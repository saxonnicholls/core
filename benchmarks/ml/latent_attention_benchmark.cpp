//
//  latent_attention_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  snicholls::ml::latent_attention_forward_sequence (multi-head latent attention) against
//  attention_forward_sequence (multi-head attention) with the same heads and query-key widths, both through the
//  reference backend, over 128 and 512 tokens of 256 channels; each against the same pass written by hand (the
//  reference backend's products as loops in index order, the RMS norms, RoPE and flash attention's recurrence with
//  std::exp, every operation in the same order). main() runs a fairness check first and refuses to benchmark if any
//  core pass and its loops by hand differ in a single bit (CLAUDE.md "Zero overhead").
//

#include <snicholls/core/linalg/reference.hpp>
#include <snicholls/core/ml/attention_layer.hpp>
#include <snicholls/core/ml/latent_attention.hpp>
#include <snicholls/core/ml/rope.hpp>

#include <benchmark/benchmark.h>

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <span>
#include <vector>

namespace {

namespace ml = snicholls::ml;
using snicholls::linalg::reference;

// 4 heads of 32 plain + 16 rotary query-key channels; MLA: values of 32, latents of 96 (query) and 64 (key-value)
constexpr ml::transformer_config mla{.d_model = 256, .n_heads = 4, .n_kv_heads = 4, .head_dim = 48, .d_ff = 512, .norm = ml::norm_kind::rms,
                                     .norm_eps = 1e-6, .rope = ml::rope_kind::half, .rope_base = 10000, .q_lora_rank = 96,
                                     .kv_lora_rank = 64, .qk_rope_dim = 16, .v_head_dim = 32};
constexpr ml::transformer_config mha{.d_model = 256, .n_heads = 4, .n_kv_heads = 4, .head_dim = 48, .d_ff = 512, .norm = ml::norm_kind::rms,
                                     .norm_eps = 1e-6, .rope = ml::rope_kind::half, .rope_base = 10000};
constexpr std::size_t d = 256, H = 4, dqk = 48, dn = 32, dr = 16, dv = 32, dc = 64, dq = 96, kvw = dn + dv, longest = 512;

const ml::rope_table<float, longest, dr> mla_rope{10000.0f};
const ml::rope_table<float, longest, dqk> mha_rope{10000.0f};

std::vector<float> stream(std::size_t size, float scale, std::uint64_t seed)
{
    std::vector<float> v(size);
    std::uint64_t x = seed * 0x9e3779b97f4a7c15ULL + 1;   // a fixed xorshift stream: identical data for both
    for (float& e : v) {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        e = scale * static_cast<float>(static_cast<double>(x >> 11) * 0x1.0p-53 - 0.5);
    }
    return v;
}

// The pieces by hand: the reference backend's product (each output a sum in index order), an RMS norm without a
// weight, RoPE's rotation, flash attention's fold
void product(const float* x, const float* w, float* y, std::size_t n, std::size_t rows, std::size_t cols)
{
    for (std::size_t t = 0; t < n; ++t)
        for (std::size_t i = 0; i < rows; ++i) {
            float s = 0.0f;
            for (std::size_t p = 0; p < cols; ++p)
                s += x[t * cols + p] * w[i * cols + p];
            y[t * rows + i] = s;
        }
}

void rms(const float* x, float* y, std::size_t n)
{
    float ms = 0.0f;
    for (std::size_t i = 0; i < n; ++i)
        ms += x[i] * x[i];
    ms = ms / static_cast<float>(n);
    const float rstd = 1.0f / std::sqrt(ms + static_cast<float>(1e-6));
    for (std::size_t i = 0; i < n; ++i)
        y[i] = x[i] * rstd * 1.0f;
}

void rotate(float* x, std::span<const float> c, std::span<const float> s)
{
    const std::size_t h = c.size();
    for (std::size_t i = 0; i < h; ++i) {
        const float x1 = x[i], x2 = x[i + h], sn = 1.0f * s[i];
        x[i] = x1 * c[i] - x2 * sn;
        x[i + h] = x1 * sn + x2 * c[i];
    }
}

void fold(float& max, float& sum, float* acc, float s, const float* v, std::size_t n)
{
    if (s > max) {
        const float c = std::exp(max - s);
        for (std::size_t i = 0; i < n; ++i)
            acc[i] = acc[i] * c + v[i];
        sum = sum * c + 1.0f;
        max = s;
    } else {
        const float w = std::exp(s - max);
        for (std::size_t i = 0; i < n; ++i)
            acc[i] += w * v[i];
        sum += w;
    }
}

struct latent_problem {
    std::size_t n;
    std::vector<float> wq_down, wq_up, wkv_down, wkv_up, wo, xs, out;
    std::vector<float> cq_raw, cq, q, ckv_raw, ckv, k_rope, kv, y, lse;
    explicit latent_problem(std::size_t tokens)
        : n(tokens), wq_down(stream(dq * d, 0.2f, 1)), wq_up(stream(H * dqk * dq, 0.3f, 2)), wkv_down(stream((dc + dr) * d, 0.2f, 3)),
          wkv_up(stream(H * kvw * dc, 0.3f, 4)), wo(stream(d * H * dv, 0.2f, 5)), xs(stream(tokens * d, 2.0f, 6)), out(tokens * d),
          cq_raw(tokens * dq), cq(tokens * dq), q(tokens * H * dqk), ckv_raw(tokens * (dc + dr)), ckv(tokens * dc), k_rope(tokens * dr),
          kv(tokens * H * kvw), y(tokens * H * dv), lse(tokens * H)
    {
    }
    [[nodiscard]] ml::latent_attention_weights<ml::dense_weights<float>, float> weights() const
    {
        return {{wq_down, dq, d}, {}, {wq_up, H * dqk, dq}, {wkv_down, dc + dr, d}, {}, {wkv_up, H * kvw, dc}, {wo, d, H * dv}, {}};
    }
};

void latent_core(latent_problem& p)
{
    ml::latent_attention_forward_sequence<mla>(reference, p.weights(), p.xs, mla_rope,
                                               ml::latent_attention_activations<float>{p.cq_raw, p.cq, p.q, p.ckv_raw, p.ckv, p.k_rope, p.kv, p.y, p.lse, {}},
                                               p.out);
}

void latent_by_hand(latent_problem& p)
{
    const std::size_t n = p.n;
    product(p.xs.data(), p.wq_down.data(), p.cq_raw.data(), n, dq, d);
    product(p.xs.data(), p.wkv_down.data(), p.ckv_raw.data(), n, dc + dr, d);
    for (std::size_t t = 0; t < n; ++t) {
        rms(p.cq_raw.data() + t * dq, p.cq.data() + t * dq, dq);
        rms(p.ckv_raw.data() + t * (dc + dr), p.ckv.data() + t * dc, dc);
    }
    product(p.cq.data(), p.wq_up.data(), p.q.data(), n, H * dqk, dq);
    for (std::size_t t = 0; t < n; ++t) {
        float* kr = p.k_rope.data() + t * dr;
        for (std::size_t i = 0; i < dr; ++i)
            kr[i] = p.ckv_raw[t * (dc + dr) + dc + i];
        rotate(kr, mla_rope.cos_at(t), mla_rope.sin_at(t));
        for (std::size_t h = 0; h < H; ++h)
            rotate(p.q.data() + t * H * dqk + h * dqk + dn, mla_rope.cos_at(t), mla_rope.sin_at(t));
    }
    product(p.ckv.data(), p.wkv_up.data(), p.kv.data(), n, H * kvw, dc);
    const float scale = 1.0f / std::sqrt(static_cast<float>(dqk));
    for (std::size_t t = 0; t < n; ++t)
        for (std::size_t h = 0; h < H; ++h) {
            const float* qh = p.q.data() + t * H * dqk + h * dqk;
            float* acc = p.y.data() + t * H * dv + h * dv;
            for (std::size_t i = 0; i < dv; ++i)
                acc[i] = 0.0f;
            float max = -std::numeric_limits<float>::infinity(), sum = 0.0f;
            for (std::size_t j = 0; j <= t; ++j) {
                const float* row = p.kv.data() + j * H * kvw + h * kvw;
                float s = 0.0f;
                for (std::size_t i = 0; i < dn; ++i)
                    s += qh[i] * row[i];
                for (std::size_t i = 0; i < dr; ++i)
                    s += qh[dn + i] * p.k_rope[j * dr + i];
                fold(max, sum, acc, scale * s, row + dn, dv);
            }
            for (std::size_t i = 0; i < dv; ++i)
                acc[i] = acc[i] / sum;
            p.lse[t * H + h] = max + std::log(sum);
        }
    product(p.y.data(), p.wo.data(), p.out.data(), n, d, H * dv);
}

struct standard_problem {
    std::size_t n;
    std::vector<float> wq, wk, wv, wo, xs, out, q, k, v, y, lse;
    explicit standard_problem(std::size_t tokens)
        : n(tokens), wq(stream(H * dqk * d, 0.2f, 7)), wk(stream(H * dqk * d, 0.2f, 8)), wv(stream(H * dqk * d, 0.2f, 9)),
          wo(stream(d * H * dqk, 0.2f, 10)), xs(stream(tokens * d, 2.0f, 6)), out(tokens * d), q(tokens * H * dqk), k(tokens * H * dqk),
          v(tokens * H * dqk), y(tokens * H * dqk), lse(tokens * H)
    {
    }
    [[nodiscard]] ml::attention_weights<ml::dense_weights<float>, float> weights() const
    {
        return {{wq, H * dqk, d}, {wk, H * dqk, d}, {wv, H * dqk, d}, {wo, d, H * dqk}};
    }
};

void standard_core(standard_problem& p)
{
    ml::attention_forward_sequence<mha>(reference, p.weights(), p.xs, mha_rope,
                                        ml::attention_activations<float>{p.q, p.k, p.v, {}, {}, p.y, p.lse}, p.out);
}

void standard_by_hand(standard_problem& p)
{
    const std::size_t n = p.n;
    product(p.xs.data(), p.wq.data(), p.q.data(), n, H * dqk, d);
    for (std::size_t h = 0; h < H; ++h) {   // the keys and values head by head, as the activations keep them
        product(p.xs.data(), p.wk.data() + h * dqk * d, p.k.data() + h * n * dqk, n, dqk, d);
        product(p.xs.data(), p.wv.data() + h * dqk * d, p.v.data() + h * n * dqk, n, dqk, d);
    }
    for (std::size_t t = 0; t < n; ++t) {
        for (std::size_t h = 0; h < H; ++h)
            rotate(p.q.data() + t * H * dqk + h * dqk, mha_rope.cos_at(t), mha_rope.sin_at(t));
        for (std::size_t h = 0; h < H; ++h)
            rotate(p.k.data() + (h * n + t) * dqk, mha_rope.cos_at(t), mha_rope.sin_at(t));
    }
    const float scale = 1.0f / std::sqrt(static_cast<float>(dqk));
    for (std::size_t t = 0; t < n; ++t)
        for (std::size_t h = 0; h < H; ++h) {
            const float* qh = p.q.data() + t * H * dqk + h * dqk;
            float acc[dqk] = {};
            float max = -std::numeric_limits<float>::infinity(), sum = 0.0f;
            for (std::size_t j = 0; j <= t; ++j) {
                const float* kj = p.k.data() + (h * n + j) * dqk;
                float s = 0.0f;
                for (std::size_t i = 0; i < dqk; ++i)
                    s += qh[i] * kj[i];
                fold(max, sum, acc, scale * s, p.v.data() + (h * n + j) * dqk, dqk);
            }
            for (std::size_t i = 0; i < dqk; ++i)
                p.y[t * H * dqk + h * dqk + i] = acc[i] / sum;
            p.lse[t * H + h] = max + std::log(sum);
        }
    product(p.y.data(), p.wo.data(), p.out.data(), n, d, H * dqk);
}

template <typename P, void (*Run)(P&)>
void run(benchmark::State& state)
{
    P p(static_cast<std::size_t>(state.range(0)));
    for (auto _ : state) {
        Run(p);
        benchmark::DoNotOptimize(p.out.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * state.range(0));
}

void BM_LatentAttentionForward_Core(benchmark::State& state) { run<latent_problem, latent_core>(state); }
void BM_LatentAttentionForward_HandWritten(benchmark::State& state) { run<latent_problem, latent_by_hand>(state); }
void BM_MultiHeadAttentionForward_Core(benchmark::State& state) { run<standard_problem, standard_core>(state); }
void BM_MultiHeadAttentionForward_HandWritten(benchmark::State& state) { run<standard_problem, standard_by_hand>(state); }
BENCHMARK(BM_LatentAttentionForward_Core)->Arg(128)->Arg(512);
BENCHMARK(BM_LatentAttentionForward_HandWritten)->Arg(128)->Arg(512);
BENCHMARK(BM_MultiHeadAttentionForward_Core)->Arg(128)->Arg(512);
BENCHMARK(BM_MultiHeadAttentionForward_HandWritten)->Arg(128)->Arg(512);

bool same_bits(const std::vector<float>& a, const std::vector<float>& b)
{
    if (a.size() != b.size())
        return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (std::bit_cast<std::uint32_t>(a[i]) != std::bit_cast<std::uint32_t>(b[i]))
            return false;
    return true;
}

bool fair()
{
    for (const std::size_t n : {std::size_t{128}, std::size_t{512}}) {
        latent_problem a(n), b(n);
        latent_core(a);
        latent_by_hand(b);
        for (const auto m : {&latent_problem::cq, &latent_problem::q, &latent_problem::ckv, &latent_problem::k_rope, &latent_problem::kv,
                             &latent_problem::y, &latent_problem::lse, &latent_problem::out})
            if (!same_bits(a.*m, b.*m))
                return false;
        standard_problem c(n), e(n);
        standard_core(c);
        standard_by_hand(e);
        for (const auto m : {&standard_problem::q, &standard_problem::k, &standard_problem::v, &standard_problem::y, &standard_problem::lse,
                             &standard_problem::out})
            if (!same_bits(c.*m, e.*m))
                return false;
    }
    return true;
}

} // namespace

int main(int argc, char** argv)
{
    if (!fair()) {
        std::fprintf(stderr, "latent_attention_benchmark: a core pass and its loops by hand differ; refusing to benchmark\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
