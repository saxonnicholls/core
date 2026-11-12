//
//  transformer_block_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  One decode step of a GPT-2-small layer (768 wide, 12 heads of 64, MLP of 3072, layer norm, tanh GELU, biases) at
//  position 256 over a filled KV cache: ml::block_forward against the same layer written as direct loops over raw
//  pointers (the same projections, the same one-pass attention fold, the same order of every multiply and add), with
//  float and with Q8_0 weights. A fairness check refuses to benchmark unless the outputs and the appended keys and
//  values agree to the bit.
//

#include <snicholls/core/ml/transformer_block.hpp>

#include <benchmark/benchmark.h>

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <span>
#include <vector>

namespace {

namespace ml = snicholls::ml;
constexpr ml::transformer_config c = ml::gpt2_small;
constexpr std::size_t d = c.d_model, f = c.d_ff, hd = c.head_dim, heads = c.n_heads, positions = 257, pos = 256;

std::vector<float> stream(std::size_t n, float scale, std::uint64_t seed)
{
    std::vector<float> v(n);
    std::uint64_t x = seed * 0x9e3779b97f4a7c15ULL + 1;
    for (float& e : v) {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        e = scale * static_cast<float>(static_cast<double>(x >> 11) * 0x1.0p-53 - 0.5);
    }
    return v;
}

struct layer_t {
    std::vector<float> ln1w = stream(d, 2, 1), ln1b = stream(d, 0.2f, 2), qkvw = stream(3 * d * d, 0.1f, 3), qkvb = stream(3 * d, 0.1f, 4),
                       ow = stream(d * d, 0.1f, 5), ob = stream(d, 0.1f, 6), ln2w = stream(d, 2, 7), ln2b = stream(d, 0.2f, 8),
                       upw = stream(f * d, 0.1f, 9), upb = stream(f, 0.1f, 10), downw = stream(d * f, 0.05f, 11), downb = stream(d, 0.1f, 12);
    std::vector<ml::block_q8_0> qkvq, oq, upq, downq;
    layer_t() : qkvq(3 * d * d / 32), oq(d * d / 32), upq(f * d / 32), downq(d * f / 32)
    {
        ml::quantize(qkvw, std::span<ml::block_q8_0>(qkvq));
        ml::quantize(ow, std::span<ml::block_q8_0>(oq));
        ml::quantize(upw, std::span<ml::block_q8_0>(upq));
        ml::quantize(downw, std::span<ml::block_q8_0>(downq));
    }

    template <typename W>
    ml::block_weights<W> weights(W q, W k, W v, W o, W up, W down) const
    {
        ml::block_weights<W> w{};
        w.norm1_weight = ln1w;
        w.norm1_bias = ln1b;
        w.attention = {q, k, v, o, std::span<const float>(qkvb).subspan(0, d), std::span<const float>(qkvb).subspan(d, d),
                       std::span<const float>(qkvb).subspan(2 * d, d), ob};
        w.norm2_weight = ln2w;
        w.norm2_bias = ln2b;
        w.mlp = {up, W{}, down, upb, {}, downb};
        return w;
    }
    ml::block_weights<ml::dense_weights<float>> dense() const
    {
        using W = ml::dense_weights<float>;
        const std::span<const float> qkv(qkvw);
        return weights(W{qkv.subspan(0, d * d), d, d}, W{qkv.subspan(d * d, d * d), d, d}, W{qkv.subspan(2 * d * d, d * d), d, d}, W{ow, d, d},
                       W{upw, f, d}, W{downw, d, f});
    }
    ml::block_weights<ml::quantized_weights<ml::block_q8_0>> q8() const
    {
        using W = ml::quantized_weights<ml::block_q8_0>;
        const std::span<const ml::block_q8_0> qkv(qkvq);
        const std::size_t m = d * d / 32;
        return weights(W{qkv.subspan(0, m), d, d}, W{qkv.subspan(m, m), d, d}, W{qkv.subspan(2 * m, m), d, d}, W{oq, d, d}, W{upq, f, d},
                       W{downq, d, f});
    }
};

const layer_t& layer()
{
    static const layer_t l;
    return l;
}

struct state_t {
    std::vector<float> x = stream(d, 2, 20), keys = stream(positions * d, 1, 21), values = stream(positions * d, 1, 22);
    ml::kv_cache_view<c> cache() { return {keys, values, 1, positions}; }
};

// ---- the same layer by hand ----

void linear_by_hand(const float* w, const float* b, const float* x, float* y, std::size_t rows, std::size_t cols)
{
    for (std::size_t i = 0; i < rows; ++i) {
        float acc = b[i];
        for (std::size_t j = 0; j < cols; ++j) {
            const float p = w[i * cols + j] * x[j];
            acc += p;
        }
        y[i] = acc;
    }
}

void linear_by_hand(const ml::block_q8_0* w, const float* b, const ml::block_q8_0* xq, float* y, std::size_t rows, std::size_t cols)
{
    for (std::size_t i = 0; i < rows; ++i)
        y[i] = b[i] + ml::dot(std::span<const ml::block_q8_0>(w + i * cols / 32, cols / 32), std::span<const ml::block_q8_0>(xq, cols / 32));
}

// The input as each storage reads it: as it is, or quantised to Q8_0 once for every product that shares it
const float* prep(const float*, const float* x, std::size_t, ml::block_q8_0*)
{
    return x;
}
const ml::block_q8_0* prep(const ml::block_q8_0*, const float* x, std::size_t cols, ml::block_q8_0* buffer)
{
    ml::quantize(std::span<const float>(x, cols), std::span<ml::block_q8_0>(buffer, cols / 32));
    return buffer;
}

template <typename Wt>
void block_by_hand(const layer_t& l, const Wt* qkv, const Wt* o, const Wt* up, const Wt* down, std::size_t block, float* x, float* keys,
                   float* values)
{
    float h[d], q[d], k[d], v[d], y[d], a[d], hidden[f];
    ml::block_q8_0 buffer[f / 32];
    ml::layer_norm(std::span<const float>(x, d), l.ln1w, l.ln1b, std::span<float>(h, d), 1e-5f);
    const auto in = prep(qkv, h, d, buffer);
    linear_by_hand(qkv, l.qkvb.data(), in, q, d, d);
    linear_by_hand(qkv + block, l.qkvb.data() + d, in, k, d, d);
    linear_by_hand(qkv + 2 * block, l.qkvb.data() + 2 * d, in, v, d, d);
    for (std::size_t h2 = 0; h2 < heads; ++h2)   // the cache: [kv_head][position][head_dim]
        for (std::size_t i = 0; i < hd; ++i) {
            keys[(h2 * positions + pos) * hd + i] = k[h2 * hd + i];
            values[(h2 * positions + pos) * hd + i] = v[h2 * hd + i];
        }
    const float scale = 1.0f / snicholls::math::sqrt(float(hd));
    for (std::size_t h2 = 0; h2 < heads; ++h2) {
        float max = -std::numeric_limits<float>::infinity(), sum = 0, acc[hd] = {};
        for (std::size_t j = 0; j <= pos; ++j) {
            const float* kj = keys + (h2 * positions + j) * hd;
            const float* vj = values + (h2 * positions + j) * hd;
            float s = 0;
            for (std::size_t i = 0; i < hd; ++i)
                s += q[h2 * hd + i] * kj[i];
            s = scale * s;
            if (s > max) {
                const float r = snicholls::math::exp(max - s);
                for (std::size_t i = 0; i < hd; ++i)
                    acc[i] = acc[i] * r + vj[i];
                sum = sum * r + 1.0f;
                max = s;
            } else {
                const float w = snicholls::math::exp(s - max);
                for (std::size_t i = 0; i < hd; ++i)
                    acc[i] += w * vj[i];
                sum += w;
            }
        }
        for (std::size_t i = 0; i < hd; ++i)
            y[h2 * hd + i] = acc[i] / sum;
    }
    linear_by_hand(o, l.ob.data(), prep(o, y, d, buffer), a, d, d);
    for (std::size_t i = 0; i < d; ++i)
        x[i] += a[i];
    ml::layer_norm(std::span<const float>(x, d), l.ln2w, l.ln2b, std::span<float>(h, d), 1e-5f);
    linear_by_hand(up, l.upb.data(), prep(up, h, d, buffer), hidden, f, d);
    for (float& e : hidden)
        e = ml::gelu_tanh(e);
    linear_by_hand(down, l.downb.data(), prep(down, hidden, f, buffer), a, d, f);
    for (std::size_t i = 0; i < d; ++i)
        x[i] += a[i];
}

void by_hand_dense(state_t& s)
{
    const layer_t& l = layer();
    block_by_hand(l, l.qkvw.data(), l.ow.data(), l.upw.data(), l.downw.data(), d * d, s.x.data(), s.keys.data(), s.values.data());
}

void by_hand_q8(state_t& s)
{
    const layer_t& l = layer();
    block_by_hand(l, l.qkvq.data(), l.oq.data(), l.upq.data(), l.downq.data(), d * d / 32, s.x.data(), s.keys.data(), s.values.data());
}

void core_dense(state_t& s)
{
    static const auto w = layer().dense();
    ml::block_forward<c>(w, s.x, pos, s.cache(), 0, ml::no_rope);
}

void core_q8(state_t& s)
{
    static const auto w = layer().q8();
    ml::block_forward<c>(w, s.x, pos, s.cache(), 0, ml::no_rope);
}

template <void (*Step)(state_t&)>
void BM_Decode(benchmark::State& state)
{
    state_t s;
    const std::vector<float> x0 = s.x;
    for (auto _ : state) {
        s.x = x0;   // the same token every iteration, so every iteration does the same work
        Step(s);
        benchmark::DoNotOptimize(s.x.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations());   // tokens
}
BENCHMARK(BM_Decode<core_dense>)->Name("BM_DecodeGpt2Layer_Float_Core")->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Decode<by_hand_dense>)->Name("BM_DecodeGpt2Layer_Float_HandWritten")->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Decode<core_q8>)->Name("BM_DecodeGpt2Layer_Q8_0_Core")->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Decode<by_hand_q8>)->Name("BM_DecodeGpt2Layer_Q8_0_HandWritten")->Unit(benchmark::kMicrosecond);

bool same(const std::vector<float>& a, const std::vector<float>& b)
{
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

template <void (*A)(state_t&), void (*B)(state_t&)>
bool agree()
{
    state_t a, b;
    A(a);
    B(b);
    return same(a.x, b.x) && same(a.keys, b.keys) && same(a.values, b.values);
}

bool fair()
{
    return agree<core_dense, by_hand_dense>() && agree<core_q8, by_hand_q8>();
}

} // namespace

int main(int argc, char** argv)
{
    if (!fair()) {
        std::fprintf(stderr, "transformer_block benchmark: fairness check FAILED - the pairs disagree; not benchmarking\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
