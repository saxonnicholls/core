//
//  self_attention_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  snicholls::ml::self_attention_head against the same head written by hand (the same projections, the same
//  one-pass recurrence, the same operations in the same order, std::exp), one causal float head of 64 channels over
//  tokens of 128, for sequences of 128 and 512 tokens; and its backward pass against attend_backward's and
//  matmul_backward's loops by hand, the same way. main() runs a fairness check first and refuses to benchmark if
//  any output differs in a single bit (CLAUDE.md "Zero overhead").
//

#include <snicholls/core/ml/self_attention.hpp>

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

constexpr std::size_t dm = 128, dh = 64;

std::vector<float> make(std::size_t n, std::uint64_t seed)
{
    std::vector<float> out(n);
    std::uint64_t x = seed;   // a fixed xorshift stream: identical data for both
    for (auto& v : out) {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        v = static_cast<float>(static_cast<double>(x >> 11) * 0x1.0p-53 * 2.0 - 1.0);
    }
    return out;
}

struct problem {
    std::size_t n;
    std::vector<float> x, wq, wk, wv, bq, bk, bv;
    std::vector<float> q, k, v, lse, out;
    std::vector<float> dout, dq, dk, dv, dwq, dwk, dwv, dbq, dbk, dbv, dx;
    explicit problem(std::size_t tokens)
        : n(tokens), x(make(tokens * dm, 1)), wq(make(dh * dm, 2)), wk(make(dh * dm, 3)), wv(make(dh * dm, 4)),
          bq(make(dh, 5)), bk(make(dh, 6)), bv(make(dh, 7)), q(tokens * dh), k(tokens * dh), v(tokens * dh), lse(tokens),
          out(tokens * dh), dout(make(tokens * dh, 8)), dq(tokens * dh), dk(tokens * dh), dv(tokens * dh), dwq(dh * dm),
          dwk(dh * dm), dwv(dh * dm), dbq(dh), dbk(dh), dbv(dh), dx(tokens * dm)
    {
    }
    [[nodiscard]] ml::self_attention_gradients<float> grads() { return {dq, dk, dv, dwq, dwk, dwv, dbq, dbk, dbv}; }
    [[nodiscard]] ml::self_attention_weights<ml::dense_weights<float>> weights() const
    {
        return {{wq, dh, dm}, {wk, dh, dm}, {wv, dh, dm}, bq, bk, bv};
    }
};

// By hand: y = W x + b per token in linear's order, then flash attention's recurrence per position
void project(const float* w, const float* b, const float* x, float* y)
{
    for (std::size_t i = 0; i < dh; ++i) {
        float acc = b[i];
        for (std::size_t j = 0; j < dm; ++j) {
            const float p = w[i * dm + j] * x[j];
            acc += p;
        }
        y[i] = acc;
    }
}

void head_by_hand(problem& p)
{
    for (std::size_t t = 0; t < p.n; ++t) {
        project(p.wq.data(), p.bq.data(), p.x.data() + t * dm, p.q.data() + t * dh);
        project(p.wk.data(), p.bk.data(), p.x.data() + t * dm, p.k.data() + t * dh);
        project(p.wv.data(), p.bv.data(), p.x.data() + t * dm, p.v.data() + t * dh);
    }
    const float scale = 1.0f / std::sqrt(static_cast<float>(dh));
    for (std::size_t t = 0; t < p.n; ++t) {
        const float* q = p.q.data() + t * dh;
        float* out = p.out.data() + t * dh;
        for (std::size_t i = 0; i < dh; ++i)
            out[i] = 0.0f;
        float max = -std::numeric_limits<float>::infinity(), sum = 0.0f;
        for (std::size_t j = 0; j <= t; ++j) {
            const float* k = p.k.data() + j * dh;
            const float* v = p.v.data() + j * dh;
            float s = 0.0f;
            for (std::size_t i = 0; i < dh; ++i)
                s += q[i] * k[i];
            s = scale * s;
            if (s == -std::numeric_limits<float>::infinity())
                continue;
            if (s > max) {
                const float c = std::exp(max - s);
                for (std::size_t i = 0; i < dh; ++i)
                    out[i] = out[i] * c + v[i];
                sum = sum * c + 1.0f;
                max = s;
            } else {
                const float w = std::exp(s - max);
                for (std::size_t i = 0; i < dh; ++i)
                    out[i] += w * v[i];
                sum += w;
            }
        }
        for (std::size_t i = 0; i < dh; ++i)
            out[i] = sum == 0.0f ? 0.0f : out[i] / sum;
        p.lse[t] = sum == 0.0f ? -std::numeric_limits<float>::infinity() : max + std::log(sum);
    }
}

// By hand: llm.c's matmul_backward for one token, then attend_backward's loop per position
void project_backward(const float* w, const float* x, const float* dy, float* dx, float* dw, float* db)
{
    for (std::size_t i = 0; i < dh; ++i) {
        const float d = dy[i];
        for (std::size_t j = 0; j < dm; ++j) {
            const float p = w[i * dm + j] * d;
            dx[j] += p;
        }
        for (std::size_t j = 0; j < dm; ++j) {
            const float p = d * x[j];
            dw[i * dm + j] += p;
        }
        db[i] += d;
    }
}

float dot(const float* a, const float* b)
{
    float s = 0.0f;
    for (std::size_t i = 0; i < dh; ++i)
        s += a[i] * b[i];
    return s;
}

void backward_by_hand(problem& p)
{
    for (std::size_t i = 0; i < p.n * dh; ++i)
        p.dk[i] = p.dv[i] = 0.0f;
    const float scale = 1.0f / std::sqrt(static_cast<float>(dh));
    for (std::size_t t = 0; t < p.n; ++t) {
        const float* q = p.q.data() + t * dh;
        const float* o = p.out.data() + t * dh;
        const float* g = p.dout.data() + t * dh;
        float* gq = p.dq.data() + t * dh;
        for (std::size_t i = 0; i < dh; ++i)
            gq[i] = 0.0f;
        if (p.lse[t] == -std::numeric_limits<float>::infinity())
            continue;
        const float big_d = dot(g, o);
        for (std::size_t j = 0; j <= t; ++j) {
            const float* k = p.k.data() + j * dh;
            const float* v = p.v.data() + j * dh;
            const float pr = std::exp(scale * dot(q, k) - p.lse[t]);
            const float ds = pr * (dot(g, v) - big_d);
            for (std::size_t i = 0; i < dh; ++i) {
                p.dv[j * dh + i] += pr * g[i];
                gq[i] += scale * ds * k[i];
                p.dk[j * dh + i] += scale * ds * q[i];
            }
        }
    }
    for (std::size_t t = 0; t < p.n; ++t) {
        const float* x = p.x.data() + t * dm;
        float* dx = p.dx.data() + t * dm;
        project_backward(p.wq.data(), x, p.dq.data() + t * dh, dx, p.dwq.data(), p.dbq.data());
        project_backward(p.wk.data(), x, p.dk.data() + t * dh, dx, p.dwk.data(), p.dbk.data());
        project_backward(p.wv.data(), x, p.dv.data() + t * dh, dx, p.dwv.data(), p.dbv.data());
    }
}

void backward_core(problem& p)
{
    ml::self_attention_head_backward<dm, dh>(p.weights(), p.x, ml::self_attention_activations<float>{p.q, p.k, p.v, p.lse},
                                             p.out, p.dout, p.dx, p.grads());
}

void run_core(problem& p)
{
    ml::self_attention_head<dm, dh>(p.weights(), p.x, ml::self_attention_activations<float>{p.q, p.k, p.v, p.lse}, p.out);
}

void BM_SelfAttentionHead_Core(benchmark::State& state)
{
    problem p(static_cast<std::size_t>(state.range(0)));
    for (auto _ : state) {
        run_core(p);
        benchmark::DoNotOptimize(p.out.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * state.range(0));
}
BENCHMARK(BM_SelfAttentionHead_Core)->Arg(128)->Arg(512);

void BM_SelfAttentionHead_HandWritten(benchmark::State& state)
{
    problem p(static_cast<std::size_t>(state.range(0)));
    for (auto _ : state) {
        head_by_hand(p);
        benchmark::DoNotOptimize(p.out.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * state.range(0));
}
BENCHMARK(BM_SelfAttentionHead_HandWritten)->Arg(128)->Arg(512);

void BM_SelfAttentionHeadBackward_Core(benchmark::State& state)
{
    problem p(static_cast<std::size_t>(state.range(0)));
    run_core(p);
    for (auto _ : state) {
        backward_core(p);
        benchmark::DoNotOptimize(p.dx.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * state.range(0));
}
BENCHMARK(BM_SelfAttentionHeadBackward_Core)->Arg(128)->Arg(512);

void BM_SelfAttentionHeadBackward_HandWritten(benchmark::State& state)
{
    problem p(static_cast<std::size_t>(state.range(0)));
    head_by_hand(p);
    for (auto _ : state) {
        backward_by_hand(p);
        benchmark::DoNotOptimize(p.dx.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * state.range(0));
}
BENCHMARK(BM_SelfAttentionHeadBackward_HandWritten)->Arg(128)->Arg(512);

bool same_bits(const std::vector<float>& a, const std::vector<float>& b)
{
    for (std::size_t i = 0; i < a.size(); ++i)
        if (std::bit_cast<std::uint32_t>(a[i]) != std::bit_cast<std::uint32_t>(b[i]))
            return false;
    return a.size() == b.size();
}

bool fair()
{
    for (const std::size_t n : {std::size_t{128}, std::size_t{512}}) {
        problem a(n), b(n);
        run_core(a);
        head_by_hand(b);
        if (!same_bits(a.q, b.q) || !same_bits(a.k, b.k) || !same_bits(a.v, b.v) || !same_bits(a.out, b.out)
            || !same_bits(a.lse, b.lse))
            return false;
        backward_core(a);
        backward_by_hand(b);
        for (const auto m : {&problem::dq, &problem::dk, &problem::dv, &problem::dwq, &problem::dwk, &problem::dwv,
                             &problem::dbq, &problem::dbk, &problem::dbv, &problem::dx})
            if (!same_bits(a.*m, b.*m))
                return false;
    }
    return true;
}

} // namespace

int main(int argc, char** argv)
{
    if (!fair()) {
        std::fprintf(stderr, "self_attention benchmark: fairness check FAILED - the pairs disagree; not benchmarking\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
