//
//  linear_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  snicholls::ml::linear_sequence and linear_sequence_backward through linalg::reference, against the same matrix
//  products written as loops over raw pointers (the reference backend's order: each output summed in index order,
//  then scaled and added; the bias after), for a GPT-2-shaped projection, 64 tokens of 256 into 256 and into 1024.
//  This is the zero-overhead check of the linalg path: views, transposed views, the extent checks and the
//  backend's dispatch against nothing. A fairness check refuses to benchmark unless every pair agrees to the bit.
//  For context, the per-token ml::linear over the same rows (not a pair: its bias joins the sum first).
//

#include <snicholls/core/ml/linear.hpp>

#include <snicholls/core/linalg/reference.hpp>

#include <benchmark/benchmark.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <span>
#include <vector>

namespace {

namespace ml = snicholls::ml;
constexpr std::size_t n = 64, cols = 256;

std::vector<float> stream(std::size_t size, float scale, std::uint64_t seed)
{
    std::vector<float> v(size);
    std::uint64_t x = seed * 0x9e3779b97f4a7c15ULL + 1;
    for (float& e : v) {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        e = scale * static_cast<float>(static_cast<double>(x >> 11) * 0x1.0p-53 - 0.5);
    }
    return v;
}

struct problem {
    std::size_t rows;
    std::vector<float> w, b, xs, ys, dys, dxs, dw, db;
    explicit problem(std::size_t r)
        : rows(r), w(stream(r * cols, 0.2f, 1)), b(stream(r, 0.1f, 2)), xs(stream(n * cols, 2, 3)), ys(n * r), dys(stream(n * r, 1, 4)),
          dxs(n * cols), dw(r * cols), db(r)
    {
    }
    ml::dense_weights<float> weights() const { return {w, rows, cols}; }
};

void forward_core(problem& p)
{
    ml::linear_sequence(snicholls::linalg::reference, p.weights(), p.xs, p.ys, p.b);
}

void forward_by_hand(problem& p)
{
    const float* w = p.w.data();
    for (std::size_t t = 0; t < n; ++t)
        for (std::size_t i = 0; i < p.rows; ++i) {
            float s = 0;
            for (std::size_t j = 0; j < cols; ++j)
                s += p.xs[t * cols + j] * w[i * cols + j];
            p.ys[t * p.rows + i] = 1.0f * s;
        }
    for (std::size_t t = 0; t < n; ++t)
        for (std::size_t i = 0; i < p.rows; ++i)
            p.ys[t * p.rows + i] += p.b[i];
}

void forward_per_token(problem& p)
{
    for (std::size_t t = 0; t < n; ++t)
        ml::linear(p.weights(), std::span<const float>(p.xs).subspan(t * cols, cols), std::span<float>(p.ys).subspan(t * p.rows, p.rows),
                   std::span<const float>(p.b));
}

void backward_core(problem& p)
{
    ml::linear_sequence_backward(snicholls::linalg::reference, p.weights(), p.xs, p.dys, p.dxs, p.dw, p.db);
}

void backward_by_hand(problem& p)
{
    const float* w = p.w.data();
    const std::size_t r = p.rows;
    for (std::size_t t = 0; t < n; ++t)   // dX += dY W
        for (std::size_t j = 0; j < cols; ++j) {
            float s = 0;
            for (std::size_t i = 0; i < r; ++i)
                s += p.dys[t * r + i] * w[i * cols + j];
            p.dxs[t * cols + j] = 1.0f * s + 1.0f * p.dxs[t * cols + j];
        }
    for (std::size_t i = 0; i < r; ++i)   // dW += dYᵀ X
        for (std::size_t j = 0; j < cols; ++j) {
            float s = 0;
            for (std::size_t t = 0; t < n; ++t)
                s += p.dys[t * r + i] * p.xs[t * cols + j];
            p.dw[i * cols + j] = 1.0f * s + 1.0f * p.dw[i * cols + j];
        }
    for (std::size_t t = 0; t < n; ++t)   // db += Σ dY
        for (std::size_t i = 0; i < r; ++i)
            p.db[i] += p.dys[t * r + i];
}

template <void (*Run)(problem&)>
void BM_Run(benchmark::State& state)
{
    problem p(static_cast<std::size_t>(state.range(0)));
    for (auto _ : state) {
        Run(p);
        benchmark::DoNotOptimize(p.ys.data());
        benchmark::DoNotOptimize(p.dw.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * std::int64_t(n));   // tokens
}
BENCHMARK(BM_Run<forward_core>)->Name("BM_LinearSequence_Core")->Arg(256)->Arg(1024)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Run<forward_by_hand>)->Name("BM_LinearSequence_HandWritten")->Arg(256)->Arg(1024)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Run<backward_core>)->Name("BM_LinearSequenceBackward_Core")->Arg(256)->Arg(1024)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Run<backward_by_hand>)->Name("BM_LinearSequenceBackward_HandWritten")->Arg(256)->Arg(1024)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Run<forward_per_token>)->Name("BM_LinearPerToken_Context")->Arg(256)->Arg(1024)->Unit(benchmark::kMicrosecond);

bool same(const std::vector<float>& a, const std::vector<float>& b)
{
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

bool fair()
{
    for (const std::size_t rows : {std::size_t{256}, std::size_t{1024}}) {
        problem a(rows), b(rows);
        forward_core(a);
        forward_by_hand(b);
        backward_core(a);
        backward_by_hand(b);
        if (!same(a.ys, b.ys) || !same(a.dxs, b.dxs) || !same(a.dw, b.dw) || !same(a.db, b.db))
            return false;
    }
    return true;
}

} // namespace

int main(int argc, char** argv)
{
    if (!fair()) {
        std::fprintf(stderr, "linear benchmark: fairness check FAILED - the pairs disagree; not benchmarking\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
