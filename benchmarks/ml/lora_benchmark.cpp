//
//  lora_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  snicholls::ml::lora_linear and lora_linear_backward (rank 16 on a frozen 1024 x 1024 float base) against the same
//  loops written by hand, over 8 rows: the base product, A x, then s B u; the backward's du, dB, dA and dx in the same
//  order. main() runs a fairness check first and refuses to benchmark if any element differs in a single bit
//  (CLAUDE.md "Zero overhead").
//

#include <snicholls/core/ml/lora.hpp>

#include <benchmark/benchmark.h>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <vector>

namespace {

namespace ml = snicholls::ml;
constexpr std::size_t rows = 1024, cols = 1024, r = 16, n = 8;
constexpr float s = 2.0f;

std::vector<float> filled(std::uint64_t seed, std::size_t count)
{
    std::vector<float> v(count);
    std::uint64_t x = 0x9e3779b97f4a7c15ULL ^ seed;   // a fixed xorshift stream: identical data for both
    for (auto& e : v) {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        e = static_cast<float>(static_cast<double>(x >> 11) * 0x1.0p-53 - 0.5);
    }
    return v;
}

const std::vector<float> w = filled(1, rows * cols), a = filled(2, r * cols), b = filled(3, rows * r), xs = filled(4, n * cols),
                         dys = filled(5, n * rows);

struct out_t {
    std::vector<float> y = std::vector<float>(n * rows), u = std::vector<float>(n * r), dx = std::vector<float>(n * cols),
                       da = std::vector<float>(r * cols), db = std::vector<float>(rows * r), du = std::vector<float>(r);
};

void core(out_t& o)
{
    const ml::lora_weights<ml::dense_weights<float>, float> lw{{w, rows, cols}, a, b, r, s};
    for (std::size_t t = 0; t < n; ++t) {
        const auto x = std::span<const float>(xs).subspan(t * cols, cols);
        const auto u = std::span<float>(o.u).subspan(t * r, r);
        ml::lora_linear(lw, x, std::span<float>(o.y).subspan(t * rows, rows), u);
        ml::lora_linear_backward(lw, x, std::span<const float>(u), std::span<const float>(dys).subspan(t * rows, rows),
                                 std::span<float>(o.dx).subspan(t * cols, cols), o.da, o.db, o.du);
    }
}

void by_hand(out_t& o)
{
    for (std::size_t t = 0; t < n; ++t) {
        const float* x = xs.data() + t * cols;
        float* y = o.y.data() + t * rows;
        float* u = o.u.data() + t * r;
        const float* dy = dys.data() + t * rows;
        float* dx = o.dx.data() + t * cols;
        for (std::size_t i = 0; i < rows; ++i) {
            float acc = 0.0f;
            for (std::size_t j = 0; j < cols; ++j) {
                const float p = w[i * cols + j] * x[j];
                acc += p;
            }
            y[i] = acc;
        }
        for (std::size_t k = 0; k < r; ++k) {
            float acc = 0.0f;
            for (std::size_t j = 0; j < cols; ++j)
                acc += a[k * cols + j] * x[j];
            u[k] = acc;
        }
        for (std::size_t i = 0; i < rows; ++i) {
            float acc = 0.0f;
            for (std::size_t k = 0; k < r; ++k)
                acc += b[i * r + k] * u[k];
            y[i] += s * acc;
        }
        float* du = o.du.data();
        for (std::size_t k = 0; k < r; ++k)
            du[k] = 0.0f;
        for (std::size_t i = 0; i < rows; ++i) {
            const float g = s * dy[i];
            for (std::size_t k = 0; k < r; ++k) {
                du[k] += b[i * r + k] * g;
                o.db[i * r + k] += g * u[k];
            }
        }
        for (std::size_t k = 0; k < r; ++k)
            for (std::size_t j = 0; j < cols; ++j)
                o.da[k * cols + j] += du[k] * x[j];
        for (std::size_t i = 0; i < rows; ++i)
            for (std::size_t j = 0; j < cols; ++j) {
                const float p = w[i * cols + j] * dy[i];
                dx[j] += p;
            }
        for (std::size_t k = 0; k < r; ++k)
            for (std::size_t j = 0; j < cols; ++j)
                dx[j] += a[k * cols + j] * du[k];
    }
}

void BM_LoraForwardBackward_Core(benchmark::State& state)
{
    out_t o;
    for (auto _ : state) {
        core(o);
        benchmark::DoNotOptimize(o.dx.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n));
}
BENCHMARK(BM_LoraForwardBackward_Core);

void BM_LoraForwardBackward_HandWritten(benchmark::State& state)
{
    out_t o;
    for (auto _ : state) {
        by_hand(o);
        benchmark::DoNotOptimize(o.dx.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n));
}
BENCHMARK(BM_LoraForwardBackward_HandWritten);

bool same(const std::vector<float>& x, const std::vector<float>& y)
{
    for (std::size_t i = 0; i < x.size(); ++i)
        if (std::bit_cast<std::uint32_t>(x[i]) != std::bit_cast<std::uint32_t>(y[i]))
            return false;
    return x.size() == y.size();
}

bool fair()
{
    out_t p, q;
    for (int i = 0; i < 2; ++i) {
        core(p);
        by_hand(q);
    }
    return same(p.y, q.y) && same(p.u, q.u) && same(p.dx, q.dx) && same(p.da, q.da) && same(p.db, q.db);
}

} // namespace

int main(int argc, char** argv)
{
    if (!fair()) {
        std::fprintf(stderr, "lora benchmark: fairness check FAILED - the pairs disagree; not benchmarking\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
