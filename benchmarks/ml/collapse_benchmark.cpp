//
//  collapse_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  snicholls::ml::vicreg_covariance_rows (with its gradient) and sigreg_moments_rows against the same loops written by
//  hand (the same operations in the same order): a batch of 256 embeddings of 128 floats, SIGReg over 64 directions
//  and 17 knots. main() runs a fairness check first and refuses to benchmark if any value differs in a single bit
//  (CLAUDE.md "Zero overhead").
//

#include <snicholls/core/ml/collapse.hpp>

#include <snicholls/core/linalg/reference.hpp>
#include <snicholls/core/math/constexpr_math.hpp>

#include <benchmark/benchmark.h>

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <vector>

namespace {

namespace ml = snicholls::ml;
namespace math = snicholls::math;
const auto& backend = snicholls::linalg::reference;

constexpr std::size_t n = 256, d = 128, m = 64, knots = 17;
const ml::sigreg_quadrature<float> quad{};

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

// VICReg's covariance term as core computes it: column means, the co-moment rows in order, the term, the gradient
float covariance_by_hand(const float* x, float* dx, float* mean, float* c)
{
    for (std::size_t j = 0; j < d; ++j)
        mean[j] = 0;
    for (std::size_t r = 0; r < n; ++r)
        for (std::size_t j = 0; j < d; ++j)
            mean[j] += x[r * d + j];
    for (std::size_t j = 0; j < d; ++j)
        mean[j] /= float(n);
    std::fill(c, c + d * d, 0.0f);
    for (std::size_t r = 0; r < n; ++r)
        for (std::size_t i = 0; i < d; ++i) {
            const float xi = x[r * d + i] - mean[i];
            for (std::size_t j = 0; j < d; ++j)
                c[i * d + j] += xi * (x[r * d + j] - mean[j]);
        }
    float total = 0;
    for (std::size_t i = 0; i < d; ++i)
        for (std::size_t j = 0; j < d; ++j) {
            c[i * d + j] /= float(n - 1);
            if (i != j)
                total += c[i * d + j] * c[i * d + j];
        }
    const float g = 1.0f * 4.0f / (float(d) * float(n - 1));
    for (std::size_t r = 0; r < n; ++r)
        for (std::size_t b = 0; b < d; ++b) {
            float s = 0;
            for (std::size_t j = 0; j < d; ++j)
                if (j != b)
                    s += (x[r * d + j] - mean[j]) * c[j * d + b];
            dx[r * d + b] += g * s;
        }
    return 1.0f * (total / float(d));
}

void moments_by_hand(const float* x, const float* dirs, float* sums)
{
    for (std::size_t r = 0; r < n; ++r)
        for (std::size_t a = 0; a < m; ++a) {
            float z = 0;
            for (std::size_t c = 0; c < d; ++c)
                z += x[r * d + c] * dirs[a * d + c];
            for (std::size_t k = 0; k < knots; ++k) {
                sums[(a * knots + k) * 2] += math::cos(quad.t[k] * z);
                sums[(a * knots + k) * 2 + 1] += math::sin(quad.t[k] * z);
            }
        }
}

struct state_t {
    std::vector<float> x = filled(1, n * d), dx = std::vector<float>(n * d), work = std::vector<float>(d + d * d), mean = std::vector<float>(d),
                       c = std::vector<float>(d * d), dirs = std::vector<float>(m * d), sums = std::vector<float>(m * knots * 2);
    state_t() { ml::sigreg_directions<float>(1, 0, d, dirs); }
};

void BM_VicregCovariance_Core(benchmark::State& state)
{
    state_t s;
    for (auto _ : state) {
        benchmark::DoNotOptimize(
            ml::vicreg_covariance_rows(backend, std::span<const float>(s.x), d, std::span<float>(s.dx), std::span<float>(s.work)));
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * std::int64_t(n));
}
BENCHMARK(BM_VicregCovariance_Core);

void BM_VicregCovariance_HandWritten(benchmark::State& state)
{
    state_t s;
    for (auto _ : state) {
        benchmark::DoNotOptimize(covariance_by_hand(s.x.data(), s.dx.data(), s.mean.data(), s.c.data()));
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * std::int64_t(n));
}
BENCHMARK(BM_VicregCovariance_HandWritten);

void BM_SigregMoments_Core(benchmark::State& state)
{
    state_t s;
    for (auto _ : state) {
        ml::sigreg_moments_rows(backend, std::span<const float>(s.x), d, std::span<const float>(s.dirs), quad, std::span<float>(s.sums));
        benchmark::DoNotOptimize(s.sums.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * std::int64_t(n));
}
BENCHMARK(BM_SigregMoments_Core);

void BM_SigregMoments_HandWritten(benchmark::State& state)
{
    state_t s;
    for (auto _ : state) {
        moments_by_hand(s.x.data(), s.dirs.data(), s.sums.data());
        benchmark::DoNotOptimize(s.sums.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * std::int64_t(n));
}
BENCHMARK(BM_SigregMoments_HandWritten);

bool same(const std::vector<float>& a, const std::vector<float>& b)
{
    for (std::size_t i = 0; i < a.size(); ++i)
        if (std::bit_cast<std::uint32_t>(a[i]) != std::bit_cast<std::uint32_t>(b[i]))
            return false;
    return true;
}

bool fair()
{
    state_t a, b;
    const float ca = ml::vicreg_covariance_rows(backend, std::span<const float>(a.x), d, std::span<float>(a.dx), std::span<float>(a.work));
    const float cb = covariance_by_hand(b.x.data(), b.dx.data(), b.mean.data(), b.c.data());
    ml::sigreg_moments_rows(backend, std::span<const float>(a.x), d, std::span<const float>(a.dirs), quad, std::span<float>(a.sums));
    moments_by_hand(b.x.data(), b.dirs.data(), b.sums.data());
    return std::bit_cast<std::uint32_t>(ca) == std::bit_cast<std::uint32_t>(cb) && same(a.dx, b.dx) && same(a.sums, b.sums);
}

} // namespace

int main(int argc, char** argv)
{
    if (!fair()) {
        std::fprintf(stderr, "collapse benchmark: fairness check FAILED - the pairs disagree; not benchmarking\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
