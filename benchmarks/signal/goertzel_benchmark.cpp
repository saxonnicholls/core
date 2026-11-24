//
//  goertzel_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  snicholls::signal::goertzel and goertzel_bank against, first, the same recurrence written out by hand with the same
//  coefficients (identical arithmetic in the same order: the fairness check requires the powers to agree to the bit),
//  and then against what they replace: fft.hpp's real transform of the whole block, reading the power of the same one
//  or eight integer bins. Those pairs compute different sums, so their fairness check is the documented bounds
//  (goertzel.hpp's plus fft.hpp's). Blocks of 205 (DTMF at 8 kHz; a Bluestein length for the FFT), 1,024 and 4,096,
//  in double and float. main() refuses to benchmark if any check fails (CLAUDE.md "Zero overhead").
//

#include <snicholls/core/signal/fft.hpp>
#include <snicholls/core/signal/goertzel.hpp>

#include <benchmark/benchmark.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

namespace {

namespace s = snicholls::signal;
constexpr std::size_t K = 8;

template <typename T>
std::vector<T> make_data(std::size_t n)
{
    std::vector<T> out(n);
    std::uint64_t x = 0x243f6a8885a308d3ULL;   // a fixed xorshift stream: identical data for every pair
    for (auto& v : out) {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        v = static_cast<T>(static_cast<double>(x >> 11) * 0x1.0p-53 - 0.5);
    }
    return out;
}

// Eight integer bins spread over the half spectrum; the first is the single-bin case.
std::array<std::size_t, K> bins_of(std::size_t n)
{
    std::array<std::size_t, K> k{};
    for (std::size_t i = 0; i < K; ++i)
        k[i] = (2 * i + 1) * n / (4 * K) + 1;
    return k;
}

template <typename T>
std::array<s::goertzel_coefficients<T>, K> coefficients_of(std::size_t n)
{
    std::array<s::goertzel_coefficients<T>, K> c{};
    const auto k = bins_of(n);
    for (std::size_t i = 0; i < K; ++i)
        c[i] = s::goertzel_coefficients<T>::at_bin(k[i], n);
    return c;
}

// The power of a final state, as core computes it: |cos w s1 - s2 + i sin w s1|^2.
template <typename T>
T power_by_hand(T s1, T s2, const s::goertzel_coefficients<T>& c)
{
    const T re = c.cos_w * s1 - s2, im = c.sin_w * s1;
    return re * re + im * im;
}

// The recurrence written out: one bin.
template <typename T>
T one_bin_by_hand(const std::vector<T>& x, const s::goertzel_coefficients<T>& c)
{
    const T coeff = c.coeff;
    T s1 = 0, s2 = 0;
    for (const T v : x) {
        const T s0 = (v - s2) + coeff * s1;
        s2 = s1;
        s1 = s0;
    }
    return power_by_hand(s1, s2, c);
}

// The recurrence written out: eight bins in one pass.
template <typename T>
std::array<T, K> bank_by_hand(const std::vector<T>& x, const std::array<T, K>& coeff, const std::array<s::goertzel_coefficients<T>, K>& c)
{
    std::array<T, K> s1{}, s2{}, p{};
    for (const T v : x)
        for (std::size_t i = 0; i < K; ++i) {
            const T s0 = (v - s2[i]) + coeff[i] * s1[i];
            s2[i] = s1[i];
            s1[i] = s0;
        }
    for (std::size_t i = 0; i < K; ++i)
        p[i] = power_by_hand(s1[i], s2[i], c[i]);
    return p;
}

template <typename T>
struct rfft_planned
{
    std::vector<s::complex<T>> storage, work, out;
    s::rfft_plan<T> plan;
    explicit rfft_planned(std::size_t n)
        : storage(s::rfft_plan<T>::storage_size(n)), work(s::rfft_plan<T>::work_size(n)), out(n / 2 + 1), plan(n, std::span<s::complex<T>>(storage)) {}
    // The whole transform, then the powers of the bins asked for.
    template <std::size_t M>
    std::array<T, M> powers(const std::vector<T>& x, const std::array<std::size_t, M>& k)
    {
        s::rfft(plan, std::span<const T>(x), std::span<s::complex<T>>(out), std::span<s::complex<T>>(work));
        std::array<T, M> p{};
        for (std::size_t i = 0; i < M; ++i)
            p[i] = s::norm(out[k[i]]);
        return p;
    }
};

template <typename T>
void BM_Goertzel_OneBin(benchmark::State& state)
{
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    const auto x = make_data<T>(n);
    s::goertzel<T> d(coefficients_of<T>(n)[0]);
    for (auto _ : state) {
        d.reset();
        d.update(x);
        benchmark::DoNotOptimize(d.power());
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n));
}
BENCHMARK_TEMPLATE(BM_Goertzel_OneBin, double)->Arg(205)->Arg(1024)->Arg(4096);
BENCHMARK_TEMPLATE(BM_Goertzel_OneBin, float)->Arg(1024);

template <typename T>
void BM_Goertzel_OneBin_HandWritten(benchmark::State& state)
{
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    const auto x = make_data<T>(n);
    const auto c = coefficients_of<T>(n)[0];
    for (auto _ : state)
        benchmark::DoNotOptimize(one_bin_by_hand(x, c));
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n));
}
BENCHMARK_TEMPLATE(BM_Goertzel_OneBin_HandWritten, double)->Arg(205)->Arg(1024)->Arg(4096);
BENCHMARK_TEMPLATE(BM_Goertzel_OneBin_HandWritten, float)->Arg(1024);

template <typename T>
void BM_Goertzel_Bank8(benchmark::State& state)
{
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    const auto x = make_data<T>(n);
    s::goertzel_bank<T, K> bank(coefficients_of<T>(n));
    for (auto _ : state) {
        bank.reset();
        bank.update(x);
        benchmark::DoNotOptimize(bank.powers());
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n));
}
BENCHMARK_TEMPLATE(BM_Goertzel_Bank8, double)->Arg(205)->Arg(1024)->Arg(4096);
BENCHMARK_TEMPLATE(BM_Goertzel_Bank8, float)->Arg(1024);

template <typename T>
void BM_Goertzel_Bank8_HandWritten(benchmark::State& state)
{
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    const auto x = make_data<T>(n);
    std::array<T, K> coeff{};
    const auto c = coefficients_of<T>(n);
    for (std::size_t i = 0; i < K; ++i)
        coeff[i] = c[i].coeff;
    for (auto _ : state)
        benchmark::DoNotOptimize(bank_by_hand(x, coeff, c));
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n));
}
BENCHMARK_TEMPLATE(BM_Goertzel_Bank8_HandWritten, double)->Arg(205)->Arg(1024)->Arg(4096);
BENCHMARK_TEMPLATE(BM_Goertzel_Bank8_HandWritten, float)->Arg(1024);

// What the detectors replace: the full real FFT of the block, reading the same bins (one or eight: the cost is the transform).
template <typename T>
void BM_Rfft_SameBins(benchmark::State& state)
{
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    const auto x = make_data<T>(n);
    rfft_planned<T> p(n);
    const auto k = bins_of(n);
    for (auto _ : state)
        benchmark::DoNotOptimize(p.powers(x, k));
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n));
}
BENCHMARK_TEMPLATE(BM_Rfft_SameBins, double)->Arg(205)->Arg(1024)->Arg(4096);
BENCHMARK_TEMPLATE(BM_Rfft_SameBins, float)->Arg(1024);

template <typename T>
bool fair_for(std::size_t n)
{
    const auto x = make_data<T>(n);
    const auto c = coefficients_of<T>(n);
    const auto k = bins_of(n);
    // Core against the recurrence by hand: bit-identical.
    s::goertzel<T> d(c[0]);
    d.update(x);
    if (d.power() != one_bin_by_hand(x, c[0]))
        return false;
    s::goertzel_bank<T, K> bank(c);
    bank.update(x);
    std::array<T, K> coeff{};
    for (std::size_t i = 0; i < K; ++i)
        coeff[i] = c[i].coeff;
    if (bank.powers() != bank_by_hand(x, coeff, c))
        return false;
    // Core against the FFT: the same bins within the bounds.
    rfft_planned<T> p(n);
    const auto ref = p.powers(x, k);
    T xn = 0;
    for (const T v : x)
        xn += v * v;
    xn = std::sqrt(xn);
    const T e = std::numeric_limits<T>::epsilon();
    for (std::size_t i = 0; i < K; ++i) {
        const double sin_w = std::fabs(std::sin(2 * 3.14159265358979323846 * static_cast<double>(k[i]) / static_cast<double>(n)));
        const T tol = e * static_cast<T>(n) * xn / static_cast<T>(std::max(sin_w, 1.0 / static_cast<double>(n)))
                      + T(2) * e * static_cast<T>(std::ceil(std::log2(4.0 * static_cast<double>(n)))) * xn;
        const T mag = std::sqrt(ref[i]);
        if (std::fabs(bank.power(i) - ref[i]) > 2 * tol * (mag + tol))
            return false;
    }
    return true;
}

bool fair()
{
    return fair_for<double>(205) && fair_for<double>(1024) && fair_for<double>(4096) && fair_for<float>(1024);
}

} // namespace

int main(int argc, char** argv)
{
    if (!fair()) {
        std::fprintf(stderr, "goertzel benchmark: fairness check FAILED - the pairs disagree; not benchmarking\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
