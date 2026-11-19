//
//  convolve_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  snicholls::signal::convolve against the same direct loop written out by hand (identical arithmetic: the fairness
//  check requires every output to agree to the bit), and the FFT forms (fft_convolve, overlap_save) against the
//  direct convolution they replace, over the same 16,384-sample signal and kernels of 16, 64 and 512 taps; those pairs
//  compute different sums, so their check is the documented bound. main() refuses to benchmark if any check fails
//  (CLAUDE.md "Zero overhead").
//

#include <snicholls/core/signal/convolve.hpp>

#include <benchmark/benchmark.h>

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

namespace {

namespace s = snicholls::signal;
using cd = s::complex<double>;

std::vector<double> make_data(std::size_t n, std::uint64_t x)
{
    std::vector<double> out(n);
    for (auto& v : out) {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        v = static_cast<double>(x >> 11) * 0x1.0p-53 - 0.5;
    }
    return out;
}

const std::vector<double> signal = make_data(16384, 0x243f6a8885a308d3ULL);

// The direct full convolution written out, in core's order.
void convolve_by_hand(const std::vector<double>& a, const std::vector<double>& b, std::vector<double>& out)
{
    const std::size_t na = a.size(), nb = b.size();
    for (std::size_t k = 0; k < na + nb - 1; ++k) {
        const std::size_t lo = k + 1 > nb ? k + 1 - nb : 0, hi = k < na - 1 ? k : na - 1;
        double acc = 0;
        for (std::size_t j = lo; j <= hi; ++j)
            acc += a[j] * b[k - j];
        out[k] = acc;
    }
}

void BM_Convolve_Core(benchmark::State& state)
{
    const auto kernel = make_data(static_cast<std::size_t>(state.range(0)), 7);
    std::vector<double> out(signal.size() + kernel.size() - 1);
    for (auto _ : state) {
        s::convolve<double>(signal, kernel, out);
        benchmark::DoNotOptimize(out.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(signal.size()));
}
BENCHMARK(BM_Convolve_Core)->Arg(16)->Arg(64)->Arg(512);

void BM_Convolve_HandWritten(benchmark::State& state)
{
    const auto kernel = make_data(static_cast<std::size_t>(state.range(0)), 7);
    std::vector<double> out(signal.size() + kernel.size() - 1);
    for (auto _ : state) {
        convolve_by_hand(signal, kernel, out);
        benchmark::DoNotOptimize(out.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(signal.size()));
}
BENCHMARK(BM_Convolve_HandWritten)->Arg(16)->Arg(64)->Arg(512);

void BM_FftConvolve_Core(benchmark::State& state)
{
    const auto kernel = make_data(static_cast<std::size_t>(state.range(0)), 7);
    const std::size_t n = s::next_power_of_two(signal.size() + kernel.size() - 1);
    std::vector<cd> st(s::fft_plan<double>::storage_size(n));
    const s::fft_plan<double> plan(n, st);
    std::vector<cd> work(s::fft_convolve_work_size(plan));
    std::vector<double> out(signal.size() + kernel.size() - 1);
    for (auto _ : state) {
        s::fft_convolve(plan, std::span<const double>(signal), std::span<const double>(kernel), std::span<double>(out), std::span<cd>(work));
        benchmark::DoNotOptimize(out.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(signal.size()));
}
BENCHMARK(BM_FftConvolve_Core)->Arg(16)->Arg(64)->Arg(512);

void BM_OverlapSave_Core(benchmark::State& state)
{
    const auto kernel = make_data(static_cast<std::size_t>(state.range(0)), 7);
    const std::size_t n = s::next_power_of_two(4 * kernel.size());
    std::vector<cd> st(s::fft_plan<double>::storage_size(n));
    const s::fft_plan<double> plan(n, st);
    std::vector<cd> work(s::block_filter_work_size(plan));
    std::vector<double> out(signal.size());
    for (auto _ : state) {
        s::overlap_save(plan, std::span<const double>(kernel), std::span<const double>(signal), std::span<double>(out), std::span<cd>(work));
        benchmark::DoNotOptimize(out.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(signal.size()));
}
BENCHMARK(BM_OverlapSave_Core)->Arg(16)->Arg(64)->Arg(512);

bool fair()
{
    for (const std::size_t taps : {std::size_t{16}, std::size_t{64}, std::size_t{512}}) {
        const auto kernel = make_data(taps, 7);
        std::vector<double> a(signal.size() + taps - 1), b(a.size()), f(a.size()), os(signal.size());
        s::convolve<double>(signal, kernel, a);
        convolve_by_hand(signal, kernel, b);
        for (std::size_t i = 0; i < a.size(); ++i)
            if (std::bit_cast<std::uint64_t>(a[i]) != std::bit_cast<std::uint64_t>(b[i]))
                return false;
        const std::size_t n = s::next_power_of_two(a.size());
        std::vector<cd> st(s::fft_plan<double>::storage_size(n));
        const s::fft_plan<double> plan(n, st);
        std::vector<cd> work(s::fft_convolve_work_size(plan));
        s::fft_convolve(plan, std::span<const double>(signal), std::span<const double>(kernel), std::span<double>(f), std::span<cd>(work));
        const std::size_t nb = s::next_power_of_two(4 * taps);
        std::vector<cd> stb(s::fft_plan<double>::storage_size(nb));
        const s::fft_plan<double> block(nb, stb);
        std::vector<cd> bw(s::block_filter_work_size(block));
        s::overlap_save(block, std::span<const double>(kernel), std::span<const double>(signal), std::span<double>(os), std::span<cd>(bw));
        double na = 0, nk = 0;
        for (const double v : signal)
            na += v * v;
        for (const double v : kernel)
            nk += v * v;
        const double bound = 4 * std::numeric_limits<double>::epsilon() * std::ceil(std::log2(static_cast<double>(n))) * std::sqrt(na * nk);
        for (std::size_t i = 0; i < a.size(); ++i)
            if (std::abs(f[i] - a[i]) > bound || (i < os.size() && std::abs(os[i] - a[i]) > bound))
                return false;
    }
    return true;
}

} // namespace

int main(int argc, char** argv)
{
    if (!fair()) {
        std::fprintf(stderr, "convolve benchmark: fairness check FAILED - the pairs disagree; not benchmarking\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
