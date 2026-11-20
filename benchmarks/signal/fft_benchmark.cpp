//
//  fft_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  snicholls::signal::fft against, first, the same radix-4 Stockham passes written out by hand over the same twiddle
//  table (identical arithmetic in the same order: the fairness check requires every output to agree to the bit), and
//  then against what it replaces: the direct O(n^2) transform over the same table, for a power of two, a mixed-radix
//  length and a prime (Bluestein). Those pairs compute different sums, so their fairness check is the documented
//  bound, 2 eps ceil(log2 L) ||x||_2. Also the real transform against the complex one of the same samples, and the
//  compile-time plan's in-place radix-2 pass. main() refuses to benchmark if any check fails (CLAUDE.md "Zero
//  overhead").
//

#include <snicholls/core/signal/fft.hpp>

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

std::vector<cd> make_data(std::size_t n)
{
    std::vector<cd> out(n);
    std::uint64_t x = 0x243f6a8885a308d3ULL;   // a fixed xorshift stream: identical data for every pair
    for (auto& v : out) {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        v.re = static_cast<double>(x >> 11) * 0x1.0p-53 - 0.5;
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        v.im = static_cast<double>(x >> 11) * 0x1.0p-53 - 0.5;
    }
    return out;
}

struct planned
{
    std::vector<cd> storage, work;
    s::fft_plan<double> plan;
    explicit planned(std::size_t n) : storage(s::fft_plan<double>::storage_size(n)), work(s::fft_plan<double>::work_size(n)), plan(n, std::span<cd>(storage)) {}
};

// The radix-4 Stockham transform written out: core's passes for a power of four, the same table, the same order.
void stockham4_by_hand(const cd* tw, std::size_t n, cd* x, cd* y)
{
    std::size_t len = n, st = 1;
    cd* src = x;
    cd* dst = y;
    while (len > 1) {
        const std::size_t m = len / 4;
        for (std::size_t j = 0; j < m; ++j) {
            const cd w1 = tw[st * j], w2 = tw[2 * st * j], w3 = tw[3 * st * j];
            for (std::size_t q = 0; q < st; ++q) {
                const cd a0 = src[q + st * j], a1 = src[q + st * (j + m)], a2 = src[q + st * (j + 2 * m)], a3 = src[q + st * (j + 3 * m)];
                const cd t0 = a0 + a2, t1 = a0 - a2, t2 = a1 + a3, d = a1 - a3;
                const cd t3{d.im, -d.re};
                cd* out = dst + st * (4 * j);
                out[q] = t0 + t2;
                out[q + st] = (t1 + t3) * w1;
                out[q + 2 * st] = (t0 - t2) * w2;
                out[q + 3 * st] = (t1 - t3) * w3;
            }
        }
        cd* t = src;
        src = dst;
        dst = t;
        len = m;
        st *= 4;
    }
    if (src != x)
        for (std::size_t i = 0; i < n; ++i)
            x[i] = src[i];
}

// The direct transform over the same table.
void dft_by_hand(const std::vector<cd>& in, std::vector<cd>& out, const std::vector<cd>& table)
{
    const std::size_t n = in.size();
    for (std::size_t k = 0; k < n; ++k) {
        cd acc{};
        std::size_t e = 0;
        for (std::size_t j = 0; j < n; ++j) {
            acc += in[j] * table[e];
            e += k;
            if (e >= n)
                e -= n;
        }
        out[k] = acc;
    }
}

void BM_Fft_Core(benchmark::State& state)
{
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    planned p(n);
    const auto data = make_data(n);
    auto x = data;
    for (auto _ : state) {
        x = data;
        s::fft(p.plan, std::span<cd>(x), std::span<cd>(p.work));
        benchmark::DoNotOptimize(x.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n));
}
BENCHMARK(BM_Fft_Core)->Arg(1024)->Arg(4096)->Arg(1000)->Arg(1009)->Arg(65536);

void BM_Fft_HandWrittenStockham(benchmark::State& state)
{
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    planned p(n);
    const auto data = make_data(n);
    auto x = data;
    for (auto _ : state) {
        x = data;
        stockham4_by_hand(p.storage.data(), n, x.data(), p.work.data());
        benchmark::DoNotOptimize(x.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n));
}
BENCHMARK(BM_Fft_HandWrittenStockham)->Arg(1024)->Arg(4096)->Arg(65536);

void BM_Dft_Direct(benchmark::State& state)
{
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    std::vector<cd> table(n), out(n);
    s::twiddles<double>(table);
    const auto data = make_data(n);
    for (auto _ : state) {
        dft_by_hand(data, out, table);
        benchmark::DoNotOptimize(out.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n));
}
BENCHMARK(BM_Dft_Direct)->Arg(1024)->Arg(1000)->Arg(1009);

void BM_Rfft_Core(benchmark::State& state)
{
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    std::vector<cd> st(s::rfft_plan<double>::storage_size(n)), work(s::rfft_plan<double>::work_size(n)), out(n / 2 + 1);
    const s::rfft_plan<double> plan(n, st);
    const auto data = make_data(n);
    std::vector<double> x(n);
    for (std::size_t i = 0; i < n; ++i)
        x[i] = data[i].re;
    for (auto _ : state) {
        s::rfft(plan, std::span<const double>(x), std::span<cd>(out), std::span<cd>(work));
        benchmark::DoNotOptimize(out.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n));
}
BENCHMARK(BM_Rfft_Core)->Arg(1024)->Arg(4096);

void BM_Fft_StaticInPlaceRadix2(benchmark::State& state)
{
    static constexpr s::static_fft_plan<double, 1024> plan{};
    const auto data = make_data(1024);
    auto x = data;
    for (auto _ : state) {
        x = data;
        plan.forward_in_place(std::span<cd, 1024>(x.data(), 1024));
        benchmark::DoNotOptimize(x.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * 1024);
}
BENCHMARK(BM_Fft_StaticInPlaceRadix2);

long double l2(const std::vector<cd>& x)
{
    long double a = 0;
    for (const auto& v : x)
        a += static_cast<long double>(v.re) * v.re + static_cast<long double>(v.im) * v.im;
    return std::sqrt(a);
}

bool within_bound(const std::vector<cd>& a, const std::vector<cd>& b, const s::fft_plan<double>& p, long double norm2)
{
    const std::size_t len = p.uses_bluestein() ? p.layout().m : p.size();
    // The bound is against the exact transform; the direct sum in double has its own error, of the same order, so allow both.
    const long double bound = 2 * 2 * std::numeric_limits<double>::epsilon() * std::ceil(std::log2(static_cast<long double>(len))) * norm2
                              + 4 * std::numeric_limits<double>::epsilon() * std::sqrt(static_cast<long double>(p.size())) * norm2;
    for (std::size_t k = 0; k < a.size(); ++k)
        if (s::abs(a[k] - b[k]) > bound)
            return false;
    return true;
}

bool fair()
{
    // Like for like: core against the hand-written Stockham passes, to the bit.
    for (const std::size_t n : {std::size_t{1024}, std::size_t{4096}, std::size_t{65536}}) {
        planned p(n);
        auto a = make_data(n), b = a;
        s::fft(p.plan, std::span<cd>(a), std::span<cd>(p.work));
        stockham4_by_hand(p.storage.data(), n, b.data(), p.work.data());
        for (std::size_t k = 0; k < n; ++k)
            if (std::bit_cast<std::uint64_t>(a[k].re) != std::bit_cast<std::uint64_t>(b[k].re)
                || std::bit_cast<std::uint64_t>(a[k].im) != std::bit_cast<std::uint64_t>(b[k].im))
                return false;
    }
    // Against the direct transform: the documented bound.
    for (const std::size_t n : {std::size_t{1024}, std::size_t{1000}, std::size_t{1009}}) {
        planned p(n);
        const auto data = make_data(n);
        auto a = data;
        std::vector<cd> b(n), table(n);
        s::twiddles<double>(table);
        s::fft(p.plan, std::span<cd>(a), std::span<cd>(p.work));
        dft_by_hand(data, b, table);
        if (!within_bound(a, b, p.plan, l2(data)))
            return false;
    }
    // The real transform is the complex one's first half; the in-place plan is the Stockham transform.
    {
        planned p(1024);
        auto c = make_data(1024);
        std::vector<double> x(1024);
        for (std::size_t i = 0; i < 1024; ++i) {
            x[i] = c[i].re;
            c[i].im = 0;
        }
        const auto real_data = c;
        s::fft(p.plan, std::span<cd>(c), std::span<cd>(p.work));
        std::vector<cd> st(s::rfft_plan<double>::storage_size(1024)), work(s::rfft_plan<double>::work_size(1024)), h(513);
        const s::rfft_plan<double> plan(1024, st);
        s::rfft(plan, std::span<const double>(x), std::span<cd>(h), std::span<cd>(work));
        std::vector<cd> first(c.begin(), c.begin() + 513);
        if (!within_bound(h, first, p.plan, l2(real_data)))
            return false;
        static constexpr s::static_fft_plan<double, 1024> sp{};
        auto d = real_data;
        sp.forward_in_place(std::span<cd, 1024>(d.data(), 1024));
        if (!within_bound(d, c, p.plan, l2(real_data)))
            return false;
    }
    return true;
}

} // namespace

int main(int argc, char** argv)
{
    if (!fair()) {
        std::fprintf(stderr, "fft benchmark: fairness check FAILED - the pairs disagree; not benchmarking\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
