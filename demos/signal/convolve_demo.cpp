//
//  convolve_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/signal/convolve.hpp>: a moving average by direct and FFT convolution in each mode, a delay
//  found by correlation, and a long signal filtered block by block by overlap-add and overlap-save. Exits non-zero if
//  a result is not what it documents.
//

#include <snicholls/core/signal/convolve.hpp>

#include <array>
#include <cstddef>
#include <cstdio>
#include <vector>

int main()
{
    std::printf("snicholls-core signal convolve demo\n");
    namespace s = snicholls::signal;
    namespace m = snicholls::math;
    using cd = s::complex<double>;
    bool ok = true;

    //! [convolve]
    // A three-point moving average, direct (also at compile time) and through the FFT.
    static constexpr std::array<double, 6> x{3, 6, 9, 3, 0, 3};
    static constexpr std::array<double, 3> avg{1.0 / 3, 1.0 / 3, 1.0 / 3};
    constexpr auto full = [] {
        std::array<double, s::convolve_size(6, 3, s::convolve_mode::full)> out{};
        s::convolve<double>(x, avg, out);
        return out;
    }();
    std::vector<double> same(6), valid(4), fast(8);
    s::convolve<double>(x, avg, same, s::convolve_mode::same);
    s::convolve<double>(x, avg, valid, s::convolve_mode::valid);
    std::vector<cd> storage(s::fft_plan<double>::storage_size(8));
    const s::fft_plan<double> plan(8, storage);   // at least 6 + 3 - 1
    std::vector<cd> work(s::fft_convolve_work_size(plan));
    s::fft_convolve(plan, std::span<const double>(x), std::span<const double>(avg), std::span<double>(fast), std::span<cd>(work));

    // Where does a pattern occur? Correlation peaks at its lag.
    const std::vector<double> signal{0, 0, 0, 1, 2, 1, 0, 0}, pattern{1, 2, 1};
    std::vector<double> corr(10), corr_fft(10);
    s::correlate<double>(signal, pattern, corr);   // lag = index - (pattern.size() - 1)
    std::vector<cd> st16(s::fft_plan<double>::storage_size(16));
    const s::fft_plan<double> p16(16, st16);
    std::vector<cd> w16(s::fft_convolve_work_size(p16));
    s::fft_correlate(p16, std::span<const double>(signal), std::span<const double>(pattern), std::span<double>(corr_fft), std::span<cd>(w16));
    //! [convolve]

    std::printf("  full:");
    for (const double v : full)
        std::printf(" %g", v);
    std::printf("\n  same: %g %g %g %g %g %g; valid: %g %g %g %g\n", same[0], same[1], same[2], same[3], same[4], same[5], valid[0], valid[1], valid[2], valid[3]);
    std::size_t lag = 0;
    for (std::size_t i = 0; i < corr.size(); ++i)
        lag = corr[i] > corr[lag] ? i : lag;
    std::printf("  the pattern starts at %zu (correlation %g, through the FFT %.12g)\n", lag - 2, corr[lag], corr_fft[lag]);
    ok &= m::near(full[2], 6.0, 1e-15) && m::near(valid[0], 6.0, 1e-15) && m::near(same[1], 6.0, 1e-15) && m::near(fast[3], full[3], 1e-14);
    ok &= lag - 2 == 3 && corr[lag] == 6.0 && m::near(corr_fft[lag], 6.0, 1e-14);

    // A long signal through a 63-tap filter in blocks of 256: overlap-add (the full convolution), overlap-save (the causal output).
    std::vector<double> longx(10000), taps(63);
    for (std::size_t i = 0; i < longx.size(); ++i)
        longx[i] = m::sin(0.001 * static_cast<double>(i * i % 7919));
    for (std::size_t i = 0; i < taps.size(); ++i)
        taps[i] = 1.0 / (1.0 + static_cast<double>(i));
    std::vector<cd> st256(s::fft_plan<double>::storage_size(256));
    const s::fft_plan<double> p256(256, st256);
    std::vector<cd> bw(s::block_filter_work_size(p256));
    std::vector<double> oa(longx.size() + taps.size() - 1), os(longx.size()), direct(oa.size());
    s::overlap_add(p256, std::span<const double>(taps), std::span<const double>(longx), std::span<double>(oa), std::span<cd>(bw));
    s::overlap_save(p256, std::span<const double>(taps), std::span<const double>(longx), std::span<double>(os), std::span<cd>(bw));
    s::convolve<double>(longx, taps, direct);
    double worst = 0;
    for (std::size_t i = 0; i < os.size(); ++i)
        worst = m::abs(oa[i] - direct[i]) + m::abs(os[i] - direct[i]) > worst ? m::abs(oa[i] - direct[i]) + m::abs(os[i] - direct[i]) : worst;
    std::printf("  10,000 samples, 63 taps, blocks of 256: overlap-add and overlap-save within %.3g of the direct convolution\n", worst);
    ok &= worst < 1e-12;

    std::printf(ok ? "  ok\n" : "  FAILED\n");
    return ok ? 0 : 1;
}
