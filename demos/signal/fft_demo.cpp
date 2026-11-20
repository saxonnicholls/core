//
//  fft_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/signal/fft.hpp>: a transform computed entirely by the compiler; run-time plans over caller
//  memory for a power of two, a mixed-radix length and a prime (Bluestein); the normalisations; a real transform and its
//  inverse; two dimensions; the in-place power-of-two plan; and the backward passes. Exits non-zero if a result is not
//  what it documents.
//

#include <snicholls/core/signal/fft.hpp>

#include <array>
#include <cstddef>
#include <cstdio>
#include <vector>

int main()
{
    std::printf("snicholls-core signal fft demo\n");
    namespace s = snicholls::signal;
    namespace m = snicholls::math;
    using cd = s::complex<double>;
    bool ok = true;

    //! [compile_time]
    // A cosine of two cycles in 16 samples: its transform, computed by the compiler, is 8 at bins 2 and 14.
    constexpr auto x = [] {
        std::array<cd, 16> v{};
        for (std::size_t i = 0; i < 16; ++i)
            v[i] = {m::cos(2 * m::pi * 2.0 * static_cast<double>(i) / 16.0), 0.0};
        return v;
    }();
    constexpr auto X = s::fft(x);   // the plan (twiddles, bit reversal) and the transform, all in constant evaluation
    static_assert(m::near(X[2].re, 8.0, 1e-13) && m::near(X[14].re, 8.0, 1e-13) && m::near(s::abs(X[5]), 0.0, 1e-13));
    // A plan for a length known when the program is written: tables as std::arrays, written by a consteval constructor.
    static constexpr s::static_fft_plan<float, 1024> plan1024{};
    // Real input: the half spectrum.
    constexpr auto H = s::rfft(std::array<double, 8>{1, 1, 1, 1, 0, 0, 0, 0});
    static_assert(H.size() == 5 && H[0].re == 4.0);
    //! [compile_time]

    //! [run_time]
    // Run-time plans over memory the caller owns; nothing allocates while transforming.
    const std::size_t n = 1000;   // 4 x 2 x 5 x 5 x 5: mixed radix
    std::vector<cd> storage(s::fft_plan<double>::storage_size(n)), work(s::fft_plan<double>::work_size(n)), data(n);
    const s::fft_plan<double> plan(n, storage);
    for (std::size_t i = 0; i < n; ++i)
        data[i] = {i == 3 ? 1.0 : 0.0, 0.0};                        // an impulse at 3
    s::fft(plan, std::span<cd>(data), std::span<cd>(work));          // X_k = e^(-2 pi i 3k / n)
    const cd bin1 = data[1];
    s::ifft(plan, std::span<cd>(data), std::span<cd>(work));         // back to the impulse
    // A prime length runs through Bluestein's algorithm; the call is the same.
    std::vector<cd> st_p(s::fft_plan<double>::storage_size(1009)), w_p(s::fft_plan<double>::work_size(1009)), d_p(1009, cd{1, 0});
    const s::fft_plan<double> prime(1009, st_p);
    s::fft(prime, std::span<cd>(d_p), std::span<cd>(w_p), s::fft_norm::ortho);   // a constant: sqrt(n) at DC
    // Real samples: n / 2 + 1 bins through a complex transform of half the length.
    std::vector<cd> st_r(s::rfft_plan<double>::storage_size(12)), w_r(s::rfft_plan<double>::work_size(12)), half(7);
    const s::rfft_plan<double> rplan(12, st_r);
    std::vector<double> samples(12), back(12);
    for (std::size_t i = 0; i < 12; ++i)
        samples[i] = m::sin(2 * m::pi * static_cast<double>(i) / 12.0);
    s::rfft(rplan, std::span<const double>(samples), std::span<cd>(half), std::span<cd>(w_r));   // -6i at bin 1
    s::irfft(rplan, std::span<const cd>(half), std::span<double>(back), std::span<cd>(w_r));
    //! [run_time]

    std::printf("  compile time: X[2] = %g, X[14] = %g; rfft of a step: DC %g, bin 1 %g%+gi\n", X[2].re, X[14].re, H[0].re, H[1].re, H[1].im);
    std::printf("  n = 1000 (radices 4 2 5 5 5): X[1] of an impulse at 3 = %.15f%+.15fi; back: x[3] = %g\n", bin1.re, bin1.im, data[3].re);
    std::printf("  n = 1009 (Bluestein, m = %zu): ortho DC of ones = %.12f = sqrt(1009)\n", prime.layout().m, d_p[0].re);
    std::printf("  rfft of one sine period in 12: bin 1 = %.12g%+.12gi; irfft round trip x[3] = %.15g\n", half[1].re, half[1].im, back[3]);
    ok &= m::near(H[1].re, 1.0, 1e-14) && m::near(H[1].im, -2.414213562373095, 1e-14);
    ok &= m::near(bin1.re, m::cos(2 * m::pi * 3 / 1000.0), 1e-15) && m::near(bin1.im, -m::sin(2 * m::pi * 3 / 1000.0), 1e-15);
    ok &= m::near(data[3].re, 1.0, 1e-15) && m::near(data[4].re, 0.0, 1e-15) && plan.layout().factor_count == 5 && !plan.uses_bluestein();
    ok &= prime.uses_bluestein() && m::near(d_p[0].re, m::sqrt(1009.0), 1e-12) && m::near(s::abs(d_p[7]), 0.0, 1e-12);
    ok &= m::near(half[1].im, -6.0, 1e-14) && m::near(back[3], 1.0, 1e-15) && rplan.bins() == 7;

    // The 1,024-point float plan from the compiler, in place (radix 2 over its bit-reversal table) and with a work buffer.
    std::vector<s::complex<float>> f(1024), g(1024), fw(plan1024.work_size);
    for (std::size_t i = 0; i < 1024; ++i)
        f[i] = g[i] = {static_cast<float>((i * 37) % 11) - 5.0f, 0.0f};
    plan1024.forward_in_place(std::span<s::complex<float>, 1024>(f.data(), 1024));
    plan1024.forward(std::span<s::complex<float>, 1024>(g.data(), 1024), std::span<s::complex<float>, plan1024.work_size>(fw.data(), plan1024.work_size));
    float worst = 0;
    for (std::size_t k = 0; k < 1024; ++k)
        worst = s::abs(f[k] - g[k]) > worst ? s::abs(f[k] - g[k]) : worst;
    std::printf("  1024-point float plan: in place vs Stockham, max difference %.3g; bit_reversal[1] = %u\n", static_cast<double>(worst),
                plan1024.bit_reversal()[1]);
    ok &= worst < 1e-3f && plan1024.bit_reversal()[1] == 512;

    // Two dimensions: a 3 x 4 array of ones has all its energy at (0, 0).
    std::vector<cd> st3(s::fft_plan<double>::storage_size(3)), st4(s::fft_plan<double>::storage_size(4)), grid(12, cd{1, 0});
    const s::fft_plan<double> p3(3, st3), p4(4, st4);
    std::vector<cd> w2(s::fftn_work_size<double>(std::array<s::fft_plan<double>, 2>{p3, p4}));
    s::fft2(p3, p4, std::span<cd>(grid), std::span<cd>(w2));
    std::printf("  fft2 of a 3 x 4 array of ones: (0, 0) = %g, (1, 2) = %.3g\n", grid[0].re, s::abs(grid[6]));
    ok &= m::near(grid[0].re, 12.0, 1e-14) && m::near(s::abs(grid[6]), 0.0, 1e-14);

    // Backward passes: the adjoint of the forward transform (here of rfft, with the half spectrum's weights).
    std::vector<cd> grad_spectrum(7, cd{0, 0});
    grad_spectrum[0] = {1, 0};   // L = Re X_0 = sum x: dL/dx = 1 everywhere
    std::vector<double> grad_x(12);
    s::rfft_backward(rplan, std::span<const cd>(grad_spectrum), std::span<double>(grad_x), std::span<cd>(w_r));
    std::printf("  rfft_backward of dL/dX_0 = 1: dL/dx = %g ... %g\n", grad_x[0], grad_x[11]);
    ok &= m::near(grad_x[0], 1.0, 1e-15) && m::near(grad_x[11], 1.0, 1e-15);

    std::printf(ok ? "  ok\n" : "  FAILED\n");
    return ok ? 0 : 1;
}
