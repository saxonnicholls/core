//
//  dct_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/signal/dct.hpp>: the orthonormal DCT-II compacting a smooth sequence's energy into its
//  first coefficients, its inverse, the type IV, the sines, a DCT at compile time, and the DCT-II's backward pass.
//  Exits non-zero if a result is not what it documents.
//

#include <snicholls/core/signal/dct.hpp>

#include <array>
#include <cstddef>
#include <cstdio>
#include <vector>

int main()
{
    std::printf("snicholls-core signal dct demo\n");
    namespace s = snicholls::signal;
    namespace m = snicholls::math;
    using cd = s::complex<double>;
    bool ok = true;

    //! [dct]
    // A smooth ramp of 32 values: its orthonormal DCT-II puts nearly all of its energy in the first few coefficients.
    const std::size_t n = 32;
    std::vector<double> x(n), y(n), back(n), first(4);
    for (std::size_t i = 0; i < n; ++i)
        x[i] = static_cast<double>(i) / static_cast<double>(n);
    std::vector<cd> storage(s::fft_plan<double>::storage_size(n));
    const s::fft_plan<double> plan(n, storage);   // types II and III: a plan of n
    std::vector<cd> work(s::dct_work_size(plan));
    s::dct2(plan, std::span<const double>(x), std::span<double>(y), std::span<cd>(work), s::dct_norm::ortho);
    s::dct2(plan, std::span<const double>(x), std::span<double>(first), std::span<cd>(work), s::dct_norm::ortho);   // only the first 4
    s::dct3(plan, std::span<const double>(y), std::span<double>(back), std::span<cd>(work), s::dct_norm::ortho);    // the inverse
    // Type IV runs on a plan of 2n.
    std::vector<cd> st4(s::fft_plan<double>::storage_size(2 * n));
    const s::fft_plan<double> plan4(2 * n, st4);
    std::vector<cd> work4(s::dct_work_size(plan4));
    std::vector<double> y4(n), y4back(n);
    s::dct4(plan4, std::span<const double>(x), std::span<double>(y4), std::span<cd>(work4), s::dct_norm::ortho);
    s::dct4(plan4, std::span<const double>(y4), std::span<double>(y4back), std::span<cd>(work4), s::dct_norm::ortho);   // its own inverse
    //! [dct]

    double energy = 0, kept = 0;
    for (std::size_t k = 0; k < n; ++k) {
        energy += y[k] * y[k];
        kept += k < 4 ? y[k] * y[k] : 0.0;
    }
    std::printf("  orthonormal DCT-II of a ramp: %.4f%% of the energy in 4 of 32 coefficients; first %.6f == %.6f\n", 100 * kept / energy, first[0], y[0]);
    std::printf("  dct3(dct2(x)) x[7] = %.15g, dct4(dct4(x)) x[7] = %.15g\n", back[7], y4back[7]);
    ok &= kept / energy > 0.999 && first[3] == y[3] && m::near(back[7], x[7], 1e-15) && m::near(y4back[7], x[7], 1e-15);

    // The sines: a DST-II of a sequence alternated in sign is the reversed DCT-II.
    std::vector<double> alt(n), st2(n), st3(n), st4v(n), st4back(n);
    for (std::size_t i = 0; i < n; ++i)
        alt[i] = i % 2 ? -x[i] : x[i];
    s::dst2(plan, std::span<const double>(alt), std::span<double>(st2), std::span<cd>(work));
    std::vector<double> c2(n);
    s::dct2(plan, std::span<const double>(x), std::span<double>(c2), std::span<cd>(work));
    s::dst3(plan, std::span<const double>(st2), std::span<double>(st3), std::span<cd>(work));   // DST-III(DST-II(v)) = 2n v
    s::dst4(plan4, std::span<const double>(x), std::span<double>(st4v), std::span<cd>(work4), s::dct_norm::ortho);
    s::dst4(plan4, std::span<const double>(st4v), std::span<double>(st4back), std::span<cd>(work4), s::dct_norm::ortho);
    std::printf("  dst2(alternated x)[0] %.12g = dct2(x)[31] %.12g; dst3(dst2(v)) / 2n = %.12g\n", st2[0], c2[31], st3[5] / 64);
    ok &= m::near(st2[0], c2[31], 1e-13) && m::near(st3[5] / 64, alt[5], 1e-14) && m::near(st4back[9], x[9], 1e-15);

    // At compile time: the DCT-II of a constant is a single coefficient.
    constexpr auto dc = [] {
        std::array<double, 8> v{1, 1, 1, 1, 1, 1, 1, 1}, out{};
        std::array<cd, 16> w{};
        s::dct2(s::static_fft_plan_v<double, 8>.plan(), std::span<const double>(v), std::span<double>(out), std::span<cd>(w), s::dct_norm::ortho);
        return out;
    }();
    static_assert(m::near(dc[0], m::sqrt(8.0), 1e-14) && m::near(dc[1], 0.0, 1e-14));

    // The backward pass of the truncated transform: dL/dx for L = first[1].
    std::vector<double> gy{0, 1, 0, 0}, gx(n);
    s::dct2_backward(plan, std::span<const double>(gy), std::span<double>(gx), std::span<cd>(work), s::dct_norm::ortho);
    std::printf("  dct2_backward: d first[1] / d x[0] = %.12g = sqrt(2/32) cos(pi / 64)\n", gx[0]);
    ok &= m::near(gx[0], m::sqrt(2.0 / 32) * m::cos(m::pi / 64), 1e-15);

    std::printf(ok ? "  ok\n" : "  FAILED\n");
    return ok ? 0 : 1;
}
