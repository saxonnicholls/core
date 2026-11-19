//
//  complex_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/signal/complex.hpp>: complex arithmetic in a static_assert, polar form, square roots, and
//  the twiddle factors with their exact quarter turns. Exits non-zero if a result is not what it documents.
//

#include <snicholls/core/signal/complex.hpp>

#include <cstdio>

int main()
{
    std::printf("snicholls-core signal::complex demo\n");
    namespace s = snicholls::signal;
    namespace m = snicholls::math;
    bool ok = true;

    //! [complex]
    using cd = s::complex<double>;
    constexpr cd a{3, 4}, b{1, -2};
    static_assert(a * b == cd{11, -2} && a + b == cd{4, 2} && s::conj(a) == cd{3, -4} && s::norm(a) == 25.0);
    constexpr double r = s::abs(a);                    // 5, at compile time
    constexpr cd q = a / b;                            // (3 + 4i) / (1 - 2i) = -1 + 2i
    constexpr cd root = s::sqrt(cd{-9, 0});            // 3i
    constexpr cd turn = s::twiddle<double>(1, 4);      // e^(-2 pi i / 4) = -i, exactly
    constexpr cd eighth = s::twiddle<double>(1, 8);    // (1 - i) / sqrt 2
    const cd p = s::polar(2.0, m::pi / 3) * s::expi(-m::pi / 3);   // 2, the angles cancel
    //! [complex]

    std::printf("  |3+4i| = %g, (3+4i)/(1-2i) = %g%+gi, sqrt(-9) = %g%+gi\n", r, q.re, q.im, root.re, root.im);
    std::printf("  twiddle(1, 4) = %g%+gi, twiddle(1, 8) = %.17g%+.17gi\n", turn.re, turn.im, eighth.re, eighth.im);
    std::printf("  2 e^(i pi/3) e^(-i pi/3) = %.15g%+.3gi, arg(i) = %.15g, times_i(1) = %gi\n", p.re, p.im, s::arg(cd{0, 1}),
                s::times_i(cd{1, 0}).im);
    ok &= r == 5.0 && m::near(q.re, -1.0, 1e-15) && m::near(q.im, 2.0, 1e-15) && m::near(root.im, 3.0, 1e-15) && root.re == 0.0;
    ok &= turn == cd{0, -1} && eighth.re == -eighth.im && m::near(eighth.re, 0.70710678118654752, 1e-16);
    ok &= m::near(p.re, 2.0, 1e-15) && m::near(p.im, 0.0, 1e-15) && m::near(s::arg(cd{0, 1}), m::pi / 2, 1e-15);
    ok &= s::times_i(cd{1, 0}) == cd{0, 1} && -a == cd{-3, -4} && a - b == cd{2, 6} && 2.0 * b == b * 2.0;

    std::printf(ok ? "  ok\n" : "  FAILED\n");
    return ok ? 0 : 1;
}
