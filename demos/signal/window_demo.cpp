//
//  window_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/signal/window.hpp>: every window as a compile-time table, periodic and symmetric, filled
//  into a span, and the COLA and NOLA checks for the STFT's usual window and hops. Exits non-zero if a result is not
//  what it documents.
//

#include <snicholls/core/signal/window.hpp>

#include <cstdio>
#include <vector>

int main()
{
    std::printf("snicholls-core signal window demo\n");
    namespace s = snicholls::signal;
    namespace m = snicholls::math;
    bool ok = true;

    //! [window]
    // Whisper's window, written by the compiler: torch.hann_window(400), periodic.
    constexpr auto hann400 = s::make_window<float, 400>(s::hann{});
    static_assert(hann400[0] == 0.0f && hann400[200] == 1.0f);
    // A symmetric Kaiser window for filter design, and the other windows, at run time into any span.
    constexpr auto kaiser31 = s::make_window<double, 31>(s::kaiser{8.6}, s::window_symmetry::symmetric);
    std::vector<double> w(9);
    s::fill_window(std::span<double>(w), s::blackman_harris{}, s::window_symmetry::symmetric);
    const double tukey_mid = s::window_value<double>(s::tukey{0.5}, 4, 9, s::window_symmetry::symmetric);
    // Reconstruction conditions for an overlap-add at a hop.
    const bool cola = s::is_cola<float>(hann400, 200);   // Hann at half overlap sums to a constant
    const bool nola = s::is_nola<float>(hann400, 160);   // Whisper's hop is not COLA, but invertible
    //! [window]

    std::printf("  hann400[100] = %.7f, kaiser31 ends %.6g, blackman-harris(9) centre %.6f\n", static_cast<double>(hann400[100]), kaiser31[0], w[4]);
    std::printf("  rectangular %g, hamming(0) %g, blackman(0) %.3g, gaussian sigma 2 at 2 from centre %.6f, tukey centre %g\n",
                s::window_value<double>(s::rectangular{}, 3, 8), s::window_value<double>(s::hamming{}, 0, 8),
                s::window_value<double>(s::blackman{}, 0, 8), s::window_value<double>(s::gaussian{2.0}, 2, 9, s::window_symmetry::symmetric),
                tukey_mid);
    std::printf("  COLA(hann, 200) %d, NOLA(hann, 160) %d, COLA(hann, 160) %d\n", cola, nola, s::is_cola<float>(hann400, 160));
    ok &= m::near(static_cast<double>(hann400[100]), 0.5, 1e-7) && kaiser31[0] == kaiser31[30] && kaiser31[15] == 1.0;
    ok &= m::near(w[4], 1.0, 1e-15) && tukey_mid == 1.0 && cola && nola && !s::is_cola<float>(hann400, 160);
    ok &= s::window_value<double>(s::hamming{}, 0, 8) == 0.54 - 0.46 && m::near(s::window_value<double>(s::gaussian{2.0}, 2, 9, s::window_symmetry::symmetric), m::exp(-0.5), 1e-15);
    ok &= s::window_denominator(8, s::window_symmetry::periodic) == 8 && m::near(m::bessel_i0(1.0), 1.2660658777520082, 1e-15);

    std::printf(ok ? "  ok\n" : "  FAILED\n");
    return ok ? 0 : 1;
}
