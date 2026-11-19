//
//  hilbert_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/signal/hilbert.hpp>: the analytic signal of an amplitude-modulated carrier, its envelope
//  and instantaneous phase, the Hilbert transform alone, and the backward pass. Exits non-zero if a result is not
//  what it documents.
//

#include <snicholls/core/signal/hilbert.hpp>

#include <cstddef>
#include <cstdio>
#include <vector>

int main()
{
    std::printf("snicholls-core signal hilbert demo\n");
    namespace s = snicholls::signal;
    namespace m = snicholls::math;
    using cd = s::complex<double>;
    bool ok = true;

    //! [hilbert]
    // A 50-cycle carrier modulated by a 2-cycle envelope, 1,000 samples: the envelope comes back from the analytic signal.
    const std::size_t n = 1000;
    std::vector<double> x(n), env(n), h(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double t = static_cast<double>(i) / static_cast<double>(n);
        x[i] = (1.0 + 0.6 * m::sin(2 * m::pi * 2 * t)) * m::cos(2 * m::pi * 50 * t);
    }
    std::vector<cd> storage(s::fft_plan<double>::storage_size(n));
    const s::fft_plan<double> plan(n, storage);
    std::vector<cd> work(n + plan.work_size()), z(n);
    s::analytic_signal(plan, std::span<const double>(x), std::span<cd>(z), std::span<cd>(work));   // x + i H(x)
    s::envelope(plan, std::span<const double>(x), std::span<double>(env), std::span<cd>(work));      // |x + i H(x)|
    s::hilbert_transform(plan, std::span<const double>(x), std::span<double>(h), std::span<cd>(work));
    //! [hilbert]

    const double expected = 1.0 + 0.6 * m::sin(2 * m::pi * 2 * 0.125);
    std::printf("  envelope at 125: %.12f (expected %.12f); real part is x: %.3g; H(x)[0] = %.6f\n", env[125], expected, m::abs(z[7].re - x[7]), h[0]);
    std::printf("  instantaneous phase advances 2 pi 50 / 1000 per sample: %.12f\n", s::arg(z[11] * s::conj(z[10])));
    ok &= m::near(env[125], expected, 1e-12) && m::near(z[7].re, x[7], 1e-14) && m::near(h[3], z[3].im, 0.0);
    ok &= m::near(s::arg(z[11] * s::conj(z[10])), 2 * m::pi * 50 / 1000, 1e-9);

    // Backward: L = Im z[105] has dL/dx = the Hilbert transform's row 105, so <dL/dx, x> = L.
    std::vector<cd> g(n, cd{0, 0});
    g[105] = {0, 1};
    std::vector<double> gx(n);
    s::analytic_signal_backward(plan, std::span<cd>(g), std::span<double>(gx), std::span<cd>(work));
    double dot = 0;
    for (std::size_t i = 0; i < n; ++i)
        dot += gx[i] * x[i];
    std::printf("  analytic_signal_backward: <dL/dx, x> = %.12f = Im z[105] = %.12f\n", dot, z[105].im);
    ok &= m::near(dot, z[105].im, 1e-12) && m::abs(z[105].im) > 0.1;

    std::printf(ok ? "  ok\n" : "  FAILED\n");
    return ok ? 0 : 1;
}
