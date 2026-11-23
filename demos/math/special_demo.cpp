//
//  special_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/math/special.hpp>: Lambert W solving x e^x = c on both branches, softplus
//  and the logistic function in their tails, log Phi at -40 where log(Phi) underflows, and the
//  entropy, Kullback-Leibler, Huber and xlogy atoms. Exits non-zero if a value is not the one
//  documented.
//

#include <snicholls/core/math/constexpr_math.hpp>
#include <snicholls/core/math/special.hpp>

#include <cstdio>

int main()
{
    std::printf("snicholls-core special demo\n");
    namespace math = snicholls::math;
    bool ok = true;

    //! [special]
    // x e^x = 5 has the one real solution W0(5); x e^x = -0.2 has two, W0(-0.2) and W-1(-0.2).
    constexpr double x = math::lambert_w0(5.0);
    constexpr double y0 = math::lambert_w0(-0.2), y1 = math::lambert_wm1(-0.2);
    static_assert(math::near(x * math::exp(x), 5.0, 1e-15));

    // softplus and the logistic function never overflow: log(1 + e^800) is 800, 1 / (1 + e^800) is 0.
    constexpr double sp_hi = math::softplus(800.0), sp_lo = math::softplus(-40.0);   // 800, e^-40
    constexpr double lg_lo = math::logistic(-700.0), llg = math::log_logistic(-800.0);   // e^-700, -800

    // log Phi(-40): Phi(-40) is 3.7e-350, below the smallest double, so log(normal_cdf(-40)) is -inf.
    constexpr double lp = math::log_normcdf(-40.0);   // -804.6084420137538
    //! [special]

    std::printf("  x e^x = 5: x = %.16f; x e^x = -0.2: %.16f and %.16f\n", x, y0, y1);
    ok &= math::near(x, 1.3267246652422002, 1e-15);   // Python decimal, Halley's iteration at 50 digits
    ok &= math::near(y0 * math::exp(y0), -0.2, 1e-15) && math::near(y1 * math::exp(y1), -0.2, 1e-15) && y1 < -1 && y0 > -1;
    ok &= math::lambert_w0(-0.36787944117144233) == -1.0 && math::is_nan(math::lambert_w0(-1.0));
    ok &= math::lambert_wm1(0.0) == -math::inf && math::is_nan(math::lambert_wm1(0.5));

    std::printf("  softplus(800) %.1f, softplus(-40) %.6g, logistic(-700) %.6g, log_logistic(-800) %.1f\n", sp_hi, sp_lo, lg_lo, llg);
    ok &= sp_hi == 800.0 && math::near(sp_lo / 4.248354255291589e-18, 1.0, 1e-15) && llg == -800.0;
    ok &= math::near(lg_lo / 9.85967654375977e-305, 1.0, 1e-14) && math::logistic(0.0) == 0.5;

    std::printf("  log Phi(-40) = %.13f; log(normal_cdf(-40)) = %g\n", lp, math::log(math::normal_cdf(-40.0)));
    ok &= math::near(lp, -804.6084420137538, 1e-15) && math::log(math::normal_cdf(-40.0)) == -math::inf;
    ok &= math::log_normcdf(0.0) == math::log(0.5) && math::near(math::log_normcdf(-5.0), -15.064998393988725, 1e-15);
    ok &= math::near(math::log_normcdf(3.0), -0.0013508099647481938, 1e-15);

    // The atoms of a convex model, with CVXPY's conventions at 0.
    const double p[] = {0.5, 0.5, 0.0}, q[] = {0.25, 0.5, 0.25};
    double H = 0, D = 0, KL = 0;
    for (int i = 0; i < 3; ++i) {
        H += math::entr(p[i]);           // 0 log 0 = 0
        D += math::rel_entr(p[i], q[i]);
        KL += math::kl_div(p[i], q[i]);  // = D here: both sum to one
    }
    std::printf("  entropy %.6f, relative entropy %.6f, KL %.6f, huber(3) %.1f, xlogy(0, 0) %.1f\n", H, D, KL, math::huber(3.0),
                math::xlogy(0.0, 0.0));
    ok &= math::near(H, math::log(2.0), 1e-15) && math::near(D, 0.5 * math::log(2.0), 1e-15) && math::near(KL, D, 1e-15);
    ok &= math::huber(0.5) == 0.25 && math::huber(3.0) == 5.0 && math::xlogy(0.0, 0.0) == 0.0 && math::rel_entr(1.0, 0.0) == math::inf;

    std::printf(ok ? "special demo: ok\n" : "special demo: FAILED\n");
    return ok ? 0 : 1;
}
