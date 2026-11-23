//
//  cones_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/optim/cones.hpp>: the projection step of a conic solver over a product of cones (zero,
//  orthant, second-order, PSD, exponential, power) and onto its dual, Moreau's decomposition checked on it; each
//  cone and set on its own with its membership test (box, rotated cone, the duals, the simplex by Condat and by
//  Michelot, the l1 and l-infinity balls, the capped simplex); svec and smat; and the Nesterov-Todd scalings of an
//  interior-point step. Exits non-zero if a result is not what it documents.
//

#include <snicholls/core/optim/cones.hpp>

#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <span>

namespace {

namespace op = snicholls::optim;

bool near(double a, double b, double tol) { return std::abs(a - b) <= tol; }

// A documented result that does not hold is reported with its line
bool check(int line, bool holds)
{
    if (!holds)
        std::printf("  not as documented at line %d\n", line);
    return holds;
}

//! [cones]
// SCS's cone layout: equality rows, then inequalities, then a second-order, a 2 x 2 PSD (svec), an exponential and a
// power cone, one after another in one vector.
constexpr std::array<op::cone_kind, 6> layout{{{op::cone::zero, 1, 0},
                                               {op::cone::nonnegative, 2, 0},
                                               {op::cone::second_order, 3, 0},
                                               {op::cone::psd, 3, 0},
                                               {op::cone::exponential, 3, 0},
                                               {op::cone::power, 3, 0.3}}};

// The projection step of the iteration, and the check a solver makes of its output
bool project_step(std::span<double> x, std::span<double> polar)
{
    std::array<double, op::psd_workspace_size(2)> ws{};   // the largest PSD block's workspace
    const std::span<const op::cone_kind> k(layout);
    std::array<double, 15> minus{};
    for (std::size_t i = 0; i < x.size(); ++i)
        minus[i] = -x[i];
    // Moreau: x = P_K(x) - P_K*(-x), and the two parts are orthogonal
    if (!op::project(k, x, std::span<double>(ws)) || !op::project_dual(k, std::span<double>(minus), std::span<double>(ws)))
        return false;
    for (std::size_t i = 0; i < x.size(); ++i)
        polar[i] = -minus[i];
    return op::in_cone(k, std::span<const double>(x), 1e-12, std::span<double>(ws));
}
//! [cones]

} // namespace

int main()
{
    std::printf("snicholls-core optim cones demo\n");
    bool ok = true;

    std::array<double, 15> x{0.4, -1, 2, 0.5, 3, 4, 1, 2.8284271247461903, 1, 1, 1, 1, 1, -0.5, 2}, polar{};
    const auto x0 = x;
    ok &= check(__LINE__, project_step(std::span<double>(x), std::span<double>(polar)));
    double ip = 0, worst = 0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        ip += x[i] * polar[i];
        worst = std::max(worst, std::abs(x[i] + polar[i] - x0[i]));
    }
    std::printf("  product of %zu cones, dimension %zu: P(x) = (%.3f | %.3f %.3f | %.3f %.3f %.3f | %.3f %.3f %.3f | %.4f %.4f %.4f | %.4f %.4f %.4f)\n",
                layout.size(), op::product_dimension(std::span<const op::cone_kind>(layout)), x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9],
                x[10], x[11], x[12], x[13], x[14]);
    std::printf("  Moreau: |x - P_K(x) - P_K°(x)| = %.1e, <P_K(x), P_K°(x)> = %.1e\n", worst, ip);
    ok &= check(__LINE__, worst < 1e-12 && std::abs(ip) < 1e-12 && x[0] == 0 && x[1] == 0 && x[2] == 2 && near(x[3], 2.75, 1e-15));
    // The PSD block [[1, 2], [2, 1]] has eigenvalues 3 and -1: its projection is [[1.5, 1.5], [1.5, 1.5]]
    std::array<double, 4> m{};
    ok &= check(__LINE__, op::smat(std::span<const double>(x).subspan(6, 3), std::span<double>(m)).is_ok() && near(m[0], 1.5, 1e-12) && near(m[1], 1.5, 1e-12));
    std::array<double, 3> back{};
    ok &= check(__LINE__, op::svec(std::span<const double>(m), std::span<double>(back)).is_ok() && near(back[1], x[7], 1e-15));
    ok &= check(__LINE__, *op::psd_order(6) == 3 && op::svec_size(3) == 6 && op::product_workspace_size(std::span<const op::cone_kind>(layout)) == 10);

    // Each cone on its own
    std::array<double, 3> z{1, 2, 3};
    op::project_zero(std::span<double>(z));
    op::project_free(std::span<double>(z));
    ok &= check(__LINE__, op::in_zero_cone(std::span<const double>(z), 0.0) && op::in_free_cone(std::span<const double>(z), 0.0));
    std::array<double, 2> o{-1, 1};
    op::project_nonnegative(std::span<double>(o));
    ok &= check(__LINE__, op::in_nonnegative_cone(std::span<const double>(o), 0.0) && o[0] == 0);
    const double inf = std::numeric_limits<double>::infinity();
    std::array<double, 3> b{-3, 0.5, 9};
    const std::array<double, 3> lo{-1, -inf, 0}, hi{1, 0, inf};
    ok &= check(__LINE__, op::project_box(std::span<double>(b), lo, hi).is_ok() && op::in_box(std::span<const double>(b), lo, hi, 0.0) && b[0] == -1 && b[1] == 0 && b[2] == 9);
    op::project_box(std::span<double>(b), -2.0, 2.0);
    ok &= check(__LINE__, b[2] == 2);
    std::array<double, 3> s{0, 3, 4};
    op::project_second_order(std::span<double>(s));
    ok &= check(__LINE__, op::in_second_order_cone(std::span<const double>(s), 1e-15) && s[0] == 2.5);
    std::array<double, 3> r{-1, 2, 1};
    ok &= check(__LINE__, op::project_rotated_second_order(std::span<double>(r)).is_ok() && op::in_rotated_second_order_cone(std::span<const double>(r), 1e-12));
    std::array<double, 3> e{1, 1, 1}, ed{1, 1, 1};
    op::project_exponential(std::span(e));
    op::project_dual_exponential(std::span(ed));
    std::printf("  exponential cone: P(1, 1, 1) = (%.6f, %.6f, %.6f), onto the dual (%.6f, %.6f, %.6f)\n", e[0], e[1], e[2], ed[0], ed[1], ed[2]);
    ok &= check(__LINE__, near(e[0], 0.42630617, 1e-6) && op::in_exponential_cone(std::span<const double, 3>(e), 1e-12) && op::in_dual_exponential_cone(std::span<const double, 3>(ed), 1e-12));
    std::array<double, 3> p{1, -0.5, 2}, pd{1, -0.5, 2};
    op::project_power(std::span(p), 0.3);
    op::project_dual_power(std::span(pd), 0.3);
    std::printf("  power cone, alpha 0.3: P(1, -0.5, 2) = (%.6f, %.6f, %.6f)\n", p[0], p[1], p[2]);
    ok &= check(__LINE__, near(p[2], 0.74004137, 1e-6) && op::in_power_cone(std::span<const double, 3>(p), 0.3, 1e-12) && op::in_dual_power_cone(std::span<const double, 3>(pd), 0.3, 1e-12));
    std::array<double, 3> psd{1, 2 * std::sqrt(2.0), 1};
    ok &= check(__LINE__, op::project_psd_fixed<2>(std::span<double, 3>(psd)).is_ok() && op::in_psd_cone_fixed<2>(std::span<const double, 3>(psd), 1e-12));
    std::array<double, 10> ws{};
    std::array<double, 3> psd_rt{1, 2 * std::sqrt(2.0), 1};
    ok &= check(__LINE__, op::project_psd(std::span<double>(psd_rt), std::span<double>(ws)).is_ok() && op::in_psd_cone(std::span<const double>(psd_rt), 1e-12, std::span<double>(ws)));

    // Sets: the simplex (Condat with a workspace, Michelot without), the balls, the capped simplex
    std::array<double, 4> w1{0.5, 0.2, -1, 0.9}, w2 = w1, sws{};
    ok &= check(__LINE__, op::project_simplex(std::span<double>(w1), 1.0, std::span<double>(sws)).is_ok() && op::project_simplex(std::span<double>(w2), 1.0).is_ok());
    std::printf("  simplex: (%.3f, %.3f, %.3f, %.3f)\n", w1[0], w1[1], w1[2], w1[3]);
    ok &= check(__LINE__, op::in_simplex(std::span<const double>(w1), 1.0, 1e-15) && near(w1[3], w2[3], 1e-15) && near(w1[0], 0.3, 1e-15) && w1[2] == 0);
    std::array<double, 3> l1{3, -1, 0.2}, l1b = l1, lws{};
    ok &= check(__LINE__, op::project_l1_ball(std::span<double>(l1), 2.0, std::span<double>(lws)).is_ok() && op::project_l1_ball(std::span<double>(l1b), 2.0).is_ok());
    ok &= check(__LINE__, op::in_l1_ball(std::span<const double>(l1), 2.0, 1e-15) && near(l1[0], 2, 1e-15) && near(l1[0], l1b[0], 1e-15));
    std::array<double, 2> li{3, -0.5};
    ok &= check(__LINE__, op::project_linf_ball(std::span<double>(li), 1.0).is_ok() && op::in_linf_ball(std::span<const double>(li), 1.0, 0.0) && li[0] == 1);
    std::array<double, 4> cs{0.9, 0.8, 0.1, -2};
    ok &= check(__LINE__, op::project_capped_simplex(std::span<double>(cs), 1.5, 0.6).is_ok() && op::in_capped_simplex(std::span<const double>(cs), 1.5, 0.6, 1e-14));
    std::printf("  capped simplex (cap 0.6, total 1.5): (%.3f, %.3f, %.3f, %.3f)\n", cs[0], cs[1], cs[2], cs[3]);

    // Nesterov-Todd: W z = W^-1 s, the scaled point lambda of an interior-point step
    const std::array<double, 3> sl{2, 0.5, -1}, du{1.5, -0.3, 0.2};
    std::array<double, 3> wbar{}, wz = du, ws_ = sl;
    const auto eta = op::nt_scaling_second_order(std::span<const double>(sl), std::span<const double>(du), std::span<double>(wbar));
    ok &= check(__LINE__, eta.is_ok());
    op::nt_apply_second_order(std::span<const double>(wbar), *eta, std::span<double>(wz));
    op::nt_apply_inverse_second_order(std::span<const double>(wbar), *eta, std::span<double>(ws_));
    std::printf("  Nesterov-Todd, second-order cone: eta %.6f, W z = (%.6f, %.6f, %.6f), W^-1 s = (%.6f, %.6f, %.6f)\n", *eta, wz[0], wz[1], wz[2], ws_[0], ws_[1], ws_[2]);
    for (std::size_t i = 0; i < 3; ++i)
        ok &= check(__LINE__, near(wz[i], ws_[i], 1e-14));
    const std::array<double, 2> so{4, 1}, zo{1, 9};
    std::array<double, 2> wo{}, a = zo, c = so;
    ok &= check(__LINE__, op::nt_scaling_nonnegative(std::span<const double>(so), std::span<const double>(zo), std::span<double>(wo)).is_ok());
    op::nt_apply_nonnegative(std::span<const double>(wo), std::span<double>(a));
    op::nt_apply_inverse_nonnegative(std::span<const double>(wo), std::span<double>(c));
    ok &= check(__LINE__, near(a[0], 2, 1e-15) && near(a[1], 3, 1e-15) && near(c[0], 2, 1e-15) && near(c[1], 3, 1e-15));
    ok &= check(__LINE__, op::dual_of(op::cone::power) == op::cone::dual_power);

    std::printf(ok ? "cones demo: ok\n" : "cones demo: FAILED\n");
    return ok ? 0 : 1;
}
