//
//  scaling_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/optim/scaling.hpp>: a badly scaled equality-constrained QP equilibrated by modified Ruiz
//  (OSQP's form), its KKT norms before and after, the scaled KKT system solved by MINRES and the solution unscaled
//  to the original's (checked against the closed form); the power-of-two variant restoring the data to the bit; and
//  the rectangular form on one matrix. Exits non-zero if a result is not what it documents.
//

#include <snicholls/core/optim/scaling.hpp>

#include <snicholls/core/linalg/krylov.hpp>
#include <snicholls/core/sparse/csc.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>

namespace {

namespace la = snicholls::linalg;
namespace op = snicholls::optim;
namespace sp = snicholls::sparse;

// The KKT matrix's column norms, as Ruiz sees them: max(‖P(:, j)‖∞, ‖A(:, j)‖∞), then ‖A(i, :)‖∞
template <std::size_t N, std::size_t M>
std::array<double, N + M> kkt_norms(const sp::csc_view<double>& p, const sp::csc_view<double>& a)
{
    std::array<double, N> pn{}, an{};
    std::array<double, M> rn{};
    sp::symmetric_upper_inf_norms(p, std::span<double>(pn));
    sp::column_inf_norms(a, std::span<double>(an));
    sp::row_inf_norms(a, std::span<double>(rn));
    std::array<double, N + M> out{};
    for (std::size_t j = 0; j < N; ++j)
        out[j] = pn[j] > an[j] ? pn[j] : an[j];
    for (std::size_t i = 0; i < M; ++i)
        out[N + i] = rn[i];
    return out;
}

bool scaling_example()
{
    bool ok = true;
    //! [scaling]
    // minimise ½ (1e4 x₁² + 1e-2 x₂²) subject to x₁ + 1e3 x₂ = 1   (l = u = 1)
    std::array<std::int32_t, 3> pp{0, 1, 2}, pi{0, 1}, ap{0, 1, 2}, ai{0, 0};
    std::array<double, 2> pv{1e4, 1e-2}, av{1, 1e3}, q{0, 0};
    const sp::csc_view<double> p{2, 2, pp, pi, pv}, a{1, 2, ap, ai, av};
    const auto before = kkt_norms<2, 1>(p, a);

    std::array<double, 2> d{};
    std::array<double, 1> e{}, l{1}, u{1};
    std::array<double, op::ruiz_workspace(2, 1)> work{};
    double c = 0;
    if (!op::ruiz_equilibrate(p, a, q, d, e, c, 10, work))
        return false;
    op::scale_bounds(std::span<const double>(e), l, u);
    const auto after = kkt_norms<2, 1>(p, a);

    // Solve the scaled KKT system [P̄ Āᵀ; Ā 0] [x̄; ȳ] = [-q̄; l̄] (symmetric indefinite) by MINRES
    const std::array<double, 9> kkt{pv[0], 0, av[0], 0, pv[1], av[1], av[0], av[1], 0};
    const std::array<double, 3> rhs{-q[0], -q[1], l[0]};
    std::array<double, 3> sol{};
    std::array<double, la::minres_workspace(3)> mw{};
    const auto r = la::minres(la::make_dense_operator(std::span<const double>(kkt), 3, 3), rhs, sol, mw, {1e-14});

    // Back to the original problem: x = D x̄, y = E ȳ / c
    std::array<double, 2> x{sol[0], sol[1]};
    std::array<double, 1> y{sol[2]};
    op::unscale_solution(std::span<const double>(d), e, c, x, y);
    //! [scaling]
    std::printf("  KKT norms before: %.1e %.1e %.1e\n", before[0], before[1], before[2]);
    std::printf("  KKT norms after:  %.3f %.3f %.3f (cost scale c = %.3e)\n", after[0], after[1], after[2], c);
    // Closed form: x₁ = -y / 1e4, x₂ = -1e5 y, y = -1 / (1e8 + 1e-4)
    const double ye = -1 / (1e8 + 1e-4), x1 = -ye / 1e4, x2 = -1e5 * ye;
    std::printf("  MINRES: %zu iterations; x = (%.6e, %.6e), y = %.6e\n", r.iterations, x[0], x[1], y[0]);
    std::printf("  closed form:        x = (%.6e, %.6e), y = %.6e\n", x1, x2, ye);
    ok &= r.converged() && std::abs(x[0] - x1) < 1e-9 * std::abs(x1) && std::abs(x[1] - x2) < 1e-9 * std::abs(x2) && std::abs(y[0] - ye) < 1e-9 * std::abs(ye);
    for (const double v : before)
        ok &= v == 1e4 || v == 1e3 || v == 1;
    for (const double v : after)
        ok &= v > 0.5 && v < 2;

    // Power-of-two factors: unscale_data restores the data to the bit
    std::array<double, 2> pv2{1e4, 1e-2}, av2{1, 1e3}, q2{3, -7};
    const sp::csc_view<double> p2{2, 2, pp, pi, pv2}, a2{1, 2, ap, ai, av2};
    if (!op::ruiz_equilibrate(p2, a2, q2, d, e, c, 10, work, {1e-4, 1e4, true, true}))
        return false;
    op::unscale_data(p2, a2, q2, std::span<const double>(d), e, c);
    const bool exact = pv2[0] == 1e4 && pv2[1] == 1e-2 && av2[0] == 1 && av2[1] == 1e3 && q2[0] == 3 && q2[1] == -7;
    std::printf("  power of two: d = (%g, %g), e = %g, c = %g; restored to the bit: %s\n", d[0], d[1], e[0], c, exact ? "yes" : "no");
    ok &= exact;

    // One rectangular matrix, as SCS equilibrates its data: Ā = R A S
    std::array<std::int32_t, 3> rp{0, 2, 4};
    std::array<std::int32_t, 4> ri{0, 1, 0, 1};
    std::array<double, 4> rv{1e3, 2, 5e-3, 4e-6};
    const sp::csc_view<double> m{2, 2, rp, ri, rv};
    std::array<double, 2> rs{}, cs{}, rn{}, cn{};
    std::array<double, op::ruiz_rectangular_workspace(2, 2)> rw{};
    if (!op::ruiz_equilibrate_rectangular(m, rs, cs, 20, rw))
        return false;
    sp::row_inf_norms(m, std::span<double>(rn));
    sp::column_inf_norms(m, std::span<double>(cn));
    std::printf("  rectangular: row norms %.4f %.4f, column norms %.4f %.4f\n", rn[0], rn[1], cn[0], cn[1]);
    for (const double v : {rn[0], rn[1], cn[0], cn[1]})
        ok &= std::abs(v - 1) < 1e-3;
    return ok;
}

} // namespace

int main()
{
    std::printf("snicholls-core scaling demo\n");
    const bool ok = scaling_example();
    std::printf(ok ? "scaling demo: ok\n" : "scaling demo: FAILED\n");
    return ok ? 0 : 1;
}
