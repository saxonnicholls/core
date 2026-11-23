//
//  prox_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/optim/prox.hpp>: a small LASSO, min ½‖Ax − b‖² + μ‖x‖₁, solved by proximal gradient (ISTA)
//  with prox_l1 and checked by its KKT conditions; then every other prox, the Moreau envelopes and the prox of a
//  conjugate on one point each. Exits non-zero if a result is not what it documents.
//

#include <snicholls/core/optim/prox.hpp>

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

constexpr std::size_t m = 6, n = 4;
// A x_true + noise, x_true = (2, 0, -1, 0)
constexpr std::array<double, m * n> A{1.0, 0.2, 0.0, 0.5,  0.3, 1.0, 0.4, 0.0,  0.0, 0.5, 1.0, 0.2,
                                      0.6, 0.0, 0.3, 1.0,  0.2, 0.7, 0.0, 0.4,  0.9, 0.1, 0.5, 0.3};
constexpr std::array<double, m> b{2.03, 0.18, -0.98, 0.92, 0.41, 1.29};
constexpr double mu = 0.1;

// g = A^T (A x - b)
std::array<double, n> gradient(const std::array<double, n>& x)
{
    std::array<double, m> r{};
    for (std::size_t i = 0; i < m; ++i) {
        r[i] = -b[i];
        for (std::size_t j = 0; j < n; ++j)
            r[i] += A[i * n + j] * x[j];
    }
    std::array<double, n> g{};
    for (std::size_t j = 0; j < n; ++j)
        for (std::size_t i = 0; i < m; ++i)
            g[j] += A[i * n + j] * r[i];
    return g;
}

//! [prox]
// ISTA: x ← prox_{t μ‖·‖₁}(x − t ∇½‖Ax − b‖²), with the step t below 1 / ‖AᵀA‖ (here ‖AᵀA‖ < 5)
std::array<double, n> lasso_ista(int iterations)
{
    std::array<double, n> x{};
    const double t = 0.2;
    for (int k = 0; k < iterations; ++k) {
        const auto g = gradient(x);
        for (std::size_t j = 0; j < n; ++j)
            x[j] -= t * g[j];
        op::prox_l1(std::span<double>(x), t * mu);
    }
    return x;
}

// KKT for the LASSO: -∇ᵢ = μ sign(xᵢ) where xᵢ ≠ 0, |∇ᵢ| ≤ μ where xᵢ = 0
double kkt_violation(const std::array<double, n>& x)
{
    const auto g = gradient(x);
    double worst = 0;
    for (std::size_t j = 0; j < n; ++j) {
        const double v = x[j] != 0 ? std::abs(g[j] + mu * (x[j] > 0 ? 1 : -1)) : std::max(0.0, std::abs(g[j]) - mu);
        worst = std::max(worst, v);
    }
    return worst;
}
//! [prox]

} // namespace

int main()
{
    std::printf("snicholls-core optim prox demo\n");
    bool ok = true;

    const auto x = lasso_ista(2000);
    const double kkt = kkt_violation(x);
    std::printf("  LASSO by ISTA, mu %.2f: x = (%.4f, %.4f, %.4f, %.4f), KKT violation %.1e\n", mu, x[0], x[1], x[2], x[3], kkt);
    ok &= check(__LINE__, kkt < 1e-10 && x[0] > 1.5 && x[2] < -0.5);

    // Every other prox on one point
    std::array<double, 2> v{3, 4};
    op::prox_l2(std::span<double>(v), 1.0);
    ok &= check(__LINE__, near(v[0], 2.4, 1e-15));
    std::array<double, 3> gv{3, 4, 0.1};
    const std::array<std::size_t, 2> groups{2, 1};
    ok &= check(__LINE__, op::prox_group_l2(std::span<double>(gv), std::span<const std::size_t>(groups), 1.0).is_ok() && gv[2] == 0);
    std::array<double, 2> li{3, 1}, li2{3, 1}, ws{};
    ok &= check(__LINE__, op::prox_linf(std::span<double>(li), 1.0, std::span<double>(ws)).is_ok());
    op::prox_linf(std::span<double>(li2), 1.0);
    ok &= check(__LINE__, li[0] == 2 && li == li2);
    std::array<double, 2> h{0.5, 10};
    op::prox_huber(std::span<double>(h), 1.0);
    ok &= check(__LINE__, near(h[0], 0.5 / 3, 1e-15) && h[1] == 8);
    std::array<double, 1> sq{3};
    op::prox_squared_l2(std::span<double>(sq), 1.0);
    ok &= check(__LINE__, sq[0] == 1);
    const double inf = std::numeric_limits<double>::infinity();
    std::array<double, 2> bx{-3, 5};
    const std::array<double, 2> lo{-1, -inf}, hi{1, inf};
    ok &= check(__LINE__, op::prox_box(std::span<double>(bx), lo, hi).is_ok() && bx[0] == -1 && bx[1] == 5);
    op::prox_nonnegative(std::span<double>(bx));
    ok &= check(__LINE__, bx[0] == 0);
    std::array<double, 1> en{3};
    op::prox_elastic_net(std::span<double>(en), 1.0, 1.0);
    ok &= check(__LINE__, en[0] == 1);
    std::array<double, 1> ent{2};
    op::prox_negative_entropy(std::span<double>(ent), 1.0);   // log p + 1 + p = 2: p = 1
    std::array<double, 1> lb{0};
    op::prox_log_barrier(std::span<double>(lb), 4.0);
    std::array<double, 1> q{5};
    const std::array<double, 1> d{2}, c{1};
    ok &= check(__LINE__, op::prox_quadratic(std::span<double>(q), d, c, 1.0).is_ok());
    std::printf("  negative entropy prox of 2: %.15f; log barrier prox of 0 with lambda 4: %.1f; quadratic: %.4f\n", ent[0], lb[0], q[0]);
    ok &= check(__LINE__, near(ent[0], 1, 1e-15) && lb[0] == 2 && near(q[0], 4.0 / 3, 1e-15));

    // Moreau envelopes, and the prox of a conjugate (|x|_1's is the clamp to [-1, 1])
    const std::array<double, 2> pt{3, 0.5};
    const double e1 = op::moreau_envelope_l1(std::span<const double>(pt), 1.0), e2 = op::moreau_envelope_l2(std::span<const double>(pt), 1.0);
    const double e3 = op::moreau_envelope_squared_l2(std::span<const double>(pt), 1.0), e4 = op::moreau_envelope_huber(std::span<const double>(pt), 1.0);
    const double e5 = op::moreau_envelope_box(std::span<const double>(pt), lo, hi, 1.0), e6 = op::moreau_envelope_nonnegative(std::span<const double>(pt), 1.0);
    std::printf("  Moreau envelopes at (3, 0.5): l1 %.4f, l2 %.4f, squared l2 %.4f, huber %.4f, box %.4f, orthant %.4f\n", e1, e2, e3, e4, e5, e6);
    ok &= check(__LINE__, near(e1, 2.625, 1e-15) && e5 == 2 && e6 == 0 && e2 > 0 && e3 > 0 && e4 > 0);
    std::array<double, 3> cv{-3, 0.5, 2}, cws{};
    ok &= check(__LINE__, op::prox_conjugate([](std::span<double> y, double t) noexcept { op::prox_l1(y, t); }, std::span<double>(cv), 0.7, std::span<double>(cws)).is_ok());
    std::printf("  prox of |x|_1's conjugate at (-3, 0.5, 2): (%.3f, %.3f, %.3f)\n", cv[0], cv[1], cv[2]);
    ok &= check(__LINE__, near(cv[0], -1, 1e-15) && near(cv[1], 0.5, 1e-15) && near(cv[2], 1, 1e-15));

    std::printf(ok ? "prox demo: ok\n" : "prox demo: FAILED\n");
    return ok ? 0 : 1;
}
