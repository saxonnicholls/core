//
//  lbfgs_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/optim/lbfgs.hpp>: L-BFGS on the 10-dimensional extended Rosenbrock function over storage
//  sized by lbfgs_workspace; a maximum-likelihood fit (a logistic regression) stopped by the value tolerance; and the
//  history and two-loop recursion used directly, with a fixed-dimension history, to show the secant equation.
//  Exits non-zero if a result is not what it documents.
//

#include <snicholls/core/optim/lbfgs.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <span>
#include <vector>

namespace {

namespace op = snicholls::optim;
using cspan = std::span<const double>;

//! [lbfgs]
// The extended Rosenbrock function: independent pairs, minimum 0 at all ones
double rosenbrock(cspan x, std::span<double> g)
{
    double f = 0;
    for (std::size_t i = 0; i + 1 < x.size(); i += 2) {
        const double a = x[i + 1] - x[i] * x[i], b = 1 - x[i];
        f += 100 * a * a + b * b;
        g[i] = -400 * x[i] * a - 2 * b;
        g[i + 1] = 200 * a;
    }
    return f;
}

bool lbfgs_example()
{
    // 1. Ten variables, seven pairs of memory; the caller owns every byte
    constexpr std::size_t n = 10, memory = 7;
    std::vector<double> x(n), work(op::lbfgs_workspace(n, memory));
    for (std::size_t i = 0; i < n; ++i)
        x[i] = i % 2 == 0 ? -1.2 : 1.0;
    const auto r = op::lbfgs_minimize<memory>(rosenbrock, std::span<double>(x), {1e-8}, work);
    std::printf("  Rosenbrock, n = 10: %s after %zu iterations, %zu evaluations, f = %.2e, x0 = %.8f\n", r.converged() ? "converged" : "failed", r.iterations, r.evaluations,
                r.value, x[0]);

    // 2. Logistic regression by maximum likelihood: P(yᵢ = 1) = σ(w₀ + w₁ tᵢ), stopped when f stops moving
    const std::array<double, 8> t{-3, -2, -1, -0.5, 0.5, 1, 2, 3}, y{0, 0, 1, 0, 1, 0, 1, 1};
    const auto nll = [&](cspan w, std::span<double> g) {
        double f = 0;
        g[0] = g[1] = 0;
        for (std::size_t i = 0; i < t.size(); ++i) {
            const double z = w[0] + w[1] * t[i], p = 1 / (1 + std::exp(-z));
            f += std::log(1 + std::exp(z)) - y[i] * z;
            g[0] += p - y[i];
            g[1] += (p - y[i]) * t[i];
        }
        return f;
    };
    std::array<double, 2> w{};
    std::array<double, op::lbfgs_workspace(2, 4)> w_work{};
    const auto fit = op::lbfgs_minimize<4>(nll, std::span<double>(w), {1e-12, 1e-14}, w_work);
    std::printf("  logistic fit: w = (%.4f, %.4f), -log L = %.6f, %zu iterations\n", w[0], w[1], fit.value, fit.iterations);

    // 3. The pieces: a fixed-dimension history and the two-loop recursion; H y = s for the newest pair
    op::lbfgs_history<double, 3, 2> history;
    const std::array<double, 2> s1{1, 0}, y1{4, 1}, s2{0.5, 1}, y2{3, 5};
    history.push(s1, y1);
    history.push(s2, y2);
    std::array<double, 2> d{};
    op::two_loop(history, y2, d);   // d = -H y₂
    std::printf("  two loop: -H y = (%.6f, %.6f), s = (%.6f, %.6f)\n", d[0], d[1], s2[0], s2[1]);

    return r.converged() && std::abs(x[0] - 1) < 1e-7 && fit.converged() && fit.gradient_norm < 1e-6 && w[1] > 0 && std::abs(d[0] + s2[0]) < 1e-12 &&
           std::abs(d[1] + s2[1]) < 1e-12;
}
//! [lbfgs]

} // namespace

int main()
{
    std::printf("snicholls-core lbfgs demo\n");
    const bool ok = lbfgs_example();
    std::printf(ok ? "lbfgs demo: ok\n" : "lbfgs demo: FAILED\n");
    return ok ? 0 : 1;
}
