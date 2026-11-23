//
//  line_search_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/optim/line_search.hpp>: Armijo backtracking and a strong-Wolfe search along steepest
//  descent on Rosenbrock's function, from the same point; a search that must expand a short first step; one along a
//  direction that does not descend; and one on an unbounded function. Exits non-zero if a result is not what it
//  documents.
//

#include <snicholls/core/optim/line_search.hpp>

#include <array>
#include <cmath>
#include <cstdio>
#include <span>

namespace {

namespace op = snicholls::optim;
using cspan = std::span<const double>;

//! [line_search]
// Rosenbrock's function with its gradient
double rosenbrock(cspan x, std::span<double> g)
{
    const double a = x[1] - x[0] * x[0], b = 1 - x[0];
    g[0] = -400 * x[0] * a - 2 * b;
    g[1] = 200 * a;
    return 100 * a * a + b * b;
}

bool line_search_example()
{
    const std::array<double, 2> x{-1.2, 1};
    std::array<double, 2> g{}, d{}, trial{}, trial_g{};
    const double f = rosenbrock(x, g);
    d = {-g[0], -g[1]};                        // steepest descent
    const double slope = g[0] * d[0] + g[1] * d[1];   // φ'(0) < 0

    // Armijo: values only; the unit step overshoots badly and is halved until f falls enough
    const auto values_only = [](cspan v) {
        std::array<double, 2> unused{};
        return rosenbrock(v, unused);
    };
    const auto a = op::backtracking_armijo(values_only, x, d, f, slope, trial);
    std::printf("  Armijo:      step %.3e, f %.4f -> %.4f, %zu evaluations\n", a.step, f, a.value, a.evaluations);

    // Strong Wolfe: also |φ'(α)| ≤ 0.9 |φ'(0)|, found by bracketing and cubic interpolation; the trial storage
    // holds the new point and its gradient, ready for the next iteration
    const auto w = op::strong_wolfe(rosenbrock, x, d, f, slope, trial, trial_g);
    std::printf("  strong Wolfe: step %.3e, f %.4f -> %.4f, φ' %.3e -> %.3e, %zu evaluations\n", w.step, f, w.value, slope, w.derivative, w.evaluations);

    // A first step far too short is expanded (doubling) until the minimum along the line is bracketed
    const auto fg = [](cspan v, std::span<double> grad) {
        grad[0] = 2 * (v[0] - 100);
        return (v[0] - 100) * (v[0] - 100);
    };
    const std::array<double, 1> x1{0}, d1{1};
    std::array<double, 1> t1{}, tg1{};
    const auto e = op::strong_wolfe(fg, x1, d1, 1e4, -200.0, t1, tg1, {1e-3, 1e-4, 0.1, 2, 1e10, 40});
    std::printf("  expanded:    step %.4f from 1e-3, %zu evaluations\n", e.step, e.evaluations);

    // Uphill: nothing is evaluated
    const auto up = op::strong_wolfe(fg, x1, d1, 1e4, 200.0, t1, tg1);
    // Unbounded below: the search stops at max_step and says so
    const auto unbounded = op::strong_wolfe([](cspan v, std::span<double> grad) { grad[0] = -1; return -v[0]; }, x1, d1, 0.0, -1.0, t1, tg1, {1, 1e-4, 0.9, 10, 1e8, 40});
    std::printf("  uphill:      %s; unbounded: %s at step %.0e\n", up.status == op::line_search_status::not_descent_direction ? "not a descent direction" : "?",
                unbounded.status == op::line_search_status::step_at_maximum ? "step_at_maximum" : "?", unbounded.step);

    return a.satisfied() && a.value <= f + 1e-4 * a.step * slope && w.satisfied() && w.value <= f + 1e-4 * w.step * slope && std::abs(w.derivative) <= 0.9 * std::abs(slope) &&
           e.satisfied() && std::abs(e.step - 100) < 10 && up.evaluations == 0 && unbounded.status == op::line_search_status::step_at_maximum;
}
//! [line_search]

} // namespace

int main()
{
    std::printf("snicholls-core line_search demo\n");
    const bool ok = line_search_example();
    std::printf(ok ? "line_search demo: ok\n" : "line_search demo: FAILED\n");
    return ok ? 0 : 1;
}
