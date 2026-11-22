//
//  assignment_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/combinatorics/assignment.hpp>: tracks matched to detections by minimum total cost, with a
//  gate (+inf) and a price for a missed detection; a reusable solver for run-time sizes, a tall matrix, maximum weight,
//  and each error. Exits non-zero if a result is not what it documents.
//

#include <snicholls/core/combinatorics/assignment.hpp>

#include <snicholls/core/linalg/fixed.hpp>

#include <array>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

namespace {

namespace cb = snicholls::combinatorics;
namespace la = snicholls::linalg;

int failures = 0;
void expect(bool ok, const char* what)
{
    std::printf("  %-64s %s\n", what, ok ? "ok" : "FAILED");
    if (!ok)
        ++failures;
}

} // namespace

int main()
{
    std::printf("assignment: the minimum-cost matching of rows to columns\n");
    constexpr double inf = std::numeric_limits<double>::infinity();

    //! [assignment]
    // Three tracks (rows) against two detections (columns 0, 1), costs from the gating distance. Columns 2..4 are
    // one dummy per track: the price of a missed detection on its own row, +inf (forbidden) elsewhere
    constexpr double miss = 4.0;
    const la::fixed_matrix<double, 3, 5> cost{{1.0, 6.0, miss, inf, inf,
                                               inf, 2.5, inf, miss, inf,      // track 1 is outside detection 0's gate
                                               2.0, 3.0, inf, inf, miss}};
    const auto a = cb::linear_assignment(cost);
    // Track 0 takes detection 0, track 1 detection 1, track 2 is missed: 1 + 2.5 + 4
    //! [assignment]
    expect(a && a->row_to_col == std::array<std::int32_t, 3>{0, 1, 4} && a->total == 7.5, "linear_assignment: tracks 0, 1 detected, track 2 missed, total 7.5");
    if (a)
        std::printf("    row_to_col = {%d, %d, %d}, total %.1f\n", a->row_to_col[0], a->row_to_col[1], a->row_to_col[2], a->total);

    // A matrix with more rows than columns: every column is assigned, two rows stay at -1
    const auto tall = cb::linear_assignment(la::fixed_matrix<double, 4, 2>{{9, 6, 1, 2, 8, 9, 7, 0}});
    expect(tall && tall->row_to_col == std::array<std::int32_t, 4>{-1, 0, -1, 1} && tall->total == 1.0, "tall matrix: solved through its transpose, free rows are -1");

    // Maximum weight: the same matrix read as scores
    const la::fixed_matrix<double, 3, 3> score{{4, 1, 3, 2, 0, 5, 3, 2, 2}};
    const auto best = cb::linear_assignment(score, cb::objective::maximise);
    const auto cheapest = cb::linear_assignment(score);
    expect(best && best->total == 11.0 && cheapest && cheapest->total == 5.0, "objective::maximise gives 11, minimise gives 5");

    // A solver sized at compile time, for matrices known at run time, reused every scan
    cb::assignment_solver<double, 8, 8> solver;
    std::vector<std::int32_t> out(8);
    const std::vector<double> scan1{3, 1, 2, 2, 3, 1};   // 2 x 3
    const std::vector<double> scan2{5, 4, 1, 9};         // 2 x 2
    const auto s1 = solver.solve(scan1, 2, 3, out);
    expect(s1 && *s1 == 2.0 && out[0] == 1 && out[1] == 2, "assignment_solver: scan 1 (2 x 3) total 2");
    const auto s2 = solver.solve(scan2, 2, 2, out);
    expect(s2 && *s2 == 5.0 && out[0] == 1 && out[1] == 0, "assignment_solver: reused for scan 2 (2 x 2) total 5");
    const auto empty = solver.solve({}, 0, 0, out);
    expect(empty && *empty == 0.0, "an empty matrix is a success with total 0");

    // The errors
    const auto blocked = cb::linear_assignment(la::fixed_matrix<double, 2, 2>{{inf, inf, 1, 2}});
    expect(!blocked && blocked.unwrap_err_unchecked() == cb::assignment_error::infeasible, "a row with every pair forbidden: infeasible");
    const auto nan = cb::linear_assignment(la::fixed_matrix<double, 1, 2>{{std::numeric_limits<double>::quiet_NaN(), 1}});
    expect(!nan && nan.unwrap_err_unchecked() == cb::assignment_error::invalid_cost, "a NaN cost: invalid_cost");
    const auto big = solver.solve(scan1, 9, 1, out);
    expect(!big && big.unwrap_err_unchecked() == cb::assignment_error::too_large, "more rows than the solver's maximum: too_large");

    std::printf(failures == 0 ? "assignment demo: ok\n" : "assignment demo: FAILED\n");
    return failures == 0 ? 0 : 1;
}
