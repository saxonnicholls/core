//
//  assignment_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/combinatorics/assignment.hpp>: optimality against brute force over every injective map
//  (square, wide and tall matrices, minimise and maximise, random and tied costs), forbidden pairs, each error, the
//  empty and one-element cases, the dummy-column construction for missed detections, solver reuse, and constexpr
//  and noexcept claims.
//

#include <snicholls/core/combinatorics/assignment.hpp>

#include <snicholls/core/linalg/fixed.hpp>
#include <snicholls/core/stats/random.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <span>
#include <utility>
#include <vector>

namespace {

namespace cb = snicholls::combinatorics;
namespace la = snicholls::linalg;
namespace st = snicholls::stats;

constexpr double inf = std::numeric_limits<double>::infinity();

// The best total over every injective map from the smaller side to the larger, by enumerating permutations of the
// larger side and reading its first min(R, C) entries; +inf when no finite matching exists
double brute_force(const std::vector<double>& cost, std::size_t rows, std::size_t cols, bool maximise)
{
    const bool tall = rows > cols;
    const std::size_t n = tall ? cols : rows, m = tall ? rows : cols;
    std::vector<std::size_t> perm(m);
    std::iota(perm.begin(), perm.end(), std::size_t{0});
    const double worst = maximise ? -inf : inf;
    double best = worst;
    do {
        double total = 0;
        bool allowed = true;
        for (std::size_t i = 0; i < n; ++i) {
            const double c = tall ? cost[perm[i] * cols + i] : cost[i * cols + perm[i]];
            allowed = allowed && std::isfinite(c);
            total += c;
        }
        if (allowed)
            best = maximise ? std::max(best, total) : std::min(best, total);
    } while (std::next_permutation(perm.begin(), perm.end()));
    return best;
}

// The assignment is a partial injection that covers the smaller side, and its reported total is its cost
void expect_valid(const std::vector<double>& cost, std::size_t rows, std::size_t cols, std::span<const std::int32_t> row_to_col, double total)
{
    std::vector<int> used(cols, 0);
    std::size_t assigned = 0;
    double sum = 0;
    for (std::size_t r = 0; r < rows; ++r) {
        const std::int32_t c = row_to_col[r];
        if (c < 0) {
            EXPECT_EQ(c, -1);
            continue;
        }
        ASSERT_LT(static_cast<std::size_t>(c), cols);
        EXPECT_EQ(used[static_cast<std::size_t>(c)]++, 0) << "column " << c << " used twice";
        ++assigned;
        sum += cost[r * cols + static_cast<std::size_t>(c)];
    }
    EXPECT_EQ(assigned, std::min(rows, cols));
    EXPECT_NEAR(sum, total, 1e-9 * (1 + std::abs(total)));
}

TEST(CombinatoricsAssignment, SolvesTheTextbookThreeByThree)
{
    // SciPy's linear_sum_assignment documentation example: rows -> columns 1, 0, 2, total 5
    const la::fixed_matrix<double, 3, 3> cost{{4, 1, 3, 2, 0, 5, 3, 2, 2}};
    const auto a = cb::linear_assignment(cost);
    ASSERT_TRUE(a);
    EXPECT_EQ(a->row_to_col, (std::array<std::int32_t, 3>{1, 0, 2}));
    EXPECT_EQ(a->total, 5.0);
}

TEST(CombinatoricsAssignment, MatchesBruteForceOnRandomMatricesOfEveryShape)
{
    st::wyrand g(2026);
    cb::assignment_solver<double, 7, 7> solver;
    std::vector<std::int32_t> out(7);
    for (std::size_t rows = 1; rows <= 6; ++rows)
        for (std::size_t cols = 1; cols <= 6; ++cols)
            for (int trial = 0; trial < 6; ++trial)
                for (const bool maximise : {false, true}) {
                    std::vector<double> cost(rows * cols);
                    for (double& c : cost)
                        c = std::floor(st::uniform_real(g) * 20) - 5;   // small integers: many ties
                    const auto s = solver.solve(cost, rows, cols, out, maximise ? cb::objective::maximise : cb::objective::minimise);
                    ASSERT_TRUE(s) << rows << "x" << cols;
                    EXPECT_EQ(*s, brute_force(cost, rows, cols, maximise)) << rows << "x" << cols << " trial " << trial;
                    expect_valid(cost, rows, cols, std::span<const std::int32_t>(out).first(rows), *s);
                }
}

TEST(CombinatoricsAssignment, MatchesBruteForceWithRealCostsAndForbiddenPairs)
{
    st::wyrand g(7);
    cb::assignment_solver<double, 6, 6> solver;
    std::vector<std::int32_t> out(6);
    int feasible = 0, infeasible = 0;
    for (int trial = 0; trial < 1000; ++trial) {
        const std::size_t rows = 1 + st::uniform_index(g, 6), cols = 1 + st::uniform_index(g, 6);
        std::vector<double> cost(rows * cols);
        for (double& c : cost)
            c = st::uniform_real(g) < 0.5 ? inf : st::uniform_real(g) * 100;
        const double expected = brute_force(cost, rows, cols, false);
        const auto s = solver.solve(cost, rows, cols, out);
        if (expected == inf) {
            ASSERT_FALSE(s);
            EXPECT_EQ(s.unwrap_err_unchecked(), cb::assignment_error::infeasible);
            ++infeasible;
        } else {
            ASSERT_TRUE(s) << trial;
            EXPECT_NEAR(*s, expected, 1e-9) << trial;
            expect_valid(cost, rows, cols, std::span<const std::int32_t>(out).first(rows), *s);
            ++feasible;
        }
    }
    EXPECT_GT(feasible, 100);
    EXPECT_GT(infeasible, 100);
}

TEST(CombinatoricsAssignment, AWideMatrixAssignsEveryRowAndLeavesColumnsFree)
{
    const la::fixed_matrix<double, 2, 4> cost{{9, 1, 8, 7, 6, 2, 9, 0}};
    const auto a = cb::linear_assignment(cost);
    ASSERT_TRUE(a);
    EXPECT_EQ(a->row_to_col, (std::array<std::int32_t, 2>{1, 3}));
    EXPECT_EQ(a->total, 1.0);
}

TEST(CombinatoricsAssignment, ATallMatrixAssignsEveryColumnAndLeavesRowsUnassigned)
{
    // The transpose of the wide case: rows 1 and 3 take columns 0 and 1, rows 0 and 2 stay at -1
    const la::fixed_matrix<double, 4, 2> cost{{9, 6, 1, 2, 8, 9, 7, 0}};
    const auto a = cb::linear_assignment(cost);
    ASSERT_TRUE(a);
    EXPECT_EQ(a->row_to_col, (std::array<std::int32_t, 4>{-1, 0, -1, 1}));
    EXPECT_EQ(a->total, 1.0);
}

TEST(CombinatoricsAssignment, MaximiseFindsTheHeaviestMatchingAndMinusInfinityForbids)
{
    const la::fixed_matrix<double, 3, 3> weight{{4, 1, 3, 2, 0, 5, 3, 2, 2}};
    const auto a = cb::linear_assignment(weight, cb::objective::maximise);
    ASSERT_TRUE(a);
    // Brute force: 4 + 5 + 2 = 11 (0, 2, 1) is the best of the six permutations
    EXPECT_EQ(a->row_to_col, (std::array<std::int32_t, 3>{0, 2, 1}));
    EXPECT_EQ(a->total, 11.0);

    const la::fixed_matrix<double, 2, 2> gated{{-inf, 1, 5, 2}};
    const auto b = cb::linear_assignment(gated, cb::objective::maximise);
    ASSERT_TRUE(b);
    EXPECT_EQ(b->row_to_col, (std::array<std::int32_t, 2>{1, 0}));
    EXPECT_EQ(b->total, 6.0);
}

TEST(CombinatoricsAssignment, PlusInfinityForbidsAPairWhenMinimising)
{
    const la::fixed_matrix<double, 2, 2> cost{{0, inf, 1, 100}};   // the one finite matching is the diagonal
    const auto a = cb::linear_assignment(cost);
    ASSERT_TRUE(a);
    EXPECT_EQ(a->row_to_col, (std::array<std::int32_t, 2>{0, 1}));
    EXPECT_EQ(a->total, 100.0);
}

TEST(CombinatoricsAssignment, ReportsInfeasibleWhenAllowedColumnsRunOut)
{
    const la::fixed_matrix<double, 2, 2> row_blocked{{inf, inf, 1, 2}};
    const auto a = cb::linear_assignment(row_blocked);
    ASSERT_FALSE(a);
    EXPECT_EQ(a.unwrap_err_unchecked(), cb::assignment_error::infeasible);

    const la::fixed_matrix<double, 3, 3> shared{{1, inf, inf, 2, inf, inf, 3, 4, 5}};   // rows 0 and 1 both need column 0
    const auto b = cb::linear_assignment(shared);
    ASSERT_FALSE(b);
    EXPECT_EQ(b.unwrap_err_unchecked(), cb::assignment_error::infeasible);

    const la::fixed_matrix<double, 3, 2> tall{{inf, inf, inf, inf, 1, 2}};   // only one row can be used
    const auto c = cb::linear_assignment(tall);
    ASSERT_FALSE(c);
    EXPECT_EQ(c.unwrap_err_unchecked(), cb::assignment_error::infeasible);
}

TEST(CombinatoricsAssignment, ReportsInvalidCostForNanAndMinusInfinity)
{
    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (const double bad : {nan, -inf}) {
        const la::fixed_matrix<double, 2, 2> cost{{1, 2, bad, 4}};
        const auto a = cb::linear_assignment(cost);
        ASSERT_FALSE(a);
        EXPECT_EQ(a.unwrap_err_unchecked(), cb::assignment_error::invalid_cost);
    }
    // Maximising negates: +inf is then minus infinity
    const la::fixed_matrix<double, 2, 2> weight{{1, inf, 3, 4}};
    const auto b = cb::linear_assignment(weight, cb::objective::maximise);
    ASSERT_FALSE(b);
    EXPECT_EQ(b.unwrap_err_unchecked(), cb::assignment_error::invalid_cost);
}

TEST(CombinatoricsAssignment, ReportsTooLargeBeyondTheMaximaOrForShortSpans)
{
    cb::assignment_solver<double, 3, 4> solver;
    std::vector<double> cost(20, 1.0);
    std::vector<std::int32_t> out(5);
    const auto expect_too_large = [](const auto& r) {
        ASSERT_FALSE(r);
        EXPECT_EQ(r.unwrap_err_unchecked(), cb::assignment_error::too_large);
    };
    expect_too_large(solver.solve(cost, 4, 2, out));                                                // rows > MaxRows
    expect_too_large(solver.solve(cost, 2, 5, out));                                                // cols > MaxCols
    expect_too_large(solver.solve(std::span<const double>(cost).first(5), 2, 3, out));              // span < rows * cols
    expect_too_large(solver.solve(cost, 3, 3, std::span<std::int32_t>(out).first(2)));              // row_to_col short
    EXPECT_TRUE(solver.solve(std::span<const double>(cost).first(12), 3, 4, out));                  // exactly at the maxima
}

TEST(CombinatoricsAssignment, AnEmptySideIsATrivialSuccess)
{
    cb::assignment_solver<double, 3, 3> solver;
    std::vector<std::int32_t> out{7, 7, 7};
    const auto none = solver.solve(std::span<const double>{}, 0, 0, out);
    ASSERT_TRUE(none);
    EXPECT_EQ(*none, 0.0);
    const auto no_cols = solver.solve(std::span<const double>{}, 3, 0, out);
    ASSERT_TRUE(no_cols);
    EXPECT_EQ(*no_cols, 0.0);
    EXPECT_EQ(out, (std::vector<std::int32_t>{-1, -1, -1}));
}

TEST(CombinatoricsAssignment, OneElementIsAssignedOrInfeasible)
{
    const auto a = cb::linear_assignment(la::fixed_matrix<double, 1, 1>{{-3.5}});
    ASSERT_TRUE(a);
    EXPECT_EQ(a->row_to_col[0], 0);
    EXPECT_EQ(a->total, -3.5);
    const auto b = cb::linear_assignment(la::fixed_matrix<double, 1, 1>{{inf}});
    ASSERT_FALSE(b);
    EXPECT_EQ(b.unwrap_err_unchecked(), cb::assignment_error::infeasible);
}

TEST(CombinatoricsAssignment, AllTiedCostsGiveAPermutationAtTheCommonTotal)
{
    la::fixed_matrix<double, 5, 5> cost{};
    for (auto& c : cost.elements)
        c = 2.0;
    const auto a = cb::linear_assignment(cost);
    ASSERT_TRUE(a);
    EXPECT_EQ(a->total, 10.0);
    auto cols = a->row_to_col;
    std::sort(cols.begin(), cols.end());
    EXPECT_EQ(cols, (std::array<std::int32_t, 5>{0, 1, 2, 3, 4}));
}

TEST(CombinatoricsAssignment, DummyColumnsLetARowStayUnassignedAtAPrice)
{
    // Two tracks, one detection. The documented construction: one dummy column per row, the miss price (5) on its
    // own row and +inf elsewhere. Track 0 is near the detection (1), track 1 far (20): track 1 takes its miss
    const la::fixed_matrix<double, 2, 3> cost{{1, 5, inf, 20, inf, 5}};
    const auto a = cb::linear_assignment(cost);
    ASSERT_TRUE(a);
    EXPECT_EQ(a->row_to_col, (std::array<std::int32_t, 2>{0, 2}));
    EXPECT_EQ(a->total, 6.0);
}

TEST(CombinatoricsAssignment, ASolverIsReusableAcrossProblemsOfDifferentShapes)
{
    cb::assignment_solver<double, 4, 4> solver;
    std::vector<std::int32_t> out(4);
    const std::vector<double> square{4, 1, 3, 2, 0, 5, 3, 2, 2};
    const std::vector<double> tall{9, 6, 1, 2, 8, 9, 7, 0};
    for (int round = 0; round < 3; ++round) {
        const auto a = solver.solve(square, 3, 3, out);
        ASSERT_TRUE(a);
        EXPECT_EQ(*a, 5.0);
        const auto b = solver.solve(tall, 4, 2, out);
        ASSERT_TRUE(b);
        EXPECT_EQ(*b, 1.0);
        EXPECT_EQ(out, (std::vector<std::int32_t>{-1, 0, -1, 1}));
    }
}

TEST(CombinatoricsAssignment, FloatCostsWork)
{
    const la::fixed_matrix<float, 2, 2> cost{{1.5f, 0.5f, 0.25f, 3.0f}};
    const auto a = cb::linear_assignment(cost);
    ASSERT_TRUE(a);
    EXPECT_EQ(a->row_to_col, (std::array<std::int32_t, 2>{1, 0}));
    EXPECT_EQ(a->total, 0.75f);
}

// Compile time: the textbook example, a tall matrix and an error, all in constant expressions
constexpr bool compile_time()
{
    const auto a = cb::linear_assignment(la::fixed_matrix<double, 3, 3>{{4, 1, 3, 2, 0, 5, 3, 2, 2}});
    const auto t = cb::linear_assignment(la::fixed_matrix<double, 3, 2>{{5, 1, 0, 9, 4, 4}});
    const auto e = cb::linear_assignment(la::fixed_matrix<double, 2, 2>{{inf, inf, 1, 1}});
    return a && a->total == 5 && a->row_to_col[0] == 1 && a->row_to_col[1] == 0 && a->row_to_col[2] == 2 && t && t->total == 1
           && t->row_to_col[0] == 1 && t->row_to_col[1] == 0 && t->row_to_col[2] == -1 && !e && e.unwrap_err_unchecked() == cb::assignment_error::infeasible;
}
static_assert(compile_time());
static_assert(noexcept(cb::linear_assignment(std::declval<const la::fixed_matrix<double, 3, 3>&>())));
static_assert(noexcept(std::declval<cb::assignment_solver<double, 4, 4>&>().solve(std::span<const double>{}, 0, 0, std::span<std::int32_t>{})));

} // namespace
