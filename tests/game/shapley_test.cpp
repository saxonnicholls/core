//
//  shapley_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/game/shapley.hpp>: the tables of Moehle, Boyd and Ang (2021), every
//  method's defining case, compile-time evaluation, several metrics at once, and the samplers.
//

#include <snicholls/core/game/shapley.hpp>

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <random>
#include <sstream>
#include <vector>

namespace {

namespace game = snicholls::game;
using game::coalition;

// Three metrics attributed in one pass: any game_value works, not just double
struct metrics {
    double risk = 0, ret = 0, turnover = 0;
    friend constexpr metrics operator+(metrics a, metrics b) noexcept { return {a.risk + b.risk, a.ret + b.ret, a.turnover + b.turnover}; }
    friend constexpr metrics operator-(metrics a, metrics b) noexcept { return {a.risk - b.risk, a.ret - b.ret, a.turnover - b.turnover}; }
    friend constexpr metrics operator*(metrics a, double w) noexcept { return {a.risk * w, a.ret * w, a.turnover * w}; }
};
static_assert(game::game_value<metrics>);

// Table 1 of the paper; bit 0 is feature x1, bit 1 is x2
constexpr std::array<metrics, 4> table_1{metrics{0.1, 5, 2}, metrics{2, 12, 30}, metrics{1.7, 8, 38}, metrics{2.3, 11, 43}};

// Section 4.1's example (1): either feature alone gives everything
constexpr auto either = [](coalition s) { return s == 0 ? 0.0 : 1.0; };

// Table 3: Bacon's return attribution, country allocation then stock selection
constexpr std::array<double, 4> table_3{6.4, 5.2, 9.4, 8.3};

// A unanimity game: worth 1 exactly when every player of `t` is in
constexpr auto unanimity(coalition t)
{
    return [t](coalition s) { return (s & t) == t ? 1.0 : 0.0; };
}

// The paper's quadratic, x'Px on the features switched on
template <std::size_t N>
struct quadratic {
    std::array<std::array<double, N>, N> p{};
    constexpr double operator()(coalition s) const noexcept
    {
        double sum = 0;
        for (std::size_t i = 0; i < N; ++i)
            for (std::size_t j = 0; j < N; ++j)
                if (game::contains(s, i) && game::contains(s, j))
                    sum += p[i][j];
        return sum;
    }
};

template <std::size_t N>
quadratic<N> random_psd(unsigned seed)
{
    std::mt19937_64 g(seed);
    std::normal_distribution<double> normal;
    std::array<std::array<double, N>, N> z{};
    for (auto& row : z)
        for (auto& x : row)
            x = normal(g);
    quadratic<N> q;
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j)
            for (std::size_t k = 0; k < N; ++k)
                q.p[i][j] += z[k][i] * z[k][j];
    return q;
}

// ---- compile time ----------------------------------------------------------------------------------

// Exact in binary fractions, so equality is exact: v = 1 for x1, 2 for x2, 4 together
constexpr std::array<double, 4> dyadic{0.0, 1.0, 2.0, 4.0};
constexpr auto at_compile_time = game::shapley<2>(game::from_table(dyadic));
static_assert(at_compile_time.values[0] == 1.5 && at_compile_time.values[1] == 2.5);
static_assert(at_compile_time.baseline == 0.0 && at_compile_time.total == 4.0 && at_compile_time.unattributed() == 0.0);
static_assert(game::shapley<2>([](coalition s) { return static_cast<double>(s); }).values[1] == 2.0 + 0.5 * -1.0 + 0.5 * 1.0);
static_assert(game::shapley<3>(unanimity(0b011)).values == std::array{0.5, 0.5, 0.0});
static_assert(game::shapley_by_permutations<3>(unanimity(0b111)).values[2] * 3 == 1.0);
static_assert(game::banzhaf<3>(unanimity(0b111)).values[0] == 0.25);
static_assert(game::sequential<2>(either).values == std::array{1.0, 0.0});
static_assert(game::permuted_sequential<2>(either, std::array<std::size_t, 2>{1, 0}).values == std::array{0.0, 1.0});

// Layout: an attribution is its values and nothing else
static_assert(sizeof(game::fixed_attribution<double, 3>) == 5 * sizeof(double));
static_assert(std::is_trivially_copyable_v<game::fixed_attribution<double, 3>>);

// The edges: no player, one player
static_assert(game::shapley<0>([](coalition) { return 7.0; }).unattributed() == 0.0);
static_assert(game::shapley<1>([](coalition s) { return s == 0 ? 2.0 : 5.0; }).values[0] == 3.0);

// ---- the paper's examples --------------------------------------------------------------------------

TEST(ShapleyPaper, Table1AttributesThreeMetricsAtOnce)
{
    const auto a = game::shapley<2>(game::from_table(table_1));
    EXPECT_NEAR(a.baseline.risk, 0.1, 1e-12);
    EXPECT_NEAR(a.values[0].risk, 1.25, 1e-12);
    EXPECT_NEAR(a.values[1].risk, 0.95, 1e-12);
    EXPECT_NEAR(a.values[0].ret, 5, 1e-12);
    EXPECT_NEAR(a.values[1].ret, 1, 1e-12);
    EXPECT_NEAR(a.values[0].turnover, 16.5, 1e-12);
    EXPECT_NEAR(a.values[1].turnover, 24.5, 1e-12);
    EXPECT_NEAR(a.total.turnover, 43, 1e-12);
}

TEST(ShapleyPaper, Table4ComparesTheThreeMethodsOnBaconsExample)
{
    const auto v = game::from_table(table_3);
    const auto shap = game::shapley<2>(v);
    EXPECT_NEAR(shap.values[0], -1.15, 1e-12);
    EXPECT_NEAR(shap.values[1], 3.05, 1e-12);
    EXPECT_NEAR(shap.unattributed(), 0, 1e-12);
    const auto one = game::one_at_a_time<2>(v);   // Brinson-Hood-Beebower
    EXPECT_NEAR(one.values[0], -1.2, 1e-12);
    EXPECT_NEAR(one.values[1], 3.0, 1e-12);
    EXPECT_NEAR(one.unattributed(), 0.1, 1e-12);
    const auto seq = game::sequential<2>(v);   // modified BHB
    EXPECT_NEAR(seq.values[0], -1.2, 1e-12);
    EXPECT_NEAR(seq.values[1], 3.1, 1e-12);
    EXPECT_NEAR(seq.unattributed(), 0, 1e-12);
}

TEST(ShapleyPaper, Example1ShowsWhereEachMethodGoesWrong)
{
    EXPECT_EQ(game::one_at_a_time<2>(either).values, (std::array{1.0, 1.0}));   // over-attributes
    EXPECT_EQ(game::one_at_a_time<2>(either).unattributed(), -1.0);
    EXPECT_EQ(game::leave_one_out<2>(either).values, (std::array{0.0, 0.0}));   // under-attributes
    EXPECT_EQ(game::leave_one_out<2>(either).unattributed(), 1.0);
    EXPECT_EQ(game::shapley<2>(either).values, (std::array{0.5, 0.5}));
}

TEST(ShapleyPaper, TheQuadraticIsAttributedP1NotThe2P1ThePaperPrints)
{
    const auto q = random_psd<6>(7);
    const auto a = game::shapley<6>(q);
    double sum = 0;
    for (std::size_t i = 0; i < 6; ++i) {
        double row = 0;
        for (std::size_t j = 0; j < 6; ++j)
            row += q.p[i][j];
        EXPECT_NEAR(a.values[i], row, 1e-9);
        sum += row;
    }
    EXPECT_NEAR(sum, a.total, 1e-9);   // 2P1 would sum to twice the total
}

// ---- agreement between the methods -----------------------------------------------------------------

TEST(Shapley, TheDefinitionAndTheFormulaAgree)
{
    std::mt19937 g(3);
    std::uniform_real_distribution<double> u(-5, 5);
    std::array<double, 32> table{};
    for (double& x : table)
        x = u(g);
    const auto fast = game::shapley<5>(game::from_table(table));
    const auto slow = game::shapley_by_permutations<5>(game::from_table(table));
    for (std::size_t i = 0; i < 5; ++i)
        EXPECT_NEAR(fast.values[i], slow.values[i], 1e-12);
    EXPECT_NEAR(fast.unattributed(), 0, 1e-12);
}

TEST(Shapley, RunTimeAndCompileTimeCountsAgree)
{
    const auto q = random_psd<5>(11);
    const auto fixed = game::shapley<5>(q);
    const auto dynamic = game::shapley(5, q);
    ASSERT_EQ(dynamic.values.size(), 5u);
    for (std::size_t i = 0; i < 5; ++i)
        EXPECT_DOUBLE_EQ(fixed.values[i], dynamic.values[i]);
    const std::vector<std::size_t> order{4, 2, 0, 1, 3};
    const auto seq = game::permuted_sequential(q, std::span<const std::size_t>(order));
    EXPECT_NEAR(seq.unattributed(), 0, 1e-12);
}

TEST(Shapley, EachGameIsEvaluatedOncePerCoalition)
{
    int calls = 0;
    const auto counted = [&calls](coalition s) {
        ++calls;
        return static_cast<double>(game::coalition_size(s));
    };
    const auto a = game::shapley<4>(counted);
    EXPECT_EQ(calls, 16);
    EXPECT_EQ(a.values, (std::array{1.0, 1.0, 1.0, 1.0}));
    calls = 0;
    static_cast<void>(game::permuted_sequential<4>(counted, std::array<std::size_t, 4>{3, 1, 0, 2}));
    EXPECT_EQ(calls, 5);
}

TEST(ShapleyBanzhaf, SharesAUnanimityDividendByHalvesNotEqually)
{
    const auto b = game::banzhaf<3>(unanimity(0b111));
    const auto s = game::shapley<3>(unanimity(0b111));
    EXPECT_DOUBLE_EQ(b.values[0], 0.25);
    EXPECT_NEAR(s.values[0], 1.0 / 3.0, 1e-15);
    EXPECT_DOUBLE_EQ(b.unattributed(), 0.25);   // Banzhaf is not full
}

TEST(ShapleyInteractions, AreTheQuadraticsCrossTermsAndZeroForAnAdditiveGame)
{
    const auto q = random_psd<4>(5);
    const auto pairs = game::shapley_interactions<4>(q);
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            EXPECT_NEAR(pairs[i][j], i == j ? 0.0 : 2 * q.p[i][j], 1e-12);
    const auto additive = game::shapley_interactions<3>([](coalition s) { return static_cast<double>(s); });
    for (const auto& row : additive)
        for (const double x : row)
            EXPECT_DOUBLE_EQ(x, 0.0);
}

TEST(ShapleyAttribution, PrintsItself)
{
    std::ostringstream os;
    game::shapley<2>(game::from_table(dyadic)).print(os);
    EXPECT_EQ(os.str(), "baseline 0, [1.5, 2.5], total 4");
}

// ---- sampling --------------------------------------------------------------------------------------

TEST(ShapleySampled, OrdersConvergeAreFullAndAreReproducible)
{
    const auto q = random_psd<10>(1);
    const auto exact = game::shapley<10>(q);
    std::mt19937_64 g(42);
    const auto est = game::shapley_sampled<10>(q, g, {.permutations = 2000, .antithetic = true});
    EXPECT_EQ(est.samples, 2000u);
    EXPECT_EQ(est.evaluations, 2u + 2000u * 2u * 9u);
    EXPECT_NEAR(est.unattributed(), 0, 1e-9);   // every sample is a full attribution
    for (std::size_t i = 0; i < 10; ++i) {
        EXPECT_GT(est.standard_errors[i], 0);
        EXPECT_NEAR(est.values[i], exact.values[i], 5 * est.standard_errors[i] + 1e-12);
    }
    std::mt19937_64 again(42);
    const auto same = game::shapley_sampled(10, q, again, {.permutations = 2000, .antithetic = true});
    for (std::size_t i = 0; i < 10; ++i)
        EXPECT_EQ(same.values[i], est.values[i]);
}

TEST(ShapleySampled, AnAdditiveGameIsExactFromOneOrder)
{
    std::mt19937 g(9);   // a 32-bit generator works too
    const auto est = game::shapley_sampled<5>([](coalition s) { return static_cast<double>(s); }, g, {.permutations = 1, .antithetic = false});
    for (std::size_t i = 0; i < 5; ++i)
        EXPECT_DOUBLE_EQ(est.values[i], static_cast<double>(game::singleton(i)));
    EXPECT_TRUE(std::isnan(est.standard_errors[0]));
}

TEST(ShapleySampled, LiftsConvergeAndScaleToFull)
{
    const auto q = random_psd<8>(2);
    const auto exact = game::shapley<8>(q);
    std::mt19937_64 g(5);
    const auto unscaled = game::shapley_sampled_lifts<8>(q, g, {.samples_per_player = 4000, .scale_to_total = false});
    for (std::size_t i = 0; i < 8; ++i)
        EXPECT_NEAR(unscaled.values[i], exact.values[i], 5 * unscaled.standard_errors[i] + 1e-12);
    const auto scaled = game::shapley_sampled_lifts(8, q, g, {.samples_per_player = 4000, .scale_to_total = true});
    EXPECT_NEAR(scaled.unattributed(), 0, 1e-9);
    EXPECT_EQ(scaled.evaluations, 2u + 8u * 4000u * 2u);
}

TEST(ShapleySampled, AMemoizedGameIsEvaluatedOncePerDistinctCoalition)
{
    int calls = 0;
    const auto backtest = [&calls](coalition s) {
        ++calls;
        return static_cast<double>(game::coalition_size(s) * game::coalition_size(s));
    };
    const game::memoized cached(backtest);
    std::mt19937_64 g(1);
    const auto est = game::shapley_sampled<4>(cached, g, {.permutations = 500});
    EXPECT_LE(calls, 16);
    EXPECT_EQ(static_cast<std::size_t>(calls), cached.evaluations());
    for (const double a : est.values)
        EXPECT_NEAR(a, 4.0, 1e-9);   // symmetric: an equal share of 16
}

// Fewer than two players: no pair, an empty or one-by-one matrix of zeros, compiled under -Werror.
constexpr auto count_game = [](snicholls::game::coalition c) { return static_cast<double>(std::popcount(c)); };
static_assert(snicholls::game::shapley_interactions<0>(count_game).empty());
static_assert(snicholls::game::shapley_interactions<1>(count_game)[0][0] == 0.0);

TEST(ShapleyInteractions, NoPairForFewerThanTwoPlayers)
{
    EXPECT_EQ(snicholls::game::shapley_interactions<1>(count_game)[0][0], 0.0);
}

} // namespace
