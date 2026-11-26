//
//  snell_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/stochastic/snell.hpp>: Hull's two-step American put at compile time,
//  the same at run time, the stopping region, the compensator, and a deep tree against the
//  closed-form limit's neighbourhood; the recombining-lattice form against the dense form, binomial and trinomial,
//  exactly and at compile time.
//

#include <snicholls/core/stochastic/snell.hpp>

#include <snicholls/core/math/constexpr_math.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <vector>

namespace {

namespace st = snicholls::stochastic;
namespace m = snicholls::math;

// Hull, Options, Futures and Other Derivatives: S0 50, K 52, u 1.2, d 0.8, r 5% a year, two one-year steps.
// The state is the number of up moves so far; it rises with probability p and is otherwise unchanged.
constexpr double s0 = 50, strike = 52, up = 1.2, down = 0.8;
constexpr double p_up = (m::exp(0.05) - down) / (up - down);
constexpr std::array<std::array<double, 3>, 3> tree{{{1 - p_up, p_up, 0}, {0, 1 - p_up, p_up}, {0, 0, 1}}};
constexpr auto put = [](std::size_t t, std::size_t j) {
    const double s = s0 * m::pow(up, static_cast<double>(j)) * m::pow(down, static_cast<double>(t) - static_cast<double>(j));
    return j > t ? 0.0 : (strike > s ? strike - s : 0.0);
};
constexpr auto american = st::snell_envelope<3, 2>(tree, put, m::exp(-0.05));

static_assert(m::near(american.value[0][0], 5.0894, 1e-4));   // Hull's American put, computed by the compiler
static_assert(american.stops(1, 0) && !american.stops(1, 1) && !american.stops(0, 0));   // exercise early at the down node
static_assert(american.compensator_increment(1, 0) > 0 && american.compensator_increment(1, 1) == 0);

TEST(Snell, TheRunTimeFormAgreesAndStopsAlongAPath)
{
    const st::markov_chain chain(3, {1 - p_up, p_up, 0, 0, 1 - p_up, p_up, 0, 0, 1});
    const auto u = st::snell_envelope(chain, 2, put, m::exp(-0.05));
    EXPECT_NEAR(u.at(0, 0), american.value[0][0], 1e-14);
    EXPECT_EQ(u.stopping_time(std::vector<std::size_t>{0, 0, 0}), 1u);   // down first: exercise at once
    EXPECT_EQ(u.stopping_time(std::vector<std::size_t>{0, 1, 2}), 2u);   // up, up: hold to the end
    EXPECT_EQ(american.stopping_time(std::array<std::size_t, 3>{0, 0, 0}), 1u);
}

TEST(Snell, TheEnvelopeIsASupermartingaleAboveThePayoff)
{
    const st::markov_chain chain(3, {1 - p_up, p_up, 0, 0, 1 - p_up, p_up, 0, 0, 1});
    const auto u = st::snell_envelope(chain, 2, put, m::exp(-0.05));
    for (std::size_t t = 0; t < 2; ++t)
        for (std::size_t s = 0; s <= t; ++s) {
            EXPECT_GE(u.at(t, s), put(t, s));
            const std::vector<double> next{u.at(t + 1, 0), u.at(t + 1, 1), u.at(t + 1, 2)};
            EXPECT_GE(u.at(t, s), m::exp(-0.05) * chain.expect(s, next) - 1e-12);
        }
}

TEST(Snell, ADeepTreeConvergesOnTheAmericanPremium)
{
    // A 200-step Cox-Ross-Rubinstein tree, S 100, K 100, r 5%, sigma 20%, one year: about 6.09 (European 5.57)
    constexpr std::size_t n = 200;
    const double dt = 1.0 / n, u = m::exp(0.2 * m::sqrt(dt)), d = 1 / u, p = (m::exp(0.05 * dt) - d) / (u - d);
    std::vector<double> matrix((n + 1) * (n + 1), 0.0);
    for (std::size_t j = 0; j < n; ++j) {
        matrix[j * (n + 1) + j] = 1 - p;
        matrix[j * (n + 1) + j + 1] = p;
    }
    matrix[n * (n + 1) + n] = 1;
    const st::markov_chain chain(n + 1, matrix);
    const auto value = st::snell_envelope(chain, n, [&](std::size_t t, std::size_t j) {
        const double s = 100 * std::pow(u, static_cast<double>(j)) * std::pow(d, static_cast<double>(t) - static_cast<double>(j));
        return j > t ? 0.0 : std::max(100 - s, 0.0);
    }, m::exp(-0.05 * dt));
    EXPECT_NEAR(value.at(0, 0), 6.09, 0.01);
}

// ---- The recombining-lattice form against the dense form on the same lattice -----------------------------------
// A lattice with stencil {p_0 .. p_{B-1}} as a dense matrix: node j moves to j + k. Rows past the last reachable
// node are never reached; they stay on the diagonal so the matrix is stochastic.
template <std::size_t States, std::size_t Branches>
constexpr auto as_matrix(const std::array<double, Branches>& stencil)
{
    std::array<std::array<double, States>, States> p{};
    for (std::size_t j = 0; j < States; ++j)
        for (std::size_t k = 0; k < Branches; ++k)
            p[j][j + k < States ? j + k : j] += stencil[k];
    return p;
}
// The lattice form agrees with the dense one bit for bit: value, first steps, and the strict stopping region's ends.
template <std::size_t Steps, std::size_t Branches, typename Payoff>
constexpr bool agrees(const std::array<double, Branches>& stencil, Payoff g, double discount)
{
    constexpr std::size_t states = (Branches - 1) * Steps + 1;
    const auto dense = st::snell_envelope<states, Steps>(as_matrix<states>(stencil), g, discount);
    const auto lattice = st::lattice_envelope<Steps>(stencil, g, discount);
    bool ok = lattice.value == dense.value[0][0];
    for (std::size_t t = 0; t < lattice.head_steps; ++t)
        for (std::size_t j = 0; j <= (Branches - 1) * t; ++j)
            ok &= lattice.head[t][j] == dense.value[t][j];
    for (std::size_t t = 0; t < Steps; ++t) {
        std::size_t first = lattice.no_node, last = lattice.no_node;
        for (std::size_t j = 0; j <= (Branches - 1) * t; ++j)
            if (dense.payoff[t][j] > dense.continuation[t][j]) {
                first = first == lattice.no_node ? j : first;
                last = j;
            }
        ok &= lattice.first_exercise[t] == first && lattice.last_exercise[t] == last;
    }
    return ok;
}

// Hull's two-step tree, as a lattice: 5.0894, and exercise at the down node only
constexpr auto hull_lattice = st::lattice_envelope<2>(std::array{1 - p_up, p_up}, put, m::exp(-0.05));
static_assert(hull_lattice.value == american.value[0][0]);
static_assert(hull_lattice.first_exercise[1] == 0 && hull_lattice.last_exercise[1] == 0 && !hull_lattice.exercises(0));
static_assert(agrees<2>(std::array{1 - p_up, p_up}, put, m::exp(-0.05)));

// A twelve-step CRR put, American and European, and a trinomial one: the lattice and the dense form agree exactly
constexpr double crr_dt = 1.0 / 12, crr_up = m::exp(0.3 * m::sqrt(crr_dt)), crr_p = (m::exp(0.04 * crr_dt) - 1 / crr_up) / (crr_up - 1 / crr_up);
constexpr auto crr_put = [](std::size_t t, std::size_t j) {
    const double s = 100 * m::pow(crr_up, 2 * static_cast<double>(j) - static_cast<double>(t));
    return s < 105 ? 105 - s : 0.0;
};
constexpr auto crr_european = [](std::size_t t, std::size_t j) { return t == 12 ? crr_put(t, j) : std::numeric_limits<double>::lowest(); };
static_assert(agrees<12>(std::array{1 - crr_p, crr_p}, crr_put, m::exp(-0.04 * crr_dt)));
static_assert(agrees<12>(std::array{1 - crr_p, crr_p}, crr_european, m::exp(-0.04 * crr_dt)));
constexpr std::array trinomial{0.25, 0.5, 0.25};
constexpr auto tri_put = [](std::size_t t, std::size_t j) {
    const double s = 100 * m::pow(1.05, static_cast<double>(j) - static_cast<double>(t));
    return s < 100 ? 100 - s : 0.0;
};
static_assert(agrees<10>(trinomial, tri_put, 0.998));

// Layout: O(Steps) state, trivially copyable, no stops table
static_assert(std::is_trivially_copyable_v<st::lattice_snell<64, 2>>);
static_assert(sizeof(st::lattice_snell<64, 2>) == sizeof(double) * (1 + 3 * 3) + 2 * 64 * sizeof(std::size_t));

TEST(SnellLattice, AgreesWithTheDenseFormAtRunTime)
{
    // The same computation at run time, where the arithmetic is the processor's rather than the compiler's
    constexpr std::size_t n = 40;
    const double dt = 1.0 / n, u = std::exp(0.25 * std::sqrt(dt)), p = (std::exp(0.03 * dt) - 1 / u) / (u - 1 / u);
    const auto g = [&](std::size_t t, std::size_t j) {
        return std::max(0.0, 90 - 100 * std::pow(u, 2 * static_cast<double>(j) - static_cast<double>(t)));
    };
    EXPECT_TRUE((agrees<n>(std::array{1 - p, p}, g, std::exp(-0.03 * dt))));
    EXPECT_TRUE((agrees<n>(trinomial, tri_put, 0.999)));
}

TEST(SnellLattice, TheBoundaryOfAnAmericanPutIsTheLowNodes)
{
    constexpr auto tree = st::lattice_envelope<12>(std::array{1 - crr_p, crr_p}, crr_put, m::exp(-0.04 * crr_dt));
    for (std::size_t t = 0; t < 12; ++t)
        if (tree.exercises(t)) {
            EXPECT_EQ(tree.first_exercise[t], 0u) << "a put is exercised from the bottom of the tree up, at step " << t;
        }
    EXPECT_TRUE(tree.exercises(11));
    EXPECT_FALSE(tree.exercises(0));
    EXPECT_GE(tree.value, crr_put(0, 0));
}

TEST(SnellLattice, AEuropeanClaimNeverExercisesAndAOneStepTreeIsOneExpectation)
{
    constexpr auto european = st::lattice_envelope<12>(std::array{1 - crr_p, crr_p}, crr_european, m::exp(-0.04 * crr_dt));
    for (std::size_t t = 0; t < 12; ++t)
        EXPECT_FALSE(european.exercises(t));
    constexpr auto one = st::lattice_envelope<1>(std::array{0.5f, 0.5f}, [](std::size_t, std::size_t j) { return j == 1 ? 2.0f : -1.0f; }, 0.5f);
    static_assert(one.value == 0.25f && one.head[1][0] == -1.0f && one.head[1][1] == 2.0f && one.first_exercise[0] == one.no_node);
    constexpr auto none = st::lattice_envelope<0>(trinomial, [](std::size_t, std::size_t) { return 7.0; });
    static_assert(none.value == 7.0 && none.head[0][0] == 7.0);
    EXPECT_EQ(one.value, 0.25f);
}

} // namespace
