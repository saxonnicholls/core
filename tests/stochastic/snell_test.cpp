//
//  snell_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/stochastic/snell.hpp>: Hull's two-step American put at compile time,
//  the same at run time, the stopping region, the compensator, and a deep tree against the
//  closed-form limit's neighbourhood.
//

#include <snicholls/core/stochastic/snell.hpp>

#include <snicholls/core/math/constexpr_math.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
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

} // namespace
