//
//  discrete_test.cpp
//  snicholls-core
//
//  Tests for <snicholls/core/tracking/discrete.hpp>: normalise and normalise_columns (a zero column left zero), the
//  hidden Markov model's prediction, predicted observation, update with one-hot and soft observations and the
//  marginal likelihood, all against hand calculation; zero_probability for an impossible observation, a zero
//  transition and a zero vector; the error's layout, and all of it in a constant expression.
//

#include <snicholls/core/tracking/discrete.hpp>

#include <snicholls/core/linalg/fixed.hpp>
#include <snicholls/core/stats/random.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace {

namespace la = snicholls::linalg;
namespace st = snicholls::stats;
namespace tr = snicholls::tracking;

using v2 = la::fixed_vector<double, 2>;
using v3 = la::fixed_vector<double, 3>;
using m22 = la::fixed_matrix<double, 2, 2>;
using m33 = la::fixed_matrix<double, 3, 3>;
using m32 = la::fixed_matrix<double, 3, 2>;

static_assert(std::is_same_v<std::underlying_type_t<tr::hmm_error>, std::uint8_t>);
static_assert(static_cast<int>(tr::hmm_error::zero_probability) == 1);

// Observations of a swimmer: rows (seen above, seen at, not seen), columns hidden (surfaced, submerged)
constexpr m32 emission() { return m32{{0.7, 0.1, 0.2, 0.3, 0.1, 0.6}}; }

template <std::size_t K>
void expect_vector(const la::fixed_vector<double, K>& got, const std::array<double, K>& want, double tol = 1e-15)
{
    for (std::size_t i = 0; i < K; ++i)
        EXPECT_NEAR(got[i], want[i], tol) << i;
}

TEST(TrackingNormalise, ScalesToSumOneAndRefusesNoMass)
{
    const auto n = tr::normalise(v3{{1, 3, 4}});
    ASSERT_TRUE(n);
    expect_vector(*n, {0.125, 0.375, 0.5});
    const auto zero = tr::normalise(v3{});
    ASSERT_FALSE(zero);
    EXPECT_EQ(zero.unwrap_err_unchecked(), tr::hmm_error::zero_probability);
    EXPECT_FALSE(tr::normalise(v2{{-1, 0.5}}));
}

TEST(TrackingNormaliseColumns, ScalesEachColumnAndLeavesAZeroColumnZero)
{
    const auto m = tr::normalise_columns(la::fixed_matrix<double, 2, 3>{{1, 0, 2, 3, 0, 2}});
    EXPECT_EQ(m, (la::fixed_matrix<double, 2, 3>{{0.25, 0, 0.5, 0.75, 0, 0.5}}));
}

TEST(TrackingHmmPredict, MatchesTheHandCalculation)
{
    // A p with A(i, j) = P(j → i): (0.9·0.5 + 0.2·0.5, 0.1·0.5 + 0.8·0.5) = (0.55, 0.45)
    const auto p = tr::hmm_predict(v2{{0.5, 0.5}}, m22{{0.9, 0.2, 0.1, 0.8}});
    ASSERT_TRUE(p);
    expect_vector(*p, {0.55, 0.45});
    // Three states: (0.16 + 0.05, 0.04 + 0.35 + 0.09, 0.10 + 0.21)
    const auto q = tr::hmm_predict(v3{{0.2, 0.5, 0.3}}, m33{{0.8, 0.1, 0, 0.2, 0.7, 0.3, 0, 0.2, 0.7}});
    ASSERT_TRUE(q);
    expect_vector(*q, {0.21, 0.48, 0.31});
    // A matrix that is not column-stochastic is normalised away
    const auto r = tr::hmm_predict(v2{{0.5, 0.5}}, m22{{1.8, 0.4, 0.2, 1.6}});
    ASSERT_TRUE(r);
    expect_vector(*r, {0.55, 0.45});
}

TEST(TrackingHmmPredict, ReportsAZeroTransition)
{
    const auto p = tr::hmm_predict(v2{{0.5, 0.5}}, m22{});
    ASSERT_FALSE(p);
    EXPECT_EQ(p.unwrap_err_unchecked(), tr::hmm_error::zero_probability);
}

TEST(TrackingHmmPredictMeasurement, MatchesTheHandCalculation)
{
    // E p, p = (0.4, 0.6): (0.28 + 0.06, 0.08 + 0.18, 0.04 + 0.36)
    const auto z = tr::hmm_predict_measurement(v2{{0.4, 0.6}}, emission());
    ASSERT_TRUE(z);
    expect_vector(*z, {0.34, 0.26, 0.40});
    const auto none = tr::hmm_predict_measurement(v2{}, emission());
    ASSERT_FALSE(none);
    EXPECT_EQ(none.unwrap_err_unchecked(), tr::hmm_error::zero_probability);
}

TEST(TrackingHmmUpdate, OneHotAndSoftObservationsMatchTheHandCalculation)
{
    const v2 p{{0.4, 0.6}};
    // Not seen: (0.1·0.4, 0.6·0.6) = (0.04, 0.36), normalised (0.1, 0.9)
    const auto hard = tr::hmm_update(p, v3{{0, 0, 1}}, emission());
    ASSERT_TRUE(hard);
    expect_vector(*hard, {0.1, 0.9});
    // Half "above", half "not seen": Eᵀ z = (0.4, 0.35), times p = (0.16, 0.21)
    const auto soft = tr::hmm_update(p, v3{{0.5, 0, 0.5}}, emission());
    ASSERT_TRUE(soft);
    expect_vector(*soft, {0.16 / 0.37, 0.21 / 0.37});
}

TEST(TrackingHmmUpdate, ReportsAnObservationImpossibleUnderThePrior)
{
    // Only the first hidden state is possible, and it never shows the second observation
    const auto u = tr::hmm_update(v2{{1, 0}}, v2{{0, 1}}, m22{{1, 0, 0, 1}});
    ASSERT_FALSE(u);
    EXPECT_EQ(u.unwrap_err_unchecked(), tr::hmm_error::zero_probability);
}

TEST(TrackingHmmLikelihood, IsTheObservationsMarginalProbability)
{
    const v2 p{{0.4, 0.6}};
    EXPECT_NEAR(tr::hmm_likelihood(p, v3{{0, 0, 1}}, emission()), 0.40, 1e-15);
    EXPECT_NEAR(tr::hmm_likelihood(p, v3{{0.5, 0, 0.5}}, emission()), 0.37, 1e-15);
    EXPECT_EQ(tr::hmm_likelihood(v2{{1, 0}}, v2{{0, 1}}, m22{{1, 0, 0, 1}}), 0.0);
    // Summed over every one-hot observation, the likelihoods are 1 for a column-stochastic E
    double s = 0;
    for (std::size_t l = 0; l < 3; ++l) {
        v3 z{};
        z[l] = 1;
        s += tr::hmm_likelihood(p, z, emission());
    }
    EXPECT_NEAR(s, 1.0, 1e-15);
}

TEST(TrackingHmmUpdate, IsBayesRuleOnRandomModels)
{
    // The update's posterior times the likelihood is the joint P(observe l, hidden k) = E(l, k) p(k)
    st::wyrand g(17);
    for (int trial = 0; trial < 200; ++trial) {
        m33 a{}, e{};
        v3 p{};
        for (std::size_t i = 0; i < 9; ++i) {
            a.elements[i] = st::uniform_real(g);
            e.elements[i] = st::uniform_real(g);
        }
        for (std::size_t i = 0; i < 3; ++i)
            p[i] = st::uniform_real(g) + 0.01;
        a = tr::normalise_columns(a);
        e = tr::normalise_columns(e);
        const auto pred = tr::hmm_predict(*tr::normalise(p), a);
        ASSERT_TRUE(pred);
        const std::size_t l = st::uniform_index(g, 3);
        v3 z{};
        z[l] = 1;
        const auto post = tr::hmm_update(*pred, z, e);
        ASSERT_TRUE(post);
        const double like = tr::hmm_likelihood(*pred, z, e);
        double sum = 0;
        for (std::size_t k = 0; k < 3; ++k) {
            EXPECT_NEAR((*post)[k] * like, e(l, k) * (*pred)[k], 1e-14);
            EXPECT_GE((*post)[k], 0.0);
            sum += (*post)[k];
        }
        EXPECT_NEAR(sum, 1.0, 1e-14);
    }
}

constexpr bool compile_time()
{
    const auto p = tr::hmm_predict(v2{{0.5, 0.5}}, m22{{0.9, 0.2, 0.1, 0.8}});
    const auto z = tr::hmm_predict_measurement(v2{{0.4, 0.6}}, emission());
    const auto u = tr::hmm_update(v2{{0.4, 0.6}}, v3{{0, 0, 1}}, emission());
    const double l = tr::hmm_likelihood(v2{{0.4, 0.6}}, v3{{0, 0, 1}}, emission());
    const auto c = tr::normalise_columns(m22{{1, 0, 3, 0}});
    const auto bad = tr::normalise(v2{});
    return p.is_ok() && (*p)[0] > 0.5499 && (*p)[0] < 0.5501 && z.is_ok() && (*z)[2] > 0.3999 && (*z)[2] < 0.4001 && u.is_ok() && (*u)[1] > 0.8999 && (*u)[1] < 0.9001
           && l > 0.3999 && l < 0.4001 && c(1, 0) == 0.75 && c(1, 1) == 0 && bad.is_err();
}
static_assert(compile_time());

static_assert(noexcept(tr::normalise(std::declval<const v2&>())));
static_assert(noexcept(tr::normalise_columns(std::declval<const m32&>())));
static_assert(noexcept(tr::hmm_predict(std::declval<const v2&>(), std::declval<const m22&>())));
static_assert(noexcept(tr::hmm_predict_measurement(std::declval<const v2&>(), std::declval<const m32&>())));
static_assert(noexcept(tr::hmm_update(std::declval<const v2&>(), std::declval<const v3&>(), std::declval<const m32&>())));
static_assert(noexcept(tr::hmm_likelihood(std::declval<const v2&>(), std::declval<const v3&>(), std::declval<const m32&>())));

} // namespace
