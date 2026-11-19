//
//  rl_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/ml/rl.hpp>: group advantages against the sample mean and std by hand, zero for a group of
//  equal rewards and for a group of one, Dr. GRPO's centring only; the k3 estimator against rho - log rho - 1 and its
//  derivative; the clipped token objective at and around both clip boundaries for either sign of the advantage; the
//  batch loss under each aggregation against a direct per-sample computation, its gradients in the log-probabilities
//  and entropies against central differences, the prompt contributing nothing; the metrics; GAE against the recursion
//  by hand, skipping prompt positions; PPO's value loss with and without its clip, against central differences; masked
//  whitening; the rollout view; compile-time use and the noexcept claims.
//

#include <snicholls/core/ml/rl.hpp>

#include <snicholls/core/linalg/reference.hpp>
#include <snicholls/core/stats/random.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>
#include <vector>

namespace {

namespace ml = snicholls::ml;
namespace stats = snicholls::stats;
constexpr auto& backend = snicholls::linalg::reference;

static_assert(std::is_trivially_copyable_v<ml::policy_loss_config> && std::is_trivially_copyable_v<ml::policy_loss_metrics<float>>);
static_assert(std::is_trivially_copyable_v<ml::rollout_view<float>>);
static_assert(noexcept(ml::kl_k3(0.0f, 0.0f)) && noexcept(ml::policy_token_loss(0.0, 0.0, 0.0, 1.0, {})));
static_assert(ml::policy_loss_config{}.clip_low == 0.2 && ml::policy_loss_config{}.kl_beta == 0.0
              && ml::policy_loss_config{}.aggregation == ml::token_aggregation::sequence_mean);

constexpr bool at_compile_time()
{
    const std::array<double, 4> rewards{1.0, 0.0, 2.0, 2.0};
    std::array<double, 4> adv{};
    ml::group_advantages<double>(backend, rewards, 2, adv, ml::advantage_scale::none);
    const auto p = ml::policy_token_loss(-1.0, -1.0, -1.0, 2.0, {.kl_beta = 0.5});
    const ml::rollout_view<double> r{.prompts = 3, .group_size = 4};
    return adv[0] == 0.5 && adv[1] == -0.5 && adv[2] == 0.0 && adv[3] == 0.0 && p.loss == -2.0 && p.d_logp == -2.0 && p.kl == 0.0
           && r.rows() == 12 && r.group_of(7) == 1 && ml::kl_k3(-1.0, -1.0) == 0.0;
}
static_assert(at_compile_time());

TEST(MlGroupAdvantages, AreTheRewardsNormalisedWithinEachGroup)
{
    const std::vector<double> rewards{1, 0, 0, 1, /**/ 0.5, 0.5, 0.5, 0.5, /**/ 3, 1, 2, 0};
    std::vector<double> adv(12);
    ml::group_advantages<double>(backend, rewards, 4, adv);
    for (std::size_t b = 0; b < 12; b += 4) {
        double mean = 0, ss = 0;
        for (std::size_t i = b; i < b + 4; ++i)
            mean += rewards[i] / 4;
        for (std::size_t i = b; i < b + 4; ++i)
            ss += (rewards[i] - mean) * (rewards[i] - mean);
        const double sd = std::sqrt(ss / 3);
        for (std::size_t i = b; i < b + 4; ++i)
            EXPECT_NEAR(adv[i], (rewards[i] - mean) / (sd + 1e-4), 1e-15) << i;
    }
    for (std::size_t i = 4; i < 8; ++i)
        EXPECT_EQ(adv[i], 0.0);   // equal rewards: no signal
    std::vector<double> centred(12);
    ml::group_advantages<double>(backend, rewards, 4, centred, ml::advantage_scale::none);
    EXPECT_EQ(centred[8], 1.5);
    EXPECT_EQ(centred[11], -1.5);
    std::vector<double> single(3);
    ml::group_advantages<double>(backend, std::vector<double>{5, -2, 7}, 1, single);
    EXPECT_EQ(single, (std::vector<double>{0, 0, 0}));
    std::vector<double> per_token(6);
    ml::expand_rows<double>(backend, std::vector<double>{1, -2}, per_token);
    EXPECT_EQ(per_token, (std::vector<double>{1, 1, 1, -2, -2, -2}));
}

TEST(MlKlK3, IsRhoMinusLogRhoMinusOneAndItsDerivative)
{
    for (const double lp : {-3.0, -1.0, -0.25})
        for (const double ref : {-4.0, -1.0, -0.5, -0.1}) {
            const double rho = std::exp(ref) / std::exp(lp);
            EXPECT_NEAR(ml::kl_k3(lp, ref), rho - std::log(rho) - 1, 1e-14);
            EXPECT_GE(ml::kl_k3(lp, ref), 0.0);
            const double h = 1e-6;
            EXPECT_NEAR(ml::kl_k3_derivative(lp, ref), (ml::kl_k3(lp + h, ref) - ml::kl_k3(lp - h, ref)) / (2 * h), 1e-8);
        }
    // Tiny differences keep their digits (expm1)
    EXPECT_NEAR(ml::kl_k3(-1.0, -1.0 + 1e-9), 0.5e-18, 1e-24);
}

TEST(MlPolicyTokenLoss, ClipsAtTheBoundariesWithTheStatedGradient)
{
    const ml::policy_loss_config c{.clip_low = 0.2, .clip_high = 0.28};
    // Positive advantage: the clip binds above 1 + eps_high
    auto inside = ml::policy_token_loss(std::log(1.1), 0.0, 0.0, 2.0, c);
    EXPECT_FALSE(inside.clipped);
    EXPECT_NEAR(inside.loss, -2.2, 1e-15);
    EXPECT_NEAR(inside.d_logp, -2.2, 1e-15);
    auto above = ml::policy_token_loss(std::log(1.5), 0.0, 0.0, 2.0, c);
    EXPECT_TRUE(above.clipped);
    EXPECT_NEAR(above.loss, -2.56, 1e-15);
    EXPECT_EQ(above.d_logp, 0.0);
    auto below = ml::policy_token_loss(std::log(0.5), 0.0, 0.0, 2.0, c);   // below 1 - eps: the unclipped term is the smaller
    EXPECT_FALSE(below.clipped);
    EXPECT_NEAR(below.d_logp, -1.0, 1e-15);
    // Negative advantage: the clip binds below 1 - eps_low
    auto low = ml::policy_token_loss(std::log(0.5), 0.0, 0.0, -1.0, c);
    EXPECT_TRUE(low.clipped);
    EXPECT_NEAR(low.loss, 0.8, 1e-15);
    EXPECT_EQ(low.d_logp, 0.0);
    auto high = ml::policy_token_loss(std::log(1.5), 0.0, 0.0, -1.0, c);
    EXPECT_FALSE(high.clipped);
    EXPECT_NEAR(high.d_logp, 1.5, 1e-15);
    // Exactly on the edge: the unclipped gradient
    const double r = std::exp(0.125);
    const auto edge = ml::policy_token_loss(0.125, 0.0, 0.0, 3.0, {.clip_low = 0.2, .clip_high = r - 1.0});
    EXPECT_FALSE(edge.clipped);
    EXPECT_EQ(edge.d_logp, -r * 3.0);
    // Zero advantage: no gradient from the surrogate, only from the KL
    const auto kl_only = ml::policy_token_loss(-1.0, -1.0, -2.0, 0.0, {.kl_beta = 0.04});
    EXPECT_NEAR(kl_only.loss, 0.04 * ml::kl_k3(-1.0, -2.0), 1e-17);
    EXPECT_NEAR(kl_only.d_logp, 0.04 * ml::kl_k3_derivative(-1.0, -2.0), 1e-17);
}

struct batch
{
    static constexpr std::size_t rows = 3, len = 4;
    std::vector<std::uint8_t> mask{0, 1, 1, 0, /**/ 0, 1, 1, 1, /**/ 0, 0, 1, 0};
    std::vector<double> old{-1, -1.2, -0.7, -2, /**/ -1, -0.3, -1.1, -0.9, /**/ -1, -1, -0.6, -1};
    std::vector<double> ref{-1, -1.0, -0.9, -2, /**/ -1, -0.4, -1.0, -1.2, /**/ -1, -1, -0.5, -1};
    std::vector<double> adv{0, 1.2, 1.2, 0, /**/ 0, -0.8, -0.8, -0.8, /**/ 0, 0, 0.3, 0};
    std::vector<double> ent{9, 2.0, 1.5, 9, /**/ 9, 0.5, 0.7, 1.1, /**/ 9, 9, 2.2, 9};
};

double direct_loss(const batch& b, const std::vector<double>& lp, const std::vector<double>& ent, const ml::policy_loss_config& c)
{
    double total_tokens = 0;
    for (const auto m : b.mask)
        total_tokens += m;
    double loss = 0;
    for (std::size_t r = 0; r < batch::rows; ++r) {
        double sum = 0, n = 0;
        for (std::size_t t = r * batch::len; t < (r + 1) * batch::len; ++t) {
            if (!b.mask[t])
                continue;
            const double ratio = std::exp(lp[t] - b.old[t]);
            const double clipped = std::fmin(std::fmax(ratio, 1 - c.clip_low), 1 + c.clip_high);
            const double rho = std::exp(b.ref[t] - lp[t]);
            sum += -std::fmin(ratio * b.adv[t], clipped * b.adv[t]) + c.kl_beta * (rho - std::log(rho) - 1) - c.entropy_coef * ent[t];
            ++n;
        }
        switch (c.aggregation) {
        case ml::token_aggregation::sequence_mean: loss += n == 0 ? 0 : sum / n / batch::rows; break;
        case ml::token_aggregation::token_mean: loss += sum / total_tokens; break;
        case ml::token_aggregation::constant: loss += sum / double(c.max_tokens == 0 ? batch::len : c.max_tokens) / batch::rows; break;
        }
    }
    return loss;
}

TEST(MlPolicyLossRows, IsTheDirectComputationUnderEachAggregationAndItsGradientIsCentralDifferences)
{
    batch b;
    std::vector<double> lp{-5, -1.0, -0.8, -5, /**/ -5, -0.35, -1.05, -0.5, /**/ -5, -5, -0.62, -5};
    for (const auto agg : {ml::token_aggregation::sequence_mean, ml::token_aggregation::token_mean, ml::token_aggregation::constant})
        for (const std::size_t max_tokens : {std::size_t{0}, std::size_t{16}}) {
            const ml::policy_loss_config c{.clip_low = 0.2, .clip_high = 0.28, .kl_beta = 0.04, .entropy_coef = 0.01, .aggregation = agg,
                                           .max_tokens = max_tokens};
            std::vector<double> dlp(12), dent(12);
            ml::policy_loss_metrics<double> m;
            const double loss = ml::policy_loss_rows<double>(backend, b.mask, lp, b.old, b.ref, b.adv, b.ent, dlp, dent, batch::rows, c, m);
            EXPECT_NEAR(loss, direct_loss(b, lp, b.ent, c), 1e-15);
            EXPECT_EQ(m.tokens, 6u);
            for (std::size_t i = 0; i < 12; ++i) {
                const double h = 1e-6;
                auto x = lp;
                x[i] += h;
                const double up = direct_loss(b, x, b.ent, c);
                x[i] -= 2 * h;
                EXPECT_NEAR(dlp[i], (up - direct_loss(b, x, b.ent, c)) / (2 * h), 1e-8) << i;
                auto e = b.ent;
                e[i] += h;
                const double eu = direct_loss(b, lp, e, c);
                e[i] -= 2 * h;
                EXPECT_NEAR(dent[i], (eu - direct_loss(b, lp, e, c)) / (2 * h), 1e-8) << i;
                if (!b.mask[i]) {
                    EXPECT_EQ(dlp[i], 0.0);
                }
            }
        }
}

TEST(MlPolicyLossRows, TheAggregationsWeighTokensAsTheirPapersSay)
{
    // Two responses of one and three tokens, every token's loss -A = -1 (ratio 1, no KL)
    const std::vector<std::uint8_t> mask{1, 0, 0, 1, 1, 1};
    const std::vector<double> zero(6, 0.0), one(6, 1.0);
    std::vector<double> d(6);
    ml::policy_loss_metrics<double> m;
    const auto run = [&](ml::token_aggregation a, std::size_t max_tokens = 0) {
        return ml::policy_loss_rows<double>(backend, mask, zero, zero, {}, one, {}, d, {}, 2, {.aggregation = a, .max_tokens = max_tokens}, m);
    };
    EXPECT_NEAR(run(ml::token_aggregation::sequence_mean), -1.0, 1e-15);   // (1/2)(1/1 + 3/3)
    EXPECT_EQ(d[0], -0.5);                                         // the short response's token weighs 1/2
    EXPECT_NEAR(d[3], -1.0 / 6, 1e-16);                            // each long response token 1/6
    EXPECT_EQ(run(ml::token_aggregation::token_mean), -1.0);       // 4 tokens / 4
    EXPECT_EQ(d[0], -0.25);
    EXPECT_EQ(d[3], -0.25);
    EXPECT_NEAR(run(ml::token_aggregation::constant), -4.0 / 6, 1e-15);   // (1/2)(1/3 + 3/3)
    EXPECT_EQ(run(ml::token_aggregation::constant, 8), -0.25);     // (1/2)(1/8 + 3/8)
    EXPECT_EQ(m.clip_fraction(), 0.0);
    EXPECT_EQ(m.mean_ratio(), 1.0);
}

TEST(MlPolicyLossMetrics, CountClipsAndMerge)
{
    const std::vector<std::uint8_t> mask{1, 1};
    const std::vector<double> lp{std::log(2.0), 0.0}, old{0.0, 0.0}, adv{1.0, 1.0};
    std::vector<double> d(2);
    ml::policy_loss_metrics<double> a, b;
    ml::policy_loss_rows<double>(backend, mask, lp, old, {}, adv, {}, d, {}, 1, {}, a);
    EXPECT_EQ(a.clip_fraction(), 0.5);
    EXPECT_EQ(a.mean_ratio(), 1.5);
    b.merge(a);
    b.merge(a);
    EXPECT_EQ(b.tokens, 4u);
    EXPECT_EQ(b.clip_fraction(), 0.5);
    EXPECT_EQ(ml::policy_loss_metrics<float>{}.mean_kl(), 0.0f);
}

TEST(MlGae, IsTheRecursionByHandOverTheResponse)
{
    // One row: prompt, then three response tokens, then padding
    const std::vector<std::uint8_t> mask{0, 1, 1, 1, 0};
    const std::vector<double> rewards{9, 0.0, 0.0, 1.0, 9}, values{9, 0.2, 0.4, 0.6, 9};
    std::vector<double> adv(5), ret(5);
    const double gamma = 0.9, lambda = 0.8;
    ml::gae_rows<double>(backend, mask, rewards, values, gamma, lambda, adv, ret, 1);
    const double d3 = 1.0 - 0.6;
    const double d2 = 0.0 + gamma * 0.6 - 0.4;
    const double d1 = 0.0 + gamma * 0.4 - 0.2;
    const double a3 = d3, a2 = d2 + gamma * lambda * a3, a1 = d1 + gamma * lambda * a2;
    EXPECT_NEAR(adv[3], a3, 1e-15);
    EXPECT_NEAR(adv[2], a2, 1e-15);
    EXPECT_NEAR(adv[1], a1, 1e-15);
    EXPECT_NEAR(ret[1], a1 + 0.2, 1e-15);
    EXPECT_EQ(adv[0], 0.0);
    EXPECT_EQ(adv[4], 0.0);
    // lambda = gamma = 1: the advantage is the sum of future rewards less the value
    ml::gae_rows<double>(backend, mask, rewards, values, 1.0, 1.0, adv, ret, 1);
    EXPECT_NEAR(adv[1], 1.0 - 0.2, 1e-15);
    EXPECT_NEAR(ret[1], 1.0, 1e-15);
}

TEST(MlValueLoss, IsHalfTheSquaredErrorClippedAndItsGradient)
{
    const std::vector<std::uint8_t> mask{0, 1, 1, 1};
    const std::vector<double> old{0, 0.5, 0.5, 0.5}, ret{0, 1.0, 1.0, 0.0};
    const std::vector<double> values{7, 0.55, 0.9, 0.9};
    std::vector<double> d(4);
    const double unclipped = ml::value_loss_rows<double>(backend, mask, values, old, ret, 0.0, d);
    EXPECT_NEAR(unclipped, (0.5 * 0.45 * 0.45 + 0.5 * 0.1 * 0.1 + 0.5 * 0.9 * 0.9) / 3, 1e-15);
    EXPECT_EQ(d[0], 0.0);
    EXPECT_NEAR(d[1], -0.45 / 3, 1e-15);
    // Clipped at 0.2: the second value clips to 0.7, whose error 0.3 is larger than 0.1, so it binds with no gradient
    const double clipped = ml::value_loss_rows<double>(backend, mask, values, old, ret, 0.2, d);
    EXPECT_NEAR(clipped, (0.5 * 0.45 * 0.45 + 0.5 * 0.3 * 0.3 + 0.5 * 0.9 * 0.9) / 3, 1e-15);
    EXPECT_EQ(d[2], 0.0);
    for (std::size_t i = 1; i < 4; ++i) {
        auto x = values;
        const double h = 1e-7;
        x[i] += h;
        std::vector<double> scratch(4);
        const double up = ml::value_loss_rows<double>(backend, mask, x, old, ret, 0.2, scratch);
        x[i] -= 2 * h;
        EXPECT_NEAR(d[i], (up - ml::value_loss_rows<double>(backend, mask, x, old, ret, 0.2, scratch)) / (2 * h), 1e-8) << i;
    }
}

TEST(MlMaskedWhiten, GivesMeanZeroAndUnitSampleStdOnTheMask)
{
    const std::vector<std::uint8_t> mask{1, 0, 1, 1, 1};
    const std::vector<double> v{1, 100, 2, 3, 6};
    std::vector<double> out(5);
    ml::masked_whiten<double>(backend, mask, v, out);
    double mean = 0, ss = 0;
    for (std::size_t i = 0; i < 5; ++i)
        mean += mask[i] ? out[i] / 4 : 0;
    for (std::size_t i = 0; i < 5; ++i)
        ss += mask[i] ? out[i] * out[i] : 0;
    EXPECT_NEAR(mean, 0.0, 1e-15);
    EXPECT_NEAR(ss / 3, 1.0, 1e-7);
    EXPECT_EQ(out[1], 0.0);
}

} // namespace
