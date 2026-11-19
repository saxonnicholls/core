//
//  samplers_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/ml/samplers.hpp>: each sampler's kept set and probabilities against a hand
//  computation (llama.cpp's semantics at 39039aa), dist's distribution by chi-square over wyrand draws,
//  mirostat v1/v2 and DRY state transitions against hand-worked cases (DRY's Z-algorithm against a naive
//  repeat count over random histories), penalties against apply_penalties, and the compile-time chain
//  against the run-time chain, in a static_assert and at run time.
//

#include <snicholls/core/ml/samplers.hpp>
#include <snicholls/core/ml/sampling.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <span>
#include <type_traits>
#include <vector>

namespace {

namespace ml = snicholls::ml;
using snicholls::stats::wyrand;
using cand = ml::token_candidates<double>;

static_assert(std::is_trivially_copyable_v<ml::token_data<float>> && sizeof(ml::token_data<float>) == 12);
static_assert(std::is_trivially_copyable_v<ml::top_p_sampler<float>> && std::is_trivially_copyable_v<ml::mirostat_sampler<float>>);
static_assert(ml::accepting_sampler<ml::penalties_sampler<float>> && !ml::accepting_sampler<ml::top_k_sampler>);
static_assert(ml::resettable_sampler<ml::mirostat_v2_sampler<float>> && !ml::resettable_sampler<ml::dist_sampler>);

// A compile-time chain on a tiny vocabulary, and the same chain built at run time, at compile time too
constexpr ml::token_id compile_time_chain(std::uint64_t seed)
{
    std::array<double, 6> logits{0.5, 2.0, -1.0, 1.5, 0.0, 1.0};
    std::array<ml::token_data<double>, 6> storage{};
    ml::sampler_chain chain{ml::penalties_sampler<double, 8>{.last_n = 8, .repeat = 1.3}, ml::top_k_sampler{4},
                            ml::temperature_sampler<double>{0.8}, ml::min_p_sampler<double>{0.1}, ml::dist_sampler{}};
    chain.accept(1);
    chain.accept(3);
    wyrand g(seed);
    return chain.sample(logits, std::span(storage), g);
}

constexpr ml::token_id compile_time_runtime_chain(std::uint64_t seed)
{
    std::array<double, 6> logits{0.5, 2.0, -1.0, 1.5, 0.0, 1.0};
    std::array<ml::token_data<double>, 6> storage{};
    ml::runtime_sampler_chain<double, 8> chain;
    chain.add(ml::penalties_sampler<double, 8>{.last_n = 8, .repeat = 1.3})
        .add(ml::top_k_sampler{4})
        .add(ml::temperature_sampler<double>{0.8})
        .add(ml::min_p_sampler<double>{0.1})
        .add(ml::dist_sampler{});
    chain.accept(1);
    chain.accept(3);
    wyrand g(seed);
    return chain.sample(logits, std::span(storage), g);
}

static_assert(compile_time_chain(7) == compile_time_runtime_chain(7) && compile_time_chain(8) == compile_time_runtime_chain(8));
static_assert(compile_time_chain(7) >= 0 && compile_time_chain(7) < 6);

constexpr std::size_t compile_time_greedy()
{
    std::array<float, 4> logits{1.0f, 3.0f, 3.0f, 2.0f};
    std::array<ml::token_data<float>, 4> storage{};
    ml::sampler_chain chain{ml::greedy_sampler{}};
    wyrand g(1);
    return static_cast<std::size_t>(chain.sample(logits, std::span(storage), g));
}
static_assert(compile_time_greedy() == 1);   // the first of the largest

struct fixture
{
    std::vector<ml::token_data<double>> storage;
    cand c;
    explicit fixture(std::vector<double> logits) : storage(logits.size())
    {
        c = ml::make_candidates(logits, std::span(storage));
    }
};

std::vector<double> logs(std::vector<double> p)
{
    for (double& v : p)
        v = std::log(v);
    return p;
}

std::vector<ml::token_id> ids(const cand& c)
{
    std::vector<ml::token_id> out;
    for (const auto& d : c.data)
        out.push_back(d.id);
    return out;
}

TEST(MlSamplersCandidates, MakeCandidatesIsIdentityOrderedAndUnsorted)
{
    fixture f({0.5, -1.0, 2.0});
    EXPECT_EQ(ids(f.c), (std::vector<ml::token_id>{0, 1, 2}));
    EXPECT_FALSE(f.c.sorted);
    EXPECT_EQ(f.c.selected, -1);
    EXPECT_EQ(f.c.data[2].logit, 2.0);
}

TEST(MlSamplersGreedy, SelectsTheFirstLargest)
{
    fixture f({1.0, 4.0, 4.0, -2.0});
    wyrand g(1);
    ml::greedy_sampler{}.apply(f.c, g);
    EXPECT_EQ(f.c.selected_token(), 1);
}

TEST(MlSamplersTemperature, ZeroKeepsOnlyTheFirstLargest)
{
    fixture f({1.0, 4.0, 4.0, -2.0});
    wyrand g(1);
    ml::temperature_sampler<double>{0.0}.apply(f.c, g);
    const double inf = std::numeric_limits<double>::infinity();
    EXPECT_EQ(f.c.data[0].logit, -inf);
    EXPECT_EQ(f.c.data[1].logit, 4.0);
    EXPECT_EQ(f.c.data[2].logit, -inf);
    EXPECT_EQ(f.c.data[3].logit, -inf);
}

TEST(MlSamplersTemperature, DividesTheLogits)
{
    fixture f({1.0, -2.0});
    wyrand g(1);
    ml::temperature_sampler<double>{0.5}.apply(f.c, g);
    EXPECT_EQ(f.c.data[0].logit, 2.0);
    EXPECT_EQ(f.c.data[1].logit, -4.0);
}

TEST(MlSamplersDist, FollowsTheSoftmaxByChiSquare)
{
    const std::vector<double> logits{0.0, 1.0, 2.0, -0.5, 0.7};
    std::vector<double> p(logits.size());
    double z = 0;
    for (std::size_t i = 0; i < p.size(); ++i)
        z += p[i] = std::exp(logits[i]);
    for (double& v : p)
        v /= z;
    std::vector<ml::token_data<double>> storage(logits.size());
    std::vector<double> counts(logits.size());
    wyrand g(0x5eed);
    const int n = 200000;
    for (int t = 0; t < n; ++t) {
        auto c = ml::make_candidates(logits, std::span(storage));
        ml::dist_sampler{}.apply(c, g);
        counts[static_cast<std::size_t>(c.selected)] += 1;
    }
    double chi2 = 0;
    for (std::size_t i = 0; i < p.size(); ++i)
        chi2 += (counts[i] - n * p[i]) * (counts[i] - n * p[i]) / (n * p[i]);
    EXPECT_LT(chi2, 27.9);   // 4 degrees of freedom: the 1e-5 upper quantile is 25.4
}

TEST(MlSamplersDist, NormalisesAndNeverSelectsAnImpossibleToken)
{
    const double inf = std::numeric_limits<double>::infinity();
    std::vector<double> logits{-inf, 0.0, -inf, 0.0};
    std::vector<ml::token_data<double>> storage(4);
    wyrand g(3);
    for (int t = 0; t < 2000; ++t) {
        auto c = ml::make_candidates(logits, std::span(storage));
        ml::dist_sampler{}.apply(c, g);
        ASSERT_TRUE(c.selected == 1 || c.selected == 3);
        EXPECT_EQ(c.data[0].p, 0.0);
        EXPECT_EQ(c.data[1].p, 0.5);
    }
    fixture one({7.0});
    ml::dist_sampler{}.apply(one.c, g);
    EXPECT_EQ(one.c.selected, 0);
    EXPECT_EQ(one.c.data[0].p, 1.0);
}

TEST(MlSamplersTopK, KeepsTheLargestSortedTiesByLowerId)
{
    fixture f({1.0, 3.0, 2.0, 3.0, 0.0});
    wyrand g(1);
    ml::top_k_sampler{3}.apply(f.c, g);
    EXPECT_EQ(ids(f.c), (std::vector<ml::token_id>{1, 3, 2}));
    EXPECT_TRUE(f.c.sorted);
    fixture all({1.0, 2.0});
    ml::top_k_sampler{0}.apply(all.c, g);
    ml::top_k_sampler{10}.apply(all.c, g);
    EXPECT_EQ(all.c.size(), 2u);
}

TEST(MlSamplersTopP, KeepsTheNucleusAndMinKeep)
{
    wyrand g(1);
    fixture f(logs({0.05, 0.5, 0.15, 0.3}));
    ml::top_p_sampler<double>{0.79}.apply(f.c, g);
    EXPECT_EQ(ids(f.c), (std::vector<ml::token_id>{1, 3}));
    EXPECT_NEAR(f.c.data[0].p, 0.5, 1e-12);   // the whole array's probabilities, not renormalised
    fixture k(logs({0.05, 0.5, 0.15, 0.3}));
    ml::top_p_sampler<double>{0.4, 3}.apply(k.c, g);
    EXPECT_EQ(ids(k.c), (std::vector<ml::token_id>{1, 3, 2}));
    fixture none(logs({0.5, 0.5}));
    ml::top_p_sampler<double>{1.0}.apply(none.c, g);
    EXPECT_FALSE(none.c.sorted);
    EXPECT_EQ(none.c.size(), 2u);
}

TEST(MlSamplersMinP, KeepsTheOrderWhenUnsortedAndFallsBackToTheSortedPath)
{
    wyrand g(1);
    fixture f(logs({0.15, 0.05, 0.5, 0.3}));
    ml::min_p_sampler<double>{0.25}.apply(f.c, g);   // cut at 0.125: 0.15, 0.5, 0.3 in their order
    EXPECT_EQ(ids(f.c), (std::vector<ml::token_id>{0, 2, 3}));
    EXPECT_FALSE(f.c.sorted);
    fixture k(logs({0.15, 0.05, 0.5, 0.3}));
    ml::min_p_sampler<double>{0.9, 2}.apply(k.c, g);   // one passes, two required: sorted, the top two
    EXPECT_EQ(ids(k.c), (std::vector<ml::token_id>{2, 3}));
    EXPECT_TRUE(k.c.sorted);
}

TEST(MlSamplersPenalties, AgreeWithApplyPenaltiesToTheBit)
{
    wyrand g(99);
    std::vector<double> logits(50);
    for (double& v : logits)
        v = snicholls::stats::uniform_real(g) * 6 - 3;
    ml::penalties_sampler<double, 16> s{.last_n = 12, .repeat = 1.4, .frequency = 0.3, .presence = 0.25};
    std::vector<ml::token_id> history;
    for (int i = 0; i < 30; ++i) {
        const auto t = static_cast<ml::token_id>(snicholls::stats::uniform_index(g, 10));
        s.accept(t);
        history.push_back(t);
    }
    std::vector<ml::token_data<double>> storage(logits.size());
    auto c = ml::make_candidates(logits, std::span(storage));
    s.apply(c, g);
    std::vector<double> expected = logits;
    const std::span<const ml::token_id> window(history.data() + history.size() - 12, 12);
    ml::apply_penalties(expected, window, ml::logit_penalties<double>{1.4, 0.25, 0.3});
    for (std::size_t i = 0; i < logits.size(); ++i)
        EXPECT_EQ(c.data[i].logit, expected[i]) << i;
    s.reset();
    auto d = ml::make_candidates(logits, std::span(storage));
    s.apply(d, g);
    EXPECT_EQ(d.data[3].logit, logits[3]);
}

TEST(MlSamplersPenalties, FindTokensInAShuffledArray)
{
    wyrand g(1);
    fixture f({2.0, -1.0, 0.5});
    std::swap(f.c.data[0], f.c.data[2]);
    ml::penalties_sampler<double, 4> s{.last_n = 4, .repeat = 2.0};
    s.accept(0);
    s.accept(1);
    s.apply(f.c, g);
    EXPECT_EQ(f.c.data[2].logit, 1.0);    // token 0, positive: halved
    EXPECT_EQ(f.c.data[1].logit, -2.0);   // token 1, negative: doubled
    EXPECT_EQ(f.c.data[0].logit, 0.5);
}

TEST(MlSamplersTypical, KeepsTheTokensNearestTheEntropy)
{
    wyrand g(1);
    const std::vector<double> p{0.4, 0.3, 0.2, 0.1};
    double h = 0;
    for (double v : p)
        h -= v * std::log(v);
    std::vector<std::size_t> order{0, 1, 2, 3};
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return std::abs(-std::log(p[a]) - h) < std::abs(-std::log(p[b]) - h); });
    std::vector<ml::token_id> expected;
    double cum = 0;
    for (std::size_t i : order) {
        expected.push_back(static_cast<ml::token_id>(i));
        cum += p[i];
        if (cum > 0.45)
            break;
    }
    fixture f(logs(p));
    ml::typical_sampler<double>{0.45}.apply(f.c, g);
    EXPECT_EQ(ids(f.c), expected);
    EXPECT_EQ(expected, (std::vector<ml::token_id>{1, 2}));   // |1.20 - 1.28| and |1.61 - 1.28| are the smallest
    fixture k(logs(p));
    ml::typical_sampler<double>{0.1, 3}.apply(k.c, g);
    EXPECT_EQ(k.c.size(), 3u);
}

TEST(MlSamplersTempExt, TheTemperatureRunsFromMinimumAtZeroEntropyToMaximumAtUniform)
{
    const ml::temp_ext_sampler<double> s{.temp = 1.0, .delta = 0.5, .exponent = 1.0};
    EXPECT_EQ(s.dynamic_temperature(0.0, 4), 0.5);
    EXPECT_NEAR(s.dynamic_temperature(std::log(4.0), 4), 1.5, 1e-15);
    const ml::temp_ext_sampler<double> wide{.temp = 0.3, .delta = 1.0, .exponent = 2.0};
    EXPECT_NEAR(wide.dynamic_temperature(0.5 * std::log(8.0), 8), 0.0 + 1.3 * 0.25, 1e-15);   // the minimum clamps at 0
}

TEST(MlSamplersTempExt, AppliesTheDynamicTemperatureAndRenormalises)
{
    wyrand g(1);
    const std::vector<double> logits{2.0, 1.0, 0.0};
    fixture f(logits);
    const ml::temp_ext_sampler<double> s{.temp = 1.0, .delta = 0.5, .exponent = 1.0};
    s.apply(f.c, g);
    double z = 0, h = 0;
    for (double v : logits)
        z += std::exp(v - 2.0);
    for (double v : logits)
        h -= std::exp(v - 2.0) / z * std::log(std::exp(v - 2.0) / z);
    const double t = 0.5 + 1.0 * h / std::log(3.0);
    double z2 = 0;
    for (double v : logits)
        z2 += std::exp(v / t - 2.0 / t);
    EXPECT_TRUE(f.c.sorted);
    EXPECT_NEAR(f.c.data[0].logit, 2.0 / t, 1e-12);
    EXPECT_NEAR(f.c.data[2].p, std::exp(0.0 / t - 2.0 / t) / z2, 1e-12);
    fixture plain({2.0, 1.0});
    ml::temp_ext_sampler<double>{.temp = 2.0, .delta = 0.0}.apply(plain.c, g);
    EXPECT_EQ(plain.c.data[0].logit, 1.0);
}

TEST(MlSamplersXtc, RemovesTheTopChoicesButTheLeastProbableOfThem)
{
    wyrand g(1);
    fixture f(logs({0.05, 0.5, 0.15, 0.3}));
    ml::xtc_sampler<double>{.probability = 1.0, .threshold = 0.1}.apply(f.c, g);
    EXPECT_EQ(ids(f.c), (std::vector<ml::token_id>{2, 0}));   // 0.5 and 0.3 removed, 0.15 kept
    fixture off(logs({0.05, 0.5, 0.15, 0.3}));
    ml::xtc_sampler<double>{.probability = 0.0, .threshold = 0.1}.apply(off.c, g);
    EXPECT_EQ(off.c.size(), 4u);
    fixture keep(logs({0.05, 0.5, 0.15, 0.3}));
    ml::xtc_sampler<double>{.probability = 1.0, .threshold = 0.1, .min_keep = 3}.apply(keep.c, g);
    EXPECT_EQ(keep.c.size(), 4u);
}

TEST(MlSamplersXtc, ActsWithItsProbability)
{
    wyrand g(42);
    int acted = 0;
    const int n = 20000;
    for (int t = 0; t < n; ++t) {
        fixture f(logs({0.6, 0.3, 0.1}));
        ml::xtc_sampler<double>{.probability = 0.3, .threshold = 0.2}.apply(f.c, g);
        acted += f.c.size() == 2 ? 1 : 0;
    }
    EXPECT_NEAR(static_cast<double>(acted) / n, 0.3, 5 * std::sqrt(0.3 * 0.7 / n));
}

TEST(MlSamplersTopNSigma, MasksBelowMaxMinusNSigma)
{
    wyrand g(1);
    const double inf = std::numeric_limits<double>::infinity();
    fixture f({3.0, 2.0, 1.0, 0.0, -inf});
    ml::top_n_sigma_sampler<double>{1.0}.apply(f.c, g);   // mean 1.5, sigma sqrt(1.25): cut at 1.882
    EXPECT_EQ(f.c.data[0].logit, 3.0);
    EXPECT_EQ(f.c.data[1].logit, 2.0);
    EXPECT_EQ(f.c.data[2].logit, -inf);
    EXPECT_EQ(f.c.data[3].logit, -inf);
}

TEST(MlSamplersMirostatV2, TruncatesBySurpriseAndMovesMu)
{
    const std::vector<double> p{0.5, 0.3, 0.15, 0.05};
    ml::mirostat_v2_sampler<double> s{.tau = 1.5, .eta = 0.2, .mu = 2.0};
    fixture f(logs(p));
    wyrand g(5);
    s.apply(f.c, g);
    ASSERT_EQ(f.c.size(), 2u);   // surprises 1, 1.74, 2.74, 4.32 bits: two at most 2
    EXPECT_NEAR(f.c.data[0].p, 0.625, 1e-12);
    const double ps = f.c.data[static_cast<std::size_t>(f.c.selected)].p;
    EXPECT_NEAR(s.mu, 2.0 - 0.2 * (-std::log2(ps) - 1.5), 1e-12);
    s.reset();
    EXPECT_EQ(s.mu, 3.0);
    ml::mirostat_v2_sampler<double> tight{.tau = 1.0, .eta = 0.1, .mu = 0.1};
    fixture one(logs(p));
    tight.apply(one.c, g);
    EXPECT_EQ(one.c.size(), 1u);   // nothing passes: the top one is kept
    EXPECT_EQ(one.c.selected_token(), 0);
    EXPECT_NEAR(tight.mu, 0.1 - 0.1 * (0.0 - 1.0), 1e-12);   // p 1 after renormalising: surprise 0
}

TEST(MlSamplersMirostat, EstimatesZipfAndMovesMu)
{
    const std::vector<double> p{0.4, 0.25, 0.15, 0.1, 0.06, 0.04};
    ml::mirostat_sampler<double> s{.n_vocab = 6, .tau = 2.0, .eta = 0.1, .m = 4, .mu = 4.0};
    double tb = 0, tt = 0;
    for (std::size_t i = 0; i < 3; ++i) {
        const double t = std::log((i + 2.0) / (i + 1.0)), b = std::log(p[i] / p[i + 1]);
        tb += t * b;
        tt += t * t;
    }
    const double sh = tb / tt, eh = sh - 1;
    const double k = std::pow(eh * std::pow(2.0, 4.0) / (1 - std::pow(6.0, -eh)), 1 / sh);
    const std::size_t kk = std::clamp(static_cast<std::size_t>(k), std::size_t{1}, std::size_t{6});
    fixture f(logs(p));
    wyrand g(11);
    s.apply(f.c, g);
    EXPECT_EQ(f.c.size(), kk);
    double z = 0;
    for (std::size_t i = 0; i < kk; ++i)
        z += p[i];
    const double ps = p[static_cast<std::size_t>(f.c.selected)] / z;
    EXPECT_NEAR(s.mu, 4.0 - 0.1 * (-std::log2(ps) - 2.0), 1e-9);
    s.reset();
    EXPECT_EQ(s.mu, 4.0);
}

// DRY over tokens a = 0, b = 1, c = 2, y = 3
std::vector<double> dry_logits(const std::vector<ml::token_id>& history, ml::dry_sampler<double, 32> s, std::size_t vocab)
{
    for (const auto t : history)
        s.accept(t);
    std::vector<double> zeros(vocab, 0.0);
    std::vector<ml::token_data<double>> storage(vocab);
    auto c = ml::make_candidates(zeros, std::span(storage));
    wyrand g(1);
    s.apply(c, g);
    std::vector<double> out(vocab);
    for (const auto& d : c.data)
        out[static_cast<std::size_t>(d.id)] = d.logit;
    return out;
}

TEST(MlSamplersDry, PenalisesTheContinuationsOfTheLlamaCppExample)
{
    const std::vector<ml::token_id> h{0, 1, 2, 2, 1, 2, 3, 0, 1, 2};   // a b c c b c y a b c
    const auto out = dry_logits(h, {.multiplier = 1.0, .base = 2.0, .allowed_length = 2, .penalty_last_n = 32}, 5);
    EXPECT_EQ(out[2], -2.0);   // c would make "a b c c": a repeat of 3, 2^(3 - 2)
    EXPECT_EQ(out[3], -1.0);   // y would make "b c y": a repeat of 2, 2^0
    EXPECT_EQ(out[0], 0.0);
    EXPECT_EQ(out[1], 0.0);   // b continues a repeat of 1 only
    EXPECT_EQ(out[4], 0.0);
}

TEST(MlSamplersDry, BreakersLimitTheMatchAndAreNeverPenalised)
{
    const std::vector<ml::token_id> h{0, 1, 2, 2, 1, 2, 3, 0, 1, 2};
    const std::array<ml::token_id, 1> y{3};
    const std::array<std::span<const ml::token_id>, 1> single{y};
    const auto out = dry_logits(h, {.multiplier = 1.0, .base = 2.0, .allowed_length = 2, .penalty_last_n = 32, .breakers = single}, 5);
    EXPECT_EQ(out[3], 0.0);    // a single-token breaker is not penalised
    EXPECT_EQ(out[2], -2.0);   // "a b c" after the y: still 3, within the limit of 3
    // A breaker "y a" three tokens back limits repeats to 2: c's repeat of 3 is clamped to 2
    const std::array<ml::token_id, 2> ya{3, 0};
    const std::array<std::span<const ml::token_id>, 1> pair{ya};
    const auto two = dry_logits(h, {.multiplier = 1.0, .base = 2.0, .allowed_length = 2, .penalty_last_n = 32, .breakers = pair}, 5);
    EXPECT_EQ(two[2], -1.0);
    EXPECT_EQ(two[3], -1.0);   // y heads a longer breaker only
    // A breaker as the last token stops everything below allowed_length
    const std::array<ml::token_id, 1> cc{2};
    const std::array<std::span<const ml::token_id>, 1> last{cc};
    const auto none = dry_logits(h, {.multiplier = 1.0, .base = 2.0, .allowed_length = 2, .penalty_last_n = 32, .breakers = last}, 5);
    EXPECT_EQ(none, std::vector<double>(5, 0.0));
}

TEST(MlSamplersDry, ShortOrDisabledHistoriesDoNothing)
{
    EXPECT_EQ(dry_logits({0, 0}, {.multiplier = 1.0, .base = 2.0, .allowed_length = 2, .penalty_last_n = 32}, 2), std::vector<double>(2, 0.0));
    EXPECT_EQ(dry_logits({0, 1, 0, 1, 0, 1}, {.multiplier = 0.0}, 2), std::vector<double>(2, 0.0));
    EXPECT_EQ(dry_logits({0, 1, 0, 1, 0, 1}, {.multiplier = 1.0, .base = 0.5}, 2), std::vector<double>(2, 0.0));
}

TEST(MlSamplersDry, TheWindowAndTheExponentClamp)
{
    // 0 1 0 1 0 1 0: token 1 would continue "0 1 0 1 0", a repeat of 5, with the whole history
    const std::vector<ml::token_id> h{0, 1, 0, 1, 0, 1, 0};
    const auto full = dry_logits(h, {.multiplier = 1.0, .base = 2.0, .allowed_length = 1, .penalty_last_n = 32}, 2);
    EXPECT_EQ(full[1], -std::pow(2.0, 4.0));
    const auto window = dry_logits(h, {.multiplier = 1.0, .base = 2.0, .allowed_length = 1, .penalty_last_n = 4}, 2);
    EXPECT_EQ(window[1], -std::pow(2.0, 1.0));   // the last four tokens, 1 0 1 0: a repeat of 2
    const auto big = dry_logits(std::vector<ml::token_id>(30, 0), {.multiplier = 1.0, .base = 1e10, .allowed_length = 1, .penalty_last_n = 32}, 1);
    const int max_exponent = static_cast<int>(88.7228391f / static_cast<float>(std::log(1e10)));
    EXPECT_EQ(big[0], -std::pow(1e10, max_exponent));
}

// A naive DRY: the repeat lengths by direct comparison, no Z-algorithm, no breakers
std::vector<double> naive_dry(const std::vector<ml::token_id>& h, double mult, double base, int allowed, std::size_t vocab)
{
    const int n = static_cast<int>(h.size());
    const auto rat = [&](int i) { return h[static_cast<std::size_t>(n - 1 - i)]; };
    std::map<ml::token_id, int> best;
    for (int k = 1; k < n; ++k) {
        int len = 0;
        while (len + k < n && rat(len) == rat(len + k))
            ++len;
        if (len >= allowed) {
            const ml::token_id next = rat(k - 1);
            best[next] = std::max(best[next], len);
        }
    }
    std::vector<double> out(vocab, 0.0);
    for (const auto& [t, len] : best)
        out[static_cast<std::size_t>(t)] -= mult * std::pow(base, len - allowed);
    return out;
}

TEST(MlSamplersDry, TheZAlgorithmMatchesANaiveRepeatCount)
{
    wyrand g(2026);
    for (int trial = 0; trial < 300; ++trial) {
        const std::size_t vocab = 2 + snicholls::stats::uniform_index(g, 3);
        std::vector<ml::token_id> h(3 + snicholls::stats::uniform_index(g, 29));
        for (auto& t : h)
            t = static_cast<ml::token_id>(snicholls::stats::uniform_index(g, vocab));
        const int allowed = 1 + static_cast<int>(snicholls::stats::uniform_index(g, 3));
        const auto got = dry_logits(h, {.multiplier = 0.5, .base = 1.5, .allowed_length = allowed, .penalty_last_n = 32}, vocab);
        const auto want = naive_dry(h, 0.5, 1.5, allowed, vocab);
        if (static_cast<int>(h.size()) <= allowed)
            continue;
        for (std::size_t i = 0; i < vocab; ++i)
            ASSERT_DOUBLE_EQ(got[i], want[i]) << "trial " << trial;
    }
}

TEST(MlSamplersLogitBias, AddsBiasesInAnyOrder)
{
    wyrand g(1);
    const double inf = std::numeric_limits<double>::infinity();
    const std::array<ml::logit_bias_entry<double>, 3> biases{{{2, 1.5}, {0, -inf}, {9, 3.0}}};
    fixture f({1.0, 1.0, 1.0});
    std::swap(f.c.data[0], f.c.data[2]);
    ml::logit_bias_sampler<double>{biases}.apply(f.c, g);
    EXPECT_EQ(f.c.data[0].logit, 2.5);   // token 2
    EXPECT_EQ(f.c.data[2].logit, -inf);  // token 0
    EXPECT_EQ(f.c.data[1].logit, 1.0);
}

TEST(MlSamplersChain, CompileTimeAndRunTimeChainsAgreeOverAGeneration)
{
    using T = float;
    const std::array<ml::token_id, 1> brk{5};
    const std::array<std::span<const ml::token_id>, 1> breakers{brk};
    ml::sampler_chain fixed{ml::penalties_sampler<T, 16>{.last_n = 16, .repeat = 1.2f, .frequency = 0.1f},
                            ml::dry_sampler<T, 16>{.multiplier = 0.8f, .base = 1.75f, .allowed_length = 2, .penalty_last_n = 16, .breakers = breakers},
                            ml::top_k_sampler{20}, ml::typical_sampler<T>{0.95f}, ml::temp_ext_sampler<T>{.temp = 0.9f, .delta = 0.3f},
                            ml::top_p_sampler<T>{0.9f}, ml::min_p_sampler<T>{0.02f}, ml::xtc_sampler<T>{.probability = 0.5f, .threshold = 0.2f},
                            ml::dist_sampler{}};
    ml::runtime_sampler_chain<T, 16> dynamic;
    std::apply([&](const auto&... s) { (dynamic.add(s), ...); }, fixed.samplers);
    EXPECT_EQ(dynamic.samplers().size(), 9u);
    wyrand g1(77), g2(77), model(5);
    std::vector<T> logits(32);
    std::vector<ml::token_data<T>> s1(32), s2(32);
    for (int step = 0; step < 200; ++step) {
        for (T& v : logits)
            v = static_cast<T>(snicholls::stats::uniform_real(model) * 8 - 4);
        const auto a = fixed.sample(logits, std::span(s1), g1);
        const auto b = dynamic.sample(logits, std::span(s2), g2);
        ASSERT_EQ(a, b) << step;
        fixed.accept(a);
        dynamic.accept(b);
    }
    fixed.reset();
    dynamic.reset();
    EXPECT_EQ(std::get<0>(fixed.samplers).history.count, 0u);
}

TEST(MlSamplersChain, MirostatChainsAreStatefulAndReproducible)
{
    const std::vector<double> logits{2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5};
    std::vector<ml::token_data<double>> storage(logits.size());
    const auto run = [&](std::uint64_t seed) {
        ml::sampler_chain chain{ml::mirostat_v2_sampler<double>{.tau = 2.0, .eta = 0.1, .mu = 4.0}};
        wyrand g(seed);
        std::vector<ml::token_id> out;
        for (int i = 0; i < 50; ++i)
            out.push_back(chain.sample(logits, std::span(storage), g));
        return std::pair{out, std::get<0>(chain.samplers).mu};
    };
    EXPECT_EQ(run(3), run(3));
    EXPECT_NE(run(3).second, 4.0);
}

TEST(MlSamplersChain, FallsBackToGreedyWhenNothingSelects)
{
    const std::vector<double> logits{0.0, 3.0, 1.0};
    std::vector<ml::token_data<double>> storage(3);
    ml::sampler_chain chain{ml::top_k_sampler{2}};
    wyrand g(1);
    EXPECT_EQ(chain.sample(logits, std::span(storage), g), 1);
    ml::runtime_sampler_chain<double> empty;
    EXPECT_EQ(empty.sample(logits, std::span(storage), g), 1);
}

TEST(MlSamplersChain, SeedsReproduceAcrossRuns)
{
    std::vector<double> logits(100);
    for (std::size_t i = 0; i < logits.size(); ++i)
        logits[i] = std::sin(static_cast<double>(i));
    std::vector<ml::token_data<double>> storage(logits.size());
    const auto draw = [&](std::uint64_t seed) {
        ml::sampler_chain chain{ml::top_k_sampler{40}, ml::temperature_sampler<double>{0.7}, ml::dist_sampler{}};
        wyrand g(seed);
        std::vector<ml::token_id> out;
        for (int i = 0; i < 20; ++i)
            out.push_back(chain.sample(logits, std::span(storage), g));
        return out;
    };
    EXPECT_EQ(draw(1), draw(1));
    EXPECT_NE(draw(1), draw(2));
}

} // namespace
