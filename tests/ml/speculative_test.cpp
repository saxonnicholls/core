//
//  speculative_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/ml/speculative.hpp>: n-gram lookup drafts, greedy verification by hand,
//  speculative sampling preserving the target distribution (chi-square over wyrand draws, as llama.cpp's
//  test-speculative-rejection-sampling.cpp checks the first token), its acceptance rate against the closed
//  form sum min(p, q), the degenerate cases exactly, the tree mask and depth, greedy and sampled tree
//  verification, and the MTP drafters; compile-time use.
//

#include <snicholls/core/ml/speculative.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace {

namespace ml = snicholls::ml;
using snicholls::stats::wyrand;

constexpr bool compile_time()
{
    const std::array<int, 9> h{1, 2, 3, 4, 9, 1, 2, 3, 7};
    const std::array<int, 7> h2{1, 2, 3, 4, 9, 2, 3};
    std::array<int, 4> d{};
    const std::size_t none = ml::ngram_draft(std::span<const int>(h), 2, 3, std::span(d));
    const std::size_t two = ml::ngram_draft(std::span<const int>(h2), 2, 3, std::span(d));
    const std::array<int, 2> draft{2, 0};
    const std::array<float, 9> logits{0, 0, 1, /**/ 1, 0, 0, /**/ 0, 3, 0};
    const auto r = ml::verify_greedy(std::span<const int>(draft), logits, 3);
    return none == 0 && two == 4 && d[0] == 4 && d[1] == 9 && d[2] == 2 && d[3] == 3 && r.accepted == 2 && r.next == 1;
}
static_assert(compile_time());
static_assert(std::is_trivially_copyable_v<ml::verify_result<int>> && std::is_trivially_copyable_v<ml::tree_verify_result<int>>);

TEST(MlSpeculativeNgram, FindsTheMostRecentLongestMatch)
{
    const std::vector<int> h{5, 1, 2, 3, 8, 1, 2, 3, 9, 4, 2, 3};
    std::array<int, 3> d{};
    // "2 3" occurs at 2 and 6 (most recent: 6, followed by 9 4 2); "4 2 3" and "3 2 3"... do not occur
    EXPECT_EQ(ml::ngram_draft(std::span<const int>(h), 2, 3, std::span(d)), 3u);
    EXPECT_EQ(d, (std::array<int, 3>{9, 4, 2}));
    // Longer first: "1 2 3" is a suffix of another history and wins over "2 3"
    const std::vector<int> h2{1, 2, 3, 7, 0, 2, 3, 6, 1, 2, 3};
    EXPECT_EQ(ml::ngram_draft(std::span<const int>(h2), 2, 3, std::span(d)), 3u);
    EXPECT_EQ(d, (std::array<int, 3>{7, 0, 2}));
    // With n_max 2 the most recent "2 3" wins
    EXPECT_EQ(ml::ngram_draft(std::span<const int>(h2), 2, 2, std::span(d)), 3u);
    EXPECT_EQ(d, (std::array<int, 3>{6, 1, 2}));
}

TEST(MlSpeculativeNgram, StopsAtTheEndOfTheHistoryAndOnNoMatch)
{
    std::array<int, 4> d{};
    const std::vector<int> h{3, 4, 3};   // "3" occurs at 0, followed by "4 3"
    EXPECT_EQ(ml::ngram_draft(std::span<const int>(h), 1, 2, std::span(d)), 2u);
    EXPECT_EQ(d[0], 4);
    EXPECT_EQ(d[1], 3);
    const std::vector<int> fresh{1, 2, 3, 4};
    EXPECT_EQ(ml::ngram_draft(std::span<const int>(fresh), 1, 3, std::span(d)), 0u);
    EXPECT_EQ(ml::ngram_draft(std::span<const int>{}, 1, 3, std::span(d)), 0u);
    const std::vector<int> one{7};
    EXPECT_EQ(ml::ngram_draft(std::span<const int>(one), 1, 3, std::span(d)), 0u);
    // A repeated token: overlapping occurrences count
    const std::vector<int> run{5, 5, 5};
    EXPECT_EQ(ml::ngram_draft(std::span<const int>(run), 2, 2, std::span(d)), 1u);
    EXPECT_EQ(d[0], 5);
}

TEST(MlSpeculativeGreedy, AcceptsThePrefixAndAddsTheTargetsToken)
{
    const std::size_t v = 4;
    const auto rows = [&](std::vector<int> best) {
        std::vector<float> x(best.size() * v, 0.0f);
        for (std::size_t i = 0; i < best.size(); ++i)
            x[i * v + static_cast<std::size_t>(best[i])] = 1.0f;
        return x;
    };
    const std::vector<int> draft{1, 2, 3};
    auto r = ml::verify_greedy(std::span<const int>(draft), rows({1, 2, 0, 3}), v);
    EXPECT_EQ(r.accepted, 2u);
    EXPECT_EQ(r.next, 0);
    r = ml::verify_greedy(std::span<const int>(draft), rows({1, 2, 3, 2}), v);
    EXPECT_EQ(r.accepted, 3u);   // all, and the bonus
    EXPECT_EQ(r.next, 2);
    r = ml::verify_greedy(std::span<const int>(draft), rows({0, 2, 3, 2}), v);
    EXPECT_EQ(r.accepted, 0u);
    EXPECT_EQ(r.next, 0);
    r = ml::verify_greedy(std::span<const int>{}, rows({3}), v);
    EXPECT_EQ(r.accepted, 0u);
    EXPECT_EQ(r.next, 3);
}

// Draws a token from q by the inverse CDF
int draw(const std::vector<double>& q, wyrand& g)
{
    const double u = snicholls::stats::uniform_real(g);
    double c = 0;
    for (std::size_t i = 0; i < q.size(); ++i) {
        c += q[i];
        if (u < c)
            return static_cast<int>(i);
    }
    return static_cast<int>(q.size() - 1);
}

double chi_square(const std::vector<double>& counts, const std::vector<double>& p, double n)
{
    double chi2 = 0;
    for (std::size_t i = 0; i < p.size(); ++i)
        if (p[i] > 0) {
            chi2 += (counts[i] - n * p[i]) * (counts[i] - n * p[i]) / (n * p[i]);
        } else {
            EXPECT_EQ(counts[i], 0.0) << "a token of target probability 0 was emitted";
        }
    return chi2;
}

TEST(MlSpeculativeSampled, TheFirstTokenIsDistributedAsTheTarget)
{
    const std::vector<double> p{0.1, 0.4, 0.05, 0.3, 0.15, 0.0};
    const std::vector<double> q{0.3, 0.1, 0.2, 0.1, 0.1, 0.2};
    std::vector<double> rows = p;
    rows.insert(rows.end(), p.begin(), p.end());   // the bonus row: any distribution
    std::vector<double> counts(p.size()), accepted(1);
    wyrand g(1234);
    const int n = 200000;
    double beta = 0;
    for (std::size_t i = 0; i < p.size(); ++i)
        beta += std::min(p[i], q[i]);
    for (int t = 0; t < n; ++t) {
        const std::array<int, 1> d{draw(q, g)};
        const auto r = ml::verify_sampled(std::span<const int>(d), rows, q, p.size(), g);
        const int first = r.accepted == 1 ? d[0] : r.next;
        counts[static_cast<std::size_t>(first)] += 1;
        accepted[0] += static_cast<double>(r.accepted);
    }
    EXPECT_LT(chi_square(counts, p, n), 25.4);   // 4 degrees of freedom (5 support tokens): the 1e-5 upper quantile
    EXPECT_NEAR(accepted[0] / n, beta, 5 * std::sqrt(beta * (1 - beta) / n));   // the acceptance rate sum min(p, q)
}

TEST(MlSpeculativeSampled, TwoDraftTokensKeepTheJointDistribution)
{
    // Independent positions: p1, p2 for the target, q1, q2 for the drafter; the emitted pair must be p1 x p2
    const std::vector<double> p1{0.5, 0.3, 0.2}, p2{0.2, 0.2, 0.6}, q1{0.2, 0.5, 0.3}, q2{0.6, 0.3, 0.1};
    std::vector<double> rows = p1, qs = q1;
    rows.insert(rows.end(), p2.begin(), p2.end());
    rows.insert(rows.end(), {1.0, 0.0, 0.0});
    qs.insert(qs.end(), q2.begin(), q2.end());
    std::vector<double> joint(9), counts(9);
    for (std::size_t a = 0; a < 3; ++a)
        for (std::size_t b = 0; b < 3; ++b)
            joint[a * 3 + b] = p1[a] * p2[b];
    wyrand g(77), gq(78);
    const int n = 200000;
    int pairs = 0;
    for (int t = 0; t < n; ++t) {
        const std::array<int, 2> d{draw(q1, gq), draw(q2, gq)};
        const auto r = ml::verify_sampled(std::span<const int>(d), rows, qs, 3, g);
        int first = 0, second = 0;
        if (r.accepted == 0) {
            first = r.next;
            // the second token would come from the next round; draw it from p2 as the target would
            second = draw(p2, g);
        } else if (r.accepted == 1) {
            first = d[0];
            second = r.next;
        } else {
            first = d[0];
            second = d[1];
            EXPECT_EQ(r.next, 0);   // the bonus row is one-hot
        }
        counts[static_cast<std::size_t>(first * 3 + second)] += 1;
        ++pairs;
    }
    EXPECT_LT(chi_square(counts, joint, pairs), 33.7);   // 8 degrees of freedom: the 1e-5 upper quantile
}

TEST(MlSpeculativeSampled, DegenerateCasesAreExact)
{
    // q = p: always accepted, never a draw for the test
    const std::vector<double> p{0.25, 0.75, 0.25, 0.75};
    const std::array<int, 1> d{1};
    wyrand g(5);
    auto r = ml::verify_sampled(std::span<const int>(d), p, std::span(p).first(2), 2, g);
    EXPECT_EQ(r.accepted, 1u);
    // The draft proposes a token the target excludes: always rejected, the residual is the target itself
    const std::vector<double> target{0.0, 1.0, 0.5, 0.5};
    const std::vector<double> q{1.0, 0.0};
    const std::array<int, 1> zero{0};
    for (int t = 0; t < 100; ++t) {
        r = ml::verify_sampled(std::span<const int>(zero), target, q, 2, g);
        EXPECT_EQ(r.accepted, 0u);
        EXPECT_EQ(r.next, 1);
    }
    // p one-hot on the draft token: accepted with probability 1 / ... p(x) >= q(x): always
    const std::vector<double> sure{0.0, 1.0, 1.0, 0.0};
    const std::vector<double> half{0.5, 0.5};
    const std::array<int, 1> one{1};
    r = ml::verify_sampled(std::span<const int>(one), sure, half, 2, g);
    EXPECT_EQ(r.accepted, 1u);
    EXPECT_EQ(r.next, 0);
}

TEST(MlSpeculativeTree, TheMaskIsEachNodesAncestorsAndItself)
{
    // root -> 0 -> 2, root -> 1 -> 3 -> 4
    const std::array<std::int32_t, 5> parent{-1, -1, 0, 1, 3};
    std::array<std::uint8_t, 25> mask{};
    ml::tree_attention_mask(std::span<const std::int32_t>(parent), std::span(mask));
    const std::array<std::uint8_t, 25> expected{1, 0, 0, 0, 0, /**/ 0, 1, 0, 0, 0, /**/ 1, 0, 1, 0, 0, /**/ 0, 1, 0, 1, 0, /**/ 0, 1, 0, 1, 1};
    EXPECT_EQ(mask, expected);
    EXPECT_EQ(ml::tree_depth(parent, 0), 1u);
    EXPECT_EQ(ml::tree_depth(parent, 2), 2u);
    EXPECT_EQ(ml::tree_depth(parent, 4), 3u);
}

TEST(MlSpeculativeTree, GreedyFollowsTheTargetsArgmax)
{
    const std::size_t v = 5;
    const std::array<int, 5> tokens{3, 1, 4, 2, 0};
    const std::array<std::int32_t, 5> parent{-1, -1, 0, 1, 3};
    // argmax at the root 1 (node 1), at node 1 2 (node 3), at node 3 4 (no child has token 4)
    const std::array<int, 6> best{1, 0, 2, 0, 4, 0};
    std::vector<float> logits(6 * v, 0.0f);
    for (std::size_t i = 0; i < 6; ++i)
        logits[i * v + static_cast<std::size_t>(best[i])] = 1.0f;
    std::array<std::int32_t, 4> path{};
    const auto r = ml::verify_tree_greedy(std::span<const int>(tokens), std::span<const std::int32_t>(parent), logits, v, std::span(path));
    EXPECT_EQ(r.length, 2u);
    EXPECT_EQ(path[0], 1);
    EXPECT_EQ(path[1], 3);
    EXPECT_EQ(r.next, 4);
    // No child matches at the root
    std::vector<float> miss(6 * v, 0.0f);
    miss[2] = 1.0f;
    const auto m = ml::verify_tree_greedy(std::span<const int>(tokens), std::span<const std::int32_t>(parent), miss, v, std::span(path));
    EXPECT_EQ(m.length, 0u);
    EXPECT_EQ(m.next, 2);
}

TEST(MlSpeculativeTree, SampledVerificationOfIndependentSiblingsPreservesTheTarget)
{
    // Two children of the root drawn independently from q; the first emitted token must follow p
    const std::vector<double> p{0.15, 0.45, 0.25, 0.15}, q{0.4, 0.1, 0.1, 0.4};
    const std::size_t v = p.size();
    const std::array<std::int32_t, 2> parent{-1, -1};
    std::vector<double> target(3 * v), proposal(3 * v);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t t = 0; t < v; ++t) {
            target[i * v + t] = p[t];
            proposal[i * v + t] = q[t];
        }
    std::vector<double> scratch(v), counts(v);
    std::array<std::int32_t, 2> path{};
    wyrand g(31), gq(32);
    const int n = 200000;
    for (int t = 0; t < n; ++t) {
        const std::array<int, 2> tokens{draw(q, gq), draw(q, gq)};
        const auto r = ml::verify_tree_sampled(std::span<const int>(tokens), std::span<const std::int32_t>(parent), target, proposal, v,
                                               std::span(path), std::span(scratch), g);
        const int first = r.length > 0 ? tokens[static_cast<std::size_t>(path[0])] : r.next;
        counts[static_cast<std::size_t>(first)] += 1;
        ASSERT_LE(r.length, 1u);
    }
    EXPECT_LT(chi_square(counts, p, n), 25.0);   // 3 degrees of freedom: the 1e-5 upper quantile is 25.9
}

TEST(MlSpeculativeTree, SampledVerificationWalksDownAChain)
{
    // A chain of two nodes whose q equals p: both accepted, then the bonus from node 1's row
    const std::size_t v = 2;
    const std::array<int, 2> tokens{1, 0};
    const std::array<std::int32_t, 2> parent{-1, 0};
    const std::vector<double> target{0.3, 0.7, 0.6, 0.4, 0.0, 1.0}, proposal = target;
    std::vector<double> scratch(v);
    std::array<std::int32_t, 2> path{};
    wyrand g(9);
    const auto r = ml::verify_tree_sampled(std::span<const int>(tokens), std::span<const std::int32_t>(parent), target, proposal, v,
                                           std::span(path), std::span(scratch), g);
    EXPECT_EQ(r.length, 2u);
    EXPECT_EQ(path[0], 0);
    EXPECT_EQ(path[1], 1);
    EXPECT_EQ(r.next, 1);
}

TEST(MlSpeculativeMtp, GreedyDraftsAreEachModulesArgmax)
{
    const std::array<float, 6> logits{0.1f, 2.0f, -1.0f, /**/ 3.0f, 0.0f, 3.0f};
    std::array<int, 2> d{};
    ml::mtp_draft_greedy(logits, 3, std::span(d));
    EXPECT_EQ(d[0], 1);
    EXPECT_EQ(d[1], 0);   // the first of equal ones
}

TEST(MlSpeculativeMtp, SampledDraftsKeepTheirDistributionsAndFollowThem)
{
    const std::vector<double> logits{0.0, 1.0, 2.0, /**/ 1.0, 1.0, -10.0};
    std::array<int, 2> d{};
    std::vector<double> q(6), counts(3);
    wyrand g(3);
    const int n = 100000;
    for (int t = 0; t < n; ++t) {
        ml::mtp_draft_sample(logits, 3, 0.5, std::span(d), std::span(q), g);
        counts[static_cast<std::size_t>(d[0])] += 1;
    }
    double z = 0;
    for (std::size_t i = 0; i < 3; ++i)
        z += std::exp(logits[i] / 0.5);
    std::vector<double> p(3);
    for (std::size_t i = 0; i < 3; ++i) {
        p[i] = std::exp(logits[i] / 0.5) / z;
        EXPECT_NEAR(q[i], p[i], 1e-12);
    }
    EXPECT_NEAR(q[3] + q[4] + q[5], 1.0, 1e-12);
    EXPECT_NEAR(q[3], q[4], 1e-15);
    EXPECT_LT(chi_square(counts, p, n), 23.0);   // 2 degrees of freedom: the 1e-5 upper quantile
}

} // namespace
