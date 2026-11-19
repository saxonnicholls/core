//
//  preference_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/ml/preference.hpp>: DPO, IPO, SimPO and ORPO against direct computations from their
//  papers' formulas in long double; every gradient against central differences; the edge cases (identical chosen and
//  rejected, beta -> 0, a large margin with no overflow, SimPO's margin gamma, ORPO's SFT term); the metrics added and
//  merged; the batch on the backend as the per-pair losses over n; the gradient through the sequence log-probabilities
//  to the logits against central differences; a tiny DPO run in which the policy's margin and accuracy rise;
//  compile-time use, triviality and the noexcept claims.
//

#include <snicholls/core/ml/preference.hpp>

#include <snicholls/core/linalg/reference.hpp>
#include <snicholls/core/ml/logprobs.hpp>
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

static_assert(std::is_trivially_copyable_v<ml::dpo> && std::is_trivially_copyable_v<ml::ipo> && std::is_trivially_copyable_v<ml::simpo>
              && std::is_trivially_copyable_v<ml::orpo>);
static_assert(std::is_trivially_copyable_v<ml::pair_loss<float>> && std::is_trivially_copyable_v<ml::preference_metrics<double>>);
static_assert(noexcept(ml::preference_loss(ml::dpo{}, 0.0f, 0.0f, 0.0f, 0.0f)));
static_assert(noexcept(ml::preference_loss(ml::orpo{}, -1.0, -2.0)));
static_assert(noexcept(std::declval<ml::preference_metrics<float>&>().merge({})));
static_assert(ml::dpo{}.beta == 0.1 && ml::simpo{}.beta == 2.0 && ml::simpo{}.gamma == 1.0 && ml::orpo{}.lambda == 0.1);

constexpr bool at_compile_time()
{
    const auto d = ml::preference_loss(ml::dpo{.beta = 0.5}, -10.0, -12.0, -11.0, -11.0);
    const auto i = ml::preference_loss(ml::ipo{.tau = 0.5}, -1.0, -1.0, -1.0, -1.0);
    ml::preference_metrics<double> m;
    m.add(d);
    return d.loss > 0.0 && d.loss < 0.6931 && d.d_chosen < 0.0 && d.d_rejected == -d.d_chosen && i.loss == 1.0 && m.accuracy() == 1.0;
}
static_assert(at_compile_time());

using ld = long double;
ld log_sigmoid_ld(ld z) { return -std::log1p(std::exp(-z)); }

template <typename P, typename F>
void expect_gradient(const P& params, double pc, double pr, double rc, double rr, F&& loss_of)
{
    const auto p = ml::preference_loss(params, pc, pr, rc, rr);
    const double h = 1e-6;
    EXPECT_NEAR(p.d_chosen, (loss_of(pc + h, pr) - loss_of(pc - h, pr)) / (2 * h), 1e-7);
    EXPECT_NEAR(p.d_rejected, (loss_of(pc, pr + h) - loss_of(pc, pr - h)) / (2 * h), 1e-7);
}

TEST(MlPreferenceDpo, IsThePapersLossAndItsGradient)
{
    const ml::dpo c{.beta = 0.1};
    const double pc = -12.5, pr = -15.25, rc = -13.0, rr = -14.0;
    const auto p = ml::preference_loss(c, pc, pr, rc, rr);
    const ld z = 0.1L * ((ld(pc) - rc) - (ld(pr) - rr));
    EXPECT_NEAR(p.loss, double(-log_sigmoid_ld(z)), 1e-15);
    EXPECT_NEAR(p.chosen_reward, 0.1 * (pc - rc), 1e-15);
    EXPECT_NEAR(p.rejected_reward, 0.1 * (pr - rr), 1e-15);
    // The paper's gradient: -beta sigma(r_l - r_w) on chosen
    EXPECT_NEAR(p.d_chosen, double(-0.1L / (1 + std::exp(z))), 1e-15);
    expect_gradient(c, pc, pr, rc, rr, [&](double a, double b) { return ml::preference_loss(c, a, b, rc, rr).loss; });
}

TEST(MlPreferenceDpo, IdenticalResponsesGiveLog2AndBetaToZeroGivesNoGradient)
{
    const auto p = ml::preference_loss(ml::dpo{.beta = 0.3}, -7.0, -7.0, -6.0, -6.0);
    EXPECT_NEAR(p.loss, std::log(2.0), 1e-15);
    EXPECT_NEAR(p.d_chosen, -0.15, 1e-15);
    EXPECT_NEAR(p.d_rejected, 0.15, 1e-15);
    const auto q = ml::preference_loss(ml::dpo{.beta = 1e-12}, -1.0, -50.0, -40.0, -2.0);
    EXPECT_NEAR(q.loss, std::log(2.0), 1e-10);
    EXPECT_NEAR(q.d_chosen, 0.0, 1e-12);
    // A huge margin: no overflow, a loss of e^-z
    const auto r = ml::preference_loss(ml::dpo{.beta = 1.0}, 0.0f, -1000.0f, 0.0f, 0.0f);
    EXPECT_EQ(r.loss, 0.0f);
    EXPECT_TRUE(std::isfinite(ml::preference_loss(ml::dpo{.beta = 1.0}, -1000.0f, 0.0f, 0.0f, 0.0f).loss));
    EXPECT_EQ(ml::preference_loss(ml::dpo{.beta = 1.0}, -1000.0f, 0.0f, 0.0f, 0.0f).loss, 1000.0f);
}

TEST(MlPreferenceIpo, IsTheSquaredDistanceToHalfTheInverseTau)
{
    const ml::ipo c{.tau = 0.25};
    const double pc = -3.0, pr = -5.5, rc = -3.5, rr = -4.0;
    const double h = (pc - rc) - (pr - rr);
    const auto p = ml::preference_loss(c, pc, pr, rc, rr);
    EXPECT_NEAR(p.loss, (h - 2.0) * (h - 2.0), 1e-15);
    expect_gradient(c, pc, pr, rc, rr, [&](double a, double b) { return ml::preference_loss(c, a, b, rc, rr).loss; });
    EXPECT_EQ(ml::preference_loss(c, -1.0, -1.0, -2.0, -2.0).loss, 4.0);   // identical: 1 / (4 tau^2)
    EXPECT_EQ(ml::preference_loss(c, -1.0, -3.0, -2.0, -2.0).loss, 0.0);   // at the target margin 1 / (2 tau)
}

TEST(MlPreferenceSimpo, IsLengthNormalisedReferenceFreeWithAMargin)
{
    const ml::simpo c{.beta = 2.5, .gamma = 1.4};
    const double pc = -0.8, pr = -1.9;
    const auto p = ml::preference_loss(c, pc, pr);
    EXPECT_NEAR(p.loss, double(-log_sigmoid_ld(2.5L * pc - 2.5L * pr - 1.4L)), 1e-15);
    EXPECT_EQ(ml::preference_loss(c, pc, pr, 123.0, -456.0).loss, p.loss);   // the reference is ignored
    expect_gradient(c, pc, pr, 0.0, 0.0, [&](double a, double b) { return ml::preference_loss(c, a, b).loss; });
    // At equal log-probabilities the margin alone sets the loss
    EXPECT_NEAR(ml::preference_loss(c, -1.0, -1.0).loss, double(-log_sigmoid_ld(-1.4L)), 1e-15);
}

TEST(MlPreferenceOrpo, IsTheSftLossPlusTheOddsRatioTerm)
{
    const ml::orpo c{.lambda = 0.25};
    const double pc = -0.6, pr = -1.7;
    const auto p = ml::preference_loss(c, pc, pr);
    const ld pw = std::exp(ld(pc)), pl = std::exp(ld(pr));
    const ld log_or = std::log((pw / (1 - pw)) / (pl / (1 - pl)));
    EXPECT_NEAR(p.loss, double(-ld(pc) - 0.25L * log_sigmoid_ld(log_or)), 1e-14);
    EXPECT_NEAR(ml::log_odds_from_logprob(-0.6), double(std::log(pw / (1 - pw))), 1e-15);
    expect_gradient(c, pc, pr, 0.0, 0.0, [&](double a, double b) { return ml::preference_loss(c, a, b).loss; });
    // The paper's eqs. 8-10: delta(d) = 1 / (1 + odds ratio), h(d) = d log P_w / (1 - P_w) - d log P_l / (1 - P_l)
    const ld delta = 1 / (1 + std::exp(log_or));
    EXPECT_NEAR(p.d_chosen, double(-1 - 0.25L * delta / (1 - pw)), 1e-14);
    EXPECT_NEAR(p.d_rejected, double(0.25L * delta / (1 - pl)), 1e-14);
    // lambda = 0 is SFT on the chosen response alone
    const auto s = ml::preference_loss(ml::orpo{.lambda = 0.0}, pc, pr);
    EXPECT_EQ(s.loss, 0.6);
    EXPECT_EQ(s.d_chosen, -1.0);
    EXPECT_EQ(s.d_rejected, 0.0);
}

TEST(MlPreferenceMetrics, AddAndMergeAgree)
{
    stats::wyrand g(5);
    ml::preference_metrics<double> all, left, right;
    for (int i = 0; i < 50; ++i) {
        const double a = -10 * stats::uniform_real(g), b = -10 * stats::uniform_real(g);
        const auto p = ml::preference_loss(ml::dpo{}, a, b, -5.0, -5.0);
        all.add(p);
        (i < 20 ? left : right).add(p);
        EXPECT_EQ(p.chosen_reward > p.rejected_reward, a > b);
    }
    left.merge(right);
    EXPECT_EQ(left.pairs, 50u);
    EXPECT_EQ(left.correct, all.correct);
    EXPECT_NEAR(left.mean_loss(), all.mean_loss(), 1e-14);
    EXPECT_NEAR(left.mean_margin(), all.mean_margin(), 1e-14);
    EXPECT_EQ(ml::preference_metrics<float>{}.accuracy(), 0.0);
    EXPECT_EQ(ml::preference_metrics<float>{}.mean_loss(), 0.0f);
    ml::preference_metrics<double> tie;
    tie.add(ml::preference_loss(ml::dpo{}, -1.0, -1.0, -1.0, -1.0));
    EXPECT_EQ(tie.accuracy(), 0.0);   // a tie is not correct
}

TEST(MlPreferenceLossRows, IsThePerPairLossesOverN)
{
    const std::vector<double> pc{-3, -4, -5}, pr{-4, -3, -6}, rc{-3.5, -3.5, -5.5}, rr{-3.5, -3.5, -5.5};
    std::vector<double> dc(3), dr(3);
    ml::preference_metrics<double> m;
    const double loss = ml::preference_loss_rows<double>(backend, ml::dpo{.beta = 0.2}, pc, pr, rc, rr, dc, dr, m);
    double sum = 0;
    for (std::size_t i = 0; i < 3; ++i) {
        const auto p = ml::preference_loss(ml::dpo{.beta = 0.2}, pc[i], pr[i], rc[i], rr[i]);
        sum += p.loss;
        EXPECT_EQ(dc[i], p.d_chosen * (1.0 / 3));
        EXPECT_EQ(dr[i], p.d_rejected * (1.0 / 3));
    }
    EXPECT_EQ(loss, sum * (1.0 / 3));
    EXPECT_EQ(m.pairs, 3u);
    EXPECT_NEAR(m.accuracy(), 2.0 / 3, 1e-15);
    // Reference-free: empty references
    const double simpo_loss = ml::preference_loss_rows<double>(backend, ml::simpo{}, pc, pr, {}, {}, dc, dr, m);
    EXPECT_NEAR(simpo_loss, (ml::preference_loss(ml::simpo{}, -3.0, -4.0).loss + ml::preference_loss(ml::simpo{}, -4.0, -3.0).loss
                             + ml::preference_loss(ml::simpo{}, -5.0, -6.0).loss) / 3, 1e-15);
    EXPECT_EQ(ml::preference_loss_rows<double>(backend, ml::dpo{}, {}, {}, {}, {}, {}, {}, m), 0.0);
}

// Policy logits for 2 pairs x (chosen, rejected) x 3 positions over 4 tokens; the DPO loss of the batch from them
struct dpo_batch
{
    static constexpr std::size_t pairs = 2, len = 3, v = 4, rows = 2 * pairs;
    std::vector<std::uint32_t> targets{1, 2, 3, /**/ 0, 0, 1, /**/ 2, 2, 0, /**/ 3, 1, 1};   // rows: c0, c1, r0, r1
    std::vector<std::uint8_t> mask{0, 1, 1, /**/ 0, 1, 1, /**/ 0, 0, 1, /**/ 0, 1, 1};
    std::vector<double> ref{-2.0, -2.5, -1.5, -3.0};

    double loss(const std::vector<double>& logits, std::vector<double>* grad = nullptr) const
    {
        std::vector<double> lp(rows * len), seq(rows), dc(pairs), dr(pairs);
        ml::token_logprobs_rows<double>(backend, logits, targets, lp);
        ml::sequence_logprobs<double>(backend, lp, mask, seq, ml::logprob_reduction::sum);
        ml::preference_metrics<double> m;
        const std::span<const double> s(seq);
        const double l = ml::preference_loss_rows<double>(backend, ml::dpo{.beta = 0.5}, s.first(pairs), s.last(pairs),
                                                          std::span<const double>(ref).first(pairs), std::span<const double>(ref).last(pairs), dc, dr, m);
        if (grad) {
            const std::vector<double> dseq{dc[0], dc[1], dr[0], dr[1]};
            std::vector<double> dtok(rows * len);
            ml::sequence_logprobs_backward<double>(backend, mask, dseq, dtok, ml::logprob_reduction::sum);
            *grad = logits;
            ml::token_logprobs_rows_backward<double>(backend, *grad, targets, dtok);
        }
        return l;
    }
};

TEST(MlPreferenceLossRows, TheGradientThroughTheLogProbabilitiesIsCentralDifferences)
{
    stats::wyrand g(6);
    dpo_batch b;
    std::vector<double> logits(dpo_batch::rows * dpo_batch::len * dpo_batch::v);
    for (auto& x : logits)
        x = 2.0 * stats::uniform_real(g) - 1.0;
    std::vector<double> grad;
    b.loss(logits, &grad);
    for (std::size_t i = 0; i < logits.size(); ++i) {
        auto x = logits;
        const double h = 1e-6;
        x[i] += h;
        const double up = b.loss(x);
        x[i] -= 2 * h;
        EXPECT_NEAR(grad[i], (up - b.loss(x)) / (2 * h), 1e-8) << i;
        if (b.mask[i / dpo_batch::v] == 0)
            EXPECT_EQ(grad[i], 0.0);
    }
}

TEST(MlPreferenceDpo, ATinyRunRaisesThePolicysMargin)
{
    // The policy: one row of logits per position, shared by every sequence; the reference: the policy at the start
    constexpr std::size_t len = 4, v = 6, pairs = 8;
    stats::wyrand g(7);
    std::vector<double> theta(len * v);
    for (auto& x : theta)
        x = 0.1 * (2.0 * stats::uniform_real(g) - 1.0);
    // Chosen responses favour token 0 at every response position, rejected ones token 5; position 0 is the prompt
    std::vector<std::uint32_t> targets(2 * pairs * len);
    std::vector<std::uint8_t> mask(2 * pairs * len);
    for (std::size_t r = 0; r < 2 * pairs; ++r)
        for (std::size_t t = 0; t < len; ++t) {
            const bool chosen = r < pairs;
            targets[r * len + t] = static_cast<std::uint32_t>(stats::uniform_real(g) < 0.7 ? (chosen ? 0 : 5) : stats::uniform_index(g, v));
            mask[r * len + t] = t == 0 ? 0 : 1;
        }
    const auto sequence_logps = [&](const std::vector<double>& th, std::vector<double>& logits, std::vector<double>& seq) {
        logits.resize(2 * pairs * len * v);
        for (std::size_t r = 0; r < 2 * pairs; ++r)
            for (std::size_t i = 0; i < len * v; ++i)
                logits[r * len * v + i] = th[i];
        std::vector<double> lp(2 * pairs * len);
        ml::token_logprobs_rows<double>(backend, logits, targets, lp);
        seq.resize(2 * pairs);
        ml::sequence_logprobs<double>(backend, lp, mask, seq, ml::logprob_reduction::sum);
    };
    std::vector<double> logits, ref;
    sequence_logps(theta, logits, ref);
    double first_margin = 0, last_margin = 0, first_accuracy = 0, last_accuracy = 0;
    for (int step = 0; step < 60; ++step) {
        std::vector<double> seq, dc(pairs), dr(pairs), dtok(2 * pairs * len);
        sequence_logps(theta, logits, seq);
        ml::preference_metrics<double> m;
        const std::span<const double> s(seq), rs(ref);
        ml::preference_loss_rows<double>(backend, ml::dpo{.beta = 0.1}, s.first(pairs), s.last(pairs), rs.first(pairs), rs.last(pairs), dc, dr, m);
        if (step == 0) {
            first_margin = m.mean_margin();
            first_accuracy = m.accuracy();
            EXPECT_NEAR(m.mean_loss(), std::log(2.0), 1e-12);   // the policy is the reference
        }
        last_margin = m.mean_margin();
        last_accuracy = m.accuracy();
        std::vector<double> dseq(dc);
        dseq.insert(dseq.end(), dr.begin(), dr.end());
        ml::sequence_logprobs_backward<double>(backend, mask, dseq, dtok, ml::logprob_reduction::sum);
        ml::token_logprobs_rows_backward<double>(backend, logits, targets, dtok);
        for (std::size_t r = 0; r < 2 * pairs; ++r)
            for (std::size_t i = 0; i < len * v; ++i)
                theta[i] -= 2.0 * logits[r * len * v + i];
    }
    EXPECT_EQ(first_margin, 0.0);
    EXPECT_GT(last_margin, 0.5);
    EXPECT_GE(last_accuracy, first_accuracy);
    EXPECT_EQ(last_accuracy, 1.0);
}

} // namespace
