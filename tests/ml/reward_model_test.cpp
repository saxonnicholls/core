//
//  reward_model_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/ml/reward_model.hpp>: the Bradley-Terry loss against -log sigma(r_w - r_l - m) and its
//  gradient against central differences, through preference_loss_rows and its metrics; the value head as a dot
//  product plus bias, its backward against central differences, the rows as the per-row calls; the last response
//  positions; gather and scatter-add as adjoints; a tiny reward model whose pairwise accuracy rises to 1; compile-time
//  use and the noexcept claims.
//

#include <snicholls/core/ml/reward_model.hpp>

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

static_assert(std::is_trivially_copyable_v<ml::bradley_terry>);
static_assert(noexcept(ml::preference_loss(ml::bradley_terry{}, 1.0f, 0.0f)));
static_assert(noexcept(ml::value_head(std::array<double, 2>{}, 0.0, std::array<double, 2>{})));

constexpr bool at_compile_time()
{
    const std::array<double, 3> w{1.0, -2.0, 0.5}, h{2.0, 1.0, 4.0};
    const std::array<std::uint8_t, 6> mask{0, 1, 1, 1, 1, 0};
    std::array<std::uint32_t, 2> pos{};
    ml::last_response_positions(mask, pos);
    return ml::value_head(w, 0.25, h) == 2.25 && pos[0] == 2 && pos[1] == 4
           && ml::preference_loss(ml::bradley_terry{}, 0.0, 0.0).d_chosen == -0.5;
}
static_assert(at_compile_time());

TEST(MlBradleyTerry, IsMinusLogSigmoidOfTheMarginAndItsGradient)
{
    for (const double m : {0.0, 1.5}) {
        const ml::bradley_terry c{.margin = m};
        const double rw = 0.7, rl = -0.4;
        const auto p = ml::preference_loss(c, rw, rl);
        EXPECT_NEAR(p.loss, std::log1p(std::exp(-(rw - rl - m))), 1e-15);
        EXPECT_EQ(p.chosen_reward, rw);
        EXPECT_EQ(p.rejected_reward, rl);
        const double h = 1e-6;
        EXPECT_NEAR(p.d_chosen, (ml::preference_loss(c, rw + h, rl).loss - ml::preference_loss(c, rw - h, rl).loss) / (2 * h), 1e-9);
        EXPECT_NEAR(p.d_rejected, (ml::preference_loss(c, rw, rl + h).loss - ml::preference_loss(c, rw, rl - h).loss) / (2 * h), 1e-9);
    }
    EXPECT_NEAR(ml::preference_loss(ml::bradley_terry{}, 2.0, 2.0).loss, std::log(2.0), 1e-15);
    // The batch and the metrics serve reward models unchanged
    const std::vector<double> rw{1.0, -1.0}, rl{0.0, 0.0};
    std::vector<double> dw(2), dl(2);
    ml::preference_metrics<double> m;
    const double loss = ml::preference_loss_rows<double>(backend, ml::bradley_terry{}, rw, rl, {}, {}, dw, dl, m);
    EXPECT_NEAR(loss, (std::log1p(std::exp(-1.0)) + std::log1p(std::exp(1.0))) / 2, 1e-15);
    EXPECT_EQ(m.accuracy(), 0.5);
}

TEST(MlValueHead, IsADotProductPlusBiasAndItsBackwardIsCentralDifferences)
{
    stats::wyrand g(8);
    constexpr std::size_t d = 7, n = 3;
    std::vector<double> w(d), hs(n * d);
    for (auto& x : w)
        x = stats::uniform_real(g) - 0.5;
    for (auto& x : hs)
        x = stats::uniform_real(g) - 0.5;
    const double bias = 0.3;
    std::vector<double> values(n);
    ml::value_head_rows<double>(backend, w, bias, hs, values);
    for (std::size_t t = 0; t < n; ++t) {
        double s = bias;
        for (std::size_t j = 0; j < d; ++j)
            s += w[j] * hs[t * d + j];
        EXPECT_EQ(values[t], s);
        EXPECT_EQ(values[t], ml::value_head(w, bias, std::span<const double>(hs).subspan(t * d, d)));
    }
    const std::vector<double> dvalues{1.0, -2.0, 0.5};
    std::vector<double> dhs(n * d, 0.0), dw(d, 0.0);
    double db = 0;
    ml::value_head_rows_backward<double>(backend, w, hs, dvalues, dhs, dw, db);
    const auto objective = [&](const std::vector<double>& ww, const std::vector<double>& hh, double bb) {
        std::vector<double> v(n);
        ml::value_head_rows<double>(backend, ww, bb, hh, v);
        return dvalues[0] * v[0] + dvalues[1] * v[1] + dvalues[2] * v[2];
    };
    const double h = 1e-6;
    for (std::size_t j = 0; j < d; ++j) {
        auto a = w, b = w;
        a[j] += h;
        b[j] -= h;
        EXPECT_NEAR(dw[j], (objective(a, hs, bias) - objective(b, hs, bias)) / (2 * h), 1e-8);
    }
    for (std::size_t i = 0; i < n * d; ++i) {
        auto a = hs, b = hs;
        a[i] += h;
        b[i] -= h;
        EXPECT_NEAR(dhs[i], (objective(w, a, bias) - objective(w, b, bias)) / (2 * h), 1e-8);
    }
    EXPECT_NEAR(db, -0.5, 1e-15);
}

TEST(MlGatherRows, GatherAndScatterAddAreAdjoints)
{
    stats::wyrand g(9);
    constexpr std::size_t width = 3, rows = 5;
    std::vector<double> xs(rows * width), ys(2 * width);
    for (auto& x : xs)
        x = stats::uniform_real(g);
    for (auto& y : ys)
        y = stats::uniform_real(g);
    const std::vector<std::uint32_t> pick{4, 1};
    std::vector<double> gathered(2 * width), scattered(rows * width, 0.0);
    ml::gather_rows<double>(backend, xs, width, pick, gathered);
    EXPECT_EQ(gathered[0], xs[12]);
    EXPECT_EQ(gathered[3], xs[3]);
    ml::scatter_add_rows<double>(backend, ys, width, pick, scattered);
    double lhs = 0, rhs = 0;
    for (std::size_t i = 0; i < ys.size(); ++i)
        lhs += gathered[i] * ys[i];
    for (std::size_t i = 0; i < xs.size(); ++i)
        rhs += xs[i] * scattered[i];
    EXPECT_NEAR(lhs, rhs, 1e-15);
}

TEST(MlRewardModel, ATinyModelLearnsToRankChosenAboveRejected)
{
    // Features of the last response token: chosen responses lean along a hidden direction, rejected ones against it
    constexpr std::size_t d = 4, pairs = 16;
    stats::wyrand g(10);
    const std::array<double, d> direction{0.5, -0.5, 0.5, 0.5};
    std::vector<double> hc(pairs * d), hr(pairs * d);
    for (std::size_t i = 0; i < pairs * d; ++i) {
        hc[i] = stats::uniform_real(g) - 0.5 + 0.4 * direction[i % d];
        hr[i] = stats::uniform_real(g) - 0.5 - 0.4 * direction[i % d];
    }
    std::vector<double> w(d, 0.0);
    double bias = 0.0, first = -1, last = -1;
    for (int step = 0; step < 100; ++step) {
        std::vector<double> rc(pairs), rr(pairs), dc(pairs), dr(pairs), dw(d, 0.0);
        ml::value_head_rows<double>(backend, w, bias, hc, rc);
        ml::value_head_rows<double>(backend, w, bias, hr, rr);
        ml::preference_metrics<double> m;
        ml::preference_loss_rows<double>(backend, ml::bradley_terry{}, rc, rr, {}, {}, dc, dr, m);
        if (first < 0)
            first = m.accuracy();
        last = m.accuracy();
        double db = 0;
        ml::value_head_rows_backward<double>(backend, w, hc, dc, {}, dw, db);
        ml::value_head_rows_backward<double>(backend, w, hr, dr, {}, dw, db);
        EXPECT_NEAR(db, 0.0, 1e-15);   // a shared bias cancels in a pairwise loss
        for (std::size_t j = 0; j < d; ++j)
            w[j] -= 1.0 * dw[j];
        bias -= db;
    }
    EXPECT_EQ(first, 0.0);   // all rewards equal at the start
    EXPECT_EQ(last, 1.0);
}

} // namespace
