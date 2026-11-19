//
//  distill_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/ml/distill.hpp>: the forward and reverse KL at a temperature against a direct computation
//  from the tempered softmaxes; zero for identical rows; Hinton's gradient (q - p) / T; both gradients against central
//  differences; finite for logits of order 1e4 and for a teacher with -infinity logits; in place equal to out of place;
//  the rows as the per-row calls with masked rows zero; teacher_topk's order and ties; the top-k loss equal to the full
//  one when the teacher has no mass outside its top-k, and its gradient against central differences; compile-time use
//  and the noexcept claims.
//

#include <snicholls/core/ml/distill.hpp>

#include <snicholls/core/linalg/reference.hpp>
#include <snicholls/core/ml/softmax.hpp>
#include <snicholls/core/stats/random.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <type_traits>
#include <vector>

namespace {

namespace ml = snicholls::ml;
namespace stats = snicholls::stats;
constexpr auto& backend = snicholls::linalg::reference;

static_assert(std::is_trivially_copyable_v<ml::forward_kl> && std::is_trivially_copyable_v<ml::reverse_kl>);
static_assert(noexcept(ml::distill_loss(ml::forward_kl{}, std::array<float, 2>{}, std::array<float, 2>{}, std::declval<std::array<float, 2>&>())));

constexpr bool at_compile_time()
{
    const std::array<double, 3> s{1.0, 2.0, 3.0};
    std::array<double, 3> g{};
    const double same = ml::distill_loss(ml::forward_kl{.temperature = 2.0}, s, s, g);
    std::array<std::uint32_t, 2> ids{};
    std::array<double, 2> vals{};
    ml::teacher_topk(std::array<double, 4>{0.5, 3.0, 3.0, -1.0}, ids, vals);
    return same == 0.0 && g[0] == 0.0 && ids[0] == 1 && ids[1] == 2 && vals[0] == 3.0;
}
static_assert(at_compile_time());

std::vector<double> softmax_at(const std::vector<double>& x, double t)
{
    std::vector<double> y(x.size()), p(x.size());
    for (std::size_t i = 0; i < x.size(); ++i)
        y[i] = x[i] / t;
    ml::softmax(y, p);
    return p;
}

double kl(const std::vector<double>& p, const std::vector<double>& q)
{
    double s = 0;
    for (std::size_t i = 0; i < p.size(); ++i)
        if (p[i] > 0)
            s += p[i] * std::log(p[i] / q[i]);
    return s;
}

std::vector<double> random_row(stats::wyrand& g, std::size_t n, double scale)
{
    std::vector<double> v(n);
    for (auto& x : v)
        x = scale * (2.0 * stats::uniform_real(g) - 1.0);
    return v;
}

TEST(MlDistill, ForwardAndReverseKlAreTheDirectComputation)
{
    stats::wyrand g(11);
    for (const double t : {1.0, 2.5, 0.5}) {
        const auto s = random_row(g, 13, 3.0), te = random_row(g, 13, 3.0);
        const auto p = softmax_at(te, t), q = softmax_at(s, t);
        std::vector<double> grad(13);
        EXPECT_NEAR(ml::distill_loss(ml::forward_kl{.temperature = t}, s, te, grad), kl(p, q), 1e-14);
        for (std::size_t i = 0; i < 13; ++i)
            EXPECT_NEAR(grad[i], (q[i] - p[i]) / t, 1e-15);   // Hinton's eq. 2
        EXPECT_NEAR(ml::distill_loss(ml::reverse_kl{.temperature = t}, s, te, grad), kl(q, p), 1e-14);
    }
}

TEST(MlDistill, IdenticalRowsGiveZeroLossAndGradient)
{
    const std::vector<double> s{0.3, -1.0, 2.0, 0.0};
    std::vector<double> g(4);
    EXPECT_NEAR(ml::distill_loss(ml::forward_kl{}, s, s, g), 0.0, 1e-16);
    for (const double x : g)
        EXPECT_NEAR(x, 0.0, 1e-16);
    EXPECT_NEAR(ml::distill_loss(ml::reverse_kl{.temperature = 3.0}, s, s, g), 0.0, 1e-16);
    for (const double x : g)
        EXPECT_NEAR(x, 0.0, 1e-16);
}

TEST(MlDistill, GradientsAreCentralDifferences)
{
    stats::wyrand g(12);
    for (const double t : {1.0, 2.0}) {
        auto s = random_row(g, 9, 2.0);
        const auto te = random_row(g, 9, 2.0);
        std::vector<double> gf(9), gr(9), scratch(9);
        ml::distill_loss(ml::forward_kl{.temperature = t}, s, te, gf, 0.75);
        ml::distill_loss(ml::reverse_kl{.temperature = t}, s, te, gr, 0.75);
        for (std::size_t i = 0; i < 9; ++i) {
            const double h = 1e-6, x = s[i];
            s[i] = x + h;
            const double fu = ml::distill_loss(ml::forward_kl{.temperature = t}, s, te, scratch);
            const double ru = ml::distill_loss(ml::reverse_kl{.temperature = t}, s, te, scratch);
            s[i] = x - h;
            const double fd = ml::distill_loss(ml::forward_kl{.temperature = t}, s, te, scratch);
            const double rd = ml::distill_loss(ml::reverse_kl{.temperature = t}, s, te, scratch);
            s[i] = x;
            EXPECT_NEAR(gf[i], 0.75 * (fu - fd) / (2 * h), 1e-9) << i;
            EXPECT_NEAR(gr[i], 0.75 * (ru - rd) / (2 * h), 1e-9) << i;
        }
    }
}

TEST(MlDistill, StaysFiniteForLargeLogitsAndImpossibleTeacherTokens)
{
    const std::vector<float> s{1e4f, 0.0f, -1e4f}, te{-1e4f, 0.0f, 1e4f};
    std::vector<float> g(3);
    const float f = ml::distill_loss(ml::forward_kl{}, s, te, g);
    EXPECT_TRUE(std::isfinite(f));
    EXPECT_EQ(f, 2e4f);
    EXPECT_TRUE(std::isfinite(ml::distill_loss(ml::reverse_kl{}, s, te, g)));
    const float inf = std::numeric_limits<float>::infinity();
    const std::vector<float> masked_teacher{0.0f, -inf, 1.0f};
    const float k = ml::distill_loss(ml::forward_kl{}, std::vector<float>{0.5f, 0.5f, 0.5f}, masked_teacher, g);
    EXPECT_TRUE(std::isfinite(k));
    for (const float x : g)
        EXPECT_TRUE(std::isfinite(x));
}

TEST(MlDistillRows, AreThePerRowCallsWithMaskedRowsZeroAndMayBeInPlace)
{
    stats::wyrand g(13);
    constexpr std::size_t n = 3, v = 5;
    const auto s = random_row(g, n * v, 2.0), te = random_row(g, n * v, 2.0);
    const std::vector<std::uint8_t> mask{1, 0, 1};
    std::vector<double> losses(n), grads(n * v);
    ml::distill_rows<double>(backend, ml::reverse_kl{.temperature = 2.0}, s, te, mask, losses, grads, 0.5);
    for (std::size_t r = 0; r < n; ++r) {
        std::vector<double> row(v);
        const double l = ml::distill_loss(ml::reverse_kl{.temperature = 2.0}, std::span<const double>(s).subspan(r * v, v),
                                          std::span<const double>(te).subspan(r * v, v), row, 0.5);
        EXPECT_EQ(losses[r], mask[r] ? l : 0.0);
        for (std::size_t i = 0; i < v; ++i)
            EXPECT_EQ(grads[r * v + i], mask[r] ? row[i] : 0.0);
    }
    auto in_place = s;
    ml::distill_rows<double>(backend, ml::reverse_kl{.temperature = 2.0}, in_place, te, mask, losses, in_place, 0.5);
    EXPECT_EQ(in_place, grads);
}

TEST(MlDistillTopk, TopkIsLargestFirstWithTiesToTheLowerIndex)
{
    const std::vector<double> logits{1.0, 5.0, 3.0, 5.0, -2.0, 3.0};
    std::vector<std::uint32_t> ids(4);
    std::vector<double> vals(4);
    ml::teacher_topk(logits, ids, vals);
    EXPECT_EQ(ids, (std::vector<std::uint32_t>{1, 3, 2, 5}));
    EXPECT_EQ(vals, (std::vector<double>{5, 5, 3, 3}));
}

TEST(MlDistillTopk, EqualsTheFullLossWhenTheTeacherHasNoTailAndItsGradientIsCentralDifferences)
{
    stats::wyrand g(14);
    constexpr std::size_t v = 8, k = 3;
    auto s = random_row(g, v, 2.0);
    std::vector<double> te(v, -std::numeric_limits<double>::infinity());
    te[6] = 1.0;
    te[1] = 0.25;
    te[3] = -0.5;
    std::vector<std::uint32_t> ids(k);
    std::vector<double> vals(k), gk(v), gf(v);
    ml::teacher_topk(te, ids, vals);
    const double topk = ml::distill_loss_topk(ml::forward_kl{.temperature = 1.5}, s, ids, vals, gk);
    const double full = ml::distill_loss(ml::forward_kl{.temperature = 1.5}, s, te, gf);
    EXPECT_NEAR(topk, full, 1e-15);
    for (std::size_t i = 0; i < v; ++i)
        EXPECT_NEAR(gk[i], gf[i], 1e-16);
    // With a tail, the top-k loss is of the renormalised teacher; its gradient is still exact for that loss
    const auto te2 = random_row(g, v, 2.0);
    ml::teacher_topk(te2, ids, vals);
    ml::distill_loss_topk(ml::forward_kl{.temperature = 1.5}, s, ids, vals, gk);
    std::vector<double> scratch(v);
    for (std::size_t i = 0; i < v; ++i) {
        const double h = 1e-6, x = s[i];
        s[i] = x + h;
        const double up = ml::distill_loss_topk(ml::forward_kl{.temperature = 1.5}, s, ids, vals, scratch);
        s[i] = x - h;
        const double down = ml::distill_loss_topk(ml::forward_kl{.temperature = 1.5}, s, ids, vals, scratch);
        s[i] = x;
        EXPECT_NEAR(gk[i], (up - down) / (2 * h), 1e-9) << i;
    }
    // The rows: per-row calls, masked rows zero
    std::vector<double> losses(2), grads(2 * v);
    std::vector<std::uint32_t> ids2(ids);
    ids2.insert(ids2.end(), ids.begin(), ids.end());
    std::vector<double> vals2(vals);
    vals2.insert(vals2.end(), vals.begin(), vals.end());
    std::vector<double> s2(s);
    s2.insert(s2.end(), s.begin(), s.end());
    ml::distill_topk_rows<double>(backend, ml::forward_kl{.temperature = 1.5}, s2, ids2, vals2, std::vector<std::uint8_t>{0, 1}, losses, grads, 1.0);
    EXPECT_EQ(losses[0], 0.0);
    EXPECT_EQ(losses[1], ml::distill_loss_topk(ml::forward_kl{.temperature = 1.5}, s, ids, vals, scratch));
    EXPECT_EQ(grads[0], 0.0);
    EXPECT_EQ(grads[v], gk[0]);
}

} // namespace
