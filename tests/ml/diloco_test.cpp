//
//  diloco_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/ml/diloco.hpp>: ordered_sum equal to a loop by hand to the bit, sensitive to the order of
//  its parts, a copy for one part, in place into the first part; ordered_mean; pseudo_gradient; the outer step with
//  Nesterov and plain momentum on a hand-computed case, as plain averaging with one worker (bit-exact under
//  Sterbenz's lemma, and the case where it is not); DiLoCo with four workers on a least-squares problem reaching the
//  loss of one worker trained for the same steps, and four workers with identical data giving one worker's bits;
//  compile-time use and the noexcept claims. bfloat16 pseudo-gradients: compress_bf16's residual exact, error feedback
//  carrying the cumulative sum to within the last round's residual, NaN, infinities, overflow, -0 and subnormals;
//  ordered_sum_bf16 as ordered_sum over the decoded parts, ordered_mean_bf16 as its stated formula.
//

#include <snicholls/core/ml/diloco.hpp>

#include <snicholls/core/stats/random.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <span>
#include <type_traits>
#include <vector>

namespace {

namespace ml = snicholls::ml;
namespace stats = snicholls::stats;

static_assert(std::is_trivially_copyable_v<ml::diloco_config>);
static_assert(ml::diloco_config{}.outer_lr == 0.7 && ml::diloco_config{}.momentum == 0.9 && ml::diloco_config{}.nesterov);

// noexcept for the standard floating-point types
static_assert(noexcept(ml::ordered_sum<float>({}, {})));
static_assert(noexcept(ml::ordered_mean<double>({}, {})));
static_assert(noexcept(ml::pseudo_gradient<float>({}, {}, {})));
static_assert(noexcept(ml::diloco_outer_step<double>({}, {}, {}, {})));

constexpr bool at_compile_time()
{
    const std::array<double, 2> a{1.0, 2.0}, b{0.5, -1.0}, c{0.25, 4.0};
    const std::array<std::span<const double>, 3> parts{a, b, c};
    std::array<double, 2> sum{}, mean{};
    ml::ordered_sum<double>(parts, sum);
    ml::ordered_mean<double>(std::span<const std::span<const double>>(parts).first(2), mean);
    std::array<double, 2> delta{};
    ml::pseudo_gradient<double>(a, b, delta);
    std::array<double, 2> global{1.0, 1.0}, momentum{};
    ml::diloco_outer_step<double>(global, delta, momentum, {.outer_lr = 1.0, .momentum = 0.0, .nesterov = false});
    return sum[0] == 1.75 && sum[1] == 5.0 && mean[0] == 0.75 && mean[1] == 0.5 && delta[0] == 0.5 && delta[1] == 3.0
           && global[0] == 0.5 && global[1] == -2.0;
}
static_assert(at_compile_time());

static_assert(noexcept(ml::compress_bf16({}, {}, {})));
static_assert(noexcept(ml::ordered_sum_bf16({}, {})));
static_assert(noexcept(ml::ordered_mean_bf16({}, {})));

constexpr bool bf16_at_compile_time()
{
    // 1 + 2^-10 is not a bfloat16: it rounds to 1, and the residual keeps the 2^-10
    const std::array<float, 3> delta{1.0f + 0x1.0p-10f, -3.0f, 0.0f};
    std::array<float, 3> residual{};
    std::array<std::uint16_t, 3> bits{};
    ml::compress_bf16(delta, residual, bits);
    if (bits[0] != 0x3f80 || residual[0] != 0x1.0p-10f || bits[1] != 0xc040 || residual[1] != 0.0f || bits[2] != 0)
        return false;
    // The next round: a zero delta sends the residual
    const std::array<float, 3> zero{};
    ml::compress_bf16(zero, residual, bits);
    if (bits[0] != 0x3a80 || residual[0] != 0.0f)   // 2^-10
        return false;
    const std::array<std::uint16_t, 2> a{0x3f80, 0x4000}, b{0x3f80, 0x4040};   // (1, 2) and (1, 3)
    const std::array<std::span<const std::uint16_t>, 2> parts{a, b};
    std::array<float, 2> sum{}, mean{};
    ml::ordered_sum_bf16(parts, sum);
    ml::ordered_mean_bf16(parts, mean);
    return sum[0] == 2.0f && sum[1] == 5.0f && mean[0] == 1.0f && mean[1] == 2.5f;
}
static_assert(bf16_at_compile_time());

std::vector<float> random_floats(stats::wyrand& g, std::size_t n, double scale)
{
    std::vector<float> v(n);
    for (auto& x : v)
        x = static_cast<float>(scale * (stats::uniform_real(g) - 0.5));
    return v;
}

bool same_bits(std::span<const float> a, std::span<const float> b)
{
    if (a.size() != b.size())
        return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (std::bit_cast<std::uint32_t>(a[i]) != std::bit_cast<std::uint32_t>(b[i]))
            return false;
    return true;
}

TEST(MlOrderedSum, EqualsALoopByHandToTheBit)
{
    stats::wyrand g(20261119);
    constexpr std::size_t workers = 5, n = 1000;
    std::vector<std::vector<float>> data;
    std::vector<std::span<const float>> parts;
    for (std::size_t k = 0; k < workers; ++k)
        data.push_back(random_floats(g, n, std::ldexp(1.0, static_cast<int>(k) * 7)));   // magnitudes 1 to 2^28
    for (const auto& d : data)
        parts.emplace_back(d);
    std::vector<float> out(n), by_hand(n);
    ml::ordered_sum<float>(parts, out);
    for (std::size_t i = 0; i < n; ++i) {
        float s = data[0][i];
        for (std::size_t k = 1; k < workers; ++k)
            s = s + data[k][i];
        by_hand[i] = s;
    }
    EXPECT_TRUE(same_bits(out, by_hand));
}

TEST(MlOrderedSum, DependsOnTheOrderOfItsParts)
{
    // (1e8 + 1) rounds to 1e8 in float (its spacing there is 8), so the 1 is lost unless the large parts cancel first
    const std::array<float, 1> big{1e8f}, one{1.0f}, minus_big{-1e8f};
    std::array<float, 1> out{};
    const std::array<std::span<const float>, 3> one_second{big, one, minus_big}, one_last{big, minus_big, one};
    ml::ordered_sum<float>(one_second, out);
    EXPECT_EQ(out[0], 0.0f);
    ml::ordered_sum<float>(one_last, out);
    EXPECT_EQ(out[0], 1.0f);
}

TEST(MlOrderedSum, OnePartIsACopy)
{
    stats::wyrand g(3);
    const auto a = random_floats(g, 257, 1e3);
    const std::array<std::span<const float>, 1> parts{std::span<const float>(a)};
    std::vector<float> out(a.size(), -1.0f);
    ml::ordered_sum<float>(parts, out);
    EXPECT_TRUE(same_bits(out, a));
}

TEST(MlOrderedSum, AccumulatesInPlaceIntoTheFirstPart)
{
    stats::wyrand g(4);
    auto a = random_floats(g, 100, 1.0);
    const auto b = random_floats(g, 100, 1.0), c = random_floats(g, 100, 1.0);
    std::vector<float> expected(a.size());
    const std::array<std::span<const float>, 3> separate{std::span<const float>(a), b, c};
    ml::ordered_sum<float>(separate, expected);
    const std::array<std::span<const float>, 3> in_place{std::span<const float>(a), b, c};
    ml::ordered_sum<float>(in_place, a);
    EXPECT_TRUE(same_bits(a, expected));
}

TEST(MlOrderedSum, EmptyRowsGiveNothing)
{
    const std::array<std::span<const float>, 2> parts{};
    ml::ordered_sum<float>(parts, std::span<float>{});
    SUCCEED();
}

TEST(MlOrderedMean, IsTheSumDividedByTheCount)
{
    stats::wyrand g(5);
    const auto a = random_floats(g, 64, 1.0), b = random_floats(g, 64, 1.0), c = random_floats(g, 64, 1.0);
    const std::array<std::span<const float>, 3> parts{std::span<const float>(a), b, c};
    std::vector<float> sum(64), mean(64);
    ml::ordered_sum<float>(parts, sum);
    ml::ordered_mean<float>(parts, mean);
    for (std::size_t i = 0; i < sum.size(); ++i)
        EXPECT_EQ(std::bit_cast<std::uint32_t>(mean[i]), std::bit_cast<std::uint32_t>(sum[i] / 3.0f)) << i;
}

TEST(MlOrderedMean, FourIdenticalPartsGiveThePartAndThreeNeedNot)
{
    // ((d + d) + d) + d is exactly 4d for every finite float d; (d + d) + d is not always 3d
    const float d = 0x1.55555cp0f;   // 3d needs more bits than a float has; it rounds, and the mean is d - 2^-22
    const std::array<float, 1> x{d};
    std::array<float, 1> out{};
    const std::array<std::span<const float>, 4> four{std::span<const float>(x), x, x, x};
    ml::ordered_mean<float>(four, out);
    EXPECT_EQ(out[0], d);
    ml::ordered_mean<float>(std::span<const std::span<const float>>(four).first(3), out);
    EXPECT_EQ(out[0], 0x1.55555ap0f);
}

TEST(MlPseudoGradient, IsGlobalMinusLocal)
{
    const std::array<float, 3> global{1.0f, -2.0f, 0.5f}, local{0.75f, -1.0f, 0.5f};
    std::array<float, 3> out{};
    ml::pseudo_gradient<float>(global, local, out);
    EXPECT_EQ(out, (std::array<float, 3>{0.25f, -1.0f, 0.0f}));
    // In place over the local weights
    std::array<float, 3> in_place = local;
    ml::pseudo_gradient<float>(global, in_place, in_place);
    EXPECT_EQ(in_place, out);
}

TEST(MlDilocoOuterStep, NesterovAndPlainMomentumOnAHandComputedCase)
{
    // mu = 0.5, lr = 0.5, g = 0.5, momentum 0.25: momentum becomes 0.125 + 0.5 = 0.625 (every value exact in binary)
    const std::array<double, 1> g{0.5};
    std::array<double, 1> global{1.0}, momentum{0.25};
    ml::diloco_outer_step<double>(global, g, momentum, {.outer_lr = 0.5, .momentum = 0.5, .nesterov = true});
    EXPECT_EQ(momentum[0], 0.625);
    EXPECT_EQ(global[0], 1.0 - 0.5 * (0.5 + 0.5 * 0.625));   // step 0.8125: 0.59375

    global = {1.0};
    momentum = {0.25};
    ml::diloco_outer_step<double>(global, g, momentum, {.outer_lr = 0.5, .momentum = 0.5, .nesterov = false});
    EXPECT_EQ(momentum[0], 0.625);
    EXPECT_EQ(global[0], 1.0 - 0.5 * 0.625);   // step 0.625: 0.6875
}

TEST(MlDilocoOuterStep, ThePapersSettingsOnTwoRounds)
{
    // Two rounds against the formula in long double, rounded to float at each statement as T's arithmetic does
    const std::array<float, 2> g1{0.1f, -0.3f}, g2{0.2f, 0.05f};
    std::array<float, 2> global{1.0f, 2.0f}, momentum{};
    std::array<float, 2> expected_global = global, expected_momentum{};
    const ml::diloco_config c{};
    for (const auto& g : {g1, g2}) {
        ml::diloco_outer_step<float>(global, g, momentum, c);
        for (std::size_t i = 0; i < 2; ++i) {
            const float mu = 0.9f, lr = 0.7f;
            expected_momentum[i] = static_cast<float>(mu * expected_momentum[i]) + g[i];
            const float step = g[i] + static_cast<float>(mu * expected_momentum[i]);
            expected_global[i] = expected_global[i] - static_cast<float>(lr * step);
        }
    }
    EXPECT_TRUE(same_bits(global, expected_global));
    EXPECT_TRUE(same_bits(momentum, expected_momentum));
}

TEST(MlDilocoOuterStep, OneWorkerAveragingGivesTheLocalWeightsUnderSterbenz)
{
    // outer_lr 1, momentum 0, no Nesterov: global - (global - local), which is local whenever global - local is exact,
    // as Sterbenz's lemma makes it when local / 2 <= global <= 2 local
    stats::wyrand g(6);
    std::vector<float> global(4096), local(4096);
    for (std::size_t i = 0; i < global.size(); ++i) {
        global[i] = static_cast<float>(1.0 + stats::uniform_real(g));                         // [1, 2)
        local[i] = global[i] * static_cast<float>(0.75 + 0.5 * stats::uniform_real(g));       // within a factor of 2
    }
    std::vector<float> delta(global.size()), momentum(global.size());
    ml::pseudo_gradient<float>(global, local, delta);
    const std::array<std::span<const float>, 1> one{std::span<const float>(delta)};
    ml::ordered_mean<float>(one, delta);
    ml::diloco_outer_step<float>(global, delta, momentum, {.outer_lr = 1.0, .momentum = 0.0, .nesterov = false});
    EXPECT_TRUE(same_bits(global, local));
}

TEST(MlDilocoOuterStep, OneWorkerAveragingIsOneRoundingAwayWhenTheDifferenceIsNotExact)
{
    // 1 - 1e-8 rounds to 1 in float, so the step lands on 0, not 1e-8
    std::array<float, 1> global{1.0f}, momentum{};
    const std::array<float, 1> local{1e-8f};
    std::array<float, 1> delta{};
    ml::pseudo_gradient<float>(global, local, delta);
    ml::diloco_outer_step<float>(global, delta, momentum, {.outer_lr = 1.0, .momentum = 0.0, .nesterov = false});
    EXPECT_EQ(global[0], 0.0f);
    EXPECT_NE(global[0], local[0]);
}

// Least squares: y = w* . x + noise, x uniform in [-1, 1]^D. Loss 0.5 (w . x - y)^2 averaged; its floor is
// sigma^2 / 2 at w = w*.
constexpr std::size_t dim = 8, batch = 8;
constexpr double sigma = 0.1;

struct problem {
    std::array<double, dim> w_star{};
    std::vector<std::array<double, dim + 1>> test;   // x then y
    problem()
    {
        stats::wyrand g(11);
        for (auto& w : w_star)
            w = 4.0 * stats::uniform_real(g) - 2.0;
        test.resize(2048);
        for (auto& row : test)
            row = sample(g);
    }
    std::array<double, dim + 1> sample(stats::wyrand& g) const
    {
        std::array<double, dim + 1> row{};
        double y = 0;
        for (std::size_t j = 0; j < dim; ++j) {
            row[j] = 2.0 * stats::uniform_real(g) - 1.0;
            y += w_star[j] * row[j];
        }
        // A noise of variance sigma^2 from the sum of 12 uniforms (Irwin-Hall), exact enough for a loss floor
        double noise = -6.0;
        for (int k = 0; k < 12; ++k)
            noise += stats::uniform_real(g);
        row[dim] = y + sigma * noise;
        return row;
    }
    double loss(std::span<const double> w) const
    {
        double total = 0;
        for (const auto& row : test) {
            double e = -row[dim];
            for (std::size_t j = 0; j < dim; ++j)
                e += w[j] * row[j];
            total += 0.5 * e * e;
        }
        return total / static_cast<double>(test.size());
    }
    // One minibatch SGD step from the worker's own stream
    void sgd_step(std::span<double> w, stats::wyrand& g, double lr) const
    {
        std::array<double, dim> grad{};
        for (std::size_t b = 0; b < batch; ++b) {
            const auto row = sample(g);
            double e = -row[dim];
            for (std::size_t j = 0; j < dim; ++j)
                e += w[j] * row[j];
            for (std::size_t j = 0; j < dim; ++j)
                grad[j] += e * row[j] / static_cast<double>(batch);
        }
        for (std::size_t j = 0; j < dim; ++j)
            w[j] -= lr * grad[j];
    }
};

constexpr double inner_lr = 0.05;

// DiLoCo: `workers` workers, `rounds` rounds of `h` local SGD steps each; worker k draws from stream(seed, stream_of(k))
template <typename StreamOf>
std::array<double, dim> diloco(const problem& p, std::size_t workers, std::size_t rounds, std::size_t h, StreamOf stream_of)
{
    std::array<double, dim> global{}, momentum{}, mean{};
    std::vector<stats::wyrand> streams;
    for (std::size_t k = 0; k < workers; ++k)
        streams.push_back(stats::wyrand::stream(2026, stream_of(k)));
    std::vector<std::array<double, dim>> deltas(workers);
    std::vector<std::span<const double>> parts(deltas.begin(), deltas.end());
    for (std::size_t r = 0; r < rounds; ++r) {
        for (std::size_t k = 0; k < workers; ++k) {
            std::array<double, dim> local = global;
            for (std::size_t t = 0; t < h; ++t)
                p.sgd_step(local, streams[k], inner_lr);
            ml::pseudo_gradient<double>(global, local, deltas[k]);
        }
        ml::ordered_mean<double>(parts, mean);
        ml::diloco_outer_step<double>(global, mean, momentum, ml::diloco_config{});
    }
    return global;
}

TEST(MlDiloco, FourWorkersReachTheLossOfOneWorkerTrainedAsLong)
{
    const problem p;
    constexpr std::size_t rounds = 20, h = 50;
    const auto w = diloco(p, 4, rounds, h, [](std::size_t k) { return k; });

    std::array<double, dim> single{};
    auto g = stats::wyrand::stream(2026, 0);
    for (std::size_t t = 0; t < rounds * h; ++t)
        p.sgd_step(single, g, inner_lr);

    const double floor = sigma * sigma / 2, start = p.loss(std::array<double, dim>{});
    const double diloco_loss = p.loss(w), single_loss = p.loss(single);
    EXPECT_GT(start, 100 * floor);
    EXPECT_LT(single_loss, 1.2 * floor);
    EXPECT_LT(diloco_loss, 1.2 * floor);
    EXPECT_LT(std::fabs(diloco_loss - single_loss), 0.1 * floor);
}

TEST(MlDiloco, FourWorkersWithIdenticalDataAreOneWorkerToTheBit)
{
    const problem p;
    const auto four = diloco(p, 4, 5, 20, [](std::size_t) { return std::size_t{0}; });
    const auto one = diloco(p, 1, 5, 20, [](std::size_t) { return std::size_t{0}; });
    for (std::size_t j = 0; j < dim; ++j)
        EXPECT_EQ(std::bit_cast<std::uint64_t>(four[j]), std::bit_cast<std::uint64_t>(one[j])) << j;
}

// ------------------------------------------------------------------------------------- bfloat16 pseudo-gradients

float decoded(std::uint16_t b)
{
    return ml::to_float(ml::bfloat16::from_bits(b));
}

// Random floats over the whole finite range, subnormals included: random bits, NaN and infinity drawn again
float any_finite(stats::wyrand& g)
{
    for (;;) {
        const float f = std::bit_cast<float>(static_cast<std::uint32_t>(g()));
        if (std::isfinite(f))
            return f;
    }
}

TEST(MlDilocoBf16, TheResidualIsExactSoTheValueIsTheBfloat16PlusTheResidual)
{
    stats::wyrand g(11);
    std::size_t subnormals = 0;
    for (int k = 0; k < 200000; ++k) {
        std::array<float, 1> delta{k % 4 == 0 ? std::bit_cast<float>(static_cast<std::uint32_t>(g()) & 0x807fffffu) : any_finite(g)};
        std::array<float, 1> residual{};
        std::array<std::uint16_t, 1> bits{};
        if (std::fpclassify(delta[0]) == FP_SUBNORMAL)
            ++subnormals;
        ml::compress_bf16(delta, residual, bits);
        const float back = decoded(bits[0]);
        if (!std::isfinite(back)) {   // overflow beyond bfloat16's largest finite
            EXPECT_EQ(residual[0], 0.0f);
            EXPECT_GT(std::fabs(delta[0]), 3.3e38f);
            continue;
        }
        ASSERT_EQ(static_cast<double>(back) + static_cast<double>(residual[0]), static_cast<double>(delta[0])) << delta[0];
        ASSERT_EQ(bits[0], ml::to_minifloat<ml::bfloat16>(static_cast<double>(delta[0])).bits) << "round to nearest even";
    }
    EXPECT_GT(subnormals, 10000u);
}

TEST(MlDilocoBf16, NonFiniteValuesTravelOnAndCarryNoResidual)
{
    const float inf = std::numeric_limits<float>::infinity(), nan = std::numeric_limits<float>::quiet_NaN();
    const std::array<float, 6> delta{nan, inf, -inf, FLT_MAX, -0.0f, -FLT_MAX};
    std::array<float, 6> residual{};
    std::array<std::uint16_t, 6> bits{};
    ml::compress_bf16(delta, residual, bits);
    EXPECT_TRUE(std::isnan(decoded(bits[0])));
    EXPECT_EQ(bits[1], 0x7f80);
    EXPECT_EQ(bits[2], 0xff80);
    EXPECT_EQ(bits[3], 0x7f80) << "FLT_MAX rounds to infinity";
    EXPECT_EQ(bits[4], 0x0000) << "-0 + a zero residual is +0 (IEEE 754), so a zero is sent as +0";
    EXPECT_EQ(bits[5], 0xff80);
    for (const float r : residual)
        EXPECT_EQ(std::bit_cast<std::uint32_t>(r), 0u);
    // A residual is never NaN: a finite delta after a NaN round sends itself
    const std::array<float, 6> next{0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
    ml::compress_bf16(next, residual, bits);
    for (const auto b : bits)
        EXPECT_EQ(b, 0x3f00);
}

TEST(MlDilocoBf16, ErrorFeedbackMakesTheCumulativeSumExactOnAGridOfExactAdditions)
{
    // Deltas are multiples of 2^-20 below 2^3 in magnitude, so delta + residual is exact in float: what was sent plus
    // the last residual is the sum of the deltas, exactly
    constexpr std::size_t n = 257, rounds = 500;
    stats::wyrand g(3);
    std::vector<float> residual(n), delta(n);
    std::vector<std::uint16_t> bits(n);
    std::vector<double> sent(n), truth(n);
    for (std::size_t r = 0; r < rounds; ++r) {
        for (std::size_t i = 0; i < n; ++i) {
            delta[i] = static_cast<float>(static_cast<std::int64_t>(stats::uniform_index(g, 1u << 22)) - (1 << 21)) * 0x1.0p-20f;
            truth[i] += static_cast<double>(delta[i]);
        }
        ml::compress_bf16(delta, residual, bits);
        for (std::size_t i = 0; i < n; ++i)
            sent[i] += static_cast<double>(decoded(bits[i]));
    }
    for (std::size_t i = 0; i < n; ++i)
        ASSERT_EQ(sent[i] + static_cast<double>(residual[i]), truth[i]) << i;
}

TEST(MlDilocoBf16, ErrorFeedbackLosesOnlyTheFloatAdditionsAndFarLessThanNoFeedback)
{
    constexpr std::size_t n = 1000, rounds = 1000;
    stats::wyrand g(5);
    std::vector<float> residual(n), no_residual(n), delta(n);
    std::vector<std::uint16_t> bits(n), plain(n);
    std::vector<double> sent(n), sent_plain(n), truth(n), bound(n);
    for (std::size_t r = 0; r < rounds; ++r) {
        for (std::size_t i = 0; i < n; ++i) {
            delta[i] = static_cast<float>(stats::uniform_real(g) - 0.3) * 1e-3f;   // a drift, as a training direction has
            truth[i] += static_cast<double>(delta[i]);
            const float v = delta[i] + residual[i];
            bound[i] += 0.5 * static_cast<double>(std::nextafter(std::fabs(v), INFINITY) - std::fabs(v));   // half an ulp of v
        }
        ml::compress_bf16(delta, residual, bits);
        std::fill(no_residual.begin(), no_residual.end(), 0.0f);
        ml::compress_bf16(delta, no_residual, plain);
        for (std::size_t i = 0; i < n; ++i) {
            sent[i] += static_cast<double>(decoded(bits[i]));
            sent_plain[i] += static_cast<double>(decoded(plain[i]));
        }
    }
    double feedback_error = 0, plain_error = 0;
    for (std::size_t i = 0; i < n; ++i) {
        ASSERT_LE(std::fabs(truth[i] - sent[i] - static_cast<double>(residual[i])), bound[i]) << i;
        feedback_error += std::fabs(truth[i] - sent[i]);
        plain_error += std::fabs(truth[i] - sent_plain[i]);
    }
    std::printf("bfloat16 over %zu rounds: mean cumulative error %.3g with error feedback, %.3g without\n", rounds,
                feedback_error / n, plain_error / n);
    EXPECT_LT(10 * feedback_error, plain_error);
}

TEST(MlDilocoBf16, TheSumIsOrderedSumOverTheDecodedPartsToTheBit)
{
    stats::wyrand g(17);
    for (std::size_t workers = 1; workers <= 5; ++workers)
        for (const std::size_t n : {std::size_t{1}, std::size_t{7}, std::size_t{1001}}) {
            std::vector<std::vector<std::uint16_t>> bits(workers, std::vector<std::uint16_t>(n));
            std::vector<std::vector<float>> floats(workers, std::vector<float>(n));
            for (std::size_t k = 0; k < workers; ++k)
                for (std::size_t i = 0; i < n; ++i) {
                    bits[k][i] = ml::to_bfloat16(static_cast<float>(stats::uniform_real(g) - 0.5) * 1e-2f).bits;
                    floats[k][i] = decoded(bits[k][i]);
                }
            const std::vector<std::span<const std::uint16_t>> bit_parts(bits.begin(), bits.end());
            const std::vector<std::span<const float>> float_parts(floats.begin(), floats.end());
            std::vector<float> a(n), b(n), mean(n);
            ml::ordered_sum_bf16(bit_parts, a);
            ml::ordered_sum<float>(float_parts, b);
            ml::ordered_mean_bf16(bit_parts, mean);
            for (std::size_t i = 0; i < n; ++i) {
                ASSERT_EQ(std::bit_cast<std::uint32_t>(a[i]), std::bit_cast<std::uint32_t>(b[i])) << workers << " " << i;
                const float expect = ml::to_float(ml::to_bfloat16(b[i])) / static_cast<float>(workers);
                ASSERT_EQ(std::bit_cast<std::uint32_t>(mean[i]), std::bit_cast<std::uint32_t>(expect)) << workers << " " << i;
            }
        }
}

TEST(MlDilocoBf16, TheMeanOfOnePartIsItsValues)
{
    const std::array<std::uint16_t, 5> part{0x3f80, 0xbf00, 0x0001, 0x8000, 0x4049};
    const std::array<std::span<const std::uint16_t>, 1> parts{part};
    std::array<float, 5> mean{};
    ml::ordered_mean_bf16(parts, mean);
    for (std::size_t i = 0; i < part.size(); ++i)
        EXPECT_EQ(std::bit_cast<std::uint32_t>(mean[i]), std::bit_cast<std::uint32_t>(decoded(part[i]))) << i;
}

TEST(MlDilocoBf16, AMeanOfIdenticalPartsWithAnExactSumIsThePart)
{
    // Two copies of d: d + d is exact and a bfloat16, halving it gives d
    const std::array<std::uint16_t, 3> d{0x3c23, 0xc2f7, 0x0042};
    const std::array<std::span<const std::uint16_t>, 2> parts{d, d};
    std::array<float, 3> mean{};
    ml::ordered_mean_bf16(parts, mean);
    for (std::size_t i = 0; i < d.size(); ++i)
        EXPECT_EQ(mean[i], decoded(d[i])) << i;
}

} // namespace
