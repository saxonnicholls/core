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
//  compile-time use and the noexcept claims.
//

#include <snicholls/core/ml/diloco.hpp>

#include <snicholls/core/stats/random.hpp>

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
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

} // namespace
