//
//  lora_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/ml/lora.hpp>: with B = 0 the output is the base's to the bit (float, bfloat16 and NF4
//  bases; one row, a sequence, heads); the output equals the merged weight's within rounding; the backward against
//  central differences for A, B and the input, and nothing for the base; the sequence and head forms against the row
//  form; QLoRA's NF4 base equal to the dense base it dequantises to, to the bit; the initialisations (B zero, A within
//  the Kaiming bound, Gaussian of std 1/r, reproducible from the seed); merge then unmerge within the stated bound;
//  the optimiser's view stepping A and B only; a tiny fit where the adapter alone reduces the loss; compile-time use.
//

#include <snicholls/core/ml/lora.hpp>

#include <snicholls/core/linalg/reference.hpp>
#include <snicholls/core/ml/minifloat.hpp>
#include <snicholls/core/ml/optim.hpp>
#include <snicholls/core/stats/random.hpp>

#include <gtest/gtest.h>

#include <array>
#include <bit>
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

static_assert(std::is_trivially_copyable_v<ml::lora_weights<ml::dense_weights<float>, float>>);
static_assert(std::is_trivially_copyable_v<ml::lora_parameters<float>>);
static_assert(noexcept(ml::lora_scaling(16.0f, 8)));

constexpr bool at_compile_time()
{
    // W = I (2x2), A = [1 1] (1x2), B = [1; 0] (2x1), s = 0.5: y = x + 0.5 (x0 + x1, 0)
    const std::array<double, 4> w{1, 0, 0, 1};
    const std::array<double, 2> a{1, 1}, b{1, 0}, x{2, 4};
    const ml::lora_weights<ml::dense_weights<double>, double> lw{{w, 2, 2}, a, b, 1, 0.5};
    std::array<double, 2> y{};
    std::array<double, 1> u{};
    ml::lora_linear(lw, x, y, u);
    std::array<double, 4> merged = w;
    ml::lora_merge<double>(merged, 2, a, b, 1, 0.5);
    ml::lora_unmerge<double>(merged, 2, a, b, 1, 0.5);
    return y[0] == 5.0 && y[1] == 4.0 && u[0] == 6.0 && merged == w && ml::lora_scaling(16.0, 8) == 2.0;
}
static_assert(at_compile_time());

template <typename T>
std::vector<T> random_vec(stats::wyrand& g, std::size_t n, double scale = 1.0)
{
    std::vector<T> v(n);
    for (auto& x : v)
        x = static_cast<T>(scale * (2.0 * stats::uniform_real(g) - 1.0));
    return v;
}

bool same_bits(std::span<const float> a, std::span<const float> b)
{
    for (std::size_t i = 0; i < a.size(); ++i)
        if (std::bit_cast<std::uint32_t>(a[i]) != std::bit_cast<std::uint32_t>(b[i]))
            return false;
    return a.size() == b.size();
}

TEST(MlLora, WithBZeroIsTheBaseToTheBit)
{
    stats::wyrand g(17);
    constexpr std::size_t rows = 6, cols = 64, r = 4, n = 3;
    const auto w = random_vec<float>(g, rows * cols), a = random_vec<float>(g, r * cols), x = random_vec<float>(g, n * cols),
               bias = random_vec<float>(g, rows);
    const std::vector<float> b(rows * r, 0.0f);
    std::vector<float> y(rows), base(rows), u(r);
    // float base
    const ml::dense_weights<float> dense{w, rows, cols};
    ml::lora_linear(ml::lora_weights<ml::dense_weights<float>, float>{dense, a, b, r, 2.0f}, std::span<const float>(x).first(cols), y, u, bias);
    ml::linear(dense, std::span<const float>(x).first(cols), base, bias);
    EXPECT_TRUE(same_bits(y, base));
    // bfloat16 base
    std::vector<ml::bfloat16> wb(w.size());
    for (std::size_t i = 0; i < w.size(); ++i)
        wb[i] = ml::to_bfloat16(w[i]);
    const ml::dense_weights<ml::bfloat16> dense_bf{wb, rows, cols};
    ml::lora_linear(ml::lora_weights<ml::dense_weights<ml::bfloat16>, float>{dense_bf, a, b, r, 2.0f}, std::span<const float>(x).first(cols), y, u);
    ml::linear(dense_bf, std::span<const float>(x).first(cols), base);
    EXPECT_TRUE(same_bits(y, base));
    // a sequence
    std::vector<float> ys(n * rows), bases(n * rows), us(n * r);
    ml::lora_sequence(backend, ml::lora_weights<ml::dense_weights<float>, float>{dense, a, b, r, 2.0f}, x, ys, us, bias);
    ml::linear_sequence(backend, dense, x, bases, bias);
    EXPECT_TRUE(same_bits(ys, bases));
    // heads, two sequences
    std::vector<float> yh(n * rows), baseh(n * rows), uh(n * r);
    const auto x2 = random_vec<float>(g, 2 * 2 * cols);
    yh.resize(4 * rows);
    baseh.resize(4 * rows);
    uh.resize(4 * r);
    ml::lora_heads(backend, ml::lora_weights<ml::dense_weights<float>, float>{dense, a, b, r, 2.0f}, x2, yh, uh, bias, 3, 2);
    ml::linear_heads(backend, dense, x2, baseh, bias, 3, 2);
    EXPECT_TRUE(same_bits(yh, baseh));
}

TEST(MlLora, IsTheMergedWeightsProductWithinRounding)
{
    stats::wyrand g(18);
    constexpr std::size_t rows = 5, cols = 7, r = 3;
    const auto w = random_vec<double>(g, rows * cols), a = random_vec<double>(g, r * cols), b = random_vec<double>(g, rows * r),
               x = random_vec<double>(g, cols);
    const ml::lora_weights<ml::dense_weights<double>, double> lw{{w, rows, cols}, a, b, r, 0.75};
    std::vector<double> y(rows), u(r), merged = w, ym(rows);
    ml::lora_linear(lw, x, y, u);
    ml::lora_merge<double>(merged, cols, a, b, r, 0.75);
    ml::linear(ml::dense_weights<double>{merged, rows, cols}, x, ym);
    for (std::size_t i = 0; i < rows; ++i)
        EXPECT_NEAR(y[i], ym[i], 1e-14);
}

TEST(MlLora, TheBackwardIsCentralDifferencesAndTheSequenceAndHeadFormsAgree)
{
    stats::wyrand g(19);
    constexpr std::size_t rows = 6, cols = 5, r = 2, n = 4, heads = 2;
    const auto w = random_vec<double>(g, rows * cols), x = random_vec<double>(g, n * cols), dy = random_vec<double>(g, n * rows);
    auto a = random_vec<double>(g, r * cols), b = random_vec<double>(g, rows * r);
    const double s = 1.5;
    const auto objective = [&](const std::vector<double>& aa, const std::vector<double>& bb, const std::vector<double>& xx) {
        double o = 0;
        for (std::size_t t = 0; t < n; ++t) {
            std::vector<double> y(rows), u(r);
            ml::lora_linear(ml::lora_weights<ml::dense_weights<double>, double>{{w, rows, cols}, aa, bb, r, s},
                            std::span<const double>(xx).subspan(t * cols, cols), y, u);
            for (std::size_t i = 0; i < rows; ++i)
                o += dy[t * rows + i] * y[i];
        }
        return o;
    };
    const ml::lora_weights<ml::dense_weights<double>, double> lw{{w, rows, cols}, a, b, r, s};
    std::vector<double> da(r * cols, 0.0), db(rows * r, 0.0), dx(n * cols, 0.0), u(r), du(r), y(rows);
    for (std::size_t t = 0; t < n; ++t) {
        const auto xt = std::span<const double>(x).subspan(t * cols, cols);
        ml::lora_linear(lw, xt, y, u);
        ml::lora_linear_backward(lw, xt, u, std::span<const double>(dy).subspan(t * rows, rows), std::span<double>(dx).subspan(t * cols, cols), da, db, du);
    }
    const double h = 1e-6;
    for (std::size_t i = 0; i < a.size(); ++i) {
        auto p = a, m = a;
        p[i] += h;
        m[i] -= h;
        EXPECT_NEAR(da[i], (objective(p, b, x) - objective(m, b, x)) / (2 * h), 1e-8);
    }
    for (std::size_t i = 0; i < b.size(); ++i) {
        auto p = b, m = b;
        p[i] += h;
        m[i] -= h;
        EXPECT_NEAR(db[i], (objective(a, p, x) - objective(a, m, x)) / (2 * h), 1e-8);
    }
    for (std::size_t i = 0; i < x.size(); ++i) {
        auto p = x, m = x;
        p[i] += h;
        m[i] -= h;
        EXPECT_NEAR(dx[i], (objective(a, b, p) - objective(a, b, m)) / (2 * h), 1e-8);
    }
    // The sequence form: the row form within rounding
    std::vector<double> ys(n * rows), us(n * r), dus(n * r), da2(r * cols, 0.0), db2(rows * r, 0.0), dx2(n * cols, 0.0);
    ml::lora_sequence(backend, lw, x, ys, us);
    ml::lora_sequence_backward(backend, lw, x, us, dy, dx2, da2, db2, dus);
    for (std::size_t i = 0; i < da.size(); ++i)
        EXPECT_NEAR(da2[i], da[i], 1e-13);
    for (std::size_t i = 0; i < db.size(); ++i)
        EXPECT_NEAR(db2[i], db[i], 1e-13);
    for (std::size_t i = 0; i < dx.size(); ++i)
        EXPECT_NEAR(dx2[i], dx[i], 1e-13);
    // Heads: the outputs regrouped head by head, the same gradients when the output gradients are regrouped too
    constexpr std::size_t d = rows / heads;
    std::vector<double> yh(n * rows), uh(n * r), dyh(n * rows), dush(n * r), da3(r * cols, 0.0), db3(rows * r, 0.0), dx3(n * cols, 0.0);
    ml::lora_heads(backend, lw, x, yh, uh, {}, heads);
    for (std::size_t hd = 0; hd < heads; ++hd)
        for (std::size_t t = 0; t < n; ++t)
            for (std::size_t k = 0; k < d; ++k) {
                EXPECT_NEAR(yh[(hd * n + t) * d + k], ys[t * rows + hd * d + k], 1e-13);
                dyh[(hd * n + t) * d + k] = dy[t * rows + hd * d + k];
            }
    ml::lora_heads_backward(backend, lw, x, uh, dyh, dx3, da3, db3, dush, heads);
    for (std::size_t i = 0; i < da.size(); ++i)
        EXPECT_NEAR(da3[i], da[i], 1e-13);
    for (std::size_t i = 0; i < db.size(); ++i)
        EXPECT_NEAR(db3[i], db[i], 1e-13);
    for (std::size_t i = 0; i < dx.size(); ++i)
        EXPECT_NEAR(dx3[i], dx[i], 1e-13);
}

TEST(MlQlora, AnNf4BaseIsTheDenseBaseItDequantisesToToTheBit)
{
    stats::wyrand g(20);
    constexpr std::size_t rows = 4, cols = 128, r = 2, n = 2;
    const auto w = random_vec<float>(g, rows * cols, 0.1), a = random_vec<float>(g, r * cols), b = random_vec<float>(g, rows * r),
               x = random_vec<float>(g, n * cols), dy = random_vec<float>(g, n * rows);
    std::vector<std::uint8_t> codes(ml::nf4_code_bytes(w.size()));
    std::vector<ml::float8_e4m3> blocks(ml::nf4_block_count(w.size()));
    std::vector<float> groups(ml::nf4_group_count(w.size())), deq(w.size());
    const float mean = ml::nf4_quantize(w, codes, blocks, groups);
    const ml::nf4_weights q{codes, blocks, groups, mean, rows, cols};
    ml::nf4_dequantize(q, deq);
    const ml::lora_weights<ml::nf4_weights, float> lq{q, a, b, r, 2.0f};
    const ml::lora_weights<ml::dense_weights<float>, float> ld{{deq, rows, cols}, a, b, r, 2.0f};
    std::vector<float> y1(rows), y2(rows), u1(r), u2(r), dx1(cols, 0.0f), dx2(cols, 0.0f), da1(r * cols, 0.0f), da2(r * cols, 0.0f),
        db1(rows * r, 0.0f), db2(rows * r, 0.0f), du(r);
    const auto x0 = std::span<const float>(x).first(cols);
    const auto dy0 = std::span<const float>(dy).first(rows);
    ml::lora_linear(lq, x0, y1, u1);
    ml::lora_linear(ld, x0, y2, u2);
    EXPECT_TRUE(same_bits(y1, y2));
    ml::lora_linear_backward(lq, x0, u1, dy0, dx1, da1, db1, du);
    ml::lora_linear_backward(ld, x0, u2, dy0, dx2, da2, db2, du);
    EXPECT_TRUE(same_bits(dx1, dx2) && same_bits(da1, da2) && same_bits(db1, db2));
    // The sequence form over the NF4 base, through the backend
    std::vector<float> ys(n * rows), us(n * r), dus(n * r), dxs(n * cols, 0.0f), das(r * cols, 0.0f), dbs(rows * r, 0.0f);
    ml::lora_sequence(backend, lq, x, ys, us);
    ml::lora_sequence_backward(backend, lq, x, us, dy, dxs, das, dbs, dus);
    for (std::size_t i = 0; i < rows; ++i)
        EXPECT_NEAR(ys[i], y1[i], 1e-5);
    for (std::size_t i = 0; i < cols; ++i)
        EXPECT_TRUE(std::isfinite(dxs[i]));
}

TEST(MlLoraInitialise, ZeroesBAndDrawsAFromTheStatedDistributions)
{
    constexpr std::size_t cols = 256, r = 8, rows = 4;
    std::vector<double> a(r * cols), b(rows * r, 7.0);
    stats::wyrand g(21), g2(21);
    ml::lora_initialise<double>(a, b, cols, g);
    for (const double v : b)
        EXPECT_EQ(v, 0.0);
    double mx = 0, ss = 0;
    for (const double v : a) {
        mx = std::fmax(mx, std::fabs(v));
        ss += v * v;
    }
    EXPECT_LE(mx, 1.0 / 16);
    EXPECT_NEAR(std::sqrt(ss / a.size()), 1.0 / 16 / std::sqrt(3.0), 0.003);   // uniform: std = bound / sqrt(3)
    std::vector<double> a2(r * cols);
    ml::lora_initialise<double>(a2, b, cols, g2);
    EXPECT_EQ(a, a2);   // the same seed, the same adapter
    ml::lora_initialise<double>(a, b, cols, g, ml::lora_init::gaussian);
    ss = 0;
    for (const double v : a) {
        EXPECT_TRUE(std::isfinite(v));
        ss += v * v;
    }
    EXPECT_NEAR(std::sqrt(ss / a.size()), 1.0 / r, 0.006);
}

TEST(MlLoraMerge, ThenUnmergeReturnsTheWeightWithinTheBound)
{
    stats::wyrand g(22);
    constexpr std::size_t rows = 16, cols = 24, r = 4;
    const auto w = random_vec<float>(g, rows * cols), a = random_vec<float>(g, r * cols), b = random_vec<float>(g, rows * r);
    auto m = w;
    ml::lora_merge<float>(m, cols, a, b, r, 4.0f);
    ml::lora_unmerge<float>(m, cols, a, b, r, 4.0f);
    std::size_t exact = 0;
    for (std::size_t i = 0; i < rows; ++i)
        for (std::size_t j = 0; j < cols; ++j) {
            float upd = 0;
            for (std::size_t k = 0; k < r; ++k)
                upd += b[i * r + k] * a[k * cols + j];
            const float bound = 2 * std::numeric_limits<float>::epsilon() * (std::fabs(w[i * cols + j]) + std::fabs(4.0f * upd));
            EXPECT_LE(std::fabs(m[i * cols + j] - w[i * cols + j]), bound);
            exact += m[i * cols + j] == w[i * cols + j] ? 1 : 0;
        }
    EXPECT_GT(exact, 0u);
}

TEST(MlLoraParameters, AnOptimiserStepsTheAdapterAloneAndTheLossFalls)
{
    // Fit y = W* x with W frozen at 0 and a rank-1 adapter: only A and B move
    constexpr std::size_t rows = 3, cols = 4, r = 1, n = 16;
    stats::wyrand g(23);
    const std::vector<double> w(rows * cols, 0.0);
    const std::vector<double> target_b{1.0, -2.0, 0.5}, target_a{0.5, 0.25, -1.0, 2.0};
    const auto xs = random_vec<double>(g, n * cols);
    std::vector<double> a(r * cols), b(rows * r), da(a.size()), db(b.size()), ma(a.size()), va(a.size()), mb(b.size()), vb(b.size());
    ml::lora_initialise<double>(a, b, cols, g);
    const ml::lora_parameters<double> params{a, b, da, db};
    EXPECT_EQ(params.size(), 7u);
    double first = -1, last = -1;
    for (std::uint64_t step = 1; step <= 300; ++step) {
        std::fill(da.begin(), da.end(), 0.0);
        std::fill(db.begin(), db.end(), 0.0);
        const ml::lora_weights<ml::dense_weights<double>, double> lw{{w, rows, cols}, a, b, r, 1.0};
        double loss = 0;
        for (std::size_t t = 0; t < n; ++t) {
            const auto x = std::span<const double>(xs).subspan(t * cols, cols);
            std::vector<double> y(rows), u(r), du(r), dy(rows);
            ml::lora_linear(lw, x, y, u);
            double proj = 0;
            for (std::size_t j = 0; j < cols; ++j)
                proj += target_a[j] * x[j];
            for (std::size_t i = 0; i < rows; ++i) {
                const double e = y[i] - target_b[i] * proj;
                loss += 0.5 * e * e / n;
                dy[i] = e / n;
            }
            ml::lora_linear_backward(lw, x, u, dy, {}, da, db, du);
        }
        if (first < 0)
            first = loss;
        last = loss;
        const auto p = params.parameters();
        const auto gr = params.gradients();
        ml::adamw_step(p[0], gr[0], ma, va, step, ml::adamw_config<double>{.learning_rate = 0.05});
        ml::adamw_step(p[1], gr[1], mb, vb, step, ml::adamw_config<double>{.learning_rate = 0.05});
    }
    for (const double v : w)
        EXPECT_EQ(v, 0.0);
    EXPECT_LT(last, first * 1e-3);
}

} // namespace
