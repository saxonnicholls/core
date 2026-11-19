//
//  hadamard_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/ml/hadamard.hpp>: the orthonormal and block transforms equal ggml's CPU transform bit
//  for bit and the rotation matrix equals llama.cpp's on every case of the committed fixture
//  (scripts/generate_hadamard_fixture.cpp); the fast transform equals Sylvester's matrix times the row; H H = n I;
//  norms are kept; the randomised transform round-trips; folding keeps W x; rotating a row with outliers lowers its
//  per-row int8 error tenfold; and compile-time use.
//

#include <snicholls/core/ml/hadamard.hpp>
#include <snicholls/core/ml/quantize.hpp>

#include "quant_rows.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace ml = snicholls::ml;

// Compile time: the transform of a unit vector is a column of H; the orthonormal form undoes itself
constexpr bool compile_time()
{
    std::array<double, 8> e{0, 0, 0, 1, 0, 0, 0, 0};
    ml::fwht(e);
    bool ok = true;
    for (std::size_t i = 0; i < 8; ++i)
        ok = ok && e[i] == double(ml::hadamard_entry(i, 3));
    std::array<float, 4> x{1.0f, 2.0f, -3.0f, 0.5f};
    ml::hadamard(x);   // n = 4: scale 1/2 exact, so the round trip is exact
    ml::hadamard(x);
    std::array<std::int8_t, 4> s{};
    ml::hadamard_signs(s, 9);
    std::array<float, 4> y{1.0f, 2.0f, -3.0f, 0.5f};
    ml::random_hadamard(y, s, 4);
    ml::random_hadamard_inverse(y, s, 4);
    return ok && x[0] == 1.0f && x[2] == -3.0f && x[3] == 0.5f && y[1] == 2.0f && y[2] == -3.0f;
}
static_assert(compile_time());
static_assert(ml::hadamard_entry(0, 7) == 1 && ml::hadamard_entry(3, 1) == -1 && ml::hadamard_entry(3, 3) == 1);
static_assert(ml::is_hadamard_order(1) && ml::is_hadamard_order(4096) && !ml::is_hadamard_order(0) && !ml::is_hadamard_order(12));
static_assert(noexcept(ml::hadamard(std::declval<std::array<float, 4>&>())));

struct fixture_row {
    int kind;
    std::size_t n, block;
    std::uint64_t x, h;
};

struct fixture {
    std::vector<fixture_row> rows;
    std::vector<std::pair<std::size_t, std::uint64_t>> matrices;
};

std::uint64_t hex(const std::string& word)
{
    return std::stoull(word.substr(word.find('=') + 1), nullptr, 16);
}

const fixture& data()
{
    static const fixture f = [] {
        fixture out;
        std::ifstream in(std::string(SNICHOLLS_CORE_FIXTURE_DIR) + "/ml/hadamard_fixture.txt");
        EXPECT_TRUE(in.good()) << "missing tests/fixtures/ml/hadamard_fixture.txt";
        std::string line;
        while (std::getline(in, line)) {
            std::istringstream ls(line);
            std::string what, a, b;
            ls >> what;
            if (what == "row") {
                fixture_row r{};
                std::string x, h;
                ls >> r.kind >> r.n >> r.block >> x >> h;
                r.x = hex(x);
                r.h = hex(h);
                out.rows.push_back(r);
            } else if (what == "matrix") {
                std::size_t n = 0;
                ls >> n >> a;
                out.matrices.emplace_back(n, hex(a));
            }
        }
        return out;
    }();
    return f;
}

TEST(MlHadamard, IsGgmlsTransformBitForBitOnEveryFixtureRow)
{
    ASSERT_EQ(data().rows.size(), 13u * 16u + 4u * 16u);
    for (const auto& r : data().rows) {
        auto x = quant_rows::row(r.kind, r.n, r.n == r.block ? 7 : 8);
        ASSERT_EQ(quant_rows::fnv_floats(x), r.x) << "input " << r.kind << " " << r.n;
        if (r.n == r.block)
            ml::hadamard(x);
        else
            ml::block_hadamard(x, r.block);
        EXPECT_EQ(quant_rows::fnv_floats(x), r.h) << quant_rows::name(r.kind) << " n " << r.n << " block " << r.block;
    }
}

TEST(MlHadamard, MatrixIsLlamaCppsRotationBitForBit)
{
    ASSERT_EQ(data().matrices.size(), 10u);
    for (const auto& [n, hash] : data().matrices) {
        std::vector<float> m(n * n);
        ml::hadamard_matrix<float>(n, m);
        EXPECT_EQ(quant_rows::fnv_floats(m), hash) << "n " << n;
    }
}

TEST(MlHadamard, FastTransformIsSylvestersMatrixTimesTheRow)
{
    for (std::size_t n = 1; n <= 256; n *= 2) {
        const auto x0 = quant_rows::row(quant_rows::uniform, n, 3);
        std::vector<double> x(x0.begin(), x0.end());
        ml::fwht(x);
        for (std::size_t i = 0; i < n; ++i) {
            double s = 0;
            for (std::size_t j = 0; j < n; ++j)
                s += ml::hadamard_entry(i, j) * double(x0[j]);
            EXPECT_NEAR(x[i], s, 1e-12 * double(n)) << n << " " << i;
        }
    }
}

TEST(MlHadamard, HTimesHIsNTimesTheIdentity)
{
    constexpr std::size_t n = 64;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t k = 0; k < n; ++k) {
            int s = 0;
            for (std::size_t j = 0; j < n; ++j)
                s += ml::hadamard_entry(i, j) * ml::hadamard_entry(j, k);
            EXPECT_EQ(s, i == k ? int(n) : 0);
        }
    // and the fast transform twice multiplies by n exactly on integers
    std::vector<double> x{3, -1, 4, 1, -5, 9, 2, -6};
    const auto x0 = x;
    ml::fwht(x);
    ml::fwht(x);
    for (std::size_t i = 0; i < x.size(); ++i)
        EXPECT_EQ(x[i], 8 * x0[i]);
}

TEST(MlHadamard, OrthonormalFormKeepsTheNormAndUndoesItself)
{
    for (std::size_t n : {std::size_t{2}, std::size_t{128}, std::size_t{2048}}) {
        const auto x0 = quant_rows::row(quant_rows::weights, n, 4);
        std::vector<double> x(x0.begin(), x0.end());
        double before = 0, after = 0;
        for (const double v : x)
            before += v * v;
        ml::hadamard(x);
        for (const double v : x)
            after += v * v;
        EXPECT_NEAR(after, before, 1e-13 * before);
        ml::hadamard(x);
        for (std::size_t i = 0; i < n; ++i)
            EXPECT_NEAR(x[i], double(x0[i]), 1e-15);
    }
}

TEST(MlHadamard, BlockFormTransformsEachBlockAlone)
{
    auto x = quant_rows::row(quant_rows::uniform, 96, 5);   // three blocks of 32: 96 is not a power of two
    auto y = x;
    ml::block_hadamard(x, 32);
    for (std::size_t b = 0; b < 96; b += 32)
        ml::hadamard(std::span<float>(y).subspan(b, 32));
    EXPECT_EQ(x, y);
}

TEST(MlHadamard, SignsAreTheSameForASeedAndBalanced)
{
    std::vector<std::int8_t> a(4096), b(4096), c(4096);
    ml::hadamard_signs(a, 1);
    ml::hadamard_signs(b, 1);
    ml::hadamard_signs(c, 2);
    EXPECT_EQ(a, b);
    EXPECT_NE(a, c);
    int sum = 0;
    for (const auto s : a) {
        EXPECT_TRUE(s == 1 || s == -1);
        sum += s;
    }
    EXPECT_LT(std::abs(sum), 4 * 64);   // 4 sigma for 4096 fair signs
    // pinned: the first draw of wyrand(1), bit by bit
    snicholls::stats::wyrand g(1);
    const std::uint64_t first = g();
    for (std::size_t k = 0; k < 64; ++k)
        EXPECT_EQ(a[k], (first >> k) & 1 ? -1 : 1);
}

TEST(MlHadamard, RandomisedTransformRoundTrips)
{
    const auto x0 = quant_rows::row(quant_rows::outliers, 512, 6);
    std::vector<std::int8_t> s(512);
    ml::hadamard_signs(s, 42);
    for (const std::size_t block : {std::size_t{512}, std::size_t{128}}) {
        auto x = x0;
        ml::random_hadamard(x, s, block);
        EXPECT_NE(x, x0);
        ml::random_hadamard_inverse(x, s, block);
        for (std::size_t i = 0; i < x.size(); ++i)
            EXPECT_NEAR(x[i], x0[i], 2e-5f * (1.0f + std::fabs(x0[i])));
    }
    // block 256 = 4^4: every scale is exact, and these integers sum exactly
    std::vector<float> ints(256);
    for (std::size_t i = 0; i < 256; ++i)
        ints[i] = float(int(i % 17) - 8);
    const auto i0 = ints;
    std::vector<std::int8_t> s256(256);
    ml::hadamard_signs(s256, 3);
    ml::random_hadamard(ints, s256, 256);
    ml::random_hadamard_inverse(ints, s256, 256);
    EXPECT_EQ(ints, i0);
}

TEST(MlHadamard, FoldingKeepsTheProduct)
{
    constexpr std::size_t rows = 8, cols = 256;
    const auto w0 = quant_rows::row(quant_rows::weights, rows * cols, 7);
    const auto x0 = quant_rows::row(quant_rows::uniform, cols, 8);
    std::vector<std::int8_t> s(cols);
    ml::hadamard_signs(s, 11);
    std::vector<double> w(w0.begin(), w0.end()), x(x0.begin(), x0.end());
    ml::rotate_rows(w, cols, s, 64);
    ml::random_hadamard(x, s, 64);
    for (std::size_t r = 0; r < rows; ++r) {
        double plain = 0, folded = 0;
        for (std::size_t j = 0; j < cols; ++j) {
            plain += double(w0[r * cols + j]) * double(x0[j]);
            folded += w[r * cols + j] * x[j];
        }
        EXPECT_NEAR(folded, plain, 1e-12);
    }
}

TEST(MlHadamard, RotatingARowWithOutliersLowersItsPerRowInt8ErrorTenfold)
{
    // a row with two outliers quantised with one scale for the row (per-channel int8, as QuaRot quantises): the
    // outliers set the scale and every other number rounds to a few levels; rotated, the outliers are spread out
    constexpr std::size_t n = 1024;
    const auto w = quant_rows::row(quant_rows::outliers, n, 9);
    std::vector<std::int8_t> s(n);
    ml::hadamard_signs(s, 5);
    const auto error = [](const std::vector<float>& v) {
        std::vector<std::int8_t> q(v.size());
        const float scale = ml::quantize_int8(v, q);
        double e = 0;
        for (std::size_t i = 0; i < v.size(); ++i) {
            const double d = double(q[i]) * double(scale) - double(v[i]);
            e += d * d;
        }
        return e;
    };
    for (const std::size_t block : {std::size_t{128}, n}) {
        auto r = w;
        ml::random_hadamard(r, s, block);
        EXPECT_LT(error(r), 0.1 * error(w)) << "block " << block;   // orthonormal: the errors compare directly
    }
    // a block format (Q4_0, a scale per 32) already confines each outlier to its block, and the rotation spreads
    // it into every block: it does not help there (documented under Known issues)
    std::vector<ml::block_q4_0> q(n / 32);
    auto r = w;
    ml::random_hadamard(r, s, n);
    ml::quantize(r, std::span<ml::block_q4_0>(q));
    EXPECT_GT(ml::to_float(q[3].d), 0.1f);   // block 3 had no outlier; rotated, its scale is the outliers' share
}

TEST(MlHadamard, NanPropagatesThroughItsBlockOnly)
{
    auto x = quant_rows::row(quant_rows::with_nan, 128, 1);   // a NaN at 7, an infinity at 67
    ml::block_hadamard(x, 64);
    for (std::size_t i = 0; i < 64; ++i)
        EXPECT_TRUE(std::isnan(x[i]));
    for (std::size_t i = 64; i < 128; ++i)
        EXPECT_TRUE(std::isinf(x[i]));
}

} // namespace
