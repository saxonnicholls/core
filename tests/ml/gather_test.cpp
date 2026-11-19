//
//  gather_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/ml/gather.hpp>: gather_rows against a loop by hand to the bit over a batch of sequences;
//  scatter_add_rows as its backward (the adjoint identity <gather(x), y> = <x, scatter_add(y)>, central differences of
//  a loss through the gather), its fixed order on a repeated index, untouched rows unchanged; scatter_rows writing in
//  order with the last write winning; concat_rows and slice_rows and slice_rows_backward; compile-time use and the
//  noexcept claims.
//

#include <snicholls/core/ml/gather.hpp>

#include <snicholls/core/linalg/reference.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace {

namespace ml = snicholls::ml;
const auto& backend = snicholls::linalg::reference;

static_assert(noexcept(ml::gather_rows(backend, std::span<const float>{}, {}, std::span<float>{}, 1)));
static_assert(noexcept(ml::scatter_add_rows(backend, std::span<const double>{}, {}, std::span<double>{}, 1)));

constexpr bool at_compile_time()
{
    // Two sequences of three rows of two; take rows {2, 0} of the first and {1, 1} of the second
    const std::array<double, 12> src{0, 1, 10, 11, 20, 21, 100, 101, 110, 111, 120, 121};
    const std::array<std::uint32_t, 4> idx{2, 0, 1, 1};
    std::array<double, 8> out{};
    ml::gather_rows(backend, std::span<const double>(src), idx, std::span<double>(out), 2, 2);
    std::array<double, 12> back{};
    ml::scatter_add_rows(backend, std::span<const double>(out), idx, std::span<double>(back), 2, 2);
    return out[0] == 20 && out[2] == 0 && out[4] == 110 && out[6] == 110 && back[4] == 20 && back[8] == 220 && back[2] == 0;
}
static_assert(at_compile_time());

std::vector<double> wave(std::size_t n, double a, double b)
{
    std::vector<double> v(n);
    for (std::size_t i = 0; i < n; ++i)
        v[i] = a * std::sin(0.37 * double(i) + b);
    return v;
}

constexpr std::size_t width = 5, seqs = 3, n = 7, k = 4;
const std::vector<std::uint32_t> indices{0, 2, 5, 6, 1, 1, 3, 4, 6, 5, 4, 0};

TEST(MlGatherRows, IsALoopByHandToTheBit)
{
    const auto src = wave(seqs * n * width, 1.3, 0.2);
    std::vector<double> got(seqs * k * width), want(got.size());
    ml::gather_rows(backend, std::span<const double>(src), indices, std::span<double>(got), width, seqs);
    for (std::size_t s = 0; s < seqs; ++s)
        for (std::size_t j = 0; j < k; ++j)
            for (std::size_t c = 0; c < width; ++c)
                want[(s * k + j) * width + c] = src[(s * n + indices[s * k + j]) * width + c];
    EXPECT_EQ(got, want);
}

TEST(MlGatherRows, ScatterAddIsItsBackward)
{
    auto src = wave(seqs * n * width, 1.3, 0.2);
    const auto y = wave(seqs * k * width, 0.7, 1.1);
    // <gather(x), y> = <x, scatter_add(y)>, and the gradient of that loss in x is scatter_add(y), by central differences
    std::vector<double> grad(src.size());
    ml::scatter_add_rows(backend, std::span<const double>(y), indices, std::span<double>(grad), width, seqs);
    auto loss = [&] {
        std::vector<double> g(y.size());
        ml::gather_rows(backend, std::span<const double>(src), indices, std::span<double>(g), width, seqs);
        double l = 0;
        for (std::size_t i = 0; i < g.size(); ++i)
            l += g[i] * y[i];
        return l;
    };
    double lhs = loss(), rhs = 0;
    for (std::size_t i = 0; i < src.size(); ++i)
        rhs += src[i] * grad[i];
    EXPECT_NEAR(lhs, rhs, 1e-12);
    for (std::size_t i = 0; i < src.size(); ++i) {
        const double keep = src[i], e = 1e-6;
        src[i] = keep + e;
        const double hi = loss();
        src[i] = keep - e;
        const double lo = loss();
        src[i] = keep;
        EXPECT_NEAR(grad[i], (hi - lo) / (2 * e), 1e-8) << i;
    }
}

TEST(MlGatherRows, ScatterAddSumsARepeatedIndexInOrderAndLeavesOtherRows)
{
    // Sequence 1 names row 1 twice: its gradient is (initial + first) + second, in that order
    std::vector<float> dsrc(seqs * n * width, 0.5f);
    const std::vector<float> dout{
        // sequence 0
        1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
        // sequence 1: rows 1, 1, 3, 4
        1e8f, 1e8f, 1e8f, 1e8f, 1e8f, 1, 1, 1, 1, 1, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6,
        // sequence 2
        7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10};
    ml::scatter_add_rows(backend, std::span<const float>(dout), indices, std::span<float>(dsrc), width, seqs);
    EXPECT_EQ(dsrc[(1 * n + 1) * width], (0.5f + 1e8f) + 1.0f);
    EXPECT_EQ(dsrc[(0 * n + 1) * width], 0.5f);   // row 1 of sequence 0 is named by none
    EXPECT_EQ(dsrc[(2 * n + 6) * width], 0.5f + 7.0f);
}

TEST(MlScatterRows, WritesEachRowToItsPositionTheLastWriteWinning)
{
    std::vector<double> dst(seqs * n * width, -1.0);
    const auto rows = wave(seqs * k * width, 1.0, 0.0);
    ml::scatter_rows(backend, std::span<const double>(rows), indices, std::span<double>(dst), width, seqs);
    for (std::size_t c = 0; c < width; ++c) {
        EXPECT_EQ(dst[(0 * n + 5) * width + c], rows[(0 * k + 2) * width + c]);
        EXPECT_EQ(dst[(1 * n + 1) * width + c], rows[(1 * k + 1) * width + c]);   // the second write to row 1
        EXPECT_EQ(dst[(0 * n + 1) * width + c], -1.0);                           // untouched
    }
    // Its backward is gather: what each written row was, gathered back
    std::vector<double> back(rows.size());
    ml::gather_rows(backend, std::span<const double>(dst), std::span<const std::uint32_t>(indices).first(k), std::span<double>(back).first(k * width),
                    width, 1);
    for (std::size_t i = 0; i < k * width; ++i)
        EXPECT_EQ(back[i], rows[i]);
}

TEST(MlConcatRows, PutsEachSequencesFirstRowsBeforeItsSecond)
{
    const auto a = wave(seqs * 2 * width, 1.0, 0.1), b = wave(seqs * 3 * width, 2.0, 0.7);
    std::vector<double> out(a.size() + b.size());
    ml::concat_rows(backend, std::span<const double>(a), std::span<const double>(b), std::span<double>(out), width, seqs);
    for (std::size_t s = 0; s < seqs; ++s) {
        for (std::size_t c = 0; c < 2 * width; ++c)
            EXPECT_EQ(out[s * 5 * width + c], a[s * 2 * width + c]);
        for (std::size_t c = 0; c < 3 * width; ++c)
            EXPECT_EQ(out[s * 5 * width + 2 * width + c], b[s * 3 * width + c]);
    }
    // Slicing the parts back out undoes it
    std::vector<double> a2(a.size()), b2(b.size());
    ml::slice_rows(backend, std::span<const double>(out), 0, 2, std::span<double>(a2), width, seqs);
    ml::slice_rows(backend, std::span<const double>(out), 2, 3, std::span<double>(b2), width, seqs);
    EXPECT_EQ(a2, a);
    EXPECT_EQ(b2, b);
}

TEST(MlSliceRows, BackwardAddsIntoTheSlicedRowsOnly)
{
    const auto d = wave(seqs * 2 * width, 1.0, 0.4);
    std::vector<double> drows(seqs * n * width, 1.0);
    ml::slice_rows_backward(backend, std::span<const double>(d), 3, 2, std::span<double>(drows), width, seqs);
    for (std::size_t s = 0; s < seqs; ++s)
        for (std::size_t r = 0; r < n; ++r)
            for (std::size_t c = 0; c < width; ++c) {
                const double want = r == 3 || r == 4 ? 1.0 + d[(s * 2 + r - 3) * width + c] : 1.0;
                EXPECT_EQ(drows[(s * n + r) * width + c], want);
            }
}

} // namespace
