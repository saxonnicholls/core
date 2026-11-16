//
//  embedding_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/ml/embedding.hpp>: the row of a token without a copy, embed and embed_with_position
//  against the table, the first and last rows, the backward pass adding into one row only and accumulating over a
//  batch, and compile-time use.
//

#include <snicholls/core/ml/embedding.hpp>

#include <snicholls/core/linalg/reference.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace {

namespace ml = snicholls::ml;

// Three tokens of two channels: row i is {10 i, 10 i + 1}
constexpr std::array<double, 6> table{0, 1, 10, 11, 20, 21};

constexpr bool compile_time()
{
    std::array<double, 2> out{};
    ml::embed(table, 2, out);
    std::array<double, 6> grad{};
    ml::embedding_backward(std::array<double, 2>{1, 2}, 1, grad);
    return out[0] == 20 && out[1] == 21 && grad[2] == 1 && grad[3] == 2 && grad[0] == 0 && ml::embedding_row(table, 1, 2)[1] == 11;
}
static_assert(compile_time());

TEST(MlEmbedding, TheRowIsAViewOfTheTable)
{
    const std::vector<float> t{1, 2, 3, 4, 5, 6};
    const auto r = ml::embedding_row(t, 2, 2);
    EXPECT_EQ(r.data(), t.data() + 4);
    EXPECT_EQ(r.size(), 2u);
}

TEST(MlEmbedding, EmbedCopiesTheFirstAndLastRows)
{
    std::array<double, 2> out{};
    ml::embed(table, 0, out);
    EXPECT_EQ(out, (std::array<double, 2>{0, 1}));
    ml::embed(table, 2, out);
    EXPECT_EQ(out, (std::array<double, 2>{20, 21}));
}

TEST(MlEmbedding, TokenPlusPositionIsGpt2sInput)
{
    const std::array<double, 4> positions{0.5, 0.25, -1, -2};   // two positions
    std::array<double, 2> out{};
    ml::embed_with_position(table, 1, positions, 1, out);
    EXPECT_EQ(out, (std::array<double, 2>{9, 9}));
}

TEST(MlEmbedding, BackwardAddsIntoTheTokensRowOverABatch)
{
    // Tokens 2, 0, 2 in a batch: row 2 gets two gradients, row 1 none
    std::vector<double> grad(6, 0.0);
    const std::array<std::size_t, 3> ids{2, 0, 2};
    const std::array<std::array<double, 2>, 3> douts{{{1, 1}, {3, 4}, {0.5, -1}}};
    for (std::size_t b = 0; b < 3; ++b)
        ml::embedding_backward(douts[b], ids[b], grad);
    EXPECT_EQ(grad, (std::vector<double>{3, 4, 0, 0, 1.5, 0}));
}

// A sequence's embeddings on a backend: embed_with_position per token, and the backward adding in token order
TEST(MlEmbedding, EmbedRowsIsEmbedWithPositionPerTokenAndItsBackwardAddsInOrder)
{
    constexpr std::size_t vocab = 4, d = 3, n = 4;
    std::vector<float> table(vocab * d), positions(n * d);
    for (std::size_t i = 0; i < table.size(); ++i)
        table[i] = 0.5f * float(i) - 2;
    for (std::size_t i = 0; i < positions.size(); ++i)
        positions[i] = 0.1f * float(i);
    const std::vector<std::uint32_t> ids{2, 0, 2, 3};   // token 2 twice: its gradient row gets both
    std::vector<float> out(n * d), want(n * d);
    ml::embed_rows(snicholls::linalg::reference, std::span<const float>(table), std::span<const float>(positions), ids, std::span<float>(out));
    for (std::size_t t = 0; t < n; ++t)
        ml::embed_with_position(table, ids[t], positions, t, std::span<float>(want).subspan(t * d, d));
    EXPECT_EQ(out, want);
    std::vector<float> dout(n * d), dtable(vocab * d), dpos(n * d), dtable_want(vocab * d);
    for (std::size_t i = 0; i < dout.size(); ++i)
        dout[i] = float(i) + 1;
    ml::embed_rows_backward(snicholls::linalg::reference, std::span<const float>(dout), ids, std::span<float>(dtable), std::span<float>(dpos));
    for (std::size_t t = 0; t < n; ++t)
        ml::embedding_backward(std::span<const float>(dout).subspan(t * d, d), ids[t], dtable_want);
    EXPECT_EQ(dtable, dtable_want);
    EXPECT_EQ(dpos, dout);
    EXPECT_EQ(dtable[2 * d], dout[0] + dout[2 * d]);   // both uses of token 2
    std::vector<float> no_positions(n * d);
    ml::embed_rows(snicholls::linalg::reference, std::span<const float>(table), std::span<const float>(), ids, std::span<float>(no_positions));
    EXPECT_EQ(no_positions[0], table[2 * d]);
}

} // namespace
