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

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
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

} // namespace
