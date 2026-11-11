//
//  embedding_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/ml/embedding.hpp>: a sequence of tokens embedded with GPT-2's token and position
//  tables, one token's row read without a copy, and the backward pass accumulating the gradient of a repeated
//  token. Exits non-zero if a result is not what it documents.
//

#include <snicholls/core/ml/embedding.hpp>

#include <array>
#include <cstddef>
#include <cstdio>

int main()
{
    std::printf("snicholls-core embedding demo\n");
    namespace ml = snicholls::ml;
    bool ok = true;

    //! [embedding]
    // A vocabulary of 4 tokens and a context of 3 positions, 2 channels each, row after row
    constexpr std::size_t channels = 2;
    const std::array<float, 4 * channels> wte{0.0f, 0.1f, 1.0f, 1.1f, 2.0f, 2.1f, 3.0f, 3.1f};
    const std::array<float, 3 * channels> wpe{0.00f, 0.00f, 0.01f, 0.02f, 0.03f, 0.04f};
    const std::array<std::size_t, 3> sequence{3, 1, 3};

    std::array<std::array<float, channels>, 3> x{};
    for (std::size_t pos = 0; pos < sequence.size(); ++pos)
        ml::embed_with_position(wte, sequence[pos], wpe, pos, x[pos]);

    // With RoPE there is no position table: the token's row alone, or a view of it
    std::array<float, channels> token1{};
    ml::embed(wte, 1, token1);
    const auto view = ml::embedding_row(wte, 2, channels);

    // Backward: each position's gradient adds into its token's row, so token 3 collects two
    std::array<float, 4 * channels> dwte{};
    const std::array<float, channels> dx{1.0f, -1.0f};
    for (const std::size_t id : sequence)
        ml::embedding_backward(dx, id, dwte);
    //! [embedding]

    for (std::size_t pos = 0; pos < 3; ++pos)
        std::printf("  position %zu, token %zu: [%.2f %.2f]\n", pos, sequence[pos], static_cast<double>(x[pos][0]), static_cast<double>(x[pos][1]));
    std::printf("  token 1 alone: [%.2f %.2f]; token 2 viewed: [%.2f %.2f]\n", static_cast<double>(token1[0]), static_cast<double>(token1[1]),
                static_cast<double>(view[0]), static_cast<double>(view[1]));
    std::printf("  gradient of token 3's row: [%.1f %.1f]\n", static_cast<double>(dwte[6]), static_cast<double>(dwte[7]));
    ok &= x[2][0] == 3.0f + 0.03f && x[1][1] == 1.1f + 0.02f && token1[0] == 1.0f && view[1] == 2.1f;
    ok &= dwte[6] == 2.0f && dwte[7] == -2.0f && dwte[2] == 1.0f && dwte[4] == 0.0f;

    std::printf(ok ? "embedding demo: ok\n" : "embedding demo: FAILED\n");
    return ok ? 0 : 1;
}
