//
//  short_conv_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/ml/short_conv.hpp>: Mamba's causal depthwise convolution of four taps over a prompt, its
//  state handed to decoding, a token decoded against the same tokens run as one sequence, and the backward pass's
//  gradients. Exits non-zero if a result is not what it documents.
//

#include <snicholls/core/linalg/reference.hpp>
#include <snicholls/core/ml/short_conv.hpp>

#include <array>
#include <cstddef>
#include <cstdio>
#include <span>

int main()
{
    std::printf("snicholls-core short convolution demo\n");
    namespace ml = snicholls::ml;
    constexpr auto& backend = snicholls::linalg::reference;
    bool ok = true;

    //! [short_conv]
    // Two channels, four taps (oldest first): channel 0 averages its last four inputs, channel 1 takes the difference
    constexpr std::size_t channels = 2, width = 4, n = 5;
    const std::array<float, channels * width> taps{0.25f, 0.25f, 0.25f, 0.25f, 0.0f, 0.0f, -1.0f, 1.0f};
    const std::array<float, channels> bias{0.0f, 0.5f};
    const ml::short_conv_weights<float> w{taps, channels, width, bias};
    const std::array<float, n * channels> prompt{4, 1, 8, 2, 4, 4, 0, 8, 4, 16};
    std::array<float, n * channels> y{};
    std::array<float, ml::short_conv_state_size(channels, width)> state{};   // zeros: a new sequence
    ml::short_conv_forward(backend, w, std::span<const float>(prompt), std::span<float>(y), 1, std::span<float>(state));

    // Decoding: the state holds the prompt's last three tokens
    const std::array<float, channels> next{8, 32};
    std::array<float, channels> out{};
    ml::short_conv_step(w, std::span<const float>(next), std::span<float>(out), std::span<float>(state));

    // Training: the gradient of Σ y in the taps, the bias and the inputs
    const std::array<float, n * channels> dy{1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    std::array<float, n * channels> dx{};
    std::array<float, channels * width> dw{};
    std::array<float, channels> db{};
    ml::short_conv_backward(backend, w, std::span<const float>(prompt), std::span<const float>(dy), std::span<float>(dx), std::span<float>(dw),
                            std::span<float>(db));
    //! [short_conv]

    std::printf("  outputs: average %.2f, difference %.2f at the last prompt token\n", static_cast<double>(y[8]), static_cast<double>(y[9]));
    std::printf("  decoded: average %.2f, difference %.2f\n", static_cast<double>(out[0]), static_cast<double>(out[1]));
    ok &= y[8] == (8 + 4 + 0 + 4) / 4.0f && y[9] == 16 - 8 + 0.5f;
    ok &= out[0] == (4 + 0 + 4 + 8) / 4.0f && out[1] == 32 - 16 + 0.5f;
    ok &= db[0] == 5 && db[1] == 5 && dw[3] == 4 + 8 + 4 + 0 + 4 && dx[0] == 1.0f;   // x_0 feeds four outputs of 0.25
    std::printf(ok ? "short convolution demo: ok\n" : "short convolution demo: FAILED\n");
    return ok ? 0 : 1;
}
