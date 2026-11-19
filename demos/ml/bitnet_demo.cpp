//
//  bitnet_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/ml/bitnet.hpp>: a BitNet b1.58 layer. Float weights become ternary with one absmean
//  scale; a token's activations become int8 with one absmax scale; BitLinear's integer product is compared with the
//  float layer it replaces, with and without its RMSNorm (SubLN); then one quantisation-aware training step's
//  gradient through the straight-through estimator. Exits non-zero if a result is not what it documents.
//

#include <snicholls/core/ml/bitnet.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>

int main()
{
    std::printf("snicholls-core bitnet demo\n");
    namespace ml = snicholls::ml;
    bool ok = true;
    constexpr std::size_t rows = 8, cols = 64;

    //! [bitnet]
    // A layer's float weights, and one token
    std::array<float, rows * cols> w{};
    std::array<float, cols> x{};
    for (std::size_t i = 0; i < w.size(); ++i)
        w[i] = 0.05f * std::sin(1.3f * float(i));
    for (std::size_t j = 0; j < cols; ++j)
        x[j] = std::cos(0.2f * float(j));

    // Absmean: ternary codes and one scale for the matrix
    std::array<std::int8_t, rows * cols> t{};
    const float gamma = ml::absmean_quantize(w, t);
    const ml::bitlinear_weights layer{t, rows, cols, gamma};

    // BitLinear: int8 activations per token, an integer sum per output, one multiplication by the scales
    std::array<float, rows> y{};
    std::array<std::int8_t, cols> codes{};
    ml::bitlinear(layer, x, y, codes);

    // As BitNet places it: an RMSNorm (SubLN) before the layer's activation quantisation
    std::array<float, cols> gain{}, normed{};
    gain.fill(1.0f);
    std::array<float, rows> y_normed{};
    ml::bitlinear_normed(layer, gain, x, y_normed, normed, codes);
    //! [bitnet]

    int zeros = 0;
    for (const auto c : t)
        zeros += c == 0;
    std::printf("  absmean scale %.5f, %d of %zu weights zero\n", double(gamma), zeros, t.size());
    double worst = 0;
    for (std::size_t i = 0; i < rows; ++i) {
        double ternary = 0;   // the layer the codes stand for, in float
        for (std::size_t j = 0; j < cols; ++j)
            ternary += double(t[i * cols + j]) * double(gamma) * double(x[j]);
        worst = std::fmax(worst, std::fabs(ternary - double(y[i])));
    }
    std::printf("  BitLinear y[0] %.5f; largest difference from the float ternary layer %.2e (activation rounding)\n", double(y[0]), worst);
    ok &= worst < 0.01;
    ok &= zeros > 0 && zeros < int(t.size());
    std::printf("  with SubLN y[0] %.5f\n", double(y_normed[0]));

    // Quantisation-aware training: the forward uses fake-quantised weights, the backward passes the gradient straight through
    std::array<float, rows * cols> fake{}, dw{}, dout{};
    ml::absmean_fake_quantize(w, fake);
    dout.fill(0.5f);
    ml::ste_backward(dout, w, 1.0f / gamma, -1, 1, dw);
    ok &= dw == dout && fake[0] == float(t[0]) / (1.0f / gamma);
    ml::ste_backward(dout, w, 1.0f / gamma, -1, 1, dw, ml::ste::clipped);
    int passed = 0;
    for (const float g : dw)
        passed += g != 0.0f;
    std::printf("  straight-through: identity passes all %zu gradients, clipped passes %d\n", dw.size(), passed);
    ok &= passed > 0 && passed <= int(dw.size());

    std::printf(ok ? "bitnet demo: ok\n" : "bitnet demo: FAILED\n");
    return ok ? 0 : 1;
}
