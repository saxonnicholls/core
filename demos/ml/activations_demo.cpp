//
//  activations_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/ml/activations.hpp>: a GELU table built by the compiler, one neuron's forward
//  and backward pass, and the tails where the textbook formulas fail. Exits non-zero if a result is not
//  what it documents.
//

#include <snicholls/core/ml/activations.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>

int main()
{
    std::printf("snicholls-core activations demo\n");
    namespace ml = snicholls::ml;
    namespace math = snicholls::math;
    bool ok = true;

    //! [activations]
    // A lookup table a kernel reads, computed by the compiler: GELU on [-4, 4] in steps of 1/2, in float.
    constexpr auto gelu_table = [] {
        std::array<float, 17> t{};
        for (std::size_t i = 0; i < t.size(); ++i)
            t[i] = ml::gelu(-4.0f + 0.5f * static_cast<float>(i));
        return t;
    }();
    static_assert(gelu_table[8] == 0.0f);

    // One neuron, y = sigmoid(w x + b), and the gradient of the loss (y - target)² / 2 in w, by the chain rule.
    // The backward pass uses the output it kept: sigmoid_derivative_from_output(y).
    constexpr double w = 0.8, b = -0.2, x = 1.5, target = 1.0;
    constexpr double y = ml::sigmoid(w * x + b);
    constexpr double dloss_dw = (y - target) * ml::sigmoid_derivative_from_output(y) * x;

    // The same backward pass for the other activations: each comes with its derivative.
    constexpr double grads[] = {ml::gelu_derivative(1.0), ml::gelu_tanh_derivative(1.0), ml::silu_derivative(1.0),
                                ml::relu_derivative(1.0), ml::relu_squared_derivative(1.0), ml::softcap_derivative(1.0, 15.0),
                                ml::sigmoid_derivative(1.0), ml::tanh_derivative_from_output(math::tanh(1.0))};
    //! [activations]

    std::printf("  gelu table: gelu(-4) %.3g, gelu(-1) %.6f, gelu(1) %.6f, gelu(4) %.6f\n", static_cast<double>(gelu_table[0]),
                static_cast<double>(gelu_table[6]), static_cast<double>(gelu_table[10]), static_cast<double>(gelu_table[16]));
    ok &= std::abs(gelu_table[10] - 0.8413447f) < 1e-6f && std::abs(gelu_table[6] + 0.1586553f) < 1e-6f;

    std::printf("  neuron: y %.6f, dloss/dw %.6f\n", y, dloss_dw);
    ok &= math::near(y, 0.7310585786300049, 1e-15) && math::near(dloss_dw, -0.07931563917640007, 1e-15);

    std::printf("  at 1: gelu' %.6f, gelu_tanh' %.6f, silu' %.6f, relu' %.0f, relu²' %.0f, softcap' %.6f, sigmoid' %.6f, tanh' %.6f\n",
                grads[0], grads[1], grads[2], grads[3], grads[4], grads[5], grads[6], grads[7]);
    ok &= math::near(grads[0], 1.0833154705876864, 1e-12) && grads[3] == 1.0 && grads[4] == 2.0 && math::near(grads[7], 0.41997434161402614, 1e-15);

    // Values at 1 of the forward functions, and the tails where the textbook formulas overflow or cancel
    std::printf("  at 1: gelu %.6f, gelu_tanh %.6f, silu %.6f, relu² %.0f, softplus %.6f, log_sigmoid %.6f, softcap %.6f, logit(0.9) %.6f\n",
                ml::gelu(1.0), ml::gelu_tanh(1.0), ml::silu(1.0), ml::relu_squared(1.0), ml::softplus(1.0), ml::log_sigmoid(1.0),
                ml::softcap(1.0, 15.0), ml::logit(0.9));
    ok &= math::near(ml::softplus(1.0), 1.3132616875182228, 1e-15) && math::near(ml::logit(0.9), 2.1972245773362196, 1e-15);
    ok &= ml::relu(-3.0) == 0.0;

    std::printf("  tails: softplus(1000) %.1f (log(1 + e^1000) is inf), log_sigmoid(-1000) %.1f (log(0) is -inf), gelu(-30) %.3g\n",
                ml::softplus(1000.0), ml::log_sigmoid(-1000.0), ml::gelu(-30.0));
    ok &= ml::softplus(1000.0) == 1000.0 && ml::log_sigmoid(-1000.0) == -1000.0 && ml::gelu(-30.0) < 0.0;

    std::printf(ok ? "activations demo: ok\n" : "activations demo: FAILED\n");
    return ok ? 0 : 1;
}
