//
//  mlp_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/ml/mlp.hpp>: one token through a ReLU² MLP and a SwiGLU MLP, and the activation each
//  kind applies. Exits non-zero if a result is not what it documents.
//

#include <snicholls/core/ml/mlp.hpp>

#include <snicholls/core/linalg/reference.hpp>

#include <array>
#include <cmath>
#include <cstdio>

int main()
{
    std::printf("snicholls-core mlp demo\n");
    namespace ml = snicholls::ml;
    using W = ml::dense_weights<float>;
    bool ok = true;

    //! [mlp]
    constexpr ml::transformer_config relu2{.d_model = 2, .n_heads = 1, .n_kv_heads = 1, .head_dim = 2, .d_ff = 2, .mlp = ml::mlp_kind::relu_squared};
    constexpr ml::transformer_config swiglu{.d_model = 2, .n_heads = 1, .n_kv_heads = 1, .head_dim = 2, .d_ff = 2, .mlp = ml::mlp_kind::swiglu};
    const std::array<float, 4> up{1, 0, 0, 1}, gate{1, 1, -1, -1}, down{1, 0, 1, 1};   // 2 x 2 each
    const std::array<float, 2> x{2.0f, -1.0f};

    std::array<float, 2> a{}, b{};
    ml::mlp_forward<relu2>(ml::mlp_weights<W>{W{up, 2, 2}, W{}, W{down, 2, 2}}, x, a);              // down(relu(up x)²)
    ml::mlp_forward<swiglu>(ml::mlp_weights<W>{W{up, 2, 2}, W{gate, 2, 2}, W{down, 2, 2}}, x, b);   // down(silu(gate x) · up x)
    //! [mlp]

    //! [mlp_backward]
    // Training: the forward pass saves its pre-activations, the backward pass reads them back
    std::array<float, 2> saved_up{}, y{};
    const ml::mlp_weights<W> relu_w{W{up, 2, 2}, W{}, W{down, 2, 2}};
    ml::mlp_forward<relu2>(relu_w, x, y, ml::mlp_activations<float>{saved_up});
    const std::array<float, 2> dy{1.0f, 0.0f};   // the gradient in output 0 only
    std::array<float, 2> dx{};
    std::array<float, 4> dup{}, ddown{};
    ml::mlp_backward<relu2>(relu_w, x, ml::mlp_activations<float>{saved_up}, dy, dx, ml::mlp_gradients<float>{dup, {}, ddown});

    // A mixture of experts: route to the 2 best of 4, then take a gradient in the weights back to the router
    const std::array<float, 4> router{0.5f, 2.0f, -1.0f, 1.0f};
    const auto route = ml::moe_gate<2>(router);
    std::array<float, 4> drouter{};
    ml::moe_gate_backward(route, std::array<float, 2>{1.0f, 0.0f}, drouter);   // more weight on the best expert
    //! [mlp_backward]

    const float s = 1.0f / (1.0f + std::exp(-1.0f)), t = -1.0f / (1.0f + std::exp(1.0f));   // silu(1), silu(-1)
    std::printf("  ReLU²: [%.4f %.4f]   SwiGLU: [%.4f %.4f]\n", static_cast<double>(a[0]), static_cast<double>(a[1]), static_cast<double>(b[0]),
                static_cast<double>(b[1]));
    std::printf("  gelu(1) %.4f, gelu_tanh(1) %.4f\n", static_cast<double>(ml::mlp_activation<ml::mlp_kind::gelu>(1.0f)),
                static_cast<double>(ml::mlp_activation<ml::mlp_kind::gelu_tanh>(1.0f)));
    ok &= a[0] == 4.0f && a[1] == 4.0f;   // h = [relu(2)², relu(-1)²] = [4, 0]
    ok &= std::abs(b[0] - 2.0f * s) < 1e-6f && std::abs(b[1] - (2.0f * s - t)) < 1e-6f;
    ok &= std::abs(ml::mlp_activation<ml::mlp_kind::gelu>(1.0f) - 0.8413447f) < 1e-6f;
    // y0 = h0 = (up x)0² with up x = (2, -1): dy0/dx0 = 2 · 2 = 4, dx1 = 0; d down[0][0] = h0 = 4; d up row 0 = 4 x
    std::printf("  backward: dx [%.1f %.1f], ddown[0] %.1f, relu²'(2) %.1f\n", static_cast<double>(dx[0]), static_cast<double>(dx[1]),
                static_cast<double>(ddown[0]), static_cast<double>(ml::mlp_activation_derivative<ml::mlp_kind::relu_squared>(2.0f)));
    ok &= y == a && dx[0] == 4.0f && dx[1] == 0.0f && ddown[0] == 4.0f && dup[0] == 8.0f && dup[1] == -4.0f && dup[2] == 0.0f;   // d up[0] = 4 x
    // Experts 1 and 3 are chosen; raising the best one's weight raises its logit and lowers the other's, equally
    std::printf("  moe: experts %zu, %zu weights %.4f %.4f; d router [%.4f %.4f %.4f %.4f]\n", route.expert[0], route.expert[1],
                static_cast<double>(route.weight[0]), static_cast<double>(route.weight[1]), static_cast<double>(drouter[0]),
                static_cast<double>(drouter[1]), static_cast<double>(drouter[2]), static_cast<double>(drouter[3]));
    ok &= route.count == 2 && route.expert[0] == 1 && route.expert[1] == 3 && drouter[0] == 0.0f && drouter[2] == 0.0f;
    ok &= drouter[1] > 0.0f && std::abs(drouter[1] + drouter[3]) < 1e-7f;

    //! [mlp_sequence]
    // A sequence of tokens at once: the products through a linalg backend, the hidden rows saved for the backward
    const std::array<float, 4> xs{2.0f, -1.0f, 2.0f, -1.0f};   // the same token twice
    std::array<float, 4> ys{}, sup{}, shidden{}, dxs{}, dup2{}, ddown2{}, scratch{};
    const ml::mlp_activations<float> save{sup, {}, shidden};
    ml::mlp_forward_sequence<relu2>(snicholls::linalg::reference, relu_w, xs, ys, save);
    const std::array<float, 4> dys{1.0f, 0.0f, 1.0f, 0.0f};
    ml::mlp_backward_sequence<relu2>(snicholls::linalg::reference, relu_w, xs, save, dys, dxs,
                                     ml::mlp_gradients<float>{dup2, {}, ddown2, {}, {}, {}, scratch});
    //! [mlp_sequence]
    std::printf("  sequence: [%.1f %.1f] twice; d up[0] %.1f, twice the per-token %.1f\n", static_cast<double>(ys[0]),
                static_cast<double>(ys[1]), static_cast<double>(dup2[0]), static_cast<double>(dup[0]));
    ok &= ys[0] == y[0] && ys[1] == y[1] && ys[2] == y[0] && ys[3] == y[1] && dxs[0] == dx[0] && dxs[2] == dx[0];
    ok &= dup2[0] == 2 * dup[0] && ddown2[0] == 2 * ddown[0];   // two identical tokens: twice the weights' gradients

    std::printf(ok ? "mlp demo: ok\n" : "mlp demo: FAILED\n");
    return ok ? 0 : 1;
}
