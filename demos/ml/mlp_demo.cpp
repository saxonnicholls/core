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

    const float s = 1.0f / (1.0f + std::exp(-1.0f)), t = -1.0f / (1.0f + std::exp(1.0f));   // silu(1), silu(-1)
    std::printf("  ReLU²: [%.4f %.4f]   SwiGLU: [%.4f %.4f]\n", static_cast<double>(a[0]), static_cast<double>(a[1]), static_cast<double>(b[0]),
                static_cast<double>(b[1]));
    std::printf("  gelu(1) %.4f, gelu_tanh(1) %.4f\n", static_cast<double>(ml::mlp_activation<ml::mlp_kind::gelu>(1.0f)),
                static_cast<double>(ml::mlp_activation<ml::mlp_kind::gelu_tanh>(1.0f)));
    ok &= a[0] == 4.0f && a[1] == 4.0f;   // h = [relu(2)², relu(-1)²] = [4, 0]
    ok &= std::abs(b[0] - 2.0f * s) < 1e-6f && std::abs(b[1] - (2.0f * s - t)) < 1e-6f;
    ok &= std::abs(ml::mlp_activation<ml::mlp_kind::gelu>(1.0f) - 0.8413447f) < 1e-6f;
    std::printf(ok ? "mlp demo: ok\n" : "mlp demo: FAILED\n");
    return ok ? 0 : 1;
}
