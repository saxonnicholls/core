//
//  self_attention_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/ml/self_attention.hpp>: one causal head over five tokens, the same head with a sliding
//  window, with RoPE, and the backward pass of a squared-error loss. Exits non-zero if a result is not
//  what it documents.
//

#include <snicholls/core/ml/self_attention.hpp>

#include <snicholls/core/ml/rope.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>

int main()
{
    std::printf("snicholls-core self_attention demo\n");
    namespace ml = snicholls::ml;
    bool ok = true;

    //! [self_attention]
    // Five tokens of 4 channels; a head of 2 channels
    constexpr std::size_t d_model = 4, d_head = 2, n = 5;
    std::array<float, n * d_model> x{};
    for (std::size_t i = 0; i < x.size(); ++i)
        x[i] = std::sin(0.7f * static_cast<float>(i));

    // Wq and Wk read channels 0 and 1; Wv reads channels 2 and 3
    const std::array<float, d_head * d_model> wq{1, 0, 0, 0, 0, 1, 0, 0};
    const std::array<float, d_head * d_model> wv{0, 0, 1, 0, 0, 0, 0, 1};
    const ml::self_attention_weights<ml::dense_weights<float>> w{{wq, d_head, d_model}, {wq, d_head, d_model},
                                                                 {wv, d_head, d_model}};

    // The caller owns what a backward pass would read: Q, K, V and each position's log-sum-exp
    std::array<float, n * d_head> q{}, k{}, v{}, out{};
    std::array<float, n> lse{};
    const ml::self_attention_activations<float> act{q, k, v, lse};

    // Causal: position t attends to tokens 0..t
    ml::self_attention_head<d_model, d_head>(w, x, act, out);

    // A sliding window of 2: position t attends to t - 1 and t
    std::array<float, n * d_head> windowed{};
    ml::self_attention_head<d_model, d_head>(w, x, act, windowed, 2);

    // RoPE from a table the compiler builds
    static constexpr ml::rope_table<float, 8, d_head> table{};
    std::array<float, n * d_head> rotated{};
    ml::self_attention_head<d_model, d_head>(w, x, act, rotated, 0, table);
    //! [self_attention]

    //! [self_attention_backward]
    // Training: run the causal head again (act now holds its Q, K, V and log-sum-exps), then for the loss
    // L = 1/2 |out - target|^2 the gradient in out is out - target
    ml::self_attention_head<d_model, d_head>(w, x, act, out);
    std::array<float, n * d_head> dout{};
    for (std::size_t i = 0; i < dout.size(); ++i)
        dout[i] = out[i] - 0.25f;   // a target of 0.25 everywhere

    // Q, K, V gradients are written; the weights', biases' and tokens' are added into, so zero them first
    std::array<float, n * d_head> dq{}, dk{}, dv{};
    std::array<float, d_head * d_model> dwq{}, dwk{}, dwv{};
    std::array<float, n * d_model> dx{};
    const ml::self_attention_gradients<float> grads{dq, dk, dv, dwq, dwk, dwv};
    ml::self_attention_head_backward<d_model, d_head>(w, x, act, out, dout, dx, grads);

    // One step of gradient descent on Wv lowers the loss
    std::array<float, d_head * d_model> wv2 = wv;
    for (std::size_t i = 0; i < wv2.size(); ++i)
        wv2[i] -= 0.5f * dwv[i];
    const ml::self_attention_weights<ml::dense_weights<float>> stepped{w.q, w.k, {wv2, d_head, d_model}};
    std::array<float, n * d_head> after{};
    ml::self_attention_head<d_model, d_head>(stepped, x, act, after);
    //! [self_attention_backward]
    auto loss = [](const std::array<float, n * d_head>& o) {
        float l = 0;
        for (float e : o)
            l += 0.5f * (e - 0.25f) * (e - 0.25f);
        return l;
    };
    std::printf("  backward: loss %.5f, after one step on Wv %.5f\n", double(loss(out)), double(loss(after)));
    ok &= loss(after) < loss(out);
    // Position 0's output is its own value, so Wv's gradient at row 0 includes token 0's channels; the tokens get one too
    ok &= dwv[0] != 0.0f && std::any_of(dx.begin(), dx.end(), [](float g) { return g != 0.0f; });

    for (std::size_t t = 0; t < n; ++t)
        std::printf("  position %zu: causal (%+.4f, %+.4f)  window 2 (%+.4f, %+.4f)  rope (%+.4f, %+.4f)\n", t,
                    double(out[t * d_head]), double(out[t * d_head + 1]), double(windowed[t * d_head]),
                    double(windowed[t * d_head + 1]), double(rotated[t * d_head]), double(rotated[t * d_head + 1]));

    // Position 0 sees only itself: its output is its own value, x[0][2..3], in every form
    for (const auto* o : {&out, &windowed, &rotated})
        ok &= (*o)[0] == x[2] && (*o)[1] == x[3];
    // Position 1 sees the same two keys causally and in a window of 2
    ok &= out[2] == windowed[2] && out[3] == windowed[3];
    // Every output is a convex combination of the values, so it lies within their range per channel
    for (std::size_t t = 0; t < n; ++t)
        for (std::size_t c = 0; c < d_head; ++c) {
            float lo = x[2 + c], hi = x[2 + c];
            for (std::size_t j = 0; j <= t; ++j) {
                lo = std::fmin(lo, x[j * d_model + 2 + c]);
                hi = std::fmax(hi, x[j * d_model + 2 + c]);
            }
            ok &= out[t * d_head + c] >= lo - 1e-6f && out[t * d_head + c] <= hi + 1e-6f;
        }

    std::printf("%s\n", ok ? "ok" : "FAILED");
    return ok ? 0 : 1;
}
