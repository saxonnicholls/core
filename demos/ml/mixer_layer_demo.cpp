//
//  mixer_layer_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/ml/mixer_layer.hpp>: a Mamba-2 layer and a gated-delta-rule layer laid out in one buffer
//  each; a prompt prefilled through the batched pass with its state kept, then decoded token by token; the layer
//  trained by gradient descent to copy the previous token, its loss falling; a hybrid stack's cache indices. Exits
//  non-zero if a result is not what it documents.
//

#include <snicholls/core/linalg/reference.hpp>
#include <snicholls/core/ml/mixer_layer.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <span>
#include <vector>

namespace {

namespace ml = snicholls::ml;
constexpr auto& backend = snicholls::linalg::reference;

std::vector<double> wave(std::size_t n, double f, double scale)
{
    std::vector<double> v(n);
    for (std::size_t i = 0; i < n; ++i)
        v[i] = scale * std::sin(f * static_cast<double>(i) + 0.3);
    return v;
}

template <ml::transformer_config C>
bool prefill_then_decode(const char* name)
{
    //! [mixer_layer]
    // One buffer of parameters, laid out as the layer's weights
    std::vector<double> params = wave(ml::mixer_parameter_count(C), 0.37, 0.3);
    const auto w = ml::mixer_weights_in<C>(std::span<const double>(params));

    // Prefill a prompt of 6 tokens with the batched pass, keeping the state it ends in
    constexpr std::size_t prompt = 6, d = C.d_model;
    const std::vector<double> tokens = wave((prompt + 2) * d, 0.11, 1.0);
    std::vector<double> act(ml::mixer_activation_values(C, prompt, 1)), out(prompt * d), state(ml::mixer_state_values(C), 0.0);
    ml::mixer_forward_sequence<C>(backend, w, std::span<const double>(tokens).first(prompt * d),
                                  ml::mixer_activations_in<C>(std::span<double>(act), prompt, 1), std::span<double>(out), 1,
                                  std::span<double>(state));

    // Decode the next two tokens from that state: a fixed number of values, whatever the prompt's length. A stack keeps
    // every mixer layer's state in a recurrent_cache (here a stack of one)
    const ml::recurrent_cache<double> cache{std::span<double>(state), ml::mixer_state_values(C)};
    std::vector<double> next(d), after(d);
    ml::mixer_forward<C>(w, std::span<const double>(tokens).subspan(prompt * d, d), cache.layer(0), std::span<double>(next));
    ml::mixer_forward<C>(w, std::span<const double>(tokens).subspan((prompt + 1) * d, d), cache.layer(0), std::span<double>(after));
    //! [mixer_layer]

    // The same eight tokens as one batched pass
    std::vector<double> act8(ml::mixer_activation_values(C, prompt + 2, 1)), out8((prompt + 2) * d);
    ml::mixer_forward_sequence<C>(backend, w, std::span<const double>(tokens), ml::mixer_activations_in<C>(std::span<double>(act8), prompt + 2, 1),
                                  std::span<double>(out8));
    double worst = 0;
    for (std::size_t i = 0; i < d; ++i)
        worst = std::fmax(worst, std::fmax(std::fabs(next[i] - out8[prompt * d + i]), std::fabs(after[i] - out8[(prompt + 1) * d + i])));
    std::printf("  %s: %zu parameters, a decoding state of %zu numbers; decoded against batched: %.1e\n", name, params.size(), state.size(), worst);
    return worst < 1e-12;
}

} // namespace

int main()
{
    std::printf("snicholls-core mixer layer demo\n");
    bool ok = prefill_then_decode<ml::tiny_mamba2>("Mamba-2");
    constexpr ml::transformer_config gdn = [] {
        auto c = ml::tiny_gated_deltanet;
        c.hybrid_period = c.hybrid_attention_offset = 0;
        return c;
    }();
    ok &= prefill_then_decode<gdn>("gated delta rule");

    //! [mixer_layer_training]
    // Train the Mamba-2 layer to output each token's predecessor: 30 steps of gradient descent on half the squared error
    constexpr ml::transformer_config C = ml::tiny_mamba2;
    constexpr std::size_t n = 8, seqs = 2, rows = n * seqs, d = C.d_model;
    std::vector<double> params = wave(ml::mixer_parameter_count(C), 0.37, 0.3), grads(params.size());
    const std::vector<double> xs = wave(rows * d, 0.23, 1.0);
    std::vector<double> target(rows * d, 0.0);
    for (std::size_t q = 0; q < seqs; ++q)
        for (std::size_t t = 1; t < n; ++t)
            for (std::size_t i = 0; i < d; ++i)
                target[(q * n + t) * d + i] = xs[(q * n + t - 1) * d + i];
    std::vector<double> act(ml::mixer_activation_values(C, rows, seqs)), out(rows * d), dout(rows * d), dxs(rows * d),
        work(ml::mixer_gradient_work<C, double>(rows, seqs));
    double first = 0, last = 0;
    for (int step = 0; step < 30; ++step) {
        const auto w = ml::mixer_weights_in<C>(std::span<const double>(params));
        const auto a = ml::mixer_activations_in<C>(std::span<double>(act), rows, seqs);
        ml::mixer_forward_sequence<C>(backend, w, std::span<const double>(xs), a, std::span<double>(out), seqs);
        double loss = 0;
        for (std::size_t i = 0; i < out.size(); ++i) {
            dout[i] = out[i] - target[i];
            loss += 0.5 * dout[i] * dout[i];
        }
        std::fill(grads.begin(), grads.end(), 0.0);
        ml::mixer_backward<C>(backend, w, std::span<const double>(xs), a, std::span<const double>(dout), std::span<double>(dxs),
                              ml::mixer_gradients_in<C>(std::span<double>(grads), std::span<double>(work)), seqs);
        for (std::size_t i = 0; i < params.size(); ++i)
            params[i] -= 0.01 * grads[i];
        (step == 0 ? first : last) = loss;
    }
    //! [mixer_layer_training]
    std::printf("  training to copy the previous token: loss %.4f -> %.4f\n", first, last);
    ok &= last < 0.7 * first;

    // A Qwen3-Next stack: layer 7 is its second attention layer, layer 6 its fifth mixer layer
    std::printf("  hybrid: layer 7 is attention slot %zu, layer 6 mixer slot %zu\n", ml::hybrid_attention_index(ml::tiny_gated_deltanet, 7),
                ml::hybrid_mixer_index(ml::tiny_gated_deltanet, 6));
    ok &= ml::hybrid_attention_index(ml::tiny_gated_deltanet, 7) == 1 && ml::hybrid_mixer_index(ml::tiny_gated_deltanet, 6) == 5;
    std::printf(ok ? "mixer layer demo: ok\n" : "mixer layer demo: FAILED\n");
    return ok ? 0 : 1;
}
