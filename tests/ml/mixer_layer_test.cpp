//
//  mixer_layer_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/ml/mixer_layer.hpp>: Mamba-2's and the gated delta rule's layers: the backward pass against
//  central differences in double over every parameter and input (two sequences, a short last chunk); decoding token by
//  token against the batched pass at every position; a prompt prefilled in pieces through the states against the
//  whole prompt; the buffers' sizes at compile time and their layouts; the hybrid stack's cache indices.
//

#include <snicholls/core/ml/mixer_layer.hpp>

#include <snicholls/core/linalg/reference.hpp>

#include "../support/mixer_fixture.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <span>
#include <vector>

namespace {

namespace ml = snicholls::ml;
using snicholls::core_testing::central_difference;
using snicholls::core_testing::max_relative_difference;
using snicholls::core_testing::uniform;
using snicholls::core_testing::weighted;
constexpr auto& ref = snicholls::linalg::reference;

constexpr ml::transformer_config mamba = ml::tiny_mamba2;
constexpr ml::transformer_config delta = [] {   // two value heads sharing one key head of 4; chunks of 4
    auto c = ml::tiny_gated_deltanet;
    c.hybrid_period = 0;
    c.hybrid_attention_offset = 0;
    return c;
}();

// tiny_mamba2: in_proj (2·8 + 2·4 + 2) x 8, conv 16 x 4 and its bias, dt_bias, A_log, D, the norm (8), out_proj 8 x 8
static_assert(ml::mixer_parameter_count(mamba) == 26 * 8 + 64 + 16 + 3 * 2 + 8 + 64);
// the rule: in_proj (2·4 + 2·8 + 2·2) x 8, conv (2·4 + 8) x 4, dt_bias, A_log, the norm (4), out_proj 8 x 8
static_assert(ml::mixer_parameter_count(delta) == 28 * 8 + 64 + 2 * 2 + 4 + 64);
static_assert(ml::mixer_state_values(mamba) == 3 * 16 + 32 && ml::mixer_state_values(delta) == 3 * 16 + 32);
static_assert(ml::mixer_activation_values(mamba, 0, 1) == 32 && ml::mixer_gradient_work<mamba, double>(0, 0) == 2 + ml::ssd_backward_work<mamba, double>());
static_assert(noexcept(ml::mixer_parameter_count(mamba)) && noexcept(ml::mixer_state_values(mamba)) && noexcept(ml::mixer_activation_values(mamba, 4, 1))
              && noexcept(ml::mixer_gradient_work<mamba, double>(4, 1)) && noexcept(ml::hybrid_attention_index(delta, 3)));
// Qwen3-Next's stack: layers 0-2 are mixers 0-2, layer 3 is attention 0, layer 4 mixer 3
static_assert(ml::hybrid_attention_index(ml::tiny_gated_deltanet, 3) == 0 && ml::hybrid_mixer_index(ml::tiny_gated_deltanet, 4) == 3
              && ml::hybrid_attention_index(ml::tiny_gated_deltanet, 8) == 2);

template <ml::transformer_config C>
struct layer
{
    std::vector<double> params;
    layer() : params(uniform<double>(ml::mixer_parameter_count(C), 1, -0.5, 0.5))
    {
        // Gates in their working range: A_log near 0, dt_bias and the norm's weight positive
        auto w = ml::mixer_weights_in<C>(std::span<const double>(params));
        const auto at = [&](std::span<const double> s) { return static_cast<std::size_t>(s.data() - params.data()); };
        for (std::size_t i = 0; i < w.norm.size(); ++i)
            params[at(w.norm) + i] += 1.0;
    }
    auto weights() const { return ml::mixer_weights_in<C>(std::span<const double>(params)); }
};

template <ml::transformer_config C>
std::vector<double> forward(const layer<C>& l, const std::vector<double>& xs, std::size_t seqs, std::vector<double>* act_out = nullptr,
                            std::span<double> states = {})
{
    const std::size_t rows = xs.size() / C.d_model;
    std::vector<double> values(ml::mixer_activation_values(C, rows, seqs)), out(xs.size());
    const auto act = ml::mixer_activations_in<C>(std::span<double>(values), rows, seqs);
    ml::mixer_forward_sequence<C>(ref, l.weights(), std::span<const double>(xs), act, std::span<double>(out), seqs, states);
    if (act_out)
        *act_out = values;
    return out;
}

template <ml::transformer_config C>
void backward_matches_central_differences()
{
    constexpr std::size_t n = 6, seqs = 2, rows = n * seqs;
    layer<C> l;
    auto xs = uniform<double>(rows * C.d_model, 2);
    const auto wo = uniform<double>(rows * C.d_model, 3);
    const auto loss = [&] { return weighted(forward<C>(l, xs, seqs), wo); };
    std::vector<double> values;
    forward<C>(l, xs, seqs, &values);
    const auto act = ml::mixer_activations_in<C>(std::span<double>(values), rows, seqs);
    std::vector<double> grads(l.params.size()), work(ml::mixer_gradient_work<C, double>(rows, seqs)), dxs(xs.size());
    ml::mixer_backward<C>(ref, l.weights(), std::span<const double>(xs), act, std::span<const double>(wo), std::span<double>(dxs),
                          ml::mixer_gradients_in<C>(std::span<double>(grads), std::span<double>(work)), seqs);
    for (std::size_t i = 0; i < xs.size(); ++i)
        EXPECT_NEAR(dxs[i], central_difference(xs, i, loss), 1e-7) << "input " << i;
    for (std::size_t i = 0; i < l.params.size(); ++i)
        EXPECT_NEAR(grads[i], central_difference(l.params, i, loss), 1e-7) << "parameter " << i;
}

TEST(MlMixerLayer, Mamba2BackwardMatchesCentralDifferences)
{
    backward_matches_central_differences<mamba>();
}

TEST(MlMixerLayer, GatedDeltaRuleBackwardMatchesCentralDifferences)
{
    backward_matches_central_differences<delta>();
}

template <ml::transformer_config C>
void decoding_is_the_batched_pass()
{
    constexpr std::size_t n = 9;
    const layer<C> l;
    const auto xs = uniform<double>(n * C.d_model, 4);
    const auto whole = forward<C>(l, xs, 1);
    std::vector<double> state(ml::mixer_state_values(C), 0.0), out(C.d_model);
    for (std::size_t t = 0; t < n; ++t) {
        ml::mixer_forward<C>(l.weights(), std::span<const double>(xs).subspan(t * C.d_model, C.d_model), std::span<double>(state),
                             std::span<double>(out));
        EXPECT_LT(max_relative_difference(out, std::span<const double>(whole).subspan(t * C.d_model, C.d_model)), 1e-13) << t;
    }
}

TEST(MlMixerLayer, DecodingIsTheBatchedPassAtEveryPosition)
{
    decoding_is_the_batched_pass<mamba>();
    decoding_is_the_batched_pass<delta>();
}

template <ml::transformer_config C>
void prefill_in_pieces()
{
    constexpr std::size_t n = 10;
    const layer<C> l;
    const auto xs = uniform<double>(n * C.d_model, 5);
    const auto whole = forward<C>(l, xs, 1);
    std::vector<double> state(ml::mixer_state_values(C), 0.0), pieces;
    for (const auto& [from, to] : {std::pair<std::size_t, std::size_t>{0, 3}, {3, 4}, {4, 10}}) {
        const std::vector<double> part(xs.begin() + static_cast<std::ptrdiff_t>(from * C.d_model), xs.begin() + static_cast<std::ptrdiff_t>(to * C.d_model));
        const auto out = forward<C>(l, part, 1, nullptr, std::span<double>(state));
        pieces.insert(pieces.end(), out.begin(), out.end());
    }
    EXPECT_LT(max_relative_difference(pieces, whole), 1e-13);
}

TEST(MlMixerLayer, APromptInPiecesIsTheWholePrompt)
{
    prefill_in_pieces<mamba>();
    prefill_in_pieces<delta>();
}

TEST(MlMixerLayer, TheLayoutsCoverTheirBuffers)
{
    const layer<mamba> m;
    const auto w = m.weights();
    EXPECT_EQ(w.in_proj.data.data(), m.params.data());
    EXPECT_EQ(w.out_proj.data.data() + w.out_proj.data.size(), m.params.data() + m.params.size());
    EXPECT_EQ(w.conv.bias.size(), 16u);
    std::vector<double> values(ml::mixer_activation_values(delta, 8, 2));
    const auto a = ml::mixer_activations_in<delta>(std::span<double>(values), 8, 2);
    EXPECT_EQ(a.work.data() + a.work.size(), values.data() + values.size());
    EXPECT_TRUE(a.gated.empty());
    EXPECT_EQ(a.checkpoints.size(), 2u * 1u * 32u);
}

} // namespace
