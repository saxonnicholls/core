//
//  transformer_config_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

#include <snicholls/core/ml/transformer_config.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <limits>
#include <type_traits>

namespace {

namespace ml = snicholls::ml;

static_assert(std::is_trivially_copyable_v<ml::transformer_config>);

static_assert(ml::gpt2_small.q_size() == 768 && ml::gpt2_small.kv_size() == 768 && ml::gpt2_small.group() == 1);
static_assert(ml::gpt2_medium.q_size() == 1024 && ml::gpt2_medium.d_ff == 4 * ml::gpt2_medium.d_model && ml::gpt2_medium.group() == 1);
static_assert(ml::gpt2_large.q_size() == 1280 && ml::gpt2_large.d_ff == 4 * ml::gpt2_large.d_model && ml::gpt2_large.group() == 1);
static_assert(ml::llama3_8b.q_size() == 4096 && ml::llama3_8b.kv_size() == 1024 && ml::llama3_8b.group() == 4);
static_assert(ml::nanochat_d20.layout() == ml::rope_layout::half);

constexpr ml::transformer_config base{.d_model = 8, .n_heads = 2, .n_kv_heads = 1, .head_dim = 4, .d_ff = 16};
static_assert(ml::valid(base));

constexpr ml::transformer_config with(ml::transformer_config c, auto change)
{
    change(c);
    return c;
}

static_assert(!ml::valid(with(base, [](auto& c) { c.d_model = 0; })));
static_assert(!ml::valid(with(base, [](auto& c) { c.d_ff = 0; })));
static_assert(!ml::valid(with(base, [](auto& c) { c.n_kv_heads = 0; })));
static_assert(!ml::valid(with(base, [](auto& c) { c.n_heads = 3; c.n_kv_heads = 2; })));   // 3 heads cannot share 2
static_assert(!ml::valid(with(base, [](auto& c) { c.rope = ml::rope_kind::half; c.head_dim = 3; })));
static_assert(!ml::valid(with(base, [](auto& c) { c.rope = ml::rope_kind::half; c.rope_base = 1; })));
static_assert(!ml::valid(with(base, [](auto& c) { c.norm_eps = 0; })));
static_assert(!ml::valid(with(base, [](auto& c) { c.attention_softcap = -1; })));
static_assert(ml::valid(with(base, [](auto& c) { c.head_dim = 3; })));   // an odd head is fine without RoPE

// A mixture of experts: top_k from 1 to experts, the expert width set, the balance settings not negative; a dense
// configuration has every expert field 0, so the presets of dense models are unchanged
constexpr auto moe_base = with(base, [](auto& c) { c.experts = 4; c.top_k = 2; c.expert_d_ff = 8; });
static_assert(ml::valid(moe_base) && moe_base.moe() && !base.moe() && !ml::gpt2_small.moe() && !ml::llama3_8b.moe());
static_assert(ml::gpt2_small.experts == 0 && ml::llama3_8b.top_k == 0 && ml::nanochat_d20.expert_d_ff == 0 && ml::gpt2_large.shared_experts == 0);
static_assert(ml::valid(with(moe_base, [](auto& c) { c.top_k = 4; c.shared_experts = 2; c.scoring = ml::moe_scoring::sigmoid; })));
static_assert(!ml::valid(with(moe_base, [](auto& c) { c.top_k = 0; })));
static_assert(!ml::valid(with(moe_base, [](auto& c) { c.top_k = 5; })));
static_assert(!ml::valid(with(moe_base, [](auto& c) { c.expert_d_ff = 0; })));
static_assert(!ml::valid(with(moe_base, [](auto& c) { c.route_scale = 0; })));
static_assert(!ml::valid(with(moe_base, [](auto& c) { c.balance_loss_weight = -0.01; })));
static_assert(!ml::valid(with(moe_base, [](auto& c) { c.z_loss_weight = -1; })));
static_assert(!ml::valid(with(moe_base, [](auto& c) { c.bias_update_rate = -1; })));
static_assert(!ml::valid(with(base, [](auto& c) { c.top_k = 1; })));            // experts' fields without experts
static_assert(!ml::valid(with(base, [](auto& c) { c.shared_experts = 1; })));
static_assert(ml::valid(ml::tiny_moe) && ml::tiny_moe.experts == 4 && ml::tiny_moe.top_k == 2 && ml::tiny_moe.shared_experts == 1);
static_assert(ml::olmoe_1b_7b.experts == 64 && ml::olmoe_1b_7b.top_k == 8 && ml::olmoe_1b_7b.expert_d_ff == 1024
              && ml::olmoe_1b_7b.scoring == ml::moe_scoring::softmax);
static_assert(ml::route_scale_v<ml::tiny_moe, float> == 1.0f && ml::balance_loss_weight_v<ml::olmoe_1b_7b, double> == 0.01
              && ml::z_loss_weight_v<ml::olmoe_1b_7b, double> == 0.001 && ml::bias_update_rate_v<ml::tiny_moe, double> == 0);

// The configuration is a template argument
template <ml::transformer_config C>
constexpr std::size_t heads_v = C.n_heads;
static_assert(heads_v<ml::llama3_8b> == 32);

// Clang reads a double member of a template argument wrongly at run time; the readers evaluate it at compile time.
// This runs the reads where the bug showed (selected by a volatile, so the optimiser cannot fold them), so a
// regression is a failure.
template <ml::transformer_config C>
double read_at_run_time(int which)
{
    return which == 0 ? ml::norm_eps_v<C, double> : which == 1 ? ml::rope_base_v<C, double> : ml::attention_softcap_v<C, double>;
}

TEST(MlTransformerConfig, TheFloatingPointMembersReadCorrectlyAtRunTime)
{
    constexpr ml::transformer_config c{.d_model = 8, .n_heads = 1, .n_kv_heads = 1, .head_dim = 8, .d_ff = 9, .norm_eps = 1e-5,
                                       .rope = ml::rope_kind::half, .rope_base = 500, .window = 7, .attention_softcap = 5};
    volatile int which[3] = {0, 1, 2};
    EXPECT_EQ(read_at_run_time<c>(which[0]), 1e-5);
    EXPECT_EQ(read_at_run_time<c>(which[1]), 500.0);
    EXPECT_EQ(read_at_run_time<c>(which[2]), 5.0);
    EXPECT_EQ((ml::norm_eps_v<ml::nanochat_d20, float>), std::numeric_limits<float>::epsilon());
}

TEST(MlTransformerConfig, PresetsHaveTheirModelsShapes)
{
    EXPECT_EQ(ml::gpt2_small.norm, ml::norm_kind::layer);
    EXPECT_TRUE(ml::gpt2_small.bias);
    EXPECT_EQ(ml::llama3_8b.mlp, ml::mlp_kind::swiglu);
    EXPECT_EQ(ml::nanochat_d20.qk_norm, ml::qk_norm_kind::after_rope);
    EXPECT_EQ(ml::nanochat_d20.mlp, ml::mlp_kind::relu_squared);
    EXPECT_EQ(with(base, [](auto& c) { c.rope = ml::rope_kind::interleaved; }).layout(), ml::rope_layout::interleaved);
    EXPECT_EQ(ml::olmoe_1b_7b.d_model, 2048u);
    EXPECT_EQ(ml::olmoe_1b_7b.n_heads, 16u);
}

// The mixture of experts' readers, at run time, where Clang's bug would show
template <ml::transformer_config C>
double read_moe_at_run_time(int which)
{
    return which == 0 ? ml::route_scale_v<C, double> : which == 1 ? ml::balance_loss_weight_v<C, double> : ml::z_loss_weight_v<C, double>;
}

TEST(MlTransformerConfig, TheMixtureOfExpertsReadersReadCorrectlyAtRunTime)
{
    constexpr auto c = with(moe_base, [](auto& x) { x.route_scale = 2.5; x.balance_loss_weight = 0.01; x.z_loss_weight = 0.001; });
    volatile int which[3] = {0, 1, 2};
    EXPECT_EQ(read_moe_at_run_time<c>(which[0]), 2.5);
    EXPECT_EQ(read_moe_at_run_time<c>(which[1]), 0.01);
    EXPECT_EQ(read_moe_at_run_time<c>(which[2]), 0.001);
}

} // namespace
