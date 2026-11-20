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
// The gated MLPs' constants (INFERENCE.md I12): gpt-oss's by default, positive, read at compile time
static_assert(base.glu_alpha == 1.702 && base.glu_limit == 7);
static_assert(!ml::valid(with(base, [](auto& c) { c.glu_limit = 0; })) && !ml::valid(with(base, [](auto& c) { c.glu_alpha = -1; })));
static_assert(ml::valid(with(base, [](auto& c) { c.mlp = ml::mlp_kind::swiglu_oai; })));
static_assert(ml::glu_alpha_v<ml::llama3_8b, float> == 1.702f && ml::glu_limit_v<ml::llama3_8b, double> == 7.0);
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
// TRAINING.md T12: the defaults leave every model as it was; each group of fields is checked
static_assert(!base.latent() && base.mtp_depth == 0 && base.rope_scaling == ml::rope_scaling_kind::none && !base.attention_sinks);
static_assert(!ml::llama3_8b.latent() && ml::valid_long_context(ml::llama3_8b) && ml::valid_long_context(ml::gpt2_small));
static_assert(!ml::valid(with(base, [](auto& c) { c.qk_rope_dim = 2; })));             // an MLA field without a latent
static_assert(!ml::valid(with(base, [](auto& c) { c.sparse_top_k = 4; })));            // sparse selection is MLA's
static_assert(!ml::valid(with(base, [](auto& c) { c.rope_factor = 0.5; })));
static_assert(!ml::valid(with(base, [](auto& c) { c.rope_scaling = ml::rope_scaling_kind::yarn; c.rope_factor = 4; })));   // no L
static_assert(ml::valid(with(base, [](auto& c) { c.rope_scaling = ml::rope_scaling_kind::ntk; c.rope_factor = 4; })));
static_assert(!ml::valid(with(base, [](auto& c) { c.mtp_loss_weight = 0.1; })));        // a weight without modules
static_assert(ml::valid(with(base, [](auto& c) { c.mtp_depth = 2; c.mtp_loss_weight = 0.3; })));
static_assert(ml::deepseek_v3.head_dim == ml::deepseek_v3.qk_nope_dim() + ml::deepseek_v3.qk_rope_dim);
static_assert(ml::rope_scaling_v<ml::deepseek_v3, float>.kind == ml::rope_scaling_kind::yarn_index_ramp
              && ml::rope_scaling_v<ml::deepseek_v3, float>.original_context == 4096);
static_assert(ml::yarn_attention_factor_v<ml::tiny_mla, double> == 1.0 && ml::yarn_attention_factor_v<ml::deepseek_v3, double> > 1.36);
// Node-limited routing: off by default, so no preset but DeepSeek-V3's changes; the groups must divide the experts and
// hold enough of them for top_k
static_assert(ml::tiny_moe.expert_groups == 0 && ml::olmoe_1b_7b.group_top_k == 0 && ml::llama3_8b.expert_groups == 0);
static_assert(ml::deepseek_v3.expert_groups == 8 && ml::deepseek_v3.group_top_k == 4 && ml::valid(ml::deepseek_v3));
static_assert(!ml::valid(with(ml::deepseek_v3, [](auto& c) { c.group_top_k = 0; })));
static_assert(!ml::valid(with(ml::deepseek_v3, [](auto& c) { c.expert_groups = 7; })));
static_assert(!ml::valid(with(ml::deepseek_v3, [](auto& c) { c.expert_groups = 128; c.group_top_k = 3; })));   // 6 experts < top 8
static_assert(ml::valid(with(ml::deepseek_v3, [](auto& c) { c.expert_groups = 128; c.group_top_k = 4; })));

// The configuration is a template argument
template <ml::transformer_config C>
constexpr std::size_t heads_v = C.n_heads;
static_assert(heads_v<ml::llama3_8b> == 32);

// The linear-time sequence mixers' fields (TRAINING.md T15): neutral defaults, consistency, the hybrid pattern
static_assert(ml::gpt2_small.mixer == ml::mixer_kind::none && ml::mixer_state_size(ml::llama3_8b) == 0);
static_assert(ml::valid(ml::tiny_mamba2) && ml::valid(ml::tiny_gated_deltanet));
static_assert(ml::mixer_state_size(ml::tiny_mamba2) == 2 * 4 * 4);
static_assert(!ml::valid(with(ml::gpt2_small, [](auto& c) { c.mixer_heads = 2; })));                 // a field without a mixer
static_assert(!ml::valid(with(ml::tiny_mamba2, [](auto& c) { c.mixer_groups = 3; })));               // 2 heads in 3 groups
static_assert(!ml::valid(with(ml::tiny_mamba2, [](auto& c) { c.mixer_chunk = 0; })));
static_assert(!ml::valid(with(ml::tiny_mamba2, [](auto& c) { c.conv_width = 0; })));                 // Mamba-2 has a convolution
static_assert(!ml::valid(with(ml::tiny_mamba2, [](auto& c) { c.mixer = ml::mixer_kind::rwkv7; })));  // RWKV's heads are square, one key head each
static_assert(ml::valid(with(ml::tiny_mamba2, [](auto& c) { c.mixer = ml::mixer_kind::rwkv7; c.mixer_groups = 2; })));
static_assert(!ml::valid(with(ml::tiny_gated_deltanet, [](auto& c) { c.hybrid_attention_offset = 4; })));
static_assert(!ml::valid(with(ml::tiny_mamba2, [](auto& c) { c.hybrid_attention_offset = 1; })));     // an offset without a period
// Qwen3-Next's pattern: three linear attention layers, then attention; Jamba's: attention at 4 of every 8
static_assert(!ml::layer_is_attention(ml::tiny_gated_deltanet, 0) && !ml::layer_is_attention(ml::tiny_gated_deltanet, 2)
              && ml::layer_is_attention(ml::tiny_gated_deltanet, 3) && ml::layer_is_attention(ml::tiny_gated_deltanet, 7));
static_assert(ml::layer_is_attention(with(ml::tiny_mamba2, [](auto& c) { c.hybrid_period = 8; c.hybrid_attention_offset = 4; }), 12));
static_assert(ml::layer_is_attention(ml::gpt2_small, 5) && !ml::layer_is_attention(ml::tiny_mamba2, 5));
static_assert(ml::mixer_scale_v<ml::tiny_mamba2, double> == 0.5);


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

TEST(MlTransformerConfig, DeepSeekV3HasItsPapersAndConfigurationsShape)
{
    // DeepSeek-V3, section 4.2, and its config.json
    constexpr auto c = ml::deepseek_v3;
    volatile int which = 0;   // the double members, read at run time through their readers
    EXPECT_EQ(c.d_model, 7168u);
    EXPECT_EQ(c.n_heads, 128u);
    EXPECT_EQ(c.qk_nope_dim(), 128u);
    EXPECT_EQ(c.qk_rope_dim, 64u);
    EXPECT_EQ(c.v_head_dim, 128u);
    EXPECT_EQ(c.q_lora_rank, 1536u);
    EXPECT_EQ(c.kv_lora_rank, 512u);
    EXPECT_EQ(c.rope, ml::rope_kind::interleaved);
    EXPECT_EQ(c.rope_original_context, 4096u);
    EXPECT_EQ(c.mtp_depth, 1u);
    EXPECT_EQ(c.experts, 256u);   // its mixture of experts (section 4.2): T10's fields
    EXPECT_EQ(c.top_k, 8u);
    EXPECT_EQ(c.shared_experts, 1u);
    EXPECT_EQ(c.expert_d_ff, 2048u);
    EXPECT_EQ(c.scoring, ml::moe_scoring::sigmoid);
    EXPECT_EQ(which == 0 ? (ml::route_scale_v<c, double>) : 0.0, 2.5);
    EXPECT_EQ(which == 0 ? (ml::bias_update_rate_v<c, double>) : 0.0, 0.001);
    EXPECT_EQ(which == 0 ? (ml::balance_loss_weight_v<c, double>) : 0.0, 0.0001);
    EXPECT_EQ(which == 0 ? (ml::rope_factor_v<c, double>) : 0.0, 40.0);
    EXPECT_EQ(which == 0 ? (ml::mtp_loss_weight_v<c, double>) : 0.0, 0.1);
    EXPECT_EQ(which == 0 ? (ml::norm_eps_v<c, double>) : 0.0, 1e-6);
    EXPECT_TRUE(ml::tiny_mla.latent());
}

} // namespace
