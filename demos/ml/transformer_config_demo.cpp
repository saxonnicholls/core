//
//  transformer_config_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/ml/transformer_config.hpp>: the presets, a configuration of one's own checked by
//  valid(), its derived sizes, and its floating-point members read through the compile-time readers. Exits non-zero
//  if a result is not what it documents.
//

#include <snicholls/core/ml/transformer_config.hpp>

#include <cstdio>

namespace ml = snicholls::ml;

//! [transformer_config]
// Grouped-query attention: 8 query heads share 2 key-value heads; RMS norm, SwiGLU, RoPE, a sliding window of 512
constexpr ml::transformer_config mine{.d_model = 512, .n_heads = 8, .n_kv_heads = 2, .head_dim = 64, .d_ff = 1536,
                                      .norm = ml::norm_kind::rms, .norm_eps = 1e-6, .mlp = ml::mlp_kind::swiglu,
                                      .rope = ml::rope_kind::half, .rope_base = 10000, .window = 512};
static_assert(ml::valid(mine));   // every template taking a configuration checks this too

// Read a floating-point member through its reader, which evaluates it at compile time
constexpr float eps = ml::norm_eps_v<mine, float>;
//! [transformer_config]

//! [moe_config]
// A mixture of experts in DeepSeek-V3's style: 16 fine-grained experts of 256, each token to 4, one shared expert,
// sigmoid scoring scaled by 2.5, the selection bias stepped by 0.001 (ml/moe.hpp)
constexpr ml::transformer_config experts = [] {
    auto c = mine;
    c.d_ff = 256;
    c.experts = 16;
    c.top_k = 4;
    c.shared_experts = 1;
    c.expert_d_ff = 256;
    c.scoring = ml::moe_scoring::sigmoid;
    c.route_scale = 2.5;
    c.bias_update_rate = 0.001;
    return c;
}();
static_assert(ml::valid(experts) && experts.moe() && !mine.moe());
//! [moe_config]
//! [transformer_config_latent]
// Multi-head latent attention with YaRN to 4x a trained 2048, a sink per head and one multi-token prediction module
// (TRAINING.md T12): every other model leaves these fields at their defaults
constexpr ml::transformer_config latent{.d_model = 512, .n_heads = 8, .n_kv_heads = 8, .head_dim = 96, .d_ff = 1536,
                                        .norm = ml::norm_kind::rms, .norm_eps = 1e-6, .mlp = ml::mlp_kind::swiglu,
                                        .rope = ml::rope_kind::interleaved, .rope_base = 10000, .q_lora_rank = 192,
                                        .kv_lora_rank = 128, .qk_rope_dim = 32, .v_head_dim = 64,
                                        .rope_scaling = ml::rope_scaling_kind::yarn_index_ramp, .rope_factor = 4,
                                        .rope_original_context = 2048, .attention_sinks = true, .mtp_depth = 1,
                                        .mtp_loss_weight = 0.3};
static_assert(ml::valid(latent) && ml::valid_long_context(latent) && latent.latent() && latent.qk_nope_dim() == 64);
constexpr auto scaling = ml::rope_scaling_v<latent, double>;          // for rope_table(base, scaling)
constexpr double temperature = ml::yarn_attention_factor_v<latent, double>;
//! [transformer_config_latent]

int main()
{
    std::printf("snicholls-core transformer_config demo\n");
    bool ok = true;
    std::printf("  mine: q %zu, kv %zu, %zu query heads per kv head, layout %s, eps %g\n", mine.q_size(), mine.kv_size(), mine.group(),
                mine.layout() == ml::rope_layout::half ? "half" : "interleaved", static_cast<double>(eps));
    std::printf("  gpt2_small %zu, gpt2_medium %zu, gpt2_large %zu wide\n", ml::gpt2_small.d_model, ml::gpt2_medium.d_model,
                ml::gpt2_large.d_model);
    ok &= ml::gpt2_medium.n_heads == 16 && ml::gpt2_large.n_heads == 20;
    std::printf("  gpt2_small %zu wide, llama3_8b %zu/%zu heads, nanochat_d20 qk norm %s\n", ml::gpt2_small.d_model, ml::llama3_8b.n_heads,
                ml::llama3_8b.n_kv_heads, ml::nanochat_d20.qk_norm == ml::qk_norm_kind::after_rope ? "after RoPE" : "other");
    constexpr ml::transformer_config broken{.d_model = 512, .n_heads = 8, .n_kv_heads = 3, .head_dim = 64, .d_ff = 1536};
    std::printf("  8 heads over 3 kv heads is %s\n", ml::valid(broken) ? "valid" : "rejected");
    ok &= mine.q_size() == 512 && mine.kv_size() == 128 && mine.group() == 4 && eps == 1e-6f && !ml::valid(broken);
    ok &= ml::rope_base_v<ml::llama3_8b, double> == 500000 && ml::attention_softcap_v<mine, double> == 0;
    std::printf("  experts: %zu of %zu wide, top %zu, %zu shared, route scale %g; olmoe_1b_7b %zu experts, top %zu\n", experts.experts,
                experts.expert_d_ff, experts.top_k, experts.shared_experts, ml::route_scale_v<experts, double>, ml::olmoe_1b_7b.experts,
                ml::olmoe_1b_7b.top_k);
    ok &= ml::route_scale_v<experts, double> == 2.5 && ml::olmoe_1b_7b.top_k == 8 && ml::valid(ml::tiny_moe);
    // A ViT encoder attends both ways (TRAINING.md T11); a sliding window is a causal model's only
    constexpr ml::transformer_config windowed_vit{.d_model = 768, .n_heads = 12, .n_kv_heads = 12, .head_dim = 64, .d_ff = 3072,
                                                  .window = 16, .mask = ml::attention_mask::bidirectional};
    std::printf("  vit_base %zu wide, mask %s; a windowed bidirectional layer is %s\n", ml::vit_base.d_model,
                ml::vit_base.mask == ml::attention_mask::bidirectional ? "bidirectional" : "other", ml::valid(windowed_vit) ? "valid" : "rejected");
    ok &= ml::vit_base.mask == ml::attention_mask::bidirectional && !ml::valid(windowed_vit) && ml::gpt2_small.mask == ml::attention_mask::causal;
    std::printf("  latent: %zu plain + %zu rotary channels, YaRN x%g from %zu, temperature %.4f, MTP weight %g\n", latent.qk_nope_dim(),
                latent.qk_rope_dim, ml::rope_factor_v<latent, double>, scaling.original_context, temperature,
                ml::mtp_loss_weight_v<latent, double>);
    std::printf("  deepseek_v3: %zu heads, latents %zu and %zu; tiny_mla %zu wide\n", ml::deepseek_v3.n_heads, ml::deepseek_v3.q_lora_rank,
                ml::deepseek_v3.kv_lora_rank, ml::tiny_mla.d_model);
    ok &= scaling.factor == 4.0 && temperature > 1.13 && temperature < 1.14 && !ml::llama3_8b.latent() && ml::deepseek_v3.latent();

    //! [transformer_config_glu]
    // gpt-oss's MLP: a clamped SwiGLU, with the constants read at compile time
    constexpr ml::transformer_config oss{.d_model = 2880, .n_heads = 64, .n_kv_heads = 8, .head_dim = 64, .d_ff = 2880,
                                         .mlp = ml::mlp_kind::swiglu_oai, .rope = ml::rope_kind::half, .rope_base = 150000};
    static_assert(ml::valid(oss) && ml::gated(oss.mlp) && !ml::gated(ml::mlp_kind::gelu_quick));
    constexpr float alpha = ml::glu_alpha_v<oss, float>, limit = ml::glu_limit_v<oss, float>;
    //! [transformer_config_glu]
    std::printf("  gpt-oss MLP: gated %d, alpha %.3f, limit %.1f\n", ml::gated(oss.mlp) ? 1 : 0, static_cast<double>(alpha),
                static_cast<double>(limit));
    ok &= alpha == 1.702f && limit == 7.0f;
    std::printf(ok ? "transformer_config demo: ok\n" : "transformer_config demo: FAILED\n");
    return ok ? 0 : 1;
}
