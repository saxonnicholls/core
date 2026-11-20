//
//  transformer_config.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::ml::transformer_config`: the architecture of a transformer layer as one compile-time value, from which every size is a constant.
/// \ingroup core
///
/// **Overview.** Transformers differ in a handful of choices: the widths, how many key-value heads the query heads
/// share, the normalisation, the MLP's activation, the positional encoding, a sliding window, a soft cap on the
/// scores, biases. `transformer_config` holds them as one structural aggregate, passed as a template parameter to
/// `attention_forward`, `mlp_forward` and `block_forward`, so every size is a constant: scratch is fixed-size, loops
/// are bounded, nothing is allocated, and one layer of code serves GPT-2, Llama and nanochat. Multi-head, grouped-query
/// and multi-query attention are one code path with `n_kv_heads` equal to `n_heads`, a divisor of it, or 1.
///
/// \snippet ml/transformer_config_demo.cpp transformer_config
///
/// **Latent attention, multi-token prediction and long context** (TRAINING.md T11), one group of fields after the others,
/// every default leaving a model as it was: multi-head latent attention's ranks and widths (`q_lora_rank`,
/// `kv_lora_rank`, `qk_rope_dim`, `v_head_dim`; `ml/latent_attention.hpp`), RoPE's rescaling for a long context
/// (`rope_scaling`, `rope_factor`, `rope_original_context`, YaRN's `β`, `α` and temperature coefficient; `ml/rope.hpp`),
/// learnable attention sinks, a sparse selection of `sparse_top_k` keys, and multi-token prediction's depth and weight
/// (`ml/multi_token_prediction.hpp`). `valid_long_context` checks them; `deepseek_v3` and `tiny_mla` set them.
///
/// \snippet ml/transformer_config_demo.cpp transformer_config_latent
///
/// **Gated MLPs** (INFERENCE.md I12). `mlp` is one of the plain activations (GELU, tanh GELU, ReLU², quick GELU) or a
/// gated linear unit (`gated(mlp)`: SwiGLU, ReGLU, the three GEGLUs, gpt-oss's clamped SwiGLU and SwiGLU with a clamp;
/// `ml/glu.hpp`). The clamped ones read `glu_alpha` and `glu_limit`, one group after the others whose defaults
/// (gpt-oss's 1.702 and 7) change no other kind.
///
/// \snippet ml/transformer_config_demo.cpp transformer_config_glu
/// **Linear-time sequence mixers** (TRAINING.md T15), one group of fields after the others, every default leaving a model
/// as it was: the mixer a layer uses in place of attention (`mixer`: Mamba-2 or the gated delta rule as layers; gated
/// linear attention and RWKV as ops), its heads, channels, groups, convolution and chunk, and the hybrid stack's pattern
/// (`hybrid_period`, `hybrid_attention_offset`, read by `layer_is_attention`). `valid_mixer` checks them;
/// `mixer_state_size` is a sequence's state; `tiny_mamba2` and `tiny_gated_deltanet` set them.
///
/// \snippet ml/transformer_config_demo.cpp mixer_config
///
/// **Relationships.** `ml/linear.hpp`, `ml/mlp.hpp`, `ml/kv_cache.hpp`, `ml/attention_layer.hpp`,
/// `ml/transformer_block.hpp`; the plan is `INFERENCE.md`. Group: \ref core.
///
/// **Mixture of experts.** With `experts > 0` the MLP is a mixture of experts (`ml/moe.hpp`, TRAINING.md T10): `top_k`
/// of `experts` routed experts of `expert_d_ff` per token, scored as `scoring` says, `shared_experts` dense experts
/// every token passes through, the balance settings, and DeepSeek-V3's node-limited routing (`expert_groups`,
/// `group_top_k`: a token's experts come from its best `group_top_k` of `expert_groups` groups; 0 and 0, the defaults,
/// route over every expert). With `experts == 0` (the default) every expert field is 0 and nothing changes for a dense
/// model.
///
/// \snippet ml/transformer_config_demo.cpp moe_config
///
/// **Attention masks.** `mask` chooses the keys a query sees (`attention_keys`, `ml/attention.hpp`): `causal` (the
/// default, with `window`), `bidirectional` (a ViT; the encoders and predictors of a joint embedding predictive
/// architecture, TRAINING.md T11) or `block_causal` with `mask_block` tokens per block.
///
/// **Implementations.** One. The presets (`gpt2_small`, `gpt2_medium`, `gpt2_large`, `llama3_8b`, `nanochat_d20`,
/// `olmoe_1b_7b`, `vit_base`, `deepseek_v3`) are the shapes of those models; `tiny_moe` is a small mixture of experts and
/// `tiny_mla` a small latent attention for tests.
///
/// **Guarantees.** `constexpr`, trivially copyable, structural (usable as a template argument). `valid(c)` checks the
/// invariants below; every class and function templated on a configuration rejects an invalid one at compile time.
///
/// **Errors.** None at run time. A configuration whose `n_heads` is not a multiple of `n_kv_heads`, with an odd
/// `head_dim` under RoPE, or with a zero width is a compile error (`static_assert` with a `snicholls::ml::` message).
///
/// **Performance.** Not applicable: a value read by the compiler.
///
/// **Tests and demos.** `tests/ml/transformer_config_test.cpp`, `demos/ml/transformer_config_demo.cpp`.
///
/// **Dependencies.** `<cstddef>`, `snicholls/core/math/constexpr_math.hpp`, `snicholls/core/ml/attention.hpp`
/// (`attention_mask`), `snicholls/core/ml/rope.hpp` (`rope_layout`, `rope_scaling`).
///
/// **Known issues.** Clang (Apple clang 17, Clang 18) miscompiles a run-time read of a `double` member of a
/// class-type template argument: it loads garbage (`C.norm_eps` reads as 6e-314), while the same read in a constant
/// expression is right, and GCC is right in both. Read the floating-point members through `norm_eps_v`,
/// `rope_base_v`, `attention_softcap_v`, the mixture of experts' `route_scale_v`, `bias_update_rate_v`,
/// `balance_loss_weight_v` and `z_loss_weight_v`, `rope_factor_v`, `mtp_loss_weight_v`, `rope_scaling_v` and
/// `yarn_attention_factor_v`, and `glu_alpha_v` and `glu_limit_v`, which evaluate them at compile time, never as `C.norm_eps` in code that runs; every header
/// here does. Wrapping the value in a class (its bits as an integer) is worse: Clang 18 crashes.
///
/// **Status.** Experimental, new in 0.8.0; the mixture-of-experts fields, `mask`, `mask_block`, `vit_base`, the fields of
/// TRAINING.md T12, `deepseek_v3` and `tiny_mla` new in 0.16.0; the gated kinds after `swiglu`, `gated`, `glu_alpha`,
/// `glu_limit` and their readers, the mixer fields, `valid_mixer`, `mixer_state_size`, `layer_is_attention`,
/// `mixer_scale_v`, `tiny_mamba2` and `tiny_gated_deltanet` new in 0.17.0.

#ifndef snicholls_core_ml_transformer_config_hpp
#define snicholls_core_ml_transformer_config_hpp

#include <snicholls/core/math/constexpr_math.hpp>
#include <snicholls/core/ml/attention.hpp>
#include <snicholls/core/ml/rope.hpp>

#include <cstddef>

namespace snicholls::ml
{

/// \brief The normalisation before attention and before the MLP.
enum class norm_kind
{
    layer,   ///< `layer_norm`: GPT-2.
    rms,     ///< `rms_norm`: Llama, nanochat.
};

/// \brief The MLP's form and activation.
enum class mlp_kind
{
    gelu,           ///< `down(gelu(up x))`, the exact GELU.
    gelu_tanh,      ///< `down(gelu_tanh(up x))`: GPT-2.
    relu_squared,   ///< `down(relu(up x)²)`: nanochat.
    swiglu,         ///< `down(silu(gate x) · up x)`: Llama.
    gelu_quick,     ///< `down(gelu_quick(up x))`: CLIP's vision encoder.
    reglu,          ///< `down(relu(gate x) · up x)` (`ml/glu.hpp`).
    geglu,          ///< `down(gelu_tanh(gate x) · up x)`: ggml's GEGLU, Gemma.
    geglu_erf,      ///< `down(gelu(gate x) · up x)`.
    geglu_quick,    ///< `down(gelu_quick(gate x) · up x)`.
    swiglu_oai,     ///< gpt-oss's clamped SwiGLU with `glu_alpha` and `glu_limit` (`glu_kind::swiglu_oai`).
    swiglu_clamp,   ///< `down(silu(min(gate x, limit)) · clamp(up x, ±limit))` with `glu_limit`.
};

/// \brief Whether an MLP of this kind has a gate projection (the gated linear units of `ml/glu.hpp`).
/// \param k The kind.
/// \return True for `swiglu`, `reglu`, the GEGLUs and the clamped SwiGLUs.
[[nodiscard]] constexpr bool gated(mlp_kind k) noexcept
{
    switch (k) {
    case mlp_kind::gelu:
    case mlp_kind::gelu_tanh:
    case mlp_kind::relu_squared:
    case mlp_kind::gelu_quick:
        return false;
    case mlp_kind::swiglu:
    case mlp_kind::reglu:
    case mlp_kind::geglu:
    case mlp_kind::geglu_erf:
    case mlp_kind::geglu_quick:
    case mlp_kind::swiglu_oai:
    case mlp_kind::swiglu_clamp:
        return true;
    }
    return false;
}

/// \brief The positional encoding applied to queries and keys.
enum class rope_kind
{
    none,          ///< No rotation (positions are learned, outside the layer: GPT-2).
    half,          ///< RoPE, the halves of a head paired (Hugging Face's `rotate_half`, nanochat).
    interleaved,   ///< RoPE, adjacent channels paired (the original Llama).
};

/// \brief Where a per-head RMS norm of queries and keys is applied, if anywhere.
enum class qk_norm_kind
{
    none,          ///< No norm: GPT-2, Llama.
    before_rope,   ///< Normalise, then rotate: Qwen 3, OLMo 2.
    after_rope,    ///< Rotate, then normalise: nanochat.
};

/// \brief How a mixture of experts scores the experts and weights the ones it chooses (`ml/moe.hpp`).
enum class moe_scoring
{
    softmax_top_k,   ///< Choose by logit; the weights a softmax over the chosen logits (Mixtral).
    softmax,         ///< A softmax over every expert; choose by probability, the weights the chosen probabilities, not renormalised (Switch, OLMoE).
    sigmoid,         ///< A sigmoid per expert; choose by it, the weights the chosen sigmoids divided by their sum (DeepSeek-V3).
};

/// \brief The linear-time sequence mixer a layer may use in place of attention (`ml/mixer_layer.hpp`, TRAINING.md T15).
enum class mixer_kind
{
    none,             ///< No mixer: every layer is attention.
    mamba2,           ///< Mamba-2's selective state space (`ml/ssd.hpp`), a layer of its own.
    gated_deltanet,   ///< The gated delta rule (`ml/delta_rule.hpp`), Qwen3-Next's linear attention, a layer of its own.
    gla,              ///< Gated linear attention (`ml/gla.hpp`): the shape of its op; no layer yet.
    rwkv6,            ///< RWKV-6's WKV (`ml/rwkv.hpp`): the shape of its op; no layer yet.
    rwkv7,            ///< RWKV-7's WKV (`ml/rwkv.hpp`): the shape of its op; no layer yet.
};

/// \brief The architecture of a transformer layer: a structural aggregate, passed as a template parameter.
/// \ingroup core
struct transformer_config
{
    std::size_t d_model = 0;       ///< The width of the residual stream.
    std::size_t n_heads = 0;       ///< The query heads.
    std::size_t n_kv_heads = 0;    ///< The key-value heads: `n_heads` (MHA), a divisor (GQA) or 1 (MQA).
    std::size_t head_dim = 0;      ///< The channels of a head.
    std::size_t d_ff = 0;          ///< The MLP's hidden width.
    norm_kind norm = norm_kind::rms;                  ///< The normalisation.
    double norm_eps = 1e-5;                                     ///< Added to the variance or mean square.
    qk_norm_kind qk_norm = qk_norm_kind::none;        ///< A per-head RMS norm of queries and keys.
    mlp_kind mlp = mlp_kind::swiglu;                  ///< The MLP.
    rope_kind rope = rope_kind::none;                 ///< The positional rotation.
    double rope_base = 10000;                                   ///< The base of RoPE's frequencies.
    std::size_t window = 0;                           ///< A sliding window of keys, counting the query's own; 0 for none.
    double attention_softcap = 0;                               ///< `cap · tanh(s / cap)` on each score; 0 for none.
    bool bias = false;                                ///< Biases on the projections and, with layer norm, a shift.
    std::size_t experts = 0;                          ///< Routed experts of a mixture-of-experts MLP; 0 for a dense MLP.
    std::size_t top_k = 0;                            ///< The routed experts each token goes to: 1 to `experts` (0 when dense).
    std::size_t shared_experts = 0;                   ///< Shared experts every token passes through, each `expert_d_ff` wide (DeepSeekMoE).
    std::size_t expert_d_ff = 0;                      ///< The hidden width of each expert (0 when dense).
    moe_scoring scoring = moe_scoring::softmax_top_k; ///< How the router scores and weights the experts.
    double route_scale = 1;                                     ///< A factor on the routed weights (DeepSeek-V3's 2.5); 1 for none.
    double bias_update_rate = 0;                                ///< DeepSeek-V3's `gamma`, the step of the selection bias (`moe_update_bias`).
    double balance_loss_weight = 0;                             ///< The auxiliary balance loss's `alpha`; 0 for none.
    double z_loss_weight = 0;                                   ///< The router z-loss's `beta` (ST-MoE); 0 for none.
    std::size_t expert_groups = 0;                    ///< Node-limited routing (DeepSeek-V3's `n_group`): the routed experts in this many equal groups; 0 for none.
    std::size_t group_top_k = 0;                      ///< The groups a token's experts may come from (DeepSeek-V3's `topk_group`); 0 without groups.

    // Joint embedding predictive architectures (TRAINING.md T11): the attention mask. The defaults are a language
    // model's, so no preset changes.
    attention_mask mask = attention_mask::causal;     ///< Which keys a query sees: causal, bidirectional or block-causal.
    std::size_t mask_block = 0;                       ///< The tokens of a block under `block_causal`; 0 otherwise.

    // Latent attention, multi-token prediction and long context (TRAINING.md T12): every default leaves a model as it was
    std::size_t q_lora_rank = 0;                      ///< MLA: the query latent's rank `d_c'`; 0 projects the query directly.
    std::size_t kv_lora_rank = 0;                     ///< MLA: the key-value latent's rank `d_c`; 0 for standard attention.
    std::size_t qk_rope_dim = 0;                      ///< MLA: the decoupled rotary channels `d_h^R` at the end of each query and key head.
    std::size_t v_head_dim = 0;                       ///< MLA: the channels of a value head.
    rope_scaling_kind rope_scaling = rope_scaling_kind::none;   ///< RoPE's rescaling for a long context.
    double rope_factor = 1;                                     ///< The rescaling's factor `s`.
    std::size_t rope_original_context = 0;            ///< YaRN's trained context `L`.
    double yarn_beta_fast = 32;                                 ///< YaRN's `β`.
    double yarn_beta_slow = 1;                                  ///< YaRN's `α`.
    double yarn_mscale = 1;                                     ///< YaRN's temperature coefficient (DeepSeek's `mscale_all_dim`).
    bool attention_sinks = false;                     ///< A learnable sink logit per query head (gpt-oss).
    std::size_t sparse_top_k = 0;                     ///< Keys each query attends to, chosen by an indexer (DeepSeek Sparse Attention); 0 for all.
    std::size_t mtp_depth = 0;                        ///< Multi-token prediction modules `D` (DeepSeek-V3); 0 for none.
    double mtp_loss_weight = 0;                                 ///< MTP's `λ`.

    // Inference operations (INFERENCE.md I12): the constants of the clamped gated MLPs, read only by `swiglu_oai` and
    // `swiglu_clamp`, so no preset changes
    double glu_alpha = 1.702;                                   ///< `swiglu_oai`'s sigmoid scale (gpt-oss's 1.702).
    double glu_limit = 7;                                       ///< The clamped SwiGLUs' limit (gpt-oss's 7).
    // Linear-time sequence mixers (TRAINING.md T15): every default leaves a model as it was. The names follow state-space
    // duality: a mixer's state per head is `mixer_state` key channels by `mixer_head_dim` value channels
    mixer_kind mixer = mixer_kind::none;              ///< The mixer of the layers that are not attention; `none` for none.
    std::size_t mixer_heads = 0;                      ///< The mixer's value heads `H` (Mamba-2's heads).
    std::size_t mixer_head_dim = 0;                   ///< The value channels of a head `P` (Mamba-2's `headdim`).
    std::size_t mixer_state = 0;                      ///< The key channels of a head `N` (Mamba-2's `d_state`).
    std::size_t mixer_groups = 0;                     ///< The key heads `G` the value heads share (Mamba-2's `ngroups`).
    std::size_t conv_width = 0;                       ///< The short causal convolution's taps `K` (Mamba-2's `d_conv`); 0 for none.
    std::size_t mixer_chunk = 0;                      ///< The chunk `L` of the chunked forms and the backward pass's checkpoints.
    std::size_t hybrid_period = 0;                    ///< Hybrid stacks: one attention layer in every `hybrid_period`; 0 for none.
    std::size_t hybrid_attention_offset = 0;          ///< Hybrid stacks: the attention layer's place in each period.

    /// \brief The width of all the query heads together.
    /// \return `n_heads · head_dim`.
    [[nodiscard]] constexpr std::size_t q_size() const noexcept { return n_heads * head_dim; }
    /// \brief The width of all the key (or value) heads together.
    /// \return `n_kv_heads · head_dim`.
    [[nodiscard]] constexpr std::size_t kv_size() const noexcept { return n_kv_heads * head_dim; }
    /// \brief The query heads that share one key-value head.
    /// \return `n_heads / n_kv_heads`.
    [[nodiscard]] constexpr std::size_t group() const noexcept { return n_heads / n_kv_heads; }
    /// \brief The RoPE layout, for `rope` (meaningful unless `rope` is `none`).
    /// \return `rope_layout::half` or `rope_layout::interleaved`.
    [[nodiscard]] constexpr rope_layout layout() const noexcept { return rope == rope_kind::interleaved ? rope_layout::interleaved : rope_layout::half; }
    /// \brief Whether the MLP is a mixture of experts.
    /// \return `experts > 0`.
    [[nodiscard]] constexpr bool moe() const noexcept { return experts > 0; }
    /// \brief Whether the attention is multi-head latent attention.
    /// \return `kv_lora_rank > 0`.
    [[nodiscard]] constexpr bool latent() const noexcept { return kv_lora_rank > 0; }
    /// \brief MLA: the non-rotary channels of a query or key head, `d_h`.
    /// \return `head_dim - qk_rope_dim`.
    [[nodiscard]] constexpr std::size_t qk_nope_dim() const noexcept { return head_dim - qk_rope_dim; }
};

/// \brief Whether the fields of TRAINING.md T12 are consistent: with no latent rank every MLA field 0; for MLA a rotary
/// part that is even, set and smaller than `head_dim`, a value head, RoPE on, one key-value head per query head and the
/// causal mask (the latent attention's only); a
/// rescaling factor of at least 1, YaRN's trained context and `β > α > 0`; a non-negative MTP weight, 0 without modules;
/// sinks and sparse attention only with MLA, the attention that implements them.
/// \param c The configuration.
/// \return Whether those fields are valid.
[[nodiscard]] constexpr bool valid_long_context(const transformer_config& c) noexcept
{
    const bool standard = c.q_lora_rank == 0 && c.qk_rope_dim == 0 && c.v_head_dim == 0;
    const bool latent = c.qk_rope_dim > 0 && c.qk_rope_dim % 2 == 0 && c.qk_rope_dim < c.head_dim && c.v_head_dim > 0
                        && c.rope != rope_kind::none && c.n_kv_heads == c.n_heads && c.mask == attention_mask::causal;
    const bool yarn = c.rope_scaling == rope_scaling_kind::yarn || c.rope_scaling == rope_scaling_kind::yarn_index_ramp;
    const bool scaling = c.rope_factor >= 1 && c.yarn_mscale >= 0
                         && (!yarn || (c.rope_original_context > 0 && c.yarn_beta_fast > c.yarn_beta_slow && c.yarn_beta_slow > 0));
    const bool mtp = c.mtp_loss_weight >= 0 && (c.mtp_depth > 0 || c.mtp_loss_weight == 0);
    const bool options = c.kv_lora_rank > 0 || (!c.attention_sinks && c.sparse_top_k == 0);   // the latent attention's options
    return (c.kv_lora_rank == 0 ? standard : latent) && scaling && mtp && options;
}

/// \brief Whether the mixer fields of TRAINING.md T15 are consistent: with no mixer every one of them 0; with one, every
/// width set, the value heads a multiple of the key heads and a chunk; RWKV with one key head per value head and square
/// heads; a short convolution of at least one tap for Mamba-2 and the gated delta rule; a hybrid period only with a mixer,
/// its attention layer inside it.
/// \param c The configuration.
/// \return Whether those fields are valid.
[[nodiscard]] constexpr bool valid_mixer(const transformer_config& c) noexcept
{
    if (c.mixer == mixer_kind::none)
        return c.mixer_heads == 0 && c.mixer_head_dim == 0 && c.mixer_state == 0 && c.mixer_groups == 0 && c.conv_width == 0
               && c.mixer_chunk == 0 && c.hybrid_period == 0 && c.hybrid_attention_offset == 0;
    const bool rwkv = c.mixer == mixer_kind::rwkv6 || c.mixer == mixer_kind::rwkv7;
    const bool conv = c.mixer == mixer_kind::mamba2 || c.mixer == mixer_kind::gated_deltanet;
    return c.mixer_heads > 0 && c.mixer_head_dim > 0 && c.mixer_state > 0 && c.mixer_groups > 0 && c.mixer_heads % c.mixer_groups == 0
           && c.mixer_chunk > 0 && (!rwkv || (c.mixer_groups == c.mixer_heads && c.mixer_state == c.mixer_head_dim))
           && (!conv || c.conv_width > 0)
           && (c.hybrid_period == 0 ? c.hybrid_attention_offset == 0 : c.hybrid_attention_offset < c.hybrid_period);
}

/// \brief The state a mixer keeps per sequence: `H · N · P` numbers (the short convolution's is `ml/short_conv.hpp`'s).
/// \param c The configuration.
/// \return The number of values; 0 with no mixer.
[[nodiscard]] constexpr std::size_t mixer_state_size(const transformer_config& c) noexcept
{
    return c.mixer_heads * c.mixer_state * c.mixer_head_dim;
}

/// \brief Whether layer `layer` of a stack is attention: every layer without a mixer, none with a mixer and no hybrid
/// period, else the layer at `hybrid_attention_offset` of each `hybrid_period` (Qwen3-Next: period 4, offset 3; Jamba's
/// pattern: period 8, offset 4).
/// \param c The configuration. \param layer The layer, from 0.
/// \return Whether it is attention; otherwise it is the configuration's mixer.
[[nodiscard]] constexpr bool layer_is_attention(const transformer_config& c, std::size_t layer) noexcept
{
    if (c.mixer == mixer_kind::none)
        return true;
    return c.hybrid_period != 0 && layer % c.hybrid_period == c.hybrid_attention_offset;
}

/// \brief Whether a configuration is consistent: every width set, `n_heads` a multiple of `n_kv_heads`, `head_dim`
/// even under RoPE, a positive `norm_eps`, a non-negative soft cap; for a mixture of experts, `top_k` from 1 to
/// `experts`, `expert_d_ff` set, a positive `route_scale` and non-negative balance settings, groups (if any) dividing
/// the experts with `group_top_k` from 1 to `expert_groups` and enough experts in the chosen groups for `top_k`, and
/// with no experts every expert field 0; a sliding window only under the causal mask, and a `mask_block` exactly under `block_causal`.
/// \param c The configuration.
/// \return Whether it is valid.
[[nodiscard]] constexpr bool valid(const transformer_config& c) noexcept
{
    const bool dense = c.experts == 0 && c.top_k == 0 && c.shared_experts == 0 && c.expert_d_ff == 0 && c.expert_groups == 0 && c.group_top_k == 0;
    const bool groups = (c.expert_groups == 0 && c.group_top_k == 0)
                        || (c.expert_groups > 0 && c.experts % c.expert_groups == 0 && c.group_top_k > 0 && c.group_top_k <= c.expert_groups
                            && c.group_top_k * (c.experts / c.expert_groups) >= c.top_k);
    const bool moe = c.experts > 0 && c.experts <= 0xffffffffu && c.top_k > 0 && c.top_k <= c.experts && c.expert_d_ff > 0
                     && c.route_scale > 0 && c.bias_update_rate >= 0 && c.balance_loss_weight >= 0 && c.z_loss_weight >= 0 && groups;
    return c.d_model > 0 && c.n_heads > 0 && c.n_kv_heads > 0 && c.head_dim > 0 && c.d_ff > 0 && c.n_heads % c.n_kv_heads == 0
           && (c.rope == rope_kind::none || c.head_dim % 2 == 0) && c.norm_eps > 0 && c.attention_softcap >= 0
           && (c.rope == rope_kind::none || c.rope_base > 1) && (dense || moe) && (c.mask == attention_mask::causal || c.window == 0)
           && ((c.mask == attention_mask::block_causal) == (c.mask_block > 0)) && valid_long_context(c) && c.glu_alpha > 0
           && c.glu_limit > 0 && valid_mixer(c);
}

/// \brief The shape of GPT-2 small (124M): 12 heads of 64, layer norm, tanh GELU, biases, learned positions.
inline constexpr transformer_config gpt2_small{.d_model = 768, .n_heads = 12, .n_kv_heads = 12, .head_dim = 64, .d_ff = 3072,
                                               .norm = norm_kind::layer, .norm_eps = 1e-5, .mlp = mlp_kind::gelu_tanh, .bias = true};

/// \brief The shape of GPT-2 medium (350M): 16 heads of 64, 1024 wide, otherwise as `gpt2_small` (24 blocks, which the
/// configuration does not carry).
inline constexpr transformer_config gpt2_medium{.d_model = 1024, .n_heads = 16, .n_kv_heads = 16, .head_dim = 64, .d_ff = 4096,
                                                .norm = norm_kind::layer, .norm_eps = 1e-5, .mlp = mlp_kind::gelu_tanh, .bias = true};

/// \brief The shape of GPT-2 large (774M): 20 heads of 64, 1280 wide, otherwise as `gpt2_small` (36 blocks).
inline constexpr transformer_config gpt2_large{.d_model = 1280, .n_heads = 20, .n_kv_heads = 20, .head_dim = 64, .d_ff = 5120,
                                               .norm = norm_kind::layer, .norm_eps = 1e-5, .mlp = mlp_kind::gelu_tanh, .bias = true};

/// \brief The shape of Llama 3 8B: 32 query heads sharing 8 key-value heads, RMS norm, SwiGLU, RoPE with base 500,000.
inline constexpr transformer_config llama3_8b{.d_model = 4096, .n_heads = 32, .n_kv_heads = 8, .head_dim = 128, .d_ff = 14336,
                                              .norm = norm_kind::rms, .norm_eps = 1e-5, .mlp = mlp_kind::swiglu,
                                              .rope = rope_kind::half, .rope_base = 500000};

/// \brief The shape of nanochat at depth 20: 10 heads of 128, parameter-free RMS norm (float's epsilon), QK norm after
/// RoPE, ReLU², no biases.
inline constexpr transformer_config nanochat_d20{.d_model = 1280, .n_heads = 10, .n_kv_heads = 10, .head_dim = 128, .d_ff = 5120,
                                                 .norm = norm_kind::rms, .norm_eps = 1.1920928955078125e-07,
                                                 .qk_norm = qk_norm_kind::after_rope, .mlp = mlp_kind::relu_squared,
                                                 .rope = rope_kind::half, .rope_base = 10000};

/// \brief A small mixture of experts for tests and demos: 8 wide, 2 heads of 4, 4 routed experts of 16 (top 2) and
/// one shared, SwiGLU, RMS norm, RoPE, scored by softmax over the top 2.
inline constexpr transformer_config tiny_moe{.d_model = 8, .n_heads = 2, .n_kv_heads = 2, .head_dim = 4, .d_ff = 16,
                                             .norm = norm_kind::rms, .mlp = mlp_kind::swiglu, .rope = rope_kind::half,
                                             .experts = 4, .top_k = 2, .shared_experts = 1, .expert_d_ff = 16};

/// \brief The shape of OLMoE-1B-7B (Muennighoff et al., 2024): 2048 wide, 16 heads of 128, 64 SwiGLU experts of 1024
/// (top 8, no shared expert), a softmax over every expert with the chosen probabilities not renormalised, the balance
/// loss at 0.01 and the router z-loss at 0.001, RMS norm, RoPE (16 blocks, which the configuration does not carry;
/// OLMoE's QK norm, over all the heads at once, is not set).
inline constexpr transformer_config olmoe_1b_7b{.d_model = 2048, .n_heads = 16, .n_kv_heads = 16, .head_dim = 128, .d_ff = 1024,
                                                .norm = norm_kind::rms, .norm_eps = 1e-5, .mlp = mlp_kind::swiglu,
                                                .rope = rope_kind::half, .rope_base = 10000, .experts = 64, .top_k = 8,
                                                .expert_d_ff = 1024, .scoring = moe_scoring::softmax,
                                                .balance_loss_weight = 0.01, .z_loss_weight = 0.001};

/// \brief The shape of ViT-B/16 (Dosovitskiy et al., 2021) as I-JEPA's and V-JEPA's encoders use it: 12 heads of 64,
/// layer norm (epsilon 1e-6, timm's), the exact GELU, biases, bidirectional attention, positions added outside the
/// layer (`ml/patch.hpp`).
inline constexpr transformer_config vit_base{.d_model = 768, .n_heads = 12, .n_kv_heads = 12, .head_dim = 64, .d_ff = 3072,
                                             .norm = norm_kind::layer, .norm_eps = 1e-6, .mlp = mlp_kind::gelu, .bias = true,
                                             .mask = attention_mask::bidirectional};

/// \brief A small MLA model for tests and demos: 16 wide, 2 heads of 6 query-key channels (4 plain, 2 rotary) and 4
/// value channels, a query latent of 6 and a key-value latent of 8, RMS norm, SwiGLU, interleaved RoPE.
inline constexpr transformer_config tiny_mla{.d_model = 16, .n_heads = 2, .n_kv_heads = 2, .head_dim = 6, .d_ff = 24,
                                             .norm = norm_kind::rms, .norm_eps = 1e-6, .mlp = mlp_kind::swiglu,
                                             .rope = rope_kind::interleaved, .rope_base = 10000, .q_lora_rank = 6,
                                             .kv_lora_rank = 8, .qk_rope_dim = 2, .v_head_dim = 4};

/// \brief The attention shape of DeepSeek-V3 (DeepSeek-AI, 2024, section 4.2; its `config.json`): 7168 wide, 128 heads
/// of 128 plain and 64 rotary query-key channels and 128 value channels, a query latent of 1536 and a key-value latent of
/// 512, RMS norm (1e-6), interleaved RoPE with base 10,000 under YaRN (factor 40 from 4,096, β 32, α 1), one MTP module
/// with `λ` 0.1 (0.3 for the first 10T tokens); its mixture of experts (section 2.1.2, `config.json`): 256 routed
/// experts of 2048 (top 8) and one shared, sigmoid affinities normalised over the chosen and scaled by 2.5, the
/// selection bias's step 0.001 and the sequence-wise balance loss at 0.0001 (section 4.2). `d_ff` is the dense MLP of
/// its first three blocks (61 blocks, which the configuration does not carry; the other 58 are mixtures of experts),
/// and its node-limited routing chooses each token's experts from 4 of 8 groups of 32 (section 2.1.2, `n_group`,
/// `topk_group`). With a vocabulary of 129,280 (also not carried) these give 671.0B parameters, 37.6B active per token,
/// as the paper's 671B and 37B (`tests/ml/composed_block_test.cpp`).
inline constexpr transformer_config deepseek_v3{.d_model = 7168, .n_heads = 128, .n_kv_heads = 128, .head_dim = 192, .d_ff = 18432,
                                                .norm = norm_kind::rms, .norm_eps = 1e-6, .mlp = mlp_kind::swiglu,
                                                .rope = rope_kind::interleaved, .rope_base = 10000, .experts = 256, .top_k = 8,
                                                .shared_experts = 1, .expert_d_ff = 2048, .scoring = moe_scoring::sigmoid,
                                                .route_scale = 2.5, .bias_update_rate = 0.001, .balance_loss_weight = 0.0001,
                                                .expert_groups = 8, .group_top_k = 4, .q_lora_rank = 1536,
                                                .kv_lora_rank = 512, .qk_rope_dim = 64, .v_head_dim = 128,
                                                .rope_scaling = rope_scaling_kind::yarn_index_ramp, .rope_factor = 40,
                                                .rope_original_context = 4096, .yarn_beta_fast = 32, .yarn_beta_slow = 1,
                                                .yarn_mscale = 1, .mtp_depth = 1, .mtp_loss_weight = 0.1};

/// \brief A small Mamba-2 model for tests and demos (`ml/mixer_layer.hpp`): 8 wide, 2 attention heads of 4 for its
/// hybrid layers, and a mixer of 2 heads of 4 value channels sharing one group of 4 state channels, a convolution of 4
/// taps, chunks of 4; RMS norm, SwiGLU, RoPE.
inline constexpr transformer_config tiny_mamba2{.d_model = 8, .n_heads = 2, .n_kv_heads = 2, .head_dim = 4, .d_ff = 16,
                                                .norm = norm_kind::rms, .mlp = mlp_kind::swiglu, .rope = rope_kind::half,
                                                .mixer = mixer_kind::mamba2, .mixer_heads = 2, .mixer_head_dim = 4,
                                                .mixer_state = 4, .mixer_groups = 1, .conv_width = 4, .mixer_chunk = 4};

/// \brief A small gated-delta-rule model for tests and demos, laid out as Qwen3-Next's hybrid stack (three linear
/// attention layers, then one attention layer: period 4, offset 3): 8 wide, attention heads as `tiny_mamba2`'s, a mixer of
/// 2 value heads of 4 sharing one key head of 4, a convolution of 4 taps, chunks of 4.
inline constexpr transformer_config tiny_gated_deltanet{.d_model = 8, .n_heads = 2, .n_kv_heads = 2, .head_dim = 4, .d_ff = 16,
                                                        .norm = norm_kind::rms, .mlp = mlp_kind::swiglu, .rope = rope_kind::half,
                                                        .mixer = mixer_kind::gated_deltanet, .mixer_heads = 2, .mixer_head_dim = 4,
                                                        .mixer_state = 4, .mixer_groups = 1, .conv_width = 4, .mixer_chunk = 4,
                                                        .hybrid_period = 4, .hybrid_attention_offset = 3};

/// \brief The scale of a linear attention's queries, `1 / sqrt(mixer_state)` (the gated delta rule's and gated linear
/// attention's), evaluated at compile time.
template <transformer_config C, math::math_number T>
inline constexpr T mixer_scale_v = T(1) / math::sqrt(static_cast<T>(C.mixer_state));

/// \brief `C.norm_eps` in `T`, evaluated at compile time (see Known issues).
template <transformer_config C, math::math_number T>
inline constexpr T norm_eps_v = static_cast<T>(C.norm_eps);

/// \brief `C.rope_base` in `T`, evaluated at compile time (see Known issues).
template <transformer_config C, math::math_number T>
inline constexpr T rope_base_v = static_cast<T>(C.rope_base);

/// \brief `C.attention_softcap` in `T`, evaluated at compile time (see Known issues).
template <transformer_config C, math::math_number T>
inline constexpr T attention_softcap_v = static_cast<T>(C.attention_softcap);

/// \brief `C.route_scale` in `T`, evaluated at compile time (see Known issues).
template <transformer_config C, math::math_number T>
inline constexpr T route_scale_v = static_cast<T>(C.route_scale);

/// \brief `C.bias_update_rate` in `T`, evaluated at compile time (see Known issues).
template <transformer_config C, math::math_number T>
inline constexpr T bias_update_rate_v = static_cast<T>(C.bias_update_rate);

/// \brief `C.balance_loss_weight` in `T`, evaluated at compile time (see Known issues).
template <transformer_config C, math::math_number T>
inline constexpr T balance_loss_weight_v = static_cast<T>(C.balance_loss_weight);

/// \brief `C.z_loss_weight` in `T`, evaluated at compile time (see Known issues).
template <transformer_config C, math::math_number T>
inline constexpr T z_loss_weight_v = static_cast<T>(C.z_loss_weight);

/// \brief `C.rope_factor` in `T`, evaluated at compile time (see Known issues).
template <transformer_config C, math::math_number T>
inline constexpr T rope_factor_v = static_cast<T>(C.rope_factor);

/// \brief `C.mtp_loss_weight` in `T`, evaluated at compile time (see Known issues).
template <transformer_config C, math::math_number T>
inline constexpr T mtp_loss_weight_v = static_cast<T>(C.mtp_loss_weight);

/// \brief `C.glu_alpha` in `T`, evaluated at compile time (see Known issues).
template <transformer_config C, math::math_number T>
inline constexpr T glu_alpha_v = static_cast<T>(C.glu_alpha);

/// \brief `C.glu_limit` in `T`, evaluated at compile time (see Known issues).
template <transformer_config C, math::math_number T>
inline constexpr T glu_limit_v = static_cast<T>(C.glu_limit);

/// \brief The configuration's RoPE rescaling in `T`, for `rope_table(base, scaling)`, every field evaluated at compile
/// time; the amplitude is 1 (a latent attention takes YaRN's temperature in its score scale).
template <transformer_config C, math::math_number T>
inline constexpr rope_scaling<T> rope_scaling_v{C.rope_scaling, static_cast<T>(C.rope_factor), C.rope_original_context,
                                                static_cast<T>(C.yarn_beta_fast), static_cast<T>(C.yarn_beta_slow), T(1)};

/// \brief `yarn_attention_factor` of the configuration in `T`: 1 unless it rescales RoPE with YaRN.
template <transformer_config C, math::math_number T>
inline constexpr T yarn_attention_factor_v
    = C.rope_scaling == rope_scaling_kind::yarn || C.rope_scaling == rope_scaling_kind::yarn_index_ramp
          ? yarn_attention_factor(static_cast<T>(C.rope_factor), static_cast<T>(C.yarn_mscale))
          : T(1);

static_assert(valid(gpt2_small) && valid(gpt2_medium) && valid(gpt2_large) && valid(llama3_8b) && valid(nanochat_d20) && valid(tiny_moe)
              && valid(olmoe_1b_7b) && valid(vit_base) && valid(tiny_mla) && valid(deepseek_v3) && valid(tiny_mamba2)
              && valid(tiny_gated_deltanet));

} // namespace snicholls::ml

#endif // snicholls_core_ml_transformer_config_hpp
