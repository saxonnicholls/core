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
/// **Relationships.** `ml/linear.hpp`, `ml/mlp.hpp`, `ml/kv_cache.hpp`, `ml/attention_layer.hpp`,
/// `ml/transformer_block.hpp`; the plan is `INFERENCE.md`. Group: \ref core.
///
/// **Mixture of experts.** With `experts > 0` the MLP is a mixture of experts (`ml/moe.hpp`, TRAINING.md T10): `top_k`
/// of `experts` routed experts of `expert_d_ff` per token, scored as `scoring` says, `shared_experts` dense experts
/// every token passes through, and the balance settings. With `experts == 0` (the default) every expert field is 0
/// and nothing changes for a dense model.
///
/// \snippet ml/transformer_config_demo.cpp moe_config
///
/// **Implementations.** One. The presets (`gpt2_small`, `gpt2_medium`, `gpt2_large`, `llama3_8b`, `nanochat_d20`,
/// `olmoe_1b_7b`) are the shapes of those models; `tiny_moe` is a small mixture of experts for tests.
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
/// **Dependencies.** `<cstddef>`, `snicholls/core/math/constexpr_math.hpp`, `snicholls/core/ml/rope.hpp` (`rope_layout`).
///
/// **Known issues.** Clang (Apple clang 17, Clang 18) miscompiles a run-time read of a `double` member of a
/// class-type template argument: it loads garbage (`C.norm_eps` reads as 6e-314), while the same read in a constant
/// expression is right, and GCC is right in both. Read the floating-point members through `norm_eps_v`,
/// `rope_base_v`, `attention_softcap_v` and the mixture of experts' `route_scale_v`, `bias_update_rate_v`,
/// `balance_loss_weight_v` and `z_loss_weight_v`, which evaluate them at compile time, never as `C.norm_eps` in code that
/// runs; every header here does. Wrapping the value in a class (its bits as an integer) is worse: Clang 18 crashes.
///
/// **Status.** Experimental, new in 0.8.0; the mixture-of-experts fields new in 0.16.0.

#ifndef snicholls_core_ml_transformer_config_hpp
#define snicholls_core_ml_transformer_config_hpp

#include <snicholls/core/math/constexpr_math.hpp>
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
};

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
};

/// \brief Whether a configuration is consistent: every width set, `n_heads` a multiple of `n_kv_heads`, `head_dim`
/// even under RoPE, a positive `norm_eps`, a non-negative soft cap; for a mixture of experts, `top_k` from 1 to
/// `experts`, `expert_d_ff` set, a positive `route_scale` and non-negative balance settings, and with no experts every
/// expert field 0.
/// \param c The configuration.
/// \return Whether it is valid.
[[nodiscard]] constexpr bool valid(const transformer_config& c) noexcept
{
    const bool dense = c.experts == 0 && c.top_k == 0 && c.shared_experts == 0 && c.expert_d_ff == 0;
    const bool moe = c.experts > 0 && c.experts <= 0xffffffffu && c.top_k > 0 && c.top_k <= c.experts && c.expert_d_ff > 0
                     && c.route_scale > 0 && c.bias_update_rate >= 0 && c.balance_loss_weight >= 0 && c.z_loss_weight >= 0;
    return c.d_model > 0 && c.n_heads > 0 && c.n_kv_heads > 0 && c.head_dim > 0 && c.d_ff > 0 && c.n_heads % c.n_kv_heads == 0
           && (c.rope == rope_kind::none || c.head_dim % 2 == 0) && c.norm_eps > 0 && c.attention_softcap >= 0
           && (c.rope == rope_kind::none || c.rope_base > 1) && (dense || moe);
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

static_assert(valid(gpt2_small) && valid(gpt2_medium) && valid(gpt2_large) && valid(llama3_8b) && valid(nanochat_d20) && valid(tiny_moe)
              && valid(olmoe_1b_7b));

} // namespace snicholls::ml

#endif // snicholls_core_ml_transformer_config_hpp
