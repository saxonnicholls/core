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
/// **Implementations.** One. The presets (`gpt2_small`, `llama3_8b`, `nanochat_d20`) are the shapes of those models.
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
/// `rope_base_v` and `attention_softcap_v`, which evaluate them at compile time, never as `C.norm_eps` in code that
/// runs; every header here does. Wrapping the value in a class (its bits as an integer) is worse: Clang 18 crashes.
///
/// **Status.** Experimental, new in 0.8.0.

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
};

/// \brief Whether a configuration is consistent: every width set, `n_heads` a multiple of `n_kv_heads`, `head_dim`
/// even under RoPE, a positive `norm_eps`, a non-negative soft cap.
/// \param c The configuration.
/// \return Whether it is valid.
[[nodiscard]] constexpr bool valid(const transformer_config& c) noexcept
{
    return c.d_model > 0 && c.n_heads > 0 && c.n_kv_heads > 0 && c.head_dim > 0 && c.d_ff > 0 && c.n_heads % c.n_kv_heads == 0
           && (c.rope == rope_kind::none || c.head_dim % 2 == 0) && c.norm_eps > 0 && c.attention_softcap >= 0
           && (c.rope == rope_kind::none || c.rope_base > 1);
}

/// \brief The shape of GPT-2 small (124M): 12 heads of 64, layer norm, tanh GELU, biases, learned positions.
inline constexpr transformer_config gpt2_small{.d_model = 768, .n_heads = 12, .n_kv_heads = 12, .head_dim = 64, .d_ff = 3072,
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

/// \brief `C.norm_eps` in `T`, evaluated at compile time (see Known issues).
template <transformer_config C, math::math_number T>
inline constexpr T norm_eps_v = static_cast<T>(C.norm_eps);

/// \brief `C.rope_base` in `T`, evaluated at compile time (see Known issues).
template <transformer_config C, math::math_number T>
inline constexpr T rope_base_v = static_cast<T>(C.rope_base);

/// \brief `C.attention_softcap` in `T`, evaluated at compile time (see Known issues).
template <transformer_config C, math::math_number T>
inline constexpr T attention_softcap_v = static_cast<T>(C.attention_softcap);

static_assert(valid(gpt2_small) && valid(llama3_8b) && valid(nanochat_d20));

} // namespace snicholls::ml

#endif // snicholls_core_ml_transformer_config_hpp
