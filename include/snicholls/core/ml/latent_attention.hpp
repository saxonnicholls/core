//
//  latent_attention.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::ml` multi-head latent attention (MLA, DeepSeek-V2 and V3): queries, keys and values from low-rank latents with decoupled RoPE; the training passes over sequences through a `linalg` backend; a KV cache holding only the latent and the rotary key; and the absorbed form for decoding.
/// \ingroup core
///
/// **Overview.** Multi-head attention caches `2 · n_h · d_h` numbers per token and layer; MLA (DeepSeek-V2, 2024,
/// section 2.1, eqs. 9 to 19) caches `d_c + d_h^R` (576 in DeepSeek-V3, against 32,768 for the same heads in MHA). For
/// a token `h_t`:
///
///     c^Q = RMSNorm(W^DQ h)              q^C_i, q^R_i = W^UQ c^Q, RoPE(W^QR c^Q)      (eqs. 12 to 14)
///     c^KV = RMSNorm(W^DKV h)            k^R = RoPE(W^KR h), one rotary key for every head  (eqs. 9, 15)
///     k^C_i, v_i = W^UK c^KV, W^UV c^KV  (eqs. 10, 11)
///     q_i = [q^C_i; q^R_i],  k_{j,i} = [k^C_{j,i}; k^R_j]                              (eqs. 16, 17)
///     o_i = Σ_j softmax_j(q_i · k_{j,i} · scale) v_{j,i},   u = W^O [o_1; ...; o_{n_h}]  (eqs. 18, 19)
///
/// with `scale = 1 / sqrt(d_h + d_h^R)`, times the square of YaRN's temperature factor when the configuration rescales
/// RoPE (DeepSeek-V3, section 4.3; `latent_attention_scale`). The RMS norms after the two down-projections are the
/// papers' "additional RMS Norm layers after the compressed latent vectors" (V2 section 3.1.2, V3 section 4.2) and the
/// released code's `q_norm` and `kv_norm`. With `q_lora_rank` 0 the query is projected directly (`W^UQ` reads `h`).
///
/// The weights are laid out as DeepSeek-V3's checkpoints are, so they load as they are: `q_up` holds each head's
/// `d_h` plain then `d_h^R` rotary rows; `kv_down` the latent's `d_c` rows then the rotary key's `d_h^R`; `kv_up` each
/// head's `d_h` key rows then its `d_v` value rows. RoPE rotates the rotary channels only (a table of `d_h^R`),
/// interleaved in DeepSeek-V3 (`rope_kind::interleaved`).
///
/// **Training.** `latent_attention_forward_sequence(backend, w, xs, rope_src, act, out, sequences)` runs a batch of
/// sequences end to end, every projection one `linear_sequence` through the backend, and saves
/// `latent_attention_activations`; `latent_attention_backward` takes the output's gradient back to every weight, the
/// norms', the sinks' and the input's. The steps that are not products are backend calls (TRAINING.md T9):
/// `latent_attention_norm_rows`, `latent_attention_rope_rows`, `latent_attention_attend_rows` and their backward passes,
/// with a host overload for any backend that a device library overloads for its own. Attention never stores the
/// `n x n` probabilities: each head folds its keys in one pass and keeps the log-sum-exp (TRAINING.md T2).
///
/// \snippet ml/latent_attention_demo.cpp latent_training
///
/// **Decoding: the absorbed form.** `latent_attention_forward(w, x, pos, cache, layer, rope_src, out)` appends the
/// token's latent and rotary key to a `latent_kv_cache_view` and attends without forming any head's keys or values:
/// since `q^C_i · (W^UK_i c_j) = ((W^UK_i)ᵀ q^C_i) · c_j` and `Σ_j p_j W^UV_i c_j = W^UV_i Σ_j p_j c_j`, each head's
/// query is taken into the latent space (`W^UK` absorbed on the query side), every head attends to the same cached rows
/// `[c_j; k^R_j]` as one multi-query attention, and `W^UV` is applied after the weighted sum, before `W^O` (V2 section
/// 2.1.2; DeepSeek-V3's `attn_impl = "absorb"`). The two forms are the same algebra summed in another order: equal
/// within rounding (the scores differ by reassociating `d_h · d_c` products, about `(d_h + d_c) · ε` relative to the
/// sum of their magnitudes; tested at 1e-12 in double and 2e-5 in float on `tiny_mla`). Pre-multiplying `W^UK` into
/// `W^UQ` and `W^UV` into `W^O` is the same algebra again, with larger matrices (`n_h · d_c · d_c'`); it is not built.
///
/// \snippet ml/latent_attention_demo.cpp latent_decode
///
/// **Options.** `attention_sinks`: a learnable logit per head (`latent_attention_weights::sinks`,
/// `ml/attention_sink.hpp`). `sparse_top_k`: each query attends to the keys `latent_attention_activations::selected`
/// names (DeepSeek Sparse Attention, from `ml/sparse_attention.hpp`'s indexer and `select_top_k`; in decoding the
/// `selected` argument). `window`: a sliding window, as for standard attention. KV-cache sizes are constant expressions:
/// `kv_cache_elements_per_token(c)`, `kv_cache_elements(c, layers, positions)`.
///
/// **Relationships.** `ml/transformer_config.hpp` (the MLA fields, `deepseek_v3`, `tiny_mla`), `ml/linear.hpp`,
/// `ml/norm.hpp`, `ml/rope.hpp` (YaRN), `ml/attention_sink.hpp` (the recurrence), `ml/latent_block.hpp` (the block).
/// Group: \ref core.
///
/// **Implementations.** One, configured at compile time.
///
/// **Guarantees.** `constexpr`, no allocation (decoding keeps `std::array`s of the configuration's sizes). Each head's
/// output is `attend`'s over its explicit keys `[k^C; k^R]` to the bit (tested), since the score's sum runs over the
/// plain channels then the rotary ones, as `attend`'s dot product over the concatenation does. Per token, training:
/// the projections, plus `n_h · visible · (d_h + d_h^R + d_v)` for attention; decoding: `n_h · d_h · d_c` to absorb the
/// query, `n_h · visible · (d_c + d_h^R)`, `n_h · d_c · d_v` for the values.
///
/// **Errors.** Preconditions, asserted: the matrices have the configuration's shapes; the norms are empty or of their
/// latent's width; the sinks have `n_heads` with `attention_sinks`; the selection has `sparse_top_k` per token, each in
/// the visible range; the cache's capacity covers the visible range. An invalid configuration, or one that is not
/// latent, is a compile error.
///
/// **Performance.** Against `attention_forward_sequence` with the same heads and widths through the reference backend
/// (`benchmarks/ml/latent_attention_benchmark.cpp`), after checking each against its loops by hand: see there.
///
/// **Tests and demos.** `tests/ml/latent_attention_test.cpp`, `demos/ml/latent_attention_demo.cpp`.
///
/// **Dependencies.** `<algorithm>`, `<array>`, `<cassert>`, `<concepts>`, `<cstddef>`, `<cstdint>`, `<limits>`,
/// `<span>`, `<type_traits>`, `linalg/backend.hpp`, `ml/attention.hpp`, `ml/attention_sink.hpp`, `ml/linear.hpp`,
/// `ml/norm.hpp`, `ml/rope.hpp`, `ml/row.hpp`, `ml/transformer_config.hpp`.
///
/// **Known issues.** Decoding needs dense `kv_up` weights (the absorbed query is a product by its transpose); a quantised
/// MLA decode is to come. A paged latent cache is not built (`paged_kv_cache_view` covers standard attention). The
/// released V3 weights are FP8 blocks of 128 x 128, which core does not read.
///
/// **Status.** Experimental, new in 0.16.0.

#ifndef snicholls_core_ml_latent_attention_hpp
#define snicholls_core_ml_latent_attention_hpp

#include <snicholls/core/linalg/backend.hpp>
#include <snicholls/core/math/constexpr_math.hpp>
#include <snicholls/core/ml/attention.hpp>
#include <snicholls/core/ml/attention_sink.hpp>
#include <snicholls/core/ml/linear.hpp>
#include <snicholls/core/ml/norm.hpp>
#include <snicholls/core/ml/rope.hpp>
#include <snicholls/core/ml/row.hpp>
#include <snicholls/core/ml/transformer_config.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <type_traits>

namespace snicholls::ml
{

/// \brief The KV cache's numbers per token and layer: `kv_lora_rank + qk_rope_dim` for MLA, `2 · kv_size()` otherwise.
/// \param c The configuration.
/// \return The elements.
[[nodiscard]] constexpr std::size_t kv_cache_elements_per_token(const transformer_config& c) noexcept
{
    return c.latent() ? c.kv_lora_rank + c.qk_rope_dim : 2 * c.kv_size();
}

/// \brief The KV cache's numbers for `layers` layers of `positions` positions.
/// \param c The configuration. \param layers The layers. \param positions The positions.
/// \return `kv_cache_elements_per_token(c) · layers · positions`.
[[nodiscard]] constexpr std::size_t kv_cache_elements(const transformer_config& c, std::size_t layers, std::size_t positions) noexcept
{
    return kv_cache_elements_per_token(c) * layers * positions;
}

/// \brief The score scale of a latent attention: `1 / sqrt(head_dim)` times the square of YaRN's temperature factor.
/// \tparam C The configuration. \tparam T The number type.
/// \return The scale, computed in `T`.
template <transformer_config C, math::math_number T>
constexpr T latent_attention_scale()
{
    constexpr T f = yarn_attention_factor_v<C, T>;
    return T(1) / math::sqrt(T(C.head_dim)) * (f * f);
}

/// \brief The weights of a latent attention, laid out as DeepSeek-V3's: views the caller fills.
/// \ingroup core
/// \tparam W A `weight_matrix`.
/// \tparam T The number type of the norms and sinks.
template <weight_matrix W, math::math_number T = float>
struct latent_attention_weights
{
    W q_down{};                    ///< `W^DQ`, `q_lora_rank x d_model`; unused when `q_lora_rank` is 0.
    std::span<const T> q_norm{};   ///< `q_lora_rank`, or empty for a parameter-free norm.
    W q_up{};                      ///< `[W^UQ; W^QR]` per head, `n_heads · head_dim x (q_lora_rank or d_model)`.
    W kv_down{};                   ///< `[W^DKV; W^KR]`, `(kv_lora_rank + qk_rope_dim) x d_model`.
    std::span<const T> kv_norm{};  ///< `kv_lora_rank`, or empty.
    W kv_up{};                     ///< `[W^UK; W^UV]` per head, `n_heads · (qk_nope_dim + v_head_dim) x kv_lora_rank`.
    W o{};                         ///< `W^O`, `d_model x n_heads · v_head_dim`.
    std::span<const T> sinks{};    ///< `n_heads` sink logits with `attention_sinks`, else empty.
};

/// \brief What `latent_attention_forward_sequence` saves for the backward pass: views the caller owns, token by token,
/// a batch's sequences end to end.
/// \ingroup core
/// \tparam T The number type.
template <math::math_number T>
struct latent_attention_activations
{
    std::span<T> cq_raw{};    ///< `rows · q_lora_rank`: `W^DQ x`; empty without query compression.
    std::span<T> cq{};        ///< `rows · q_lora_rank`: the normalised query latent; empty without.
    std::span<T> q{};         ///< `rows · n_heads · head_dim`: the queries, rotary parts rotated.
    std::span<T> ckv_raw{};   ///< `rows · (kv_lora_rank + qk_rope_dim)`: `kv_down x`, the rotary key not yet rotated.
    std::span<T> ckv{};       ///< `rows · kv_lora_rank`: the normalised latent `c^KV`.
    std::span<T> k_rope{};    ///< `rows · qk_rope_dim`: the rotated shared key `k^R`.
    std::span<T> kv{};        ///< `rows · n_heads · (qk_nope_dim + v_head_dim)`: each head's `k^C` then `v`.
    std::span<T> y{};         ///< `rows · n_heads · v_head_dim`: the heads' outputs, `W^O`'s input.
    std::span<T> lse{};       ///< `rows · n_heads`: each head's log-sum-exp.
    std::span<const std::uint32_t> selected{};   ///< `rows · sparse_top_k` positions in the token's sequence: the keys each
                                                 ///< token attends to, the first `min(sparse_top_k, visible)` used; read only.
};

/// \brief The gradients of a latent attention, and its working spaces: views the caller owns.
/// \ingroup core
/// \tparam T The number type.
template <math::math_number T>
struct latent_attention_gradients
{
    std::span<T> q_down{};    ///< Added into, `W^DQ`'s size (empty without query compression).
    std::span<T> q_norm{};    ///< Added into, `q_lora_rank`; or empty.
    std::span<T> q_up{};      ///< Added into.
    std::span<T> kv_down{};   ///< Added into.
    std::span<T> kv_norm{};   ///< Added into, `kv_lora_rank`; or empty.
    std::span<T> kv_up{};     ///< Added into.
    std::span<T> o{};         ///< Added into.
    std::span<T> sinks{};     ///< Added into, `n_heads`; with `attention_sinks`, else empty.
    std::span<T> y{};         ///< Working space, `rows · n_heads · v_head_dim`.
    std::span<T> q{};         ///< Written, `rows · n_heads · head_dim`: the gradients in the projected queries.
    std::span<T> kv{};        ///< Written, `rows · n_heads · (qk_nope_dim + v_head_dim)`.
    std::span<T> k_rope{};    ///< Written, `rows · qk_rope_dim`.
    std::span<T> ckv{};       ///< Written, `rows · kv_lora_rank`.
    std::span<T> ckv_raw{};   ///< Written, `rows · (kv_lora_rank + qk_rope_dim)`.
    std::span<T> cq{};        ///< Written, `rows · q_lora_rank`; empty without query compression.
    std::span<T> cq_raw{};    ///< Written, `rows · q_lora_rank`; empty without.
};

namespace detail::latent_attention
{
    template <transformer_config C>
    struct dims
    {
        static_assert(valid(C), "snicholls::ml::latent_attention: the transformer_config is not valid");
        static_assert(C.latent(), "snicholls::ml::latent_attention: the transformer_config has no kv_lora_rank");
        static constexpr std::size_t h = C.n_heads, dqk = C.head_dim, dn = C.qk_nope_dim(), dr = C.qk_rope_dim,
                                     dv = C.v_head_dim, dc = C.kv_lora_rank, dq = C.q_lora_rank, d = C.d_model,
                                     kvw = dn + dv, qw = h * dqk, kvn = h * kvw, yw = h * dv, cw = dc + dr;
    };

    template <typename T>
    constexpr T sink_of(std::span<const T> sinks, std::size_t h, bool on)
    {
        return on ? sinks[h] : -std::numeric_limits<T>::infinity();
    }

    // The keys token t of a sequence attends to: their count, and the position of key j
    template <transformer_config C>
    struct keys
    {
        key_range r;
        std::span<const std::uint32_t> sel;
        [[nodiscard]] constexpr std::size_t size() const
        {
            return C.sparse_top_k == 0 || sel.empty() ? r.size() : std::min(sel.size(), r.size());
        }
        [[nodiscard]] constexpr std::size_t operator[](std::size_t j) const
        {
            if (C.sparse_top_k == 0 || sel.empty())
                return r.first + j;
            assert(sel[j] >= r.first && sel[j] < r.last);
            return sel[j];
        }
    };

    template <transformer_config C, typename T>
    constexpr keys<C> keys_of(const latent_attention_activations<T>& act, std::size_t row, std::size_t t)
    {
        const key_range r = attention_window(t, C.window);
        if constexpr (C.sparse_top_k > 0) {
            assert(act.selected.size() >= (row + 1) * C.sparse_top_k);
            return {r, act.selected.subspan(row * C.sparse_top_k, C.sparse_top_k)};
        } else {
            return {r, {}};
        }
    }
} // namespace detail::latent_attention

/// \name The steps of the latent attention's batched passes, on the backend (TRAINING.md T9)
/// Each runs on the host for any backend, sequence by sequence and token by token; a device library overloads them for
/// its own backend type.
/// @{

/// \brief The RMS norms of the latents: `cq = norm(cq_raw)` and `ckv = norm(ckv_raw[:kv_lora_rank])` for every row.
/// \tparam C The configuration. \param w The weights (the norms'). \param act The activations. \param rows The rows.
template <transformer_config C, typename B, typename W, math::math_number T>
constexpr void latent_attention_norm_rows(const B&, const latent_attention_weights<W, T>& w, const latent_attention_activations<T>& act,
                                          std::size_t rows)
{
    using D = detail::latent_attention::dims<C>;
    constexpr T eps = norm_eps_v<C, T>;
    for (std::size_t r = 0; r < rows; ++r) {
        if constexpr (D::dq > 0)
            rms_norm(act.cq_raw.subspan(r * D::dq, D::dq), w.q_norm, act.cq.subspan(r * D::dq, D::dq), eps);
        rms_norm(act.ckv_raw.subspan(r * D::cw, D::dc), w.kv_norm, act.ckv.subspan(r * D::dc, D::dc), eps);
    }
}

/// \brief The decoupled RoPE: each token's shared key `k_rope = RoPE(ckv_raw[kv_lora_rank:])` and its heads' rotary
/// query channels, in place, at the token's position in its sequence.
/// \tparam C The configuration. \param act The activations. \param n The tokens of each sequence.
/// \param rope_src A rope source of `qk_rope_dim` covering `n - 1`. \param sequences The sequences.
template <transformer_config C, typename B, math::math_number T, typename R>
constexpr void latent_attention_rope_rows(const B&, const latent_attention_activations<T>& act, std::size_t n, const R& rope_src,
                                          std::size_t sequences = 1)
{
    using D = detail::latent_attention::dims<C>;
    for (std::size_t s = 0; s < sequences; ++s)
        for (std::size_t t = 0; t < n; ++t) {
            const std::size_t row = s * n + t;
            const auto kr = act.k_rope.subspan(row * D::dr, D::dr);
            std::copy_n(act.ckv_raw.begin() + std::ptrdiff_t(row * D::cw + D::dc), D::dr, kr.begin());
            rope(kr, rope_src.cos_at(t), rope_src.sin_at(t), C.layout());
            for (std::size_t h = 0; h < D::h; ++h)
                rope(act.q.subspan(row * D::qw + h * D::dqk + D::dn, D::dr), rope_src.cos_at(t), rope_src.sin_at(t), C.layout());
        }
}

/// \brief Every token's heads over the keys it sees in its own sequence (all visible, or its selection): `act.y` and
/// `act.lse`. \tparam C The configuration. \param w The weights (the sinks). \param act The activations.
/// \param n The tokens of each sequence. \param sequences The sequences.
template <transformer_config C, typename B, typename W, math::math_number T>
constexpr void latent_attention_attend_rows(const B&, const latent_attention_weights<W, T>& w, const latent_attention_activations<T>& act,
                                            std::size_t n, std::size_t sequences = 1)
{
    using D = detail::latent_attention::dims<C>;
    const T scale = latent_attention_scale<C, T>();
    for (std::size_t s = 0; s < sequences; ++s)
        for (std::size_t t = 0; t < n; ++t) {
            const std::size_t row = s * n + t, base = s * n;
            const auto keys = detail::latent_attention::keys_of<C>(act, row, t);
            for (std::size_t h = 0; h < D::h; ++h) {
                const auto q = std::span<const T>(act.q).subspan(row * D::qw + h * D::dqk, D::dqk);
                act.lse[row * D::h + h] = detail::attention_sink::forward(
                    keys.size(),
                    [&](std::size_t j) {
                        const std::size_t k = base + keys[j];
                        T acc = T(0);
                        for (std::size_t i = 0; i < D::dn; ++i)
                            acc += q[i] * act.kv[k * D::kvn + h * D::kvw + i];
                        for (std::size_t i = 0; i < D::dr; ++i)
                            acc += q[D::dn + i] * act.k_rope[k * D::dr + i];
                        return scale * acc;
                    },
                    [&](std::size_t j) { return std::span<const T>(act.kv).subspan((base + keys[j]) * D::kvn + h * D::kvw + D::dn, D::dv); },
                    act.y.subspan(row * D::yw + h * D::dv, D::dv), detail::latent_attention::sink_of(w.sinks, h, C.attention_sinks));
            }
        }
}

/// \brief The backward pass of `latent_attention_attend_rows`, from the heads' output gradients `dy`: `grads.q` written;
/// `grads.kv`, `grads.k_rope` and `grads.sinks` added into (zero the first two before).
/// \tparam C The configuration. \param w The weights. \param act The activations. \param dy The gradients in `act.y`.
/// \param grads The gradients. \param n The tokens of each sequence. \param sequences The sequences.
template <transformer_config C, typename B, typename W, math::math_number T>
constexpr void latent_attention_attend_rows_backward(const B&, const latent_attention_weights<W, T>& w,
                                                     const latent_attention_activations<T>& act, std::type_identity_t<std::span<const T>> dy,
                                                     const latent_attention_gradients<T>& grads, std::size_t n, std::size_t sequences = 1)
{
    using D = detail::latent_attention::dims<C>;
    const T scale = latent_attention_scale<C, T>();
    for (std::size_t s = 0; s < sequences; ++s)
        for (std::size_t t = 0; t < n; ++t) {
            const std::size_t row = s * n + t, base = s * n;
            const auto keys = detail::latent_attention::keys_of<C>(act, row, t);
            for (std::size_t h = 0; h < D::h; ++h) {
                const auto q = std::span<const T>(act.q).subspan(row * D::qw + h * D::dqk, D::dqk);
                const auto gq = grads.q.subspan(row * D::qw + h * D::dqk, D::dqk);
                const auto g = dy.subspan(row * D::yw + h * D::dv, D::dv);
                std::fill(gq.begin(), gq.end(), T(0));
                const T sink = detail::latent_attention::sink_of(w.sinks, h, C.attention_sinks);
                const T dsink = detail::attention_sink::backward(
                    keys.size(),
                    [&](std::size_t j) {
                        const std::size_t k = base + keys[j];
                        T acc = T(0);
                        for (std::size_t i = 0; i < D::dn; ++i)
                            acc += q[i] * act.kv[k * D::kvn + h * D::kvw + i];
                        for (std::size_t i = 0; i < D::dr; ++i)
                            acc += q[D::dn + i] * act.k_rope[k * D::dr + i];
                        return scale * acc;
                    },
                    [&](std::size_t j) { return std::span<const T>(act.kv).subspan((base + keys[j]) * D::kvn + h * D::kvw + D::dn, D::dv); },
                    std::span<const T>(act.y).subspan(row * D::yw + h * D::dv, D::dv), g, act.lse[row * D::h + h], sink,
                    [&](std::size_t j, T p, T ds) {
                        const std::size_t k = base + keys[j], at = k * D::kvn + h * D::kvw;
                        for (std::size_t i = 0; i < D::dv; ++i)
                            grads.kv[at + D::dn + i] += p * g[i];
                        for (std::size_t i = 0; i < D::dn; ++i) {
                            gq[i] += scale * ds * act.kv[at + i];
                            grads.kv[at + i] += scale * ds * q[i];
                        }
                        for (std::size_t i = 0; i < D::dr; ++i) {
                            gq[D::dn + i] += scale * ds * act.k_rope[k * D::dr + i];
                            grads.k_rope[k * D::dr + i] += scale * ds * q[D::dn + i];
                        }
                    });
                if constexpr (C.attention_sinks)
                    grads.sinks[h] += dsink;
            }
        }
}

/// \brief The backward pass of `latent_attention_rope_rows`, in place: the rotary channels of `grads.q` and every row
/// of `grads.k_rope` rotated back (`rope_inverse`). \tparam C The configuration. \param grads The gradients.
/// \param n The tokens of each sequence. \param rope_src The rope source. \param sequences The sequences.
template <transformer_config C, typename B, math::math_number T, typename R>
constexpr void latent_attention_rope_rows_backward(const B&, const latent_attention_gradients<T>& grads, std::size_t n, const R& rope_src,
                                                   std::size_t sequences = 1)
{
    using D = detail::latent_attention::dims<C>;
    for (std::size_t s = 0; s < sequences; ++s)
        for (std::size_t t = 0; t < n; ++t) {
            const std::size_t row = s * n + t;
            rope_inverse(grads.k_rope.subspan(row * D::dr, D::dr), rope_src.cos_at(t), rope_src.sin_at(t), C.layout());
            for (std::size_t h = 0; h < D::h; ++h)
                rope_inverse(grads.q.subspan(row * D::qw + h * D::dqk + D::dn, D::dr), rope_src.cos_at(t), rope_src.sin_at(t),
                             C.layout());
        }
}

/// \brief The backward pass of `latent_attention_norm_rows`: `grads.cq_raw` from `grads.cq`, and `grads.ckv_raw` as the
/// latent's norm backward from `grads.ckv` beside the rotary key's gradient `grads.k_rope`; the norms' weights' gradients
/// added into. \tparam C The configuration. \param w The weights. \param act The activations. \param grads The gradients.
/// \param rows The rows.
template <transformer_config C, typename B, typename W, math::math_number T>
constexpr void latent_attention_norm_rows_backward(const B&, const latent_attention_weights<W, T>& w,
                                                   const latent_attention_activations<T>& act, const latent_attention_gradients<T>& grads,
                                                   std::size_t rows)
{
    using D = detail::latent_attention::dims<C>;
    constexpr T eps = norm_eps_v<C, T>;
    std::array<T, D::dc> scratch{};
    for (std::size_t r = 0; r < rows; ++r) {
        if constexpr (D::dq > 0) {
            std::array<T, D::dq> sq{};
            const auto x = std::span<const T>(act.cq_raw).subspan(r * D::dq, D::dq);
            const T rstd = rms_norm(x, w.q_norm, sq, eps);
            rms_norm_backward(std::span<const T>(grads.cq).subspan(r * D::dq, D::dq), x, w.q_norm, rstd,
                              grads.cq_raw.subspan(r * D::dq, D::dq), grads.q_norm);
        }
        const auto x = std::span<const T>(act.ckv_raw).subspan(r * D::cw, D::dc);
        const T rstd = rms_norm(x, w.kv_norm, scratch, eps);
        rms_norm_backward(std::span<const T>(grads.ckv).subspan(r * D::dc, D::dc), x, w.kv_norm, rstd,
                          grads.ckv_raw.subspan(r * D::cw, D::dc), grads.kv_norm);
        std::copy_n(grads.k_rope.begin() + std::ptrdiff_t(r * D::dr), D::dr, grads.ckv_raw.begin() + std::ptrdiff_t(r * D::cw + D::dc));
    }
}

/// @}

/// \brief A latent attention over a batch of sequences, its products through a `linalg` backend, saving what the
/// backward pass needs.
/// \tparam C A latent configuration.
/// \param backend A `linalg::blas_backend`.
/// \param w The weights, dense in the sequence's type.
/// \param xs `sequences · n` tokens of `d_model`, end to end (normalised inputs).
/// \param rope_src A rope source of `qk_rope_dim` channels covering `n - 1`.
/// \param act Receives the activations (its `selected` is read with `sparse_top_k`).
/// \param out Receives `sequences · n` outputs of `d_model`; may not alias `xs`.
/// \param sequences The sequences, at least one; attention stays within each, positions restarting at 0.
template <transformer_config C, typename B, math::math_number T, typename R>
    requires linalg::blas_backend<B, T> && rope_source<R>
constexpr void latent_attention_forward_sequence(const B& backend, const latent_attention_weights<dense_weights<T>, T>& w,
                                                 std::type_identity_t<std::span<const T>> xs, const R& rope_src,
                                                 const latent_attention_activations<T>& act, std::type_identity_t<std::span<T>> out,
                                                 std::size_t sequences = 1)
{
    using D = detail::latent_attention::dims<C>;
    const std::size_t rows = xs.size() / D::d, n = rows / sequences;
    assert(sequences > 0 && xs.size() == n * sequences * D::d && out.size() == xs.size());
    assert(act.q.size() == rows * D::qw && act.ckv_raw.size() == rows * D::cw && act.ckv.size() == rows * D::dc
           && act.k_rope.size() == rows * D::dr && act.kv.size() == rows * D::kvn && act.y.size() == rows * D::yw
           && act.lse.size() == rows * D::h && (D::dq == 0 || (act.cq_raw.size() == rows * D::dq && act.cq.size() == rows * D::dq)));
    assert(!C.attention_sinks || w.sinks.size() == D::h);
    if constexpr (D::dq > 0)
        linear_sequence(backend, w.q_down, xs, act.cq_raw);
    linear_sequence(backend, w.kv_down, xs, act.ckv_raw);
    latent_attention_norm_rows<C>(backend, w, act, rows);
    if constexpr (D::dq > 0)
        linear_sequence(backend, w.q_up, std::span<const T>(act.cq), act.q);
    else
        linear_sequence(backend, w.q_up, xs, act.q);
    latent_attention_rope_rows<C>(backend, act, n, rope_src, sequences);
    linear_sequence(backend, w.kv_up, std::span<const T>(act.ckv), act.kv);
    latent_attention_attend_rows<C>(backend, w, act, n, sequences);
    linear_sequence(backend, w.o, std::span<const T>(act.y), out);
}

/// \brief The backward pass of `latent_attention_forward_sequence`, from what it saved: every weight's, norm's and sink's
/// gradient added into, the input's added into `dxs`.
///
/// `W^O`'s backward, the heads' (one pass per head from the saved log-sum-exp), RoPE undone, `kv_up`'s and `q_up`'s
/// backward into the latents, the norms', then `kv_down`'s and `q_down`'s into the input.
/// \tparam C A latent configuration.
/// \param backend A `linalg::blas_backend`.
/// \param w The weights the forward pass used.
/// \param xs The tokens it read.
/// \param rope_src Its rope source.
/// \param act What it saved.
/// \param dout The gradient in its output.
/// \param dxs Added into, `xs`' size; or empty.
/// \param grads The gradients and working spaces.
/// \param sequences The sequences the forward pass took.
template <transformer_config C, typename B, math::math_number T, typename R>
    requires linalg::blas_backend<B, T> && rope_source<R>
constexpr void latent_attention_backward(const B& backend, const latent_attention_weights<dense_weights<T>, T>& w,
                                         std::type_identity_t<std::span<const T>> xs, const R& rope_src,
                                         const latent_attention_activations<T>& act, std::type_identity_t<std::span<const T>> dout,
                                         std::type_identity_t<std::span<T>> dxs, const latent_attention_gradients<T>& grads,
                                         std::size_t sequences = 1)
{
    using D = detail::latent_attention::dims<C>;
    const std::size_t rows = xs.size() / D::d, n = rows / sequences;
    assert(sequences > 0 && rows == n * sequences && dout.size() == xs.size() && (dxs.empty() || dxs.size() == xs.size()));
    assert(grads.y.size() == rows * D::yw && grads.q.size() == rows * D::qw && grads.kv.size() == rows * D::kvn
           && grads.k_rope.size() == rows * D::dr && grads.ckv.size() == rows * D::dc && grads.ckv_raw.size() == rows * D::cw
           && (D::dq == 0 || (grads.cq.size() == rows * D::dq && grads.cq_raw.size() == rows * D::dq)));
    fill_rows(backend, grads.y, T(0));
    linear_sequence_backward(backend, w.o, std::span<const T>(act.y), dout, grads.y, grads.o);
    fill_rows(backend, grads.kv, T(0));
    fill_rows(backend, grads.k_rope, T(0));
    latent_attention_attend_rows_backward<C>(backend, w, act, std::span<const T>(grads.y), grads, n, sequences);
    latent_attention_rope_rows_backward<C>(backend, grads, n, rope_src, sequences);
    fill_rows(backend, grads.ckv, T(0));
    linear_sequence_backward(backend, w.kv_up, std::span<const T>(act.ckv), std::span<const T>(grads.kv), grads.ckv, grads.kv_up);
    if constexpr (D::dq > 0) {
        fill_rows(backend, grads.cq, T(0));
        linear_sequence_backward(backend, w.q_up, std::span<const T>(act.cq), std::span<const T>(grads.q), grads.cq, grads.q_up);
    } else {
        linear_sequence_backward(backend, w.q_up, xs, std::span<const T>(grads.q), dxs, grads.q_up);
    }
    latent_attention_norm_rows_backward<C>(backend, w, act, grads, rows);
    linear_sequence_backward(backend, w.kv_down, xs, std::span<const T>(grads.ckv_raw), dxs, grads.kv_down);
    if constexpr (D::dq > 0)
        linear_sequence_backward(backend, w.q_down, xs, std::span<const T>(grads.cq_raw), dxs, grads.q_down);
}

/// \brief A latent KV cache an attention can use: `append(layer, pos, row)` one token's `[c^KV; k^R]`, and `row(layer,
/// pos)` it back.
template <typename S, typename T>
concept latent_kv_store = requires(const S& s, std::size_t n, std::span<const T> r) {
    s.append(n, n, r);
    { s.row(n, n) } -> std::convertible_to<std::span<const T>>;
};

/// \brief A latent KV cache over the caller's memory: `[layer][slot][kv_lora_rank + qk_rope_dim]`, position `p` in slot
/// `p % capacity`; each row the normalised latent `c^KV` then the rotated key `k^R`.
/// \ingroup core
/// \tparam C A latent configuration.
/// \tparam T The number type.
template <transformer_config C, math::math_number T = float>
class latent_kv_cache_view
{
    static_assert(valid(C) && C.latent(), "snicholls::ml::latent_kv_cache_view: the transformer_config is not a valid latent one");

public:
    /// \brief A row's width: `kv_cache_elements_per_token(C)`.
    static constexpr std::size_t width = C.kv_lora_rank + C.qk_rope_dim;

    /// \brief The elements the storage needs.
    /// \param layers The layers. \param capacity The slots per layer.
    /// \return `layers · capacity · width`.
    [[nodiscard]] static constexpr std::size_t required_size(std::size_t layers, std::size_t capacity) noexcept
    {
        return layers * capacity * width;
    }

    /// \brief An empty view.
    constexpr latent_kv_cache_view() noexcept = default;

    /// \brief A view over the caller's storage.
    /// \param rows `required_size(layers, capacity)` elements.
    /// \param layers The layers.
    /// \param capacity The slots per layer: the longest sequence, or at least the window.
    constexpr latent_kv_cache_view(std::span<T> rows, std::size_t layers, std::size_t capacity) noexcept
        : rows_(rows), layers_(layers), capacity_(capacity)
    {
        assert(rows.size() == required_size(layers, capacity));
        assert(C.window == 0 || capacity == 0 || capacity >= C.window);
    }

    /// \brief Stores one position's row.
    /// \param layer The layer. \param pos The position, in slot `pos % capacity`. \param r `width` numbers.
    template <row Row>
        requires std::same_as<row_value_t<Row>, T>
    constexpr void append(std::size_t layer, std::size_t pos, const Row& r) const
    {
        const auto x = as_row(r);
        assert(layer < layers_ && capacity_ > 0 && x.size() == width);
        std::copy(x.begin(), x.end(), rows_.begin() + std::ptrdiff_t(offset(layer, pos)));
    }

    /// \brief One position's row: `[c^KV; k^R]`.
    /// \param layer The layer. \param pos The position, among the last `capacity` appended.
    /// \return `width` numbers.
    [[nodiscard]] constexpr std::span<const T> row(std::size_t layer, std::size_t pos) const
    {
        assert(layer < layers_ && capacity_ > 0);
        return std::span<const T>(rows_).subspan(offset(layer, pos), width);
    }

    /// \brief The layers. \return The number given at construction.
    [[nodiscard]] constexpr std::size_t layers() const noexcept { return layers_; }
    /// \brief The slots per layer. \return The number given at construction.
    [[nodiscard]] constexpr std::size_t capacity() const noexcept { return capacity_; }

private:
    constexpr std::size_t offset(std::size_t layer, std::size_t pos) const noexcept { return (layer * capacity_ + pos % capacity_) * width; }

    std::span<T> rows_{};
    std::size_t layers_ = 0;
    std::size_t capacity_ = 0;
};

/// \brief A latent KV cache that owns fixed-size storage.
/// \ingroup core
/// \tparam C A latent configuration. \tparam T The number type. \tparam Layers The layers. \tparam Capacity The slots.
template <transformer_config C, math::math_number T, std::size_t Layers, std::size_t Capacity>
class latent_kv_cache
{
public:
    /// \brief A view of this cache, for the attention functions.
    /// \return The view.
    [[nodiscard]] constexpr latent_kv_cache_view<C, T> view() noexcept { return {rows_, Layers, Capacity}; }

private:
    std::array<T, latent_kv_cache_view<C, T>::required_size(Layers, Capacity)> rows_{};
};

/// \brief A latent attention for one token at position `pos`, in the absorbed form: projects, normalises, rotates,
/// caches `[c^KV; k^R]`, and attends to the cached rows with each head's query taken into the latent space.
/// \tparam C A latent configuration.
/// \param w The weights: dense (the absorbed query is a product by `kv_up`'s transpose).
/// \param x The token's (normalised) input, `d_model`.
/// \param pos Its position.
/// \param cache The latent cache; this token's row is appended to `layer` at `pos`.
/// \param layer The layer.
/// \param rope_src A rope source of `qk_rope_dim` covering `pos`.
/// \param out The output, `d_model`; may not alias `x`.
/// \param selected With `sparse_top_k`, the positions to attend to (each visible), or empty for every visible one.
template <transformer_config C, typename S, math::math_number T, row X, latent_kv_store<T> Cache, typename R, mutable_row Out>
    requires std::same_as<row_value_t<X>, T> && std::same_as<row_value_t<Out>, T> && rope_source<R>
constexpr void latent_attention_forward(const latent_attention_weights<dense_weights<S>, T>& w, const X& x, std::size_t pos,
                                        const Cache& cache, std::size_t layer, const R& rope_src, Out&& out,
                                        std::span<const std::uint32_t> selected = {})
{
    using D = detail::latent_attention::dims<C>;
    constexpr T eps = norm_eps_v<C, T>;
    const auto xs = as_row(x);
    assert(xs.size() == D::d && w.kv_up.rows == D::kvn && w.kv_up.cols == D::dc && (!C.attention_sinks || w.sinks.size() == D::h));
    std::array<T, D::qw> q{};
    if constexpr (D::dq > 0) {
        std::array<T, D::dq> cq{};
        linear(w.q_down, xs, cq);
        rms_norm(cq, w.q_norm, cq, eps);
        linear(w.q_up, cq, q);
    } else {
        linear(w.q_up, xs, q);
    }
    std::array<T, D::cw> c{};
    linear(w.kv_down, xs, c);
    const auto latent = std::span<T>(c).first(D::dc);
    rms_norm(latent, w.kv_norm, latent, eps);
    rope(std::span<T>(c).subspan(D::dc, D::dr), rope_src.cos_at(pos), rope_src.sin_at(pos), C.layout());
    cache.append(layer, pos, c);
    const auto keys = detail::latent_attention::keys<C>{attention_window(pos, C.window), selected};
    const T scale = latent_attention_scale<C, T>();
    std::array<T, D::yw> y{};
    std::array<T, D::cw> qa{};
    std::array<T, D::dc> o{};
    for (std::size_t h = 0; h < D::h; ++h) {
        const auto qh = std::span<T>(q).subspan(h * D::dqk, D::dqk);
        rope(qh.subspan(D::dn, D::dr), rope_src.cos_at(pos), rope_src.sin_at(pos), C.layout());
        for (std::size_t k = 0; k < D::dc; ++k) {   // (W^UK_h)ᵀ q^C_h
            T acc = T(0);
            for (std::size_t i = 0; i < D::dn; ++i) {
                const T p = detail::linear::widen<T>(w.kv_up.row(h * D::kvw + i)[k]) * qh[i];
                acc += p;
            }
            qa[k] = acc;
        }
        std::copy_n(qh.begin() + std::ptrdiff_t(D::dn), D::dr, qa.begin() + std::ptrdiff_t(D::dc));
        detail::attention_sink::forward(
            keys.size(), [&](std::size_t j) { return scale * detail::attention::dot(std::span<const T>(qa), cache.row(layer, keys[j])); },
            [&](std::size_t j) { return cache.row(layer, keys[j]).first(D::dc); }, std::span<T>(o),
            detail::latent_attention::sink_of(w.sinks, h, C.attention_sinks));
        for (std::size_t v = 0; v < D::dv; ++v) {   // W^UV_h ō_h
            const auto r = w.kv_up.row(h * D::kvw + D::dn + v);
            T acc = T(0);
            for (std::size_t k = 0; k < D::dc; ++k) {
                const T p = detail::linear::widen<T>(r[k]) * o[k];
                acc += p;
            }
            y[h * D::dv + v] = acc;
        }
    }
    linear(w.o, y, out);
}

} // namespace snicholls::ml

#endif // snicholls_core_ml_latent_attention_hpp
