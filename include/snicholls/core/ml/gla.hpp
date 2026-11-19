//
//  gla.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::ml` gated linear attention (GLA): the recurrent form, the chunked form, the backward pass, and one token at a time for decoding.
/// \ingroup core
///
/// **Overview.** Gated linear attention (Yang, Wang, Shen, Panda and Kim, 2023, section 4.1 eq. 3) is linear attention
/// whose state decays by a data-dependent gate per key channel, `S_t = Diag(α_t) S_{t-1} + k_tᵀ v_t`, `o_t = q_t S_t`.
/// Here the gate is given by its logarithm, `gk = log α ≤ 0` (as flash-linear-attention's `gk`; the paper's
/// `α = σ(x W_α¹ W_α² + b)^{1/16}` is `gk = log_sigmoid(...) / 16`, section 4.4, the layer's business), and the queries
/// are scaled by `1 / sqrt(N)` (`mixer_scale_v`). Per head `h` of `P` value channels, in key group `g = h / (H / G)`:
///
///     S_t[n][p] = S_{t-1}[n][p] · exp(gk_t[n]) + v_t[p] · k_t[n],     o_t[p] = Σ_n S_t[n][p] · (q_t[n] · scale)
///
/// which is llama.cpp's `GGML_OP_GATED_LINEAR_ATTN` (commit 39039aa, `ggml_compute_forward_gla_f32`, the same products
/// in the same order; tested against a transcription), with the gate passed as its logarithm.
///
/// The chunked form is the paper's (section 4.2, eq. 4, in log space): within a chunk with cumulative gates
/// `G_j[n] = Σ_{i≤j} gk_i[n]`, `o_j = scale · (Σ_n q_j[n] e^{G_j[n]} S_prev[n] + Σ_{i≤j} (Σ_n q_j[n] k_i[n]
/// e^{G_j[n] - G_i[n]}) v_i)`, and the state passed on is `e^{G_last} ⊙ S_prev + Σ_i (e^{G_last - G_i} ⊙ k_i) v_iᵀ`. Every
/// exponent is of a difference that is not positive, so nothing overflows; the paper's secondary chunking, which exists
/// to run the quadratic part on tensor cores, is a device's form of the same sums.
///
/// - `gla_forward<C>(backend, in, o, sequences, states, checkpoints)` and `gla_forward_chunked<C>(..., work)`.
/// - `gla_backward<C>(backend, in, dout, grads, sequences, checkpoints, dstates, work)`.
/// - `gla_step<C>(in, o, state)`: decoding, with `gla_state_size(C)` values.
///
/// \snippet ml/gla_demo.cpp gla
///
/// **Relationships.** `ml/recurrence.hpp`, `ml/rwkv.hpp` (RWKV-6 is this recurrence read before the update, with a
/// bonus), `ml/delta_rule.hpp`, `ml/transformer_config.hpp` (`N` is `mixer_state`, the key channels). Group: \ref core.
///
/// **Implementations.** The recurrent and chunked forms; one backward pass for both.
///
/// **Guarantees.** `constexpr`, no allocation (the chunked form's cumulative gates are in `work`), sums in a stated order
/// (the recurrent output over `n` ascending, as ggml's). The chunked form is the recurrent within rounding (tested); the
/// step is the recurrent forward to the bit. Each row's input gradients are written.
///
/// **Errors.** Preconditions, asserted: `q` and `k` `rows · G · N`, `v` and `o` `rows · H · P`, `gk` `rows · H · N`.
///
/// **Performance.** `benchmarks/ml/gla_benchmark.cpp`, the recurrent form against the loop by hand; BENCHMARKS.md.
///
/// **Tests and demos.** `tests/ml/gla_test.cpp`, `demos/ml/gla_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<cassert>`, `<cstddef>`, `<span>`, `<type_traits>`, `math/constexpr_math.hpp`,
/// `ml/recurrence.hpp`, `ml/transformer_config.hpp`.
///
/// **Known issues.** No layer yet (`mixer_kind::gla` is the op's shape): the projections, low-rank gate and output gate
/// of the paper's section 4.4 are not here.
///
/// **Status.** Experimental, new in 0.17.0.

#ifndef snicholls_core_ml_gla_hpp
#define snicholls_core_ml_gla_hpp

#include <snicholls/core/math/constexpr_math.hpp>
#include <snicholls/core/ml/recurrence.hpp>
#include <snicholls/core/ml/transformer_config.hpp>

#include <array>
#include <cassert>
#include <cstddef>
#include <span>
#include <type_traits>

namespace snicholls::ml
{

/// \brief The inputs of gated linear attention over `rows` rows: views the caller owns.
/// \ingroup core
/// \tparam T The number type.
template <math::math_number T>
struct gla_inputs
{
    std::span<const T> q{};    ///< `rows · G · N`: the queries.
    std::span<const T> k{};    ///< `rows · G · N`: the keys.
    std::span<const T> v{};    ///< `rows · H · P`: the values.
    std::span<const T> gk{};   ///< `rows · H · N`: the log-gates, not positive.
};

/// \brief The gradients of gated linear attention, shaped as `gla_inputs`, every one written.
/// \ingroup core
/// \tparam T The number type.
template <math::math_number T>
struct gla_gradients
{
    std::span<T> q{};    ///< Written.
    std::span<T> k{};    ///< Written.
    std::span<T> v{};    ///< Written.
    std::span<T> gk{};   ///< Written.
};

/// \brief A sequence's state: `H · N · P` values. \param c The configuration. \return The number of values.
[[nodiscard]] constexpr std::size_t gla_state_size(const transformer_config& c) noexcept
{
    return mixer_state_size(c);
}

/// \brief The working space of `gla_forward_chunked`: a chunk's cumulative gates.
/// \param c The configuration. \return `mixer_chunk · mixer_state`.
[[nodiscard]] constexpr std::size_t gla_chunk_work(const transformer_config& c) noexcept
{
    return c.mixer_chunk * c.mixer_state;
}

namespace detail::gla
{
    template <transformer_config C, typename T>
    struct scan
    {
        using value_type = T;
        static constexpr std::size_t H = C.mixer_heads, P = C.mixer_head_dim, N = C.mixer_state, G = C.mixer_groups;
        static constexpr std::size_t state_size = H * N * P, work_size = 0;
        static constexpr T scale = mixer_scale_v<C, T>;

        gla_inputs<T> in{};
        std::span<T> o{};
        std::span<const T> dout{};
        gla_gradients<T> g{};

        constexpr void update(std::size_t row, std::span<T> state, bool output) const
        {
            for (std::size_t h = 0; h < H; ++h) {
                const std::size_t grp = h / (H / G);
                const auto q = in.q.subspan((row * G + grp) * N, N), k = in.k.subspan((row * G + grp) * N, N);
                const auto v = in.v.subspan((row * H + h) * P, P);
                const auto s = state.subspan(h * N * P, N * P);
                std::array<T, P> out{};
                for (std::size_t n = 0; n < N; ++n) {
                    const T qn = q[n] * scale, gate = math::exp(in.gk[(row * H + h) * N + n]);
                    for (std::size_t p = 0; p < P; ++p) {
                        s[n * P + p] = s[n * P + p] * gate + v[p] * k[n];
                        out[p] += s[n * P + p] * qn;
                    }
                }
                if (output)
                    for (std::size_t p = 0; p < P; ++p)
                        o[(row * H + h) * P + p] = out[p];
            }
        }
        constexpr void step(std::size_t row, std::span<T> state) const { update(row, state, true); }
        constexpr void advance(std::size_t row, std::span<T> state) const { update(row, state, false); }

        constexpr void step_backward(std::size_t row, std::span<const T> prev, std::span<T> dstate, std::span<T>) const
        {
            for (std::size_t i = 0; i < G * N; ++i)
                g.q[row * G * N + i] = g.k[row * G * N + i] = T(0);
            for (std::size_t h = 0; h < H; ++h) {
                const std::size_t grp = h / (H / G);
                const auto q = in.q.subspan((row * G + grp) * N, N), k = in.k.subspan((row * G + grp) * N, N);
                const auto v = in.v.subspan((row * H + h) * P, P);
                const auto d = dout.subspan((row * H + h) * P, P);
                const auto s0 = prev.subspan(h * N * P, N * P);
                const auto ds = dstate.subspan(h * N * P, N * P);
                std::array<T, P> dv{};
                for (std::size_t n = 0; n < N; ++n) {
                    const T gate = math::exp(in.gk[(row * H + h) * N + n]);
                    T dq = T(0), dk = T(0), dg = T(0);
                    for (std::size_t p = 0; p < P; ++p) {
                        const T st = s0[n * P + p] * gate + v[p] * k[n];
                        dq += d[p] * st;
                        const T dsp = ds[n * P + p] + q[n] * scale * d[p];
                        dk += dsp * v[p];
                        dv[p] += dsp * k[n];
                        dg += dsp * s0[n * P + p];
                        ds[n * P + p] = dsp * gate;
                    }
                    g.q[(row * G + grp) * N + n] += dq * scale;
                    g.k[(row * G + grp) * N + n] += dk;
                    g.gk[(row * H + h) * N + n] = dg * gate;
                }
                for (std::size_t p = 0; p < P; ++p)
                    g.v[(row * H + h) * P + p] = dv[p];
            }
        }
    };

    template <transformer_config C, typename T>
    constexpr std::size_t rows_of(const gla_inputs<T>& in)
    {
        [[maybe_unused]] constexpr std::size_t H = C.mixer_heads, P = C.mixer_head_dim, N = C.mixer_state, G = C.mixer_groups;
        static_assert(valid(C) && C.mixer_heads > 0, "snicholls::ml::gla: the configuration has no mixer shape");
        const std::size_t rows = in.v.size() / (H * P);
        assert(in.v.size() == rows * H * P && in.q.size() == rows * G * N && in.k.size() == rows * G * N && in.gk.size() == rows * H * N);
        return rows;
    }
} // namespace detail::gla

/// \brief Gated linear attention over `sequences` sequences laid end to end, token by token.
/// \tparam C The configuration.
/// \param in The inputs. \param o Receives `rows · H · P` outputs. \param sequences The sequences.
/// \param states `sequences · gla_state_size(C)`: the initial states, replaced by the final.
/// \param checkpoints The states at each chunk's start for the backward pass, or empty.
template <transformer_config C, typename B, math::math_number T>
constexpr void gla_forward(const B&, const gla_inputs<T>& in, std::type_identity_t<std::span<T>> o, std::size_t sequences,
                           std::type_identity_t<std::span<T>> states, std::type_identity_t<std::span<T>> checkpoints = {})
{
    const std::size_t rows = detail::gla::rows_of<C>(in);
    assert(sequences > 0 && rows % sequences == 0 && o.size() == in.v.size());
    recurrence_forward(detail::gla::scan<C, T>{in, o, {}, {}}, rows / sequences, sequences, C.mixer_chunk, states, checkpoints);
}

/// \brief Gated linear attention's chunked form, over chunks of `C.mixer_chunk` tokens: the recurrent form's result
/// within rounding, the same checkpoints.
/// \tparam C The configuration.
/// \param in The inputs. \param o Receives the outputs. \param sequences The sequences.
/// \param states The initial states, replaced by the final. \param checkpoints The states at each chunk's start, or empty.
/// \param work `gla_chunk_work(C)` values.
template <transformer_config C, typename B, math::math_number T>
constexpr void gla_forward_chunked(const B&, const gla_inputs<T>& in, std::type_identity_t<std::span<T>> o, std::size_t sequences,
                                   std::type_identity_t<std::span<T>> states, std::type_identity_t<std::span<T>> checkpoints,
                                   std::type_identity_t<std::span<T>> work)
{
    constexpr std::size_t H = C.mixer_heads, P = C.mixer_head_dim, N = C.mixer_state, G = C.mixer_groups, L = C.mixer_chunk;
    constexpr std::size_t S = H * N * P;
    constexpr T scale = mixer_scale_v<C, T>;
    const std::size_t rows = detail::gla::rows_of<C>(in);
    assert(sequences > 0 && rows % sequences == 0 && o.size() == in.v.size() && states.size() == sequences * S
           && work.size() == gla_chunk_work(C));
    const std::size_t n = rows / sequences, m = recurrence_checkpoints(n, L);
    assert(checkpoints.empty() || checkpoints.size() == sequences * m * S);
    for (std::size_t sq = 0; sq < sequences; ++sq)
        for (std::size_t ck = 0; ck < m; ++ck) {
            const std::size_t first = sq * n + ck * L, len = n - ck * L < L ? n - ck * L : L;
            const auto state = states.subspan(sq * S, S);
            if (!checkpoints.empty())
                for (std::size_t i = 0; i < S; ++i)
                    checkpoints[(sq * m + ck) * S + i] = state[i];
            for (std::size_t h = 0; h < H; ++h) {
                const std::size_t grp = h / (H / G);
                const auto s = state.subspan(h * N * P, N * P);
                const auto cg = work;   // cg[j · N + n]: the cumulative log-gate
                for (std::size_t j = 0; j < len; ++j)
                    for (std::size_t q = 0; q < N; ++q)
                        cg[j * N + q] = (j == 0 ? T(0) : cg[(j - 1) * N + q]) + in.gk[((first + j) * H + h) * N + q];
                for (std::size_t j = 0; j < len; ++j) {
                    const auto qj = in.q.subspan(((first + j) * G + grp) * N, N);
                    const auto oj = o.subspan(((first + j) * H + h) * P, P);
                    for (std::size_t p = 0; p < P; ++p)
                        oj[p] = T(0);
                    for (std::size_t i = 0; i <= j; ++i) {
                        const auto ki = in.k.subspan(((first + i) * G + grp) * N, N);
                        T a = T(0);
                        for (std::size_t q = 0; q < N; ++q)
                            a += qj[q] * ki[q] * math::exp(cg[j * N + q] - cg[i * N + q]);
                        const auto vi = in.v.subspan(((first + i) * H + h) * P, P);
                        for (std::size_t p = 0; p < P; ++p)
                            oj[p] += a * vi[p];
                    }
                    for (std::size_t q = 0; q < N; ++q) {
                        const T qd = qj[q] * math::exp(cg[j * N + q]);
                        for (std::size_t p = 0; p < P; ++p)
                            oj[p] += qd * s[q * P + p];
                    }
                    for (std::size_t p = 0; p < P; ++p)
                        oj[p] *= scale;
                }
                for (std::size_t q = 0; q < N; ++q) {
                    const T last = cg[(len - 1) * N + q], total = math::exp(last);
                    for (std::size_t p = 0; p < P; ++p) {
                        T add = T(0);
                        for (std::size_t i = 0; i < len; ++i)
                            add += math::exp(last - cg[i * N + q]) * in.k[((first + i) * G + grp) * N + q] * in.v[((first + i) * H + h) * P + p];
                        s[q * P + p] = total * s[q * P + p] + add;
                    }
                }
            }
        }
}

/// \brief The working space of `gla_backward`. \tparam C The configuration. \return The number of values.
template <transformer_config C, math::math_number T>
[[nodiscard]] constexpr std::size_t gla_backward_work() noexcept
{
    return recurrence_backward_work<detail::gla::scan<C, T>>(C.mixer_chunk);
}

/// \brief The backward pass of gated linear attention from a forward pass's checkpoints.
/// \tparam C The configuration.
/// \param in The forward pass's inputs. \param dout The gradient in its outputs. \param grads Receives the gradients.
/// \param sequences The sequences. \param checkpoints What the forward pass saved.
/// \param dstates The gradient in the final states, replaced by that in the initial.
/// \param work `gla_backward_work<C, T>()` values.
template <transformer_config C, typename B, math::math_number T>
constexpr void gla_backward(const B&, const gla_inputs<T>& in, std::type_identity_t<std::span<const T>> dout, const gla_gradients<T>& grads,
                            std::size_t sequences, std::type_identity_t<std::span<const T>> checkpoints,
                            std::type_identity_t<std::span<T>> dstates, std::type_identity_t<std::span<T>> work)
{
    const std::size_t rows = detail::gla::rows_of<C>(in);
    assert(sequences > 0 && rows % sequences == 0 && dout.size() == in.v.size() && grads.q.size() == in.q.size()
           && grads.k.size() == in.k.size() && grads.v.size() == in.v.size() && grads.gk.size() == in.gk.size());
    recurrence_backward(detail::gla::scan<C, T>{in, {}, dout, grads}, rows / sequences, sequences, C.mixer_chunk, checkpoints, dstates, work);
}

/// \brief One token of gated linear attention for decoding.
/// \tparam C The configuration.
/// \param in One row's inputs. \param o Receives its `H · P` outputs. \param state `gla_state_size(C)`, updated.
template <transformer_config C, math::math_number T>
constexpr void gla_step(const gla_inputs<T>& in, std::type_identity_t<std::span<T>> o, std::type_identity_t<std::span<T>> state)
{
    [[maybe_unused]] const std::size_t rows = detail::gla::rows_of<C>(in);
    assert(rows == 1 && o.size() == in.v.size() && state.size() == gla_state_size(C));
    detail::gla::scan<C, T>{in, o, {}, {}}.step(0, state);
}

} // namespace snicholls::ml

#endif // snicholls_core_ml_gla_hpp
