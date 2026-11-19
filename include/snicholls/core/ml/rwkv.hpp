//
//  rwkv.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::ml` RWKV's sequence mixing: token shift, RWKV-6's WKV with its bonus, and RWKV-7's WKV (the generalised delta rule), each forward over a sequence, backward, and one token at a time for decoding.
/// \ingroup core
///
/// **Overview.** RWKV mixes tokens twice: a token shift, which interpolates each channel between the token and the one
/// before it (`lerp(x_t, x_{t-1}, μ) = x_t + (x_{t-1} - x_t) ⊙ μ`; RWKV-6, Peng et al., 2024, eq. 3; RWKV-7, Peng et
/// al., 2025, eq. 3; llama.cpp's graphs compute it as `(x_prev - x) · μ + x`), and a WKV recurrence over a matrix state
/// per head, here laid out `N` key rows of `P` value channels (`N = P`, the head size).
///
/// RWKV-6 (eq. 19 and its recurrent form, eqs. 21-22): the state is read before it is updated, with a per-channel bonus
/// `u` for the current token, and the decay `w_t = exp(-exp(d_t))` (eq. 18) is passed as its logarithm `lw = -exp(d)`:
///
///     o_t[p] = Σ_n r_t[n] · (v_t[p] k_t[n] · u[n] + S_{t-1}[n][p]),     S_t[n][p] = S_{t-1}[n][p] · e^{lw_t[n]} + v_t[p] k_t[n]
///
/// RWKV-7 (eq. 17, `wkv_t = wkv_{t-1}(diag(w_t) - κ̂_tᵀ(a_t ⊙ κ̂_t)) + v_tᵀ k̃_t`, written here for the transposed state
/// with ggml's general `a` and `b`: the model passes `a = -κ̂` and `b = κ̂ ⊙ a_t`), its decay `w = exp(-e^{-0.5}
/// sigmoid(d))` (eq. 12) also passed as its logarithm:
///
///     sa[p] = Σ_n a_t[n] S_{t-1}[n][p],   S_t[n][p] = S_{t-1}[n][p] e^{lw_t[n]} + v_t[p] k_t[n] + sa[p] b_t[n],   o_t[p] = Σ_n S_t[n][p] r_t[n]
///
/// Both are llama.cpp's `GGML_OP_RWKV_WKV6` and `GGML_OP_RWKV_WKV7` (commit 39039aa, `ggml_compute_forward_rwkv_wkv6_f32`
/// and `_wkv7_f32`'s scalar paths: the same products in the same order, tested against transcriptions), with the decay
/// as its logarithm and the state transposed.
///
/// - `token_shift(backend, xs, mu, out, sequences, states)`, `token_shift_backward(...)`, `token_shift_step(...)`.
/// - `wkv6_forward<C>`, `wkv6_backward<C>`, `wkv6_step<C>`; `wkv7_forward<C>`, `wkv7_backward<C>`, `wkv7_step<C>`.
///
/// \snippet ml/rwkv_demo.cpp rwkv
///
/// **Relationships.** `ml/recurrence.hpp` (the driver), `ml/gla.hpp` (RWKV-6 is gated linear attention read before the
/// update, plus the bonus), `ml/delta_rule.hpp` (RWKV-7 generalises it), `ml/short_conv.hpp` (token shift is a
/// two-tap convolution). Group: \ref core.
///
/// **Implementations.** The recurrent forms; the token shift on rows.
///
/// **Guarantees.** `constexpr`, no allocation, the sums in ggml's order. Decoding gives the sequence forward's bits at
/// every position (tested). Every row's input gradients are written; RWKV-6's `u` and the token shift's `μ` are
/// parameters, added into.
///
/// **Errors.** Preconditions, asserted: a configuration with one key head per head and square heads (`valid_mixer`), the
/// views `rows · H · N`, `u` and `μ` of their widths.
///
/// **Performance.** `benchmarks/ml/rwkv_benchmark.cpp`, both recurrences against the loops by hand; BENCHMARKS.md.
///
/// **Tests and demos.** `tests/ml/rwkv_test.cpp`, `demos/ml/rwkv_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<cassert>`, `<cstddef>`, `<span>`, `<type_traits>`, `math/constexpr_math.hpp`,
/// `ml/recurrence.hpp`, `ml/transformer_config.hpp`.
///
/// **Known issues.** No chunked form (RWKV-7's paper gives only the closed form, eq. 18, and says prefill is chunked;
/// RWKV-6's chunked form is `ml/gla.hpp`'s with a strictly causal block and the bonus), and no layer: RWKV-6's
/// data-dependent interpolation and low-rank decay (eqs. 14-17) and RWKV-7's in-context learning rate, value residual
/// and gate (eqs. 4-14, 20-22) are a layer's, not here.
///
/// **Status.** Experimental, new in 0.17.0.

#ifndef snicholls_core_ml_rwkv_hpp
#define snicholls_core_ml_rwkv_hpp

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

/// \brief Token shift over `sequences` sequences laid end to end: `out_t = (x_{t-1} - x_t) · μ + x_t`.
/// \param xs `sequences · n` rows of `μ.size()`.
/// \param mu The interpolation per channel.
/// \param out Receives the shifted rows; may not alias `xs`.
/// \param sequences The sequences.
/// \param states `sequences` rows: the row before each sequence, replaced by its last row; or empty: zeros, nothing kept.
template <typename B, math::math_number T>
constexpr void token_shift(const B&, std::type_identity_t<std::span<const T>> xs, std::type_identity_t<std::span<const T>> mu,
                           std::span<T> out, std::size_t sequences = 1, std::type_identity_t<std::span<T>> states = {})
{
    const std::size_t d = mu.size();
    assert(d > 0 && sequences > 0 && xs.size() % (d * sequences) == 0 && out.size() == xs.size()
           && (states.empty() || states.size() == sequences * d));
    const std::size_t n = xs.size() / (d * sequences);
    for (std::size_t s = 0; s < sequences; ++s) {
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t c = 0; c < d; ++c) {
                const std::size_t t = s * n + i;
                const T prev = i > 0 ? xs[(t - 1) * d + c] : states.empty() ? T(0) : states[s * d + c];
                out[t * d + c] = (prev - xs[t * d + c]) * mu[c] + xs[t * d + c];
            }
        if (!states.empty() && n > 0)
            for (std::size_t c = 0; c < d; ++c)
                states[s * d + c] = xs[(s * n + n - 1) * d + c];
    }
}

/// \brief The backward pass of `token_shift`: `dxs` written, `dmu` added into.
/// \param xs The forward pass's rows. \param mu Its interpolation. \param dout The gradient in its output.
/// \param dxs Receives the gradient in `xs`. \param dmu `μ.size()`: added into. \param sequences The sequences.
/// \param initial The rows before each sequence the forward pass read, or empty for zeros.
/// \param dinitial Receives their gradient, or empty to skip it.
template <typename B, math::math_number T>
constexpr void token_shift_backward(const B&, std::type_identity_t<std::span<const T>> xs, std::type_identity_t<std::span<const T>> mu,
                                    std::type_identity_t<std::span<const T>> dout, std::span<T> dxs,
                                    std::type_identity_t<std::span<T>> dmu, std::size_t sequences = 1,
                                    std::type_identity_t<std::span<const T>> initial = {}, std::type_identity_t<std::span<T>> dinitial = {})
{
    const std::size_t d = mu.size();
    assert(d > 0 && sequences > 0 && xs.size() % (d * sequences) == 0 && dout.size() == xs.size() && dxs.size() == xs.size()
           && dmu.size() == d && (initial.empty() || initial.size() == sequences * d) && (dinitial.empty() || dinitial.size() == sequences * d));
    const std::size_t n = xs.size() / (d * sequences);
    for (std::size_t s = 0; s < sequences; ++s)
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t c = 0; c < d; ++c) {
                const std::size_t t = s * n + i;
                const T prev = i > 0 ? xs[(t - 1) * d + c] : initial.empty() ? T(0) : initial[s * d + c];
                const T g = dout[t * d + c];
                dmu[c] += (prev - xs[t * d + c]) * g;
                dxs[t * d + c] = g - g * mu[c] + (i + 1 < n ? dout[(t + 1) * d + c] * mu[c] : T(0));
                if (i == 0 && !dinitial.empty())
                    dinitial[s * d + c] = g * mu[c];
            }
}

/// \brief One token of token shift for decoding. \param x The token. \param mu The interpolation.
/// \param out Receives the shifted row. \param state The previous token, replaced by `x`.
template <math::math_number T>
constexpr void token_shift_step(std::type_identity_t<std::span<const T>> x, std::type_identity_t<std::span<const T>> mu,
                                std::span<T> out, std::type_identity_t<std::span<T>> state)
{
    assert(x.size() == mu.size() && out.size() == x.size() && state.size() == x.size());
    for (std::size_t c = 0; c < x.size(); ++c) {
        out[c] = (state[c] - x[c]) * mu[c] + x[c];
        state[c] = x[c];
    }
}

/// \brief RWKV-6's WKV inputs over `rows` rows, every view `rows · H · N` but `u`: views the caller owns.
/// \ingroup core
/// \tparam T The number type.
template <math::math_number T>
struct wkv6_inputs
{
    std::span<const T> r{};    ///< The receptance.
    std::span<const T> k{};    ///< The keys.
    std::span<const T> v{};    ///< The values.
    std::span<const T> lw{};   ///< The log-decay `-exp(d)`, negative.
    std::span<const T> u{};    ///< `H · N`: the bonus (RWKV's `time_first`).
};

/// \brief RWKV-6's WKV gradients, shaped as `wkv6_inputs`.
/// \ingroup core
/// \tparam T The number type.
template <math::math_number T>
struct wkv6_gradients
{
    std::span<T> r{};    ///< Written.
    std::span<T> k{};    ///< Written.
    std::span<T> v{};    ///< Written.
    std::span<T> lw{};   ///< Written.
    std::span<T> u{};    ///< Added into.
};

/// \brief RWKV-7's WKV inputs over `rows` rows, every view `rows · H · N`: views the caller owns.
/// \ingroup core
/// \tparam T The number type.
template <math::math_number T>
struct wkv7_inputs
{
    std::span<const T> r{};    ///< The receptance.
    std::span<const T> lw{};   ///< The log-decay, negative.
    std::span<const T> k{};    ///< The replacement keys `k̃`.
    std::span<const T> v{};    ///< The values.
    std::span<const T> a{};    ///< The removal direction, `-κ̂` in RWKV-7.
    std::span<const T> b{};    ///< Its weight, `κ̂ ⊙ a_t` in RWKV-7.
};

/// \brief RWKV-7's WKV gradients, shaped as `wkv7_inputs`, every one written.
/// \ingroup core
/// \tparam T The number type.
template <math::math_number T>
struct wkv7_gradients
{
    std::span<T> r{};    ///< Written.
    std::span<T> lw{};   ///< Written.
    std::span<T> k{};    ///< Written.
    std::span<T> v{};    ///< Written.
    std::span<T> a{};    ///< Written.
    std::span<T> b{};    ///< Written.
};

/// \brief A sequence's WKV state: `H · N · N` values. \param c The configuration. \return The number of values.
[[nodiscard]] constexpr std::size_t wkv_state_size(const transformer_config& c) noexcept
{
    return mixer_state_size(c);
}

namespace detail::rwkv
{
    template <transformer_config C>
    constexpr void check_shape()
    {
        static_assert(valid(C) && C.mixer_heads > 0 && C.mixer_groups == C.mixer_heads && C.mixer_state == C.mixer_head_dim,
                      "snicholls::ml::rwkv: the configuration needs RWKV's shape (one key head per head, square heads)");
    }

    template <transformer_config C, typename T>
    struct wkv6
    {
        using value_type = T;
        static constexpr std::size_t H = C.mixer_heads, N = C.mixer_state;
        static constexpr std::size_t state_size = H * N * N, work_size = 0;

        wkv6_inputs<T> in{};
        std::span<T> o{};
        std::span<const T> dout{};
        wkv6_gradients<T> g{};

        constexpr void update(std::size_t row, std::span<T> state, bool output) const
        {
            for (std::size_t h = 0; h < H; ++h) {
                const std::size_t base = (row * H + h) * N;
                const auto s = state.subspan(h * N * N, N * N);
                std::array<T, N> out{};
                for (std::size_t n = 0; n < N; ++n) {
                    const T k = in.k[base + n], r = in.r[base + n], u = in.u[h * N + n], w = math::exp(in.lw[base + n]);
                    for (std::size_t p = 0; p < N; ++p) {
                        const T kv = in.v[base + p] * k, prev = s[n * N + p];
                        out[p] += (kv * u + prev) * r;
                        s[n * N + p] = prev * w + kv;
                    }
                }
                if (output)
                    for (std::size_t p = 0; p < N; ++p)
                        o[base + p] = out[p];
            }
        }
        constexpr void step(std::size_t row, std::span<T> state) const { update(row, state, true); }
        constexpr void advance(std::size_t row, std::span<T> state) const { update(row, state, false); }

        constexpr void step_backward(std::size_t row, std::span<const T> prev, std::span<T> dstate, std::span<T>) const
        {
            for (std::size_t h = 0; h < H; ++h) {
                const std::size_t base = (row * H + h) * N;
                const auto s0 = prev.subspan(h * N * N, N * N);
                const auto ds = dstate.subspan(h * N * N, N * N);
                std::array<T, N> dv{};
                for (std::size_t n = 0; n < N; ++n) {
                    const T k = in.k[base + n], r = in.r[base + n], u = in.u[h * N + n], w = math::exp(in.lw[base + n]);
                    T dr = T(0), dk = T(0), du = T(0), dw = T(0);
                    for (std::size_t p = 0; p < N; ++p) {
                        const T v = in.v[base + p], kv = v * k, d = dout[base + p];
                        dr += d * (kv * u + s0[n * N + p]);
                        const T dt = d * r;                        // the gradient in kv · u + prev
                        const T dkv = dt * u + ds[n * N + p];      // kv feeds the output and the state
                        du += dt * kv;
                        dw += ds[n * N + p] * s0[n * N + p];
                        dk += dkv * v;
                        dv[p] += dkv * k;
                        ds[n * N + p] = ds[n * N + p] * w + dt;
                    }
                    g.r[base + n] = dr;
                    g.k[base + n] = dk;
                    g.lw[base + n] = dw * w;
                    g.u[h * N + n] += du;
                }
                for (std::size_t p = 0; p < N; ++p)
                    g.v[base + p] = dv[p];
            }
        }
    };

    template <transformer_config C, typename T>
    struct wkv7
    {
        using value_type = T;
        static constexpr std::size_t H = C.mixer_heads, N = C.mixer_state;
        static constexpr std::size_t state_size = H * N * N, work_size = 0;

        wkv7_inputs<T> in{};
        std::span<T> o{};
        std::span<const T> dout{};
        wkv7_gradients<T> g{};

        constexpr void update(std::size_t row, std::span<T> state, bool output) const
        {
            for (std::size_t h = 0; h < H; ++h) {
                const std::size_t base = (row * H + h) * N;
                const auto s = state.subspan(h * N * N, N * N);
                std::array<T, N> sa{}, out{};
                for (std::size_t n = 0; n < N; ++n)
                    for (std::size_t p = 0; p < N; ++p)
                        sa[p] += in.a[base + n] * s[n * N + p];
                for (std::size_t n = 0; n < N; ++n) {
                    const T k = in.k[base + n], w = math::exp(in.lw[base + n]), b = in.b[base + n], r = in.r[base + n];
                    for (std::size_t p = 0; p < N; ++p) {
                        s[n * N + p] = s[n * N + p] * w + in.v[base + p] * k + sa[p] * b;
                        out[p] += s[n * N + p] * r;
                    }
                }
                if (output)
                    for (std::size_t p = 0; p < N; ++p)
                        o[base + p] = out[p];
            }
        }
        constexpr void step(std::size_t row, std::span<T> state) const { update(row, state, true); }
        constexpr void advance(std::size_t row, std::span<T> state) const { update(row, state, false); }

        constexpr void step_backward(std::size_t row, std::span<const T> prev, std::span<T> dstate, std::span<T>) const
        {
            for (std::size_t h = 0; h < H; ++h) {
                const std::size_t base = (row * H + h) * N;
                const auto s0 = prev.subspan(h * N * N, N * N);
                const auto ds = dstate.subspan(h * N * N, N * N);
                std::array<T, N> sa{}, dsa{}, dv{};
                for (std::size_t n = 0; n < N; ++n)
                    for (std::size_t p = 0; p < N; ++p)
                        sa[p] += in.a[base + n] * s0[n * N + p];
                for (std::size_t n = 0; n < N; ++n) {
                    const T k = in.k[base + n], w = math::exp(in.lw[base + n]), b = in.b[base + n], r = in.r[base + n];
                    T dr = T(0), dw = T(0), dk = T(0), db = T(0);
                    for (std::size_t p = 0; p < N; ++p) {
                        const T v = in.v[base + p], d = dout[base + p];
                        dr += d * (s0[n * N + p] * w + v * k + sa[p] * b);
                        const T dsp = ds[n * N + p] + r * d;
                        dw += dsp * s0[n * N + p];
                        dk += dsp * v;
                        dv[p] += dsp * k;
                        db += dsp * sa[p];
                        dsa[p] += dsp * b;
                        ds[n * N + p] = dsp * w;
                    }
                    g.r[base + n] = dr;
                    g.lw[base + n] = dw * w;
                    g.k[base + n] = dk;
                    g.b[base + n] = db;
                }
                for (std::size_t n = 0; n < N; ++n) {   // sa = aᵀ S_{t-1}: into a and the previous state
                    T da = T(0);
                    for (std::size_t p = 0; p < N; ++p) {
                        da += dsa[p] * s0[n * N + p];
                        ds[n * N + p] += in.a[base + n] * dsa[p];
                    }
                    g.a[base + n] = da;
                }
                for (std::size_t p = 0; p < N; ++p)
                    g.v[base + p] = dv[p];
            }
        }
    };
} // namespace detail::rwkv

/// \brief RWKV-6's WKV over `sequences` sequences laid end to end.
/// \tparam C The configuration (RWKV's shape).
/// \param in The inputs. \param o Receives `rows · H · N` outputs. \param sequences The sequences.
/// \param states `sequences · wkv_state_size(C)`: the initial states, replaced by the final.
/// \param checkpoints The states at each chunk's start for the backward pass, or empty.
template <transformer_config C, typename B, math::math_number T>
constexpr void wkv6_forward(const B&, const wkv6_inputs<T>& in, std::type_identity_t<std::span<T>> o, std::size_t sequences,
                            std::type_identity_t<std::span<T>> states, std::type_identity_t<std::span<T>> checkpoints = {})
{
    detail::rwkv::check_shape<C>();
    const std::size_t rows = in.r.size() / (C.mixer_heads * C.mixer_state);
    assert(sequences > 0 && rows % sequences == 0 && in.k.size() == in.r.size() && in.v.size() == in.r.size() && in.lw.size() == in.r.size()
           && in.u.size() == C.mixer_heads * C.mixer_state && o.size() == in.r.size());
    recurrence_forward(detail::rwkv::wkv6<C, T>{in, o, {}, {}}, rows / sequences, sequences, C.mixer_chunk, states, checkpoints);
}

/// \brief The working space of `wkv6_backward` and `wkv7_backward`. \tparam C The configuration.
/// \return The number of values.
template <transformer_config C, math::math_number T>
[[nodiscard]] constexpr std::size_t wkv_backward_work() noexcept
{
    return recurrence_backward_work<detail::rwkv::wkv6<C, T>>(C.mixer_chunk);
}

/// \brief The backward pass of RWKV-6's WKV from a forward pass's checkpoints.
/// \tparam C The configuration.
/// \param in The forward pass's inputs. \param dout The gradient in its outputs. \param grads Receives the gradients.
/// \param sequences The sequences. \param checkpoints What the forward pass saved.
/// \param dstates The gradient in the final states, replaced by that in the initial.
/// \param work `wkv_backward_work<C, T>()` values.
template <transformer_config C, typename B, math::math_number T>
constexpr void wkv6_backward(const B&, const wkv6_inputs<T>& in, std::type_identity_t<std::span<const T>> dout, const wkv6_gradients<T>& grads,
                             std::size_t sequences, std::type_identity_t<std::span<const T>> checkpoints,
                             std::type_identity_t<std::span<T>> dstates, std::type_identity_t<std::span<T>> work)
{
    detail::rwkv::check_shape<C>();
    const std::size_t rows = in.r.size() / (C.mixer_heads * C.mixer_state);
    assert(sequences > 0 && rows % sequences == 0 && dout.size() == in.r.size() && grads.r.size() == in.r.size()
           && grads.k.size() == in.r.size() && grads.v.size() == in.r.size() && grads.lw.size() == in.r.size() && grads.u.size() == in.u.size());
    recurrence_backward(detail::rwkv::wkv6<C, T>{in, {}, dout, grads}, rows / sequences, sequences, C.mixer_chunk, checkpoints, dstates, work);
}

/// \brief One token of RWKV-6's WKV for decoding.
/// \tparam C The configuration.
/// \param in One row's inputs. \param o Receives its outputs. \param state `wkv_state_size(C)`, updated.
template <transformer_config C, math::math_number T>
constexpr void wkv6_step(const wkv6_inputs<T>& in, std::type_identity_t<std::span<T>> o, std::type_identity_t<std::span<T>> state)
{
    detail::rwkv::check_shape<C>();
    assert(in.r.size() == C.mixer_heads * C.mixer_state && o.size() == in.r.size() && state.size() == wkv_state_size(C));
    detail::rwkv::wkv6<C, T>{in, o, {}, {}}.step(0, state);
}

/// \brief RWKV-7's WKV over `sequences` sequences laid end to end.
/// \tparam C The configuration (RWKV's shape).
/// \param in The inputs. \param o Receives `rows · H · N` outputs. \param sequences The sequences.
/// \param states `sequences · wkv_state_size(C)`: the initial states, replaced by the final.
/// \param checkpoints The states at each chunk's start for the backward pass, or empty.
template <transformer_config C, typename B, math::math_number T>
constexpr void wkv7_forward(const B&, const wkv7_inputs<T>& in, std::type_identity_t<std::span<T>> o, std::size_t sequences,
                            std::type_identity_t<std::span<T>> states, std::type_identity_t<std::span<T>> checkpoints = {})
{
    detail::rwkv::check_shape<C>();
    const std::size_t rows = in.r.size() / (C.mixer_heads * C.mixer_state);
    assert(sequences > 0 && rows % sequences == 0 && in.lw.size() == in.r.size() && in.k.size() == in.r.size() && in.v.size() == in.r.size()
           && in.a.size() == in.r.size() && in.b.size() == in.r.size() && o.size() == in.r.size());
    recurrence_forward(detail::rwkv::wkv7<C, T>{in, o, {}, {}}, rows / sequences, sequences, C.mixer_chunk, states, checkpoints);
}

/// \brief The backward pass of RWKV-7's WKV from a forward pass's checkpoints.
/// \tparam C The configuration.
/// \param in The forward pass's inputs. \param dout The gradient in its outputs. \param grads Receives the gradients.
/// \param sequences The sequences. \param checkpoints What the forward pass saved.
/// \param dstates The gradient in the final states, replaced by that in the initial.
/// \param work `wkv_backward_work<C, T>()` values.
template <transformer_config C, typename B, math::math_number T>
constexpr void wkv7_backward(const B&, const wkv7_inputs<T>& in, std::type_identity_t<std::span<const T>> dout, const wkv7_gradients<T>& grads,
                             std::size_t sequences, std::type_identity_t<std::span<const T>> checkpoints,
                             std::type_identity_t<std::span<T>> dstates, std::type_identity_t<std::span<T>> work)
{
    detail::rwkv::check_shape<C>();
    const std::size_t rows = in.r.size() / (C.mixer_heads * C.mixer_state);
    assert(sequences > 0 && rows % sequences == 0 && dout.size() == in.r.size() && grads.r.size() == in.r.size()
           && grads.lw.size() == in.r.size() && grads.k.size() == in.r.size() && grads.v.size() == in.r.size()
           && grads.a.size() == in.r.size() && grads.b.size() == in.r.size());
    recurrence_backward(detail::rwkv::wkv7<C, T>{in, {}, dout, grads}, rows / sequences, sequences, C.mixer_chunk, checkpoints, dstates, work);
}

/// \brief One token of RWKV-7's WKV for decoding.
/// \tparam C The configuration.
/// \param in One row's inputs. \param o Receives its outputs. \param state `wkv_state_size(C)`, updated.
template <transformer_config C, math::math_number T>
constexpr void wkv7_step(const wkv7_inputs<T>& in, std::type_identity_t<std::span<T>> o, std::type_identity_t<std::span<T>> state)
{
    detail::rwkv::check_shape<C>();
    assert(in.r.size() == C.mixer_heads * C.mixer_state && o.size() == in.r.size() && state.size() == wkv_state_size(C));
    detail::rwkv::wkv7<C, T>{in, o, {}, {}}.step(0, state);
}

} // namespace snicholls::ml

#endif // snicholls_core_ml_rwkv_hpp
