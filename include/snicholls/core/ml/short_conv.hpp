//
//  short_conv.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::ml` short causal depthwise convolution (Mamba's `conv1d`): over a batch of sequences with a state carried in and out, its backward pass, and one token at a time for decoding.
/// \ingroup core
///
/// **Overview.** Mamba, Mamba-2, the gated delta rule's layers (Qwen3-Next, Kimi Linear) and DeltaNet mix each channel
/// with its own last few tokens before the recurrence: a depthwise causal convolution of `K` taps (usually 4),
/// `y_t[c] = Σ_{j<K} w[c][j] · x_{t-K+1+j}[c] + b[c]`, the oldest tap first (llama.cpp's `GGML_OP_SSM_CONV` at commit
/// 39039aa, `ggml_compute_forward_ssm_conv_f32`, with the bias added after it as its graphs do). Before a sequence's
/// first token the inputs are the state, the `K - 1` inputs that came before it (zeros for a new sequence), so a
/// sequence can be fed in pieces; decoding keeps exactly that state.
///
/// - `short_conv_forward(backend, w, xs, ys, sequences, states)`: rows of `w.channels`, `sequences` sequences end to end.
/// - `short_conv_backward(backend, w, xs, dys, dxs, dw, db, sequences, initial, dinitial)`.
/// - `short_conv_step(w, x, y, state)`: one token; the forward pass's arithmetic, so its bits.
///
/// \snippet ml/short_conv_demo.cpp short_conv
///
/// **Relationships.** `ml/mixer_layer.hpp` (Mamba-2's and the gated delta rule's layers), `ml/rwkv.hpp` (token shift,
/// a convolution of two taps with weights that sum to one). Group: \ref core.
///
/// **Implementations.** One, on the host for any backend (TRAINING.md T9): a device library overloads the two batched
/// passes for its own backend type.
///
/// **Guarantees.** `constexpr`, no allocation, each sum in tap order from zero, then the bias. The state is
/// `short_conv_state_size(channels, width)` values per sequence, `K - 1` rows of `channels`, oldest first. The step and
/// the forward pass give the same bits (tested). Weight gradients are added into; `dxs` is written.
///
/// **Errors.** Preconditions, asserted: `w.data` `channels · width`, the bias empty or `channels`, the rows a multiple of
/// `channels · sequences`, the states empty or `sequences · short_conv_state_size`.
///
/// **Performance.** `benchmarks/ml/short_conv_benchmark.cpp`, 512 tokens of 2,560 channels and 4 taps (Release, AppleClang 17, Intel Xeon W-3245, macOS, 2026-11-19):
/// 1.08 ms, the loop by hand 1.11 ms (checked to the bit). The widths 2, 3 and 4 run with the tap count known to the
/// compiler. BENCHMARKS.md has the current numbers.
///
/// **Tests and demos.** `tests/ml/short_conv_test.cpp`, `demos/ml/short_conv_demo.cpp`.
///
/// **Dependencies.** `<algorithm>`, `<cassert>`, `<cstddef>`, `<span>`, `<type_traits>`, `math/constexpr_math.hpp`.
///
/// **Known issues.** None.
///
/// **Status.** Experimental, new in 0.17.0.

#ifndef snicholls_core_ml_short_conv_hpp
#define snicholls_core_ml_short_conv_hpp

#include <snicholls/core/math/constexpr_math.hpp>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <span>
#include <type_traits>

namespace snicholls::ml
{

template <math::math_number T>
struct short_conv_weights;

namespace detail::short_conv
{
    // The tokens past the first K - 1 read only the inputs: the same sums as the general loop, without its branch, with
    // the common widths' tap counts known to the compiler (fully unrolled)
    template <std::size_t Taps, bool Bias, typename T>
    constexpr void body(const T* taps, const T* bias, std::size_t ch, std::size_t width, const T* x, T* y, std::size_t first, std::size_t n)
    {
        const std::size_t K = Taps == 0 ? width : Taps;
        for (std::size_t i = first; i < n; ++i) {
            const T* window = x + (i + 1 - K) * ch;
            T* out = y + i * ch;
            for (std::size_t c = 0; c < ch; ++c) {
                T sum = T(0);
                for (std::size_t j = 0; j < K; ++j)
                    sum += taps[c * K + j] * window[j * ch + c];
                if constexpr (Bias)
                    out[c] = sum + bias[c];
                else
                    out[c] = sum;
            }
        }
    }

    template <typename W, typename T>
    constexpr void body_for_width(const W& w, const T* x, T* y, std::size_t first, std::size_t n)
    {
        const T* bias = w.bias.empty() ? nullptr : w.bias.data();
        const auto run = [&](auto taps, auto with_bias) {
            body<decltype(taps)::value, decltype(with_bias)::value>(w.data.data(), bias, w.channels, w.width, x, y, first, n);
        };
        const auto by_width = [&](auto with_bias) {
            switch (w.width) {
            case 2: run(std::integral_constant<std::size_t, 2>{}, with_bias); break;
            case 3: run(std::integral_constant<std::size_t, 3>{}, with_bias); break;
            case 4: run(std::integral_constant<std::size_t, 4>{}, with_bias); break;
            default: run(std::integral_constant<std::size_t, 0>{}, with_bias); break;
            }
        };
        if (bias == nullptr)
            by_width(std::false_type{});
        else
            by_width(std::true_type{});
    }
} // namespace detail::short_conv

/// \brief A depthwise convolution's taps over the caller's memory: `channels` rows of `width` taps, oldest first, and an
/// optional bias.
/// \ingroup core
/// \tparam T The number type.
template <math::math_number T>
struct short_conv_weights
{
    std::span<const T> data{};   ///< `channels · width` taps, channel by channel, the oldest input's first.
    std::size_t channels = 0;    ///< The channels.
    std::size_t width = 0;       ///< The taps `K`, at least 1.
    std::span<const T> bias{};   ///< `channels`, or empty for none.
};

/// \brief The state a sequence carries: its last `width - 1` inputs.
/// \param channels The channels. \param width The taps.
/// \return `(width - 1) · channels`.
[[nodiscard]] constexpr std::size_t short_conv_state_size(std::size_t channels, std::size_t width) noexcept
{
    return width == 0 ? 0 : (width - 1) * channels;
}

/// \brief The convolution over `sequences` sequences laid end to end: `y_t[c] = Σ_j w[c][j] x_{t-K+1+j}[c] + b[c]`.
/// \param w The taps and bias.
/// \param xs `sequences · n` rows of `channels`.
/// \param ys Receives the outputs, as `xs`; may not alias it.
/// \param sequences The sequences.
/// \param states `sequences · short_conv_state_size`: each sequence's earlier inputs, replaced by its last `width - 1`
/// inputs (for the next piece or for decoding); or empty: zeros before each sequence, and nothing kept.
template <typename B, math::math_number T>
constexpr void short_conv_forward(const B&, const short_conv_weights<T>& w, std::type_identity_t<std::span<const T>> xs,
                                  std::type_identity_t<std::span<T>> ys, std::size_t sequences = 1,
                                  std::type_identity_t<std::span<T>> states = {})
{
    const std::size_t ch = w.channels, K = w.width, sk = short_conv_state_size(ch, K);
    assert(K > 0 && w.data.size() == ch * K && (w.bias.empty() || w.bias.size() == ch) && sequences > 0);
    assert(xs.size() % (ch * sequences) == 0 && ys.size() == xs.size() && (states.empty() || states.size() == sequences * sk));
    const std::size_t n = xs.size() / (ch * sequences);
    for (std::size_t s = 0; s < sequences; ++s) {
        const auto x = xs.subspan(s * n * ch, n * ch);
        const auto y = ys.subspan(s * n * ch, n * ch);
        const auto st = states.empty() ? std::span<T>{} : states.subspan(s * sk, sk);
        const std::size_t head = n < K - 1 ? n : K - 1;   // the tokens that read the state
        for (std::size_t i = 0; i < head; ++i)
            for (std::size_t c = 0; c < ch; ++c) {
                T sum = T(0);
                for (std::size_t j = 0; j < K; ++j) {
                    const std::size_t back = K - 1 - j;   // inputs before token i
                    const T in = back <= i ? x[(i - back) * ch + c] : st.empty() ? T(0) : st[(K - 1 + i - back) * ch + c];
                    sum += w.data[c * K + j] * in;
                }
                y[i * ch + c] = w.bias.empty() ? sum : sum + w.bias[c];
            }
        detail::short_conv::body_for_width(w, x.data(), y.data(), head, n);
        for (std::size_t r = 0; r + 1 < K && !st.empty(); ++r) {   // the last K - 1 inputs, oldest first
            const std::size_t from = r + n;                          // in the old state's rows, then x's
            for (std::size_t c = 0; c < ch; ++c)
                st[r * ch + c] = from < K - 1 ? st[from * ch + c] : x[(from - (K - 1)) * ch + c];
        }
    }
}

/// \brief The backward pass of `short_conv_forward`: `dxs` written, `dw` and `db` added into.
/// \param w The taps the forward pass used.
/// \param xs Its inputs.
/// \param dys The gradient in its outputs.
/// \param dxs Receives the gradient in its inputs.
/// \param dw `channels · width`: added into.
/// \param db `channels`: added into; or empty.
/// \param sequences The sequences.
/// \param initial The states the forward pass started from, or empty for zeros.
/// \param dinitial Receives the gradient in them, as `initial`; or empty to skip it.
template <typename B, math::math_number T>
constexpr void short_conv_backward(const B&, const short_conv_weights<T>& w, std::type_identity_t<std::span<const T>> xs,
                                   std::type_identity_t<std::span<const T>> dys, std::type_identity_t<std::span<T>> dxs,
                                   std::type_identity_t<std::span<T>> dw, std::type_identity_t<std::span<T>> db,
                                   std::size_t sequences = 1, std::type_identity_t<std::span<const T>> initial = {},
                                   std::type_identity_t<std::span<T>> dinitial = {})
{
    const std::size_t ch = w.channels, K = w.width, sk = short_conv_state_size(ch, K);
    assert(K > 0 && w.data.size() == ch * K && sequences > 0 && xs.size() % (ch * sequences) == 0);
    assert(dys.size() == xs.size() && dxs.size() == xs.size() && dw.size() == ch * K && (db.empty() || db.size() == ch));
    assert((initial.empty() || initial.size() == sequences * sk) && (dinitial.empty() || dinitial.size() == sequences * sk));
    const std::size_t n = xs.size() / (ch * sequences);
    for (std::size_t s = 0; s < sequences; ++s) {
        const auto x = xs.subspan(s * n * ch, n * ch);
        const auto dy = dys.subspan(s * n * ch, n * ch);
        const auto dx = dxs.subspan(s * n * ch, n * ch);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t c = 0; c < ch; ++c) {
                T sum = T(0);   // the outputs token i feeds: i + K - 1 - j
                for (std::size_t j = 0; j < K; ++j)
                    if (i + K - 1 - j < n)
                        sum += w.data[c * K + j] * dy[(i + K - 1 - j) * ch + c];
                dx[i * ch + c] = sum;
            }
        for (std::size_t c = 0; c < ch; ++c)
            for (std::size_t j = 0; j < K; ++j) {
                T sum = T(0);
                for (std::size_t i = 0; i < n; ++i) {
                    const std::size_t back = K - 1 - j;
                    const T in = back <= i ? x[(i - back) * ch + c] : initial.empty() ? T(0) : initial[s * sk + (K - 1 + i - back) * ch + c];
                    sum += dy[i * ch + c] * in;
                }
                dw[c * K + j] += sum;
            }
        if (!db.empty())
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t c = 0; c < ch; ++c)
                    db[c] += dy[i * ch + c];
        if (!dinitial.empty())
            for (std::size_t r = 0; r + 1 < K; ++r)   // state row r feeds output i through tap r - i
                for (std::size_t c = 0; c < ch; ++c) {
                    T sum = T(0);
                    for (std::size_t i = 0; i <= r && i < n; ++i)
                        sum += w.data[c * K + (r - i)] * dy[i * ch + c];
                    dinitial[s * sk + r * ch + c] = sum;
                }
    }
}

/// \brief One token of the convolution for decoding: `y` from the state and `x`, then `x` appended to the state.
/// \param w The taps and bias.
/// \param x The token's `channels` inputs.
/// \param y Receives its `channels` outputs; may not alias `x`.
/// \param state `short_conv_state_size(channels, width)`: the earlier inputs, updated.
template <math::math_number T>
constexpr void short_conv_step(const short_conv_weights<T>& w, std::type_identity_t<std::span<const T>> x,
                               std::type_identity_t<std::span<T>> y, std::type_identity_t<std::span<T>> state)
{
    const std::size_t ch = w.channels, K = w.width;
    assert(K > 0 && x.size() == ch && y.size() == ch && state.size() == short_conv_state_size(ch, K));
    for (std::size_t c = 0; c < ch; ++c) {
        T sum = T(0);
        for (std::size_t j = 0; j < K; ++j)
            sum += w.data[c * K + j] * (j + 1 < K ? state[j * ch + c] : x[c]);
        y[c] = w.bias.empty() ? sum : sum + w.bias[c];
    }
    for (std::size_t r = 0; r + 1 < K; ++r)
        for (std::size_t c = 0; c < ch; ++c)
            state[r * ch + c] = r + 2 < K ? state[(r + 1) * ch + c] : x[c];
}

} // namespace snicholls::ml

#endif // snicholls_core_ml_short_conv_hpp
