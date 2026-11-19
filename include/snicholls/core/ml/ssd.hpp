//
//  ssd.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::ml` Mamba-2's selective state space (SSD): the recurrent scan, the chunked form (quadratic within a chunk, the state passed between chunks), the backward pass, and one token at a time for decoding.
/// \ingroup core
///
/// **Overview.** Mamba-2 (Dao and Gu, 2024, "Transformers are SSMs", section 2.4 and eqs. 2a-2b) restricts the state
/// transition of a selective SSM to a scalar per head and time, `h_t = a_t h_{t-1} + B_t x_t`, `y_t = C_tᵀ h_t`, with
/// `a_t = exp(Δ_t A)` (Mamba's discretisation, section 2.1 remark 1). Per head `h` of `P` channels, in group
/// `g = h / (H / G)` whose `B` and `C` it shares (section 7.2, grouped-value attention), with the state `S` of `N` rows of
/// `P` (state-space duality's keys by values):
///
///     S_t[n][p] = exp(dt_t A) · S_{t-1}[n][p] + B_t[n] · (x_t[p] · dt_t),     y_t[p] = Σ_n C_t[n] · S_t[n][p]
///
/// which is llama.cpp's `GGML_OP_SSM_SCAN` for a scalar `A` (commit 39039aa, `ggml_compute_forward_ssm_scan_f32`, the
/// same products in the same order; tested against a transcription of it), except that `dt` arrives here already
/// positive (the layer applies `softplus(dt + dt_bias)`, which ggml does inside the op) and the state is laid out
/// `[n][p]` rather than `[p][n]`.
///
/// The chunked form is the paper's SSD algorithm (section 6, its listing `ssd`): within a chunk of `L` tokens with
/// cumulative log-decays `a_j = Σ_{i≤j} dt_i A`,
/// `y_j = Σ_{i≤j} (C_j · B_i) exp(a_j - a_i) dt_i x_i + exp(a_j) C_jᵀ S_prev` (the diagonal block and the left factor),
/// and the state passed on is `exp(a_last) S_prev + Σ_i exp(a_last - a_i) dt_i B_i x_iᵀ` (the right and centre factors).
/// It gives the recurrence's result within rounding: relative `L · ε` in the exponents' sums (tested in float and
/// double).
///
/// - `ssd_forward<C>(backend, in, y, sequences, states, checkpoints)`: the scan, `recurrence_forward` over `ssd_step`'s
///   arithmetic.
/// - `ssd_forward_chunked<C>(backend, in, y, sequences, states, checkpoints)`: the chunked form, the same checkpoints.
/// - `ssd_backward<C>(backend, in, dy, grads, sequences, checkpoints, dstates, work)`: backpropagation through time from
///   either forward pass's checkpoints (`ml/recurrence.hpp`).
/// - `ssd_step<C>(in, y, state)`: one token for decoding, with the fixed-size state `ssd_state_size(C)`.
///
/// \snippet ml/ssd_demo.cpp ssd
///
/// **Relationships.** `ml/recurrence.hpp` (the driver), `ml/mixer_layer.hpp` (the Mamba-2 layer: projections, the
/// short convolution, `softplus`, the `D` skip, the gate and the grouped norm), `ml/transformer_config.hpp` (`H` is
/// `mixer_heads`, `P` `mixer_head_dim`, `N` `mixer_state`, `G` `mixer_groups`, `L` `mixer_chunk`). Group: \ref core.
///
/// **Implementations.** The scan and the chunked form; one backward pass for both.
///
/// **Guarantees.** `constexpr`, no allocation (the chunk's cumulative decays are a fixed-size array of `L`), every sum
/// in a stated order: the scan's `y` over `n` ascending, the chunked form's diagonal block over `i` ascending then the
/// inter-chunk term. Decoding with `ssd_step` gives `ssd_forward`'s bits at every position (tested). The gradients in the
/// inputs of each row are written; `grads.a` is added into.
///
/// **Errors.** Preconditions, asserted, as `ml/recurrence.hpp`'s and the views' sizes (`rows · H · P` for `x` and `y`,
/// `rows · H` for `dt`, `H` for `a`, `rows · G · N` for `b` and `c`).
///
/// **Performance.** `benchmarks/ml/ssd_benchmark.cpp`: the scan against the loop by hand (the same arithmetic in the same
/// order, checked to the bit), and the chunked form beside it; the results are in BENCHMARKS.md.
///
/// **Tests and demos.** `tests/ml/ssd_test.cpp`, `demos/ml/ssd_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<cassert>`, `<cstddef>`, `<span>`, `<type_traits>`, `math/constexpr_math.hpp`,
/// `ml/recurrence.hpp`, `ml/transformer_config.hpp`.
///
/// **Known issues.** The chunked form here is the reference, a token's row at a time: a device runs it as batched
/// products over chunks (and its backward pass chunked) and checks against this.
///
/// **Status.** Experimental, new in 0.17.0.

#ifndef snicholls_core_ml_ssd_hpp
#define snicholls_core_ml_ssd_hpp

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

/// \brief The inputs of the selective scan over `rows` rows: views the caller owns.
/// \ingroup core
/// \tparam T The number type.
template <math::math_number T>
struct ssd_inputs
{
    std::span<const T> x{};    ///< `rows · H · P`: the values (Mamba-2's `x`).
    std::span<const T> dt{};   ///< `rows · H`: the step sizes, positive (after `softplus`).
    std::span<const T> a{};    ///< `H`: each head's `A`, negative.
    std::span<const T> b{};    ///< `rows · G · N`: the keys (Mamba-2's `B`).
    std::span<const T> c{};    ///< `rows · G · N`: the queries (Mamba-2's `C`).
};

/// \brief The gradients of the selective scan: views the caller owns, shaped as `ssd_inputs`.
/// \ingroup core
/// \tparam T The number type.
template <math::math_number T>
struct ssd_gradients
{
    std::span<T> x{};    ///< Written.
    std::span<T> dt{};   ///< Written.
    std::span<T> a{};    ///< Added into.
    std::span<T> b{};    ///< Written.
    std::span<T> c{};    ///< Written.
};

/// \brief A sequence's state: `H · N · P` values, head by head, `N` rows of `P`.
/// \param c The configuration. \return The number of values.
[[nodiscard]] constexpr std::size_t ssd_state_size(const transformer_config& c) noexcept
{
    return mixer_state_size(c);
}

namespace detail::ssd
{
    template <transformer_config C, typename T>
    struct scan
    {
        using value_type = T;
        static constexpr std::size_t H = C.mixer_heads, P = C.mixer_head_dim, N = C.mixer_state, G = C.mixer_groups;
        static constexpr std::size_t state_size = H * N * P, work_size = N * P;

        ssd_inputs<T> in{};
        std::span<T> y{};
        std::span<const T> dy{};
        ssd_gradients<T> g{};

        constexpr void update(std::size_t row, std::span<T> state, bool output) const
        {
            for (std::size_t h = 0; h < H; ++h) {
                const T dt = in.dt[row * H + h], alpha = math::exp(dt * in.a[h]);
                const std::size_t grp = h / (H / G);
                const auto x = in.x.subspan((row * H + h) * P, P);
                const auto b = in.b.subspan((row * G + grp) * N, N), c = in.c.subspan((row * G + grp) * N, N);
                const auto s = state.subspan(h * N * P, N * P);
                std::array<T, P> xdt{}, out{};
                for (std::size_t p = 0; p < P; ++p)
                    xdt[p] = x[p] * dt;
                for (std::size_t n = 0; n < N; ++n)
                    for (std::size_t p = 0; p < P; ++p) {
                        s[n * P + p] = s[n * P + p] * alpha + b[n] * xdt[p];
                        out[p] += s[n * P + p] * c[n];
                    }
                if (output)
                    for (std::size_t p = 0; p < P; ++p)
                        y[(row * H + h) * P + p] = out[p];
            }
        }
        constexpr void step(std::size_t row, std::span<T> state) const { update(row, state, true); }
        constexpr void advance(std::size_t row, std::span<T> state) const { update(row, state, false); }

        constexpr void step_backward(std::size_t row, std::span<const T> prev, std::span<T> dstate, std::span<T> work) const
        {
            for (std::size_t i = 0; i < G * N; ++i)
                g.b[row * G * N + i] = g.c[row * G * N + i] = T(0);
            for (std::size_t h = 0; h < H; ++h) {
                const T dt = in.dt[row * H + h], A = in.a[h], alpha = math::exp(dt * A);
                const std::size_t grp = h / (H / G);
                const auto x = in.x.subspan((row * H + h) * P, P);
                const auto b = in.b.subspan((row * G + grp) * N, N), c = in.c.subspan((row * G + grp) * N, N);
                const auto dyh = dy.subspan((row * H + h) * P, P);
                const auto s0 = prev.subspan(h * N * P, N * P);
                const auto ds = dstate.subspan(h * N * P, N * P);
                const auto db = g.b.subspan((row * G + grp) * N, N), dc = g.c.subspan((row * G + grp) * N, N);
                T dalpha = T(0);
                std::array<T, P> dxdt{};
                for (std::size_t n = 0; n < N; ++n) {
                    T dcn = T(0), dbn = T(0);
                    for (std::size_t p = 0; p < P; ++p) {
                        const T st = s0[n * P + p] * alpha + b[n] * (x[p] * dt);
                        dcn += dyh[p] * st;
                        const T d = ds[n * P + p] + c[n] * dyh[p];
                        dbn += d * (x[p] * dt);
                        dxdt[p] += b[n] * d;
                        dalpha += d * s0[n * P + p];
                        ds[n * P + p] = d * alpha;
                    }
                    dc[n] += dcn;
                    db[n] += dbn;
                }
                T ddt = T(0);
                for (std::size_t p = 0; p < P; ++p) {
                    g.x[(row * H + h) * P + p] = dxdt[p] * dt;
                    ddt += dxdt[p] * x[p];
                }
                g.dt[row * H + h] = ddt + dalpha * alpha * A;
                g.a[h] += dalpha * alpha * dt;
            }
            static_cast<void>(work);
        }
    };

    template <transformer_config C, typename T>
    constexpr void check(const ssd_inputs<T>& in, std::size_t rows)
    {
        [[maybe_unused]] constexpr std::size_t H = C.mixer_heads, P = C.mixer_head_dim, N = C.mixer_state, G = C.mixer_groups;
        static_assert(valid(C) && C.mixer_heads > 0, "snicholls::ml::ssd: the configuration has no mixer shape");
        assert(in.x.size() == rows * H * P && in.dt.size() == rows * H && in.a.size() == H && in.b.size() == rows * G * N
               && in.c.size() == rows * G * N);
        static_cast<void>(rows);
        static_cast<void>(in);
    }
} // namespace detail::ssd

/// \brief The selective scan over `sequences` sequences laid end to end, token by token.
/// \tparam C The configuration (its mixer shape).
/// \param in The inputs, `rows = sequences · n` rows.
/// \param y Receives `rows · H · P` outputs.
/// \param sequences The sequences.
/// \param states `sequences · ssd_state_size(C)`: the initial states (zeros for new sequences), replaced by the final.
/// \param checkpoints `sequences · recurrence_checkpoints(n, C.mixer_chunk) · ssd_state_size(C)` for the backward
/// pass, or empty.
template <transformer_config C, typename B, math::math_number T>
constexpr void ssd_forward(const B&, const ssd_inputs<T>& in, std::type_identity_t<std::span<T>> y, std::size_t sequences,
                           std::type_identity_t<std::span<T>> states, std::type_identity_t<std::span<T>> checkpoints = {})
{
    const std::size_t rows = in.dt.size() / C.mixer_heads;
    detail::ssd::check<C>(in, rows);
    assert(sequences > 0 && rows % sequences == 0 && y.size() == in.x.size());
    recurrence_forward(detail::ssd::scan<C, T>{in, y, {}, {}}, rows / sequences, sequences, C.mixer_chunk, states, checkpoints);
}

/// \brief The selective scan's chunked form (SSD), over chunks of `C.mixer_chunk` tokens: the recurrence's result
/// within rounding, the same checkpoints.
/// \tparam C The configuration.
/// \param in The inputs. \param y Receives the outputs. \param sequences The sequences.
/// \param states The initial states, replaced by the final. \param checkpoints The states at each chunk's start, or empty.
template <transformer_config C, typename B, math::math_number T>
constexpr void ssd_forward_chunked(const B&, const ssd_inputs<T>& in, std::type_identity_t<std::span<T>> y, std::size_t sequences,
                                   std::type_identity_t<std::span<T>> states, std::type_identity_t<std::span<T>> checkpoints = {})
{
    constexpr std::size_t H = C.mixer_heads, P = C.mixer_head_dim, N = C.mixer_state, G = C.mixer_groups, L = C.mixer_chunk;
    constexpr std::size_t S = H * N * P;
    const std::size_t rows = in.dt.size() / H;
    detail::ssd::check<C>(in, rows);
    assert(sequences > 0 && rows % sequences == 0 && y.size() == in.x.size() && states.size() == sequences * S);
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
                std::array<T, L> la{};   // the cumulative log-decay
                for (std::size_t j = 0; j < len; ++j)
                    la[j] = (j == 0 ? T(0) : la[j - 1]) + in.dt[(first + j) * H + h] * in.a[h];
                for (std::size_t j = 0; j < len; ++j) {
                    const auto cj = in.c.subspan(((first + j) * G + grp) * N, N);
                    const auto yj = y.subspan(((first + j) * H + h) * P, P);
                    for (std::size_t p = 0; p < P; ++p)
                        yj[p] = T(0);
                    for (std::size_t i = 0; i <= j; ++i) {   // the diagonal block
                        const auto bi = in.b.subspan(((first + i) * G + grp) * N, N);
                        T cb = T(0);
                        for (std::size_t q = 0; q < N; ++q)
                            cb += cj[q] * bi[q];
                        const T w = cb * math::exp(la[j] - la[i]) * in.dt[(first + i) * H + h];
                        const auto xi = in.x.subspan(((first + i) * H + h) * P, P);
                        for (std::size_t p = 0; p < P; ++p)
                            yj[p] += w * xi[p];
                    }
                    const T decay = math::exp(la[j]);   // the state carried in
                    for (std::size_t p = 0; p < P; ++p) {
                        T cs = T(0);
                        for (std::size_t q = 0; q < N; ++q)
                            cs += cj[q] * s[q * P + p];
                        yj[p] += decay * cs;
                    }
                }
                const T total = math::exp(la[len - 1]);   // the state passed on
                for (std::size_t q = 0; q < N; ++q)
                    for (std::size_t p = 0; p < P; ++p) {
                        T add = T(0);
                        for (std::size_t i = 0; i < len; ++i)
                            add += math::exp(la[len - 1] - la[i]) * in.dt[(first + i) * H + h] * in.b[((first + i) * G + grp) * N + q]
                                   * in.x[((first + i) * H + h) * P + p];
                        s[q * P + p] = total * s[q * P + p] + add;
                    }
            }
        }
}

/// \brief The working space of `ssd_backward`.
/// \tparam C The configuration. \return The number of values.
template <transformer_config C, math::math_number T>
[[nodiscard]] constexpr std::size_t ssd_backward_work() noexcept
{
    return recurrence_backward_work<detail::ssd::scan<C, T>>(C.mixer_chunk);
}

/// \brief The backward pass of the selective scan from a forward pass's checkpoints (either form's).
/// \tparam C The configuration.
/// \param in The forward pass's inputs.
/// \param dy The gradient in its outputs.
/// \param grads Receives the gradients in the inputs (`a` added into).
/// \param sequences The sequences.
/// \param checkpoints What the forward pass saved.
/// \param dstates `sequences · ssd_state_size(C)`: the gradient in the final states, replaced by that in the initial.
/// \param work `ssd_backward_work<C, T>()` values.
template <transformer_config C, typename B, math::math_number T>
constexpr void ssd_backward(const B&, const ssd_inputs<T>& in, std::type_identity_t<std::span<const T>> dy, const ssd_gradients<T>& grads,
                            std::size_t sequences, std::type_identity_t<std::span<const T>> checkpoints,
                            std::type_identity_t<std::span<T>> dstates, std::type_identity_t<std::span<T>> work)
{
    const std::size_t rows = in.dt.size() / C.mixer_heads;
    detail::ssd::check<C>(in, rows);
    assert(sequences > 0 && rows % sequences == 0 && dy.size() == in.x.size() && grads.x.size() == in.x.size()
           && grads.dt.size() == in.dt.size() && grads.a.size() == in.a.size() && grads.b.size() == in.b.size()
           && grads.c.size() == in.c.size());
    recurrence_backward(detail::ssd::scan<C, T>{in, {}, dy, grads}, rows / sequences, sequences, C.mixer_chunk, checkpoints, dstates, work);
}

/// \brief One token of the selective scan for decoding: `y` from the state, which is updated.
/// \tparam C The configuration.
/// \param in One row's inputs. \param y Receives its `H · P` outputs. \param state `ssd_state_size(C)`, updated.
template <transformer_config C, math::math_number T>
constexpr void ssd_step(const ssd_inputs<T>& in, std::type_identity_t<std::span<T>> y, std::type_identity_t<std::span<T>> state)
{
    detail::ssd::check<C>(in, 1);
    assert(y.size() == C.mixer_heads * C.mixer_head_dim && state.size() == ssd_state_size(C));
    detail::ssd::scan<C, T>{in, y, {}, {}}.step(0, state);
}

} // namespace snicholls::ml

#endif // snicholls_core_ml_ssd_hpp
