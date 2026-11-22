//
//  delta_rule.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::ml` the gated delta rule (Gated DeltaNet, Qwen3-Next's and Kimi Linear's linear attention): the recurrent form, the chunked form through the WY / UT representation and a triangular solve, the backward pass, and one token at a time for decoding.
/// \ingroup core
///
/// **Overview.** The delta rule writes into a linear attention's state only what the state does not already predict:
/// `S_t = S_{t-1}(I - β_t k_t k_tᵀ) + β_t v_t k_tᵀ` (DeltaNet; Yang et al., 2024, arXiv 2406.06484, section 3.1), and
/// Gated DeltaNet decays the state first, `S_t = S_{t-1}(α_t (I - β_t k_t k_tᵀ)) + β_t v_t k_tᵀ` (Yang, Kautz and
/// Hatamizadeh, 2024, arXiv 2412.06464, eq. 10), `o_t = S_t q_t`. Here, per head `h` of `P` value channels in key group
/// `g` (the state laid out `N` key rows of `P`), with the gate as its logarithm `g_t = log α_t` (one per head, Qwen3-Next's
/// `-exp(A_log) · softplus(a + dt_bias)`, or one per key channel, Kimi Delta Attention's) and the queries scaled by
/// `1 / sqrt(N)`:
///
///     S ← e^{g_t} S;   δ[p] = (v_t[p] - Σ_n S[n][p] k_t[n]) · β_t;   S[n][p] += k_t[n] δ[p];   o_t[p] = (Σ_n S[n][p] q_t[n]) · scale
///
/// which is llama.cpp's `GGML_OP_GATED_DELTA_NET` (commit 39039aa, `ggml_compute_forward_gated_delta_net_one_chunk`,
/// with its per-channel gate for Kimi; tested against a transcription), the state transposed.
///
/// **The chunked form** is the WY representation with the UT transform (2406.06484 eqs. 8-11; with the gates,
/// 2412.06464 section 3.3). Within a chunk of `L` tokens from the state `S_0`, with cumulative gates `G_j = Σ_{i≤j} g_i`
/// (per channel), the recurrence unrolls to `δ_j + β_j Σ_{i<j} A_{ji} δ_i = β_j (v_j - S_0ᵀ (e^{G_j} ⊙ k_j))` with
/// `A_{ji} = Σ_n k_j[n] k_i[n] e^{G_j[n] - G_i[n]}`: a unit lower-triangular system `(I + strictLower(diag(β) A)) [U | W]
/// = diag(β) [V | e^{G} ⊙ K]`, solved with the backend's `linalg::triangular_solve` (the paper's `T = (I + ...)⁻¹
/// diag(β)`, never formed), so `Δ = U - W S_0`. Then `o_j = scale · ((e^{G_j} ⊙ q_j)ᵀ S_0 + Σ_{i≤j} (Σ_n q_j[n] k_i[n]
/// e^{G_j[n] - G_i[n]}) Δ_i)` and the state passed on is `e^{G_last} ⊙ S_0 + Σ_i (e^{G_last - G_i} ⊙ k_i) Δ_iᵀ`. Every
/// exponent is of a difference that is not positive. It is the recurrence within rounding (tested in float and double).
///
/// - `delta_rule_forward<C>(backend, in, o, sequences, states, checkpoints)`,
///   `delta_rule_forward_chunked<C>(..., work)`, `delta_rule_backward<C>(...)`, `delta_rule_step<C>(in, o, state)`.
/// - `l2_normalise_rows(backend, x, out, width, eps)` and its backward pass: the queries' and keys' `x / sqrt(Σ x² + eps)`
///   (flash-linear-attention's `l2norm`, llama.cpp's `build_gdn_l2_norm`).
///
/// \snippet ml/delta_rule_demo.cpp delta_rule
///
/// **Relationships.** `ml/recurrence.hpp`, `ml/gla.hpp` (the same chunking without the solve), `ml/rwkv.hpp` (RWKV-7
/// generalises the rule), `ml/mixer_layer.hpp` (Qwen3-Next's layer), `linalg/backend.hpp` (`triangular_solve`). Group:
/// \ref core.
///
/// **Implementations.** The recurrent and chunked forms; one backward pass for both. Gates per head or per key channel,
/// told apart by `g`'s size.
///
/// **Guarantees.** `constexpr` (on a `constexpr` backend, as core's reference), no allocation, sums in a stated order
/// (ggml's: the state's product with `k` and with `q` over `n` ascending). Decoding gives the recurrent forward's bits at
/// every position (tested). Each row's input gradients are written.
///
/// **Errors.** Preconditions, asserted: `q` and `k` `rows · G · N`, `v` and `o` `rows · H · P`, `g` `rows · H` or
/// `rows · H · N`, `beta` `rows · H`; `work` `delta_rule_chunk_work(C)`. The unit lower-triangular solve cannot fail.
///
/// **Performance.** `benchmarks/ml/delta_rule_benchmark.cpp`, 256 tokens of 8 value heads of 64 sharing 4 key heads
/// (Release, AppleClang 17, Intel Xeon W-3245, macOS, 2026-11-19): the recurrent form 4.27 ms, the loop by hand 4.20 ms (checked to the bit); the chunked WY form 109 ms, a
/// reference for a device's kernel. BENCHMARKS.md has the current numbers.
///
/// **Tests and demos.** `tests/ml/delta_rule_test.cpp`, `demos/ml/delta_rule_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<cassert>`, `<cstddef>`, `<span>`, `<type_traits>`, `linalg/backend.hpp`,
/// `math/constexpr_math.hpp`, `ml/recurrence.hpp`, `ml/transformer_config.hpp`.
///
/// **Known issues.** The keys should be normalised (the rule is stable for `β ‖k‖² ≤ 2`); the layer does it.
///
/// **Status.** Experimental, new in 0.17.0.

#ifndef snicholls_core_ml_delta_rule_hpp
#define snicholls_core_ml_delta_rule_hpp

#include <snicholls/core/linalg/backend.hpp>
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

/// \brief The inputs of the gated delta rule over `rows` rows: views the caller owns.
/// \ingroup core
/// \tparam T The number type.
template <math::math_number T>
struct delta_rule_inputs
{
    std::span<const T> q{};      ///< `rows · G · N`: the queries (normalised).
    std::span<const T> k{};      ///< `rows · G · N`: the keys (normalised).
    std::span<const T> v{};      ///< `rows · H · P`: the values.
    std::span<const T> g{};      ///< `rows · H` (a gate per head) or `rows · H · N` (per key channel): the log-gates, not positive.
    std::span<const T> beta{};   ///< `rows · H`: the writing strengths, in (0, 1).
};

/// \brief The gradients of the gated delta rule, shaped as `delta_rule_inputs`, every one written.
/// \ingroup core
/// \tparam T The number type.
template <math::math_number T>
struct delta_rule_gradients
{
    std::span<T> q{};      ///< Written.
    std::span<T> k{};      ///< Written.
    std::span<T> v{};      ///< Written.
    std::span<T> g{};      ///< Written.
    std::span<T> beta{};   ///< Written.
};

/// \brief A sequence's state: `H · N · P` values. \param c The configuration. \return The number of values.
[[nodiscard]] constexpr std::size_t delta_rule_state_size(const transformer_config& c) noexcept
{
    return mixer_state_size(c);
}

/// \brief The working space of `delta_rule_forward_chunked`: a chunk's cumulative gates, its triangular matrix, the
/// solve's `[U | W]` and `Δ`.
/// \param c The configuration. \return `L·N + L·L + L·(P+N) + L·P`.
[[nodiscard]] constexpr std::size_t delta_rule_chunk_work(const transformer_config& c) noexcept
{
    const std::size_t L = c.mixer_chunk, N = c.mixer_state, P = c.mixer_head_dim;
    return L * N + L * L + L * (P + N) + L * P;
}

/// \brief `out = x / sqrt(Σ x² + eps)` for each row of `width`: the queries' and keys' normalisation.
/// \param x The rows. \param out Receives the normalised rows; may be `x`. \param width The row's width.
/// \param eps Added to the sum of squares.
template <typename B, math::math_number T>
constexpr void l2_normalise_rows(const B&, std::type_identity_t<std::span<const T>> x, std::span<T> out,
                                 std::size_t width, std::type_identity_t<T> eps)
{
    assert(width > 0 && x.size() % width == 0 && out.size() == x.size());
    for (std::size_t r = 0; r < x.size() / width; ++r) {
        T ss = T(0);
        for (std::size_t i = 0; i < width; ++i)
            ss += x[r * width + i] * x[r * width + i];
        const T inv = T(1) / math::sqrt(ss + eps);
        for (std::size_t i = 0; i < width; ++i)
            out[r * width + i] = x[r * width + i] * inv;
    }
}

/// \brief The backward pass of `l2_normalise_rows`: `dx = (dy - y (y · dy)) / sqrt(Σ x² + eps)`.
/// \param x The forward pass's rows. \param dy The gradient in its output. \param dx Receives the gradient in `x`;
/// may be `dy`. \param width The row's width. \param eps The forward pass's.
template <typename B, math::math_number T>
constexpr void l2_normalise_rows_backward(const B&, std::type_identity_t<std::span<const T>> x, std::type_identity_t<std::span<const T>> dy,
                                          std::span<T> dx, std::size_t width, std::type_identity_t<T> eps)
{
    assert(width > 0 && x.size() % width == 0 && dy.size() == x.size() && dx.size() == x.size());
    for (std::size_t r = 0; r < x.size() / width; ++r) {
        T ss = T(0), yd = T(0);
        for (std::size_t i = 0; i < width; ++i)
            ss += x[r * width + i] * x[r * width + i];
        const T inv = T(1) / math::sqrt(ss + eps);
        for (std::size_t i = 0; i < width; ++i)
            yd += x[r * width + i] * inv * dy[r * width + i];
        for (std::size_t i = 0; i < width; ++i)
            dx[r * width + i] = (dy[r * width + i] - x[r * width + i] * inv * yd) * inv;
    }
}

namespace detail::delta_rule
{
    template <transformer_config C, typename T>
    struct scan
    {
        using value_type = T;
        static constexpr std::size_t H = C.mixer_heads, P = C.mixer_head_dim, N = C.mixer_state, G = C.mixer_groups;
        static constexpr std::size_t state_size = H * N * P, work_size = N * P;
        static constexpr T scale = mixer_scale_v<C, T>;

        delta_rule_inputs<T> in{};
        std::span<T> o{};
        std::span<const T> dout{};
        delta_rule_gradients<T> g{};
        bool per_channel = false;

        constexpr T gate(std::size_t row, std::size_t h, std::size_t n) const
        {
            return per_channel ? in.g[(row * H + h) * N + n] : in.g[row * H + h];
        }

        constexpr void update(std::size_t row, std::span<T> state, bool output) const
        {
            for (std::size_t h = 0; h < H; ++h) {
                const std::size_t grp = h / (H / G);
                const auto q = in.q.subspan((row * G + grp) * N, N), k = in.k.subspan((row * G + grp) * N, N);
                const auto v = in.v.subspan((row * H + h) * P, P);
                const T beta = in.beta[row * H + h];
                const auto s = state.subspan(h * N * P, N * P);
                std::array<T, P> delta{};
                for (std::size_t n = 0; n < N; ++n) {
                    const T a = math::exp(gate(row, h, n));
                    for (std::size_t p = 0; p < P; ++p)
                        s[n * P + p] *= a;
                }
                for (std::size_t n = 0; n < N; ++n)
                    for (std::size_t p = 0; p < P; ++p)
                        delta[p] += s[n * P + p] * k[n];
                for (std::size_t p = 0; p < P; ++p)
                    delta[p] = (v[p] - delta[p]) * beta;
                for (std::size_t n = 0; n < N; ++n)
                    for (std::size_t p = 0; p < P; ++p)
                        s[n * P + p] += k[n] * delta[p];
                if (output) {
                    std::array<T, P> out{};
                    for (std::size_t n = 0; n < N; ++n)
                        for (std::size_t p = 0; p < P; ++p)
                            out[p] += s[n * P + p] * q[n];
                    for (std::size_t p = 0; p < P; ++p)
                        o[(row * H + h) * P + p] = out[p] * scale;
                }
            }
        }
        constexpr void step(std::size_t row, std::span<T> state) const { update(row, state, true); }
        constexpr void advance(std::size_t row, std::span<T> state) const { update(row, state, false); }

        constexpr void step_backward(std::size_t row, std::span<const T> prev, std::span<T> dstate, std::span<T> work) const
        {
            for (std::size_t i = 0; i < G * N; ++i)
                g.q[row * G * N + i] = g.k[row * G * N + i] = T(0);
            for (std::size_t h = 0; h < H; ++h) {
                const std::size_t grp = h / (H / G);
                const auto q = in.q.subspan((row * G + grp) * N, N), k = in.k.subspan((row * G + grp) * N, N);
                const auto v = in.v.subspan((row * H + h) * P, P);
                const auto d = dout.subspan((row * H + h) * P, P);
                const T beta = in.beta[row * H + h];
                const auto s0 = prev.subspan(h * N * P, N * P);
                const auto ds = dstate.subspan(h * N * P, N * P);
                const auto sd = work.first(N * P);   // the decayed state
                const auto dq = g.q.subspan((row * G + grp) * N, N), dk = g.k.subspan((row * G + grp) * N, N);
                std::array<T, P> u{}, delta{}, ddelta{}, du{};
                for (std::size_t n = 0; n < N; ++n) {
                    const T a = math::exp(gate(row, h, n));
                    for (std::size_t p = 0; p < P; ++p) {
                        sd[n * P + p] = s0[n * P + p] * a;
                        u[p] += sd[n * P + p] * k[n];
                    }
                }
                for (std::size_t p = 0; p < P; ++p)
                    delta[p] = (v[p] - u[p]) * beta;
                for (std::size_t n = 0; n < N; ++n) {   // o = scale · Sᵀ q with S = sd + k δᵀ
                    T dqn = T(0), dkn = T(0);
                    for (std::size_t p = 0; p < P; ++p) {
                        dqn += d[p] * (sd[n * P + p] + k[n] * delta[p]);
                        ds[n * P + p] += q[n] * scale * d[p];
                        dkn += ds[n * P + p] * delta[p];
                        ddelta[p] += ds[n * P + p] * k[n];
                    }
                    dq[n] += dqn * scale;
                    dk[n] += dkn;
                }
                T dbeta = T(0);
                for (std::size_t p = 0; p < P; ++p) {
                    dbeta += ddelta[p] * (v[p] - u[p]);
                    g.v[(row * H + h) * P + p] = ddelta[p] * beta;
                    du[p] = -ddelta[p] * beta;
                }
                g.beta[row * H + h] = dbeta;
                T dg = T(0);
                for (std::size_t n = 0; n < N; ++n) {   // u = sdᵀ k, sd = e^g s0
                    const T a = math::exp(gate(row, h, n));
                    T dkn = T(0), dgn = T(0);
                    for (std::size_t p = 0; p < P; ++p) {
                        dkn += sd[n * P + p] * du[p];
                        const T dsd = ds[n * P + p] + k[n] * du[p];
                        dgn += dsd * sd[n * P + p];
                        ds[n * P + p] = dsd * a;
                    }
                    dk[n] += dkn;
                    if (per_channel)
                        g.g[(row * H + h) * N + n] = dgn;
                    else
                        dg += dgn;
                }
                if (!per_channel)
                    g.g[row * H + h] = dg;
            }
        }
    };

    template <transformer_config C, typename T>
    constexpr std::size_t rows_of(const delta_rule_inputs<T>& in)
    {
        [[maybe_unused]] constexpr std::size_t H = C.mixer_heads, P = C.mixer_head_dim, N = C.mixer_state, G = C.mixer_groups;
        static_assert(valid(C) && C.mixer_heads > 0, "snicholls::ml::delta_rule: the configuration has no mixer shape");
        static_cast<void>(N), static_cast<void>(G);   // read only by the assert (MSVC ignores [[maybe_unused]] there, C4189)
        const std::size_t rows = in.v.size() / (H * P);
        assert(in.v.size() == rows * H * P && in.q.size() == rows * G * N && in.k.size() == rows * G * N && in.beta.size() == rows * H
               && (in.g.size() == rows * H || in.g.size() == rows * H * N));
        return rows;
    }

    template <transformer_config C, typename T>
    constexpr scan<C, T> make(const delta_rule_inputs<T>& in, std::span<T> o, std::span<const T> dout, const delta_rule_gradients<T>& g,
                              std::size_t rows)
    {
        return {in, o, dout, g, C.mixer_state > 1 && in.g.size() == rows * C.mixer_heads * C.mixer_state};
    }
} // namespace detail::delta_rule

/// \brief The gated delta rule over `sequences` sequences laid end to end, token by token.
/// \tparam C The configuration.
/// \param in The inputs. \param o Receives `rows · H · P` outputs. \param sequences The sequences.
/// \param states `sequences · delta_rule_state_size(C)`: the initial states, replaced by the final.
/// \param checkpoints The states at each chunk's start for the backward pass, or empty.
template <transformer_config C, typename B, math::math_number T>
constexpr void delta_rule_forward(const B&, const delta_rule_inputs<T>& in, std::type_identity_t<std::span<T>> o, std::size_t sequences,
                                  std::type_identity_t<std::span<T>> states, std::type_identity_t<std::span<T>> checkpoints = {})
{
    const std::size_t rows = detail::delta_rule::rows_of<C>(in);
    assert(sequences > 0 && rows % sequences == 0 && o.size() == in.v.size());
    recurrence_forward(detail::delta_rule::make<C, T>(in, o, {}, {}, rows), rows / sequences, sequences, C.mixer_chunk, states, checkpoints);
}

/// \brief The gated delta rule's chunked form through the WY / UT representation, chunks of `C.mixer_chunk` tokens: the
/// recurrent form's result within rounding, the same checkpoints.
/// \tparam C The configuration.
/// \param backend A `linalg::blas_backend`: the triangular solve is its.
/// \param in The inputs. \param o Receives the outputs. \param sequences The sequences.
/// \param states The initial states, replaced by the final. \param checkpoints The states at each chunk's start, or empty.
/// \param work `delta_rule_chunk_work(C)` values.
template <transformer_config C, typename B, math::math_number T>
    requires linalg::blas_backend<B, T>
constexpr void delta_rule_forward_chunked(const B& backend, const delta_rule_inputs<T>& in, std::type_identity_t<std::span<T>> o,
                                          std::size_t sequences, std::type_identity_t<std::span<T>> states,
                                          std::type_identity_t<std::span<T>> checkpoints, std::type_identity_t<std::span<T>> work)
{
    constexpr std::size_t H = C.mixer_heads, P = C.mixer_head_dim, N = C.mixer_state, G = C.mixer_groups, L = C.mixer_chunk;
    constexpr std::size_t S = H * N * P;
    constexpr T scale = mixer_scale_v<C, T>;
    const std::size_t rows = detail::delta_rule::rows_of<C>(in);
    const auto sc = detail::delta_rule::make<C, T>(in, o, {}, {}, rows);
    assert(sequences > 0 && rows % sequences == 0 && o.size() == in.v.size() && states.size() == sequences * S
           && work.size() == delta_rule_chunk_work(C));
    const std::size_t n = rows / sequences, m = recurrence_checkpoints(n, L);
    assert(checkpoints.empty() || checkpoints.size() == sequences * m * S);
    const auto cg = work.first(L * N), tri = work.subspan(L * N, L * L), uw = work.subspan(L * N + L * L, L * (P + N)),
               dl = work.subspan(L * N + L * L + L * (P + N), L * P);
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
                const auto key = [&](std::size_t j) { return in.k.subspan(((first + j) * G + grp) * N, N); };
                for (std::size_t j = 0; j < len; ++j)
                    for (std::size_t q = 0; q < N; ++q)
                        cg[j * N + q] = (j == 0 ? T(0) : cg[(j - 1) * N + q]) + sc.gate(first + j, h, q);
                // (I + strictLower(diag(β) A)) [U | W] = diag(β) [V | e^G ⊙ K], unit lower triangular
                for (std::size_t j = 0; j < len; ++j) {
                    const T beta = in.beta[(first + j) * H + h];
                    for (std::size_t i = 0; i < len; ++i) {
                        T a = T(0);
                        if (i < j)
                            for (std::size_t q = 0; q < N; ++q)
                                a += key(j)[q] * key(i)[q] * math::exp(cg[j * N + q] - cg[i * N + q]);
                        tri[j * len + i] = i < j ? beta * a : i == j ? T(1) : T(0);
                    }
                    for (std::size_t p = 0; p < P; ++p)
                        uw[j * (P + N) + p] = beta * in.v[((first + j) * H + h) * P + p];
                    for (std::size_t q = 0; q < N; ++q)
                        uw[j * (P + N) + P + q] = beta * key(j)[q] * math::exp(cg[j * N + q]);
                }
                [[maybe_unused]] const auto solved = linalg::triangular_solve(
                    backend, linalg::matrix_view<const T, linalg::layout_right>(tri.data(), {len, len}),
                    linalg::matrix_view<T, linalg::layout_right>(uw.data(), {len, P + N}), linalg::triangle::lower,
                    linalg::diagonal::implicit_unit);
                assert(solved.is_ok());
                for (std::size_t j = 0; j < len; ++j)   // Δ = U - W S_0
                    for (std::size_t p = 0; p < P; ++p) {
                        T ws = T(0);
                        for (std::size_t q = 0; q < N; ++q)
                            ws += uw[j * (P + N) + P + q] * s[q * P + p];
                        dl[j * P + p] = uw[j * (P + N) + p] - ws;
                    }
                for (std::size_t j = 0; j < len; ++j) {
                    const auto qj = in.q.subspan(((first + j) * G + grp) * N, N);
                    const auto oj = o.subspan(((first + j) * H + h) * P, P);
                    for (std::size_t p = 0; p < P; ++p)
                        oj[p] = T(0);
                    for (std::size_t q = 0; q < N; ++q) {
                        const T qd = qj[q] * math::exp(cg[j * N + q]);
                        for (std::size_t p = 0; p < P; ++p)
                            oj[p] += qd * s[q * P + p];
                    }
                    for (std::size_t i = 0; i <= j; ++i) {
                        T a = T(0);
                        for (std::size_t q = 0; q < N; ++q)
                            a += qj[q] * key(i)[q] * math::exp(cg[j * N + q] - cg[i * N + q]);
                        for (std::size_t p = 0; p < P; ++p)
                            oj[p] += a * dl[i * P + p];
                    }
                    for (std::size_t p = 0; p < P; ++p)
                        oj[p] *= scale;
                }
                for (std::size_t q = 0; q < N; ++q) {
                    const T last = cg[(len - 1) * N + q], total = math::exp(last);
                    for (std::size_t p = 0; p < P; ++p) {
                        T add = T(0);
                        for (std::size_t i = 0; i < len; ++i)
                            add += math::exp(last - cg[i * N + q]) * key(i)[q] * dl[i * P + p];
                        s[q * P + p] = total * s[q * P + p] + add;
                    }
                }
            }
        }
}

/// \brief The working space of `delta_rule_backward`. \tparam C The configuration. \return The number of values.
template <transformer_config C, math::math_number T>
[[nodiscard]] constexpr std::size_t delta_rule_backward_work() noexcept
{
    return recurrence_backward_work<detail::delta_rule::scan<C, T>>(C.mixer_chunk);
}

/// \brief The backward pass of the gated delta rule from a forward pass's checkpoints (either form's).
/// \tparam C The configuration.
/// \param in The forward pass's inputs. \param dout The gradient in its outputs. \param grads Receives the gradients.
/// \param sequences The sequences. \param checkpoints What the forward pass saved.
/// \param dstates The gradient in the final states, replaced by that in the initial.
/// \param work `delta_rule_backward_work<C, T>()` values.
template <transformer_config C, typename B, math::math_number T>
constexpr void delta_rule_backward(const B&, const delta_rule_inputs<T>& in, std::type_identity_t<std::span<const T>> dout,
                                   const delta_rule_gradients<T>& grads, std::size_t sequences,
                                   std::type_identity_t<std::span<const T>> checkpoints, std::type_identity_t<std::span<T>> dstates,
                                   std::type_identity_t<std::span<T>> work)
{
    const std::size_t rows = detail::delta_rule::rows_of<C>(in);
    assert(sequences > 0 && rows % sequences == 0 && dout.size() == in.v.size() && grads.q.size() == in.q.size()
           && grads.k.size() == in.k.size() && grads.v.size() == in.v.size() && grads.g.size() == in.g.size()
           && grads.beta.size() == in.beta.size());
    recurrence_backward(detail::delta_rule::make<C, T>(in, {}, dout, grads, rows), rows / sequences, sequences, C.mixer_chunk, checkpoints,
                        dstates, work);
}

/// \brief One token of the gated delta rule for decoding.
/// \tparam C The configuration.
/// \param in One row's inputs. \param o Receives its `H · P` outputs. \param state `delta_rule_state_size(C)`, updated.
template <transformer_config C, math::math_number T>
constexpr void delta_rule_step(const delta_rule_inputs<T>& in, std::type_identity_t<std::span<T>> o, std::type_identity_t<std::span<T>> state)
{
    const std::size_t rows = detail::delta_rule::rows_of<C>(in);
    assert(rows == 1 && o.size() == in.v.size() && state.size() == delta_rule_state_size(C));
    detail::delta_rule::make<C, T>(in, o, {}, {}, rows).step(0, state);
}

} // namespace snicholls::ml

#endif // snicholls_core_ml_delta_rule_hpp
