//
//  probe.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::ml` probes of frozen features: `mean_pool_rows` and the attentive probe (one learned query, one cross-attention, as V-JEPA's `AttentivePooler`), forward and backward.
/// \ingroup core
///
/// **Overview.** A self-supervised encoder is judged by how well a small model on its frozen features does a task:
/// a linear classifier on the mean of its token rows (a linear probe), or on an attentive pooling of them, which V-JEPA
/// (Bardes et al., 2024) uses for every evaluation (TRAINING.md T11).
///
/// - `mean_pool_rows(b, rows, width, out, sequences, lengths)`: each sequence's mean row over its valid rows (all, or
///   the first `lengths[s]`), summed in order and divided once. `mean_pool_rows_backward` spreads the gradient back
///   evenly.
/// - `attentive_probe_forward(b, w, xs, act, out, heads, sequences, lengths)`: one learned query `q0` attends over each
///   sequence's rows, as V-JEPA's `CrossAttentionBlock` (read from its code 2026-11-19): `h = layer_norm(x)` (affine,
///   `norm1`), `q = Wq q0 + bq`, `K = h Wkᵀ + bk`, `V = h Wvᵀ + bv`, each head `y_h = softmax(q_h·K_h / sqrt(d_h)) V_h`
///   over the valid rows, and `out = q0 + y`: V-JEPA's `CrossAttention` has no output projection, and its block adds
///   the attention to the query. The rest of V-JEPA's block, `out = out + mlp(norm2(out))`, and the classifier are
///   `block_norm` / `layer_norm`, `mlp_forward` and `linear` (`ml/mlp.hpp`, `ml/linear.hpp`) on the pooled row.
/// - `attentive_probe_backward(...)`: every gradient (the query's, the projections', the biases', the norm's and, if
///   asked, the features'), the softmax rebuilt from the saved log-sum-exps (TRAINING.md T2).
///
/// \snippet ml/probe_demo.cpp probe
///
/// **Relationships.** `ml/linear.hpp` (`linear_sequence`, the projections through a `linalg` backend),
/// `ml/norm.hpp`, `ml/attention.hpp` (the same softmax arithmetic as `attend`, over strided heads), `ml/jepa.hpp`.
/// Group: \ref core.
///
/// **Implementations.** One, on the host for any `linalg::blas_backend`; the projections are the backend's.
///
/// **Guarantees.** `constexpr`, no allocation; the caller's views. Each head's softmax takes its maximum score first,
/// then sums `e^(s - max)` and the weighted values in row order.
///
/// **Errors.** Preconditions, asserted: the shapes; `heads` divides `width`; at least one valid row per sequence for
/// the attentive probe (a sequence with none gives a zero `y` and passes no gradient).
///
/// **Performance.** Not benchmarked: an evaluation, run once per checkpoint; its products are the backend's.
///
/// **Tests and demos.** `tests/ml/probe_test.cpp`, `demos/ml/probe_demo.cpp`.
///
/// **Dependencies.** `<cassert>`, `<cstddef>`, `<limits>`, `<span>`, `<type_traits>`, `linalg/backend.hpp`,
/// `ml/linear.hpp`, `ml/norm.hpp`.
///
/// **Known issues.** One query and depth one, V-JEPA's defaults (`num_queries = 1`, `depth = 1`); more queries are more
/// calls.
///
/// **Status.** Experimental, new in 0.16.0.

#ifndef snicholls_core_ml_probe_hpp
#define snicholls_core_ml_probe_hpp

#include <snicholls/core/linalg/backend.hpp>
#include <snicholls/core/math/constexpr_math.hpp>
#include <snicholls/core/ml/linear.hpp>
#include <snicholls/core/ml/norm.hpp>

#include <cassert>
#include <cstddef>
#include <limits>
#include <span>
#include <type_traits>

namespace snicholls::ml
{

namespace detail::probe
{
    constexpr std::size_t valid_rows(std::span<const std::size_t> lengths, std::size_t s, std::size_t n) noexcept
    {
        return lengths.empty() ? n : lengths[s];
    }
} // namespace detail::probe

/// \brief Each sequence's mean row: `out_s = (Σ_{r < len_s} x_{s, r}) / len_s`, rows in order.
/// \param rows `sequences · n` rows of `width`.
/// \param width The values of a row.
/// \param out Receives `sequences` rows.
/// \param sequences The sequences.
/// \param lengths The valid rows of each sequence, or empty for all.
template <typename B, math::math_number T>
constexpr void mean_pool_rows(const B&, std::type_identity_t<std::span<const T>> rows, std::size_t width, std::span<T> out,
                              std::size_t sequences = 1, std::span<const std::size_t> lengths = {})
{
    assert(width > 0 && sequences > 0 && rows.size() % (width * sequences) == 0 && out.size() == sequences * width
           && (lengths.empty() || lengths.size() == sequences));
    const std::size_t n = rows.size() / width / sequences;
    for (std::size_t s = 0; s < sequences; ++s) {
        const std::size_t len = detail::probe::valid_rows(lengths, s, n);
        assert(len <= n);
        for (std::size_t c = 0; c < width; ++c)
            out[s * width + c] = T(0);
        for (std::size_t r = 0; r < len; ++r)
            for (std::size_t c = 0; c < width; ++c)
                out[s * width + c] += rows[(s * n + r) * width + c];
        for (std::size_t c = 0; c < width; ++c)
            out[s * width + c] = len == 0 ? T(0) : out[s * width + c] / T(len);
    }
}

/// \brief The backward pass of `mean_pool_rows`: `drows(s, r) += dout_s / len_s` for each valid row.
/// \param dout The gradients in the pooled rows, `sequences · width`.
/// \param width The values of a row.
/// \param drows Added into, `sequences · n` rows.
/// \param sequences The sequences.
/// \param lengths The forward pass's lengths.
template <typename B, math::math_number T>
constexpr void mean_pool_rows_backward(const B&, std::type_identity_t<std::span<const T>> dout, std::size_t width, std::span<T> drows,
                                       std::size_t sequences = 1, std::span<const std::size_t> lengths = {})
{
    assert(width > 0 && sequences > 0 && drows.size() % (width * sequences) == 0 && dout.size() == sequences * width);
    const std::size_t n = drows.size() / width / sequences;
    for (std::size_t s = 0; s < sequences; ++s) {
        const std::size_t len = detail::probe::valid_rows(lengths, s, n);
        for (std::size_t r = 0; r < len; ++r)
            for (std::size_t c = 0; c < width; ++c)
                drows[(s * n + r) * width + c] += dout[s * width + c] / T(len);
    }
}

/// \brief The weights of an attentive probe: views the caller fills.
/// \ingroup core
/// \tparam T The number type.
template <math::math_number T>
struct attentive_probe_weights
{
    std::span<const T> query{};         ///< `width`: the learned query `q0`.
    dense_weights<T> q{};               ///< `width x width`.
    dense_weights<T> k{};               ///< `width x width`.
    dense_weights<T> v{};               ///< `width x width`.
    std::span<const T> q_bias{};        ///< `width`, or empty.
    std::span<const T> k_bias{};        ///< `width`, or empty.
    std::span<const T> v_bias{};        ///< `width`, or empty.
    std::span<const T> norm_weight{};   ///< `width`, or empty: the layer norm of the features.
    std::span<const T> norm_bias{};     ///< `width`, or empty.
    T norm_eps = T(1e-6);               ///< The norm's epsilon (V-JEPA's layer norms: 1e-6).
};

/// \brief What `attentive_probe_forward` saves: views the caller owns.
/// \ingroup core
/// \tparam T The number type.
template <math::math_number T>
struct attentive_probe_activations
{
    std::span<T> h{};     ///< `sequences · n · width`: the normalised features.
    std::span<T> k{};     ///< `sequences · n · width`: the keys.
    std::span<T> v{};     ///< `sequences · n · width`: the values.
    std::span<T> q{};     ///< `width`: the projected query.
    std::span<T> y{};     ///< `sequences · width`: the heads' outputs.
    std::span<T> lse{};   ///< `sequences · heads`: each head's log-sum-exp.
};

/// \brief The gradients of an attentive probe: views the caller owns.
/// \ingroup core
/// \tparam T The number type.
template <math::math_number T>
struct attentive_probe_gradients
{
    std::span<T> query{};         ///< `width`: added into.
    std::span<T> wq{};            ///< `width · width`: added into.
    std::span<T> wk{};            ///< Added into.
    std::span<T> wv{};            ///< Added into.
    std::span<T> q_bias{};        ///< Added into, or empty.
    std::span<T> k_bias{};        ///< Added into, or empty.
    std::span<T> v_bias{};        ///< Added into, or empty.
    std::span<T> norm_weight{};   ///< Added into, or empty.
    std::span<T> norm_bias{};     ///< Added into, or empty.
    std::span<T> dq{};            ///< `width`: working space, written.
    std::span<T> dk{};            ///< `sequences · n · width`: working space, written.
    std::span<T> dv{};            ///< `sequences · n · width`: working space, written.
    std::span<T> dh{};            ///< `sequences · n · width`: working space, written.
};

/// \brief The attentive probe: one learned query attends over each sequence's rows; `out_s = q0 + y_s`.
/// \param backend A `linalg::blas_backend`.
/// \param w The weights.
/// \param xs `sequences · n` feature rows of `width`.
/// \param act Receives the activations.
/// \param out Receives `sequences` rows of `width`.
/// \param heads The heads, dividing `width`.
/// \param sequences The sequences.
/// \param lengths The valid rows of each sequence, or empty for all.
template <typename B, math::math_number T>
    requires linalg::blas_backend<B, T>
constexpr void attentive_probe_forward(const B& backend, const attentive_probe_weights<T>& w, std::type_identity_t<std::span<const T>> xs,
                                       const attentive_probe_activations<T>& act, std::type_identity_t<std::span<T>> out, std::size_t heads,
                                       std::size_t sequences = 1, std::span<const std::size_t> lengths = {})
{
    const std::size_t d = w.query.size(), n = d == 0 || sequences == 0 ? 0 : xs.size() / d / sequences;
    assert(d > 0 && heads > 0 && d % heads == 0 && xs.size() == sequences * n * d && out.size() == sequences * d
           && act.h.size() == xs.size() && act.k.size() == xs.size() && act.v.size() == xs.size() && act.q.size() == d
           && act.y.size() == sequences * d && act.lse.size() == sequences * heads && (lengths.empty() || lengths.size() == sequences));
    for (std::size_t r = 0; r < sequences * n; ++r)
        layer_norm(xs.subspan(r * d, d), w.norm_weight, w.norm_bias, act.h.subspan(r * d, d), w.norm_eps);
    linear_sequence(backend, w.q, w.query, act.q, w.q_bias);
    linear_sequence(backend, w.k, std::span<const T>(act.h), act.k, w.k_bias);
    linear_sequence(backend, w.v, std::span<const T>(act.h), act.v, w.v_bias);
    const std::size_t dh = d / heads;
    const T scale = T(1) / math::sqrt(T(dh));
    for (std::size_t s = 0; s < sequences; ++s) {
        const std::size_t len = detail::probe::valid_rows(lengths, s, n);
        for (std::size_t h = 0; h < heads; ++h) {
            const auto qh = std::span<const T>(act.q).subspan(h * dh, dh);
            const auto yh = act.y.subspan(s * d + h * dh, dh);
            T max = -std::numeric_limits<T>::infinity();
            for (std::size_t j = 0; j < len; ++j) {
                T sc = T(0);
                for (std::size_t i = 0; i < dh; ++i)
                    sc += qh[i] * act.k[(s * n + j) * d + h * dh + i];
                max = scale * sc > max ? scale * sc : max;
            }
            for (T& v : yh)
                v = T(0);
            T sum = T(0);
            for (std::size_t j = 0; j < len; ++j) {
                T sc = T(0);
                for (std::size_t i = 0; i < dh; ++i)
                    sc += qh[i] * act.k[(s * n + j) * d + h * dh + i];
                const T e = math::exp(scale * sc - max);
                sum += e;
                for (std::size_t i = 0; i < dh; ++i)
                    yh[i] += e * act.v[(s * n + j) * d + h * dh + i];
            }
            for (T& v : yh)
                v = sum == T(0) ? T(0) : v / sum;
            act.lse[s * heads + h] = sum == T(0) ? -std::numeric_limits<T>::infinity() : max + math::log(sum);
        }
        for (std::size_t c = 0; c < d; ++c)
            out[s * d + c] = w.query[c] + act.y[s * d + c];
    }
}

/// \brief The backward pass of `attentive_probe_forward`: the weights' gradients added into, the features' added into
/// `dxs` if it is not empty.
/// \param backend A `linalg::blas_backend`.
/// \param w The weights the forward pass used.
/// \param xs The features it read.
/// \param act What it saved.
/// \param dout The gradients in its output, `sequences · width`.
/// \param dxs Added into, `xs`' size; or empty.
/// \param grads The gradients and working spaces.
/// \param heads The forward pass's heads.
/// \param sequences The sequences.
/// \param lengths The forward pass's lengths.
template <typename B, math::math_number T>
    requires linalg::blas_backend<B, T>
constexpr void attentive_probe_backward(const B& backend, const attentive_probe_weights<T>& w, std::type_identity_t<std::span<const T>> xs,
                                        const attentive_probe_activations<T>& act, std::type_identity_t<std::span<const T>> dout,
                                        std::type_identity_t<std::span<T>> dxs, const attentive_probe_gradients<T>& grads, std::size_t heads,
                                        std::size_t sequences = 1, std::span<const std::size_t> lengths = {})
{
    const std::size_t d = w.query.size(), n = d == 0 || sequences == 0 ? 0 : xs.size() / d / sequences;
    assert(dout.size() == sequences * d && (dxs.empty() || dxs.size() == xs.size()) && grads.dq.size() == d
           && grads.dk.size() == xs.size() && grads.dv.size() == xs.size() && grads.dh.size() == xs.size() && grads.query.size() == d);
    const std::size_t dh = d / heads;
    const T scale = T(1) / math::sqrt(T(dh));
    for (T& v : grads.dq)
        v = T(0);
    for (T& v : grads.dk)
        v = T(0);
    for (T& v : grads.dv)
        v = T(0);
    for (T& v : grads.dh)
        v = T(0);
    for (std::size_t s = 0; s < sequences; ++s) {
        for (std::size_t c = 0; c < d; ++c)
            grads.query[c] += dout[s * d + c];   // the residual
        const std::size_t len = detail::probe::valid_rows(lengths, s, n);
        for (std::size_t h = 0; h < heads; ++h) {
            const T lse = act.lse[s * heads + h];
            if (lse == -std::numeric_limits<T>::infinity())
                continue;
            const auto qh = std::span<const T>(act.q).subspan(h * dh, dh);
            const auto g = dout.subspan(s * d + h * dh, dh);
            T big = T(0);
            for (std::size_t i = 0; i < dh; ++i)
                big += g[i] * act.y[s * d + h * dh + i];
            for (std::size_t j = 0; j < len; ++j) {
                const std::size_t at = (s * n + j) * d + h * dh;
                T sc = T(0), gv = T(0);
                for (std::size_t i = 0; i < dh; ++i) {
                    sc += qh[i] * act.k[at + i];
                    gv += g[i] * act.v[at + i];
                }
                const T p = math::exp(scale * sc - lse);
                const T ds = p * (gv - big);
                for (std::size_t i = 0; i < dh; ++i) {
                    grads.dv[at + i] += p * g[i];
                    grads.dq[h * dh + i] += scale * ds * act.k[at + i];
                    grads.dk[at + i] += scale * ds * qh[i];
                }
            }
        }
    }
    linear_sequence_backward(backend, w.q, w.query, std::span<const T>(grads.dq), grads.query, grads.wq, grads.q_bias);
    linear_sequence_backward(backend, w.k, std::span<const T>(act.h), std::span<const T>(grads.dk), grads.dh, grads.wk, grads.k_bias);
    linear_sequence_backward(backend, w.v, std::span<const T>(act.h), std::span<const T>(grads.dv), grads.dh, grads.wv, grads.v_bias);
    if (dxs.empty() && grads.norm_weight.empty() && grads.norm_bias.empty())
        return;
    for (std::size_t r = 0; r < sequences * n; ++r) {
        const auto x = xs.subspan(r * d, d);
        const auto stats = layer_norm(x, w.norm_weight, w.norm_bias, grads.dk.subspan(r * d, d), w.norm_eps);   // dk is spent
        const auto dx = grads.dv.subspan(r * d, d);                                                             // so is dv
        layer_norm_backward(std::span<const T>(grads.dh).subspan(r * d, d), x, w.norm_weight, stats, dx, grads.norm_weight, grads.norm_bias);
        if (!dxs.empty())
            for (std::size_t c = 0; c < d; ++c)
                dxs[r * d + c] += dx[c];
    }
}

} // namespace snicholls::ml

#endif // snicholls_core_ml_probe_hpp
