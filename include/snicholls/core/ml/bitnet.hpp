//
//  bitnet.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::ml` BitNet b1.58: absmean ternary weights, per-token absmax activations, the straight-through estimator for quantisation-aware training, the ternary x int8 dot product and BitLinear, all `constexpr`.
/// \ingroup core
///
/// **Overview.** BitNet b1.58 trains and runs a transformer whose linear layers have weights in {-1, 0, +1} times one
/// scale and 8-bit activations, so a matrix-vector product is additions and subtractions of int8 numbers and one
/// multiplication per output. The definitions, as BitNet's code has them (microsoft/BitNet 0b341e5):
///
/// | Step | Definition | Source |
/// |---|---|---|
/// | weights | `γ = max(mean |W|, 1e-5)`, `W_q = clamp(round(W / γ), -1, 1)`, `W ≈ W_q · γ` | `weight_quant`, `utils/convert-hf-to-gguf-bitnet.py`, `utils/preprocess-huggingface-bitnet.py` |
/// | activations, per token | `s = 127 / max(max |x|, 1e-5)`, `x_q = clamp(round(x · s), -128, 127)`, `x ≈ x_q / s` | `BitLinear.quant_input`, `gpu/model.py` |
/// | BitLinear | `y_i = (sum_j W_q,ij · x_q,j) / s · γ` | `bitlinear_int8xint2` epilogue, `gpu/bitnet_kernels/bitnet_kernels.h` |
/// | norms | a pre-norm (RMSNorm) before `wqkv` and `w13`; SubLN, an RMSNorm, before `wo` (`attn_sub_norm`) and before `w2` (`ffn_sub_norm`, on `relu(x1)^2 · x3`), eps 1e-5 | `Attention`, `FeedForward`, `TransformerBlock`, `gpu/model.py` |
///
/// `round` is PyTorch's, ties to even. So every BitLinear's input is the output of an RMSNorm: `bitlinear_normed`
/// is the norm and the layer together, `bitlinear` the layer alone.
///
/// \snippet ml/bitnet_demo.cpp bitnet
///
/// - `absmean_quantize(w, q)` and `absmean_fake_quantize(w, out)`: ternary weights, and the float values the training
///   forward uses (`round(W · s).clamp(-1, 1) / s`, `s = 1 / γ`).
/// - `absmax_quantize<Bits>(x, q)` and `absmax_fake_quantize<Bits>(x, out)`: per-token activations, `Bits` = 8 as
///   BitNet b1.58 (`Q_b = 2^(b-1)` as the BitNet paper writes it, the range `[-Q_b, Q_b - 1]`), 4 for a 4-bit variant.
/// - `ste_backward(dout, x, s, qmin, qmax, dx, mode)`: the gradient of a fake-quantised tensor. `ste::identity` is
///   what `x + (quant(x) - x).detach()` gives, the form BitNet trains with (recalled from "The Era of 1-bit LLMs:
///   Training Tips, Code and FAQ", linked from the README, not in the clone); `ste::clipped` zeroes the gradient
///   where the value was clipped (`x · s` outside `[qmin - 1/2, qmax + 1/2]`), the usual variant.
/// - `ternary_dot(t, q)`: the integer dot product of ternary and int8 codes.
/// - `bitlinear(w, x, y)`, `bitlinear_normed(w, norm_weight, x, y)`: the layer's inference forward.
///
/// **Relationships.** `ml/bitnet_kernels.hpp` (BitNet's packed I2_S, TL1 and TL2 layouts and their kernels);
/// `ml/quant_ternary.hpp` (ggml's TQ1_0 and TQ2_0, which store the same ternary weights per 256); `ml/norm.hpp`
/// (`rms_norm`); `ml/linear.hpp` (float and ggml-block projections). Group: \ref core.
///
/// **Implementations.** One, the scalar reference.
///
/// **Guarantees.** `constexpr`, no allocation, linear in the row (the layer in rows x columns). The integer sums are
/// exact; the float steps are BitNet's, in its order, so a result is the reference's to the bit given the same
/// scale. `noexcept`.
///
/// **Errors.** Preconditions, asserted: matching lengths. A row of zeros quantises to zeros with scale `127 / 1e-5`
/// (activations) or `γ = 1e-5` (weights), as the reference's clamp makes it. A NaN quantises to 0 (PyTorch's cast of
/// NaN is undefined).
///
/// **Performance.** Against a hand-written loop (`benchmarks/ml/bitnet_benchmark.cpp`); numbers in BENCHMARKS.md.
///
/// **Tests and demos.** `tests/ml/bitnet_test.cpp`, `demos/ml/bitnet_demo.cpp`.
///
/// **Dependencies.** `<cassert>`, `<cmath>`, `<cstddef>`, `<cstdint>`, `<span>`, `<type_traits>`, `ml/norm.hpp`,
/// `ml/quantize.hpp`, `ml/row.hpp`.
///
/// **Known issues.** The weights' mean |W| is summed in `double` and rounded once; PyTorch sums `float`s pairwise,
/// so for a large tensor its γ can differ in the last bit. BitNet a4.8's activations (4-bit for the inputs of
/// attention and the FFN, sparsified 8-bit elsewhere, arXiv 2411.04965) are not defined in the clone; only the
/// generic `absmax_quantize<4>` is here, and a4.8's own quantiser is deferred.
///
/// **Status.** Experimental, new in 0.17.0.

#ifndef snicholls_core_ml_bitnet_hpp
#define snicholls_core_ml_bitnet_hpp

#include <snicholls/core/ml/norm.hpp>
#include <snicholls/core/ml/quantize.hpp>
#include <snicholls/core/ml/row.hpp>

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>

namespace snicholls::ml
{

/// \brief The floor BitNet clamps a scale's denominator to: mean |W| for weights, max |x| for activations.
inline constexpr float bitnet_eps = 1e-5f;

namespace detail::bitnet
{
    // round half to even (PyTorch's round), then clamp to [lo, hi]; NaN to 0
    constexpr int round_clamp(float v, int lo, int hi)
    {
        if (v != v)
            return 0;
        if (v <= float(lo))
            return lo;
        if (v >= float(hi))
            return hi;
        const int r = quantize::nearest_int(v);   // exact for |v| below 2^22
        return r < lo ? lo : r > hi ? hi : r;
    }
} // namespace detail::bitnet

/// \brief BitNet b1.58's absmean weight quantisation: `γ = max(mean |w|, eps)`, `q = clamp(round(w / γ), -1, 1)`.
/// \param w The weights (a whole matrix: one scale per tensor, as BitNet).
/// \param q Receives `w.size()` codes in {-1, 0, 1}.
/// \param eps The floor of `γ`.
/// \return `γ`: `w ≈ q · γ`.
template <row W>
    requires std::same_as<row_value_t<W>, float>
constexpr float absmean_quantize(const W& w, std::span<std::int8_t> q, float eps = bitnet_eps) noexcept
{
    const auto in = as_row(w);
    assert(q.size() == in.size());
    double sum = 0;
    for (const float v : in)
        sum += v < 0 ? -double(v) : double(v);
    const float mean = in.empty() ? 0.0f : float(sum / double(in.size()));
    const float gamma = mean > eps ? mean : eps;
    const float s = 1.0f / gamma;
    for (std::size_t i = 0; i < in.size(); ++i) {
        const float v = in[i] * s;
        q[i] = static_cast<std::int8_t>(detail::bitnet::round_clamp(v, -1, 1));
    }
    return gamma;
}

/// \brief The training forward of absmean quantisation: `round(w · s).clamp(-1, 1) / s`, `s = 1 / γ`.
/// \param w The weights.
/// \param out Receives the fake-quantised weights.
/// \param eps The floor of `γ`.
/// \return `γ`.
template <row W, mutable_row Out>
    requires std::same_as<row_value_t<W>, float> && std::same_as<row_value_t<Out>, float>
constexpr float absmean_fake_quantize(const W& w, Out&& out, float eps = bitnet_eps) noexcept
{
    const auto in = as_row(w);
    const auto o = as_mutable_row(out);
    assert(o.size() == in.size());
    double sum = 0;
    for (const float v : in)
        sum += v < 0 ? -double(v) : double(v);
    const float mean = in.empty() ? 0.0f : float(sum / double(in.size()));
    const float gamma = mean > eps ? mean : eps;
    const float s = 1.0f / gamma;
    for (std::size_t i = 0; i < in.size(); ++i) {
        const float v = in[i] * s;
        o[i] = float(detail::bitnet::round_clamp(v, -1, 1)) / s;
    }
    return gamma;
}

/// \brief Per-token absmax quantisation to `Bits` bits: `s = (2^(Bits-1) - 1) / max(max |x|, eps)`, `q = clamp(round(x · s), -2^(Bits-1), 2^(Bits-1) - 1)`.
/// \tparam Bits 2 to 8: 8 for BitNet b1.58's activations.
/// \param x One token's activations.
/// \param q Receives `x.size()` codes.
/// \param eps The floor of max |x|.
/// \return `s`: `x ≈ q / s`.
template <int Bits = 8, row X>
    requires(Bits >= 2 && Bits <= 8 && std::same_as<row_value_t<X>, float>)
constexpr float absmax_quantize(const X& x, std::span<std::int8_t> q, float eps = bitnet_eps) noexcept
{
    const auto in = as_row(x);
    assert(q.size() == in.size());
    constexpr int top = (1 << (Bits - 1)) - 1;
    float amax = 0.0f;
    for (const float v : in)
        amax = detail::quantize::fmax(amax, detail::quantize::abs(v));
    const float s = float(top) / (amax > eps ? amax : eps);
    for (std::size_t i = 0; i < in.size(); ++i) {
        const float v = in[i] * s;
        q[i] = static_cast<std::int8_t>(detail::bitnet::round_clamp(v, -top - 1, top));
    }
    return s;
}

/// \brief The training forward of per-token absmax quantisation: `round(x · s).clamp(-2^(Bits-1), 2^(Bits-1) - 1) / s`.
/// \tparam Bits 2 to 8.
/// \param x One token's activations.
/// \param out Receives the fake-quantised activations.
/// \param eps The floor of max |x|.
/// \return `s`.
template <int Bits = 8, row X, mutable_row Out>
    requires(Bits >= 2 && Bits <= 8 && std::same_as<row_value_t<X>, float> && std::same_as<row_value_t<Out>, float>)
constexpr float absmax_fake_quantize(const X& x, Out&& out, float eps = bitnet_eps) noexcept
{
    const auto in = as_row(x);
    const auto o = as_mutable_row(out);
    assert(o.size() == in.size());
    constexpr int top = (1 << (Bits - 1)) - 1;
    float amax = 0.0f;
    for (const float v : in)
        amax = detail::quantize::fmax(amax, detail::quantize::abs(v));
    const float s = float(top) / (amax > eps ? amax : eps);
    for (std::size_t i = 0; i < in.size(); ++i) {
        const float v = in[i] * s;
        o[i] = float(detail::bitnet::round_clamp(v, -top - 1, top)) / s;
    }
    return s;
}

/// \brief The straight-through estimator's two forms.
/// \ingroup core
enum class ste
{
    identity,   ///< `dx = dout` everywhere: the gradient of `x + (quant(x) - x).detach()`, as BitNet trains.
    clipped,    ///< `dx = dout` where `x · s` was not clipped, 0 where it was.
};

/// \brief The backward pass of a fake-quantised tensor through the straight-through estimator.
/// \param dout The gradient of the fake-quantised values.
/// \param x The values before quantisation.
/// \param s The scale the forward multiplied by (`1 / γ` for weights, `absmax_quantize`'s `s` for activations).
/// \param qmin The smallest code (-1 for weights, -128 for int8 activations).
/// \param qmax The largest code.
/// \param dx Receives the gradient of `x` (overwritten).
/// \param mode `ste::identity` or `ste::clipped`.
template <row D, row X, mutable_row DX>
    requires std::same_as<row_value_t<D>, float> && std::same_as<row_value_t<X>, float> && std::same_as<row_value_t<DX>, float>
constexpr void ste_backward(const D& dout, const X& x, float s, int qmin, int qmax, DX&& dx, ste mode = ste::identity) noexcept
{
    const auto g = as_row(dout);
    const auto in = as_row(x);
    const auto out = as_mutable_row(dx);
    assert(g.size() == in.size() && out.size() == in.size());
    const float lo = float(qmin) - 0.5f, hi = float(qmax) + 0.5f;
    for (std::size_t i = 0; i < in.size(); ++i) {
        const float v = in[i] * s;
        out[i] = mode == ste::identity || (v >= lo && v <= hi) ? g[i] : 0.0f;
    }
}

/// \brief The dot product of ternary codes and int8 codes: `sum t_j · q_j`, exact.
/// \param t Codes in {-1, 0, 1}.
/// \param q Codes in [-128, 127].
/// \return The sum.
constexpr std::int32_t ternary_dot(std::span<const std::int8_t> t, std::span<const std::int8_t> q) noexcept
{
    assert(t.size() == q.size());
    std::int32_t s = 0;
    for (std::size_t j = 0; j < t.size(); ++j)
        s += std::int32_t(t[j]) * std::int32_t(q[j]);
    return s;
}

/// \brief A BitLinear layer's weights: a row-major ternary matrix and its absmean scale.
/// \ingroup core
struct bitlinear_weights
{
    std::span<const std::int8_t> ternary;  ///< `rows · cols` codes in {-1, 0, 1}, row `i` the weights of output `i`.
    std::size_t rows = 0;                  ///< Outputs.
    std::size_t cols = 0;                  ///< Inputs.
    float scale = 1.0f;                    ///< `γ`.
};

/// \brief BitLinear's inference forward for one token: the activations quantised per token, then `y_i = (sum_j W_ij · x_q,j) / s · γ`.
/// \param w The weights.
/// \param x The token's `w.cols` inputs.
/// \param y Receives `w.rows` outputs.
/// \param codes Scratch for `w.cols` int8 codes.
template <row X, mutable_row Y>
    requires std::same_as<row_value_t<X>, float> && std::same_as<row_value_t<Y>, float>
constexpr void bitlinear(const bitlinear_weights& w, const X& x, Y&& y, std::span<std::int8_t> codes) noexcept
{
    const auto in = as_row(x);
    const auto out = as_mutable_row(y);
    assert(in.size() == w.cols && out.size() == w.rows && codes.size() == w.cols && w.ternary.size() == w.rows * w.cols);
    const float s = absmax_quantize<8>(in, codes);
    for (std::size_t i = 0; i < w.rows; ++i) {
        const std::int32_t acc = ternary_dot(w.ternary.subspan(i * w.cols, w.cols), codes);
        const float a = float(acc) / s;
        out[i] = a * w.scale;
    }
}

/// \brief An RMSNorm (BitNet's pre-norm or SubLN) and then BitLinear: `bitlinear(w, rms_norm(x, norm_weight, eps))`.
/// \param w The weights.
/// \param norm_weight The norm's `w.cols` gains.
/// \param x The token's `w.cols` inputs.
/// \param y Receives `w.rows` outputs.
/// \param normed Scratch for `w.cols` floats.
/// \param codes Scratch for `w.cols` int8 codes.
/// \param eps The norm's epsilon, 1e-5 in BitNet.
template <row X, row G, mutable_row Y>
    requires std::same_as<row_value_t<X>, float> && std::same_as<row_value_t<G>, float> && std::same_as<row_value_t<Y>, float>
constexpr void bitlinear_normed(const bitlinear_weights& w, const G& norm_weight, const X& x, Y&& y, std::span<float> normed,
                                std::span<std::int8_t> codes, float eps = bitnet_eps) noexcept
{
    rms_norm(x, norm_weight, normed, eps);
    bitlinear(w, std::span<const float>(normed), y, codes);
}

} // namespace snicholls::ml

#endif // snicholls_core_ml_bitnet_hpp
