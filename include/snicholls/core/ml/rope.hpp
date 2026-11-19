//
//  rope.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::ml` rotary position embedding: the frequencies, a table of cosines and sines the compiler can build, and the rotation of a head, forward and inverse.
/// \ingroup core
///
/// **Overview.** RoPE (Su et al., 2021), as LLaMA, GPT-NeoX and nanochat use it: each pair of channels of a query
/// or key head is rotated by the angle `pos · base^(-2i/d)`, so that the dot product of a query and a key depends on
/// their distance only.
///
/// - `rope_frequency<T>(i, d, base)`: `base^(-2i/d)`, the angular frequency of pair `i` of a head of `d` channels.
/// - `rope_angles(pos, cos, sin, base)`: one position's row of `d/2` cosines and sines.
/// - `rope_table<T, Positions, HeadDim>`: every position's rows, built by a `constexpr` constructor: in a
///   `constexpr` variable for a table the compiler holds (a fixed context on an FPGA, a `float` table a GPU kernel
///   reads), or at start-up for a long context, which would exceed the compiler's evaluation limits.
/// - `rope(x, cos, sin, layout)` rotates a head by `+θ`; `rope_inverse` by `-θ`. The rotation is orthogonal, so
///   the backward pass of `rope` is `rope_inverse` applied to the gradient, and the other way round.
/// - `rope_source<R>`: anything with `cos_at(pos)` and `sin_at(pos)`, such as `rope_table`; `no_rope` stands in where
///   a model has no RoPE (`ml/attention_layer.hpp`, `ml/self_attention.hpp`).
/// - Long context: `rope_scaling<T>` rescales the frequencies (`rope_scaled_frequency`) for a context `s` times the
///   trained one: position interpolation (`linear`, every frequency over `s`; Chen et al., 2023), `ntk` (the base raised to
///   `base · s^(d/(d-2))`; YaRN section 3.1), and YaRN's NTK-by-parts (Peng et al., 2023, eqs. 10 to 13): pairs that turn
///   fewer than `α` times over the trained context are interpolated, more than `β` times kept, a ramp `yarn_ramp` between.
///   `yarn` is the paper's ramp, linear in the rotations `r`; `yarn_index_ramp` is its published code's, which
///   DeepSeek-V3 also uses: linear in the pair index between the correction dimensions (`yarn_correction_dim`) rounded
///   down and up. `yarn_attention_factor(s) = 0.1 ln s + 1` is YaRN's temperature, applied squared to the scores: as the
///   table's `amplitude` (queries and keys each scaled) or in a softmax scale. `rope_angles(pos, cos, sin, base,
///   scaling)` and `rope_table(base, scaling)` build the scaled rows, in the table's type and at compile time in a
///   `constexpr` variable. Every method is plain RoPE at a factor of 1.
/// - `rope_layout::half` pairs channel `i` with `i + d/2` (GPT-NeoX, Hugging Face LLaMA, nanochat);
///   `rope_layout::interleaved` pairs `2i` with `2i + 1` (Meta's LLaMA, GPT-J).
///
/// \snippet ml/rope_demo.cpp rope
///
/// \snippet ml/rope_demo.cpp rope_scaling
///
/// **Relationships.** `ml/attention.hpp` takes the rotated queries and keys; `ml/row.hpp` (the rows). Group: \ref core.
///
/// **Implementations.** One.
///
/// **Guarantees.** `constexpr`, no allocation, linear in the head. `rope_table` is trivially copyable. A rotation
/// keeps a pair's length up to rounding.
///
/// **Errors.** Preconditions, asserted: the head has an even number of channels, and `cos` and `sin` have half as
/// many. Position 0 is the identity.
///
/// **Performance.** Not benchmarked: four multiplications and two additions per pair, against a table read.
///
/// **Tests and demos.** `tests/ml/rope_test.cpp`, `demos/ml/rope_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<cassert>`, `<concepts>`, `<cstddef>`, `<span>`,
/// `snicholls/core/math/constexpr_math.hpp`, `ml/row.hpp`.
///
/// **Known issues.** nanochat's `apply_rotary_emb` rotates by `-θ` (`y1 = x1 cos + x2 sin`): its forward pass is
/// `rope_inverse` here. Either direction is a valid RoPE as long as queries and keys agree. LLaMA 3's own long-context
/// rescaling (a ramp over wavelengths with its own constants) is not built in. YaRN's paper and its code differ in the
/// ramp's shape between the bounds (both are offered); gpt-oss computes the paper's form. A table with an amplitude other
/// than 1 is not orthogonal: `rope_inverse` is still `rope`'s backward pass (the transpose), but no longer its inverse.
///
/// **Status.** Experimental, new in 0.7.0; the rescalings new in 0.16.0.

#ifndef snicholls_core_ml_rope_hpp
#define snicholls_core_ml_rope_hpp

#include <snicholls/core/math/constexpr_math.hpp>
#include <snicholls/core/ml/row.hpp>

#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <span>

namespace snicholls::ml
{

/// \brief Which channels of a head RoPE rotates together.
/// \ingroup core
enum class rope_layout
{
    half,          ///< Channel `i` with `i + d/2`: GPT-NeoX, Hugging Face LLaMA, nanochat.
    interleaved,   ///< Channel `2i` with `2i + 1`: Meta's LLaMA, GPT-J.
};

/// \brief A source of RoPE's cosines and sines by position, such as `rope_table`.
template <typename R>
concept rope_source = requires(const R& r, std::size_t pos) {
    r.cos_at(pos);
    r.sin_at(pos);
};

/// \brief The rope source of a model without RoPE.
/// \ingroup core
struct no_rope_t
{
};

/// \brief Pass where a model has no RoPE.
inline constexpr no_rope_t no_rope{};

/// \brief The angular frequency of pair `i` of a head of `head_dim` channels: `base^(-2i / head_dim)`.
/// \tparam T A `math::math_number`.
/// \param i The pair, below `head_dim / 2`.
/// \param head_dim The channels of the head.
/// \param base The base: 10,000 in GPT-NeoX and nanochat, 500,000 in LLaMA 3.
/// \return The frequency, in radians per position.
template <math::math_number T>
constexpr T rope_frequency(std::size_t i, std::size_t head_dim, T base = T(10000))
{
    return T(1) / math::pow(base, T(2 * i) / T(head_dim));
}

/// \brief One position's cosines and sines: `cos[i] = cos(pos · f_i)`, `sin[i] = sin(pos · f_i)`.
/// \param pos The position.
/// \param cos Receives `d/2` cosines.
/// \param sin Receives `d/2` sines, `cos`'s size.
/// \param base The base of the frequencies.
template <mutable_row C, mutable_row S>
    requires std::same_as<row_value_t<C>, row_value_t<S>>
constexpr void rope_angles(std::size_t pos, C&& cos, S&& sin, std::type_identity_t<row_value_t<C>> base = 10000)
{
    using T = row_value_t<C>;
    const auto c = as_mutable_row(cos);
    const auto s = as_mutable_row(sin);
    assert(c.size() == s.size());
    for (std::size_t i = 0; i < c.size(); ++i) {
        const T angle = T(pos) * rope_frequency<T>(i, 2 * c.size(), base);
        c[i] = math::cos(angle);
        s[i] = math::sin(angle);
    }
}

/// \brief How RoPE's frequencies are rescaled to reach past the context a model was trained on.
/// \ingroup core
enum class rope_scaling_kind
{
    none,              ///< The frequencies as trained.
    linear,            ///< Position interpolation (Chen et al., 2023): every frequency divided by the factor.
    ntk,               ///< NTK-aware: the base raised to `base · s^(d / (d - 2))` (YaRN, Peng et al., 2023, section 3.1).
    yarn,              ///< YaRN's "NTK-by-parts" as the paper writes it (eqs. 10 to 13): the ramp linear in `r = L / λ`.
    yarn_index_ramp,   ///< YaRN as its published code, Hugging Face and DeepSeek-V3 compute it: the ramp linear in the
                       ///< pair index between the rounded correction dimensions.
};

/// \brief The rescaling of RoPE's frequencies for a long context, and the amplitude of the table's cosines and sines.
/// \ingroup core
/// \tparam T A `math::math_number`.
template <math::math_number T>
struct rope_scaling
{
    rope_scaling_kind kind = rope_scaling_kind::none;   ///< The method.
    T factor = T(1);                                    ///< `s = L' / L`, the new context over the trained one.
    std::size_t original_context = 0;                   ///< `L`, the trained context: YaRN only.
    T beta_fast = T(32);                                ///< YaRN's `β`: pairs turning more than `β` times in `L` keep their frequency.
    T beta_slow = T(1);                                 ///< YaRN's `α`: pairs turning fewer than `α` times are interpolated.
    T amplitude = T(1);                                 ///< Multiplies every cosine and sine: YaRN's `sqrt(1/t)` when the
                                                        ///< temperature is applied through the table (see `yarn_attention_factor`).
};

/// \brief YaRN's ramp `γ(r)`: 0 below `alpha`, 1 above `beta`, linear between (YaRN, eq. 11).
/// \tparam T A `math::math_number`.
/// \param r The rotations of a pair over the trained context, `L / λ`.
/// \param alpha The lower bound (`beta_slow`).
/// \param beta The upper bound (`beta_fast`), above `alpha`.
/// \return `γ` in [0, 1].
template <math::math_number T>
constexpr T yarn_ramp(T r, T alpha, T beta) noexcept
{
    return r < alpha ? T(0) : r > beta ? T(1) : (r - alpha) / (beta - alpha);
}

/// \brief The pair index at which a pair turns `rotations` times over `context` positions, as YaRN's code finds it:
/// `head_dim · ln(context / (rotations · 2π)) / (2 ln base)`.
/// \tparam T A `math::math_number`.
/// \param rotations The rotations.
/// \param head_dim The channels the rotation covers.
/// \param base The base of the frequencies.
/// \param context The trained context `L`.
/// \return The (fractional) pair index.
template <math::math_number T>
constexpr T yarn_correction_dim(T rotations, std::size_t head_dim, T base, std::size_t context)
{
    return T(head_dim) * math::log(T(context) / (rotations * T(2) * T(math::pi))) / (T(2) * math::log(base));
}

/// \brief YaRN's attention temperature as a factor on the scores' square root: `sqrt(1/t) = 0.1 · mscale · ln s + 1`,
/// 1 for `s <= 1` (YaRN, definition 2; DeepSeek-V3, section 4.3, with `mscale` 1).
///
/// The scores are multiplied by its square: through the table (`rope_scaling::amplitude`, which scales queries and
/// keys alike, as YaRN and gpt-oss do), or through the softmax scale (DeepSeek-V3, whose rotation covers part of a head
/// only: `latent_attention_scale`).
/// \tparam T A `math::math_number`.
/// \param factor `s`.
/// \param mscale The coefficient's multiplier (DeepSeek's `mscale_all_dim`), 1 for YaRN's.
/// \return The factor.
template <math::math_number T>
constexpr T yarn_attention_factor(T factor, T mscale = T(1))
{
    return factor <= T(1) ? T(1) : T(0.1) * mscale * math::log(factor) + T(1);
}

/// \brief The angular frequency of pair `i`, rescaled as `scaling` says; `rope_frequency` itself with `none`, or with
/// a factor of 1 under any method.
/// \tparam T A `math::math_number`.
/// \param i The pair, below `head_dim / 2`.
/// \param head_dim The channels the rotation covers (more than 2 for `ntk`).
/// \param base The base.
/// \param scaling The method and its parameters.
/// \return The frequency, in radians per position.
template <math::math_number T>
constexpr T rope_scaled_frequency(std::size_t i, std::size_t head_dim, T base, const rope_scaling<T>& scaling)
{
    const T f = rope_frequency<T>(i, head_dim, base), s = scaling.factor;
    if (scaling.kind == rope_scaling_kind::none || s == T(1))
        return f;
    switch (scaling.kind) {
    case rope_scaling_kind::linear:
        return f / s;
    case rope_scaling_kind::ntk:
        assert(head_dim > 2);
        return rope_frequency<T>(i, head_dim, base * math::pow(s, T(head_dim) / T(head_dim - 2)));
    case rope_scaling_kind::yarn: {
        assert(scaling.original_context > 0 && scaling.beta_fast > scaling.beta_slow);
        const T r = T(scaling.original_context) * f / (T(2) * T(math::pi));   // L / λ, λ = 2π / θ
        const T g = yarn_ramp(r, scaling.beta_slow, scaling.beta_fast);
        return (T(1) - g) * f / s + g * f;
    }
    default: {   // yarn_index_ramp
        assert(scaling.original_context > 0 && scaling.beta_fast > scaling.beta_slow);
        const T lo_dim = math::floor(yarn_correction_dim(scaling.beta_fast, head_dim, base, scaling.original_context));
        const T hi_dim = -math::floor(-yarn_correction_dim(scaling.beta_slow, head_dim, base, scaling.original_context));
        const T lo = lo_dim < T(0) ? T(0) : lo_dim;
        T hi = hi_dim > T(head_dim - 1) ? T(head_dim - 1) : hi_dim;
        if (hi == lo)
            hi += T(0.001);
        const T ramp = yarn_ramp(T(i), lo, hi);   // clamp((i - lo) / (hi - lo), 0, 1)
        const T smooth = T(1) - ramp;
        return f / s * (T(1) - smooth) + f * smooth;
    }
    }
}

/// \brief One position's cosines and sines under a rescaling: `cos[i] = a · cos(pos · f'_i)`, `sin[i] = a · sin(pos · f'_i)`,
/// with `f'` from `rope_scaled_frequency` and `a` the scaling's amplitude.
/// \param pos The position.
/// \param cos Receives `d/2` cosines.
/// \param sin Receives `d/2` sines, `cos`'s size.
/// \param base The base of the frequencies.
/// \param scaling The rescaling.
template <mutable_row C, mutable_row S>
    requires std::same_as<row_value_t<C>, row_value_t<S>>
constexpr void rope_angles(std::size_t pos, C&& cos, S&& sin, std::type_identity_t<row_value_t<C>> base,
                           const rope_scaling<row_value_t<C>>& scaling)
{
    using T = row_value_t<C>;
    const auto c = as_mutable_row(cos);
    const auto s = as_mutable_row(sin);
    assert(c.size() == s.size());
    if (scaling.amplitude == T(1) && (scaling.kind == rope_scaling_kind::none || scaling.factor == T(1))) {
        // The plain angles through the plain function, so that they are its bits however each call is optimised (a
        // compiler may turn pow(base, 0.5) into sqrt(base) where it sees the exponent, and a platform's pow differs)
        rope_angles(pos, c, s, base);
        return;
    }
    for (std::size_t i = 0; i < c.size(); ++i) {
        const T angle = T(pos) * rope_scaled_frequency<T>(i, 2 * c.size(), base, scaling);
        c[i] = scaling.amplitude * math::cos(angle);
        s[i] = scaling.amplitude * math::sin(angle);
    }
}

/// \brief Every position's cosines and sines, `Positions` rows of `HeadDim / 2`.
/// \ingroup core
/// \tparam T A `math::math_number`: `float` for a table a GPU kernel reads.
/// \tparam Positions The longest context.
/// \tparam HeadDim The channels of a head, even.
template <math::math_number T, std::size_t Positions, std::size_t HeadDim>
    requires(HeadDim % 2 == 0 && HeadDim > 0)
struct rope_table
{
    std::array<std::array<T, HeadDim / 2>, Positions> cos{};   ///< `cos[pos][i]`.
    std::array<std::array<T, HeadDim / 2>, Positions> sin{};   ///< `sin[pos][i]`.

    /// \brief Builds the table: in a `constexpr` variable, at compile time.
    /// \param base The base of the frequencies.
    constexpr explicit rope_table(T base = T(10000))
    {
        for (std::size_t p = 0; p < Positions; ++p)
            rope_angles(p, cos[p], sin[p], base);
    }

    /// \brief Builds the table with rescaled frequencies (`rope_scaling`): at compile time in a `constexpr` variable.
    /// \param base The base of the frequencies.
    /// \param scaling The rescaling; with `none` and amplitude 1 the table is the plain one to the bit.
    constexpr rope_table(T base, const rope_scaling<T>& scaling)
    {
        for (std::size_t p = 0; p < Positions; ++p)
            rope_angles(p, cos[p], sin[p], base, scaling);
    }

    /// \brief One position's cosines.
    /// \param pos The position, below `Positions`.
    /// \return A span of `HeadDim / 2`.
    [[nodiscard]] constexpr std::span<const T, HeadDim / 2> cos_at(std::size_t pos) const { return cos[pos]; }
    /// \brief One position's sines.
    /// \param pos The position, below `Positions`.
    /// \return A span of `HeadDim / 2`.
    [[nodiscard]] constexpr std::span<const T, HeadDim / 2> sin_at(std::size_t pos) const { return sin[pos]; }
};

namespace detail::rope
{
    template <typename T>
    constexpr void rotate(std::span<T> x, std::span<const T> c, std::span<const T> s, rope_layout layout, T sign)
    {
        const std::size_t h = c.size();
        assert(x.size() == 2 * h && s.size() == h);
        for (std::size_t i = 0; i < h; ++i) {
            T& a = layout == rope_layout::half ? x[i] : x[2 * i];
            T& b = layout == rope_layout::half ? x[i + h] : x[2 * i + 1];
            const T x1 = a, x2 = b, sn = sign * s[i];
            a = x1 * c[i] - x2 * sn;
            b = x1 * sn + x2 * c[i];
        }
    }
} // namespace detail::rope

/// \brief Rotates a head by `+θ`, in place: `(x1, x2) -> (x1 cos - x2 sin, x1 sin + x2 cos)` per pair.
/// \param x The head, `2 · cos.size()` channels.
/// \param cos The position's cosines, from `rope_angles` or a `rope_table`.
/// \param sin The position's sines.
/// \param layout Which channels form a pair.
template <mutable_row X, row C, row S>
    requires std::same_as<row_value_t<X>, row_value_t<C>> && std::same_as<row_value_t<X>, row_value_t<S>>
constexpr void rope(X&& x, const C& cos, const S& sin, rope_layout layout = rope_layout::half)
{
    detail::rope::rotate(as_mutable_row(x), as_row(cos), as_row(sin), layout, row_value_t<X>(1));
}

/// \brief Rotates a head by `-θ`, in place: the inverse of `rope`, and its backward pass.
/// \param x The head, or the gradient in `rope`'s output.
/// \param cos The position's cosines.
/// \param sin The position's sines.
/// \param layout Which channels form a pair.
template <mutable_row X, row C, row S>
    requires std::same_as<row_value_t<X>, row_value_t<C>> && std::same_as<row_value_t<X>, row_value_t<S>>
constexpr void rope_inverse(X&& x, const C& cos, const S& sin, rope_layout layout = rope_layout::half)
{
    detail::rope::rotate(as_mutable_row(x), as_row(cos), as_row(sin), layout, row_value_t<X>(-1));
}

} // namespace snicholls::ml

#endif // snicholls_core_ml_rope_hpp
