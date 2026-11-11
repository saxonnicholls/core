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
/// - `rope_layout::half` pairs channel `i` with `i + d/2` (GPT-NeoX, Hugging Face LLaMA, nanochat);
///   `rope_layout::interleaved` pairs `2i` with `2i + 1` (Meta's LLaMA, GPT-J).
///
/// \snippet ml/rope_demo.cpp rope
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
/// `rope_inverse` here. Either direction is a valid RoPE as long as queries and keys agree. LLaMA 3's long-context
/// rescaling of the frequencies is not built in; scale the table's angles yourself.
///
/// **Status.** Experimental, new in 0.7.0.

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
