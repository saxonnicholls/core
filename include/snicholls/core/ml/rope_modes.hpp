//
//  rope_modes.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::ml` RoPE as ggml names its modes: NORM and NEOX mapped to core's layouts, multimodal RoPE (Qwen2-VL's sections, Qwen3-VL's interleaved sections), vision RoPE in two dimensions, per-pair frequency factors (LongRoPE's short and long), the shift of a rotated key to a new position, and their angle tables, all `constexpr`.
/// \ingroup core
///
/// **Overview.** `ml/rope.hpp` rotates a head by one position. Multimodal models rotate by several coordinates at once
/// (time, height, width, and an extra one), and serving stacks name the layouts by ggml's modes. This header adds those
/// on top of `rope`, `rope_inverse` and `rope_frequency`, as ggml computes them (`ggml_rope_cache_init`,
/// `ggml_mrope_cache_init` and `rotate_pairs` in `ggml/src/ggml-cpu/ops.cpp`, llama.cpp 39039aa):
///
/// | `rope_mode` | ggml | Pairs | Angle of pair `i` |
/// |---|---|---|---|
/// | `normal` | `GGML_ROPE_TYPE_NORMAL` (0) | `2i`, `2i + 1`: `rope_layout::interleaved` | `p · f_i` |
/// | `neox` | `GGML_ROPE_TYPE_NEOX` (2) | `i`, `i + d/2`: `rope_layout::half` | `p · f_i` |
/// | `mrope` | `GGML_ROPE_TYPE_MROPE` (8) | as `neox` | `p_a · f_i`, `a` the section of `i mod S` (Qwen2-VL) |
/// | `imrope` | `GGML_ROPE_TYPE_IMROPE` (40) | as `neox` | `a` chosen by `i mod 3` (Qwen3-VL's interleaved sections) |
/// | `vision` | `GGML_ROPE_TYPE_VISION` (24) | as `neox`, over the whole head | `p_a · base^(-2j / (d/2))`, `j` the pair's index in its section |
///
/// with `f_i = base^(-2i/d) / factor_i` and `S = t + h + w + e` the sections' pairs. A vision encoder with sections
/// `{d/4, d/4, ...}` and positions (row, column) is the axial RoPE of `ml/patch.hpp` (`axial_rope_table`), to the bit.
///
/// - `rope_mode_layout(mode)` and `rope_mode_of(layout)`: ggml's names for core's `rope_layout`.
/// - `rope_section_axis(mode, sections, i)`: which coordinate pair `i` rotates by.
/// - `rope_mode_angles(position, cos, sin, base, mode, sections, factors, amplitude)`: one token's cosines and sines;
///   `rope_mode_table<T, Positions, HeadDim>` holds many tokens' (built by a `constexpr` constructor, a `rope_source` by
///   token index).
/// - LongRoPE (Phi-3; llama.cpp's `rope_long` / `rope_short` tensors): `factors` divides each pair's frequency;
///   `longrope_factors(short, long, context, original)` chooses as `llama_model::get_rope_factors` does (long when the
///   context exceeds the trained one) and `longrope_attention_factor(s, L) = sqrt(1 + ln s / ln L)` is the amplitude the
///   converter writes (`conversion/phi.py`).
/// - Context shift: a key rotated at position `p` is moved to `p + δ` by rotating it by `δ` (the rotations compose), as
///   llama.cpp's K-shift does (`llama_kv_cache::build_rope_shift`): `rope_shift(x, delta, base, layout, factors)`.
///   `ml/kv_cells.hpp` applies it to a cache.
/// - Partial rotation (`n_rot < head_dim`, Phi-2, GLM): pass the head's first `n_rot` channels as the row; the rest are
///   left alone, as ggml copies them.
///
/// \snippet ml/rope_modes_demo.cpp rope_modes
///
/// **Relationships.** Built on `ml/rope.hpp`; `ml/patch.hpp` (`axial_rope_table`, the same angles in vision mode);
/// `ml/kv_cells.hpp` (the K-shift over a cache). Group: \ref core.
///
/// **Implementations.** One each.
///
/// **Guarantees.** `constexpr`, no allocation, linear in the head. `rope_mode_table` is trivially copyable. Each angle
/// is `coordinate · rope_frequency(...) / factor` with one call to `rope_frequency` per pair, so the `neox` and `normal`
/// modes with no factors and amplitude 1 are `rope_angles` to the bit.
///
/// **Errors.** Preconditions, asserted: `cos` and `sin` have one size; `factors` is empty or has their size; for the
/// sectioned modes at least one of `t`, `h`, `w` is set; the factors are positive.
///
/// **Performance.** Not benchmarked: a table read and `rope`'s four multiplications per pair; the tables are built
/// once.
///
/// **Tests and demos.** `tests/ml/rope_modes_test.cpp`, `demos/ml/rope_modes_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<cassert>`, `<concepts>`, `<cstddef>`, `<cstdint>`, `<span>`, `ml/rope.hpp`,
/// `ml/row.hpp`, `snicholls/core/math/constexpr_math.hpp`.
///
/// **Known issues.** ggml builds each frequency as a running product (`theta *= theta_scale`) in `float`; core calls
/// `rope_frequency` (`pow`) per pair, so the angles agree to a few units in the last place, not to the bit. ggml's
/// YaRN ramp inside `rope_yarn` is not repeated here: combine YaRN with `rope_scaling` in `ml/rope.hpp`. llama.cpp's
/// K-shift of a multimodal cache rotates the whole head as `neox` by one position; `rope_shift` does the same.
///
/// **Status.** Experimental, new in 0.17.0.

#ifndef snicholls_core_ml_rope_modes_hpp
#define snicholls_core_ml_rope_modes_hpp

#include <snicholls/core/math/constexpr_math.hpp>
#include <snicholls/core/ml/rope.hpp>
#include <snicholls/core/ml/row.hpp>

#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <span>

namespace snicholls::ml
{

/// \brief ggml's RoPE modes; the values are its `GGML_ROPE_TYPE_*` constants.
/// \ingroup core
enum class rope_mode : int
{
    normal = 0,    ///< Adjacent channels paired (`rope_layout::interleaved`): Llama as Meta wrote it, GPT-J.
    neox = 2,      ///< The halves paired (`rope_layout::half`): GPT-NeoX, Hugging Face's Llama, Qwen.
    mrope = 8,     ///< Multimodal RoPE with contiguous sections (Qwen2-VL, Qwen2.5-VL).
    vision = 24,   ///< RoPE of a vision encoder: sections over the whole head, each axis's frequencies from its own start.
    imrope = 40,   ///< Multimodal RoPE with interleaved sections (Qwen3-VL).
};

/// \brief Core's pairing of a ggml mode.
/// \param m The mode.
/// \return `interleaved` for `normal`, `half` for every other.
[[nodiscard]] constexpr rope_layout rope_mode_layout(rope_mode m) noexcept
{
    return m == rope_mode::normal ? rope_layout::interleaved : rope_layout::half;
}

/// \brief ggml's name for a core pairing.
/// \param l The layout.
/// \return `normal` for `interleaved`, `neox` for `half`.
[[nodiscard]] constexpr rope_mode rope_mode_of(rope_layout l) noexcept
{
    return l == rope_layout::interleaved ? rope_mode::normal : rope_mode::neox;
}

/// \brief The pairs each coordinate rotates in the sectioned modes (ggml's `sections[4]`, Qwen2-VL's `mrope_section`).
/// \ingroup core
struct rope_sections
{
    std::size_t t = 0;   ///< Time (or the text position).
    std::size_t h = 0;   ///< Height.
    std::size_t w = 0;   ///< Width.
    std::size_t e = 0;   ///< An extra coordinate (a vision encoder's fourth).

    /// \brief The pairs of one period of the sections.
    /// \return `t + h + w + e`.
    [[nodiscard]] constexpr std::size_t total() const noexcept { return t + h + w + e; }
};

/// \brief A token's coordinates: one position for text, up to four for the sectioned modes.
/// \ingroup core
struct rope_position
{
    std::int64_t t = 0;   ///< Time, or the position of a text token (the only one `normal` and `neox` read).
    std::int64_t h = 0;   ///< Height.
    std::int64_t w = 0;   ///< Width.
    std::int64_t e = 0;   ///< The extra coordinate.

    /// \brief One coordinate.
    /// \param axis 0 to 3: `t`, `h`, `w`, `e`.
    /// \return The coordinate.
    [[nodiscard]] constexpr std::int64_t at(std::size_t axis) const noexcept { return axis == 0 ? t : axis == 1 ? h : axis == 2 ? w : e; }
};

/// \brief The coordinate pair `i` rotates by: 0 (`t`), 1 (`h`), 2 (`w`) or 3 (`e`), as `ggml_mrope_cache_init` chooses.
/// \param m The mode; `normal` and `neox` always give 0.
/// \param s The sections.
/// \param i The pair.
/// \return The axis.
[[nodiscard]] constexpr std::size_t rope_section_axis(rope_mode m, const rope_sections& s, std::size_t i) noexcept
{
    if (m == rope_mode::normal || m == rope_mode::neox || s.total() == 0)
        return 0;
    const std::size_t sector = i % s.total();
    if (m == rope_mode::imrope) {
        if (sector % 3 == 1 && sector < 3 * s.h)
            return 1;
        if (sector % 3 == 2 && sector < 3 * s.w)
            return 2;
        if (sector % 3 == 0 && sector < 3 * s.t)
            return 0;
        return 3;
    }
    return sector < s.t ? 0 : sector < s.t + s.h ? 1 : sector < s.t + s.h + s.w ? 2 : 3;
}

namespace detail::rope_modes
{
    // The first pair of axis a's section in one period (vision mode restarts each axis's frequencies there)
    constexpr std::size_t section_start(const rope_sections& s, std::size_t a) noexcept
    {
        return a == 0 ? 0 : a == 1 ? s.t : a == 2 ? s.t + s.h : s.t + s.h + s.w;
    }

    template <typename T>
    constexpr T frequency(rope_mode m, const rope_sections& s, std::size_t i, std::size_t pairs, T base)
    {
        if (m != rope_mode::vision)
            return rope_frequency<T>(i, 2 * pairs, base);
        const std::size_t sector = i % s.total(), j = sector - section_start(s, rope_section_axis(m, s, i));
        return rope_frequency<T>(j, pairs, base);   // ggml: n_dims = d/2, theta_scale = base^(-2 / n_dims)
    }
} // namespace detail::rope_modes

/// \brief One token's cosines and sines in a ggml mode: `cos[i] = a · cos(p_axis(i) · f_i / factor_i)`, likewise `sin`.
/// \param position The token's coordinates (`t` alone for `normal` and `neox`).
/// \param cos Receives `d/2` cosines.
/// \param sin Receives `d/2` sines.
/// \param base The base of the frequencies.
/// \param mode The mode.
/// \param sections The sections, for `mrope`, `imrope` and `vision`.
/// \param factors Per-pair divisors of the frequencies (LongRoPE's, a GGUF's `rope_freqs`), `d/2` or empty.
/// \param amplitude Multiplies every cosine and sine (LongRoPE's attention factor; ggml's `attn_factor`).
template <mutable_row C, mutable_row S>
    requires std::same_as<row_value_t<C>, row_value_t<S>>
constexpr void rope_mode_angles(const rope_position& position, C&& cos, S&& sin, std::type_identity_t<row_value_t<C>> base, rope_mode mode,
                                const rope_sections& sections = {}, std::span<const row_value_t<C>> factors = {},
                                std::type_identity_t<row_value_t<C>> amplitude = 1)
{
    using T = row_value_t<C>;
    const auto c = as_mutable_row(cos);
    const auto s = as_mutable_row(sin);
    const bool sectioned = mode != rope_mode::normal && mode != rope_mode::neox;
    assert(c.size() == s.size() && (factors.empty() || factors.size() == c.size()));
    assert(!sectioned || sections.t > 0 || sections.h > 0 || sections.w > 0);
    (void)sectioned;
    for (std::size_t i = 0; i < c.size(); ++i) {
        const std::size_t axis = rope_section_axis(mode, sections, i);
        T f = detail::rope_modes::frequency<T>(mode, sections, i, c.size(), base);
        if (!factors.empty()) {
            assert(factors[i] > T(0));
            f = f / factors[i];
        }
        const T angle = T(position.at(axis)) * f;
        c[i] = amplitude == T(1) ? math::cos(angle) : amplitude * math::cos(angle);
        s[i] = amplitude == T(1) ? math::sin(angle) : amplitude * math::sin(angle);
    }
}

/// \brief Many tokens' cosines and sines in a ggml mode: a `rope_source` by token index, built by a `constexpr`
/// constructor (in a `constexpr` variable for a fixed grid, at start-up for a long one).
/// \ingroup core
/// \tparam T A `math::math_number`.
/// \tparam Positions The tokens.
/// \tparam HeadDim The channels rotated, even.
template <math::math_number T, std::size_t Positions, std::size_t HeadDim>
    requires(HeadDim % 2 == 0 && HeadDim > 0)
struct rope_mode_table
{
    std::array<std::array<T, HeadDim / 2>, Positions> cos{};   ///< `cos[token][pair]`.
    std::array<std::array<T, HeadDim / 2>, Positions> sin{};   ///< `sin[token][pair]`.

    /// \brief Builds the table from each token's coordinates.
    /// \param positions The tokens' coordinates, `Positions` of them.
    /// \param base The base.
    /// \param mode The mode.
    /// \param sections The sections, for the sectioned modes.
    /// \param factors Per-pair divisors, `HeadDim / 2` or empty.
    /// \param amplitude Multiplies every entry.
    constexpr rope_mode_table(std::span<const rope_position, Positions> positions, T base, rope_mode mode, const rope_sections& sections = {},
                              std::span<const T> factors = {}, T amplitude = T(1))
    {
        for (std::size_t p = 0; p < Positions; ++p)
            rope_mode_angles(positions[p], cos[p], sin[p], base, mode, sections, factors, amplitude);
    }

    /// \brief One token's cosines.
    /// \param token The token, below `Positions`.
    /// \return A span of `HeadDim / 2`.
    [[nodiscard]] constexpr std::span<const T, HeadDim / 2> cos_at(std::size_t token) const { return cos[token]; }
    /// \brief One token's sines.
    /// \param token The token, below `Positions`.
    /// \return A span of `HeadDim / 2`.
    [[nodiscard]] constexpr std::span<const T, HeadDim / 2> sin_at(std::size_t token) const { return sin[token]; }
};

/// \brief The coordinates of a grid of image patches for `vision` mode (row, column), in raster order, as the
/// multimodal projectors feed them.
/// \param rows The patch rows.
/// \param columns The patch columns.
/// \param out Receives `rows · columns` positions.
constexpr void vision_rope_positions(std::size_t rows, std::size_t columns, std::span<rope_position> out)
{
    assert(out.size() == rows * columns);
    for (std::size_t r = 0; r < rows; ++r)
        for (std::size_t c = 0; c < columns; ++c)
            out[r * columns + c] = {static_cast<std::int64_t>(r), static_cast<std::int64_t>(c), 0, 0};
}

/// \brief LongRoPE's choice of factors: the long ones when the context exceeds the trained one, as
/// `llama_model::get_rope_factors` chooses.
/// \tparam T A `math::math_number`.
/// \param short_factors The factors for contexts up to `original_context`.
/// \param long_factors The factors beyond.
/// \param context The context the model runs with.
/// \param original_context The context it was trained on.
/// \return One of the two.
template <math::math_number T>
[[nodiscard]] constexpr std::span<const T> longrope_factors(std::span<const T> short_factors, std::span<const T> long_factors,
                                                            std::size_t context, std::size_t original_context) noexcept
{
    return context > original_context ? long_factors : short_factors;
}

/// \brief LongRoPE's attention factor, the amplitude of its cosines and sines: `sqrt(1 + ln s / ln L)` for `s > 1`, else
/// 1 (Phi-3; llama.cpp's `conversion/phi.py`).
/// \tparam T A `math::math_number`.
/// \param scale `s`, the longest context over the trained one.
/// \param original_context `L`, the trained context, above 1.
/// \return The factor.
template <math::math_number T>
[[nodiscard]] constexpr T longrope_attention_factor(T scale, std::size_t original_context)
{
    return scale > T(1) ? math::sqrt(T(1) + math::log(scale) / math::log(T(original_context))) : T(1);
}

/// \brief Moves a key rotated at position `p` to `p + delta`, in place: a rotation by `delta` (llama.cpp's K-shift).
/// \param x The rotated head (or its first `n_rot` channels).
/// \param delta The shift, negative to move back.
/// \param base The base of the frequencies.
/// \param layout The pairing the key was rotated with.
/// \param factors Per-pair divisors it was rotated with, `x.size() / 2` or empty.
template <mutable_row X>
constexpr void rope_shift(X&& x, std::int64_t delta, std::type_identity_t<row_value_t<X>> base, rope_layout layout = rope_layout::half,
                          std::span<const row_value_t<X>> factors = {})
{
    using T = row_value_t<X>;
    const auto v = as_mutable_row(x);
    const std::size_t h = v.size() / 2;
    assert(v.size() % 2 == 0 && (factors.empty() || factors.size() == h));
    if (delta == 0)
        return;
    // rope by rope_angles(|delta|) (rope_inverse for a negative shift), one pair at a time so no table is needed
    const T distance = T(delta < 0 ? -delta : delta), sign = delta < 0 ? T(-1) : T(1);
    for (std::size_t i = 0; i < h; ++i) {
        T f = rope_frequency<T>(i, 2 * h, base);
        if (!factors.empty())
            f = f / factors[i];
        const T angle = distance * f, c = math::cos(angle), sn = sign * math::sin(angle);
        T& a = layout == rope_layout::half ? v[i] : v[2 * i];
        T& b = layout == rope_layout::half ? v[i + h] : v[2 * i + 1];
        const T x1 = a, x2 = b;
        a = x1 * c - x2 * sn;
        b = x1 * sn + x2 * c;
    }
}

} // namespace snicholls::ml

#endif // snicholls_core_ml_rope_modes_hpp
