//
//  patch.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::ml` patches and positions for vision transformers: `patch_grid`, `patchify_rows` / `unpatchify_rows` for images, video tubelets and spectrograms, fixed sine-cosine position embeddings in two and three dimensions, axial RoPE, and `gathered_rope` for tokens at scattered positions.
/// \ingroup core
///
/// **Overview.** A vision transformer reads an image as a sequence of patches (Dosovitskiy et al., 2021); I-JEPA
/// (Assran et al., 2023) does, and V-JEPA (Bardes et al., 2024) reads a video as tubelets, a patch over a few frames.
/// This header turns pixels into token rows and back, and gives each token its position.
///
/// - `patch_grid`: a volume of `frames x height x width x channels` values (channels last: an image is one frame; a
///   spectrogram is one frame of one channel, frequency by time, with rectangular patches, as Audio-JEPA reads one),
///   cut into tubelets of `tubelet x patch_height x patch_width`. Its tokens are numbered in raster order, frame
///   group, then row, then column: `token = (gt · grid_height + gy) · grid_width + gx`; a token's row is its values in
///   the order `(dt, dy, dx, c)`, as MAE's and I-JEPA's `patchify` lay them out (`'nchpwq->nhwpqc'`).
/// - `patch_source(g, token, element)`: where element `element` of token `token`'s row lives in the volume.
/// - `patchify_rows(b, g, volumes, tokens, samples)`: every token's row, each value read once from the volume (a
///   gather: nothing is copied but the rows). `unpatchify_rows` is its inverse; the two are permutations, so each is
///   the other's backward pass, to the bit. The patch embedding that follows is a `linear_sequence` (`ml/linear.hpp`),
///   as a strided convolution of stride `p` is.
/// - `sincos_position_2d(dim, gh, gw, out, layout)` and `sincos_position_3d(dim, gt, gh, gw, out, uniform_power)`:
///   the fixed sine-cosine embeddings of MAE, I-JEPA and V-JEPA, one row of `dim` per token. In one dimension, with
///   `h = d / 2` and `ω_i = 1 / 10000^(i / h)`, the row of position `p` is `[sin(p ω_0) .. sin(p ω_{h-1}), cos(p ω_0)
///   .. cos(p ω_{h-1})]`. In two, half the channels encode the row and half the column: `[row, column]`
///   (`sincos_layout::row_first`, V-JEPA's `get_2d_sincos_pos_embed`) or `[column, row]` (`column_first`, as MAE's
///   and I-JEPA's code order them, recalled, see Known issues). In three, `[frame, row, column]` with `dim / 2`,
///   `dim / 4`, `dim / 4` channels, or with `uniform_power` `2 · ceil(dim / 6)` each, the row cut to `dim` (V-JEPA's
///   `get_3d_sincos_pos_embed`). Computed in `double` and rounded once to `T`.
/// - `sincos_table_2d<T, Dim, Gh, Gw>` and `sincos_table_3d<T, Dim, Gt, Gh, Gw>`: the same, built by a `constexpr`
///   constructor, so a fixed grid's table is a constant the compiler holds (or is built at start-up when it exceeds
///   the compiler's evaluation limits, as `rope_table`).
/// - `axial_rope_table<T, HeadDim, Gt, Gh, Gw>`: RoPE in two or three dimensions (axial: each axis rotates its own
///   share of the head's channel pairs by its own coordinate; Heo et al., 2024, "Rotary Position Embedding for Vision
///   Transformer", and V-JEPA 2's 3-D RoPE, recalled). A `rope_source` by token index, so a configured attention layer
///   rotates by it unchanged.
/// - `gathered_rope<R>`: a `rope_source` over another one at a list of positions: the context encoder's tokens sit at
///   their patches' positions, not at `0, 1, 2, ...`. Its `sequence(s)` gives sequence `s` of a batch its own list
///   (`ml/attention_layer.hpp` asks for it).
///
/// \snippet ml/patch_demo.cpp patchify
///
/// \snippet ml/patch_demo.cpp positions
///
/// **Relationships.** `ml/linear.hpp` (the patch embedding), `ml/rope.hpp` (`rope_source`, `rope_frequency`),
/// `ml/gather.hpp` and `ml/masking.hpp` (the tokens a model reads), `ml/jepa.hpp`. TRAINING.md T11. Group: \ref core.
///
/// **Implementations.** One each; `patchify_rows` and `unpatchify_rows` on the host for any backend, for a device
/// library to overload (TRAINING.md T9).
///
/// **Guarantees.** `constexpr`, no allocation, linear in the values. `patch_grid` is trivially copyable and
/// structural. `patchify_rows` then `unpatchify_rows` is the identity to the bit. A table built at compile time and the
/// same function at run time agree within the last bits of `math::sin` and `math::cos`, which defer to `<cmath>` at run
/// time.
///
/// **Errors.** Preconditions, asserted (`valid(grid)`): the frames, height and width divide by the tubelet and patch
/// sizes; the sizes of the spans; an even share of channels per axis (`dim` a multiple of 4 in two dimensions, of 8 in
/// three; even and at least 6 with `uniform_power`).
///
/// **Performance.** `patchify_rows` against the loops by hand (`benchmarks/ml/patch_benchmark.cpp`): see
/// `BENCHMARKS.md`. The tables are built once.
///
/// **Tests and demos.** `tests/ml/patch_test.cpp`, `demos/ml/patch_demo.cpp`.
///
/// **Dependencies.** `<algorithm>`, `<array>`, `<cassert>`, `<cstddef>`, `<cstdint>`, `<span>`, `<type_traits>`,
/// `snicholls/core/math/constexpr_math.hpp`, `ml/rope.hpp`.
///
/// **Known issues.** MAE's `get_2d_sincos_pos_embed` (which I-JEPA copies) builds its grid as `meshgrid(grid_w,
/// grid_h)` and gives its first half to that grid's first component, the column; V-JEPA's builds `[row, column]`. The
/// layout of MAE and I-JEPA is recalled, not checked against their code here: pass the one the checkpoint was trained
/// with. V-JEPA 2's split of a head among its axes (`2 · ((head_dim / 3) / 2)` channels each) is recalled; this header
/// splits the pairs evenly, `(HeadDim / 2) / axes` each, the rest unrotated.
///
/// **Status.** Experimental, new in 0.16.0.

#ifndef snicholls_core_ml_patch_hpp
#define snicholls_core_ml_patch_hpp

#include <snicholls/core/math/constexpr_math.hpp>
#include <snicholls/core/ml/rope.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>

namespace snicholls::ml
{

/// \brief A volume of `frames x height x width x channels` values cut into tubelets of `tubelet x patch_height x
/// patch_width`: an image (one frame), a video, a spectrogram (one frame, one channel).
/// \ingroup core
struct patch_grid
{
    std::size_t frames = 1;         ///< The frames of the volume; 1 for an image.
    std::size_t height = 0;         ///< Its rows.
    std::size_t width = 0;          ///< Its columns.
    std::size_t channels = 1;       ///< Its channels, last.
    std::size_t tubelet = 1;        ///< The frames of a tubelet; 1 for an image.
    std::size_t patch_height = 0;   ///< The rows of a patch.
    std::size_t patch_width = 0;    ///< The columns of a patch.

    /// \brief The frame groups. \return `frames / tubelet`.
    [[nodiscard]] constexpr std::size_t grid_frames() const noexcept { return frames / tubelet; }
    /// \brief The patch rows. \return `height / patch_height`.
    [[nodiscard]] constexpr std::size_t grid_height() const noexcept { return height / patch_height; }
    /// \brief The patch columns. \return `width / patch_width`.
    [[nodiscard]] constexpr std::size_t grid_width() const noexcept { return width / patch_width; }
    /// \brief The tokens of one volume. \return `grid_frames · grid_height · grid_width`.
    [[nodiscard]] constexpr std::size_t tokens() const noexcept { return grid_frames() * grid_height() * grid_width(); }
    /// \brief The values of a token's row. \return `tubelet · patch_height · patch_width · channels`.
    [[nodiscard]] constexpr std::size_t token_size() const noexcept { return tubelet * patch_height * patch_width * channels; }
    /// \brief The values of one volume. \return `frames · height · width · channels`.
    [[nodiscard]] constexpr std::size_t values() const noexcept { return frames * height * width * channels; }
};

/// \brief Whether a grid cuts evenly: every size set and divided by its tubelet or patch size.
/// \param g The grid.
/// \return Whether it is valid.
[[nodiscard]] constexpr bool valid(const patch_grid& g) noexcept
{
    return g.frames > 0 && g.height > 0 && g.width > 0 && g.channels > 0 && g.tubelet > 0 && g.patch_height > 0
           && g.patch_width > 0 && g.frames % g.tubelet == 0 && g.height % g.patch_height == 0 && g.width % g.patch_width == 0;
}

/// \brief Where element `element` of token `token`'s row lives in the volume.
/// \param g The grid.
/// \param token The token, below `g.tokens()`.
/// \param element The element of its row, below `g.token_size()`, in the order `(dt, dy, dx, c)`.
/// \return The offset in the volume, `((f · height + y) · width + x) · channels + c`.
[[nodiscard]] constexpr std::size_t patch_source(const patch_grid& g, std::size_t token, std::size_t element) noexcept
{
    const std::size_t gw = g.grid_width(), gh = g.grid_height();
    const std::size_t gx = token % gw, gy = token / gw % gh, gt = token / gw / gh;
    const std::size_t c = element % g.channels, dx = element / g.channels % g.patch_width,
                      dy = element / g.channels / g.patch_width % g.patch_height,
                      dt = element / g.channels / g.patch_width / g.patch_height;
    const std::size_t f = gt * g.tubelet + dt, y = gy * g.patch_height + dy, x = gx * g.patch_width + dx;
    return ((f * g.height + y) * g.width + x) * g.channels + c;
}

/// \brief Every token's row of every volume: `tokens(s, t)[e] = volumes(s)[patch_source(g, t, e)]`.
/// \param g The grid.
/// \param volumes `samples · g.values()` values.
/// \param tokens Receives `samples · g.tokens()` rows of `g.token_size()`.
/// \param samples The volumes.
template <typename B, math::math_number T>
constexpr void patchify_rows(const B&, const patch_grid& g, std::type_identity_t<std::span<const T>> volumes, std::span<T> tokens,
                             std::size_t samples = 1) noexcept(std::is_floating_point_v<T>)
{
    assert(valid(g) && volumes.size() == samples * g.values() && tokens.size() == volumes.size());
    const std::size_t row = g.token_size(), run = g.patch_width * g.channels;
    for (std::size_t s = 0; s < samples; ++s)
        for (std::size_t t = 0; t < g.tokens(); ++t)
            for (std::size_t e = 0; e < row; e += run)   // a patch row's values are contiguous in the volume
                std::copy_n(volumes.begin() + std::ptrdiff_t(s * g.values() + patch_source(g, t, e)), run,
                            tokens.begin() + std::ptrdiff_t((s * g.tokens() + t) * row + e));
}

/// \brief The inverse of `patchify_rows`, and its backward pass: every token's row written back into its volume.
/// \param g The grid.
/// \param tokens `samples · g.tokens()` rows of `g.token_size()`.
/// \param volumes Receives `samples · g.values()` values.
/// \param samples The volumes.
template <typename B, math::math_number T>
constexpr void unpatchify_rows(const B&, const patch_grid& g, std::type_identity_t<std::span<const T>> tokens, std::span<T> volumes,
                               std::size_t samples = 1) noexcept(std::is_floating_point_v<T>)
{
    assert(valid(g) && volumes.size() == samples * g.values() && tokens.size() == volumes.size());
    const std::size_t row = g.token_size(), run = g.patch_width * g.channels;
    for (std::size_t s = 0; s < samples; ++s)
        for (std::size_t t = 0; t < g.tokens(); ++t)
            for (std::size_t e = 0; e < row; e += run)
                std::copy_n(tokens.begin() + std::ptrdiff_t((s * g.tokens() + t) * row + e), run,
                            volumes.begin() + std::ptrdiff_t(s * g.values() + patch_source(g, t, e)));
}

/// \brief The order of the two halves of a 2-D sine-cosine embedding.
enum class sincos_layout
{
    row_first,      ///< `[row, column]`: V-JEPA's `get_2d_sincos_pos_embed`.
    column_first,   ///< `[column, row]`: MAE's and I-JEPA's (recalled, see the file's Known issues).
};

namespace detail::patch
{
    // One axis of a sine-cosine embedding: width channels (even) of position p, [sin(p w_i) ..., cos(p w_i) ...] with
    // w_i = 1 / 10000^(i / (width/2)), in double, rounded once to T; the first out.size() of them written
    template <typename T>
    constexpr void sincos_axis(std::span<T> out, std::size_t width, std::size_t p)
    {
        const std::size_t h = width / 2;
        for (std::size_t i = 0; i < h; ++i) {
            const double omega = 1.0 / math::pow(10000.0, double(i) / double(h));
            const double a = double(p) * omega;
            if (i < out.size())
                out[i] = static_cast<T>(math::sin(a));
            if (h + i < out.size())
                out[h + i] = static_cast<T>(math::cos(a));
        }
    }

    constexpr std::size_t uniform_axis(std::size_t dim) noexcept { return (dim + 5) / 6 * 2; }
} // namespace detail::patch

/// \brief The fixed 2-D sine-cosine position embedding of a `gh x gw` grid: one row of `dim` per token, raster order.
/// \param dim The channels, a multiple of 4.
/// \param gh The grid's rows.
/// \param gw The grid's columns.
/// \param out Receives `gh · gw · dim` values.
/// \param layout The order of the halves.
template <math::math_number T>
constexpr void sincos_position_2d(std::size_t dim, std::size_t gh, std::size_t gw, std::span<T> out,
                                  sincos_layout layout = sincos_layout::row_first)
{
    assert(dim % 4 == 0 && out.size() == gh * gw * dim);
    const std::size_t a = dim / 2;
    for (std::size_t y = 0; y < gh; ++y)
        for (std::size_t x = 0; x < gw; ++x) {
            const auto row = out.subspan((y * gw + x) * dim, dim);
            const bool rows = layout == sincos_layout::row_first;
            detail::patch::sincos_axis(row.first(a), a, rows ? y : x);
            detail::patch::sincos_axis(row.last(a), a, rows ? x : y);
        }
}

/// \brief The fixed 3-D sine-cosine position embedding of a `gt x gh x gw` grid: `[frame, row, column]` per token.
/// \param dim The channels: a multiple of 8, or with `uniform_power` even and at least 6.
/// \param gt The frame groups.
/// \param gh The rows.
/// \param gw The columns.
/// \param out Receives `gt · gh · gw · dim` values.
/// \param uniform_power `2 · ceil(dim / 6)` channels per axis, the row cut to `dim` (V-JEPA's `uniform_power`);
/// otherwise `dim / 2` for the frame and `dim / 4` for the row and the column.
template <math::math_number T>
constexpr void sincos_position_3d(std::size_t dim, std::size_t gt, std::size_t gh, std::size_t gw, std::span<T> out,
                                  bool uniform_power = false)
{
    assert(out.size() == gt * gh * gw * dim && (uniform_power ? dim % 2 == 0 && dim >= 6 : dim % 8 == 0));
    const std::size_t u = detail::patch::uniform_axis(dim);
    const std::size_t dd = uniform_power ? u : dim / 2, dh = uniform_power ? u : dim / 4, dw = dh;
    for (std::size_t t = 0; t < gt; ++t)
        for (std::size_t y = 0; y < gh; ++y)
            for (std::size_t x = 0; x < gw; ++x) {
                const auto row = out.subspan(((t * gh + y) * gw + x) * dim, dim);
                const std::size_t widths[3] = {dd, dh, dw}, coords[3] = {t, y, x};
                std::size_t at = 0;   // the three axes side by side, the last cut at dim
                for (std::size_t k = 0; k < 3 && at < dim; at += widths[k], ++k)
                    detail::patch::sincos_axis(row.subspan(at, std::min(widths[k], dim - at)), widths[k], coords[k]);
            }
}

/// \brief A 2-D sine-cosine table built by a `constexpr` constructor: `Gh · Gw` rows of `Dim`.
/// \ingroup core
/// \tparam T The number type. \tparam Dim The channels, a multiple of 4. \tparam Gh The rows. \tparam Gw The columns.
template <math::math_number T, std::size_t Dim, std::size_t Gh, std::size_t Gw>
    requires(Dim % 4 == 0 && Gh > 0 && Gw > 0)
struct sincos_table_2d
{
    std::array<T, Gh * Gw * Dim> values{};   ///< Token by token.

    /// \brief Builds the table. \param layout The order of the halves.
    constexpr explicit sincos_table_2d(sincos_layout layout = sincos_layout::row_first) { sincos_position_2d<T>(Dim, Gh, Gw, values, layout); }
    /// \brief One token's row. \param token The token, raster order. \return `Dim` values.
    [[nodiscard]] constexpr std::span<const T, Dim> row(std::size_t token) const { return std::span<const T, Dim>(values.data() + token * Dim, Dim); }
};

/// \brief A 3-D sine-cosine table built by a `constexpr` constructor: `Gt · Gh · Gw` rows of `Dim`.
/// \ingroup core
/// \tparam T The number type. \tparam Dim The channels. \tparam Gt The frame groups. \tparam Gh The rows.
/// \tparam Gw The columns.
template <math::math_number T, std::size_t Dim, std::size_t Gt, std::size_t Gh, std::size_t Gw>
    requires(Dim % 2 == 0 && Dim >= 6 && Gt > 0 && Gh > 0 && Gw > 0)
struct sincos_table_3d
{
    std::array<T, Gt * Gh * Gw * Dim> values{};   ///< Token by token.

    /// \brief Builds the table. \param uniform_power As `sincos_position_3d`.
    constexpr explicit sincos_table_3d(bool uniform_power = false) { sincos_position_3d<T>(Dim, Gt, Gh, Gw, values, uniform_power); }
    /// \brief One token's row. \param token The token, raster order. \return `Dim` values.
    [[nodiscard]] constexpr std::span<const T, Dim> row(std::size_t token) const { return std::span<const T, Dim>(values.data() + token * Dim, Dim); }
};

/// \brief Axial RoPE over a grid of tokens: a `rope_source` by token index (raster order).
///
/// The head's `HeadDim / 2` channel pairs are shared among the axes, `(HeadDim / 2) / axes` each in order (frame, row,
/// column with three axes, `Gt > 1`; row, column with two), pair `i` of an axis rotating by `coordinate ·
/// rope_frequency(i, 2 · share, base)`; pairs left over are not rotated (cosine 1, sine 0).
/// \ingroup core
/// \tparam T The number type. \tparam HeadDim The channels of a head, even. \tparam Gt The frame groups (1 for an
/// image). \tparam Gh The rows. \tparam Gw The columns.
template <math::math_number T, std::size_t HeadDim, std::size_t Gt, std::size_t Gh, std::size_t Gw>
    requires(HeadDim % 2 == 0 && Gt > 0 && Gh > 0 && Gw > 0 && HeadDim / 2 >= (Gt > 1 ? 3 : 2))
struct axial_rope_table
{
    static constexpr std::size_t axes = Gt > 1 ? 3 : 2;                ///< The axes rotated.
    static constexpr std::size_t share = HeadDim / 2 / axes;           ///< The pairs of each axis.
    std::array<std::array<T, HeadDim / 2>, Gt * Gh * Gw> cos{};        ///< `cos[token][pair]`.
    std::array<std::array<T, HeadDim / 2>, Gt * Gh * Gw> sin{};        ///< `sin[token][pair]`.

    /// \brief Builds the table. \param base The base of the frequencies.
    constexpr explicit axial_rope_table(T base = T(10000))
    {
        for (std::size_t token = 0; token < Gt * Gh * Gw; ++token) {
            const std::size_t x = token % Gw, y = token / Gw % Gh, t = token / Gw / Gh;
            const std::size_t coords[3] = {axes == 3 ? t : y, axes == 3 ? y : x, x};
            for (std::size_t p = 0; p < HeadDim / 2; ++p) {
                const std::size_t a = p / share;
                if (a >= axes) {
                    cos[token][p] = T(1);
                    sin[token][p] = T(0);
                    continue;
                }
                const T angle = T(coords[a]) * rope_frequency<T>(p % share, 2 * share, base);
                cos[token][p] = math::cos(angle);
                sin[token][p] = math::sin(angle);
            }
        }
    }
    /// \brief One token's cosines. \param token The token. \return `HeadDim / 2` values.
    [[nodiscard]] constexpr std::span<const T, HeadDim / 2> cos_at(std::size_t token) const { return cos[token]; }
    /// \brief One token's sines. \param token The token. \return `HeadDim / 2` values.
    [[nodiscard]] constexpr std::span<const T, HeadDim / 2> sin_at(std::size_t token) const { return sin[token]; }
};

/// \brief A `rope_source` at a list of positions: `cos_at(t)` is `source.cos_at(positions[t])`.
///
/// For a batch of sequences of `n` tokens each, `positions` holds every sequence's `n` positions end to end, and
/// `sequence(s)` is sequence `s`'s own source (the batched attention steps ask for it).
/// \ingroup core
/// \tparam R The underlying `rope_source`.
template <rope_source R>
struct gathered_rope
{
    const R* source = nullptr;                    ///< The positions' cosines and sines.
    std::span<const std::uint32_t> positions{};   ///< Each token's position, sequence by sequence.
    std::size_t n = 0;                            ///< The tokens of a sequence; 0 for one sequence of `positions.size()`.

    /// \brief Token `t`'s cosines. \param t The token. \return The source's at its position.
    [[nodiscard]] constexpr auto cos_at(std::size_t t) const { return source->cos_at(positions[t]); }
    /// \brief Token `t`'s sines. \param t The token. \return The source's at its position.
    [[nodiscard]] constexpr auto sin_at(std::size_t t) const { return source->sin_at(positions[t]); }
    /// \brief Sequence `s`'s positions. \param s The sequence. \return A source over its `n` positions.
    [[nodiscard]] constexpr gathered_rope sequence(std::size_t s) const
    {
        const std::size_t k = n == 0 ? positions.size() : n;
        return {source, positions.subspan(s * k, k), k};
    }
};

} // namespace snicholls::ml

#endif // snicholls_core_ml_patch_hpp
