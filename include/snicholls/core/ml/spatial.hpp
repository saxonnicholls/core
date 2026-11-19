//
//  spatial.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::ml` spatial operations of vision and audio encoders: max and average pooling of rank 1 to 3 with backward, upscaling (nearest, bilinear, bicubic) with its adjoint, zero and reflect padding, roll, SAM's window partition and decomposed relative-position bias, all `constexpr` over the caller's spans.
/// \ingroup core
///
/// **Overview.** The data movement and resampling around the convolutions of `ml/conv.hpp`, as ggml computes them, for
/// one sample of `[channels][spatial...]` (row-major, the last dimension fastest):
///
/// - `pool_shape<Rank>` and `pool_forward(s, kind, x, y)` / `pool_backward(s, kind, x, dy, dx)`: max or average over
///   windows of `kernel` taps every `stride`, `padding` on both sides; output `(in + 2p - k) / s + 1`. The maximum
///   ignores the padding (a window entirely in it gives the lowest finite value, ggml's `-FLT_MAX`); its gradient goes
///   to the first maximal tap. The average divides by the window's taps inside the input (`count_padding = false`,
///   ggml's `pool_1d`) or by every tap (`true`, ggml's `pool_2d`): `pool_1d_shape` and `pool_2d_shape` make each
///   ggml's way.
/// - `upscale(mode, align_corners, x, in_h, in_w, y, out_h, out_w)` resizes each `[h][w]` plane: `nearest`
///   (`floor(i / scale)`), `bilinear` and `bicubic` (Keys' kernel with `a = -0.75`, PyTorch's) at pixel offset 0.5,
///   or with corners aligned (offset 0, scale `(out - 1) / (in - 1)`), borders clamped: ggml's `ggml_interpolate`
///   modes. `upscale_backward` is the adjoint (every output's gradient spread back with the same weights).
/// - `pad(extents, left, right, x, y)` adds zeros on each side of each dimension (ggml's `ggml_pad_ext`); `pad_backward`
///   crops (its adjoint). `pad_reflect_1d(rows, width, left, right, x, y)` mirrors each row without repeating its
///   edge (ggml's `ggml_pad_reflect_1d`, PyTorch's `reflect`); `pad_reflect_1d_backward` folds the mirrored gradients
///   back.
/// - `roll(extents, shifts, x, y)`: `y[i] = x[(i - shift) mod n]` per dimension (ggml's `ggml_roll`, `torch.roll`);
///   its adjoint is the roll by `-shift`.
/// - `window_partition(h, w, c, window, x, windows)` cuts `[h][w][c]` into `ceil(h / window) · ceil(w / window)`
///   windows `[window][window][c]`, zero-padded at the bottom and right (SAM, ggml's `ggml_win_part`);
///   `window_unpartition` (`ggml_win_unpart`) crops them back and is the partition's adjoint, and the other way round.
/// - SAM's decomposed relative positions: `rel_pos_gather(rel, size, out)` gives `out[q][k] = rel[q - k + size - 1]`
///   (ggml's `ggml_get_rel_pos`, query and key sizes equal); `rel_pos_scores(q, rh, rw, h, w, c, rel_h, rel_w)` the
///   einsums `rel_h[qh][qw][kh] = q · Rh[qh][kh]` and `rel_w[qh][qw][kw] = q · Rw[qw][kw]`; `add_rel_pos(attn, rel_h,
///   rel_w, h, w)` adds them to the `(h w) x (h w)` scores (ggml's `ggml_add_rel_pos`), and `add_rel_pos_backward`
///   sums the scores' gradient back into them.
///
/// \snippet ml/spatial_demo.cpp spatial
///
/// **Relationships.** `ml/conv.hpp` (the convolutions these surround), `ml/patch.hpp`, `ml/row.hpp`. Group: \ref core.
///
/// **Implementations.** One each.
///
/// **Guarantees.** `constexpr`, no allocation, no exceptions; outputs are written, never accumulated, except
/// `add_rel_pos` (in place on the scores) and the documented gradients. Linear in the output for every operation but
/// the relative-position einsums (`h w (h + w) c`). `pool_shape` is a structural aggregate.
///
/// **Errors.** Preconditions, asserted: spans of the sizes the arguments imply; a valid pool shape (kernel within the
/// padded input, positive strides); reflection shorter than the row; a window and sizes above 0.
///
/// **Performance.** Not benchmarked: data movement, or a few multiply-adds per output.
///
/// **Tests and demos.** `tests/ml/spatial_test.cpp`, `demos/ml/spatial_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<cassert>`, `<concepts>`, `<cstddef>`, `<cstdint>`, `<limits>`, `<span>`,
/// `snicholls/core/math/constexpr_math.hpp`, `ml/conv.hpp` (its shape helpers), `ml/row.hpp`.
///
/// **Known issues.** ggml's average pool counts the padding in 2-D and not in 1-D: the defaults of `pool_1d_shape` and
/// `pool_2d_shape` follow it, so a 2-D average near a padded border is smaller than PyTorch's with
/// `count_include_pad = False`. ggml's antialiased bilinear mode is not built in. ggml's maximum starts from `-FLT_MAX`, so a window of -infinities gives `-FLT_MAX` there and -infinity here. ggml's `add_rel_pos` adds the two
/// terms in an order that depends on the position; here `(attn + rel_h) + rel_w`, SAM's, so a score may differ from
/// ggml's in its last bit.
///
/// **Status.** Experimental, new in 0.17.0.

#ifndef snicholls_core_ml_spatial_hpp
#define snicholls_core_ml_spatial_hpp

#include <snicholls/core/math/constexpr_math.hpp>
#include <snicholls/core/ml/conv.hpp>
#include <snicholls/core/ml/row.hpp>

#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>

namespace snicholls::ml
{

/// \brief Max or average pooling.
enum class pool_kind
{
    max,   ///< The largest tap inside the input.
    avg,   ///< The mean of the taps (see `pool_shape::count_padding`).
};

/// \brief The shape of a pooling of rank `Rank` over `[channels][input...]`.
/// \ingroup core
/// \tparam Rank The spatial dimensions: 1, 2 or 3.
template <std::size_t Rank>
    requires(Rank >= 1 && Rank <= 3)
struct pool_shape
{
    std::size_t channels = 0;                                          ///< Channels, pooled independently.
    std::array<std::size_t, Rank> input{};                             ///< The input's extent per dimension.
    std::array<std::size_t, Rank> kernel{};                            ///< The window's extent per dimension.
    std::array<std::size_t, Rank> stride = detail::conv::ones<Rank>(); ///< The stride per dimension.
    std::array<std::size_t, Rank> padding{};                           ///< Padding on both sides per dimension.
    bool count_padding = false;                                        ///< The average divides by every tap (true) or only those inside the input.

    /// \brief Whether the shape is consistent: channels set, positive strides, the kernel within the padded input.
    /// \return Whether it is valid.
    [[nodiscard]] constexpr bool valid() const noexcept
    {
        if (channels == 0)
            return false;
        for (std::size_t d = 0; d < Rank; ++d)
            if (kernel[d] == 0 || stride[d] == 0 || input[d] + 2 * padding[d] < kernel[d])
                return false;
        return true;
    }
    /// \brief The output's extent per dimension, `(in + 2p - k) / s + 1`.
    /// \return The extents.
    [[nodiscard]] constexpr std::array<std::size_t, Rank> output() const noexcept
    {
        std::array<std::size_t, Rank> o{};
        for (std::size_t d = 0; d < Rank; ++d)
            o[d] = (input[d] + 2 * padding[d] - kernel[d]) / stride[d] + 1;
        return o;
    }
    /// \brief The elements of the input.
    /// \return `channels · Π input`.
    [[nodiscard]] constexpr std::size_t x_size() const noexcept { return channels * detail::conv::product(input); }
    /// \brief The elements of the output.
    /// \return `channels · Π output()`.
    [[nodiscard]] constexpr std::size_t y_size() const noexcept { return channels * detail::conv::product(output()); }

    /// \brief Shapes compare by every field.
    /// \return Whether they are equal.
    friend constexpr bool operator==(const pool_shape&, const pool_shape&) = default;
};

/// \brief A 1-D pooling as ggml's `ggml_pool_1d`: the average over the taps inside the input.
/// \param channels The rows.
/// \param width The input's width.
/// \param kernel The window.
/// \param stride The stride.
/// \param padding The padding on both sides.
/// \return The shape.
[[nodiscard]] constexpr pool_shape<1> pool_1d_shape(std::size_t channels, std::size_t width, std::size_t kernel, std::size_t stride,
                                                    std::size_t padding = 0) noexcept
{
    return {channels, {width}, {kernel}, {stride}, {padding}, false};
}

/// \brief A 2-D pooling as ggml's `ggml_pool_2d`: the average over every tap, the padding included.
/// \param channels The planes.
/// \param height The input's height.
/// \param width The input's width.
/// \param kernel_h The window's height.
/// \param kernel_w The window's width.
/// \param stride_h The vertical stride.
/// \param stride_w The horizontal stride.
/// \param padding_h The vertical padding on both sides.
/// \param padding_w The horizontal padding on both sides.
/// \return The shape.
[[nodiscard]] constexpr pool_shape<2> pool_2d_shape(std::size_t channels, std::size_t height, std::size_t width, std::size_t kernel_h,
                                                    std::size_t kernel_w, std::size_t stride_h, std::size_t stride_w,
                                                    std::size_t padding_h = 0, std::size_t padding_w = 0) noexcept
{
    return {channels, {height, width}, {kernel_h, kernel_w}, {stride_h, stride_w}, {padding_h, padding_w}, true};
}

/// \brief How `upscale` resamples.
enum class upscale_mode
{
    nearest,    ///< The source pixel `floor(i / scale)`.
    bilinear,   ///< The four neighbours, weighted linearly; borders clamped.
    bicubic,    ///< The sixteen neighbours, Keys' cubic with `a = -0.75` (PyTorch's); borders clamped.
};

namespace detail::spatial
{
    // The input position tap k of output o reads, or `none` in the padding
    template <std::size_t Rank>
    constexpr std::size_t source(const pool_shape<Rank>& s, const std::array<std::size_t, Rank>& o, const std::array<std::size_t, Rank>& k,
                                 std::size_t none) noexcept
    {
        std::size_t at = 0;
        for (std::size_t d = 0; d < Rank; ++d) {
            const std::size_t padded = o[d] * s.stride[d] + k[d];
            if (padded < s.padding[d] || padded - s.padding[d] >= s.input[d])
                return none;
            at = at * s.input[d] + (padded - s.padding[d]);
        }
        return at;
    }

    // One output of a pool: the value, the first maximal tap (max) and the divisor (avg)
    template <typename T>
    struct pooled
    {
        T value;
        std::size_t argmax;
        std::size_t count;
    };

    template <std::size_t Rank, typename T>
    constexpr pooled<T> pool_one(const pool_shape<Rank>& s, pool_kind kind, std::span<const T> plane, const std::array<std::size_t, Rank>& o)
    {
        const std::size_t is = conv::product(s.input), ks = conv::product(s.kernel);
        pooled<T> r{kind == pool_kind::max ? std::numeric_limits<T>::lowest() : T(0), is, 0};
        for (std::size_t t = 0; t < ks; ++t) {
            const std::size_t at = source(s, o, conv::unflatten<Rank>(t, s.kernel), is);
            if (at == is)
                continue;
            ++r.count;
            if (kind == pool_kind::avg)
                r.value += plane[at];
            else if (r.argmax == is || plane[at] > r.value) {
                r.value = plane[at];
                r.argmax = at;
            }
        }
        if (kind == pool_kind::avg)
            r.count = s.count_padding ? ks : r.count;
        return r;
    }

    constexpr std::size_t clamp_index(std::int64_t i, std::size_t n) noexcept
    {
        return i < 0 ? 0 : static_cast<std::size_t>(i) >= n ? n - 1 : static_cast<std::size_t>(i);
    }

    // Keys' cubic convolution kernel with a = -0.75, as ggml writes it
    template <typename T>
    constexpr std::array<T, 4> cubic_weights(T x) noexcept
    {
        const T a = T(-0.75);
        const auto w1 = [a](T v) { return ((a + T(2)) * v - (a + T(3))) * v * v + T(1); };
        const auto w2 = [a](T v) { return ((a * v - T(5) * a) * v + T(8) * a) * v - T(4) * a; };
        return {w2(x + T(1)), w1(x), w1(T(1) - x), w2(T(2) - x)};
    }

    // Visit every (output index, input index, weight) of an upscale of one plane
    template <typename T, typename F>
    constexpr void upscale_taps(upscale_mode mode, bool align_corners, std::size_t in_h, std::size_t in_w, std::size_t out_h, std::size_t out_w, F&& f)
    {
        T sf0 = T(out_w) / T(in_w), sf1 = T(out_h) / T(in_h), offset = T(0.5);
        if (align_corners) {
            offset = T(0);
            sf0 = out_w > 1 && in_w > 1 ? T(out_w - 1) / T(in_w - 1) : sf0;
            sf1 = out_h > 1 && in_h > 1 ? T(out_h - 1) / T(in_h - 1) : sf1;
        }
        for (std::size_t i1 = 0; i1 < out_h; ++i1)
            for (std::size_t i0 = 0; i0 < out_w; ++i0) {
                const std::size_t o = i1 * out_w + i0;
                if (mode == upscale_mode::nearest) {
                    const auto y0 = static_cast<std::size_t>(T(i1) / sf1), x0 = static_cast<std::size_t>(T(i0) / sf0);
                    f(o, (y0 < in_h ? y0 : in_h - 1) * in_w + (x0 < in_w ? x0 : in_w - 1), T(1));
                    continue;
                }
                const T y = (T(i1) + offset) / sf1 - offset, x = (T(i0) + offset) / sf0 - offset;
                const auto fy = static_cast<std::int64_t>(math::floor(y)), fx = static_cast<std::int64_t>(math::floor(x));
                if (mode == upscale_mode::bilinear) {   // the corners clamped, the fractions from the clamped corner
                    const std::size_t y0 = clamp_index(fy, in_h), y1 = clamp_index(fy + 1, in_h);
                    const std::size_t x0 = clamp_index(fx, in_w), x1 = clamp_index(fx + 1, in_w);
                    T dy = y - T(y0), dx = x - T(x0);
                    dy = dy < T(0) ? T(0) : dy > T(1) ? T(1) : dy;
                    dx = dx < T(0) ? T(0) : dx > T(1) ? T(1) : dx;
                    f(o, y0 * in_w + x0, (T(1) - dx) * (T(1) - dy));
                    f(o, y0 * in_w + x1, dx * (T(1) - dy));
                    f(o, y1 * in_w + x0, (T(1) - dx) * dy);
                    f(o, y1 * in_w + x1, dx * dy);
                    continue;
                }
                const auto wy = cubic_weights(y - T(fy)), wx = cubic_weights(x - T(fx));   // bicubic
                for (std::int64_t j = 0; j < 4; ++j)
                    for (std::int64_t i = 0; i < 4; ++i)
                        f(o, clamp_index(fy + j - 1, in_h) * in_w + clamp_index(fx + i - 1, in_w),
                          wy[static_cast<std::size_t>(j)] * wx[static_cast<std::size_t>(i)]);
            }
    }

    // One plane of upscale, each output by ggml's own expression (ggml_compute_forward_upscale_f32)
    template <typename T>
    constexpr void upscale_plane(upscale_mode mode, bool align_corners, std::span<const T> src, std::size_t in_h, std::size_t in_w,
                                 std::span<T> dst, std::size_t out_h, std::size_t out_w)
    {
        T sf0 = T(out_w) / T(in_w), sf1 = T(out_h) / T(in_h), offset = T(0.5);
        if (align_corners) {
            offset = T(0);
            sf0 = out_w > 1 && in_w > 1 ? T(out_w - 1) / T(in_w - 1) : sf0;
            sf1 = out_h > 1 && in_h > 1 ? T(out_h - 1) / T(in_h - 1) : sf1;
        }
        const auto at = [&](std::int64_t y, std::int64_t x) { return src[clamp_index(y, in_h) * in_w + clamp_index(x, in_w)]; };
        for (std::size_t i1 = 0; i1 < out_h; ++i1)
            for (std::size_t i0 = 0; i0 < out_w; ++i0) {
                T& out = dst[i1 * out_w + i0];
                if (mode == upscale_mode::nearest) {
                    out = at(static_cast<std::int64_t>(T(i1) / sf1), static_cast<std::int64_t>(T(i0) / sf0));
                    continue;
                }
                const T y = (T(i1) + offset) / sf1 - offset, x = (T(i0) + offset) / sf0 - offset;
                const auto fy = static_cast<std::int64_t>(math::floor(y)), fx = static_cast<std::int64_t>(math::floor(x));
                if (mode == upscale_mode::bilinear) {
                    const std::size_t y0 = clamp_index(fy, in_h), x0 = clamp_index(fx, in_w);
                    T dy = y - T(y0), dx = x - T(x0);
                    dy = dy < T(0) ? T(0) : dy > T(1) ? T(1) : dy;
                    dx = dx < T(0) ? T(0) : dx > T(1) ? T(1) : dx;
                    const T a = at(fy, fx), b = at(fy, fx + 1), c = at(fy + 1, fx), d = at(fy + 1, fx + 1);
                    const T ta = a * (T(1) - dx) * (T(1) - dy), tb = b * dx * (T(1) - dy), tc = c * (T(1) - dx) * dy, td = d * dx * dy;
                    out = ta + tb + tc + td;
                    continue;
                }
                const auto wx = cubic_weights(x - T(fx)), wy = cubic_weights(y - T(fy));
                std::array<T, 4> rows{};
                for (std::int64_t j = 0; j < 4; ++j) {
                    T r = T(0);
                    for (std::int64_t i = 0; i < 4; ++i) {
                        const T term = at(fy + j - 1, fx + i - 1) * wx[static_cast<std::size_t>(i)];
                        r = i == 0 ? term : r + term;
                    }
                    rows[static_cast<std::size_t>(j)] = r;
                }
                T v = T(0);
                for (std::size_t j = 0; j < 4; ++j) {
                    const T term = rows[j] * wy[j];
                    v = j == 0 ? term : v + term;
                }
                out = v;
            }
    }

    template <std::size_t Rank>
    constexpr std::size_t flat(const std::array<std::size_t, Rank>& c, const std::array<std::size_t, Rank>& e) noexcept
    {
        std::size_t at = 0;
        for (std::size_t d = 0; d < Rank; ++d)
            at = at * e[d] + c[d];
        return at;
    }
} // namespace detail::spatial

/// \brief Max or average pooling of each channel.
/// \tparam Rank The spatial dimensions.
/// \param s The shape.
/// \param kind Max or average.
/// \param x The input, `s.x_size()`.
/// \param y Receives `s.y_size()`.
template <std::size_t Rank, row X, mutable_row Y>
    requires std::same_as<row_value_t<X>, row_value_t<Y>>
constexpr void pool_forward(const pool_shape<Rank>& s, pool_kind kind, const X& x, Y&& y)
{
    using T = row_value_t<X>;
    const auto in = as_row(x);
    const auto out = as_mutable_row(y);
    assert(s.valid() && in.size() == s.x_size() && out.size() == s.y_size());
    const std::size_t is = detail::conv::product(s.input), os = detail::conv::product(s.output());
    for (std::size_t c = 0; c < s.channels; ++c)
        for (std::size_t p = 0; p < os; ++p) {
            const auto r = detail::spatial::pool_one(s, kind, in.subspan(c * is, is), detail::conv::unflatten<Rank>(p, s.output()));
            out[c * os + p] = kind == pool_kind::max ? r.value : r.count == 0 ? T(0) : r.value / T(r.count);
        }
}

/// \brief The backward pass of `pool_forward`: `dx` written; the maximum's gradient to its first maximal tap, the
/// average's spread over its taps, divided by the same count.
/// \tparam Rank The spatial dimensions.
/// \param s The shape.
/// \param kind Max or average.
/// \param x The forward's input, `s.x_size()` (read for the maximum).
/// \param dy The gradient in the output, `s.y_size()`.
/// \param dx Receives the gradient in the input, `s.x_size()`.
template <std::size_t Rank, row X, row DY, mutable_row DX>
    requires std::same_as<row_value_t<X>, row_value_t<DY>> && std::same_as<row_value_t<X>, row_value_t<DX>>
constexpr void pool_backward(const pool_shape<Rank>& s, pool_kind kind, const X& x, const DY& dy, DX&& dx)
{
    using T = row_value_t<X>;
    const auto in = as_row(x);
    const auto g = as_row(dy);
    const auto gx = as_mutable_row(dx);
    assert(s.valid() && in.size() == s.x_size() && g.size() == s.y_size() && gx.size() == s.x_size());
    const std::size_t is = detail::conv::product(s.input), os = detail::conv::product(s.output()), ks = detail::conv::product(s.kernel);
    for (T& v : gx)
        v = T(0);
    for (std::size_t c = 0; c < s.channels; ++c)
        for (std::size_t p = 0; p < os; ++p) {
            const auto o = detail::conv::unflatten<Rank>(p, s.output());
            const auto r = detail::spatial::pool_one(s, kind, in.subspan(c * is, is), o);
            if (kind == pool_kind::max) {
                if (r.argmax != is)
                    gx[c * is + r.argmax] += g[c * os + p];
                continue;
            }
            if (r.count == 0)
                continue;
            const T share = g[c * os + p] / T(r.count);
            for (std::size_t t = 0; t < ks; ++t) {
                const std::size_t at = detail::spatial::source(s, o, detail::conv::unflatten<Rank>(t, s.kernel), is);
                if (at != is)
                    gx[c * is + at] += share;
            }
        }
}

/// \brief Resizes each `[in_h][in_w]` plane of `x` to `[out_h][out_w]`, as ggml's `ggml_interpolate`.
/// \param mode Nearest, bilinear or bicubic.
/// \param align_corners Pixel offset 0 and scale `(out - 1) / (in - 1)` (ggml's `GGML_SCALE_FLAG_ALIGN_CORNERS`), else offset 0.5 and scale `out / in`.
/// \param x The planes, `planes · in_h · in_w`.
/// \param in_h The input's height, above 0.
/// \param in_w The input's width, above 0.
/// \param y Receives `planes · out_h · out_w`.
/// \param out_h The output's height.
/// \param out_w The output's width.
template <row X, mutable_row Y>
    requires std::same_as<row_value_t<X>, row_value_t<Y>>
constexpr void upscale(upscale_mode mode, bool align_corners, const X& x, std::size_t in_h, std::size_t in_w, Y&& y, std::size_t out_h,
                       std::size_t out_w)
{
    const auto in = as_row(x);
    const auto out = as_mutable_row(y);
    assert(in_h > 0 && in_w > 0 && in.size() % (in_h * in_w) == 0);
    const std::size_t planes = in.size() / (in_h * in_w), ps = out_h * out_w;
    assert(out.size() == planes * ps);
    for (std::size_t c = 0; c < planes; ++c)
        detail::spatial::upscale_plane(mode, align_corners, in.subspan(c * in_h * in_w, in_h * in_w), in_h, in_w,
                                       out.subspan(c * ps, ps), out_h, out_w);
}

/// \brief The adjoint of `upscale`: each output's gradient added back to the source pixels with the same weights.
/// \param mode The forward's mode.
/// \param align_corners The forward's alignment.
/// \param dy The gradient in the output, `planes · out_h · out_w`.
/// \param out_h The output's height.
/// \param out_w The output's width.
/// \param dx Receives the gradient in the input, `planes · in_h · in_w`.
/// \param in_h The input's height, above 0.
/// \param in_w The input's width, above 0.
template <row DY, mutable_row DX>
    requires std::same_as<row_value_t<DY>, row_value_t<DX>>
constexpr void upscale_backward(upscale_mode mode, bool align_corners, const DY& dy, std::size_t out_h, std::size_t out_w, DX&& dx,
                                std::size_t in_h, std::size_t in_w)
{
    using T = row_value_t<DY>;
    const auto g = as_row(dy);
    const auto gx = as_mutable_row(dx);
    assert(in_h > 0 && in_w > 0 && gx.size() % (in_h * in_w) == 0);
    const std::size_t planes = gx.size() / (in_h * in_w), ps = out_h * out_w;
    assert(g.size() == planes * ps);
    for (T& v : gx)
        v = T(0);
    for (std::size_t c = 0; c < planes; ++c) {
        const auto src = g.subspan(c * ps, ps);
        const auto dst = gx.subspan(c * in_h * in_w, in_h * in_w);
        detail::spatial::upscale_taps<T>(mode, align_corners, in_h, in_w, out_h, out_w,
                                         [&](std::size_t o, std::size_t i, T w) {
                                             const T term = src[o] * w;
                                             dst[i] += term;
                                         });
    }
}

/// \brief Zero padding: `left[d]` zeros before and `right[d]` after each dimension (ggml's `ggml_pad_ext`).
/// \tparam Rank The dimensions, the channels included if they are padded too.
/// \param extents The input's extents.
/// \param left The zeros before, per dimension.
/// \param right The zeros after, per dimension.
/// \param x The input, `Π extents`.
/// \param y Receives `Π (extents + left + right)`.
template <std::size_t Rank, row X, mutable_row Y>
    requires std::same_as<row_value_t<X>, row_value_t<Y>>
constexpr void pad(const std::array<std::size_t, Rank>& extents, const std::array<std::size_t, Rank>& left,
                   const std::array<std::size_t, Rank>& right, const X& x, Y&& y)
{
    using T = row_value_t<X>;
    const auto in = as_row(x);
    const auto out = as_mutable_row(y);
    std::array<std::size_t, Rank> padded{};
    for (std::size_t d = 0; d < Rank; ++d)
        padded[d] = extents[d] + left[d] + right[d];
    assert(in.size() == detail::conv::product(extents) && out.size() == detail::conv::product(padded));
    for (std::size_t i = 0; i < out.size(); ++i) {
        auto c = detail::conv::unflatten<Rank>(i, padded);
        bool inside = true;
        for (std::size_t d = 0; d < Rank; ++d) {
            inside = inside && c[d] >= left[d] && c[d] - left[d] < extents[d];
            c[d] -= inside ? left[d] : 0;
        }
        out[i] = inside ? in[detail::spatial::flat(c, extents)] : T(0);
    }
}

/// \brief The adjoint of `pad`: the interior of the padded gradient, cropped.
/// \tparam Rank The dimensions.
/// \param extents The unpadded extents.
/// \param left The zeros before, per dimension.
/// \param right The zeros after, per dimension.
/// \param dy The gradient in the padded output.
/// \param dx Receives the gradient in the input, `Π extents`.
template <std::size_t Rank, row DY, mutable_row DX>
    requires std::same_as<row_value_t<DY>, row_value_t<DX>>
constexpr void pad_backward(const std::array<std::size_t, Rank>& extents, const std::array<std::size_t, Rank>& left,
                            const std::array<std::size_t, Rank>& right, const DY& dy, DX&& dx)
{
    const auto g = as_row(dy);
    const auto gx = as_mutable_row(dx);
    std::array<std::size_t, Rank> padded{};
    for (std::size_t d = 0; d < Rank; ++d)
        padded[d] = extents[d] + left[d] + right[d];
    assert(gx.size() == detail::conv::product(extents) && g.size() == detail::conv::product(padded));
    for (std::size_t i = 0; i < gx.size(); ++i) {
        auto c = detail::conv::unflatten<Rank>(i, extents);
        for (std::size_t d = 0; d < Rank; ++d)
            c[d] += left[d];
        gx[i] = g[detail::spatial::flat(c, padded)];
    }
}

/// \brief Reflect padding of each row: `y = [x[left] ... x[1], x, x[w-2] ... x[w-1-right]]`, the edge not repeated
/// (ggml's `ggml_pad_reflect_1d`).
/// \param rows The rows.
/// \param width Each row's width.
/// \param left The mirrored elements before, below `width`.
/// \param right The mirrored elements after, below `width`.
/// \param x The input, `rows · width`.
/// \param y Receives `rows · (width + left + right)`.
template <row X, mutable_row Y>
    requires std::same_as<row_value_t<X>, row_value_t<Y>>
constexpr void pad_reflect_1d(std::size_t rows, std::size_t width, std::size_t left, std::size_t right, const X& x, Y&& y)
{
    const auto in = as_row(x);
    const auto out = as_mutable_row(y);
    const std::size_t w = width + left + right;
    assert(left < width && right < width && in.size() == rows * width && out.size() == rows * w);
    for (std::size_t r = 0; r < rows; ++r)
        for (std::size_t i = 0; i < w; ++i) {
            const std::size_t j = i < left ? left - i : i - left < width ? i - left : 2 * (width - 1) - (i - left);
            out[r * w + i] = in[r * width + j];
        }
}

/// \brief The adjoint of `pad_reflect_1d`: each padded gradient added back to the element it mirrored, in output order.
/// \param rows The rows.
/// \param width Each row's unpadded width.
/// \param left The mirrored elements before.
/// \param right The mirrored elements after.
/// \param dy The gradient in the output, `rows · (width + left + right)`.
/// \param dx Receives the gradient in the input, `rows · width`.
template <row DY, mutable_row DX>
    requires std::same_as<row_value_t<DY>, row_value_t<DX>>
constexpr void pad_reflect_1d_backward(std::size_t rows, std::size_t width, std::size_t left, std::size_t right, const DY& dy, DX&& dx)
{
    using T = row_value_t<DY>;
    const auto g = as_row(dy);
    const auto gx = as_mutable_row(dx);
    const std::size_t w = width + left + right;
    assert(left < width && right < width && gx.size() == rows * width && g.size() == rows * w);
    for (T& v : gx)
        v = T(0);
    for (std::size_t r = 0; r < rows; ++r)
        for (std::size_t i = 0; i < w; ++i) {
            const std::size_t j = i < left ? left - i : i - left < width ? i - left : 2 * (width - 1) - (i - left);
            gx[r * width + j] += g[r * w + i];
        }
}

/// \brief Rolls each dimension by its shift with wraparound: `y[i] = x[(i - shift) mod n]` (ggml's `ggml_roll`); the
/// adjoint is the roll by `-shift`.
/// \tparam Rank The dimensions.
/// \param extents The extents.
/// \param shifts The shift per dimension, any sign and size.
/// \param x The input, `Π extents`.
/// \param y Receives `Π extents`; not `x`.
template <std::size_t Rank, row X, mutable_row Y>
    requires std::same_as<row_value_t<X>, row_value_t<Y>>
constexpr void roll(const std::array<std::size_t, Rank>& extents, const std::array<std::ptrdiff_t, Rank>& shifts, const X& x, Y&& y)
{
    const auto in = as_row(x);
    const auto out = as_mutable_row(y);
    assert(in.size() == detail::conv::product(extents) && out.size() == in.size());
    for (std::size_t i = 0; i < out.size(); ++i) {
        auto c = detail::conv::unflatten<Rank>(i, extents);
        for (std::size_t d = 0; d < Rank; ++d) {
            const auto n = static_cast<std::ptrdiff_t>(extents[d]);
            const std::ptrdiff_t s = ((shifts[d] % n) + n) % n;
            c[d] = static_cast<std::size_t>((static_cast<std::ptrdiff_t>(c[d]) - s + n) % n);
        }
        out[i] = in[detail::spatial::flat(c, extents)];
    }
}

/// \brief The windows `window_partition` cuts a `height x width` grid into: `ceil(h / window) · ceil(w / window)`.
/// \param height The grid's height.
/// \param width The grid's width.
/// \param window The window's side, above 0.
/// \return The windows.
[[nodiscard]] constexpr std::size_t window_count(std::size_t height, std::size_t width, std::size_t window) noexcept
{
    return ((height + window - 1) / window) * ((width + window - 1) / window);
}

/// \brief SAM's window partition: `[h][w][c]` into `window_count` windows `[window][window][c]`, row of windows by
/// row, zero beyond the grid (ggml's `ggml_win_part`).
/// \param height The grid's height.
/// \param width The grid's width.
/// \param channels The channels per position.
/// \param window The window's side, above 0.
/// \param x The grid, `height · width · channels`.
/// \param windows Receives `window_count · window² · channels`.
template <row X, mutable_row W>
    requires std::same_as<row_value_t<X>, row_value_t<W>>
constexpr void window_partition(std::size_t height, std::size_t width, std::size_t channels, std::size_t window, const X& x, W&& windows)
{
    using T = row_value_t<X>;
    const auto in = as_row(x);
    const auto out = as_mutable_row(windows);
    assert(window > 0 && in.size() == height * width * channels && out.size() == window_count(height, width, window) * window * window * channels);
    const std::size_t across = (width + window - 1) / window;
    for (std::size_t i = 0; i < out.size(); ++i) {
        const std::size_t ch = i % channels, wx = (i / channels) % window, wy = (i / (channels * window)) % window,
                          win = i / (channels * window * window);
        const std::size_t gy = (win / across) * window + wy, gx = (win % across) * window + wx;
        out[i] = gy < height && gx < width ? in[(gy * width + gx) * channels + ch] : T(0);
    }
}

/// \brief SAM's window unpartition, the adjoint of `window_partition`: the windows back into `[h][w][c]`, the padding
/// dropped (ggml's `ggml_win_unpart`).
/// \param height The grid's height.
/// \param width The grid's width.
/// \param channels The channels per position.
/// \param window The window's side, above 0.
/// \param windows The windows, `window_count · window² · channels`.
/// \param x Receives the grid, `height · width · channels`.
template <row W, mutable_row X>
    requires std::same_as<row_value_t<X>, row_value_t<W>>
constexpr void window_unpartition([[maybe_unused]] std::size_t height, std::size_t width, std::size_t channels, std::size_t window, const W& windows, X&& x)
{
    const auto in = as_row(windows);
    const auto out = as_mutable_row(x);
    assert(window > 0 && out.size() == height * width * channels && in.size() == window_count(height, width, window) * window * window * channels);
    const std::size_t across = (width + window - 1) / window;
    for (std::size_t i = 0; i < out.size(); ++i) {
        const std::size_t ch = i % channels, gx = (i / channels) % width, gy = i / (channels * width);
        const std::size_t win = (gy / window) * across + gx / window;
        out[i] = in[((win * window + gy % window) * window + gx % window) * channels + ch];
    }
}

/// \brief SAM's `get_rel_pos` for equal query and key sizes: `out[q][k][c] = rel[q - k + size - 1][c]` (ggml's `ggml_get_rel_pos`).
/// \param rel The table, `(2 size - 1) · channels`.
/// \param size The positions along the axis.
/// \param out Receives `size · size · channels`.
template <row R, mutable_row Out>
    requires std::same_as<row_value_t<R>, row_value_t<Out>>
constexpr void rel_pos_gather(const R& rel, std::size_t size, Out&& out)
{
    const auto r = as_row(rel);
    const auto o = as_mutable_row(out);
    assert(size > 0 && r.size() % (2 * size - 1) == 0);
    const std::size_t c = r.size() / (2 * size - 1);
    assert(o.size() == size * size * c);
    for (std::size_t q = 0; q < size; ++q)
        for (std::size_t k = 0; k < size; ++k)
            for (std::size_t i = 0; i < c; ++i)
                o[(q * size + k) * c + i] = r[(q + size - 1 - k) * c + i];
}

/// \brief SAM's decomposed relative-position terms: `rel_h[qh][qw][kh] = Σ_c q[qh][qw][c] Rh[qh][kh][c]` and
/// `rel_w[qh][qw][kw] = Σ_c q[qh][qw][c] Rw[qw][kw][c]`.
/// \param q One head's queries, `h · w · c`.
/// \param rh `rel_pos_gather` of the height table, `h · h · c`.
/// \param rw `rel_pos_gather` of the width table, `w · w · c`.
/// \param h The window's height.
/// \param w The window's width.
/// \param c The head's channels.
/// \param rel_h Receives `h · w · h`.
/// \param rel_w Receives `h · w · w`.
template <row Q, row RH, row RW, mutable_row OH, mutable_row OW>
    requires std::same_as<row_value_t<Q>, row_value_t<RH>> && std::same_as<row_value_t<Q>, row_value_t<RW>>
             && std::same_as<row_value_t<Q>, row_value_t<OH>> && std::same_as<row_value_t<Q>, row_value_t<OW>>
constexpr void rel_pos_scores(const Q& q, const RH& rh, const RW& rw, std::size_t h, std::size_t w, std::size_t c, OH&& rel_h, OW&& rel_w)
{
    using T = row_value_t<Q>;
    const auto qs = as_row(q);
    const auto a = as_row(rh);
    const auto b = as_row(rw);
    const auto oh = as_mutable_row(rel_h);
    const auto ow = as_mutable_row(rel_w);
    assert(qs.size() == h * w * c && a.size() == h * h * c && b.size() == w * w * c && oh.size() == h * w * h && ow.size() == h * w * w);
    for (std::size_t y = 0; y < h; ++y)
        for (std::size_t x = 0; x < w; ++x) {
            const auto qv = qs.subspan((y * w + x) * c, c);
            for (std::size_t k = 0; k < h; ++k) {
                T s = T(0);
                for (std::size_t i = 0; i < c; ++i)
                    s += qv[i] * a[(y * h + k) * c + i];
                oh[(y * w + x) * h + k] = s;
            }
            for (std::size_t k = 0; k < w; ++k) {
                T s = T(0);
                for (std::size_t i = 0; i < c; ++i)
                    s += qv[i] * b[(x * w + k) * c + i];
                ow[(y * w + x) * w + k] = s;
            }
        }
}

/// \brief Adds SAM's decomposed relative positions to one head's scores, in place:
/// `attn[(qh, qw)][(kh, kw)] = (attn + rel_h[qh][qw][kh]) + rel_w[qh][qw][kw]` (ggml's `ggml_add_rel_pos`).
/// \param attn The scores, `(h w) · (h w)`, updated.
/// \param rel_h `h · w · h`.
/// \param rel_w `h · w · w`.
/// \param h The window's height.
/// \param w The window's width.
template <mutable_row A, row RH, row RW>
    requires std::same_as<row_value_t<A>, row_value_t<RH>> && std::same_as<row_value_t<A>, row_value_t<RW>>
constexpr void add_rel_pos(A&& attn, const RH& rel_h, const RW& rel_w, std::size_t h, std::size_t w)
{
    const auto s = as_mutable_row(attn);
    const auto a = as_row(rel_h);
    const auto b = as_row(rel_w);
    const std::size_t n = h * w;
    assert(s.size() == n * n && a.size() == n * h && b.size() == n * w);
    for (std::size_t q = 0; q < n; ++q)
        for (std::size_t k = 0; k < n; ++k) {
            const auto t = s[q * n + k] + a[q * h + k / w];
            s[q * n + k] = t + b[q * w + k % w];
        }
}

/// \brief The backward pass of `add_rel_pos` in its terms: `drel_h[q][kh] += Σ_kw dattn[q][(kh, kw)]` and
/// `drel_w[q][kw] += Σ_kh dattn[q][(kh, kw)]` (the scores' own gradient passes through unchanged).
/// \param dattn The gradient in the scores, `(h w) · (h w)`.
/// \param h The window's height.
/// \param w The window's width.
/// \param drel_h Accumulates `h · w · h`.
/// \param drel_w Accumulates `h · w · w`.
template <row DA, mutable_row DH, mutable_row DW>
    requires std::same_as<row_value_t<DA>, row_value_t<DH>> && std::same_as<row_value_t<DA>, row_value_t<DW>>
constexpr void add_rel_pos_backward(const DA& dattn, std::size_t h, std::size_t w, DH&& drel_h, DW&& drel_w)
{
    const auto g = as_row(dattn);
    const auto a = as_mutable_row(drel_h);
    const auto b = as_mutable_row(drel_w);
    const std::size_t n = h * w;
    assert(g.size() == n * n && a.size() == n * h && b.size() == n * w);
    for (std::size_t q = 0; q < n; ++q)
        for (std::size_t k = 0; k < n; ++k) {
            a[q * h + k / w] += g[q * n + k];
            b[q * w + k % w] += g[q * n + k];
        }
}

} // namespace snicholls::ml

#endif // snicholls_core_ml_spatial_hpp
