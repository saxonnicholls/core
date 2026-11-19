//
//  conv.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::ml` convolutions of rank 1, 2 and 3: `im2col` / `col2im`, the convolution forward and backward (direct and through `im2col`), groups and depthwise, and the transposed convolution, all `constexpr` over the caller's spans.
/// \ingroup core
///
/// **Overview.** The convolutions of vision and audio encoders (a ViT's patch embedding, Whisper's two 1-D
/// convolutions, a U-Net's 2-D and transposed layers, a video model's 3-D stem), for one sample at a time. The rank is
/// a template parameter: `conv_shape<1>`, `<2>` and `<3>` are the same code.
///
/// - Layout, as PyTorch and ggml for one sample: the input `[in_channels][spatial...]`, the output
///   `[out_channels][output...]`, the weights `[out_channels][in_channels / groups][kernel...]`, every spatial index
///   row-major (depth, height, width).
/// - `conv_shape<Rank>`: channels, groups, and per dimension the input extent, kernel, stride, padding (both sides) and
///   dilation. Its derived sizes are `constexpr` (`output()`, `output_size()`, `column_size()`, ...), so a
///   `constexpr conv_shape` sizes `std::array` scratch at compile time. The output extent is ggml's and PyTorch's
///   `(in + 2p - d (k - 1) - 1) / s + 1`.
/// - `im2col(shape, x, cols)`: the columns `[output position][in_channels · kernel]`, channel-major then kernel, zero
///   where the window reaches into the padding: ggml's `ggml_im2col` layout. `col2im` is its adjoint (overlapping
///   entries summed).
/// - `conv_forward(shape, x, w, bias, y)`: the direct loop, no scratch. `conv_forward_im2col(shape, x, w, bias, y,
///   cols)`: `im2col` and the matrix product, the same sums in the same order, so the two agree to the bit.
/// - `conv_backward(shape, x, w, dy, dx, dw, db)`: `dx` written (`col2im` of `Wᵀ dy`), `dw` and `db` added (they sum over
///   the samples of a batch). An empty span skips that gradient.
/// - Groups: channel group `g` of `groups` convolves only with its own input channels. `depthwise_shape(...)` is a 2-D
///   shape with `groups == channels` (ggml's `ggml_conv_2d_dw`).
/// - `conv_transpose_shape<Rank>` and `conv_transpose_forward` / `conv_transpose_backward`: the transposed
///   ("deconvolution") layer, output `(in - 1) s - 2p + d (k - 1) + 1`, weights `[in_channels][out_channels /
///   groups][kernel...]` (PyTorch's `ConvTranspose` layout, ggml's `ggml_conv_transpose_1d` / `_2d_p0` with padding
///   0 and dilation 1). It is exactly the adjoint, in its input, of the convolution `as_conv()` returns, so its
///   forward is that convolution's data gradient and its backward that convolution's forward.
///
/// \snippet ml/conv_demo.cpp conv
///
/// **Relationships.** `ml/row.hpp` (the rows), `ml/spatial.hpp` (pooling, padding, upscaling), `ml/patch.hpp` (a
/// stride-equals-kernel convolution is a patch embedding). Group: \ref core.
///
/// **Implementations.** Two forwards: direct (`conv_forward`, no scratch) and lowered (`conv_forward_im2col`, the
/// columns in the caller's scratch, the form a GPU library replaces with its matrix product). They agree to the bit.
///
/// **Guarantees.** `constexpr`, no allocation, no exceptions. Each multiply and add is its own statement, summed per
/// output in the order (input channel, kernel position) from 0, the bias added last. Complexity:
/// `out_channels · output_size · (in_channels / groups) · kernel_size` multiply-adds for each pass. `conv_shape` is a
/// structural aggregate (usable as a template argument).
///
/// **Errors.** Preconditions, asserted: `shape.valid()` (channels divisible by the groups, positive strides and
/// dilations, a kernel that fits the padded input); every span the size the shape says, or empty for an optional
/// bias or gradient.
///
/// **Performance.** Benchmarked against the 7-deep loop written by hand, same order of summation, with a fairness check
/// to the bit (`benchmarks/ml/conv_benchmark.cpp`, `BENCHMARKS.md`): a 3x3 convolution, padding 1, of 64 channels of
/// 32x32 floats into 64 runs 1.08 billion multiply-adds a second by hand, 1.02 billion through `conv_forward_im2col`
/// and 0.89 billion through `conv_forward`, whose kernel extents are not known to the compiler (Xeon W-3245, -O3,
/// Apple clang 17, 2026-11-19).
///
/// **Tests and demos.** `tests/ml/conv_test.cpp`, `demos/ml/conv_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<cassert>`, `<concepts>`, `<cstddef>`, `<span>`, `ml/row.hpp`.
///
/// **Known issues.** ggml's transposed convolutions have no padding or dilation, and its CPU kernel sums over input
/// channels first (a vector dot product) then over positions: the same value, possibly not the same last bit. No
/// `output_padding` (PyTorch's extra row on one side of a transposed convolution): pad the output instead.
///
/// **Status.** Experimental, new in 0.17.0.

#ifndef snicholls_core_ml_conv_hpp
#define snicholls_core_ml_conv_hpp

#include <snicholls/core/ml/row.hpp>

#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <span>

namespace snicholls::ml
{

namespace detail::conv
{
    template <std::size_t Rank>
    constexpr std::array<std::size_t, Rank> ones() noexcept
    {
        std::array<std::size_t, Rank> a{};
        for (auto& v : a)
            v = 1;
        return a;
    }

    template <std::size_t Rank>
    constexpr std::size_t product(const std::array<std::size_t, Rank>& a) noexcept
    {
        std::size_t p = 1;
        for (const std::size_t v : a)
            p *= v;
        return p;
    }

    // The coordinates of flat index i of a row-major box of extents e
    template <std::size_t Rank>
    constexpr std::array<std::size_t, Rank> unflatten(std::size_t i, const std::array<std::size_t, Rank>& e) noexcept
    {
        std::array<std::size_t, Rank> c{};
        for (std::size_t d = Rank; d-- > 0;) {
            c[d] = i % e[d];
            i /= e[d];
        }
        return c;
    }
} // namespace detail::conv

/// \brief The shape of a convolution of rank `Rank` over one sample.
/// \ingroup core
/// \tparam Rank The spatial dimensions: 1, 2 or 3.
template <std::size_t Rank>
    requires(Rank >= 1 && Rank <= 3)
struct conv_shape
{
    std::size_t in_channels = 0;                                         ///< Input channels.
    std::size_t out_channels = 0;                                        ///< Output channels.
    std::array<std::size_t, Rank> input{};                               ///< The input's extent per dimension.
    std::array<std::size_t, Rank> kernel{};                              ///< The kernel's extent per dimension.
    std::array<std::size_t, Rank> stride = detail::conv::ones<Rank>();   ///< The stride per dimension.
    std::array<std::size_t, Rank> padding{};                             ///< Zeros on both sides per dimension.
    std::array<std::size_t, Rank> dilation = detail::conv::ones<Rank>(); ///< The spacing of the kernel's taps.
    std::size_t groups = 1;                                              ///< Channel groups: 1 dense, `in_channels` depthwise.

    /// \brief Whether the shape is consistent: channels set and divisible by `groups`, strides and dilations positive,
    /// and the dilated kernel within the padded input.
    /// \return Whether it is valid.
    [[nodiscard]] constexpr bool valid() const noexcept
    {
        if (in_channels == 0 || out_channels == 0 || groups == 0 || in_channels % groups != 0 || out_channels % groups != 0)
            return false;
        for (std::size_t d = 0; d < Rank; ++d)
            if (kernel[d] == 0 || stride[d] == 0 || dilation[d] == 0 || input[d] + 2 * padding[d] < dilation[d] * (kernel[d] - 1) + 1)
                return false;
        return true;
    }
    /// \brief The output's extent per dimension, `(in + 2p - d (k - 1) - 1) / s + 1`.
    /// \return The extents.
    [[nodiscard]] constexpr std::array<std::size_t, Rank> output() const noexcept
    {
        std::array<std::size_t, Rank> o{};
        for (std::size_t d = 0; d < Rank; ++d)
            o[d] = (input[d] + 2 * padding[d] - dilation[d] * (kernel[d] - 1) - 1) / stride[d] + 1;
        return o;
    }
    /// \brief The input positions of one channel.
    /// \return The product of `input`.
    [[nodiscard]] constexpr std::size_t input_size() const noexcept { return detail::conv::product(input); }
    /// \brief The output positions of one channel.
    /// \return The product of `output()`.
    [[nodiscard]] constexpr std::size_t output_size() const noexcept { return detail::conv::product(output()); }
    /// \brief The taps of the kernel.
    /// \return The product of `kernel`.
    [[nodiscard]] constexpr std::size_t kernel_size() const noexcept { return detail::conv::product(kernel); }
    /// \brief The input channels each output channel reads, `in_channels / groups`.
    /// \return The channels.
    [[nodiscard]] constexpr std::size_t group_in_channels() const noexcept { return in_channels / groups; }
    /// \brief The output channels of each group, `out_channels / groups`.
    /// \return The channels.
    [[nodiscard]] constexpr std::size_t group_out_channels() const noexcept { return out_channels / groups; }
    /// \brief The elements of the input, `in_channels · input_size()`.
    /// \return The size.
    [[nodiscard]] constexpr std::size_t x_size() const noexcept { return in_channels * input_size(); }
    /// \brief The elements of the output, `out_channels · output_size()`.
    /// \return The size.
    [[nodiscard]] constexpr std::size_t y_size() const noexcept { return out_channels * output_size(); }
    /// \brief The elements of the weights, `out_channels · (in_channels / groups) · kernel_size()`.
    /// \return The size.
    [[nodiscard]] constexpr std::size_t weight_size() const noexcept { return out_channels * group_in_channels() * kernel_size(); }
    /// \brief The elements of one row of `im2col`'s columns, `in_channels · kernel_size()`.
    /// \return The size.
    [[nodiscard]] constexpr std::size_t column_width() const noexcept { return in_channels * kernel_size(); }
    /// \brief The elements of `im2col`'s columns, `output_size() · column_width()`.
    /// \return The size.
    [[nodiscard]] constexpr std::size_t column_size() const noexcept { return output_size() * column_width(); }

    /// \brief Shapes compare by every field.
    /// \return Whether they are equal.
    friend constexpr bool operator==(const conv_shape&, const conv_shape&) = default;
};

/// \brief A 2-D depthwise convolution's shape: one filter per channel (`groups == channels`), ggml's `ggml_conv_2d_dw`.
/// \param channels The channels, in and out.
/// \param height The input's height.
/// \param width The input's width.
/// \param kernel_h The kernel's height.
/// \param kernel_w The kernel's width.
/// \param stride The stride in both dimensions.
/// \param padding The padding in both dimensions.
/// \param dilation The dilation in both dimensions.
/// \return The shape; its weights are `[channels][1][kernel_h][kernel_w]`.
[[nodiscard]] constexpr conv_shape<2> depthwise_shape(std::size_t channels, std::size_t height, std::size_t width, std::size_t kernel_h,
                                                      std::size_t kernel_w, std::size_t stride = 1, std::size_t padding = 0,
                                                      std::size_t dilation = 1) noexcept
{
    return {channels, channels, {height, width}, {kernel_h, kernel_w}, {stride, stride}, {padding, padding}, {dilation, dilation}, channels};
}

namespace detail::conv
{
    // Steps row-major coordinates c within extents e; false after the last
    template <std::size_t Rank>
    constexpr bool next(std::array<std::size_t, Rank>& c, const std::array<std::size_t, Rank>& e) noexcept
    {
        for (std::size_t d = Rank; d-- > 0;) {
            if (++c[d] < e[d])
                return true;
            c[d] = 0;
        }
        return false;
    }

    // The taps of dimension D on, inside the input: the range of each dimension's taps found once, no division
    template <std::size_t D, std::size_t Rank, typename F>
    constexpr void taps(const conv_shape<Rank>& s, const std::array<std::ptrdiff_t, Rank>& base, std::size_t t, std::size_t at, F& f)
    {
        if constexpr (D == Rank) {
            f(t, at);
        } else {
            const auto dil = static_cast<std::ptrdiff_t>(s.dilation[D]), n = static_cast<std::ptrdiff_t>(s.input[D]);
            std::size_t k = 0;
            std::ptrdiff_t pos = base[D];
            for (; k < s.kernel[D] && pos < 0; ++k)   // the taps in the leading padding
                pos += dil;
            for (; k < s.kernel[D] && pos < n; ++k, pos += dil)
                taps<D + 1>(s, base, t * s.kernel[D] + k, at * s.input[D] + static_cast<std::size_t>(pos), f);
        }
    }

    // Where output o's window starts in each dimension of the input (negative in the padding)
    template <std::size_t Rank>
    constexpr std::array<std::ptrdiff_t, Rank> window(const conv_shape<Rank>& s, const std::array<std::size_t, Rank>& o) noexcept
    {
        std::array<std::ptrdiff_t, Rank> base{};
        for (std::size_t d = 0; d < Rank; ++d)
            base[d] = static_cast<std::ptrdiff_t>(o[d] * s.stride[d]) - static_cast<std::ptrdiff_t>(s.padding[d]);
        return base;
    }

    // Calls f(tap, input position) for each tap of output o inside the input, taps in row-major order
    template <std::size_t Rank, typename F>
    constexpr void for_each_tap(const conv_shape<Rank>& s, const std::array<std::size_t, Rank>& o, F&& f)
    {
        taps<0>(s, window(s, o), 0, 0, f);
    }

    constexpr bool fits(std::size_t n, std::size_t expected) noexcept { return n == 0 || n == expected; }

    // y[oc][p] = sum over (ic of the group, tap) of w * x, from 0, in that order; then + bias
    template <std::size_t Rank, typename T>
    constexpr void forward(const conv_shape<Rank>& s, std::span<const T> x, std::span<const T> w, std::span<const T> bias, std::span<T> y)
    {
        const std::size_t cg = s.group_in_channels(), og = s.group_out_channels(), ks = s.kernel_size(), os = s.output_size(),
                          is = s.input_size();
        const auto out = s.output();
        for (std::size_t oc = 0; oc < s.out_channels; ++oc) {
            const std::size_t g = oc / og;
            std::array<std::size_t, Rank> o{};
            for (std::size_t p = 0; p < os; ++p, next(o, out)) {
                T sum = T(0);
                const T* wr = w.data() + oc * cg * ks;
                const T* xr = x.data() + g * cg * is;
                const auto base = window(s, o);
                for (std::size_t c = 0; c < cg; ++c, wr += ks, xr += is) {
                    auto f = [&](std::size_t t, std::size_t at) {
                        const T term = wr[t] * xr[at];
                        sum += term;
                    };
                    taps<0>(s, base, 0, 0, f);
                }
                y[oc * os + p] = bias.empty() ? sum : sum + bias[oc];
            }
        }
    }

    // dx = Wᵀ dy scattered back (written), in the order (oc, p, c, tap)
    template <std::size_t Rank, typename T>
    constexpr void data_gradient(const conv_shape<Rank>& s, std::span<const T> w, std::span<const T> dy, std::span<T> dx)
    {
        const std::size_t cg = s.group_in_channels(), og = s.group_out_channels(), ks = s.kernel_size(), os = s.output_size(),
                          is = s.input_size();
        const auto out = s.output();
        for (T& v : dx)
            v = T(0);
        for (std::size_t oc = 0; oc < s.out_channels; ++oc) {
            const std::size_t g = oc / og;
            std::array<std::size_t, Rank> o{};
            for (std::size_t p = 0; p < os; ++p, next(o, out)) {
                const T gy = dy[oc * os + p];
                for (std::size_t c = 0; c < cg; ++c) {
                    const T* wr = w.data() + (oc * cg + c) * ks;
                    T* xr = dx.data() + (g * cg + c) * is;
                    for_each_tap(s, o, [&](std::size_t t, std::size_t at) {
                        const T term = wr[t] * gy;
                        xr[at] += term;
                    });
                }
            }
        }
    }

    // The input position tap k of output o reads, or input_size() in the padding; no division
    template <std::size_t Rank>
    constexpr std::size_t source(const conv_shape<Rank>& s, const std::array<std::size_t, Rank>& o, const std::array<std::size_t, Rank>& k) noexcept
    {
        std::size_t at = 0;
        for (std::size_t d = 0; d < Rank; ++d) {
            const std::size_t padded = o[d] * s.stride[d] + k[d] * s.dilation[d];   // in the padded input
            if (padded < s.padding[d] || padded - s.padding[d] >= s.input[d])
                return s.input_size();
            at = at * s.input[d] + (padded - s.padding[d]);
        }
        return at;
    }

    // dw[oc][c][t] += Σ_p dy[oc][p] · x[ic][src(p, t)], the sum over p in order, then added once
    template <std::size_t Rank, typename T>
    constexpr void weight_gradient(const conv_shape<Rank>& s, std::span<const T> x, std::span<const T> dy, std::span<T> dw)
    {
        const std::size_t cg = s.group_in_channels(), og = s.group_out_channels(), ks = s.kernel_size(), os = s.output_size(),
                          is = s.input_size();
        const auto out = s.output();
        for (std::size_t oc = 0; oc < s.out_channels; ++oc) {
            const std::size_t g = oc / og;
            for (std::size_t c = 0; c < cg; ++c) {
                const T* xr = x.data() + (g * cg + c) * is;
                const T* gr = dy.data() + oc * os;
                std::array<std::size_t, Rank> k{};
                for (std::size_t t = 0; t < ks; ++t, next(k, s.kernel)) {
                    T sum = T(0);
                    std::array<std::size_t, Rank> o{};
                    for (std::size_t p = 0; p < os; ++p, next(o, out)) {
                        const std::size_t at = source(s, o, k);
                        if (at == is)
                            continue;
                        const T term = gr[p] * xr[at];
                        sum += term;
                    }
                    dw[(oc * cg + c) * ks + t] += sum;
                }
            }
        }
    }

    template <typename T>
    constexpr void bias_gradient(std::size_t channels, std::size_t positions, std::span<const T> dy, std::span<T> db)
    {
        for (std::size_t oc = 0; oc < channels; ++oc) {
            T sum = T(0);
            for (std::size_t p = 0; p < positions; ++p)
                sum += dy[oc * positions + p];
            db[oc] += sum;
        }
    }
} // namespace detail::conv

/// \brief `im2col`: the columns `[output position][in_channels · kernel_size]`, input channel then kernel tap, zero in
/// the padding (ggml's `ggml_im2col` layout).
/// \tparam Rank The spatial dimensions.
/// \param s The shape.
/// \param x The input, `s.x_size()`.
/// \param cols Receives `s.column_size()`.
template <std::size_t Rank, row X, mutable_row Cols>
    requires std::same_as<row_value_t<X>, row_value_t<Cols>>
constexpr void im2col(const conv_shape<Rank>& s, const X& x, Cols&& cols)
{
    using T = row_value_t<X>;
    const auto in = as_row(x);
    const auto c = as_mutable_row(cols);
    assert(s.valid() && in.size() == s.x_size() && c.size() == s.column_size());
    const std::size_t ks = s.kernel_size(), is = s.input_size(), width = s.column_width();
    const auto out = s.output();
    for (T& v : c)
        v = T(0);
    std::array<std::size_t, Rank> o{};
    for (std::size_t p = 0; p < s.output_size(); ++p, detail::conv::next(o, out))
        for (std::size_t ic = 0; ic < s.in_channels; ++ic) {
            T* dst = c.data() + p * width + ic * ks;
            const T* src = in.data() + ic * is;
            detail::conv::for_each_tap(s, o, [&](std::size_t t, std::size_t at) { dst[t] = src[at]; });
        }
}

/// \brief `col2im`, the adjoint of `im2col`: each column entry added back to the input position it was read from (the
/// padding's entries dropped), in column order.
/// \tparam Rank The spatial dimensions.
/// \param s The shape.
/// \param cols The columns, `s.column_size()`.
/// \param x Receives the input-shaped sums, `s.x_size()`: overwritten.
template <std::size_t Rank, row Cols, mutable_row X>
    requires std::same_as<row_value_t<X>, row_value_t<Cols>>
constexpr void col2im(const conv_shape<Rank>& s, const Cols& cols, X&& x)
{
    using T = row_value_t<X>;
    const auto c = as_row(cols);
    const auto out_x = as_mutable_row(x);
    assert(s.valid() && out_x.size() == s.x_size() && c.size() == s.column_size());
    const std::size_t ks = s.kernel_size(), is = s.input_size(), width = s.column_width();
    const auto out = s.output();
    for (T& v : out_x)
        v = T(0);
    std::array<std::size_t, Rank> o{};
    for (std::size_t p = 0; p < s.output_size(); ++p, detail::conv::next(o, out))
        for (std::size_t ic = 0; ic < s.in_channels; ++ic) {
            const T* src = c.data() + p * width + ic * ks;
            T* dst = out_x.data() + ic * is;
            detail::conv::for_each_tap(s, o, [&](std::size_t t, std::size_t at) { dst[at] += src[t]; });
        }
}

/// \brief The convolution, by the direct loop: `y[oc][p] = Σ_{ic in oc's group, tap} w · x + bias[oc]`.
/// \tparam Rank The spatial dimensions.
/// \param s The shape.
/// \param x The input, `s.x_size()`.
/// \param w The weights, `s.weight_size()`.
/// \param bias `out_channels`, or empty for none.
/// \param y Receives `s.y_size()`.
template <std::size_t Rank, row X, row W, row B, mutable_row Y>
    requires std::same_as<row_value_t<X>, row_value_t<W>> && std::same_as<row_value_t<X>, row_value_t<B>>
             && std::same_as<row_value_t<X>, row_value_t<Y>>
constexpr void conv_forward(const conv_shape<Rank>& s, const X& x, const W& w, const B& bias, Y&& y)
{
    const auto in = as_row(x);
    const auto ws = as_row(w);
    const auto b = as_row(bias);
    const auto out = as_mutable_row(y);
    assert(s.valid() && in.size() == s.x_size() && ws.size() == s.weight_size() && detail::conv::fits(b.size(), s.out_channels)
           && out.size() == s.y_size());
    detail::conv::forward(s, in, ws, b, out);
}

/// \brief The convolution through `im2col` and the matrix product `y_g = W_g · cols_gᵀ` per group: the same sums in
/// the same order as `conv_forward`, so the same bits.
/// \tparam Rank The spatial dimensions.
/// \param s The shape.
/// \param x The input, `s.x_size()`.
/// \param w The weights, `s.weight_size()`.
/// \param bias `out_channels`, or empty for none.
/// \param y Receives `s.y_size()`.
/// \param cols Scratch for the columns, `s.column_size()`: overwritten.
template <std::size_t Rank, row X, row W, row B, mutable_row Y, mutable_row Cols>
    requires std::same_as<row_value_t<X>, row_value_t<W>> && std::same_as<row_value_t<X>, row_value_t<B>>
             && std::same_as<row_value_t<X>, row_value_t<Y>> && std::same_as<row_value_t<X>, row_value_t<Cols>>
constexpr void conv_forward_im2col(const conv_shape<Rank>& s, const X& x, const W& w, const B& bias, Y&& y, Cols&& cols)
{
    using T = row_value_t<X>;
    const auto ws = as_row(w);
    const auto b = as_row(bias);
    const auto out = as_mutable_row(y);
    const auto c = as_mutable_row(cols);
    assert(ws.size() == s.weight_size() && detail::conv::fits(b.size(), s.out_channels) && out.size() == s.y_size());
    im2col(s, x, c);
    const std::size_t cg = s.group_in_channels(), og = s.group_out_channels(), ks = s.kernel_size(), os = s.output_size(),
                      width = s.column_width(), inner = cg * ks;
    for (std::size_t oc = 0; oc < s.out_channels; ++oc) {
        const std::size_t first = (oc / og) * inner;
        const auto wrow = ws.subspan(oc * inner, inner);
        for (std::size_t p = 0; p < os; ++p) {
            const auto crow = std::span<const T>(c).subspan(p * width + first, inner);
            T sum = T(0);
            for (std::size_t j = 0; j < inner; ++j) {
                const T term = wrow[j] * crow[j];
                sum += term;
            }
            out[oc * os + p] = b.empty() ? sum : sum + b[oc];
        }
    }
}

/// \brief The convolution's backward pass: `dx` written, `dw` and `db` added; an empty span skips that gradient.
/// \tparam Rank The spatial dimensions.
/// \param s The shape.
/// \param x The forward's input, `s.x_size()` (read for `dw`).
/// \param w The weights, `s.weight_size()` (read for `dx`).
/// \param dy The gradient in the output, `s.y_size()`.
/// \param dx Receives the gradient in the input, `s.x_size()`, or empty.
/// \param dw Accumulates the gradient in the weights, `s.weight_size()`, or empty.
/// \param db Accumulates the gradient in the bias, `out_channels`, or empty.
template <std::size_t Rank, row X, row W, row DY, mutable_row DX, mutable_row DW, mutable_row DB>
    requires std::same_as<row_value_t<X>, row_value_t<W>> && std::same_as<row_value_t<X>, row_value_t<DY>>
             && std::same_as<row_value_t<X>, row_value_t<DX>> && std::same_as<row_value_t<X>, row_value_t<DW>>
             && std::same_as<row_value_t<X>, row_value_t<DB>>
constexpr void conv_backward(const conv_shape<Rank>& s, const X& x, const W& w, const DY& dy, DX&& dx, DW&& dw, DB&& db)
{
    const auto in = as_row(x);
    const auto ws = as_row(w);
    const auto g = as_row(dy);
    const auto gx = as_mutable_row(dx);
    const auto gw = as_mutable_row(dw);
    const auto gb = as_mutable_row(db);
    assert(s.valid() && g.size() == s.y_size() && detail::conv::fits(gx.size(), s.x_size())
           && detail::conv::fits(gw.size(), s.weight_size()) && detail::conv::fits(gb.size(), s.out_channels));
    if (!gx.empty()) {
        assert(ws.size() == s.weight_size());
        detail::conv::data_gradient(s, ws, g, gx);
    }
    if (!gw.empty()) {
        assert(in.size() == s.x_size());
        detail::conv::weight_gradient(s, in, g, gw);
    }
    if (!gb.empty())
        detail::conv::bias_gradient(s.out_channels, s.output_size(), g, gb);
}

/// \brief The shape of a transposed convolution of rank `Rank`: output `(in - 1) s - 2p + d (k - 1) + 1`, weights
/// `[in_channels][out_channels / groups][kernel...]`.
/// \ingroup core
/// \tparam Rank The spatial dimensions: 1, 2 or 3.
template <std::size_t Rank>
    requires(Rank >= 1 && Rank <= 3)
struct conv_transpose_shape
{
    std::size_t in_channels = 0;                                         ///< Input channels.
    std::size_t out_channels = 0;                                        ///< Output channels.
    std::array<std::size_t, Rank> input{};                               ///< The input's extent per dimension.
    std::array<std::size_t, Rank> kernel{};                              ///< The kernel's extent per dimension.
    std::array<std::size_t, Rank> stride = detail::conv::ones<Rank>();   ///< The stride per dimension.
    std::array<std::size_t, Rank> padding{};                             ///< Rows removed from both sides of the output per dimension.
    std::array<std::size_t, Rank> dilation = detail::conv::ones<Rank>(); ///< The spacing of the kernel's taps.
    std::size_t groups = 1;                                              ///< Channel groups.

    /// \brief The output's extent per dimension, `(in - 1) s - 2p + d (k - 1) + 1`.
    /// \return The extents.
    [[nodiscard]] constexpr std::array<std::size_t, Rank> output() const noexcept
    {
        std::array<std::size_t, Rank> o{};
        for (std::size_t d = 0; d < Rank; ++d)
            o[d] = (input[d] - 1) * stride[d] + dilation[d] * (kernel[d] - 1) + 1 - 2 * padding[d];
        return o;
    }
    /// \brief The convolution this layer is the adjoint of: its input is this output, its output this input.
    /// \return The shape; the weights' layout is the same memory.
    [[nodiscard]] constexpr conv_shape<Rank> as_conv() const noexcept
    {
        return {out_channels, in_channels, output(), kernel, stride, padding, dilation, groups};
    }
    /// \brief Whether the shape is consistent: a positive input and `as_conv()` valid, its output this input.
    /// \return Whether it is valid.
    [[nodiscard]] constexpr bool valid() const noexcept
    {
        for (std::size_t d = 0; d < Rank; ++d)
            if (input[d] == 0 || kernel[d] == 0 || stride[d] == 0 || dilation[d] == 0
                || (input[d] - 1) * stride[d] + dilation[d] * (kernel[d] - 1) + 1 <= 2 * padding[d])
                return false;
        return as_conv().valid() && as_conv().output() == input;
    }
    /// \brief The elements of the input, `in_channels · Π input`.
    /// \return The size.
    [[nodiscard]] constexpr std::size_t x_size() const noexcept { return in_channels * detail::conv::product(input); }
    /// \brief The elements of the output, `out_channels · Π output()`.
    /// \return The size.
    [[nodiscard]] constexpr std::size_t y_size() const noexcept { return out_channels * detail::conv::product(output()); }
    /// \brief The elements of the weights, `in_channels · (out_channels / groups) · Π kernel`.
    /// \return The size.
    [[nodiscard]] constexpr std::size_t weight_size() const noexcept { return as_conv().weight_size(); }

    /// \brief Shapes compare by every field.
    /// \return Whether they are equal.
    friend constexpr bool operator==(const conv_transpose_shape&, const conv_transpose_shape&) = default;
};

/// \brief The transposed convolution: `y = Wᵀ x` scattered as `as_conv()`'s data gradient, plus the bias.
/// \tparam Rank The spatial dimensions.
/// \param s The shape.
/// \param x The input, `s.x_size()`.
/// \param w The weights, `s.weight_size()`, `[in_channels][out_channels / groups][kernel...]`.
/// \param bias `out_channels`, or empty for none.
/// \param y Receives `s.y_size()`.
template <std::size_t Rank, row X, row W, row B, mutable_row Y>
    requires std::same_as<row_value_t<X>, row_value_t<W>> && std::same_as<row_value_t<X>, row_value_t<B>>
             && std::same_as<row_value_t<X>, row_value_t<Y>>
constexpr void conv_transpose_forward(const conv_transpose_shape<Rank>& s, const X& x, const W& w, const B& bias, Y&& y)
{
    const auto in = as_row(x);
    const auto ws = as_row(w);
    const auto b = as_row(bias);
    const auto out = as_mutable_row(y);
    assert(s.valid() && in.size() == s.x_size() && ws.size() == s.weight_size() && detail::conv::fits(b.size(), s.out_channels)
           && out.size() == s.y_size());
    detail::conv::data_gradient(s.as_conv(), ws, in, out);
    if (!b.empty()) {
        const std::size_t positions = s.y_size() / s.out_channels;
        for (std::size_t c = 0; c < s.out_channels; ++c)
            for (std::size_t p = 0; p < positions; ++p)
                out[c * positions + p] += b[c];
    }
}

/// \brief The transposed convolution's backward pass: `dx` written (`as_conv()`'s forward of `dy`), `dw` and `db`
/// added; an empty span skips that gradient.
/// \tparam Rank The spatial dimensions.
/// \param s The shape.
/// \param x The forward's input, `s.x_size()` (read for `dw`).
/// \param w The weights, `s.weight_size()` (read for `dx`).
/// \param dy The gradient in the output, `s.y_size()`.
/// \param dx Receives the gradient in the input, `s.x_size()`, or empty.
/// \param dw Accumulates the gradient in the weights, or empty.
/// \param db Accumulates the gradient in the bias, `out_channels`, or empty.
template <std::size_t Rank, row X, row W, row DY, mutable_row DX, mutable_row DW, mutable_row DB>
    requires std::same_as<row_value_t<X>, row_value_t<W>> && std::same_as<row_value_t<X>, row_value_t<DY>>
             && std::same_as<row_value_t<X>, row_value_t<DX>> && std::same_as<row_value_t<X>, row_value_t<DW>>
             && std::same_as<row_value_t<X>, row_value_t<DB>>
constexpr void conv_transpose_backward(const conv_transpose_shape<Rank>& s, const X& x, const W& w, const DY& dy, DX&& dx, DW&& dw,
                                       DB&& db)
{
    using T = row_value_t<X>;
    const auto in = as_row(x);
    const auto ws = as_row(w);
    const auto g = as_row(dy);
    const auto gx = as_mutable_row(dx);
    const auto gw = as_mutable_row(dw);
    const auto gb = as_mutable_row(db);
    const auto c = s.as_conv();
    assert(s.valid() && g.size() == s.y_size() && detail::conv::fits(gx.size(), s.x_size())
           && detail::conv::fits(gw.size(), s.weight_size()) && detail::conv::fits(gb.size(), s.out_channels));
    if (!gx.empty()) {
        assert(ws.size() == s.weight_size());
        detail::conv::forward(c, g, ws, std::span<const T>{}, gx);
    }
    if (!gw.empty()) {
        assert(in.size() == s.x_size());
        detail::conv::weight_gradient(c, g, in, gw);   // the convolution's "input" is dy, its "output gradient" x
    }
    if (!gb.empty())
        detail::conv::bias_gradient(s.out_channels, s.y_size() / s.out_channels, g, gb);
}

} // namespace snicholls::ml

#endif // snicholls_core_ml_conv_hpp
