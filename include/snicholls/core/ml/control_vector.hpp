//
//  control_vector.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::ml` control vectors: a direction per layer added to the residual stream after layers `il_start` to `il_end`, in llama.cpp's layout, with the sum of several at given strengths and the backward pass for training one.
/// \ingroup core
///
/// **Overview.** Representation engineering (Zou et al., 2023; "steering vectors") shifts a model's behaviour by adding a
/// fixed direction to the residual stream between blocks. llama.cpp applies one after each block's output
/// (`build_cvec` after `ffn_out`, `src/models/llama.cpp`), for the layers from `il_start` to `il_end` inclusive
/// (`llama_adapter_cvec::tensor_for` and `apply_to`, `src/llama-adapter.cpp`, llama.cpp 39039aa). Its buffer holds
/// `n_embd` values for each layer from 1 on (layer 0 has none: "never present"), and `common_control_vector_load` sums
/// several files, each scaled by its strength.
///
/// - `control_vector_view<T>{data, width, layer_start, layer_end}`: llama.cpp's buffer and range; `direction(il)`
///   gives layer `il`'s row, or an empty span outside the range or the buffer.
/// - `apply_control_vector(cv, il, rows)`: `x += direction(il)` for every row of a batch, after block `il`.
/// - `add_control_vector(sum, data, strength)`: `sum += strength · data`, to combine vectors before use.
/// - `control_vector_backward(cv, il, drows, ddirection)`: the gradient passes unchanged to the block's output, and
///   the direction's gradient is the sum of the rows' (training a steering vector).
///
/// \snippet ml/control_vector_demo.cpp control_vector
///
/// **Relationships.** Applied between `block_forward` calls (`ml/transformer_block.hpp`); `ml/lora.hpp` is the other
/// adapter llama.cpp loads. Group: \ref core.
///
/// **Implementations.** One.
///
/// **Guarantees.** `constexpr`, no allocation, linear in the rows. A disabled vector (`layer_start` -1, as
/// `llama_adapter_cvec::apply` with no data) changes nothing.
///
/// **Errors.** Preconditions, asserted: `width > 0`; the rows a multiple of `width`; `sum` and `data` the same size.
///
/// **Performance.** Not benchmarked: one addition per value.
///
/// **Tests and demos.** `tests/ml/control_vector_test.cpp`, `demos/ml/control_vector_demo.cpp`.
///
/// **Dependencies.** `<cassert>`, `<cstddef>`, `<cstdint>`, `<span>`, `snicholls/core/math/constexpr_math.hpp`.
///
/// **Known issues.** llama.cpp's loader also accepts vectors with fewer layers than the model (the rest are zero);
/// `direction` returns an empty span past the buffer, the same effect.
///
/// **Status.** Experimental, new in 0.17.0.

#ifndef snicholls_core_ml_control_vector_hpp
#define snicholls_core_ml_control_vector_hpp

#include <snicholls/core/math/constexpr_math.hpp>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <span>

namespace snicholls::ml
{

/// \brief A control vector over the caller's buffer: `width` values per layer from layer 1 on, applied to the layers
/// from `layer_start` to `layer_end` inclusive.
/// \ingroup core
/// \tparam T The number type.
template <math::math_number T>
struct control_vector_view
{
    std::span<const T> data{};        ///< Layer `il`'s direction at `width · (il - 1)`, for `il >= 1`.
    std::size_t width = 0;            ///< The residual stream's width (`n_embd`).
    std::int32_t layer_start = -1;    ///< The first layer it applies to; -1 disables it.
    std::int32_t layer_end = -1;      ///< The last layer it applies to.

    /// \brief Layer `il`'s direction.
    /// \param il The layer.
    /// \return `width` values, or an empty span if the layer is outside the range or the buffer (or is layer 0).
    [[nodiscard]] constexpr std::span<const T> direction(std::int32_t il) const noexcept
    {
        if (il < 1 || layer_start < 0 || il < layer_start || il > layer_end)
            return {};
        const std::size_t at = width * std::size_t(il - 1);
        return at + width <= data.size() ? data.subspan(at, width) : std::span<const T>{};
    }
};

/// \brief Adds layer `il`'s direction to every row of a batch, after block `il`: `x += direction(il)`.
/// \tparam T A `math::math_number`.
/// \param cv The control vector.
/// \param il The layer just computed.
/// \param rows The residual rows, `width` each: updated.
template <math::math_number T>
constexpr void apply_control_vector(const control_vector_view<T>& cv, std::int32_t il, std::span<T> rows)
{
    assert(cv.width > 0 && rows.size() % cv.width == 0);
    const auto d = cv.direction(il);
    if (d.empty())
        return;
    for (std::size_t r = 0; r < rows.size(); r += cv.width)
        for (std::size_t c = 0; c < cv.width; ++c)
            rows[r + c] += d[c];
}

/// \brief Adds one control vector into a sum at a strength: `sum += strength · data` (`common_control_vector_load`).
/// \tparam T A `math::math_number`.
/// \param sum The combined buffer: updated.
/// \param data Another vector's buffer, `sum`'s size.
/// \param strength Its scale.
template <math::math_number T>
constexpr void add_control_vector(std::span<T> sum, std::type_identity_t<std::span<const T>> data, std::type_identity_t<T> strength)
{
    assert(sum.size() == data.size());
    for (std::size_t i = 0; i < sum.size(); ++i)
        sum[i] += strength * data[i];
}

/// \brief The backward pass of `apply_control_vector`: the rows' gradients pass through unchanged, and layer `il`'s
/// direction gains their sum.
/// \tparam T A `math::math_number`.
/// \param cv The control vector of the forward pass.
/// \param il The layer.
/// \param drows The gradients in the residual rows after the addition.
/// \param ddirection Added into: the gradient in layer `il`'s direction, `width` (unchanged if the layer is outside the
/// range).
template <math::math_number T>
constexpr void control_vector_backward(const control_vector_view<T>& cv, std::int32_t il, std::type_identity_t<std::span<const T>> drows,
                                       std::span<T> ddirection)
{
    assert(cv.width > 0 && drows.size() % cv.width == 0 && ddirection.size() == cv.width);
    if (cv.direction(il).empty())
        return;
    for (std::size_t r = 0; r < drows.size(); r += cv.width)
        for (std::size_t c = 0; c < cv.width; ++c)
            ddirection[c] += drows[r + c];
}

} // namespace snicholls::ml

#endif // snicholls_core_ml_control_vector_hpp
