//
//  row.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::ml::row` and `mutable_row`: what the `ml` functions accept as a row of numbers, and the spans they work on.
/// \ingroup core
///
/// **Overview.** A row is any contiguous, sized range of a `math::math_number`: `std::vector<float>`,
/// `std::array<double, N>`, `std::span<const float>`, a C array, a row of an `md_span`. The `ml` functions take
/// rows as they are, so a caller never writes `std::span<const float>(v)` to call them; inside, each works on the
/// `std::span` that `as_row` or `as_mutable_row` gives, which costs nothing.
///
/// \snippet ml/row_demo.cpp row
///
/// **Relationships.** Used by `ml/softmax.hpp`, `ml/norm.hpp`, `ml/attention.hpp`, `ml/optim.hpp`,
/// `ml/sampling.hpp`, `ml/embedding.hpp`. Group: \ref core.
///
/// **Implementations.** One.
///
/// **Guarantees.** `constexpr`, `noexcept`, no copy: a span over the caller's elements.
///
/// **Errors.** None.
///
/// **Performance.** Not benchmarked: a pointer and a size.
///
/// **Tests and demos.** `tests/ml/row_test.cpp`, `demos/ml/row_demo.cpp`.
///
/// **Dependencies.** `<ranges>`, `<span>`, `<type_traits>`, `snicholls/core/math/constexpr_math.hpp`.
///
/// **Known issues.** None.
///
/// **Status.** Experimental, new in 0.6.0.

#ifndef snicholls_core_ml_row_hpp
#define snicholls_core_ml_row_hpp

#include <snicholls/core/math/constexpr_math.hpp>

#include <ranges>
#include <span>
#include <type_traits>

namespace snicholls::ml
{

/// \brief A row of numbers: a contiguous, sized range whose elements are a `math::math_number`.
/// \ingroup core
template <typename R>
concept row = std::ranges::contiguous_range<R> && std::ranges::sized_range<R> && math::math_number<std::ranges::range_value_t<R>>;

/// \brief A row whose elements can be written.
/// \ingroup core
template <typename R>
concept mutable_row = row<R> && !std::is_const_v<std::remove_reference_t<std::ranges::range_reference_t<R>>>;

/// \brief The number type of a row.
/// \ingroup core
template <row R>
using row_value_t = std::ranges::range_value_t<R>;

/// \brief A row as a span of its elements, read-only.
/// \tparam R A `row`.
/// \param r The row.
/// \return A span over `r`'s elements.
template <row R>
constexpr std::span<const row_value_t<R>> as_row(const R& r) noexcept
{
    return {std::ranges::data(r), std::ranges::size(r)};
}

/// \brief A row as a span of its elements, writable.
/// \tparam R A `mutable_row`.
/// \param r The row.
/// \return A span over `r`'s elements.
template <mutable_row R>
constexpr std::span<row_value_t<R>> as_mutable_row(R&& r) noexcept
{
    return {std::ranges::data(r), std::ranges::size(r)};
}

} // namespace snicholls::ml

#endif // snicholls_core_ml_row_hpp
