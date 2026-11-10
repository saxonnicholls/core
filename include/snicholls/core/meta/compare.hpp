//
//  compare.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `synth_three_way`: compare two values with `<=>` if they have it, otherwise with `<`, as `std::tuple` does.
/// \ingroup core
///
/// **Overview.** A generic `operator<=>` over members (a tuple-like, a reflected aggregate, a key of
/// several parts) must compare members that may have `<=>` or only `<`. The standard specifies how,
/// for `std::tuple` and `std::pair`, as *synth-three-way*: the member's own `<=>` when it has one,
/// otherwise a `std::weak_ordering` built from two `<`. It never names the function; this does.
///
/// \snippet meta/compare_demo.cpp compare
/// \snippet meta/compare_demo.cpp compare_use
///
/// **Relationships.** `meta/concepts.hpp`; objects with fields can compare through it. Group: \ref core.
///
/// **Guarantees.** `constexpr`; `noexcept` exactly when the comparison it makes is. The result type is
/// the member's own ordering, or `std::weak_ordering` when synthesised.
///
/// **Errors.** None: a type with neither `<=>` nor `<` does not satisfy `synth_three_way_comparable`.
///
/// **Performance.** The comparison it makes; not benchmarked.
///
/// **Tests and demos.** `tests/meta/compare_test.cpp`, `demos/meta/compare_demo.cpp`.
///
/// **Dependencies.** `<compare>`, `<concepts>`, `<utility>`.
///
/// **Known issues.** None known.
///
/// **Status.** Stable.

#ifndef snicholls_core_meta_compare_hpp
#define snicholls_core_meta_compare_hpp

#include <compare>
#include <concepts>
#include <utility>

namespace snicholls {

    /// \brief The standard's exposition-only *boolean-testable*: converts to `bool`, and so does its negation.
    /// \ingroup core
    template <typename B>
    concept boolean_testable = std::convertible_to<B, bool> && requires(B&& b) {
        { !std::forward<B>(b) } -> std::convertible_to<bool>;
    };

    /// \brief Comparable by synth-three-way: `a < b` and `b < a` are boolean-testable.
    /// \ingroup core
    template <typename T>
    concept synth_three_way_comparable = requires(const T& a, const T& b) {
        { a < b } -> boolean_testable;
        { b < a } -> boolean_testable;
    };

    /// \brief Synth-three-way for a type with a usable `<=>`: its own result.
    /// \ingroup core
    /// \tparam T The type.
    /// \param a One value.
    /// \param b Another.
    /// \return `a <=> b`.
    template <typename T>
        requires synth_three_way_comparable<T> && std::three_way_comparable<T>
    [[nodiscard]] constexpr auto synth_three_way(const T& a, const T& b) noexcept(noexcept(a <=> b))
    {
        return a <=> b;
    }

    /// \brief Synth-three-way for a type with only `<`: a weak ordering from at most two comparisons.
    /// \ingroup core
    /// \tparam T The type.
    /// \param a One value.
    /// \param b Another.
    /// \return `less`, `greater` or `equivalent`.
    template <typename T>
        requires synth_three_way_comparable<T> && (!std::three_way_comparable<T>)
    [[nodiscard]] constexpr std::weak_ordering synth_three_way(const T& a, const T& b) noexcept(noexcept(static_cast<bool>(a < b)))
    {
        if (static_cast<bool>(a < b))
            return std::weak_ordering::less;
        if (static_cast<bool>(b < a))
            return std::weak_ordering::greater;
        return std::weak_ordering::equivalent;
    }

    /// \brief The ordering `synth_three_way` gives for `T`.
    /// \ingroup core
    template <synth_three_way_comparable T>
    using synth_three_way_result_t = decltype(synth_three_way(std::declval<const T&>(), std::declval<const T&>()));

} // namespace snicholls

#endif /* snicholls_core_meta_compare_hpp */
