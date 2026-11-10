//
//  values.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief Facts about a pack of values at compile time: distinct, increasing, decreasing.
/// \ingroup core
///
/// **Overview.** Tables written by hand into a program (enumeration codes, tenor buckets, the
/// breakpoints of a schedule) carry rules: no code twice, buckets in order. These state the rule
/// where the table is, as a `static_assert`, for any values usable as template arguments.
///
/// \snippet meta/values_demo.cpp values
///
/// **Relationships.** `meta/concepts.hpp` (`distinct_static_names_v` for names); `constexpr_map.hpp`.
/// Group: \ref core.
///
/// **Guarantees.** Compile time only. An empty pack or a single value is distinct and monotone.
///
/// **Errors.** None.
///
/// **Performance.** Compile time only.
///
/// **Tests and demos.** `tests/meta/values_test.cpp`, `demos/meta/values_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<cstddef>`, `<type_traits>`.
///
/// **Known issues.** The values must share a type once `std::common_type` has had its say.
///
/// **Status.** New in 0.1.0.

#ifndef snicholls_core_meta_values_hpp
#define snicholls_core_meta_values_hpp

#include <array>
#include <cstddef>
#include <type_traits>

namespace snicholls {

    namespace detail {

        template <auto... Vs>
        consteval auto value_array() noexcept
        {
            using common = std::common_type_t<decltype(Vs)...>;
            return std::array<common, sizeof...(Vs)>{static_cast<common>(Vs)...};
        }

        template <typename Compare, auto... Vs>
        consteval bool adjacent_all(Compare compare) noexcept
        {
            if constexpr (sizeof...(Vs) < 2) {
                return true;
            } else {
                constexpr auto values = value_array<Vs...>();
                for (std::size_t i = 1; i < values.size(); ++i)
                    if (!compare(values[i - 1], values[i]))
                        return false;
                return true;
            }
        }

        template <auto... Vs>
        consteval bool distinct() noexcept
        {
            if constexpr (sizeof...(Vs) < 2) {
                return true;
            } else {
                constexpr auto values = value_array<Vs...>();
                for (std::size_t i = 0; i < values.size(); ++i)
                    for (std::size_t j = i + 1; j < values.size(); ++j)
                        if (values[i] == values[j])
                            return false;
                return true;
            }
        }

    } // namespace detail

    /// \brief No two of the values are equal.
    /// \ingroup core
    template <auto... Vs>
    inline constexpr bool all_distinct_v = detail::distinct<Vs...>();

    /// \brief Each value is less than the next.
    /// \ingroup core
    template <auto... Vs>
    inline constexpr bool strictly_increasing_v = detail::adjacent_all<decltype([](auto a, auto b) { return a < b; }), Vs...>({});

    /// \brief Each value is at most the next.
    /// \ingroup core
    template <auto... Vs>
    inline constexpr bool increasing_v = detail::adjacent_all<decltype([](auto a, auto b) { return !(b < a); }), Vs...>({});

    /// \brief Each value is greater than the next.
    /// \ingroup core
    template <auto... Vs>
    inline constexpr bool strictly_decreasing_v = detail::adjacent_all<decltype([](auto a, auto b) { return b < a; }), Vs...>({});

    /// \brief Each value is at least the next.
    /// \ingroup core
    template <auto... Vs>
    inline constexpr bool decreasing_v = detail::adjacent_all<decltype([](auto a, auto b) { return !(a < b); }), Vs...>({});

} // namespace snicholls

#endif /* snicholls_core_meta_values_hpp */
