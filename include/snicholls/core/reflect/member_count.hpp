//
//  member_count.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::member_count_v<S>`: the member count of a plain aggregate struct.
/// \ingroup core
///
/// Member count of a plain aggregate struct, detected at compile time (no N at call sites).
///
/// Technique (as Boost.PFR): find the largest N for which S{x0, ..., xN-1} is valid, where each
/// x converts to any member type. Binary search, so cost is logarithmic in SNICHOLLS_MAX_STRUCT_MEMBERS.
///
/// Supported members: values of any type, including move-only (unique_ptr), const, std::optional,
/// std::any, std::variant, nested aggregates and std::array.
/// Not supported: reference members (rejected with a clear error), C-array members (miscounted via
/// brace elision - use std::array), bit-fields (cannot be bound by reference), and aggregates with base
/// classes or unions.

#ifndef snicholls_core_reflect_member_count_hpp
#define snicholls_core_reflect_member_count_hpp

#include <cstddef>
#include <type_traits>
#include <utility>

#ifndef SNICHOLLS_MAX_STRUCT_MEMBERS
/// \brief The largest member count `member_count` detects (default 256). Define it before the first include to lower it;
/// `detail/struct_binders.hpp` is generated for up to 256 and rejects a larger value.
#define SNICHOLLS_MAX_STRUCT_MEMBERS 256
#endif

namespace snicholls {

namespace detail {

// Converts to any member type T (by T&&, so move-only members count correctly). Only ever used in
// unevaluated requires-expressions, so it is declared but deliberately never defined (and not
// constexpr/inline, which would demand a definition).
struct any_member
{
    std::size_t index;

    template <typename T>
    operator T&&() const noexcept;
};

template <typename S, std::size_t... I>
constexpr bool is_initialisable_with(std::index_sequence<I...>) noexcept
{
    return requires { S{any_member{I}...}; };
}

template <typename S, std::size_t Lo, std::size_t Hi>
constexpr std::size_t search_member_count() noexcept
{
    if constexpr (Lo == Hi) {
        return Lo;
    } else {
        constexpr std::size_t mid = (Lo + Hi + 1) / 2;
        if constexpr (is_initialisable_with<S>(std::make_index_sequence<mid>{}))
            return search_member_count<S, mid, Hi>();
        else
            return search_member_count<S, Lo, mid - 1>();
    }
}

} // namespace detail

/// \brief A plain aggregate struct this library can bind: class type, aggregate, not a union (any cv/ref).
/// \ingroup core
/// \tparam S the type.
template <typename S>
concept plain_aggregate = std::is_class_v<std::remove_cvref_t<S>> && std::is_aggregate_v<std::remove_cvref_t<S>> &&
                          !std::is_union_v<std::remove_cvref_t<S>>;

/// \brief The member count of `S` as a `std::integral_constant`; `member_count_v<S>` is its value.
/// \ingroup core
/// \tparam S a plain aggregate struct (any cv/ref).
template <plain_aggregate S>
struct member_count
    : std::integral_constant<std::size_t,
                             detail::search_member_count<std::remove_cvref_t<S>, 0, SNICHOLLS_MAX_STRUCT_MEMBERS + 1>()>
{
    static_assert(member_count::value <= SNICHOLLS_MAX_STRUCT_MEMBERS,
                  "snicholls::member_count: struct has more than SNICHOLLS_MAX_STRUCT_MEMBERS members");
    static_assert(member_count::value > 0 || std::is_empty_v<std::remove_cvref_t<S>>,
                  "snicholls::member_count: cannot count members - reference members are not supported");
};

/// \brief The member count of a plain aggregate struct, detected at compile time.
/// \ingroup core
///
/// \overview
/// So call sites never pass N (`snicholls::to_tuple(s)`, not `to_tuple<11>(s)`). The technique (as Boost.PFR: the largest N for which `S{x0, ..., xN-1}` is valid, by binary search) is in the file description above.
///
/// \relations
/// - Used by `snicholls::tie`, `snicholls::to_tuple` and `snicholls::from_tuple` (`reflect/struct_tuple.hpp`).
/// - Module: \ref core.
///
/// \impls
/// None: a single implementation.
///
/// \contracts
/// - Compile time only; cost logarithmic in `SNICHOLLS_MAX_STRUCT_MEMBERS` (256).
///
/// \errors
/// Reference members are a clear compile error.
///
/// \examples
/// \snippet member_count_demo.cpp count
///
/// \perf
/// Not benchmarked.
///
/// \testing
/// - Tests: tests/reflect/member_count_test.cpp; compile-fail cases (reference member, over the limit) in
///   tests/compile_fail/member_count_*.cpp.
/// - Demo: demos/reflect/member_count_demo.cpp.
///
/// \deps
/// CMake target `snicholls::core`; `<cstddef>`, `<type_traits>`, `<utility>`.
///
/// \gotchas
/// - C arrays are miscounted (use `std::array`); bit-fields, base classes and unions are not supported.
///
/// \status
/// Stable. Rules: CLAUDE.md, "Struct / tuple conversion rules".
///
/// \tparam S a plain aggregate struct.
template <plain_aggregate S>
inline constexpr std::size_t member_count_v = member_count<S>::value;

} // namespace snicholls

#endif /* snicholls_core_reflect_member_count_hpp */
