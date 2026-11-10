//
//  struct_tuple.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  The struct code is unchanged but for one static_assert in from_tuple that makes narrowing an error on GCC
//  as well (GCC only warns about it inside a template); the guard, namespace and documentation are core's.
//

/// \file
/// \brief Conversions between plain aggregate structs and `std::tuple`s.
/// \ingroup core
///
/// Conversions between plain aggregate structs and std::tuples. Everything is
/// constexpr; tie() is unconditionally noexcept, every other conversion is noexcept exactly when the
/// member copies/moves/assignments it performs are (e.g. not for std::string copies, which may throw).
///
/// \code
///   auto refs = snicholls::tie(s);              // std::tuple of references (lvalues only)
///   auto vals = snicholls::to_tuple(s);         // std::tuple of values; moves from rvalues
///   S s       = snicholls::from_tuple<S>(tup);  // tuple-like -> struct
/// \endcode
///
/// Structs (plain aggregates, member count detected automatically, see reflect/member_count.hpp):
/// one element per member; nested structs stay as they are.
///
/// Strict input: sizes must match exactly, and arithmetic conversions must not narrow - both are
/// compile-time errors with a snicholls:: message.
///
/// The rules, as CLAUDE.md "Struct / tuple conversion rules" states them:
/// - Everything is `constexpr`. `tie` is always `noexcept`; every other conversion is `noexcept` **exactly when**
///   the copies/moves/assignments it performs are (a `std::string` copy may throw `bad_alloc`, so copying one is
///   not `noexcept`; moving it is). Never declare a conversion unconditionally `noexcept` when it can allocate -
///   that turns `bad_alloc` into `std::terminate`.
/// - Input is strict: sizes must match exactly and arithmetic conversions must not narrow (brace-init rules, so
///   `int -> double` and `int -> bool` are rejected). Errors are compile-time `static_assert`s with a `snicholls::`
///   message.
/// - Plain structs: members of any type incl. move-only, const, `std::optional`/`any`/`variant`, nested aggregates,
///   `std::array`. Not supported: reference members (clear error), C arrays (miscounted - use `std::array`),
///   bit-fields, base classes, unions. Up to `SNICHOLLS_MAX_STRUCT_MEMBERS` (256) members.

#ifndef snicholls_core_reflect_struct_tuple_hpp
#define snicholls_core_reflect_struct_tuple_hpp

#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#include <snicholls/core/detail/struct_binders.hpp>
#include <snicholls/core/reflect/member_count.hpp>

namespace snicholls {

/// \brief Anything with `std::tuple_size`: `std::tuple`, `std::pair`, `std::array`.
///
/// Anything with std::tuple_size: std::tuple, std::pair, std::array.
/// \ingroup core
/// \tparam T the type (any cv/ref).
template <typename T>
concept tuple_like = requires { std::tuple_size<std::remove_cvref_t<T>>::value; };

/// \brief A plain aggregate struct (`snicholls::plain_aggregate`).
///
/// Core knows no object model, so this is exactly `plain_aggregate`.
/// \ingroup core
/// \tparam S the type.
template <typename S>
concept plain_struct = plain_aggregate<S>;

namespace detail {

// get<I> for any tuple-like: std::get for std types, the type's own get (via ADL) otherwise.
template <std::size_t I, typename Tup>
constexpr decltype(auto) element(Tup&& tup) noexcept
{
    using std::get;
    return get<I>(static_cast<Tup&&>(tup));
}

// Leaves of a tie are references; nested tuples of references stay tuples.
template <typename T>
struct value_of
{
    using type = std::remove_cvref_t<T>;
};

template <typename... Ts>
struct value_of<std::tuple<Ts...>>
{
    using type = std::tuple<typename value_of<Ts>::type...>;
};

template <typename T>
using value_of_t = typename value_of<T>::type;

// --- struct -> tie --------------------------------------------------------------

template <typename S>
constexpr auto tie_struct(S& s) noexcept
{
    return binder<member_count_v<S>>::tie(s);
}

// tie for lvalues, tuple of rvalue references for rvalues (moves members out).
template <typename S>
constexpr auto forward_tie_struct(S&& s) noexcept
{
    if constexpr (std::is_lvalue_reference_v<S>) {
        return tie_struct(s);
    } else {
        return std::apply([](auto&... m) noexcept { return std::forward_as_tuple(std::move(m)...); }, tie_struct(s));
    }
}

template <typename S>
using struct_tie_t = decltype(forward_tie_struct(std::declval<S>()));

// --- tuple -> struct ----------------------------------------------------------------

template <typename S, typename Tup, std::size_t... I>
constexpr S struct_from_tuple(Tup&& tup, std::index_sequence<I...>) noexcept(
    noexcept(S{element<I>(static_cast<Tup&&>(tup))...}))
{
    return S{element<I>(static_cast<Tup&&>(tup))...};
}

// Whether S{get<I>(tup)...} is well-formed: every element converts to its member without narrowing. Asked in a
// requires-expression, where narrowing is ill-formed on every compiler; GCC otherwise only warns (-Wnarrowing)
// about a narrowing brace-initialisation inside a template, which would let a narrowing from_tuple compile there.
template <typename S, typename Tup, std::size_t... I>
constexpr bool brace_initialisable_from(std::index_sequence<I...>) noexcept
{
    return requires(Tup&& tup) { S{element<I>(static_cast<Tup&&>(tup))...}; };
}

template <typename S, typename Tup>
constexpr bool converts_without_narrowing() noexcept
{
    if constexpr (std::tuple_size_v<std::remove_cvref_t<Tup>> != member_count_v<S>)
        return true;   // the size static_assert reports this one
    else
        return brace_initialisable_from<S, Tup>(std::make_index_sequence<member_count_v<S>>{});
}

// The noexcept-specification of from_tuple. Ill-formed input (wrong size, narrowing) gives false here instead of an
// error, so the error comes from from_tuple's static_asserts with their snicholls:: message: MSVC stops at an
// ill-formed noexcept-specification and never reaches the body. Well-formed input gives exactly the noexcept of the
// brace-initialisation, as before.
template <typename S, typename Tup>
constexpr bool nothrow_from_tuple() noexcept
{
    if constexpr (std::tuple_size_v<std::remove_cvref_t<Tup>> != member_count_v<S>)
        return false;
    else if constexpr (!brace_initialisable_from<S, Tup>(std::make_index_sequence<member_count_v<S>>{}))
        return false;
    else
        return noexcept(struct_from_tuple<S>(std::declval<Tup>(), std::make_index_sequence<member_count_v<S>>{}));
}

} // namespace detail

// --- public API -----------------------------------------------------------------------

/// \brief References to every member of a plain struct: a `std::tuple` of references.
/// \ingroup core
///
/// \overview
/// The first of the struct / tuple conversions; all are summarised in the file description above. Use them at
/// boundaries with tuple-based code (`std::apply`, comparison through `std::tuple`, C APIs, legacy structs).
///
/// \relations
/// - `snicholls::to_tuple`, `snicholls::from_tuple`; member count from `snicholls::member_count_v`.
/// - A consuming library may add overloads of the same names for its own objects.
/// - Module: \ref core.
///
/// \impls
/// | Function | From | To |
/// |---|---|---|
/// | `tie` | struct (lvalue) | tuple of references |
/// | `to_tuple` | struct | tuple of values |
/// | `from_tuple<S>` | tuple-like | struct |
///
/// \contracts
/// - Everything is `constexpr`. `tie` is always `noexcept`; every other conversion is `noexcept` exactly when the
///   copies, moves and assignments it performs are (a `std::string` copy may throw).
/// - Strict input: sizes must match exactly and arithmetic conversions must not narrow (compile-time errors with a
///   `snicholls::` message).
///
/// \errors
/// Only what the element copies throw (`std::bad_alloc`); misuse is a compile error.
///
/// \examples
/// \snippet struct_tuple_demo.cpp tie
/// \snippet struct_tuple_demo.cpp to_tuple
///
/// \perf
/// `tie`, `to_tuple` and `from_tuple` are benchmarked like for like against hand-written member access
/// (`benchmarks/reflect/struct_tuple_benchmark.cpp`, with a fairness check); see `BENCHMARKS.md`.
///
/// \testing
/// - Tests: tests/reflect/struct_tuple_test.cpp; compile-fail cases (size mismatch, narrowing) in
///   tests/compile_fail/struct_tuple_*.cpp.
/// - Demo: demos/reflect/struct_tuple_demo.cpp.
/// - Benchmark: benchmarks/reflect/struct_tuple_benchmark.cpp.
///
/// \deps
/// CMake target `snicholls::core`; standard library only.
///
/// \gotchas
/// - Plain structs: no reference members, C arrays (use `std::array`), bit-fields, base classes or unions; up to
///   `SNICHOLLS_MAX_STRUCT_MEMBERS` (256) members.
/// - Migrating from a `to_tuple<N>(s)` / `to_struct<S>(tuple)` pair: they are `snicholls::to_tuple(s)` and
///   `snicholls::from_tuple<S>(tuple)`.
///
/// \status
/// Stable. Rules: CLAUDE.md, "Struct / tuple conversion rules".
///
/// \tparam S the struct type.
/// \param s the struct (lvalue).
/// \return a tuple of references.
template <typename S>
    requires plain_struct<S>
constexpr auto tie(S& s) noexcept
{
    return detail::tie_struct(s);
}

/// \brief Values of every member (plain structs). Moves from rvalues.
/// \ingroup core
/// \tparam S the struct type.
/// \param s the struct.
/// \return a tuple of values.
template <typename S>
    requires plain_struct<S>
constexpr auto to_tuple(S&& s) noexcept(
    std::is_nothrow_constructible_v<detail::value_of_t<decltype(detail::tie_struct(std::declval<S&>()))>,
                                    detail::struct_tie_t<S&&>>)
{
    using result = detail::value_of_t<decltype(detail::tie_struct(s))>;
    return result(detail::forward_tie_struct(static_cast<S&&>(s)));
}

/// \brief Tuple-like to plain struct (brace-initialised: narrowing is a compile-time error).
/// \ingroup core
/// \tparam S the struct type.
/// \tparam Tup a tuple-like with one element per member.
/// \param tup the tuple.
/// \return the struct.
template <typename S, typename Tup>
    requires plain_struct<S> && tuple_like<Tup>
constexpr S from_tuple(Tup&& tup) noexcept(detail::nothrow_from_tuple<S, Tup>())
{
    static_assert(std::tuple_size_v<std::remove_cvref_t<Tup>> == member_count_v<S>,
                  "snicholls::from_tuple: tuple size must equal the struct's number of members");
    static_assert(detail::converts_without_narrowing<S, Tup>(),
                  "snicholls::from_tuple: every element must convert to its member without narrowing");
    return detail::struct_from_tuple<S>(static_cast<Tup&&>(tup), std::make_index_sequence<member_count_v<S>>{});
}

} // namespace snicholls

#endif /* snicholls_core_reflect_struct_tuple_hpp */
