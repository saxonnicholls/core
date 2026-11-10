//
//  overloaded.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  The run-time std::any dispatch those variants had is snicholls/core/invoke_with.hpp.
//

/// \file
/// \brief `snicholls::overloaded` and `snicholls::first_match`: several callables as one, resolved at compile time.
/// \ingroup core
///
/// **Overview.** Two ways to make one callable out of several. They differ only in which
/// callable a call picks:
///
/// - `overloaded{f, g, h}` is an overload set. A call is resolved by ordinary overload
///   resolution: the best match wins and an ambiguity is a compile error. It is the usual
///   companion to `std::visit`, one handler per alternative:
///
///   \snippet overloaded_demo.cpp visit
///
/// - `first_match{f, g, h}` is an `if constexpr` chain without the `if constexpr`. A call
///   goes to the first callable, in the order written, that accepts the arguments; the rest
///   are not considered, so overlapping conditions are settled by order and nothing is ever
///   ambiguous. With concept-constrained lambdas every callable states its condition and the
///   chain reads top to bottom, like the chain it replaces:
///
///   \snippet overloaded_demo.cpp first_match
///
/// Which one: `overloaded` when every argument type has exactly one handler (a variant's
/// alternatives, a type registry's types); `first_match` when the handlers are conditions
/// that can overlap (integral, enumeration, range, streamable, anything) and the order is the
/// rule. A `first_match` can be an element of an `overloaded` and the other way round.
///
/// Both accept lambdas, function objects and function pointers; a pointer is wrapped in
/// `snicholls::fn_ptr`, which calls it with its exact signature. Both are resolved at compile
/// time and cost nothing at run time. For a dispatch decided at run time, over `std::any`
/// arguments, see `snicholls::invoke_any` in `snicholls/core/invoke_with.hpp`.
///
/// **Relationships.** `std::visit`; base-encode-decode's type registry `dispatch`;
/// `snicholls::invoke_any`, the run-time counterpart; `snicholls::function_ref` refers to
/// either. Group: \ref core.
///
/// **Guarantees.** Both are aggregates that derive from each callable: no storage beyond the
/// callables, no indirection, and `std::is_empty_v` when every callable is empty. Copyable and
/// movable exactly when every callable is. `constexpr` whenever the chosen callable is. A
/// `first_match` call is `noexcept` exactly when the chosen callable's call is, and is
/// SFINAE-friendly: `std::is_invocable_v` is false, not an error, when no callable accepts the
/// arguments. A `first_match` considers a callable through the same constness as itself, so a
/// `mutable` lambda is skipped when the `first_match` is const.
///
/// **Errors.** None of their own. An ambiguous or missing `overloaded` overload, or a
/// `first_match` that no callable accepts, is a compile error at the call site.
///
/// **Performance.** No run-time footprint: a call is the chosen callable's call, inlined.
/// Not benchmarked. `benchmarks/invoke_with_benchmark.cpp` measures the run-time path
/// against a direct call through an `overloaded`.
///
/// **Tests and demos.** `tests/overloaded_test.cpp`, `demos/overloaded_demo.cpp`.
///
/// **Dependencies.** `<cstddef>`, `<type_traits>`, `<utility>` and
/// `snicholls/core/meta/type_list.hpp` (for `first_match`'s choice).
///
/// **Known issues.**
/// - `first_match` asks "does this callable accept these arguments?" (`std::is_invocable`), and
///   implicit conversions count: `[](bool)` accepts an `int`, `[](double)` accepts an `int`,
///   `[](const std::string&)` accepts a string literal. Put the exact cases first, or constrain
///   them: `[](std::same_as<bool> auto)`.
/// - `overloaded` has the opposite trap: an exactly matching generic lambda beats a converting
///   concrete one, so `overloaded{[](int) {...}, [](const auto&) {...}}` called with a `short`
///   takes the generic lambda. `first_match` takes the `int` one.
/// - To decide whether a generic lambda accepts the arguments, the compiler instantiates its
///   body (the return type is deduced from it), for every callable, whichever one is chosen. So
///   each body must be well-formed for everything its constraint admits: a
///   `[](std::integral auto n) { std::from_chars(..., n); }` fails to compile when called with a
///   `bool`, even behind a `[](std::same_as<bool> auto)` that would take it, because `bool` is
///   integral and `std::from_chars` has no `bool` overload. Constrain each branch to what its
///   body can take, or state the return type (`-> std::string`) so the body is not needed.
/// - A callable must be a class type or a function pointer. A C-variadic function pointer and a
///   `final` class cannot be bases: wrap them in a lambda. Two callables of the same type (the
///   same function pointer type twice) cannot both be bases either.
///
/// **Status.** `overloaded` stable. `first_match` and `fn_ptr` new in 0.1.0.

#ifndef snicholls_core_overloaded_hpp
#define snicholls_core_overloaded_hpp

#include <snicholls/core/meta/concepts.hpp>

#include <snicholls/core/meta/type_list.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

/// \brief Makes every empty base cost nothing on MSVC too (`__declspec(empty_bases)`); empty elsewhere.
/// \ingroup core
///
/// MSVC applies the empty-base optimisation to the first empty base only, so without this an
/// `overloaded` of a function pointer and an empty lambda would be padded past the pointer.
#if defined(_MSC_VER) && !defined(__clang__)
#define SNICHOLLS_EMPTY_BASES __declspec(empty_bases)
#else
#define SNICHOLLS_EMPTY_BASES
#endif

namespace snicholls {

    /// \brief A function pointer as a callable class, so that it can be an element of an overload set.
    /// \ingroup core
    ///
    /// `overloaded{&f, g}` and `first_match{&f, g}` wrap a pointer in this through their deduction
    /// guides; it is rarely named directly. The call has the pointer's exact signature, so it takes
    /// part in overload resolution exactly as the function would, and `snicholls::invoke_any` can
    /// read its parameter types. Defined for `R(*)(Args...)` and `R(*)(Args...) noexcept`; a
    /// C-variadic pointer is rejected with a message (wrap it in a lambda).
    /// \tparam P The function pointer type.
    template <typename P>
    struct fn_ptr {
        static_assert(always_false_v<P>,
                      "snicholls::fn_ptr: only a non-variadic function pointer can be an element of an overload set; "
                      "wrap anything else in a lambda");
    };

    /// \brief `snicholls::fn_ptr` for `R(*)(Args...)`.
    /// \tparam R    The result type.
    /// \tparam Args The parameter types.
    template <typename R, typename... Args>
    struct fn_ptr<R (*)(Args...)> {
        /// \brief The pointer type.
        using pointer = R (*)(Args...);

        /// \brief The function called.
        pointer target;

        /// \brief Wrap `p`. Implicit, so that `overloaded{&f, g}` initialises this base from `&f`.
        /// \param p The function; not null.
        constexpr fn_ptr(pointer p) noexcept : target(p) {}   // NOLINT(google-explicit-constructor)

        /// \brief Call the function.
        /// \param args The arguments, forwarded.
        /// \return Its result.
        /// \throws Whatever the function throws.
        constexpr R operator()(Args... args) const { return target(std::forward<Args>(args)...); }
    };

    /// \brief `snicholls::fn_ptr` for `R(*)(Args...) noexcept`.
    /// \tparam R    The result type.
    /// \tparam Args The parameter types.
    template <typename R, typename... Args>
    struct fn_ptr<R (*)(Args...) noexcept> {
        /// \brief The pointer type.
        using pointer = R (*)(Args...) noexcept;

        /// \brief The function called.
        pointer target;

        /// \brief Wrap `p`. Implicit, so that `overloaded{&f, g}` initialises this base from `&f`.
        /// \param p The function; not null.
        constexpr fn_ptr(pointer p) noexcept : target(p) {}   // NOLINT(google-explicit-constructor)

        /// \brief Call the function.
        /// \param args The arguments, forwarded.
        /// \return Its result.
        constexpr R operator()(Args... args) const noexcept { return target(std::forward<Args>(args)...); }
    };

    namespace detail {

        template <typename F>
        inline constexpr bool is_function_pointer_v = std::is_pointer_v<F> && std::is_function_v<std::remove_pointer_t<F>>;

        // What an element passed to overloaded{...} / first_match{...} becomes: a function pointer is
        // wrapped, everything else is itself.
        template <typename F>
        using set_element_t = std::conditional_t<is_function_pointer_v<F>, fn_ptr<F>, F>;

        // The index of the first true flag, or sizeof...(Bs) when none is. A loop, not a chain.
        template <bool... Bs>
        constexpr std::size_t first_true() noexcept
        {
            constexpr bool flags[]{Bs..., false};   // the trailing false keeps the array non-empty
            std::size_t i = 0;
            while (i < sizeof...(Bs) && !flags[i])
                ++i;
            return i;
        }

        // The first of Fs (each already `F&` or `const F&`) invocable with Args. Only instantiated when
        // one is: first_match's call operators are constrained on exactly that.
        template <typename List, typename... Args>
        struct first_accepting;

        template <typename... Fs, typename... Args>
        struct first_accepting<meta::type_list<Fs...>, Args...> {
            static constexpr std::size_t index = first_true<std::is_invocable_v<Fs, Args...>...>();
            using type = meta::at_t<meta::type_list<Fs...>, index>;
        };

        template <typename List, typename... Args>
        using first_accepting_t = typename first_accepting<List, Args...>::type;

    } // namespace detail

    /// \brief An overload set built from several callables: `overloaded{[](int) {...}, [](double) {...}}`.
    /// \ingroup core
    ///
    /// A call is resolved by overload resolution over every callable's `operator()`: the best
    /// match wins, an ambiguity or no match is a compile error. See the file description for when
    /// to prefer `snicholls::first_match`.
    /// \tparam Fs The callable types; each becomes a base. A function pointer is wrapped in `snicholls::fn_ptr`
    ///            by the deduction guide.
    template <typename... Fs>
    struct SNICHOLLS_EMPTY_BASES overloaded : Fs... {
        using Fs::operator()...;
    };

    /// \brief Deduction guide: `overloaded{f, g}` deduces `overloaded<F, G>`, wrapping function pointers.
    /// \tparam Fs The callable types as written.
    template <typename... Fs>
    overloaded(Fs...) -> overloaded<detail::set_element_t<Fs>...>;

    /// \brief An ordered set of callables: a call goes to the first one, in the order written, that accepts the arguments.
    /// \ingroup core
    ///
    /// The compile-time replacement for an `if constexpr` / `else if constexpr` chain over a
    /// type. Each callable's parameter list is its condition, usually a concept-constrained
    /// `auto` parameter (`[](std::integral auto n) {...}`), and a plain `[](const auto&) {...}`
    /// last is the `else`. "Accepts" is `std::is_invocable`, implicit conversions included.
    /// \tparam Fs The callable types; each becomes a base. A function pointer is wrapped in `snicholls::fn_ptr`
    ///            by the deduction guide.
    template <typename... Fs>
    struct SNICHOLLS_EMPTY_BASES first_match : Fs... {
        /// \brief Call the first callable that accepts the arguments.
        /// \tparam Args The argument types.
        /// \param args The arguments, forwarded to the chosen callable.
        /// \return Whatever the chosen callable returns, as it returns it.
        /// \throws Whatever the chosen callable throws; `noexcept` exactly when its call is.
        template <typename... Args>
            requires(std::is_invocable_v<Fs&, Args...> || ...)
        constexpr decltype(auto) operator()(Args&&... args) noexcept(
            std::is_nothrow_invocable_v<detail::first_accepting_t<meta::type_list<Fs&...>, Args...>, Args...>)
        {
            using chosen = detail::first_accepting_t<meta::type_list<Fs&...>, Args...>;
            return static_cast<chosen>(*this)(std::forward<Args>(args)...);
        }

        /// \brief Call the first callable that accepts the arguments through a const `first_match`.
        /// \tparam Args The argument types.
        /// \param args The arguments, forwarded to the chosen callable.
        /// \return Whatever the chosen callable returns, as it returns it.
        /// \throws Whatever the chosen callable throws; `noexcept` exactly when its call is.
        template <typename... Args>
            requires(std::is_invocable_v<const Fs&, Args...> || ...)
        constexpr decltype(auto) operator()(Args&&... args) const noexcept(
            std::is_nothrow_invocable_v<detail::first_accepting_t<meta::type_list<const Fs&...>, Args...>, Args...>)
        {
            using chosen = detail::first_accepting_t<meta::type_list<const Fs&...>, Args...>;
            return static_cast<chosen>(*this)(std::forward<Args>(args)...);
        }
    };

    /// \brief Deduction guide: `first_match{f, g}` deduces `first_match<F, G>`, wrapping function pointers.
    /// \tparam Fs The callable types as written.
    template <typename... Fs>
    first_match(Fs...) -> first_match<detail::set_element_t<Fs>...>;
}

#endif /* snicholls_core_overloaded_hpp */
