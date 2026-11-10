//
//  invoke_with.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  The candidates are tried with checks instead of by catching exceptions, the callable's result comes back
//  instead of being dropped, and no match is a snicholls::result instead of a thrown exception.
//

/// \file
/// \brief `snicholls::invoke_with` and `invoke_any`: call an overload set with arguments that exist only at run time.
/// \ingroup core
///
/// **Overview.** `snicholls::overloaded` and `snicholls::first_match` choose the callable at
/// compile time, from the static types of the arguments. `invoke_with(set, source)` is the slow
/// option for arguments that exist only at run time: a command parsed from a line of text, a
/// message off a socket, a JSON-RPC call. The *source* is any object that says how many arguments
/// there are and hands out argument `i` as a requested type, or refuses
/// (`snicholls::argument_source`). `invoke_with` tries the set's callables in the order written and
/// calls the first whose parameter list the source can fill: the same count, and every parameter
/// type supplied. The callable's result comes back converted to `R` (`std::any` by default; empty,
/// or `R{}`, for `void`); otherwise an `overload_mismatch` says why nothing was called.
///
/// `invoke_any(set, {args...})` is `invoke_with` over `snicholls::any_source`: a
/// `std::span<const std::any>`, each `std::any` holding exactly the parameter's type (its
/// `std::remove_cvref_t`; a `std::any` has no conversions, so an `int` parameter takes a `std::any`
/// holding an `int`, never a `long` or a `short`):
///
/// \snippet invoke_with_demo.cpp commands
///
/// A source of your own is a few lines: `size()`, and `get<T>(i)` returning either a pointer into
/// the source (`const T*`, null to refuse) or a value (`std::optional<T>`, empty to refuse; moved
/// into a by-value or `&&` parameter). The demo's `token_source` fills parameters from the words
/// of a command line:
///
/// \snippet invoke_with_demo.cpp source
///
/// Every callable needs a parameter list the library can read, `snicholls::callable_traits`:
/// exactly one non-template `operator()` (a non-generic lambda, a function object, a
/// `std::function`), or a function pointer. A generic lambda has no single parameter list and is
/// rejected at compile time with a message. The first argument can also be a single callable
/// rather than a set, and a set inside a set is flattened.
///
/// **Relationships.** `snicholls::overloaded` and `snicholls::first_match`, whose callables it
/// reads; `snicholls::result`, which carries the outcome; `snicholls::any_source`, the source
/// behind `invoke_any`. A table of named sets is one `std::map` of `std::function`s away:
///
/// \snippet invoke_with_demo.cpp table
///
/// Group: \ref core.
///
/// **Guarantees.** The set and the source are not modified. From a pointer source, a parameter
/// taken by value or by `const&` binds the source's object directly (no copy for `const&`) and one
/// taken by `&&` gets a copy; from a value source the value is moved into a by-value or `&&`
/// parameter. The candidates are tried in order and the first match is called, once; whatever it
/// throws propagates and nothing else is tried. Not `noexcept`: converting the result to `R` can
/// allocate, the source may allocate, and the callable may throw. Complexity: at most one
/// `get<T>(i)` per parameter per candidate before the call (for `any_source`, one `type_info`
/// comparison each).
///
/// **Errors.**
/// - Returns `snicholls::err(overload_mismatch::arity)` when no callable takes `source.size()`
///   arguments, and `err(overload_mismatch::types)` when one does but the source refuses one of
///   its parameter types. `snicholls::error_text` spells both out, so `unwrap()` on a miss says
///   which.
/// - Compile-time, each with a `snicholls::invoke_with:` message: a callable whose parameter list
///   cannot be read; a parameter that is a non-const lvalue reference (a source lends or gives,
///   never shares); a callable not callable through the set's constness (a `mutable` lambda in a
///   const set); a result that does not convert to `R` (for `std::any`, a non-copyable result); a
///   `get<T>(i)` that returns neither a pointer nor an optional.
/// - \throws std::bad_alloc from `R` or the source, and whatever the chosen callable throws.
///
/// **Performance.** `benchmarks/invoke_with_benchmark.cpp` runs the same 4,096 `invoke_any` calls
/// three ways, after a fairness check that they agree: through `invoke_any`, through the same
/// `std::any_cast` chain written by hand, and as the direct compile-time call through the
/// `overloaded`. Release, Apple clang 17, Xeon W-3245 (a loaded machine, so indicative only),
/// 2026-11-05: 27 ns per call through `invoke_any`, 22 ns by hand, 6 ns direct. The difference to
/// the hand-written chain is the `snicholls::result` and the `std::any` moves on the way out.
///
/// **Tests and demos.** `tests/invoke_with_test.cpp`, `demos/invoke_with_demo.cpp`, and the
/// compile-fail cases `tests/compile_fail/invoke_with_generic_lambda.cpp` and
/// `tests/compile_fail/invoke_with_mutable_reference.cpp`.
///
/// **Dependencies.** `<any>`, `<concepts>`, `<cstddef>`, `<initializer_list>`, `<span>`,
/// `<type_traits>`, `<utility>`, `snicholls/core/meta/callable_traits.hpp`, `snicholls/core/overloaded.hpp`,
/// `snicholls/core/meta/type_list.hpp` and `snicholls/core/result.hpp`. Heavier than it looks:
/// `result.hpp` brings `<format>` and `<string>`. The compile-time sets in `overloaded.hpp` do not
/// include this header, so code that only visits a variant pays none of it.
///
/// **Known issues.**
/// - `any_source` matches the dynamic type exactly, so build the arguments with the parameter
///   types in mind: `std::any{42}` holds an `int`, `std::any{"x"}` holds a `const char*`, not a
///   `std::string`.
/// - Two callables with the same parameter types once references are stripped (`(int)` and
///   `(const int&)`) both match; the first written is called.
/// - An `operator()` with a ref-qualifier (`&`, `&&`) or `volatile` is not read.
///
/// **Status.** New in 0.5.0, experimental.

#ifndef snicholls_core_invoke_with_hpp
#define snicholls_core_invoke_with_hpp

#include <snicholls/core/meta/callable_traits.hpp>
#include <snicholls/core/overloaded.hpp>
#include <snicholls/core/meta/type_list.hpp>
#include <snicholls/core/result.hpp>

#include <any>
#include <concepts>
#include <cstddef>
#include <initializer_list>
#include <span>
#include <type_traits>
#include <utility>

namespace snicholls {

    /// \brief Why `snicholls::invoke_with` called nothing.
    /// \ingroup core
    enum class overload_mismatch : unsigned char {
        arity,   ///< No callable takes that many arguments.
        types,   ///< A callable takes that many, but the source refuses one of its parameter types for every one.
    };

    /// \brief The text of an `overload_mismatch`, found by `snicholls::error_text` and so by `unwrap()`'s message.
    /// \ingroup core
    /// \param m The mismatch.
    /// \return A static string.
    [[nodiscard]] constexpr const char* snicholls_error_text(overload_mismatch m) noexcept
    {
        return m == overload_mismatch::arity ? "no overload takes this many arguments"
                                             : "no overload takes arguments of these types";
    }

    /// \brief What `snicholls::invoke_with` reads arguments from: a count, and argument `i` as a `T` or a refusal.
    /// \ingroup core
    ///
    /// `size()` is the number of arguments. `get<T>(i)`, for every object type `T` a callable may
    /// ask for, returns either `const T*` (a pointer into the source, null to refuse) or
    /// `std::optional<T>` (a value, empty to refuse). `snicholls::any_source` is the `std::any` one;
    /// a JSON source, a command-line source and the like are written the same way.
    template <typename S>
    concept argument_source = requires(const S& s, std::size_t i) {
        { s.size() } -> std::convertible_to<std::size_t>;
        s.template get<int>(i);
    };

    /// \brief The `std::any` argument source: argument `i` is the object inside `args[i]` when that holds exactly a `T`.
    /// \ingroup core
    struct any_source {
        /// \brief The arguments.
        std::span<const std::any> args;

        /// \brief The number of arguments.
        /// \return `args.size()`.
        [[nodiscard]] constexpr std::size_t size() const noexcept { return args.size(); }

        /// \brief Argument `i` as a `T`.
        /// \tparam T The parameter type, references and cv stripped.
        /// \param i The index; less than `size()`.
        /// \return A pointer to the object inside `args[i]`, or null when it holds another type.
        template <typename T>
        [[nodiscard]] const T* get(std::size_t i) const noexcept
        {
            return std::any_cast<T>(&args[i]);
        }
    };

    namespace detail {

        // The callables of a set, flattened; a single callable is a set of one.
        template <typename T>
        struct callables_of {
            using type = meta::type_list<T>;
        };
        template <typename... Fs>
        struct callables_of<overloaded<Fs...>> {
            using type = meta::concat_t<typename callables_of<Fs>::type...>;
        };
        template <typename... Fs>
        struct callables_of<first_match<Fs...>> {
            using type = meta::concat_t<typename callables_of<Fs>::type...>;
        };
        template <typename T>
        using callables_of_t = typename callables_of<std::remove_cv_t<T>>::type;

        // A callable F reached through a set S, with S's constness: `F&` or `const F&`.
        template <typename S, typename F>
        using element_ref_t = std::conditional_t<std::is_const_v<S>, const F&, F&>;

        // A non-const lvalue reference can bind neither a borrowed const object nor a value about to go.
        template <typename A>
        inline constexpr bool bindable_from_source_v =
            !(std::is_lvalue_reference_v<A> && !std::is_const_v<std::remove_reference_t<A>>);

        // What get<T>(i) hands back: contextually bool, and *h a T.
        template <typename H, typename T>
        concept holds = requires(H& h) {
            static_cast<bool>(h);
            { *h } -> std::convertible_to<const T&>;
        };

        // A borrowed argument (a pointer into the source) reaching a parameter of type A: by const
        // reference when A binds one, as a copy when A is an rvalue reference.
        template <typename A>
        using borrowed_t =
            std::conditional_t<std::is_rvalue_reference_v<A>, std::remove_cvref_t<A>, const std::remove_cvref_t<A>&>;

        template <typename A, typename H>
            requires std::is_pointer_v<H>
        constexpr borrowed_t<A> take(H h)
        {
            return *h;
        }

        // An owned argument (a value the source produced) reaching a parameter of type A: moved unless A
        // is a const lvalue reference.
        template <typename A>
        using owned_t =
            std::conditional_t<std::is_lvalue_reference_v<A>, const std::remove_cvref_t<A>&, std::remove_cvref_t<A>&&>;

        template <typename A, typename H>
            requires(!std::is_pointer_v<H>)
        constexpr owned_t<A> take(H& h)
        {
            return static_cast<owned_t<A>>(*h);
        }

        // Call f and keep its result in out as an R: R{} for void, otherwise the value converted.
        template <typename R, typename F, typename... X>
            requires std::is_void_v<std::invoke_result_t<F, X...>>
        void call_into(R& out, F&& f, X&&... x)
        {
            std::forward<F>(f)(std::forward<X>(x)...);
            out = R{};
        }

        template <typename R, typename F, typename... X>
            requires(!std::is_void_v<std::invoke_result_t<F, X...>>)
        void call_into(R& out, F&& f, X&&... x)
        {
            static_assert(std::is_constructible_v<R, std::invoke_result_t<F, X...>>,
                          "snicholls::invoke_with: a callable's result must convert to R (for std::any, a copyable "
                          "type); return void or a convertible type");
            out = R(std::forward<F>(f)(std::forward<X>(x)...));
        }

        // One candidate whose arity already matches: call it if the source supplies every parameter.
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(push)
#pragma warning(disable : 4702)   // unreachable only for a callable that always throws, which MSVC inlines
#endif
        template <typename R, typename F, argument_source S, typename... A, std::size_t... I>
        bool try_call(F&& f, [[maybe_unused]] const S& source, R& out, meta::type_list<A...>,
                      std::index_sequence<I...>)
        {
            static_assert((bindable_from_source_v<A> && ...),
                          "snicholls::invoke_with: a parameter cannot be a non-const lvalue reference, because a source "
                          "lends a const object or gives a value; take it by value, by const& or by &&");
            static_assert(std::is_invocable_v<F, borrowed_t<A>...>,
                          "snicholls::invoke_with: the callable cannot be called through the set's constness (a "
                          "mutable lambda needs a non-const set)");
            // One get per parameter: a refusal (null, or an empty optional) is a parameter the source cannot fill.
            return [&](auto... held) {
                static_assert((holds<decltype(held), std::remove_cvref_t<A>> && ...),
                              "snicholls::invoke_with: a source's get<T>(i) must return a const T* or a std::optional<T>");
                if (!(held && ...))
                    return false;
                call_into(out, std::forward<F>(f), take<A>(held)...);
                return true;
            }(source.template get<std::remove_cvref_t<A>>(I)...);
        }
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(pop)
#endif

        // One candidate: read its parameter list, check the arity, then the parameters.
        template <typename F, typename R, typename Set, argument_source S>
        bool try_one(Set& set, const S& source, R& out, bool& arity_seen)
        {
            static_assert(single_signature_callable<F>,
                          "snicholls::invoke_with: every callable needs exactly one non-template operator() or to be "
                          "a function pointer, so that its parameter types can be read; a generic lambda has none");
            using params = typename callable_traits<F>::arguments;
            if (source.size() != meta::size_v<params>)
                return false;
            arity_seen = true;
            return try_call(static_cast<element_ref_t<Set, F>>(set), source, out, params{},
                            std::make_index_sequence<meta::size_v<params>>{});
        }

        // The candidates in order; the first that is called ends the fold.
        template <typename R, typename Set, argument_source S, typename... Fs>
        result<R, overload_mismatch> dispatch_with([[maybe_unused]] Set& set, [[maybe_unused]] const S& source,
                                                   meta::type_list<Fs...>)
        {
            static_assert(std::is_default_constructible_v<R>,
                          "snicholls::invoke_with: R must be default constructible; it is what a void callable returns");
            R out{};
            bool arity_seen = false;
            if ((try_one<Fs>(set, source, out, arity_seen) || ...))
                return result<R, overload_mismatch>(std::in_place, std::move(out));
            return err(arity_seen ? overload_mismatch::types : overload_mismatch::arity);
        }

    } // namespace detail

    /// \brief Call the first callable of `f` whose parameter list the source can fill, chosen at run time.
    /// \ingroup core
    /// \tparam R The result type every callable's result converts to; `std::any` by default. `R{}` stands for `void`.
    /// \tparam F A `snicholls::overloaded`, a `snicholls::first_match`, or one callable with a readable parameter list.
    /// \tparam S An `snicholls::argument_source`.
    /// \param f      The set or callable. Not modified, beyond what the chosen callable does to its own state.
    /// \param source The arguments.
    /// \return The chosen callable's result as an `R`, or the `overload_mismatch`.
    /// \throws std::bad_alloc from `R` or the source; whatever the chosen callable throws.
    template <typename R = std::any, typename F, argument_source S>
    [[nodiscard]] result<R, overload_mismatch> invoke_with(F&& f, const S& source)
    {
        return detail::dispatch_with<R>(f, source, detail::callables_of_t<std::remove_reference_t<F>>{});
    }

    /// \brief `invoke_with` over `snicholls::any_source`: the arguments are `std::any`s holding exactly the parameter types.
    /// \ingroup core
    /// \tparam R The result type; `std::any` by default.
    /// \tparam F A `snicholls::overloaded`, a `snicholls::first_match`, or one callable with a readable parameter list.
    /// \param f    The set or callable.
    /// \param args The arguments.
    /// \return As `invoke_with`.
    /// \throws As `invoke_with`.
    template <typename R = std::any, typename F>
    [[nodiscard]] result<R, overload_mismatch> invoke_any(F&& f, std::span<const std::any> args)
    {
        return invoke_with<R>(std::forward<F>(f), any_source{args});
    }

    /// \brief `invoke_any(f, {a, b, c})`: the arguments written in place, each converted to a `std::any`.
    /// \ingroup core
    /// \tparam R The result type; `std::any` by default.
    /// \tparam F A `snicholls::overloaded`, a `snicholls::first_match`, or one callable with a readable parameter list.
    /// \param f    The set or callable.
    /// \param args The arguments; `{42, std::string{"x"}}` holds an `int` and a `std::string`.
    /// \return As `invoke_with`.
    /// \throws As `invoke_with`.
    template <typename R = std::any, typename F>
    [[nodiscard]] result<R, overload_mismatch> invoke_any(F&& f, std::initializer_list<std::any> args)
    {
        return invoke_with<R>(std::forward<F>(f), any_source{std::span<const std::any>(args.begin(), args.size())});
    }
}

#endif /* snicholls_core_invoke_with_hpp */
