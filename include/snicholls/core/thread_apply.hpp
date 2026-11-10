//
//  thread_apply.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  It has nothing to do with std::thread; hence thread_apply and not thread. The form for types is
//  meta::thread_t.
//

/// \file
/// \brief `thread_apply` and `outer`: Mathematica's `Thread` and `Outer` over tuples and arrays, resolved per element at compile time.
/// \ingroup core
///
/// **Overview.** `Thread[f[{a, b}, x]]` is `{f[a, x], f[b, x]}`: `f` is applied across its list
/// arguments position by position, and an argument that is not a list is repeated. `thread_apply`
/// is that for tuple-like values, so the elements may have different types, and each call is
/// resolved on its own: an `overloaded` or `first_match` picks a branch per element.
///
/// \snippet thread_apply_demo.cpp thread_apply
///
/// | Wolfram                          | here                                        |
/// | -------------------------------- | ------------------------------------------- |
/// | `Thread[f[{a, b}]]`              | `thread_apply(f, std::tuple{a, b})`         |
/// | `Thread[f[{a, b}, x]]`           | `thread_apply(f, std::tuple{a, b}, x)`      |
/// | `Thread[f[{a, b}, {x, y}]]`      | `thread_apply(f, tuple_ab, tuple_xy)`       |
/// | `Thread[f[x]]`, no list to thread | `thread_apply(f, x)`, which is `f(x)`       |
///
/// Wolfram's position specifications are for types (`meta::thread_t`); here, `whole(t)` marks one
/// tuple to be repeated rather than threaded.
///
/// `outer(f, a, b)` is `Outer[f, a, b]`: every combination rather than corresponding elements.
///
/// \snippet thread_apply_demo.cpp outer
///
/// **Relationships.** `meta/type_list.hpp` for `meta::thread_t`; `overloaded.hpp` for the callables;
/// `zip.hpp` for ranges whose length is known only at run time. Group: \ref core.
///
/// **Guarantees.** `constexpr`, no allocation of its own. Calls are made in order, left to right. A
/// repeated argument reaches `f` as a const lvalue, every time. A threaded rvalue tuple's elements are moved out,
/// each once. The result is a `std::tuple` of the results by value, or `void` if every call returns `void`.
/// A tuple-like is anything `std::tuple_size` knows: `std::tuple`, `std::pair`, `std::array`.
///
/// **Errors.** Compile errors with a message: tuples of different lengths, and a mix of `void` and non-`void` results.
/// Exceptions are whatever `f` throws.
///
/// **Performance.** Not benchmarked: it expands to the calls written out.
///
/// **Tests and demos.** `tests/thread_apply_test.cpp`, `demos/thread_apply_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<cstddef>`, `<functional>`, `<tuple>`, `<type_traits>`, `<utility>`.
///
/// **Known issues.** It threads one level, as `Thread` does; nested tuples reach `f` whole.
///
/// **Status.** New in 0.5.0.

#ifndef snicholls_core_thread_apply_hpp
#define snicholls_core_thread_apply_hpp

#include <array>
#include <cstddef>
#include <functional>
#include <tuple>
#include <type_traits>
#include <utility>

namespace snicholls {

    /// \brief One argument to repeat rather than thread; what `snicholls::whole` returns.
    /// \ingroup core
    /// \tparam T The argument's type.
    template <typename T>
    struct whole_value {
        const T& value;   ///< The argument, not owned.
    };

    /// \brief Mark a tuple to reach every call whole: `thread_apply(f, tuple, whole(other_tuple))`.
    /// \ingroup core
    /// \tparam T The argument's type.
    /// \param value The argument; it must outlive the result.
    /// \return The marked argument.
    template <typename T>
    [[nodiscard]] constexpr whole_value<T> whole(const T& value) noexcept
    {
        return {value};
    }

    namespace detail {

        template <typename T>
        concept threadable = requires { std::tuple_size<std::remove_cvref_t<T>>::value; };

        template <std::size_t I, threadable Arg>
        constexpr decltype(auto) threaded_element(Arg&& arg)
        {
            using std::get;
            return get<I>(std::forward<Arg>(arg));
        }

        template <std::size_t I, typename T>
        constexpr const T& threaded_element(const whole_value<T>& arg) noexcept
        {
            return arg.value;
        }

        template <std::size_t I, typename Arg>
            requires(!threadable<Arg>)
        constexpr const std::remove_cvref_t<Arg>& threaded_element(const Arg& arg) noexcept
        {
            return arg;
        }

        template <typename... Args>
        consteval std::size_t threaded_length() noexcept
        {
            std::size_t length = 0;
            const auto take = [&length]<typename Arg>() noexcept {
                if constexpr (threadable<Arg>) {
                    const std::size_t here = std::tuple_size_v<std::remove_cvref_t<Arg>>;
                    length = length == 0 || here < length ? here : length;
                }
            };
            (take.template operator()<Args>(), ...);
            return length;
        }

        template <typename... Args>
        consteval bool threaded_lengths_agree() noexcept
        {
            const std::size_t shortest = threaded_length<Args...>();
            const auto agrees = [shortest]<typename Arg>() noexcept {
                if constexpr (threadable<Arg>)
                    return std::tuple_size_v<std::remove_cvref_t<Arg>> == shortest;
                else
                    return true;
            };
            return (true && ... && agrees.template operator()<Args>());
        }

        template <std::size_t I, typename F, typename ArgsTuple, std::size_t... As>
        constexpr decltype(auto) thread_call(F& f, ArgsTuple& args, std::index_sequence<As...>)
        {
            return std::invoke(f, threaded_element<I>(std::forward<std::tuple_element_t<As, ArgsTuple>>(std::get<As>(args)))...);
        }

        template <typename F, typename... Args, std::size_t... Is>
        constexpr auto thread_calls(F& f, std::index_sequence<Is...>, Args&&... args)
        {
            std::tuple<Args&&...> refs(std::forward<Args>(args)...);
            const std::index_sequence_for<Args...> positions{};
            using result_at = std::tuple<decltype(thread_call<Is>(f, refs, positions))...>;
            constexpr std::size_t voids = (std::size_t{0} + ... + std::is_void_v<std::tuple_element_t<Is, result_at>>);
            static_assert(voids == 0 || voids == sizeof...(Is), "snicholls::thread_apply: the calls must all return void or all return a value");
            if constexpr (voids != 0)
                (static_cast<void>(thread_call<Is>(f, refs, positions)), ...);
            else
                return std::tuple<std::remove_cvref_t<std::tuple_element_t<Is, result_at>>...>{thread_call<Is>(f, refs, positions)...};
        }

    } // namespace detail

    /// \brief Apply `f` across its tuple arguments position by position, repeating every other argument.
    /// \ingroup core
    /// \tparam F The callable's type.
    /// \tparam Args The arguments' types.
    /// \param f    Called once per position; each call resolves on its own, so an `overloaded` may pick a branch per element.
    /// \param args The arguments: tuple-likes of one length are threaded, anything else repeated.
    /// \return A `std::tuple` of the results by value, or `void` if the calls return `void`.
    /// \throws Whatever `f` throws.
    template <typename F, typename... Args>
        requires(detail::threadable<Args> || ...)
    constexpr auto thread_apply(F&& f, Args&&... args)
    {
        static_assert(detail::threaded_lengths_agree<Args...>(), "snicholls::thread_apply: the threaded tuples must all be the same length");
        return detail::thread_calls(f, std::make_index_sequence<detail::threaded_length<Args...>()>{}, std::forward<Args>(args)...);
    }

    /// \brief With no tuple to thread, `Thread[f[x]]` is `f[x]`: the plain call.
    /// \ingroup core
    /// \tparam F The callable's type.
    /// \tparam Args The arguments' types.
    /// \param f    The callable.
    /// \param args The arguments.
    /// \return What `f` returns.
    /// \throws Whatever `f` throws.
    template <typename F, typename... Args>
        requires(!(detail::threadable<Args> || ...))
    constexpr decltype(auto) thread_apply(F&& f, Args&&... args)
    {
        return std::invoke(std::forward<F>(f), std::forward<Args>(args)...);
    }

    // ---- Outer (Mathematica) -----------------------------------------------------------------------------------------

    namespace detail {

        template <typename T>
        inline constexpr bool outer_is_array = false;
        template <typename T, std::size_t N>
        inline constexpr bool outer_is_array<std::array<T, N>> = N > 0;

        // A row of results: a std::array when the list was one, otherwise a std::tuple.
        template <bool Array, typename... Rs>
        struct outer_row {
            using type = std::tuple<Rs...>;
        };
        template <typename R, typename... Rs>
        struct outer_row<true, R, Rs...> {
            using type = std::array<R, 1 + sizeof...(Rs)>;
        };

        template <typename F, typename List, std::size_t... Is>
        constexpr auto outer_last(F& f, List& list, std::index_sequence<Is...>)
        {
            using std::get;
            using row = typename outer_row<outer_is_array<std::remove_cvref_t<List>>,
                                           std::remove_cvref_t<std::invoke_result_t<F&, decltype(get<Is>(list))>>...>::type;
            return row{std::invoke(f, get<Is>(list))...};   // braces: the calls are made in order
        }

        template <typename F, typename List, typename... Rest>
        constexpr auto outer_rows(F& f, List& list, Rest&... rest);

        template <typename F, typename List, typename... Rest, std::size_t... Is>
        constexpr auto outer_nest(F& f, List& list, std::index_sequence<Is...>, Rest&... rest)
        {
            using std::get;
            const auto bound = [&f](auto& element) {
                return [&f, &element](auto&&... more) -> decltype(auto) { return std::invoke(f, element, std::forward<decltype(more)>(more)...); };
            };
            using row = typename outer_row<outer_is_array<std::remove_cvref_t<List>>,
                                           decltype(outer_rows(std::declval<decltype(bound(get<Is>(list)))&>(), rest...))...>::type;
            return [&]<std::size_t... Js>(std::index_sequence<Js...>) {
                [[maybe_unused]] auto binders = std::tuple{bound(get<Js>(list))...};   // unused for an empty list
                return row{outer_rows(std::get<Js>(binders), rest...)...};
            }(std::index_sequence<Is...>{});
        }

        template <typename F, typename List, typename... Rest>
        constexpr auto outer_rows(F& f, List& list, Rest&... rest)
        {
            constexpr std::size_t n = std::tuple_size_v<std::remove_cvref_t<List>>;
            if constexpr (sizeof...(Rest) == 0)
                return outer_last(f, list, std::make_index_sequence<n>{});
            else
                return outer_nest(f, list, std::make_index_sequence<n>{}, rest...);
        }

    } // namespace detail

    /// \brief Mathematica's `Outer[f, list1, list2, ...]`: `f` of every combination, nested one level per list.
    ///
    /// `outer(f, {a, b}, {x, y})` is `{{f(a, x), f(a, y)}, {f(b, x), f(b, y)}}`. Each call is resolved on its own, so an
    /// `overloaded` picks a branch per pair of types. A level is a `std::array` when its list was one (and not empty),
    /// otherwise a `std::tuple`; the type-level form is `meta::outer_with_t`.
    /// \ingroup core
    /// \tparam F     The callable's type.
    /// \tparam Lists The lists: tuple-likes (`std::tuple`, `std::pair`, `std::array`).
    /// \param f     Called once per combination, in row-major order (the last list varies fastest), never with `void` result.
    /// \param lists The lists; their elements are passed as lvalues, never moved, since each is used more than once.
    /// \return The nested results.
    /// \throws Whatever `f` throws.
    template <typename F, typename... Lists>
        requires(sizeof...(Lists) > 0 && (detail::threadable<Lists> && ...))
    [[nodiscard]] constexpr auto outer(F&& f, Lists&&... lists)
    {
        return detail::outer_rows(f, lists...);
    }

    /// \brief Wolfram's `Inner[f, list1, list2, g]`: `g(f(a0, b0), f(a1, b1), ...)`, the generalised dot product of two tuples.
    ///
    /// `inner(times, a, b, plus)` is the dot product; with an overload set for `f`, each pair is resolved on its own.
    /// \ingroup core
    /// \tparam F  The pairwise callable's type.
    /// \tparam A  The first tuple-like.
    /// \tparam B  The second, the same length.
    /// \tparam G  The combining callable's type, called with every pairwise result.
    /// \param f The pairwise function.
    /// \param a The first.
    /// \param b The second.
    /// \param g The combining function.
    /// \return `g` of the pairwise results.
    /// \throws Whatever `f` and `g` throw.
    template <typename F, typename A, typename B, typename G>
        requires detail::threadable<A> && detail::threadable<B>
    [[nodiscard]] constexpr decltype(auto) inner(F&& f, A&& a, B&& b, G&& g)
    {
        static_assert(std::tuple_size_v<std::remove_cvref_t<A>> == std::tuple_size_v<std::remove_cvref_t<B>>,
                      "snicholls::inner: the tuples must be the same length");
        return std::apply(std::forward<G>(g), thread_apply(std::forward<F>(f), std::forward<A>(a), std::forward<B>(b)));
    }

} // namespace snicholls

#endif /* snicholls_core_thread_apply_hpp */
