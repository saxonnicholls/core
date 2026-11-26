//
//  bind.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::fn::bind` with placeholders `arg<N>` / `_1` ... `_9`: arguments bound in any position, `constexpr`.
/// \ingroup core
///
/// **Overview.** `bind(f, a, _2, _1)(x, y)` is `f(a, y, x)`. Each bound argument is a value (stored,
/// and passed to `f`), a placeholder `arg<N>` (the call's N-th argument, counting from one), or another
/// `bind` (called with the same arguments, its result passed on):
///
/// \snippet bind_demo.cpp bind
///
/// Use `bind` when the arguments must be reordered or repeated, or a nested call made, and a lambda
/// would be longer; use `partial` / `partial_right` (`functional/curry.hpp`) when only the leading or
/// trailing arguments are bound. Never `std::bind`: it is not `constexpr`-friendly before C++20, not
/// SFINAE-friendly, and it treats any `std::is_bind_expression` type specially.
///
/// **Relationships.** `functional/curry.hpp` (`partial`, `curry`), `functional/combinators.hpp` (the
/// closures). Group: \ref core.
///
/// **Implementations / variants.** One. The placeholders are `fn::arg<N>`, and `_1` ... `_9` in
/// `snicholls::fn::placeholders`, which is not pulled in by `using namespace snicholls::fn`: write
/// `using namespace snicholls::fn::placeholders;` where they are wanted, so they never meet
/// `std::placeholders` or another library's.
///
/// **Guarantees.** `constexpr`. Bound values are stored by value (decayed), passed as lvalues from an
/// lvalue closure and moved from an rvalue one. A placeholder forwards its argument with the
/// value category it was given; a nested `bind` sees the arguments as lvalues. Arguments no
/// placeholder names are ignored, as `std::bind` does. The call is `noexcept` exactly when `f`'s and
/// the nested calls are; trivially copyable when the bound values are.
///
/// **Errors.** A placeholder beyond the number of arguments given makes the call ill-formed
/// (`std::is_invocable_v` false).
///
/// **Examples.** \snippet bind_demo.cpp nested
///
/// **Performance.** Not benchmarked separately: a call is `f`'s call with the arguments rearranged,
/// inlined, as the curry benchmark (`benchmarks/functional/combinators_benchmark.cpp`) shows for the
/// same closures.
///
/// **Tests and demos.** `tests/functional/bind_test.cpp`, `demos/functional/bind_demo.cpp`.
///
/// **Dependencies.** `<cstddef>`, `<functional>`, `<tuple>`, `<type_traits>`, `<utility>` and
/// `snicholls/core/functional/combinators.hpp`.
///
/// **Known issues and gotchas.** A placeholder named twice with an rvalue argument forwards it twice,
/// so the second use may see a moved-from value; take it by const reference in `f` or pass an lvalue.
///
/// **Status.** Experimental, new in 0.27.0.

#ifndef snicholls_core_functional_bind_hpp
#define snicholls_core_functional_bind_hpp

#include <snicholls/core/functional/combinators.hpp>

#include <cstddef>
#include <functional>
#include <tuple>
#include <type_traits>
#include <utility>

namespace snicholls::fn {

    /// \brief The placeholder for a call's `N`-th argument, counting from one.
    /// \ingroup core
    /// \tparam N The argument's position; at least 1.
    template <std::size_t N>
    struct arg_t {
        static_assert(N >= 1, "snicholls::fn::arg: placeholders count from 1");
        /// \brief The position, counting from one.
        static constexpr std::size_t position = N;
    };

    /// \brief `arg<N>`: the call's `N`-th argument, counting from one, in a `bind`.
    /// \ingroup core
    /// \tparam N The argument's position; at least 1.
    template <std::size_t N>
    inline constexpr arg_t<N> arg{};

    /// \brief `_1` ... `_9`, short for `arg<1>` ... `arg<9>`; opt in with `using namespace snicholls::fn::placeholders;`.
    /// \ingroup core
    namespace placeholders {
        inline constexpr arg_t<1> _1{}; ///< The first argument.
        inline constexpr arg_t<2> _2{}; ///< The second argument.
        inline constexpr arg_t<3> _3{}; ///< The third argument.
        inline constexpr arg_t<4> _4{}; ///< The fourth argument.
        inline constexpr arg_t<5> _5{}; ///< The fifth argument.
        inline constexpr arg_t<6> _6{}; ///< The sixth argument.
        inline constexpr arg_t<7> _7{}; ///< The seventh argument.
        inline constexpr arg_t<8> _8{}; ///< The eighth argument.
        inline constexpr arg_t<9> _9{}; ///< The ninth argument.
    } // namespace placeholders

    namespace detail {

        struct bind_op;

        template <class T>
        inline constexpr bool is_placeholder_v = false;
        template <std::size_t N>
        inline constexpr bool is_placeholder_v<arg_t<N>> = true;

        template <class T>
        inline constexpr bool is_bind_v = false;
        template <class... Ts>
        inline constexpr bool is_bind_v<closure<bind_op, Ts...>> = true;

        // What a bound argument becomes in the call; Args is a std::tuple of references to the call's arguments.
        // Three cases that cannot overlap: a placeholder, a nested bind, a value.
        // The position is checked in the template parameters, not a requires-clause, so that a compiler that
        // substitutes into the return type first (Apple Clang 17) never names an element that does not exist.
        template <class B, class Args>
        inline constexpr bool names_an_argument_v = false;
        template <std::size_t N, class Args>
        inline constexpr bool names_an_argument_v<arg_t<N>, Args> = (N <= std::tuple_size_v<Args>);

        template <class B, class Args,
                  std::enable_if_t<names_an_argument_v<std::remove_cvref_t<B>, std::remove_reference_t<Args>>, int> = 0>
        constexpr auto resolve(B&&, Args&& args)
            SNICHOLLS_FN_RETURNS(std::get<std::remove_cvref_t<B>::position - 1>(std::move(args)))

        template <class B, class Args>
            requires is_bind_v<std::remove_cvref_t<B>>
        constexpr auto resolve(B&& b, Args&& args)
            SNICHOLLS_FN_RETURNS(detail::apply_tuple(std::forward<B>(b), args, tuple_indices<Args>{}))

        template <class B, class Args>
            requires(!is_placeholder_v<std::remove_cvref_t<B>> && !is_bind_v<std::remove_cvref_t<B>>)
        constexpr B&& resolve(B&& b, Args&&) noexcept
        {
            return std::forward<B>(b);
        }

        // Part 0 is f, parts 1... the bound arguments.
        struct bind_op {
            template <std::size_t... I, class S, class Args>
            static constexpr auto with(std::index_sequence<0, I...>, S&& s, Args&& args)
                SNICHOLLS_FN_RETURNS(std::invoke(detail::part<0>(std::forward<S>(s)),
                                                 detail::resolve(detail::part<I>(std::forward<S>(s)), args)...))

            template <std::size_t... I, class S, class... A>
            static constexpr auto call(std::index_sequence<I...> is, S&& s, A&&... a)
                SNICHOLLS_FN_RETURNS(bind_op::with(is, std::forward<S>(s), std::forward_as_tuple(std::forward<A>(a)...)))
        };

    } // namespace detail

    /// \brief `bind(f, bound...)(args...)` is `f(resolved...)`: each bound value passed, each `arg<N>` replaced by
    ///        the `N`-th argument, each nested `bind` called with the arguments.
    /// \ingroup core
    ///
    /// `bind(std::minus<>{}, _2, _1)(1, 10) == 9`. The bound values are stored by value.
    inline constexpr detail::variadic_maker<detail::bind_op> bind{};

} // namespace snicholls::fn

#endif /* snicholls_core_functional_bind_hpp */
