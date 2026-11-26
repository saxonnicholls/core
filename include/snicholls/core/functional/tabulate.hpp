//
//  tabulate.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::fn::tabulate<N>(f)`: a function of an index, memoised at compile time into a table lookup.
/// \ingroup core
///
/// **Overview.** `tabulate<N>(f)` calls `f(0)` ... `f(N - 1)` once, when it is built (by the compiler,
/// for a `constexpr` variable), and returns a function that looks the answer up:
///
/// \snippet tabulate_demo.cpp tabulate
///
/// It is memoisation that can be `constexpr`: the domain is known and small, the values are computed
/// before the program runs, and the hot path is one indexed load. Use it for a pure function of a
/// small index that is called often: binomial coefficients, a discount factor per day, a lookup by
/// enumerator. The table itself, as a `std::array`, is `snicholls::table<N>(f)` (`snicholls/core/iterate.hpp`),
/// which this calls.
///
/// **Run-time memoisation is deliberately absent.** A cache filled as calls arrive needs mutable state
/// and, for an unbounded domain, allocation; both are outside core's rules (compile time first, no
/// hidden state, allocation-free hot paths). Where that is wanted it belongs to the caller, or to a
/// library above core, wrapped around a pure function from here.
///
/// **Relationships.** `snicholls::table` (the array), `snicholls::constexpr_map` and `containers/static_map`
/// (tables keyed by something other than an index), `functional/combinators.hpp`. Group: \ref core.
///
/// **Implementations / variants.** One.
///
/// **Guarantees.** `constexpr`: building and calling. The closure holds the `std::array` of the `N`
/// values and nothing else, so it is trivially copyable when the values are. A call returns the value
/// by value, `noexcept` when copying it is. Building is `noexcept` when `f` and the moves are.
/// \pre The index is less than `N` (checked with `assert`; at compile time an out-of-range index does
/// not compile).
///
/// **Errors.** None; see the precondition.
///
/// **Examples.** \snippet tabulate_demo.cpp binomial
///
/// **Performance.** One indexed load per call; not benchmarked.
///
/// **Tests and demos.** `tests/functional/tabulate_test.cpp`, `demos/functional/tabulate_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<cassert>`, `<cstddef>`, `<type_traits>`, `<utility>`,
/// `snicholls/core/iterate.hpp` and `snicholls/core/functional/combinators.hpp`.
///
/// **Known issues and gotchas.** A large `N` costs compile time (each value is evaluated by the
/// compiler) and the table's size in the binary; keep tables to what the hot path needs.
///
/// **Status.** Experimental, new in 0.27.0.

#ifndef snicholls_core_functional_tabulate_hpp
#define snicholls_core_functional_tabulate_hpp

#include <snicholls/core/functional/combinators.hpp>
#include <snicholls/core/iterate.hpp>

#include <array>
#include <cassert>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace snicholls::fn {

    namespace detail {

        struct tabulate_op {
            template <std::size_t... I, class S, class V = typename std::remove_cvref_t<part_t<0, S>>::value_type>
            static constexpr V call(std::index_sequence<I...>, S&& s,
                                    std::size_t i) noexcept(std::is_nothrow_copy_constructible_v<V>)
            {
                assert((i < std::tuple_size_v<std::remove_cvref_t<part_t<0, S>>>));
                return detail::part<0>(s)[i];
            }
        };

    } // namespace detail

    /// \brief `f` over the indices `0` ... `N - 1`, computed once into a table: a function of `std::size_t` that looks
    ///        `f(i)` up.
    /// \ingroup core
    /// \tparam N The size of the domain.
    /// \tparam F The function's type, called with a `std::size_t`.
    /// \param f The function; called `N` times, here, and never again.
    /// \return A callable `i -> f(i)` holding the `N` values.
    template <std::size_t N, class F>
    [[nodiscard]] constexpr auto tabulate(F&& f) noexcept(noexcept(snicholls::table<N>(f)))
    {
        return detail::make<detail::tabulate_op>(snicholls::table<N>(f));
    }

} // namespace snicholls::fn

#endif /* snicholls_core_functional_tabulate_hpp */
