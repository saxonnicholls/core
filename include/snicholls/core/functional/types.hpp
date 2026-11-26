//
//  types.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief Type-level functional programming under functional's names: `fn::types::map_t`, `filter_t`, `foldl_t`,
///        `foldr_t`, `zip_t`, `zip_with_t` and friends, aliases of `snicholls::meta::type_list`'s algorithms.
/// \ingroup core
///
/// **Overview.** A list of types is a list too. `meta::type_list` (`meta/type_list.hpp`) owns the
/// algorithms; this header names them as the value-level functions are named, so code that moves
/// between levels reads the same:
///
/// | value level (`fn::`)         | type level (`fn::types::`)        | `meta::`                    |
/// | ---------------------------- | --------------------------------- | --------------------------- |
/// | `map(f, xs)`                 | `map_t<L, F>`                     | `transform_t<L, F>`         |
/// | `filter(p, xs)`              | `filter_t<L, P>`                  | `filter_t<L, P>`            |
/// | `foldl_of(f, z, xs)`         | `foldl_t<L, Z, F>`                | `fold_left_t<L, Z, F>`      |
/// | `foldr_of(f, z, xs)`         | `foldr_t<L, Z, F>`                | `fold_right_t<L, Z, F>`     |
/// | `zip(a, b)`, `zip_with`      | `zip_t<A, B>`, `zip_with_t<F, A, B>` | `zip_t`, `zip_with_t`    |
/// | `reverse`, `concat`          | `reverse_t<L>`, `concat_t<Ls...>` | the same                    |
/// | `take<K>`, `drop<K>`         | `take_t<L, K>`, `drop_t<L, K>`    | the same                    |
/// | `nub`, `sort_by`             | `nub_t<L>`, `sort_by_t<L, Less>`  | `unique_t`, `sort_t`        |
/// | `length`, `elem`             | `length_v<L>`, `elem_v<L, T>`     | `size_v`, `contains_v`      |
///
/// \snippet functional/types_demo.cpp types
///
/// **Relationships.** `meta/type_list.hpp`, which owns every algorithm here (nothing is
/// reimplemented); `functional/list.hpp`, the value level. Group: \ref core.
///
/// **Implementations / variants.** Aliases only.
///
/// **Guarantees.** Compile time only; no run-time footprint.
///
/// **Errors.** Those of `meta::type_list`.
///
/// **Examples.** See the overview.
///
/// **Performance.** Not applicable.
///
/// **Tests and demos.** `tests/functional/types_test.cpp`, `demos/functional/types_demo.cpp`.
///
/// **Dependencies.** `meta/type_list.hpp`.
///
/// **Known issues and gotchas.** The metafunctions are class templates (`template <class> class F`),
/// not lambdas: a type-level function is a template, as in `meta::`.
///
/// **Status.** Experimental, new in 0.27.0.

#ifndef snicholls_core_functional_types_hpp
#define snicholls_core_functional_types_hpp

#include <snicholls/core/meta/type_list.hpp>

#include <cstddef>

/// \brief Type-level functional programming: aliases of `snicholls::meta`'s list algorithms under functional's names.
/// \ingroup core
namespace snicholls::fn::types {

    /// \brief A list of types.
    /// \tparam Ts The types.
    template <class... Ts>
    using list = meta::type_list<Ts...>;

    /// \brief `F<T>` of every type: `map`.
    template <class L, template <class...> class F>
    using map_t = meta::transform_t<L, F>;

    /// \brief The types for which `P<T>::value` holds: `filter`.
    template <class L, template <class> class P>
    using filter_t = meta::filter_t<L, P>;

    /// \brief Left fold: `F<F<Z, T0>, T1>...`.
    template <class L, class Z, template <class, class> class F>
    using foldl_t = meta::fold_left_t<L, Z, F>;

    /// \brief Right fold: `F<T0, F<T1, ... Z>>`.
    template <class L, class Z, template <class, class> class F>
    using foldr_t = meta::fold_right_t<L, Z, F>;

    /// \brief `std::pair<A_i, B_i>` for the length of the shorter.
    template <class A, class B>
    using zip_t = meta::zip_t<A, B>;

    /// \brief `F<A_i, B_i>` for the length of the shorter.
    template <template <class...> class F, class A, class B>
    using zip_with_t = meta::zip_with_t<F, A, B>;

    /// \brief The types in the opposite order.
    template <class L>
    using reverse_t = meta::reverse_t<L>;

    /// \brief The lists joined.
    template <class... Ls>
    using concat_t = meta::concat_t<Ls...>;

    /// \brief The first `K` types.
    template <class L, std::size_t K>
    using take_t = meta::take_t<L, K>;

    /// \brief The types after the first `K`.
    template <class L, std::size_t K>
    using drop_t = meta::drop_t<L, K>;

    /// \brief The types without repeats, first occurrences kept.
    template <class L>
    using nub_t = meta::unique_t<L>;

    /// \brief The types sorted by `Less<A, B>::value`.
    template <class L, template <class, class> class Less>
    using sort_by_t = meta::sort_t<L, Less>;

    /// \brief The number of types.
    template <class L>
    inline constexpr std::size_t length_v = meta::size_v<L>;

    /// \brief Whether `T` is one of the types.
    template <class L, class T>
    inline constexpr bool elem_v = meta::contains_v<L, T>;

} // namespace snicholls::fn::types

#endif /* snicholls_core_functional_types_hpp */
