//
//  ocaml.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief OCaml's standard modules over core's types: `fn::ocaml::option` (Option), `fn::ocaml::result` (Result) and
///        `fn::ocaml::list` (List), with OCaml's names and argument orders; and the idioms for `|>`, `@@`, labelled
///        and optional arguments and functors. All `constexpr`.
/// \ingroup core
///
/// **Overview.** For code read by people who think in OCaml, the same operations under OCaml's names:
///
/// | OCaml                                     | here                                                  |
/// | ----------------------------------------- | ----------------------------------------------------- |
/// | `x \|> f \|> g`                           | `x \| fn::then(f) \| fn::then(g)`                     |
/// | `f @@ x`                                  | `fn::call(f, x)`                                      |
/// | `Option.map / bind / join / fold`         | `ocaml::option::map(f, o)`, `bind(o, f)`, `join(o)`, `fold(none, some, o)` |
/// | `Option.value ~default`, `get`, `is_some`, `is_none`, `some`, `none` | the same names                  |
/// | `Result.map / map_error / bind / join / fold` | `ocaml::result::map(f, r)`, `map_error(f, r)`, `bind(r, f)`, `join(r)`, `fold(ok, error, r)` |
/// | `Result.value ~default`, `get_ok`, `get_error`, `is_ok`, `is_error`, `to_option` | the same names     |
/// | `List.map / filter / fold_left / fold_right / rev / length / mem / exists / for_all / find_opt / assoc_opt / partition / combine / split / sort / init` | `ocaml::list::...` |
///
/// \snippet ocaml_demo.cpp pipeline
///
/// **Labelled and optional arguments.** OCaml's `f ~rate ~spread ?(floor = 0.) price` is a
/// designated-initialiser struct in C++20: the labels are member names, the optional ones have default
/// member initialisers, and the call reads `f({.rate = 1.085, .spread = 25}, price)`. The compiler
/// checks the names and their order; an omitted member takes its default:
///
/// \snippet ocaml_demo.cpp labelled
///
/// **Functors.** An OCaml functor is a module parameterised by a module: `module Make (Ord : ORDERED) = struct ... end`.
/// In C++ the module type is a concept, the module a struct of static functions and types, and the functor a
/// class template constrained by the concept: `template <ordered Ord> struct make_set { ... };`, applied as
/// `make_set<by_price>`. The demo works one through:
///
/// \snippet ocaml_demo.cpp functor
///
/// **Relationships.** `functional/typeclasses.hpp`, `functional/monad.hpp`, `functional/list.hpp` (the
/// operations underneath), `snicholls::result`. Group: \ref core.
///
/// **Implementations / variants.** The modules are namespaces of function objects; most are the
/// functional/ ones under another name, a few adapt the argument order (`fold_right f l init`).
///
/// **Guarantees.** Those of the operations they name: `constexpr`, allocation-free.
///
/// **Errors.** `option::get` and `result::get_ok` / `get_error` of the wrong case are precondition
/// violations (OCaml raises `Invalid_argument`); they are checked with `assert`.
///
/// **Examples.** See the overview.
///
/// **Performance.** That of the operations underneath; not benchmarked separately.
///
/// **Tests and demos.** `tests/functional/ocaml_test.cpp`, `demos/functional/ocaml_demo.cpp`.
///
/// **Dependencies.** `<cassert>`, `<optional>`, `<utility>`, `functional/monad.hpp` and `functional/list.hpp`.
///
/// **Known issues and gotchas.** `fn::ocaml::result` is a namespace, so with `using namespace
/// snicholls::fn::ocaml;` and `using namespace snicholls;` both open, `result` is ambiguous: name the
/// type `snicholls::result`. OCaml's `List` is a linked list; here it is `std::array` (and ranges for
/// the searches), so `cons` and pattern matching on `::` are left out.
///
/// **Status.** Experimental, new in 0.27.0.

#ifndef snicholls_core_functional_ocaml_hpp
#define snicholls_core_functional_ocaml_hpp

#include <snicholls/core/functional/list.hpp>
#include <snicholls/core/functional/monad.hpp>

#include <cassert>
#include <optional>
#include <utility>

/// \brief OCaml's standard modules under OCaml's names.
/// \ingroup core
namespace snicholls::fn::ocaml {

    namespace detail {

        struct option_bind_fn {
            template <class O, class F>
            constexpr auto operator()(O&& o, F&& f) const SNICHOLLS_FN_RETURNS(fn::mbind(std::forward<O>(o), std::forward<F>(f)))
        };

        struct option_fold_fn {
            template <class N, class S, class O>
            constexpr auto operator()(N&& none, S&& some, O&& o) const
                SNICHOLLS_FN_RETURNS(fn::maybe(std::forward<N>(none), std::forward<S>(some), std::forward<O>(o)))
        };

        struct get_fn {
            template <class O>
            constexpr decltype(auto) operator()(O&& o) const
            {
                assert(o.has_value());
                return *std::forward<O>(o);
            }
        };

        struct is_some_fn {
            template <class T>
            constexpr bool operator()(const std::optional<T>& o) const noexcept { return o.has_value(); }
        };

        struct is_none_fn {
            template <class T>
            constexpr bool operator()(const std::optional<T>& o) const noexcept { return !o.has_value(); }
        };

        struct some_fn {
            template <class T>
            constexpr std::optional<std::decay_t<T>> operator()(T&& x) const { return std::forward<T>(x); }
        };

        struct option_value_fn {
            template <class O, class D>
            constexpr auto operator()(O&& o, D&& fallback) const
                SNICHOLLS_FN_RETURNS(fn::value_or(std::forward<O>(o), std::forward<D>(fallback)))
        };

        struct map_error_fn {
            template <class F, class R>
            constexpr auto operator()(F&& f, R&& r) const SNICHOLLS_FN_RETURNS(std::forward<R>(r).map_err(std::forward<F>(f)))
        };

        struct result_fold_fn {
            template <class Ok, class Error, class R>
            constexpr auto operator()(Ok&& on_ok, Error&& on_error, R&& r) const
                SNICHOLLS_FN_RETURNS(fn::either(std::forward<Error>(on_error), std::forward<Ok>(on_ok), std::forward<R>(r)))
        };

        struct get_ok_fn {
            template <class R>
            constexpr decltype(auto) operator()(R&& r) const
            {
                assert(r.is_ok());
                return std::forward<R>(r).unwrap_unchecked();
            }
        };

        struct get_error_fn {
            template <class R>
            constexpr decltype(auto) operator()(R&& r) const
            {
                assert(r.is_err());
                return std::forward<R>(r).unwrap_err_unchecked();
            }
        };

        struct is_ok_fn {
            template <class R>
            constexpr bool operator()(const R& r) const noexcept { return r.is_ok(); }
        };

        struct is_error_fn {
            template <class R>
            constexpr bool operator()(const R& r) const noexcept { return r.is_err(); }
        };

        struct to_option_fn {
            template <class R>
            constexpr auto operator()(R&& r) const SNICHOLLS_FN_RETURNS(std::forward<R>(r).ok())
        };

        struct fold_right_fn {
            template <class F, class L, class Z>
            constexpr auto operator()(F&& f, L&& l, Z&& z) const
                SNICHOLLS_FN_RETURNS(fn::foldr_of(std::forward<F>(f), std::forward<Z>(z), std::forward<L>(l)))
        };

        // OCaml's compare-style sort: cmp(a, b) < 0 when a comes first.
        struct sort_fn {
            template <class Cmp, class L>
            constexpr auto operator()(Cmp&& cmp, L&& l) const
            {
                return fn::sort_by([&cmp](const auto& a, const auto& b) { return std::invoke(cmp, a, b) < 0; }, std::forward<L>(l));
            }
        };

    } // namespace detail

    /// \brief OCaml's `Option` module over `std::optional`.
    /// \ingroup core
    namespace option {
        /// \brief `Option.map f o`.
        inline constexpr auto& map = fn::fmap;
        /// \brief `Option.bind o f`.
        inline constexpr detail::option_bind_fn bind{};
        /// \brief `Option.join oo`.
        inline constexpr auto& join = fn::join;
        /// \brief `Option.fold ~none ~some o`.
        inline constexpr detail::option_fold_fn fold{};
        /// \brief `Option.value o ~default`.
        inline constexpr detail::option_value_fn value{};
        /// \brief `Option.get o`; \pre `o` holds a value.
        inline constexpr detail::get_fn get{};
        /// \brief `Option.is_some o`.
        inline constexpr detail::is_some_fn is_some{};
        /// \brief `Option.is_none o`.
        inline constexpr detail::is_none_fn is_none{};
        /// \brief `Option.some x`.
        inline constexpr detail::some_fn some{};
        /// \brief `Option.none`, convertible to any optional.
        inline constexpr std::nullopt_t none = std::nullopt;
    } // namespace option

    /// \brief OCaml's `Result` module over `snicholls::result`.
    /// \ingroup core
    namespace result {
        /// \brief `Result.map f r`.
        inline constexpr auto& map = fn::fmap;
        /// \brief `Result.map_error f r`.
        inline constexpr detail::map_error_fn map_error{};
        /// \brief `Result.bind r f`.
        inline constexpr detail::option_bind_fn bind{};
        /// \brief `Result.join rr`.
        inline constexpr auto& join = fn::join;
        /// \brief `Result.fold ~ok ~error r`.
        inline constexpr detail::result_fold_fn fold{};
        /// \brief `Result.value r ~default`.
        inline constexpr detail::option_value_fn value{};
        /// \brief `Result.get_ok r`; \pre `r` is a success.
        inline constexpr detail::get_ok_fn get_ok{};
        /// \brief `Result.get_error r`; \pre `r` is an error.
        inline constexpr detail::get_error_fn get_error{};
        /// \brief `Result.is_ok r`.
        inline constexpr detail::is_ok_fn is_ok{};
        /// \brief `Result.is_error r`.
        inline constexpr detail::is_error_fn is_error{};
        /// \brief `Result.to_option r`.
        inline constexpr detail::to_option_fn to_option{};
    } // namespace result

    /// \brief OCaml's `List` module over `std::array` (and ranges for the searches).
    /// \ingroup core
    namespace list {
        /// \brief `List.map f l`.
        inline constexpr auto& map = fn::map;
        /// \brief `List.filter p l`.
        inline constexpr auto& filter = fn::filter;
        /// \brief `List.fold_left f init l`.
        inline constexpr auto& fold_left = fn::foldl_of;
        /// \brief `List.fold_right f l init`.
        inline constexpr detail::fold_right_fn fold_right{};
        /// \brief `List.rev l`.
        inline constexpr auto& rev = fn::reverse;
        /// \brief `List.length l`.
        inline constexpr auto& length = fn::length;
        /// \brief `List.mem x l`.
        inline constexpr auto& mem = fn::elem;
        /// \brief `List.exists p l`.
        inline constexpr auto& exists = fn::any;
        /// \brief `List.for_all p l`.
        inline constexpr auto& for_all = fn::all;
        /// \brief `List.find_opt p l`.
        inline constexpr auto& find_opt = fn::find;
        /// \brief `List.assoc_opt k l`.
        inline constexpr auto& assoc_opt = fn::lookup;
        /// \brief `List.partition p l`.
        inline constexpr auto& partition = fn::partition;
        /// \brief `List.combine a b` (OCaml raises on different lengths; here the shorter wins).
        inline constexpr auto& combine = fn::zip;
        /// \brief `List.split l`.
        inline constexpr auto& split = fn::unzip;
        /// \brief `List.concat_map f l`.
        inline constexpr auto& concat_map = fn::concat_map;
        /// \brief `List.sort cmp l`, `cmp` returning negative, zero or positive; stable.
        inline constexpr detail::sort_fn sort{};
        /// \brief `List.init n f`, `n` a template argument: `init<N>(f)`.
        /// \tparam N The length.
        /// \param f The function of the index.
        /// \return `{f(0), ..., f(N - 1)}`.
        template <std::size_t N, class F>
        [[nodiscard]] constexpr auto init(F&& f) SNICHOLLS_FN_RETURNS(snicholls::table<N>(std::forward<F>(f)))
    } // namespace list

} // namespace snicholls::fn::ocaml

#endif /* snicholls_core_functional_ocaml_hpp */
