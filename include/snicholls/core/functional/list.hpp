//
//  list.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief Haskell's Data.List and OCaml's List over `std::array` (lists whose length is known at compile time) and,
///        for the folds and searches, any input range; Foldable's `length`, `null`, `elem`, `sum`, `product`,
///        `maximum`, `minimum`, `fold_map`; and `fst`, `snd`, `map_tuple`. All `constexpr`.
/// \ingroup core
///
/// **Overview.** A `std::array` is a list with its length in its type, so each operation returns a
/// type whose length the compiler knows: `take<K>` an `array<T, K>`, `concat` an `array<T, N * M>`,
/// `transpose` an `array<array<T, N>, M>`. An operation whose length depends on the values (`filter`,
/// `nub`, `group`, `span`) returns a `containers::inplace_vector<T, N>`, an array with a size, so
/// nothing allocates:
///
/// \snippet list_demo.cpp list
///
/// | Haskell / OCaml                                    | here                                                    | returns                     |
/// | -------------------------------------------------- | ------------------------------------------------------- | --------------------------- |
/// | `map`, `List.map`                                  | `map(f, xs)`                                            | `array<R, N>`               |
/// | `filter`, `List.filter`                            | `filter(p, xs)`                                         | `inplace_vector<T, N>`      |
/// | `foldl1`, `foldr1`                                 | `foldl1(f, xs)`, `foldr1(f, xs)` (N > 0)                | `T`                         |
/// | `scanl`, `scanr`                                   | `scanl(f, z, xs)`, `scanr(f, z, xs)`                    | `array<Z, N + 1>`           |
/// | `zip`, `zip3`, `zipWith`, `unzip`                  | `zip(a, b)`, `zip3(a, b, c)`, `zip_with(f, a, b)`, `unzip(ps)` | arrays of the shorter length |
/// | `concat`, `concatMap`                              | `concat(xss)`, `concat_map(f, xs)`                      | `array<T, N * M>`           |
/// | `take`, `drop`, `splitAt`                          | `take<K>(xs)`, `drop<K>(xs)`, `split_at<K>(xs)`         | arrays                      |
/// | `span`, `break`                                    | `span(p, xs)`, `break_(p, xs)`                          | pair of `inplace_vector`    |
/// | `reverse`, `intersperse`, `intercalate`            | `reverse(xs)`, `intersperse(x, xs)`, `intercalate(sep, xss)` | arrays                 |
/// | `group`, `groupBy`, `partition`, `nub`             | `group(xs)`, `group_by(eq, xs)`, `partition(p, xs)`, `nub(xs)` | `inplace_vector`s     |
/// | `sort`, `sortBy`, `sortOn`                         | `sort(xs)`, `sort_by(less, xs)`, `sort_on(key, xs)` (stable) | `array<T, N>`          |
/// | `lookup`, `elem`, `find`, `findIndex`, `all`, `any` | `lookup(k, pairs)`, `elem(x, xs)`, `find(p, xs)`, `find_index(p, xs)`, `all(p, xs)`, `any(p, xs)` | any range |
/// | `replicate`, `transpose`                           | `replicate<N>(x)`, `transpose(xss)`                     | arrays                      |
/// | `subsequences`, `permutations`                     | `subsequences(xs)`, `permutations(xs)`                  | `2^N` / `N!` lists          |
/// | `length`, `null`, `sum`, `product`, `maximum`, `minimum`, `foldMap` | the same names, snake case        | any range                   |
/// | `fst`, `snd`, (`bimap` on tuples)                  | `fst(p)`, `snd(p)`, `map_tuple(f, t)`                   |                             |
///
/// `foldl`, `foldr`, `foldl_of`, `foldr_of` and `unfold` are in `functional/fold.hpp`; infinite lists
/// (`iterate`, `repeat`, `cycle`, `takeWhile` on them) in `functional/lazy.hpp`. For lists known only
/// at run time, `std::views` (`transform`, `filter`, `take`, `drop`, `reverse`) and `<algorithm>` are
/// the tools; the searches and folds here accept any input range as well.
///
/// **Relationships.** `functional/fold.hpp`, `functional/lazy.hpp`, `functional/instances.hpp`
/// (`concat_map` is the array monad's `mbind`), `snicholls::fold_list` (`scanl` calls it),
/// `snicholls::permutations` (`permutations` calls it), `containers::inplace_vector`. Group: \ref core.
///
/// **Implementations / variants.** One each.
///
/// **Guarantees.** `constexpr`, allocation-free (C++20's `std::sort`-family algorithms are `constexpr`;
/// `sort_by` is stable, an insertion sort, suited to the short lists that are sorted at compile time).
/// Elements are copied out of the input, which is never modified. `inplace_vector` results need a
/// default-constructible element type.
///
/// **Errors.** `foldl1` / `foldr1` of an empty array, and `subsequences` of more than 20 elements, do not
/// compile. `maximum` / `minimum` of an empty range is a precondition violation (checked with `assert`).
///
/// **Examples.** \snippet list_demo.cpp search
///
/// **Performance.** Loops over the elements; `nub` and `group_by` are quadratic, as in Haskell. Not
/// benchmarked separately.
///
/// **Tests and demos.** `tests/functional/list_test.cpp`, `demos/functional/list_demo.cpp`.
///
/// **Dependencies.** `<algorithm>`, `<array>`, `<cassert>`, `<cstddef>`, `<functional>`, `<iterator>`,
/// `<optional>`, `<ranges>`, `<tuple>`, `<type_traits>`, `<utility>`, `containers/inplace_vector.hpp`,
/// `combinatorics/permutations.hpp`, `iterate.hpp` and `functional/instances.hpp`.
///
/// **Known issues and gotchas.** `fn::zip` and `fn::partition` share their names with `snicholls::zip`
/// (run-time ranges walked together) and with `std::partition`; with both namespaces opened, qualify
/// them. `permutations` grows as `N!`: twelve elements are already half a billion lists.
///
/// **Status.** Experimental, new in 0.27.0.

#ifndef snicholls_core_functional_list_hpp
#define snicholls_core_functional_list_hpp

#include <snicholls/core/combinatorics/permutations.hpp>
#include <snicholls/core/containers/inplace_vector.hpp>
#include <snicholls/core/functional/instances.hpp>
#include <snicholls/core/iterate.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <functional>
#include <iterator>
#include <optional>
#include <ranges>
#include <tuple>
#include <type_traits>
#include <utility>

namespace snicholls::fn {

    namespace detail {

        template <class A>
        using elem_t = typename std::remove_cvref_t<A>::value_type;

        template <class A>
        inline constexpr std::size_t size_v = std::tuple_size_v<std::remove_cvref_t<A>>;

        template <class R>
        using range_value_t = std::remove_cvref_t<std::ranges::range_reference_t<R>>;

        // An array of N elements, element J being make(J); the elements need not be default-constructible.
        template <std::size_t N, class Make>
        constexpr auto generate(Make&& make)
        {
            return [&]<std::size_t... J>(std::index_sequence<J...>) {
                using T = std::remove_cvref_t<decltype(make(std::size_t{0}))>;
                return std::array<T, N>{{make(J)...}};
            }(std::make_index_sequence<N>{});
        }

        template <class T, std::size_t N>
        using vec = containers::inplace_vector<T, N>;

        struct map_fn {
            template <class F, std_array A>
            constexpr auto operator()(F&& f, A&& xs) const
                SNICHOLLS_FN_RETURNS(functor_instance<std::remove_cvref_t<A>>::fmap(std::forward<F>(f), std::forward<A>(xs)))
        };

        struct filter_fn {
            template <class P, std_array A>
            constexpr vec<elem_t<A>, size_v<A>> operator()(P&& p, const A& xs) const
            {
                vec<elem_t<A>, size_v<A>> out;
                for (const auto& x : xs)
                    if (std::invoke(p, x)) out.push_back(x);
                return out;
            }
        };

        struct foldl1_fn {
            template <class F, std_array A>
                requires(size_v<A> > 0)
            constexpr elem_t<A> operator()(F&& f, const A& xs) const
            {
                elem_t<A> acc = xs[0];
                for (std::size_t i = 1; i < size_v<A>; ++i) acc = std::invoke(f, std::move(acc), xs[i]);
                return acc;
            }
        };

        struct foldr1_fn {
            template <class F, std_array A>
                requires(size_v<A> > 0)
            constexpr elem_t<A> operator()(F&& f, const A& xs) const
            {
                elem_t<A> acc = xs[size_v<A> - 1];
                for (std::size_t i = size_v<A> - 1; i-- > 0;) acc = std::invoke(f, xs[i], std::move(acc));
                return acc;
            }
        };

        struct scanl_fn {
            template <class F, class Z, std_array A>
            constexpr auto operator()(F&& f, Z&& z, const A& xs) const
                SNICHOLLS_FN_RETURNS(snicholls::fold_list(f, std::remove_cvref_t<Z>(std::forward<Z>(z)), xs))
        };

        struct scanr_fn {
            template <class F, class Z, std_array A, class V = std::remove_cvref_t<Z>>
            constexpr std::array<V, size_v<A> + 1> operator()(F&& f, Z&& z, const A& xs) const
            {
                std::array<V, size_v<A> + 1> out{};
                out[size_v<A>] = std::forward<Z>(z);
                for (std::size_t i = size_v<A>; i-- > 0;) out[i] = std::invoke(f, xs[i], std::as_const(out[i + 1]));
                return out;
            }
        };

        struct zip_with_fn {
            template <class F, std_array A, std_array B, std::size_t N = std::min(size_v<A>, size_v<B>)>
            constexpr auto operator()(F&& f, const A& a, const B& b) const
            {
                return detail::generate<N>([&](std::size_t i) { return std::invoke(f, a[i], b[i]); });
            }
        };

        struct zip_fn {
            template <std_array A, std_array B>
            constexpr auto operator()(const A& a, const B& b) const
            {
                return zip_with_fn{}([](const auto& x, const auto& y) { return std::pair{x, y}; }, a, b);
            }
        };

        struct zip3_fn {
            template <std_array A, std_array B, std_array C, std::size_t N = std::min({size_v<A>, size_v<B>, size_v<C>})>
            constexpr auto operator()(const A& a, const B& b, const C& c) const
            {
                return detail::generate<N>([&](std::size_t i) { return std::tuple{a[i], b[i], c[i]}; });
            }
        };

        struct unzip_fn {
            template <std_array A>
            constexpr auto operator()(const A& ps) const
            {
                return std::pair{detail::generate<size_v<A>>([&](std::size_t i) { return std::get<0>(ps[i]); }),
                                 detail::generate<size_v<A>>([&](std::size_t i) { return std::get<1>(ps[i]); })};
            }
        };

        struct concat_fn {
            template <std_array A, std::size_t M = size_v<elem_t<A>>>
            constexpr auto operator()(const A& xss) const
            {
                return detail::generate<size_v<A> * M>([&](std::size_t i) { return xss[i / M][i % M]; });
            }
        };

        struct concat_map_fn {
            template <class F, std_array A>
            constexpr auto operator()(F&& f, A&& xs) const
                SNICHOLLS_FN_RETURNS(monad_instance<std::remove_cvref_t<A>>::bind(std::forward<A>(xs), std::forward<F>(f)))
        };

        struct reverse_fn {
            template <std_array A>
            constexpr auto operator()(const A& xs) const
            {
                return detail::generate<size_v<A>>([&](std::size_t i) { return xs[size_v<A> - 1 - i]; });
            }
        };

        struct intersperse_fn {
            template <class X, std_array A, std::size_t N = size_v<A>>
            constexpr auto operator()(const X& x, const A& xs) const
            {
                return detail::generate<(N == 0 ? 0 : 2 * N - 1)>(
                    [&](std::size_t i) { return i % 2 == 0 ? xs[i / 2] : elem_t<A>(x); });
            }
        };

        struct intercalate_fn {
            template <std_array S, std_array A, std::size_t N = size_v<A>, std::size_t M = size_v<elem_t<A>>,
                      std::size_t K = size_v<S>>
            constexpr auto operator()(const S& sep, const A& xss) const
            {
                constexpr std::size_t stride = M + K;
                return detail::generate<(N == 0 ? 0 : N * M + (N - 1) * K)>([&](std::size_t i) {
                    const std::size_t at = i % stride;
                    return at < M ? xss[i / stride][at] : sep[at - M];
                });
            }
        };

        struct span_fn {
            template <class P, std_array A>
            constexpr auto operator()(P&& p, const A& xs) const
            {
                std::pair<vec<elem_t<A>, size_v<A>>, vec<elem_t<A>, size_v<A>>> out;
                std::size_t i = 0;
                for (; i < size_v<A> && std::invoke(p, xs[i]); ++i) out.first.push_back(xs[i]);
                for (; i < size_v<A>; ++i) out.second.push_back(xs[i]);
                return out;
            }
        };

        struct break_fn {
            template <class P, std_array A>
            constexpr auto operator()(P&& p, const A& xs) const
            {
                return span_fn{}([&](const auto& x) { return !std::invoke(p, x); }, xs);
            }
        };

        struct group_by_fn {
            template <class Eq, std_array A, std::size_t N = size_v<A>>
            constexpr vec<vec<elem_t<A>, N>, N> operator()(Eq&& eq, const A& xs) const
            {
                vec<vec<elem_t<A>, N>, N> out;
                for (std::size_t i = 0; i < N;) {
                    vec<elem_t<A>, N> g;
                    g.push_back(xs[i]);
                    std::size_t j = i + 1;
                    for (; j < N && std::invoke(eq, xs[i], xs[j]); ++j) g.push_back(xs[j]);
                    out.push_back(g);
                    i = j;
                }
                return out;
            }
        };

        struct group_fn {
            template <std_array A>
            constexpr auto operator()(const A& xs) const SNICHOLLS_FN_RETURNS(group_by_fn{}(std::equal_to<>{}, xs))
        };

        struct partition_fn {
            template <class P, std_array A>
            constexpr auto operator()(P&& p, const A& xs) const
            {
                std::pair<vec<elem_t<A>, size_v<A>>, vec<elem_t<A>, size_v<A>>> out;
                for (const auto& x : xs) (std::invoke(p, x) ? out.first : out.second).push_back(x);
                return out;
            }
        };

        struct nub_fn {
            template <std_array A>
            constexpr vec<elem_t<A>, size_v<A>> operator()(const A& xs) const
            {
                vec<elem_t<A>, size_v<A>> out;
                for (const auto& x : xs)
                    if (std::find(out.begin(), out.end(), x) == out.end()) out.push_back(x);
                return out;
            }
        };

        struct sort_by_fn {
            template <class Less, std_array A>
            constexpr std::remove_cvref_t<A> operator()(Less&& less, A xs) const
            {
                for (std::size_t i = 1; i < size_v<A>; ++i)
                    for (std::size_t j = i; j > 0 && std::invoke(less, xs[j], xs[j - 1]); --j) std::swap(xs[j], xs[j - 1]);
                return xs;
            }
        };

        struct sort_on_fn {
            template <class Key, std_array A>
            constexpr auto operator()(Key&& key, const A& xs) const
            {
                return sort_by_fn{}([&](const auto& a, const auto& b) { return std::invoke(key, a) < std::invoke(key, b); }, xs);
            }
        };

        struct sort_fn {
            template <std_array A>
            constexpr auto operator()(const A& xs) const SNICHOLLS_FN_RETURNS(sort_by_fn{}(std::less<>{}, xs))
        };

        struct transpose_fn {
            template <std_array A, std::size_t N = size_v<A>, std::size_t M = size_v<elem_t<A>>>
            constexpr auto operator()(const A& xss) const
            {
                return detail::generate<M>([&](std::size_t j) {
                    return detail::generate<N>([&](std::size_t i) { return xss[i][j]; });
                });
            }
        };

        struct subsequences_fn {
            template <std_array A, std::size_t N = size_v<A>>
                requires(N <= 20)
            constexpr auto operator()(const A& xs) const
            {
                std::array<vec<elem_t<A>, N>, (std::size_t{1} << N)> out{};
                // Haskell's order: by the binary counter, lowest element varying fastest.
                for (std::size_t mask = 0; mask < out.size(); ++mask)
                    for (std::size_t i = 0; i < N; ++i)
                        if (mask >> i & 1U) out[mask].push_back(xs[i]);
                return out;
            }
        };

        struct permutations_fn {
            template <std_array A>
            constexpr auto operator()(const A& xs) const SNICHOLLS_FN_RETURNS(snicholls::permutations(xs))
        };

        // ---- over any range ---------------------------------------------------------------------------------------

        struct lookup_fn {
            template <class K, std::ranges::input_range R>
            constexpr auto operator()(const K& k, R&& pairs) const
            {
                using V = std::remove_cvref_t<decltype(std::get<1>(*std::ranges::begin(pairs)))>;
                for (const auto& p : pairs)
                    if (std::get<0>(p) == k) return std::optional<V>(std::get<1>(p));
                return std::optional<V>();
            }
        };

        struct elem_fn {
            template <class X, std::ranges::input_range R>
            constexpr bool operator()(const X& x, R&& xs) const
            {
                for (const auto& y : xs)
                    if (y == x) return true;
                return false;
            }
        };

        struct find_fn {
            template <class P, std::ranges::input_range R>
            constexpr std::optional<range_value_t<R>> operator()(P&& p, R&& xs) const
            {
                for (const auto& x : xs)
                    if (std::invoke(p, x)) return x;
                return std::nullopt;
            }
        };

        struct find_index_fn {
            template <class P, std::ranges::input_range R>
            constexpr std::optional<std::size_t> operator()(P&& p, R&& xs) const
            {
                std::size_t i = 0;
                for (const auto& x : xs) {
                    if (std::invoke(p, x)) return i;
                    ++i;
                }
                return std::nullopt;
            }
        };

        struct all_fn {
            template <class P, std::ranges::input_range R>
            constexpr bool operator()(P&& p, R&& xs) const
            {
                for (const auto& x : xs)
                    if (!std::invoke(p, x)) return false;
                return true;
            }
        };

        struct any_fn {
            template <class P, std::ranges::input_range R>
            constexpr bool operator()(P&& p, R&& xs) const
            {
                for (const auto& x : xs)
                    if (std::invoke(p, x)) return true;
                return false;
            }
        };

        struct length_fn {
            template <std::ranges::input_range R>
            constexpr std::size_t operator()(R&& xs) const
            {
                std::size_t n = 0;
                for (auto it = std::ranges::begin(xs); it != std::ranges::end(xs); ++it) ++n;
                return n;
            }
            template <tuple_like T>
                requires(!std::ranges::input_range<T>)
            constexpr std::size_t operator()(const T&) const noexcept
            {
                return std::tuple_size_v<std::remove_cvref_t<T>>;
            }
            template <class T>
            constexpr std::size_t operator()(const std::optional<T>& o) const noexcept
            {
                return o.has_value() ? 1 : 0;
            }
        };

        struct null_fn {
            template <class S>
            constexpr auto operator()(S&& s) const SNICHOLLS_FN_RETURNS(length_fn{}(std::forward<S>(s)) == 0)
        };

        struct sum_fn {
            template <std::ranges::input_range R, class V = range_value_t<R>>
            constexpr V operator()(R&& xs) const
            {
                V acc = V(0);
                for (const auto& x : xs) acc = acc + x;
                return acc;
            }
        };

        struct product_fn {
            template <std::ranges::input_range R, class V = range_value_t<R>>
            constexpr V operator()(R&& xs) const
            {
                V acc = V(1);
                for (const auto& x : xs) acc = acc * x;
                return acc;
            }
        };

        template <bool Max>
        struct extreme_fn {
            template <std::ranges::input_range R, class V = range_value_t<R>>
            constexpr V operator()(R&& xs) const
            {
                auto it = std::ranges::begin(xs);
                assert(it != std::ranges::end(xs));
                V best = *it;
                for (++it; it != std::ranges::end(xs); ++it)
                    if (Max ? best < *it : *it < best) best = *it;
                return best;
            }
        };

        struct fold_map_fn {
            template <class F, std::ranges::input_range R,
                      class M = std::remove_cvref_t<std::invoke_result_t<F&, std::ranges::range_reference_t<R>>>>
                requires monoid<M>
            constexpr M operator()(F&& f, R&& xs) const
            {
                M acc = mempty<M>;
                for (auto&& x : xs) acc = fn::combine(acc, std::invoke(f, x));
                return acc;
            }
        };

        struct fst_fn {
            template <class P>
            constexpr auto operator()(P&& p) const SNICHOLLS_FN_RETURNS(std::remove_cvref_t<decltype(std::get<0>(std::forward<P>(p)))>(std::get<0>(std::forward<P>(p))))
        };

        struct snd_fn {
            template <class P>
            constexpr auto operator()(P&& p) const SNICHOLLS_FN_RETURNS(std::remove_cvref_t<decltype(std::get<1>(std::forward<P>(p)))>(std::get<1>(std::forward<P>(p))))
        };

        struct map_tuple_fn {
            template <class F, tuple_like T>
            constexpr auto operator()(F&& f, T&& t) const
                SNICHOLLS_FN_RETURNS(detail::map_elements(f, std::forward<T>(t), tuple_indices<T>{}))
        };

    } // namespace detail

    /// \brief `map(f, xs)`: `f` of every element of a `std::array`, into an array of the same length.
    /// \ingroup core
    inline constexpr detail::map_fn map{};
    /// \brief `filter(p, xs)`: the elements for which `p` holds, in order, in an `inplace_vector<T, N>`.
    /// \ingroup core
    inline constexpr detail::filter_fn filter{};
    /// \brief `foldl1(f, xs)`: `foldl` from the first element; the array must not be empty.
    /// \ingroup core
    inline constexpr detail::foldl1_fn foldl1{};
    /// \brief `foldr1(f, xs)`: `foldr` from the last element; the array must not be empty.
    /// \ingroup core
    inline constexpr detail::foldr1_fn foldr1{};
    /// \brief `scanl(f, z, xs)`: `{z, f(z, x0), f(f(z, x0), x1), ...}`, `N + 1` values (`snicholls::fold_list`).
    /// \ingroup core
    inline constexpr detail::scanl_fn scanl{};
    /// \brief `scanr(f, z, xs)`: `{f(x0, f(x1, ... z)), ..., f(x_last, z), z}`, `N + 1` values.
    /// \ingroup core
    inline constexpr detail::scanr_fn scanr{};
    /// \brief `zip_with(f, a, b)`: `f(a_i, b_i)` for the length of the shorter.
    /// \ingroup core
    inline constexpr detail::zip_with_fn zip_with{};
    /// \brief `zip(a, b)`: `std::pair{a_i, b_i}` for the length of the shorter.
    /// \ingroup core
    inline constexpr detail::zip_fn zip{};
    /// \brief `zip3(a, b, c)`: `std::tuple{a_i, b_i, c_i}` for the length of the shortest.
    /// \ingroup core
    inline constexpr detail::zip3_fn zip3{};
    /// \brief `unzip(ps)`: an array of pairs as a pair of arrays.
    /// \ingroup core
    inline constexpr detail::unzip_fn unzip{};
    /// \brief `concat(xss)`: an array of `N` arrays of `M` as one array of `N * M`.
    /// \ingroup core
    inline constexpr detail::concat_fn concat{};
    /// \brief `concat_map(f, xs)`: `concat(map(f, xs))`, `f` returning arrays of one length; the array monad's bind.
    /// \ingroup core
    inline constexpr detail::concat_map_fn concat_map{};
    /// \brief `reverse(xs)`: the elements in the opposite order.
    /// \ingroup core
    inline constexpr detail::reverse_fn reverse{};
    /// \brief `intersperse(x, xs)`: `x` between every two elements, `2N - 1` values.
    /// \ingroup core
    inline constexpr detail::intersperse_fn intersperse{};
    /// \brief `intercalate(sep, xss)`: the arrays of `xss` joined with the array `sep` between them.
    /// \ingroup core
    inline constexpr detail::intercalate_fn intercalate{};
    /// \brief `span(p, xs)`: the longest prefix for which `p` holds, and the rest.
    /// \ingroup core
    inline constexpr detail::span_fn span{};
    /// \brief `break_(p, xs)`: Haskell's `break`, the longest prefix for which `p` fails, and the rest.
    /// \ingroup core
    inline constexpr detail::break_fn break_{};
    /// \brief `group_by(eq, xs)`: runs of adjacent elements equal (by `eq`) to the run's first.
    /// \ingroup core
    inline constexpr detail::group_by_fn group_by{};
    /// \brief `group(xs)`: runs of adjacent equal elements.
    /// \ingroup core
    inline constexpr detail::group_fn group{};
    /// \brief `partition(p, xs)`: the elements for which `p` holds, and the others, each in order.
    /// \ingroup core
    inline constexpr detail::partition_fn partition{};
    /// \brief `nub(xs)`: the elements without repeats, first occurrences kept, in order.
    /// \ingroup core
    inline constexpr detail::nub_fn nub{};
    /// \brief `sort_by(less, xs)`: a stable sort by `less`.
    /// \ingroup core
    inline constexpr detail::sort_by_fn sort_by{};
    /// \brief `sort_on(key, xs)`: a stable sort by `key(x)`.
    /// \ingroup core
    inline constexpr detail::sort_on_fn sort_on{};
    /// \brief `sort(xs)`: a stable sort by `<`.
    /// \ingroup core
    inline constexpr detail::sort_fn sort{};
    /// \brief `transpose(xss)`: an array of `N` arrays of `M` as `M` arrays of `N`.
    /// \ingroup core
    inline constexpr detail::transpose_fn transpose{};
    /// \brief `subsequences(xs)`: all `2^N` sub-lists, in Haskell's order, for `N` up to 20.
    /// \ingroup core
    inline constexpr detail::subsequences_fn subsequences{};
    /// \brief `permutations(xs)`: all `N!` orderings (`snicholls::permutations`, lexicographic by position).
    /// \ingroup core
    inline constexpr detail::permutations_fn permutations{};
    /// \brief `lookup(k, pairs)`: the second of the first pair whose first is `k`, if any (OCaml's `List.assoc_opt`).
    /// \ingroup core
    inline constexpr detail::lookup_fn lookup{};
    /// \brief `elem(x, xs)`: whether `x` is an element (OCaml's `List.mem`).
    /// \ingroup core
    inline constexpr detail::elem_fn elem{};
    /// \brief `find(p, xs)`: the first element for which `p` holds, if any.
    /// \ingroup core
    inline constexpr detail::find_fn find{};
    /// \brief `find_index(p, xs)`: the position of the first element for which `p` holds, if any.
    /// \ingroup core
    inline constexpr detail::find_index_fn find_index{};
    /// \brief `all(p, xs)`: whether `p` holds of every element (true for none).
    /// \ingroup core
    inline constexpr detail::all_fn all{};
    /// \brief `any(p, xs)`: whether `p` holds of some element (false for none).
    /// \ingroup core
    inline constexpr detail::any_fn any{};
    /// \brief `length(xs)`: the number of elements of a range, a tuple's size, or 0 / 1 for an optional (Foldable).
    /// \ingroup core
    inline constexpr detail::length_fn length{};
    /// \brief `null(xs)`: whether `length(xs) == 0`.
    /// \ingroup core
    inline constexpr detail::null_fn null{};
    /// \brief `sum(xs)`: the elements added, from 0.
    /// \ingroup core
    inline constexpr detail::sum_fn sum{};
    /// \brief `product(xs)`: the elements multiplied, from 1.
    /// \ingroup core
    inline constexpr detail::product_fn product{};
    /// \brief `maximum(xs)`: the largest element, the first of equals. \pre `xs` is not empty.
    /// \ingroup core
    inline constexpr detail::extreme_fn<true> maximum{};
    /// \brief `minimum(xs)`: the smallest element, the first of equals. \pre `xs` is not empty.
    /// \ingroup core
    inline constexpr detail::extreme_fn<false> minimum{};
    /// \brief `fold_map(f, xs)`: `f` of every element combined in `f`'s monoid, from `mempty` (Foldable's `foldMap`).
    /// \ingroup core
    inline constexpr detail::fold_map_fn fold_map{};
    /// \brief `fst(p)`: the first element of a pair or tuple, as a value.
    /// \ingroup core
    inline constexpr detail::fst_fn fst{};
    /// \brief `snd(p)`: the second element of a pair or tuple, as a value.
    /// \ingroup core
    inline constexpr detail::snd_fn snd{};
    /// \brief `map_tuple(f, t)`: `f` of every element of a tuple, pair or array, as a `std::tuple`.
    /// \ingroup core
    inline constexpr detail::map_tuple_fn map_tuple{};

    /// \brief `take<K>(xs)`: the first `K` elements (`K` at most `N`).
    /// \ingroup core
    /// \tparam K How many.
    /// \param xs The array.
    /// \return An `array<T, K>`.
    template <std::size_t K, detail::std_array A>
        requires(K <= detail::size_v<A>)
    [[nodiscard]] constexpr auto take(const A& xs)
    {
        return detail::generate<K>([&](std::size_t i) { return xs[i]; });
    }

    /// \brief `drop<K>(xs)`: the elements after the first `K` (`K` at most `N`).
    /// \ingroup core
    /// \tparam K How many to skip.
    /// \param xs The array.
    /// \return An `array<T, N - K>`.
    template <std::size_t K, detail::std_array A>
        requires(K <= detail::size_v<A>)
    [[nodiscard]] constexpr auto drop(const A& xs)
    {
        return detail::generate<detail::size_v<A> - K>([&](std::size_t i) { return xs[K + i]; });
    }

    /// \brief `split_at<K>(xs)`: `std::pair{take<K>(xs), drop<K>(xs)}`.
    /// \ingroup core
    /// \tparam K Where to split.
    /// \param xs The array.
    /// \return The two arrays.
    template <std::size_t K, detail::std_array A>
        requires(K <= detail::size_v<A>)
    [[nodiscard]] constexpr auto split_at(const A& xs)
    {
        return std::pair{fn::take<K>(xs), fn::drop<K>(xs)};
    }

    /// \brief `replicate<N>(x)`: `N` copies of `x`.
    /// \ingroup core
    /// \tparam N How many.
    /// \param x The value.
    /// \return An `array<T, N>`.
    template <std::size_t N, class T>
    [[nodiscard]] constexpr std::array<T, N> replicate(const T& x)
    {
        return detail::generate<N>([&](std::size_t) { return x; });
    }

} // namespace snicholls::fn

#endif /* snicholls_core_functional_list_hpp */
