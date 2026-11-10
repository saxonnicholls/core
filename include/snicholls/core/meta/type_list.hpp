//
//  type_list.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::meta::type_list` and the compile-time algorithms over type lists.
/// \ingroup core
///
/// snicholls::meta::type_list - one compile-time type list for the whole library (std only, no recursion limits in
/// the common operations: they expand packs rather than recurse element by element).
///
/// Consolidated from earlier type-list and cross-product implementations, with these fixes:
/// - take/drop on an empty list with N == 0 was an ambiguous specialisation (hard error);
/// - sort was not stable (equal elements swapped) - now a stable merge sort;
/// - unique kept the *last* occurrence despite saying "keep first" - now keeps the first;
/// - zip of unequal lengths was a hard error - now truncates to the shorter list, like Python's zip;
/// - merge/'s cross_product zipped instead of forming the product - cartesian_product is n-ary and correct.
/// String sets need no special machinery: wrap values with meta::constant\<V> (equal values are the same type),
/// e.g. constant\<snicholls::fixed_string{"usd"}>, and use the ordinary set operations.
///
/// Conventions (Boost.MP11 style):
/// - metafunction arguments are templates used directly: transform_t\<L, std::add_pointer_t>;
/// - predicates are traits with ::value: filter_t\<L, std::is_integral>;
/// - binary functions (folds, zip_with) are templates used directly: fold_left_t\<L, Init, F> applies F\<Acc, T>;
/// - results are type_list<...>; indices out of range are clamped where that is meaningful (take/drop/erase).
///
/// \code
///   size_v empty_v at_t front_t back_t                    contains_v count_v index_of_v (npos)
///   push_front_t push_back_t pop_front_t pop_back_t       concat_t (variadic) reverse_t flatten_t (one level)
///   take_t drop_t erase_at_t insert_at_t replace_at_t     replace_t replace_if_t remove_t remove_if_t filter_t
///   transform_t fold_left_t fold_right_t                   any_of_v all_of_v none_of_v
///   unique_t set_union_t set_intersection_t set_difference_t set_symmetric_difference_t
///   zip_t zip_with_t unzip<L>::first/second partition<L, P>::first/second
///   sort_t (stable) group_by_t group_adjacent_t group_adjacent_by_key_t topological_sort_t
///   cartesian_product_t<Ls...> product_with_t<F, Ls...> cartesian_power_t<L, N>
///   outer_product_t<Ls...> outer_with_t<F, Ls...>         dot_product_t<Ls...> dot_with_t<F, Ls...>
///   thread_t<f<Args...>, Head, Spec> thread_with_t<F, Args...>     permutations_t<L, R> permutations_with_t<F, L, R>
///   partition_n_t<L, N, D> (Wolfram's Partition)      find_if_t<L, P, Default> find_index_if_v<L, P>
///   subsets_t<L> subsets_of_size_t<L, K> inner_t<F, L1, L2, G> fold_list_t<L, Init, F> nest_list_t<F, X, N> nest_t
///   range_t<N> map_indexed_t<L, F> rotate_left_t rotate_right_t riffle_t<L, S> tally_t<L>
///   rename_t<L, Template> as_type_list_t<Template<Ts...>>  for_each_type<L>(f) for_each_tag<L>(f) make_array<L>(f)
/// \endcode

#ifndef snicholls_core_meta_type_list_hpp
#define snicholls_core_meta_type_list_hpp

#include <snicholls/core/combinatorics/permutations.hpp>
#include <snicholls/core/combinatorics/subsets.hpp>

#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace snicholls::meta {

/// \brief One compile-time type list for the whole library, with the algorithms over it.
/// \ingroup core
///
/// \overview
/// `type_list<Ts...>` and its algorithms (access, search, transform, filter, fold, set operations, zip, stable sort, grouping, topological sort, cartesian products) as aliases and constants. The operations expand packs rather than recursing element by element, so there are no recursion limits in the common operations. The origin and the full list are in the file description above.
///
/// \relations
/// - Used for enum and object reflection in consuming libraries; `snicholls::meta::for_each_type` and `make_array` bridge to run time.
/// - Module: \ref core.
///
/// \impls
/// One class template; every algorithm is an alias (`*_t`) or constant (`*_v`).
///
/// \contracts
/// - Compile time only; `for_each_type` is `noexcept` when the callable is.
///
/// \errors
/// Misuse (index out of range, a cycle in a topological sort) is a compile error.
///
/// \examples
/// \snippet type_list_demo.cpp demo
///
/// \perf
/// Not benchmarked.
///
/// \testing
/// - Tests: tests/meta/type_list_test.cpp.
/// - Demo: demos/meta/type_list_demo.cpp.
///
/// \deps
/// CMake target `snicholls::core`; standard library only.
///
/// \gotchas
/// None known.
///
/// \status
/// Stable.
///
/// \tparam Ts the types.
template <typename... Ts>
struct type_list
{
    /// \brief The number of elements.
    static constexpr std::size_t size = sizeof...(Ts);
};

/// \brief "Not found" for `index_of_v`.
inline constexpr std::size_t npos = static_cast<std::size_t>(-1);

/// \brief A value as a type: equal values give the same type, so values.
///
/// (numbers, fixed strings) can live in type lists.
template <auto V>
struct constant
{
    /// \brief The value.
    static constexpr auto value = V;
};

template <typename T>
struct is_type_list : std::false_type {};
template <typename... Ts>
struct is_type_list<type_list<Ts...>> : std::true_type {};
/// \brief Whether `T` is a `type_list` (compile-time constant).
template <typename T>
inline constexpr bool is_type_list_v = is_type_list<T>::value;

// --- conversion to and from other variadic templates (std::tuple, std::variant, ...) ---------------------------

template <typename List, template <typename...> class Target>
struct rename;
/// \brief The rename<.
template <template <typename...> class Source, typename... Ts, template <typename...> class Target>
struct rename<Source<Ts...>, Target>
{
    /// \brief The type.
    using type = Target<Ts...>;
};
/// \brief `Source<Ts...>` re-instantiated as `Target<Ts...>`.
template <typename List, template <typename...> class Target>
using rename_t = typename rename<List, Target>::type;

/// \brief Any pack template `X<Ts...>` as `type_list<Ts...>`.
template <typename Variadic>
using as_type_list_t = rename_t<Variadic, type_list>;

// --- size and element access ---------------------------------------------------------------------------------------

/// \brief The number of types in `List`.
template <typename List>
inline constexpr std::size_t size_v = List::size;
/// \brief Whether `List` is empty.
template <typename List>
inline constexpr bool empty_v = List::size == 0;

namespace detail {

template <std::size_t I, typename T>
struct indexed
{
    using type = T;
};

template <typename Seq, typename... Ts>
struct indexed_set;
template <std::size_t... Is, typename... Ts>
struct indexed_set<std::index_sequence<Is...>, Ts...> : indexed<Is, Ts>... {};

template <std::size_t I, typename T>
indexed<I, T> select(const indexed<I, T>&);   // unevaluated only: deduces T from the unique base for I

template <std::size_t I, typename List>
struct at_impl;
template <std::size_t I, typename... Ts>
struct at_impl<I, type_list<Ts...>>
{
    static_assert(I < sizeof...(Ts), "meta::at_t: index out of range");
    using type =
        typename decltype(detail::select<I>(std::declval<indexed_set<std::index_sequence_for<Ts...>, Ts...>>()))::type;
};

} // namespace detail

/// \brief The type at index `I`.
template <typename List, std::size_t I>
using at_t = typename detail::at_impl<I, List>::type;
/// \brief The first type.
template <typename List>
using front_t = at_t<List, 0>;
/// \brief The last type.
template <typename List>
using back_t = at_t<List, size_v<List> - 1>;

// --- queries ---------------------------------------------------------------------------------------------------------

namespace detail {

template <typename List, typename T>
struct count_impl;
template <typename... Ts, typename T>
struct count_impl<type_list<Ts...>, T> : std::integral_constant<std::size_t, (std::size_t{0} + ... + std::is_same_v<Ts, T>)> {};

template <typename T, typename... Ts>
consteval std::size_t index_of_impl(type_list<Ts...>) noexcept
{
    constexpr bool matches[]{std::is_same_v<T, Ts>..., false};
    for (std::size_t i = 0; i < sizeof...(Ts); ++i) {
        if (matches[i]) {
            return i;
        }
    }
    return npos;
}

template <typename List, template <typename> class Pred>
struct predicate_count;
template <typename... Ts, template <typename> class Pred>
struct predicate_count<type_list<Ts...>, Pred>
    : std::integral_constant<std::size_t, (std::size_t{0} + ... + (Pred<Ts>::value ? 1 : 0))> {};

} // namespace detail

/// \brief How many times `T` occurs in `List`.
template <typename List, typename T>
inline constexpr std::size_t count_v = detail::count_impl<List, T>::value;
/// \brief Whether `T` occurs in `List`.
template <typename List, typename T>
inline constexpr bool contains_v = count_v<List, T> != 0;
/// \brief The first index of `T` in `List`, or `npos`.
template <typename List, typename T>
inline constexpr std::size_t index_of_v = detail::index_of_impl<T>(List{});   // npos when absent

/// \brief Whether `Pred` holds for some type.
template <typename List, template <typename> class Pred>
inline constexpr bool any_of_v = detail::predicate_count<List, Pred>::value != 0;
/// \brief Whether `Pred` holds for every type.
template <typename List, template <typename> class Pred>
inline constexpr bool all_of_v = detail::predicate_count<List, Pred>::value == size_v<List>;
/// \brief Whether `Pred` holds for no type.
template <typename List, template <typename> class Pred>
inline constexpr bool none_of_v = !any_of_v<List, Pred>;

// --- building ---------------------------------------------------------------------------------------------------------

namespace detail {

template <typename... Lists>
struct concat_impl
{
    using type = type_list<>;
};
template <typename... Ts>
struct concat_impl<type_list<Ts...>>
{
    using type = type_list<Ts...>;
};
template <typename... Ts, typename... Us, typename... Rest>
struct concat_impl<type_list<Ts...>, type_list<Us...>, Rest...> : concat_impl<type_list<Ts..., Us...>, Rest...> {};

template <typename List, typename T>
struct push_front_impl;
template <typename... Ts, typename T>
struct push_front_impl<type_list<Ts...>, T>
{
    using type = type_list<T, Ts...>;
};

template <typename List, typename T>
struct push_back_impl;
template <typename... Ts, typename T>
struct push_back_impl<type_list<Ts...>, T>
{
    using type = type_list<Ts..., T>;
};

template <typename List>
struct pop_front_impl;
template <typename T, typename... Ts>
struct pop_front_impl<type_list<T, Ts...>>
{
    using type = type_list<Ts...>;
};

template <typename List, typename Seq>
struct pick;
template <typename List, std::size_t... Is>
struct pick<List, std::index_sequence<Is...>>
{
    using type = type_list<at_t<List, Is>...>;
};

template <std::size_t Offset, std::size_t... Is>
constexpr std::index_sequence<(Offset + Is)...> shift(std::index_sequence<Is...>) noexcept;

// The elements at [Begin, End) of List; End is clamped to the size.
template <typename List, std::size_t Begin, std::size_t End>
struct slice
{
    static constexpr std::size_t end = End < List::size ? End : List::size;
    static constexpr std::size_t begin = Begin < end ? Begin : end;
    using type = typename pick<List, decltype(shift<begin>(std::make_index_sequence<end - begin>{}))>::type;
};

} // namespace detail

/// \brief The lists concatenated.
template <typename... Lists>
using concat_t = typename detail::concat_impl<Lists...>::type;
/// \brief `List` with `Ts...` prepended.
template <typename List, typename T>
using push_front_t = typename detail::push_front_impl<List, T>::type;
/// \brief `List` with `Ts...` appended.
template <typename List, typename T>
using push_back_t = typename detail::push_back_impl<List, T>::type;
/// \brief `List` without its first type.
template <typename List>
using pop_front_t = typename detail::pop_front_impl<List>::type;

/// \brief First N elements / all but the first N (N clamped to the size).
/// \ingroup core
template <typename List, std::size_t N>
using take_t = typename detail::slice<List, 0, N>::type;
/// \brief `List` without its first `N` types.
template <typename List, std::size_t N>
using drop_t = typename detail::slice<List, N, List::size>::type;
/// \brief `List` without its last type.
template <typename List>
using pop_back_t = take_t<List, (List::size == 0 ? 0 : List::size - 1)>;

/// \brief `List` without the type at index `I`.
template <typename List, std::size_t I>
using erase_at_t = concat_t<take_t<List, I>, drop_t<List, I + 1>>;
/// \brief `List` with `T` inserted at index `I`.
template <typename List, std::size_t I, typename T>
using insert_at_t = concat_t<take_t<List, I>, type_list<T>, drop_t<List, I>>;
/// \brief `List` with the type at index `I` replaced by `T`.
template <typename List, std::size_t I, typename T>
using replace_at_t = concat_t<take_t<List, I>, type_list<T>, drop_t<List, I + 1>>;

// --- element-wise ------------------------------------------------------------------------------------------------------

namespace detail {

template <typename List, template <typename...> class F>
struct transform_impl;
template <typename... Ts, template <typename...> class F>
struct transform_impl<type_list<Ts...>, F>
{
    using type = type_list<F<Ts>...>;
};

template <typename List, template <typename> class Pred, bool Keep>
struct filter_impl;
template <typename... Ts, template <typename> class Pred, bool Keep>
struct filter_impl<type_list<Ts...>, Pred, Keep>
{
    using type = concat_t<std::conditional_t<static_cast<bool>(Pred<Ts>::value) == Keep, type_list<Ts>, type_list<>>...>;
};

template <typename Old>
struct same_as_pred
{
    template <typename T>
    using fn = std::is_same<T, Old>;
};

template <typename List, template <typename> class Pred, template <typename...> class F>
struct replace_if_impl;
template <typename... Ts, template <typename> class Pred, template <typename...> class F>
struct replace_if_impl<type_list<Ts...>, Pred, F>
{
    template <typename T, bool Replace>
    struct one
    {
        using type = T;
    };
    template <typename T>
    struct one<T, true>
    {
        using type = F<T>;
    };
    using type = type_list<typename one<Ts, static_cast<bool>(Pred<Ts>::value)>::type...>;
};

template <typename New>
struct constant_fn
{
    template <typename>
    using fn = New;
};

template <typename T>
struct as_list
{
    using type = type_list<T>;
};
template <typename... Ts>
struct as_list<type_list<Ts...>>
{
    using type = type_list<Ts...>;
};

template <typename T>
using as_list_t = typename as_list<T>::type;

template <typename List>
struct reverse_impl;
template <typename... Ts>
struct reverse_impl<type_list<Ts...>>
{
    using list = type_list<Ts...>;
    template <std::size_t... Is>
    static type_list<at_t<list, sizeof...(Ts) - 1 - Is>...> make(std::index_sequence<Is...>);
    using type = decltype(make(std::index_sequence_for<Ts...>{}));
};

} // namespace detail

/// \brief Each type mapped through `F`.
template <typename List, template <typename...> class F>
using transform_t = typename detail::transform_impl<List, F>::type;
/// \brief The types for which `Pred` holds.
template <typename List, template <typename> class Pred>
using filter_t = typename detail::filter_impl<List, Pred, true>::type;
/// \brief The types for which `Pred` does not hold.
template <typename List, template <typename> class Pred>
using remove_if_t = typename detail::filter_impl<List, Pred, false>::type;
/// \brief `List` without any `T`.
template <typename List, typename T>
using remove_t = remove_if_t<List, detail::same_as_pred<T>::template fn>;
/// \brief Types for which `Pred` holds replaced by `T`.
template <typename List, template <typename> class Pred, template <typename...> class F>
using replace_if_t = typename detail::replace_if_impl<List, Pred, F>::type;
/// \brief Every `Old` replaced by `New`.
template <typename List, typename Old, typename New>
using replace_t = replace_if_t<List, detail::same_as_pred<Old>::template fn, detail::constant_fn<New>::template fn>;
/// \brief `List` reversed.
template <typename List>
using reverse_t = typename detail::reverse_impl<List>::type;
/// \brief One level: type_list\<a, type_list\<b, c>, type_list\<type_list\<d>>> -> type_list\<a, b, c.
///
/// Type_list\<d>>.
/// \ingroup core
template <typename List>
using flatten_t = rename_t<typename detail::transform_impl<List, detail::as_list_t>::type, concat_t>;

// --- folds ------------------------------------------------------------------------------------------------------------

namespace detail {

template <typename List, typename Acc, template <typename, typename> class F>
struct fold_left_impl
{
    using type = Acc;
};
template <typename T, typename... Ts, typename Acc, template <typename, typename> class F>
struct fold_left_impl<type_list<T, Ts...>, Acc, F> : fold_left_impl<type_list<Ts...>, F<Acc, T>, F> {};

template <typename List, typename Acc, template <typename, typename> class F>
struct fold_right_impl
{
    using type = Acc;
};
template <typename T, typename... Ts, typename Acc, template <typename, typename> class F>
struct fold_right_impl<type_list<T, Ts...>, Acc, F>
{
    using type = F<typename fold_right_impl<type_list<Ts...>, Acc, F>::type, T>;
};

} // namespace detail

/// \brief F\<F\<F\<Init, T0>, T1>, T2> ...
/// \ingroup core
template <typename List, typename Init, template <typename, typename> class F>
using fold_left_t = typename detail::fold_left_impl<List, Init, F>::type;
/// \brief F\<F\<F\<Init, Tn>, Tn-1>, ...>, T0>
/// \ingroup core
template <typename List, typename Init, template <typename, typename> class F>
using fold_right_t = typename detail::fold_right_impl<List, Init, F>::type;

// --- sets (order of first appearance is kept) ----------------------------------------------------------------------

namespace detail {

template <typename Acc, typename T>
using push_back_unique = std::conditional_t<contains_v<Acc, T>, Acc, push_back_t<Acc, T>>;

template <typename Set>
struct in_set
{
    template <typename T>
    using fn = std::bool_constant<contains_v<Set, T>>;
};

} // namespace detail

/// \brief `List` without duplicates (first occurrence kept).
template <typename List>
using unique_t = fold_left_t<List, type_list<>, detail::push_back_unique>;
/// \brief Types in either list (duplicates removed).
template <typename L1, typename L2>
using set_union_t = unique_t<concat_t<L1, L2>>;
/// \brief Types in both lists.
template <typename L1, typename L2>
using set_intersection_t = filter_t<unique_t<L1>, detail::in_set<L2>::template fn>;
namespace detail {

template <typename List, template <typename> class Pred>
struct find_index_if_impl;
template <typename... Ts, template <typename> class Pred>
struct find_index_if_impl<type_list<Ts...>, Pred>
{
    static consteval std::size_t find() noexcept
    {
        constexpr bool hits[]{static_cast<bool>(Pred<Ts>::value)..., false};
        for (std::size_t i = 0; i < sizeof...(Ts); ++i)
            if (hits[i])
                return i;
        return npos;
    }
    static constexpr std::size_t value = find();
};

template <typename List, std::size_t I, typename Default>
struct at_or
{
    using type = at_t<List, I>;
};
template <typename List, typename Default>
struct at_or<List, npos, Default>
{
    using type = Default;
};

} // namespace detail

/// \brief The index of the first type for which `Pred<T>::value` holds, or `npos`.
/// \ingroup core
template <typename List, template <typename> class Pred>
inline constexpr std::size_t find_index_if_v = detail::find_index_if_impl<List, Pred>::value;

/// \brief The first type for which `Pred<T>::value` holds, or `Default` when none does: unlike
/// `front_t<filter_t<L, P>>`, not an error when nothing matches.
/// \ingroup core
template <typename List, template <typename> class Pred, typename Default = void>
using find_if_t = typename detail::at_or<List, find_index_if_v<List, Pred>, Default>::type;

/// \brief Types in the first list but not the second.
template <typename L1, typename L2>
using set_difference_t = remove_if_t<unique_t<L1>, detail::in_set<L2>::template fn>;
/// \brief Types in exactly one of the lists.
template <typename L1, typename L2>
using set_symmetric_difference_t = concat_t<set_difference_t<L1, L2>, set_difference_t<L2, L1>>;

// --- zip / unzip / partition ---------------------------------------------------------------------------------------

namespace detail {

// F is variadic so that both std::pair and std::tuple bind without P0522 relaxed matching (off by default in clang < 19).
template <template <typename...> class F, typename L1, typename L2, typename Seq>
struct zip_impl;
template <template <typename...> class F, typename L1, typename L2, std::size_t... Is>
struct zip_impl<F, L1, L2, std::index_sequence<Is...>>
{
    using type = type_list<F<at_t<L1, Is>, at_t<L2, Is>>...>;
};

template <typename P>
using first_of = typename P::first_type;
template <typename P>
using second_of = typename P::second_type;

} // namespace detail

/// \brief type_list\<F\<A0, B0>, F\<A1, B1>, ...>, truncated to the shorter list.
/// \ingroup core
template <template <typename...> class F, typename L1, typename L2>
using zip_with_t = typename detail::zip_impl<F, L1, L2, std::make_index_sequence<(L1::size < L2::size ? L1::size : L2::size)>>::type;
/// \brief Pairs of corresponding types.
template <typename L1, typename L2>
using zip_t = zip_with_t<std::pair, L1, L2>;

/// \brief A list of pairs split into two lists.
template <typename Zipped>
struct unzip
{
    /// \brief The first list.
    using first = transform_t<Zipped, detail::first_of>;
    /// \brief The second list.
    using second = transform_t<Zipped, detail::second_of>;
};

/// \brief The types split by `Pred` into (holds, does not hold).
template <typename List, template <typename> class Pred>
struct partition
{
    using first = filter_t<List, Pred>;   ///< Pred true
    using second = remove_if_t<List, Pred>;   ///< Pred false
};

// --- sort (stable) ----------------------------------------------------------------------------------------------------

namespace detail {

template <typename A, typename B, template <typename, typename> class Less>
struct merge;
template <typename... Bs, template <typename, typename> class Less>
struct merge<type_list<>, type_list<Bs...>, Less>
{
    using type = type_list<Bs...>;
};
template <typename A, typename... As, template <typename, typename> class Less>
struct merge<type_list<A, As...>, type_list<>, Less>
{
    using type = type_list<A, As...>;
};
template <typename A, typename... As, typename B, typename... Bs, template <typename, typename> class Less>
struct merge<type_list<A, As...>, type_list<B, Bs...>, Less>
{
    // Take from the right only when strictly less: equal elements keep their order.
    using type = std::conditional_t<static_cast<bool>(Less<B, A>::value),
                                    push_front_t<typename merge<type_list<A, As...>, type_list<Bs...>, Less>::type, B>,
                                    push_front_t<typename merge<type_list<As...>, type_list<B, Bs...>, Less>::type, A>>;
};

template <typename List, template <typename, typename> class Less>
struct sort_impl
{
    static constexpr std::size_t half = List::size / 2;
    using type = typename merge<typename sort_impl<take_t<List, half>, Less>::type,
                                typename sort_impl<drop_t<List, half>, Less>::type, Less>::type;
};
template <template <typename, typename> class Less>
struct sort_impl<type_list<>, Less>
{
    using type = type_list<>;
};
template <typename T, template <typename, typename> class Less>
struct sort_impl<type_list<T>, Less>
{
    using type = type_list<T>;
};

} // namespace detail

/// \brief `List` stably sorted by the comparator.
template <typename List, template <typename, typename> class Less>
using sort_t = typename detail::sort_impl<List, Less>::type;

// --- grouping ---------------------------------------------------------------------------------------------------------

namespace detail {

template <template <typename, typename> class Equiv, typename H>
struct equiv_to
{
    template <typename T>
    using fn = std::bool_constant<static_cast<bool>(Equiv<H, T>::value)>;
};

template <typename List, template <typename, typename> class Equiv>
struct group_by_impl
{
    using type = type_list<>;
};
template <typename H, typename... Ts, template <typename, typename> class Equiv>
struct group_by_impl<type_list<H, Ts...>, Equiv>
{
    using parts = partition<type_list<Ts...>, equiv_to<Equiv, H>::template fn>;
    using type = push_front_t<typename group_by_impl<typename parts::second, Equiv>::type,
                              push_front_t<typename parts::first, H>>;
};

template <template <typename, typename> class Equiv>
struct adjacent
{
    template <typename Groups, typename T>
    struct step
    {
        using type = type_list<type_list<T>>;   // first element starts the first group
    };
    template <typename... Gs, typename T>
        requires(sizeof...(Gs) > 0)
    struct step<type_list<Gs...>, T>
    {
        using groups = type_list<Gs...>;
        using last = back_t<groups>;
        using type = std::conditional_t<static_cast<bool>(Equiv<front_t<last>, T>::value),
                                        push_back_t<pop_back_t<groups>, push_back_t<last, T>>,
                                        push_back_t<groups, type_list<T>>>;
    };
    template <typename Groups, typename T>
    using fn = typename step<Groups, T>::type;
};

template <template <typename> class Key>
struct same_key
{
    template <typename A, typename B>
    using fn = std::is_same<Key<A>, Key<B>>;
};

} // namespace detail

/// \brief Stable: every element equivalent to the first of a group joins it, wherever it appears.
/// \ingroup core
template <typename List, template <typename, typename> class Equiv>
using group_by_t = typename detail::group_by_impl<List, Equiv>::type;
/// \brief Stable: runs of adjacent elements equivalent to the run's first element.
/// \ingroup core
template <typename List, template <typename, typename> class Equiv>
using group_adjacent_t = fold_left_t<List, type_list<>, detail::adjacent<Equiv>::template fn>;
/// \brief group_adjacent_t with Equiv = "Key\<A> is Key\<B>", e.g.
///
/// Key = std::remove_cvref_t.
/// \ingroup core
template <typename List, template <typename> class Key>
using group_adjacent_by_key_t = group_adjacent_t<List, detail::same_key<Key>::template fn>;

// --- topological sort ----------------------------------------------------------------------------------------------

namespace detail {

template <typename Remaining, template <typename, typename> class Depends>
struct topo_impl;
template <template <typename, typename> class Depends>
struct topo_impl<type_list<>, Depends>
{
    using type = type_list<>;
};
template <typename... Ts, template <typename, typename> class Depends>
struct topo_impl<type_list<Ts...>, Depends>
{
    template <typename Node>
    static constexpr bool blocked = (false || ... || (!std::is_same_v<Node, Ts> && static_cast<bool>(Depends<Node, Ts>::value)));

    static consteval std::size_t first_free() noexcept
    {
        constexpr bool free_nodes[]{!blocked<Ts>...};
        for (std::size_t i = 0; i < sizeof...(Ts); ++i) {
            if (free_nodes[i]) {
                return i;
            }
        }
        return npos;
    }

    static constexpr std::size_t pick = first_free();
    static_assert(pick != npos, "meta::topological_sort_t: the dependency graph has a cycle");
    using list = type_list<Ts...>;
    using type = push_front_t<typename topo_impl<erase_at_t<list, pick>, Depends>::type,
                              at_t<list, (pick == npos ? 0 : pick)>>;
};

} // namespace detail

/// \brief Depends\<A, B>::value means A depends on B, so B comes first.
///
/// Stable: the earliest free node is taken each step.
/// \ingroup core
template <typename List, template <typename, typename> class Depends>
using topological_sort_t = typename detail::topo_impl<List, Depends>::type;

// --- products ---------------------------------------------------------------------------------------------------------

namespace detail {

template <typename H, typename Lists>
struct prepend_each;
template <typename H, typename... Lists>
struct prepend_each<H, type_list<Lists...>>
{
    using type = type_list<push_front_t<Lists, H>...>;
};

template <typename... Lists>
struct product_impl
{
    using type = type_list<type_list<>>;   // the product of no lists has one (empty) tuple
};
template <typename... Ts, typename... Rest>
struct product_impl<type_list<Ts...>, Rest...>
{
    using tail = typename product_impl<Rest...>::type;
    using type = concat_t<typename prepend_each<Ts, tail>::type...>;
};

template <template <typename...> class F>
struct apply_to_list
{
    template <typename List>
    using fn = rename_t<List, F>;
};

template <std::size_t, typename T>
using repeat = T;

template <typename List, typename Seq>
struct power_impl;
template <typename List, std::size_t... Is>
struct power_impl<List, std::index_sequence<Is...>>
{
    using type = typename product_impl<repeat<Is, List>...>::type;
};

} // namespace detail

/// \brief Every combination, first list varying slowest: \<a,b> x \<x,y> -> <\<a,x>,\<a,y>,\<b,x>,\<b,y>>.
/// \ingroup core
template <typename... Lists>
using cartesian_product_t = typename detail::product_impl<Lists...>::type;
/// \brief cartesian_product_t with each combination applied to F.
///
/// Product_with_t\<std::pair, L1, L2> -> \<pair\<a,x>, ...>.
/// \ingroup core
template <template <typename...> class F, typename... Lists>
using product_with_t = transform_t<cartesian_product_t<Lists...>, detail::apply_to_list<F>::template fn>;
/// \brief All length-N combinations of List (N == 0 gives one empty combination).
/// \ingroup core
template <typename List, std::size_t N>
using cartesian_power_t = typename detail::power_impl<List, std::make_index_sequence<N>>::type;

/// \brief The outer product: every combination, first list varying slowest. `cartesian_product_t` by its other name.
/// \ingroup core
template <typename... Lists>
using outer_product_t = cartesian_product_t<Lists...>;
/// \brief The outer product with each combination applied to F: `product_with_t` by its other name.
/// \ingroup core
template <template <typename...> class F, typename... Lists>
using outer_with_t = product_with_t<F, Lists...>;

// --- Thread (Mathematica) and the dot product -----------------------------------------------------------------------

/// \brief Thread over every argument: Wolfram's `All`, the default.
/// \ingroup core
struct thread_all
{
    /// \brief Whether the specification fits this many arguments.
    /// \return Always true.
    static constexpr bool valid(int) noexcept { return true; }
    /// \brief Whether this 1-based position is threaded.
    /// \return Always true.
    static constexpr bool selects(int, int) noexcept { return true; }
};

/// \brief Thread over no argument: Wolfram's `None`. The result is the expression unchanged.
/// \ingroup core
struct thread_none
{
    /// \brief Whether the specification fits this many arguments.
    /// \return Always true.
    static constexpr bool valid(int) noexcept { return true; }
    /// \brief Whether this 1-based position is threaded.
    /// \return Always false.
    static constexpr bool selects(int, int) noexcept { return false; }
};

/// \brief Thread over the first N arguments, or the last -N when N is negative: Wolfram's `n` and `-n`.
/// \ingroup core
/// \tparam N The count.
template <int N>
struct thread_first
{
    /// \brief Whether the specification fits this many arguments.
    /// \param total The number of arguments.
    /// \return Whether there are that many.
    static constexpr bool valid(int total) noexcept { return (N < 0 ? -N : N) <= total; }
    /// \brief Whether this 1-based position is threaded.
    /// \param position The position.
    /// \param total    The number of arguments.
    /// \return Whether it is among those chosen.
    static constexpr bool selects(int position, int total) noexcept { return N >= 0 ? position <= N : position > total + N; }
};

/// \brief Thread over one argument: Wolfram's `{n}`. Positions count from 1, as in Wolfram; negative ones from the end.
/// \ingroup core
/// \tparam N The position.
template <int N>
struct thread_at
{
    /// \brief Whether the specification fits this many arguments.
    /// \param total The number of arguments.
    /// \return Whether the position exists.
    static constexpr bool valid(int total) noexcept { return N != 0 && (N < 0 ? -N : N) <= total; }
    /// \brief Whether this 1-based position is threaded.
    /// \param position The position.
    /// \param total    The number of arguments.
    /// \return Whether it is the one chosen.
    static constexpr bool selects(int position, int total) noexcept { return position == (N < 0 ? total + 1 + N : N); }
};

/// \brief Thread over positions M to N in steps of S: Wolfram's `{m, n}` and `{m, n, s}`, so `<1, -1, 2>` is every other argument.
/// \ingroup core
/// \tparam M The first position.
/// \tparam N The last position.
/// \tparam S The step; not zero.
template <int M, int N, int S = 1>
struct thread_range
{
    static_assert(S != 0, "snicholls::meta::thread_range: the step must not be zero");

    /// \brief Whether the specification fits this many arguments.
    /// \param total The number of arguments.
    /// \return Whether both ends exist.
    static constexpr bool valid(int total) noexcept
    {
        return M != 0 && N != 0 && (M < 0 ? -M : M) <= total && (N < 0 ? -N : N) <= total;
    }
    /// \brief Whether this 1-based position is threaded.
    /// \param position The position.
    /// \param total    The number of arguments.
    /// \return Whether the walk from M to N lands on it.
    static constexpr bool selects(int position, int total) noexcept
    {
        const int first = M < 0 ? total + 1 + M : M;
        const int last = N < 0 ? total + 1 + N : N;
        const int distance = S > 0 ? position - first : first - position;
        const int length = S > 0 ? last - first : first - last;
        return distance >= 0 && distance <= length && distance % (S > 0 ? S : -S) == 0;
    }
};

namespace detail {

template <typename Arg, template <typename...> class Head>
inline constexpr bool has_head_v = false;
template <template <typename...> class Head, typename... Ts>
inline constexpr bool has_head_v<Head<Ts...>, Head> = true;

template <typename Arg, bool Threaded, std::size_t I>
struct thread_element
{
    using type = Arg;   // repeated, as Wolfram copies an argument without the head
};
template <typename Arg, std::size_t I>
struct thread_element<Arg, true, I>
{
    using type = at_t<rename_t<Arg, type_list>, I>;
};

template <typename Arg, bool Threaded>
inline constexpr std::size_t thread_length_v = npos;
template <typename Arg>
inline constexpr std::size_t thread_length_v<Arg, true> = rename_t<Arg, type_list>::size;

template <template <typename...> class Head, template <typename...> class F, typename Spec, typename Positions, typename... Args>
struct thread_impl;
template <template <typename...> class Head, template <typename...> class F, typename Spec, std::size_t... Ps, typename... Args>
struct thread_impl<Head, F, Spec, std::index_sequence<Ps...>, Args...>
{
    static constexpr int total = static_cast<int>(sizeof...(Args));
    static_assert(Spec::valid(total), "snicholls::meta::thread_t: the position specification does not fit the number of arguments");

    static constexpr bool threaded_at_all =
        (false || ... || (has_head_v<Args, Head> && Spec::selects(static_cast<int>(Ps) + 1, total)));

    template <typename Arg, std::size_t P>
    static constexpr bool threads = has_head_v<Arg, Head> && Spec::selects(static_cast<int>(P) + 1, total);

    static consteval std::size_t shortest() noexcept
    {
        std::size_t length = npos;
        ((length = thread_length_v<Args, threads<Args, Ps>> < length ? thread_length_v<Args, threads<Args, Ps>> : length), ...);
        return length == npos ? 0 : length;
    }
    static constexpr std::size_t length = shortest();
    static_assert((true && ... && (!threads<Args, Ps> || thread_length_v<Args, threads<Args, Ps>> == length)),
                  "snicholls::meta::thread_t: the threaded arguments must all be the same length (zip_t truncates)");

    template <std::size_t I>
    using row = F<typename thread_element<Args, threads<Args, Ps>, I>::type...>;

    template <typename Rows>
    struct threaded;
    template <std::size_t... Is>
    struct threaded<std::index_sequence<Is...>>
    {
        using type = Head<row<Is>...>;
    };

    template <bool Any, typename Dummy = void>
    struct choose
    {
        using type = F<Args...>;   // nothing to thread over: Thread[f[x]] is f[x]
    };
    template <typename Dummy>
    struct choose<true, Dummy> : threaded<std::make_index_sequence<length>> {};

    using type = typename choose<threaded_at_all>::type;
};

template <typename Expression, template <typename...> class Head, typename Spec>
struct thread_expression;
template <template <typename...> class F, typename... Args, template <typename...> class Head, typename Spec>
struct thread_expression<F<Args...>, Head, Spec> : thread_impl<Head, F, Spec, std::index_sequence_for<Args...>, Args...> {};

} // namespace detail

/// \brief Mathematica's `Thread[f[args], h, spec]`: `thread_t<f<type_list<a, b>, x>>` is `type_list<f<a, x>, f<b, x>>`.
///
/// `f` is threaded over the arguments whose head is `Head`, and the result has that head. Other arguments
/// are repeated. Threaded arguments must be of one length. With nothing to thread, the result is the
/// expression itself. `thread_t<type_list<L1, L2>>` is the transpose, as in Wolfram. One level only. As Wolfram
/// warns of `Thread` evaluating first, naming `f<type_list<...>, x>` must be harmless: where `f` is an alias
/// or is constrained, use `thread_with_t`, its `MapThread`-like form.
/// \ingroup core
/// \tparam Expression The expression: any specialisation of a template of types.
/// \tparam Head       The head to thread over: `type_list`, `std::tuple` or any other template of types.
/// \tparam Spec       Which arguments: `thread_all`, `thread_none`, `thread_first`, `thread_at`, `thread_range`.
template <typename Expression, template <typename...> class Head = type_list, typename Spec = thread_all>
using thread_t = typename detail::thread_expression<Expression, Head, Spec>::type;

/// \brief `thread_t` with the function and arguments apart: `thread_with_t<F, type_list<a, b>, x>` is `type_list<F<a, x>, F<b, x>>`.
/// \ingroup core
/// \tparam F    The template to apply.
/// \tparam Args The arguments: type lists of one length are threaded, anything else repeated.
template <template <typename...> class F, typename... Args>
using thread_with_t = typename detail::thread_impl<type_list, F, thread_all, std::index_sequence_for<Args...>, Args...>::type;

/// \brief The dot product: corresponding elements together. `<a, b> . <x, y>` is `<<a, x>, <b, y>>`; lengths must match.
///
/// `zip_t` is the two-list form that truncates and gives pairs; this takes any number of lists and refuses a mismatch.
/// \ingroup core
/// \tparam Lists The lists, at least one.
template <typename... Lists>
    requires(sizeof...(Lists) > 0 && (is_type_list_v<Lists> && ...))
using dot_product_t = thread_with_t<type_list, Lists...>;

/// \brief The dot product with each row applied to F: `dot_with_t<F, <a, b>, <x, y>>` is `<F<a, x>, F<b, y>>`.
/// \ingroup core
/// \tparam F     The template to apply.
/// \tparam Lists The lists, at least one.
template <template <typename...> class F, typename... Lists>
    requires(sizeof...(Lists) > 0 && (is_type_list_v<Lists> && ...))
using dot_with_t = thread_with_t<F, Lists...>;

// --- permutations -----------------------------------------------------------------------------------------------------

namespace detail {

template <typename List, std::size_t R>
struct permutations_impl
{
    using distinct = unique_t<List>;

    template <typename Ds>
    struct counts_of;
    template <typename... Ds>
    struct counts_of<type_list<Ds...>>
    {
        static constexpr std::array<std::size_t, sizeof...(Ds)> value{count_v<List, Ds>...};
    };

    static consteval std::size_t row_count() noexcept
    {
        std::array<std::size_t, distinct::size> remaining = counts_of<distinct>::value;
        std::array<std::size_t, R> row{};
        std::size_t count = 0;
        auto tally = [&count](const std::array<std::size_t, R>&) noexcept { ++count; };
        snicholls::detail::for_each_arrangement(remaining, row, 0, tally);
        return count;
    }

    static consteval std::array<std::array<std::size_t, R>, row_count()> table() noexcept
    {
        std::array<std::array<std::size_t, R>, row_count()> rows{};
        std::array<std::size_t, distinct::size> remaining = counts_of<distinct>::value;
        std::array<std::size_t, R> row{};
        std::size_t next = 0;
        auto store = [&rows, &next](const std::array<std::size_t, R>& current) noexcept { rows[next++] = current; };
        snicholls::detail::for_each_arrangement(remaining, row, 0, store);
        return rows;
    }

    static constexpr std::array<std::array<std::size_t, R>, row_count()> rows = table();

    // Read through a function: MSVC rejects a subscript of `rows` inside the pack expansion below.
    static constexpr std::size_t cell(std::size_t i, std::size_t c) noexcept { return rows[i][c]; }

    template <std::size_t I, typename Columns>
    struct row_at;
    template <std::size_t I, std::size_t... Columns>
    struct row_at<I, std::index_sequence<Columns...>>
    {
        using type = type_list<at_t<distinct, cell(I, Columns)>...>;
    };

    template <typename Rows>
    struct all_rows;
    template <std::size_t... Is>
    struct all_rows<std::index_sequence<Is...>>
    {
        using type = type_list<typename row_at<Is, std::make_index_sequence<R>>::type...>;
    };

    using type = typename all_rows<std::make_index_sequence<row_count()>>::type;
};

} // namespace detail

/// \brief Every distinct ordering of `R` elements of `List`: `permutations_t<<a, b>>` is `<<a, b>, <b, a>>`.
///
/// Rows are in lexicographic order, with types ranked by first appearance. A repeated type gives each distinct row once
/// (`<a, a, b>` has 3 rows, not 6), and `R` above the size gives no rows. Wolfram's `Permutations[list, {r}]`. The
/// instantiations are one per element of the result; the walk is a constexpr function, not templates.
/// \ingroup core
/// \tparam List The list.
/// \tparam R    The length of each row; all of `List` by default.
template <typename List, std::size_t R = List::size>
using permutations_t = typename detail::permutations_impl<List, R>::type;

/// \brief `permutations_t` with each row applied to F.
/// \ingroup core
/// \tparam F    The template to apply.
/// \tparam List The list.
/// \tparam R    The length of each row; all of `List` by default.
template <template <typename...> class F, typename List, std::size_t R = List::size>
using permutations_with_t = transform_t<permutations_t<List, R>, detail::apply_to_list<F>::template fn>;

// --- Partition (Mathematica) -----------------------------------------------------------------------------------------

namespace detail {

template <typename List, std::size_t N, std::size_t D, typename Starts>
struct partition_n_impl;
template <typename List, std::size_t N, std::size_t D, std::size_t... Ks>
struct partition_n_impl<List, N, D, std::index_sequence<Ks...>>
{
    using type = type_list<typename slice<List, Ks * D, Ks * D + N>::type...>;
};

} // namespace detail

/// \brief Wolfram's `Partition[list, n, d]` for types: sublists of N types starting every D, full ones only.
///
/// `partition_n_t<<a, b, c, d>, 2>` is `<<a, b>, <c, d>>`; with D = 1, every window. (`partition<L, P>` is the split by a
/// predicate.)
/// \ingroup core
/// \tparam List The list.
/// \tparam N    The sublists' length, above zero.
/// \tparam D    The offset, above zero; N by default.
template <typename List, std::size_t N, std::size_t D = N>
    requires(N > 0 && D > 0)
using partition_n_t = typename detail::partition_n_impl<List, N, D,
                                                        std::make_index_sequence<(List::size < N ? 0 : (List::size - N) / D + 1)>>::type;

// --- more of Wolfram's list functions ----------------------------------------------------------------------------

namespace detail {

template <typename Seq>
struct range_impl;
template <std::size_t... Is>
struct range_impl<std::index_sequence<Is...>>
{
    using type = type_list<constant<Is>...>;
};

template <typename List, template <typename, typename> class F, typename Seq>
struct map_indexed_impl;
template <typename... Ts, template <typename, typename> class F, std::size_t... Is>
struct map_indexed_impl<type_list<Ts...>, F, std::index_sequence<Is...>>
{
    using type = type_list<F<Ts, constant<Is>>...>;
};

template <typename List, typename Separator, typename Seq>
struct riffle_impl;
template <typename List, typename Separator, std::size_t... Is>
struct riffle_impl<List, Separator, std::index_sequence<Is...>>
{
    using type = type_list<std::conditional_t<Is % 2 == 0, at_t<List, (Is % 2 == 0 ? Is / 2 : 0)>, Separator>...>;
};

template <typename Acc, typename List, template <typename, typename> class F>
struct fold_list_impl
{
    using type = type_list<Acc>;
};
template <typename Acc, typename T, typename... Ts, template <typename, typename> class F>
struct fold_list_impl<Acc, type_list<T, Ts...>, F>
{
    using type = push_front_t<typename fold_list_impl<F<Acc, T>, type_list<Ts...>, F>::type, Acc>;
};

template <template <typename> class F, typename X, std::size_t N>
struct nest_list_impl
{
    using type = push_front_t<typename nest_list_impl<F, F<X>, N - 1>::type, X>;
};
template <template <typename> class F, typename X>
struct nest_list_impl<F, X, 0>
{
    using type = type_list<X>;
};

template <typename List, typename Unique>
struct tally_impl;
template <typename List, typename... Us>
struct tally_impl<List, type_list<Us...>>
{
    using type = type_list<type_list<Us, constant<count_v<List, Us>>>...>;
};

template <typename List, std::size_t K, typename Rows>
struct subsets_impl;
template <typename List, std::size_t K, std::size_t... Rs>
struct subsets_impl<List, K, std::index_sequence<Rs...>>
{
    static constexpr auto table = snicholls::index_combinations<List::size, K>();
    template <std::size_t R, typename Columns>
    struct row;
    template <std::size_t R, std::size_t... Cs>
    struct row<R, std::index_sequence<Cs...>>
    {
        using type = type_list<at_t<List, table[R][Cs]>...>;
    };
    using type = type_list<typename row<Rs, std::make_index_sequence<K>>::type...>;
};

template <typename List, typename Sizes>
struct all_subsets_impl;
template <typename List, std::size_t... Ks>
struct all_subsets_impl<List, std::index_sequence<Ks...>>
{
    using type = concat_t<typename subsets_impl<List, Ks, std::make_index_sequence<snicholls::combination_count_v<List::size, Ks>>>::type...>;
};

} // namespace detail

/// \brief `Range`: `type_list<constant<0>, ..., constant<N - 1>>`.
/// \ingroup core
template <std::size_t N>
using range_t = typename detail::range_impl<std::make_index_sequence<N>>::type;

/// \brief `MapIndexed[f, list]`: `type_list<F<T0, constant<0>>, F<T1, constant<1>>, ...>`.
/// \ingroup core
template <typename List, template <typename, typename> class F>
using map_indexed_t = typename detail::map_indexed_impl<List, F, std::make_index_sequence<List::size>>::type;

/// \brief `RotateLeft[list, n]`: the first `N` types moved to the end (cyclically).
/// \ingroup core
template <typename List, std::size_t N>
using rotate_left_t = concat_t<drop_t<List, (List::size == 0 ? 0 : N % List::size)>, take_t<List, (List::size == 0 ? 0 : N % List::size)>>;

/// \brief `RotateRight[list, n]`: the last `N` types moved to the front (cyclically).
/// \ingroup core
template <typename List, std::size_t N>
using rotate_right_t = rotate_left_t<List, (List::size == 0 ? 0 : List::size - N % List::size)>;

/// \brief `Riffle[list, x]`: `Separator` between every two types.
/// \ingroup core
template <typename List, typename Separator>
using riffle_t = typename detail::riffle_impl<List, Separator, std::make_index_sequence<(List::size == 0 ? 0 : 2 * List::size - 1)>>::type;

/// \brief `FoldList[f, x, list]`: `type_list<Init, F<Init, T0>, F<F<Init, T0>, T1>, ...>`.
/// \ingroup core
template <typename List, typename Init, template <typename, typename> class F>
using fold_list_t = typename detail::fold_list_impl<Init, List, F>::type;

/// \brief `NestList[f, x, n]`: `type_list<X, F<X>, F<F<X>>, ...>`, `N + 1` types.
/// \ingroup core
template <template <typename> class F, typename X, std::size_t N>
using nest_list_t = typename detail::nest_list_impl<F, X, N>::type;

/// \brief `Nest[f, x, n]`: `F` applied `N` times to `X`.
/// \ingroup core
template <template <typename> class F, typename X, std::size_t N>
using nest_t = back_t<nest_list_t<F, X, N>>;

/// \brief `Tally[list]`: `type_list<type_list<T, constant<count>>...>` for each distinct type, in order of first appearance.
/// \ingroup core
template <typename List>
using tally_t = typename detail::tally_impl<List, unique_t<List>>::type;

/// \brief `Subsets[list, {k}]`: every `K`-element sublist, in order of position.
/// \ingroup core
template <typename List, std::size_t K>
using subsets_of_size_t =
    typename detail::subsets_impl<List, K, std::make_index_sequence<snicholls::combination_count_v<List::size, K>>>::type;

/// \brief `Subsets[list]`: every sublist, by size and then in order of position: `2^n` of them.
/// \ingroup core
template <typename List>
using subsets_t = typename detail::all_subsets_impl<List, std::make_index_sequence<List::size + 1>>::type;

/// \brief `Inner[f, list1, list2, g]`: `G<F<A0, B0>, F<A1, B1>, ...>`, the generalised dot product; lengths must match.
/// \ingroup core
template <template <typename...> class F, typename L1, typename L2, template <typename...> class G>
    requires(L1::size == L2::size)
using inner_t = rename_t<zip_with_t<F, L1, L2>, G>;

// --- reify (types -> run time) ---------------------------------------------------------------------------------------

namespace detail {

template <typename F, typename... Ts>
constexpr void for_each_type_impl(F& f, type_list<Ts...>) noexcept((noexcept(f.template operator()<Ts>()) && ...))
{
    (static_cast<void>(f.template operator()<Ts>()), ...);
}

template <typename F, typename... Ts>
constexpr auto make_array_impl(F& f, type_list<Ts...>)
{
    using value_type = std::common_type_t<decltype(f.template operator()<Ts>())...>;
    return std::array<value_type, sizeof...(Ts)>{{static_cast<value_type>(f.template operator()<Ts>())...}};
}

} // namespace detail

/// \brief Calls f.template operator()\<T>() for each T in order: for_each_type\<L>([]\<typename T>() { ...
///
/// }).
/// \ingroup core
/// \param f the callable.
template <typename List, typename F>
constexpr void for_each_type(F&& f) noexcept(noexcept(detail::for_each_type_impl(f, List{})))
{
    detail::for_each_type_impl(f, List{});
}

/// \brief Calls f(std::type_identity\<T>{}) for each T in order, so an `overloaded` or `first_match` can choose per type.
///
/// `for_each_type` calls a template with no arguments, which an overload set cannot choose on; a tag gives it one:
/// `for_each_tag<L>(overloaded{[](std::type_identity<int>) {...}, [](std::type_identity<double>) {...}})`.
/// \ingroup core
/// \param f the callable.
template <typename List, typename F>
constexpr void for_each_tag(F&& f)
{
    for_each_type<List>([&f]<typename T>() { static_cast<void>(f(std::type_identity<T>{})); });
}

/// \brief std::array of f.template operator()\<T>() for each T.
///
/// Make_array\<L>([]\<typename T>() { return sizeof(T); }).
/// \ingroup core
/// \param f the callable.
/// \return the result.
template <typename List, typename F>
    requires(List::size > 0)
constexpr auto make_array(F&& f)
{
    return detail::make_array_impl(f, List{});
}

} // namespace snicholls::meta

#endif /* snicholls_core_meta_type_list_hpp */
