//
//  rust.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief Rust's `Option` methods over `std::optional` (`fn::rust::option`), `std::cmp`'s ordering helpers
///        (`fn::rust::cmp`) and `Into` (`fn::rust::into<T>`), each the one Rust has and C++20 lacks. All `constexpr`.
/// \ingroup core
///
/// **Overview.** `snicholls::result` already carries Rust's `Result` methods as members (`map`, `map_err`,
/// `map_or`, `map_or_else`, `and_then`, `or_else`, `and_`, `or_`, `inspect`, `inspect_err`, `is_ok_and`,
/// `is_err_and`, `ok`, `err`, `flatten`, `transpose`, `unwrap_or`, `unwrap_or_else`, `unwrap_or_default`, `iter`).
/// C++20's `std::optional` has only `value_or`; this header gives it the rest of Rust's `Option`, as free
/// functions taking the optional first, in Rust's argument order:
///
/// | Rust                                  | here (`fn::rust::option::`)          | already in core / std                 |
/// | ------------------------------------- | ------------------------------------ | ------------------------------------- |
/// | `o.is_some_and(p)`, `o.is_none_or(p)` | `is_some_and(o, p)`, `is_none_or(o, p)` |                                    |
/// | `o.map_or(d, f)`, `o.map_or_else(d, f)` | `map_or(o, d, f)`, `map_or_else(o, d, f)` |                                |
/// | `o.ok_or(e)`, `o.ok_or_else(f)`       | `ok_or(o, e)`, `ok_or_else(o, f)`: a `snicholls::result` |                   |
/// | `o.filter(p)`, `o.inspect(f)`         | `filter(o, p)`, `inspect(o, f)`      |                                       |
/// | `o.and(b)`, `o.or(b)`, `o.xor(b)`, `o.or_else(f)` | `and_(o, b)`, `or_(o, b)`, `xor_(o, b)`, `or_else(o, f)` |        |
/// | `o.unwrap_or_else(f)`, `o.unwrap_or_default()` | `unwrap_or_else(o, f)`, `unwrap_or_default(o)` | `unwrap_or` is `o.value_or(d)` |
/// | `o.zip(b)`, `o.zip_with(b, f)`, `o.unzip()` | `zip(o, b)`, `zip_with(o, b, f)`, `unzip(o)` |                     |
/// | `o.transpose()`                       | `transpose(o)`: `optional<result<V, E>>` to `result<optional<V>, E>` | |
/// | `o.take()`, `o.take_if(p)`, `o.replace(v)` | `take(o)`, `take_if(o, p)`, `replace(o, v)` |                      |
/// | `o.insert(v)`, `o.get_or_insert(v)`, `o.get_or_insert_with(f)` | `insert(o, v)`, `get_or_insert(o, v)`, `get_or_insert_with(o, f)` | |
/// | `o.map(f)`, `o.and_then(f)`, `o.flatten()` |                                  | `fn::fmap(f, o)`, `fn::mbind(o, f)`, `fn::join(o)` |
/// | `o.unwrap()`, `o.expect(m)`, `o.is_some()` |                                  | `*o` / `o.value()`, `o.has_value()`  |
/// | `Ordering::then`, `then_with`, `reverse` | `cmp::then(a, b)`, `cmp::then_with(a, f)`, `cmp::reverse(a)` | `std::is_lt` and friends |
/// | `Ord::cmp` by a key, `.then_with` on keys | `cmp::comparing(key)`, `cmp::lexicographic(c1, c2, ...)` |         |
/// | `std::cmp::Reverse`                   | `cmp::reversed(c)`; `cmp::as_less(c)` for `std::ranges::sort` |       |
/// | `x.into()` (`Into<T>`)                | `into<T>(x)`: an implicit, non-narrowing conversion | `TryInto` is `snicholls::numeric_cast` |
/// | `Default::default()`, `From`          |                                      | `T{}`, `std::default_initializable`, a constructor |
///
/// \snippet rust_demo.cpp option
///
/// In a pipeline the receiver is bound last: `o | fn::then(fn::partial_right(option::filter, p))`, or a
/// plain lambda. `fn::rust::option` is a namespace of function objects, so each can be passed and curried.
///
/// **Ordering.** Rust's `Ordering` is C++'s comparison categories (`std::strong_ordering` and friends, from
/// `<=>`). `cmp::then` keeps the first non-equal ordering (the category is their common one), `cmp::reverse`
/// flips one (`0 <=> o`), and `cmp::comparing(key)` builds a three-way comparator from a key, through
/// `snicholls::synth_three_way` so a key with only `<` works:
///
/// \snippet rust_demo.cpp cmp
///
/// **Relationships.** `snicholls::result` (Rust's `Result`), `functional/rust_iter.hpp` (iterators and slices),
/// `functional/monad.hpp` and `functional/typeclasses.hpp` (`fmap`, `mbind`, `join`), `functional/ocaml.hpp` (the
/// same idea for OCaml), `meta/compare.hpp`, `checked.hpp` (`numeric_cast`). Group: \ref core.
///
/// **Implementations / variants.** One overload per value category of the optional: an lvalue is copied
/// from, an rvalue moved from, as Rust's `self` by value moves.
///
/// **Guarantees.** `constexpr`. `is_some_and`, `is_none_or`, `map_or`, `map_or_else`, `take`, `into` and the `cmp`
/// functions (but not the comparator `lexicographic` returns) are `noexcept` exactly when the calls and the copies or
/// moves they make are; the others do not claim it.
/// `take`, `take_if`, `replace`, `insert` and the `get_or_insert` family modify the optional they are given,
/// as Rust's `&mut self` methods do; `insert` and `get_or_insert*` return a reference into it, valid while it
/// holds that value.
///
/// **Errors.** None of their own: a callable's exception propagates, and the optional is then unchanged.
///
/// **Examples.** \snippet rust_demo.cpp into
///
/// **Performance.** A branch on `has_value()` and the call; not benchmarked.
///
/// **Tests and demos.** `tests/functional/rust_test.cpp`, `demos/functional/rust_demo.cpp`,
/// `demos/functional/rust_showcase_demo.cpp` (Option, Result and iterator chains end to end).
///
/// **Dependencies.** `<compare>`, `<concepts>`, `<cstddef>`, `<functional>`, `<optional>`, `<type_traits>`, `<utility>`,
/// `snicholls/core/result.hpp` (heavier than it looks: `<string_view>` and `<stdexcept>`), `meta/compare.hpp`,
/// `meta/concepts.hpp` and `functional/combinators.hpp`.
///
/// **Known issues and gotchas.** `and`, `or` and `xor` are C++ keywords, so the functions are `and_`, `or_` and
/// `xor_`, as `result`'s members are. `fn::rust::option::zip` is not `snicholls::zip` (ranges): qualify it.
/// `into<T>` refuses a narrowing conversion by brace-initialisation's rules, as core's struct conversions do:
/// `double` to `int` (Rust has no `From<f64> for i32`) and also `int` to `double`, which Rust allows.
///
/// **Status.** Experimental, new in 0.27.0.

#ifndef snicholls_core_functional_rust_hpp
#define snicholls_core_functional_rust_hpp

#include <snicholls/core/functional/combinators.hpp>
#include <snicholls/core/meta/compare.hpp>
#include <snicholls/core/meta/concepts.hpp>
#include <snicholls/core/result.hpp>

#include <compare>
#include <concepts>
#include <cstddef>
#include <functional>
#include <optional>
#include <type_traits>
#include <utility>

/// \brief Rust's standard library vocabulary, where C++20 lacks it.
/// \ingroup core
namespace snicholls::fn::rust {

    namespace detail {

        template <class O>
        concept optional_ = specialisation_of<O, std::optional>;

        template <class O>
        using value_t = typename std::remove_cvref_t<O>::value_type;

        template <class O>
        using deref_t = decltype(*std::declval<O>());

        template <class T>
        concept ordering = std::same_as<T, std::strong_ordering> || std::same_as<T, std::weak_ordering> ||
                           std::same_as<T, std::partial_ordering>;

        struct is_some_and_fn {
            template <optional_ O, class P>
            constexpr bool operator()(O&& o, P&& p) const noexcept(std::is_nothrow_invocable_v<P, deref_t<O>>)
            {
                return o.has_value() && static_cast<bool>(std::invoke(std::forward<P>(p), *std::forward<O>(o)));
            }
        };

        struct is_none_or_fn {
            template <optional_ O, class P>
            constexpr bool operator()(O&& o, P&& p) const noexcept(std::is_nothrow_invocable_v<P, deref_t<O>>)
            {
                return !o.has_value() || static_cast<bool>(std::invoke(std::forward<P>(p), *std::forward<O>(o)));
            }
        };

        struct map_or_fn {
            template <optional_ O, class U, class F>
            constexpr U operator()(O&& o, U fallback, F&& f) const
                noexcept(std::is_nothrow_invocable_r_v<U, F, deref_t<O>> && std::is_nothrow_move_constructible_v<U>)
            {
                if (!o.has_value()) return fallback;
                return std::invoke(std::forward<F>(f), *std::forward<O>(o));
            }
        };

        struct map_or_else_fn {
            template <optional_ O, class D, class F,
                      class R = std::common_type_t<std::invoke_result_t<D>, std::invoke_result_t<F, deref_t<O>>>>
            constexpr R operator()(O&& o, D&& fallback, F&& f) const
                noexcept(std::is_nothrow_invocable_r_v<R, D> && std::is_nothrow_invocable_r_v<R, F, deref_t<O>>)
            {
                if (!o.has_value()) return std::invoke(std::forward<D>(fallback));
                return std::invoke(std::forward<F>(f), *std::forward<O>(o));
            }
        };

        struct ok_or_fn {
            template <optional_ O, class E, class R = snicholls::result<value_t<O>, std::decay_t<E>>>
            constexpr R operator()(O&& o, E&& e) const
            {
                if (o.has_value()) return snicholls::ok(*std::forward<O>(o));
                return snicholls::err(std::forward<E>(e));
            }
        };

        struct ok_or_else_fn {
            template <optional_ O, class F,
                      class R = snicholls::result<value_t<O>, std::remove_cvref_t<std::invoke_result_t<F>>>>
            constexpr R operator()(O&& o, F&& f) const
            {
                if (o.has_value()) return snicholls::ok(*std::forward<O>(o));
                return snicholls::err(std::invoke(std::forward<F>(f)));
            }
        };

        struct filter_fn {
            template <optional_ O, class P>
            constexpr std::remove_cvref_t<O> operator()(O&& o, P&& p) const
            {
                if (o.has_value() && std::invoke(std::forward<P>(p), std::as_const(*o))) return std::forward<O>(o);
                return std::nullopt;
            }
        };

        struct inspect_fn {
            // An lvalue optional is returned by reference, a temporary by value (Rust returns self).
            template <optional_ O, class F, class R = std::conditional_t<std::is_lvalue_reference_v<O>, O, std::remove_cvref_t<O>>>
            constexpr R operator()(O&& o, F&& f) const
            {
                if (o.has_value()) std::invoke(std::forward<F>(f), std::as_const(*o));
                return std::forward<O>(o);
            }
        };

        struct and_fn {
            template <optional_ O, optional_ B>
            constexpr std::remove_cvref_t<B> operator()(const O& o, B&& b) const
            {
                if (o.has_value()) return std::forward<B>(b);
                return std::nullopt;
            }
        };

        struct or_fn {
            template <optional_ O>
            constexpr std::remove_cvref_t<O> operator()(O&& o, std::remove_cvref_t<O> b) const
            {
                if (o.has_value()) return std::forward<O>(o);
                return b;
            }
        };

        struct xor_fn {
            template <optional_ O>
            constexpr std::remove_cvref_t<O> operator()(O&& o, std::remove_cvref_t<O> b) const
            {
                if (o.has_value() && !b.has_value()) return std::forward<O>(o);
                if (!o.has_value() && b.has_value()) return b;
                return std::nullopt;
            }
        };

        struct or_else_fn {
            template <optional_ O, class F>
                requires std::convertible_to<std::invoke_result_t<F>, std::remove_cvref_t<O>>
            constexpr std::remove_cvref_t<O> operator()(O&& o, F&& f) const
            {
                if (o.has_value()) return std::forward<O>(o);
                return std::invoke(std::forward<F>(f));
            }
        };

        struct unwrap_or_else_fn {
            template <optional_ O, class F>
            constexpr value_t<O> operator()(O&& o, F&& f) const
            {
                if (o.has_value()) return *std::forward<O>(o);
                return std::invoke(std::forward<F>(f));
            }
        };

        struct unwrap_or_default_fn {
            template <optional_ O>
                requires std::default_initializable<value_t<O>>
            constexpr value_t<O> operator()(O&& o) const
            {
                if (o.has_value()) return *std::forward<O>(o);
                return value_t<O>{};
            }
        };

        struct zip_fn {
            template <optional_ A, optional_ B, class R = std::optional<std::pair<value_t<A>, value_t<B>>>>
            constexpr R operator()(A&& a, B&& b) const
            {
                if (a.has_value() && b.has_value()) return R(std::in_place, *std::forward<A>(a), *std::forward<B>(b));
                return std::nullopt;
            }
        };

        struct zip_with_fn {
            template <optional_ A, optional_ B, class F,
                      class R = std::optional<std::remove_cvref_t<std::invoke_result_t<F, deref_t<A>, deref_t<B>>>>>
            constexpr R operator()(A&& a, B&& b, F&& f) const
            {
                if (a.has_value() && b.has_value())
                    return R(std::invoke(std::forward<F>(f), *std::forward<A>(a), *std::forward<B>(b)));
                return std::nullopt;
            }
        };

        struct unzip_fn {
            template <optional_ O, class P = value_t<O>,
                      class R = std::pair<std::optional<typename P::first_type>, std::optional<typename P::second_type>>>
            constexpr R operator()(O&& o) const
            {
                if (!o.has_value()) return R{};
                return R{std::forward<O>(o)->first, std::forward<O>(o)->second};
            }
        };

        struct transpose_fn {
            template <optional_ O, class Res = value_t<O>>
                requires specialisation_of<Res, snicholls::result>
            constexpr auto operator()(O&& o) const
                -> snicholls::result<std::optional<typename Res::value_type>, typename Res::error_type>
            {
                if (!o.has_value()) return snicholls::ok(std::optional<typename Res::value_type>());
                if (o->is_ok()) return snicholls::ok(std::optional<typename Res::value_type>(std::forward<O>(o)->unwrap_unchecked()));
                return snicholls::err(std::forward<O>(o)->unwrap_err_unchecked());
            }
        };

        struct take_fn {
            template <class T>
            constexpr std::optional<T> operator()(std::optional<T>& o) const noexcept(std::is_nothrow_move_constructible_v<T>)
            {
                std::optional<T> out = std::move(o);
                o.reset();
                return out;
            }
        };

        struct take_if_fn {
            template <class T, class P>
            constexpr std::optional<T> operator()(std::optional<T>& o, P&& p) const
            {
                if (o.has_value() && std::invoke(std::forward<P>(p), *o)) return take_fn{}(o);
                return std::nullopt;
            }
        };

        struct replace_fn {
            template <class T, class U>
            constexpr std::optional<T> operator()(std::optional<T>& o, U&& v) const
            {
                std::optional<T> old = take_fn{}(o);
                o.emplace(std::forward<U>(v));
                return old;
            }
        };

        struct insert_fn {
            template <class T, class U>
            constexpr T& operator()(std::optional<T>& o, U&& v) const
            {
                return o.emplace(std::forward<U>(v));
            }
        };

        struct get_or_insert_with_fn {
            template <class T, class F>
            constexpr T& operator()(std::optional<T>& o, F&& f) const
            {
                if (!o.has_value()) o.emplace(std::invoke(std::forward<F>(f)));
                return *o;
            }
        };

        struct get_or_insert_fn {
            template <class T, class U>
            constexpr T& operator()(std::optional<T>& o, U&& v) const
            {
                if (!o.has_value()) o.emplace(std::forward<U>(v));
                return *o;
            }
        };

        // ---- cmp ------------------------------------------------------------------------------------------------

        struct then_fn {
            template <ordering A, ordering B, class R = std::common_comparison_category_t<A, B>>
            constexpr R operator()(A a, B b) const noexcept
            {
                return a != 0 ? R(a) : R(b);
            }
        };

        struct then_with_fn {
            template <ordering A, class F, ordering B = std::invoke_result_t<F>,
                      class R = std::common_comparison_category_t<A, B>>
            constexpr R operator()(A a, F&& f) const noexcept(std::is_nothrow_invocable_v<F>)
            {
                return a != 0 ? R(a) : R(std::invoke(std::forward<F>(f)));
            }
        };

        struct reverse_fn {
            template <ordering A>
            constexpr A operator()(A a) const noexcept
            {
                return 0 <=> a;
            }
        };

        struct comparing_op {
            template <std::size_t... I, class S, class A, class B>
            static constexpr auto call(std::index_sequence<I...>, S&& s, const A& a, const B& b)
                SNICHOLLS_FN_RETURNS(snicholls::synth_three_way(std::invoke(fn::detail::part<0>(s), a),
                                                                std::invoke(fn::detail::part<0>(s), b)))
        };

        // The first comparator's ordering if it is not equal, else the next one's.
        struct lexicographic_op {
            template <class A, class B, class C>
            static constexpr auto step(const A& a, const B& b, const C& c)
            {
                return std::invoke(c, a, b);
            }

            template <class A, class B, class C, class C2, class... Cs>
            static constexpr auto step(const A& a, const B& b, const C& c, const C2& c2, const Cs&... cs)
            {
                return then_with_fn{}(std::invoke(c, a, b), [&] { return step(a, b, c2, cs...); });
            }

            template <std::size_t... I, class S, class A, class B>
            static constexpr auto call(std::index_sequence<I...>, S&& s, const A& a, const B& b)
                SNICHOLLS_FN_RETURNS(step(a, b, fn::detail::part<I>(s)...))
        };

        struct reversed_op {
            template <std::size_t... I, class S, class A, class B>
            static constexpr auto call(std::index_sequence<I...>, S&& s, const A& a, const B& b)
                SNICHOLLS_FN_RETURNS(std::invoke(fn::detail::part<0>(s), b, a))
        };

        struct as_less_op {
            template <std::size_t... I, class S, class A, class B>
            static constexpr auto call(std::index_sequence<I...>, S&& s, const A& a, const B& b)
                SNICHOLLS_FN_RETURNS(std::invoke(fn::detail::part<0>(s), a, b) < 0)
        };

        struct lexicographic_fn {
            template <class... C>
                requires(sizeof...(C) > 0)
            constexpr auto operator()(C&&... c) const noexcept(fn::detail::nothrow_store_v<C...>)
            {
                return fn::detail::make<lexicographic_op>(std::forward<C>(c)...);
            }
        };

        template <class T>
        struct into_fn {
            template <class X>
                requires std::convertible_to<X, T> && requires(X&& x) { T{std::forward<X>(x)}; }
            constexpr T operator()(X&& x) const noexcept(std::is_nothrow_convertible_v<X, T>)
            {
                return std::forward<X>(x);
            }
        };

    } // namespace detail

    /// \brief Rust's `Option` methods over `std::optional`, the optional first.
    /// \ingroup core
    namespace option {
        /// \brief `is_some_and(o, p)`: whether `o` holds a value and `p` of it holds.
        inline constexpr detail::is_some_and_fn is_some_and{};
        /// \brief `is_none_or(o, p)`: whether `o` is empty or `p` of its value holds.
        inline constexpr detail::is_none_or_fn is_none_or{};
        /// \brief `map_or(o, d, f)`: `f(*o)`, or `d` when empty; the result has `d`'s type.
        inline constexpr detail::map_or_fn map_or{};
        /// \brief `map_or_else(o, d, f)`: `f(*o)`, or `d()` when empty.
        inline constexpr detail::map_or_else_fn map_or_else{};
        /// \brief `ok_or(o, e)`: `ok(*o)`, or `err(e)` when empty, as a `snicholls::result`.
        inline constexpr detail::ok_or_fn ok_or{};
        /// \brief `ok_or_else(o, f)`: `ok(*o)`, or `err(f())` when empty; `f` runs only then.
        inline constexpr detail::ok_or_else_fn ok_or_else{};
        /// \brief `filter(o, p)`: `o` if it holds a value of which `p` holds, else empty.
        inline constexpr detail::filter_fn filter{};
        /// \brief `inspect(o, f)`: calls `f(*o)` if `o` holds a value; returns `o` (a reference for an lvalue).
        inline constexpr detail::inspect_fn inspect{};
        /// \brief `and_(o, b)`: `b` if `o` holds a value, else empty (of `b`'s type).
        inline constexpr detail::and_fn and_{};
        /// \brief `or_(o, b)`: `o` if it holds a value, else `b`.
        inline constexpr detail::or_fn or_{};
        /// \brief `xor_(o, b)`: whichever of the two holds a value, if exactly one does; else empty.
        inline constexpr detail::xor_fn xor_{};
        /// \brief `or_else(o, f)`: `o` if it holds a value, else `f()`.
        inline constexpr detail::or_else_fn or_else{};
        /// \brief `unwrap_or_else(o, f)`: `*o`, or `f()` when empty.
        inline constexpr detail::unwrap_or_else_fn unwrap_or_else{};
        /// \brief `unwrap_or_default(o)`: `*o`, or `T{}` when empty.
        inline constexpr detail::unwrap_or_default_fn unwrap_or_default{};
        /// \brief `zip(a, b)`: `std::pair{*a, *b}` if both hold values, else empty.
        inline constexpr detail::zip_fn zip{};
        /// \brief `zip_with(a, b, f)`: `f(*a, *b)` if both hold values, else empty.
        inline constexpr detail::zip_with_fn zip_with{};
        /// \brief `unzip(o)`: an optional pair as a pair of optionals.
        inline constexpr detail::unzip_fn unzip{};
        /// \brief `transpose(o)`: `optional<result<V, E>>` as `result<optional<V>, E>`; empty is `ok(nullopt)`.
        inline constexpr detail::transpose_fn transpose{};
        /// \brief `take(o)`: moves the value out and leaves `o` empty.
        inline constexpr detail::take_fn take{};
        /// \brief `take_if(o, p)`: `take(o)` if `o` holds a value of which `p` holds (`p` sees it mutable), else empty.
        inline constexpr detail::take_if_fn take_if{};
        /// \brief `replace(o, v)`: puts `v` in `o` and returns what was there.
        inline constexpr detail::replace_fn replace{};
        /// \brief `insert(o, v)`: puts `v` in `o`, dropping any value, and returns a reference to it.
        inline constexpr detail::insert_fn insert{};
        /// \brief `get_or_insert(o, v)`: the value in `o`, putting `v` there first if it is empty.
        inline constexpr detail::get_or_insert_fn get_or_insert{};
        /// \brief `get_or_insert_with(o, f)`: the value in `o`, putting `f()` there first if it is empty.
        inline constexpr detail::get_or_insert_with_fn get_or_insert_with{};
    } // namespace option

    /// \brief Rust's `std::cmp` over C++'s comparison categories.
    /// \ingroup core
    namespace cmp {
        /// \brief `then(a, b)`: `a` unless it is equal, then `b` (`Ordering::then`).
        inline constexpr detail::then_fn then{};
        /// \brief `then_with(a, f)`: `a` unless it is equal, then `f()`; `f` runs only then.
        inline constexpr detail::then_with_fn then_with{};
        /// \brief `reverse(a)`: less becomes greater and greater less (`Ordering::reverse`).
        inline constexpr detail::reverse_fn reverse{};
        /// \brief `comparing(key)`: the comparator `(a, b) -> key(a) <=> key(b)`, by `synth_three_way`.
        inline constexpr fn::detail::unary_maker<detail::comparing_op> comparing{};
        /// \brief `lexicographic(c1, c2, ...)`: the comparator that asks `c1`, then `c2` on a tie, and so on.
        inline constexpr detail::lexicographic_fn lexicographic{};
        /// \brief `reversed(c)`: the comparator `(a, b) -> c(b, a)` (`std::cmp::Reverse`).
        inline constexpr fn::detail::unary_maker<detail::reversed_op> reversed{};
        /// \brief `as_less(c)`: the predicate `(a, b) -> c(a, b) < 0`, for `std::ranges::sort` and friends.
        inline constexpr fn::detail::unary_maker<detail::as_less_op> as_less{};
    } // namespace cmp

    /// \brief `into<T>(x)`: `x` converted to `T` implicitly and without narrowing (Rust's `Into<T>`).
    /// \ingroup core
    /// \tparam T The type converted to.
    template <class T>
    inline constexpr detail::into_fn<T> into{};

} // namespace snicholls::fn::rust

#endif /* snicholls_core_functional_rust_hpp */
