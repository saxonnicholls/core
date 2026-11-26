//
//  lazy.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief Lazy evaluation, `constexpr`: `lazy` (a thunk evaluated at most once) and infinite streams with
///        `iterate`, `repeat`, `cycle`, `unfold`, `map`, `filter`, `zip_with`, `scanl`, `drop`, `drop_while`,
///        `take` and `take_while`, finished into `std::array`s at compile time.
/// \ingroup core
///
/// **Overview.** Haskell's lists are lazy: `take 5 (filter even (iterate (*3) 1))` describes an infinite
/// list and computes five elements of it. Here an infinite list is a `fn::stream`: a state and a step
/// `state -> {value, next state}`, a value type that describes the list without computing it. The
/// stream operations, in `namespace snicholls::fn::stream`, return new streams; `take<N>` and
/// `take_while<N>` compute, into a `std::array` or a `containers::inplace_vector`:
///
/// \snippet lazy_demo.cpp stream
///
/// `lazy(f)` is a thunk: `f` is called the first time the value is wanted, and the value kept.
///
/// | Haskell                          | here                                   |
/// | -------------------------------- | -------------------------------------- |
/// | `iterate f x`                    | `stream::iterate(f, x)`                |
/// | `repeat x`, `cycle xs`           | `stream::repeat(x)`, `stream::cycle(xs)` |
/// | `unfoldr` (infinite)             | `stream::unfold(seed, step)`           |
/// | `map`, `filter`, `zipWith`, `zip`, `scanl` | `stream::map`, `filter`, `zip_with`, `zip`, `scanl` |
/// | `drop n`, `dropWhile p`          | `stream::drop(n, s)`, `stream::drop_while(p, s)` |
/// | `take n`, `takeWhile p`          | `stream::take<N>(s)`, `stream::take_while<N>(p, s)` (at most `N`) |
/// | `head`, `tail`, `xs !! n`        | `stream::head(s)`, `stream::tail(s)`, `stream::nth(s, n)` |
///
/// **Relationships.** `fn::unfold` (`functional/fold.hpp`, which `take` calls), `functional/list.hpp`
/// (the same operations on finite arrays), `containers::inplace_vector`, `std::views` (lazy views of
/// run-time ranges). Group: \ref core.
///
/// **Implementations / variants.** A stream is one class template; each operation wraps the step of
/// the stream it is given. `drop` and `drop_while` advance the state when called.
///
/// **Guarantees.** `constexpr`, allocation-free. A stream is a value; taking from it does not change it.
/// Trivially copyable when its state and step are. `take<N>` calls the step exactly `N` times.
/// `lazy` evaluates `f` at most once per object (copies share nothing).
///
/// **Errors.** None. `filter`, `drop_while` and `take_while` on a stream with no element that passes
/// (or none that fails) do not terminate, as in Haskell; at compile time the compiler's step limit
/// stops them.
///
/// **Examples.** \snippet lazy_demo.cpp lazy
///
/// **Performance.** A take of a mapped, filtered stream is the loop that computes it.
/// `benchmarks/functional/lazy_benchmark.cpp`, `take<256>` of a filtered, mapped `iterate` against the
/// hand-written loop, fairness-checked: medians 509 ns and 530 ns, equal within noise (macOS x86-64 Xeon W-3245, Apple Clang 17, Release, 2026-11-26, 5 repetitions).
///
/// **Tests and demos.** `tests/functional/lazy_test.cpp`, `demos/functional/lazy_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<cstddef>`, `<functional>`, `<optional>`, `<type_traits>`, `<utility>`,
/// `functional/fold.hpp` and `containers/inplace_vector.hpp`.
///
/// **Known issues and gotchas.** A `lazy` keeps its value in a `mutable` member, so a `constexpr lazy`
/// variable cannot be forced at compile time (its value cannot change after it is created); create and
/// force it inside one constant evaluation, or use a plain `constexpr` value.
///
/// **Status.** Experimental, new in 0.27.0.

#ifndef snicholls_core_functional_lazy_hpp
#define snicholls_core_functional_lazy_hpp

#include <snicholls/core/containers/inplace_vector.hpp>
#include <snicholls/core/functional/fold.hpp>

#include <array>
#include <cstddef>
#include <functional>
#include <optional>
#include <type_traits>
#include <utility>

namespace snicholls::fn {

    /// \brief A thunk: `f()` evaluated the first time it is wanted, then kept (Haskell's call by need).
    /// \ingroup core
    /// \tparam F The function's type, called with no arguments.
    template <class F>
    class lazy {
    public:
        /// \brief The value's type.
        using value_type = std::remove_cvref_t<std::invoke_result_t<const F&>>;

        /// \brief Wrap `f`, unevaluated.
        /// \param f The function.
        constexpr explicit lazy(F f) noexcept(std::is_nothrow_move_constructible_v<F>) : f_(std::move(f)) {}

        /// \brief The value, computed on the first call.
        /// \return A reference to the kept value, valid while this object lives.
        /// \throws Whatever `f` throws; nothing is kept then.
        constexpr const value_type& force() const
        {
            if (!value_) value_.emplace(std::invoke(f_));
            return *value_;
        }

        /// \brief Whether the value has been computed.
        /// \return True after the first `force()`.
        [[nodiscard]] constexpr bool forced() const noexcept { return value_.has_value(); }

    private:
        F f_;
        mutable std::optional<value_type> value_;
    };

    /// \brief Deduction guide: `lazy{f}` holds a copy of `f`.
    /// \tparam F The function's type.
    template <class F>
    lazy(F) -> lazy<F>;

    /// \brief An infinite stream: a state and a step, `state -> std::pair{value, next state}`.
    /// \ingroup core
    ///
    /// Built by the operations in `namespace snicholls::fn::stream`; finished by `stream::take` and `stream::take_while`.
    /// \tparam S    The state's type.
    /// \tparam Step The step's type.
    template <class S, class Step>
    struct stream_of {
        S state;   ///< The current state.
        Step step; ///< `state -> {value, next state}`.

        /// \brief The first value and the next state.
        /// \return `step(state)`.
        constexpr auto next() const SNICHOLLS_FN_RETURNS(std::invoke(step, state))
    };

    /// \brief Deduction guide: `stream_of{state, step}`.
    /// \tparam S    The state's type.
    /// \tparam Step The step's type.
    template <class S, class Step>
    stream_of(S, Step) -> stream_of<S, Step>;

    /// \brief The stream operations, Haskell's list functions on infinite lists.
    /// \ingroup core
    namespace stream {

        namespace detail {

            template <class St>
            using value_t = std::remove_cvref_t<decltype(std::declval<const St&>().next().first)>;

            struct iterate_fn {
                template <class F, class X>
                constexpr auto operator()(F f, X x) const
                {
                    return stream_of{std::move(x), [f](const X& s) { return std::pair{s, std::invoke(f, s)}; }};
                }
            };

            struct repeat_fn {
                template <class X>
                constexpr auto operator()(X x) const
                {
                    return stream_of{std::move(x), [](const X& s) { return std::pair{s, s}; }};
                }
            };

            struct cycle_fn {
                template <class T, std::size_t N>
                    requires(N > 0)
                constexpr auto operator()(const std::array<T, N>& xs) const
                {
                    return stream_of{std::size_t{0}, [xs](std::size_t i) { return std::pair{xs[i], (i + 1) % N}; }};
                }
            };

            struct unfold_fn {
                template <class S, class Step>
                constexpr auto operator()(S seed, Step step) const
                {
                    return stream_of{std::move(seed), std::move(step)};
                }
            };

            struct map_fn {
                template <class F, class St>
                constexpr auto operator()(F f, St st) const
                {
                    return stream_of{st.state, [f, step = st.step](const auto& s) {
                                         auto r = std::invoke(step, s);
                                         return std::pair{std::invoke(f, std::move(r.first)), std::move(r.second)};
                                     }};
                }
            };

            struct filter_fn {
                template <class P, class St>
                constexpr auto operator()(P p, St st) const
                {
                    return stream_of{st.state, [p, step = st.step](const auto& s) {
                                         auto r = std::invoke(step, s);
                                         while (!std::invoke(p, std::as_const(r.first))) r = std::invoke(step, r.second);
                                         return r;
                                     }};
                }
            };

            struct zip_with_fn {
                template <class F, class A, class B>
                constexpr auto operator()(F f, A a, B b) const
                {
                    return stream_of{std::pair{a.state, b.state}, [f, sa = a.step, sb = b.step](const auto& s) {
                                         auto ra = std::invoke(sa, s.first);
                                         auto rb = std::invoke(sb, s.second);
                                         return std::pair{std::invoke(f, std::move(ra.first), std::move(rb.first)),
                                                          std::pair{std::move(ra.second), std::move(rb.second)}};
                                     }};
                }
            };

            struct make_pair_fn {
                template <class A, class B>
                constexpr std::pair<A, B> operator()(A a, B b) const
                {
                    return {std::move(a), std::move(b)};
                }
            };

            struct scanl_fn {
                template <class F, class Z, class St>
                constexpr auto operator()(F f, Z z, St st) const
                {
                    return stream_of{std::pair{std::move(z), st.state}, [f, step = st.step](const auto& s) {
                                         auto r = std::invoke(step, s.second);
                                         Z next = std::invoke(f, s.first, std::move(r.first));
                                         return std::pair{s.first, std::pair{std::move(next), std::move(r.second)}};
                                     }};
                }
            };

            struct drop_fn {
                template <class St>
                constexpr St operator()(std::size_t n, St st) const
                {
                    for (std::size_t i = 0; i < n; ++i) st.state = st.next().second;
                    return st;
                }
            };

            struct drop_while_fn {
                template <class P, class St>
                constexpr St operator()(P p, St st) const
                {
                    for (auto r = st.next(); std::invoke(p, std::as_const(r.first)); r = st.next()) st.state = std::move(r.second);
                    return st;
                }
            };

            struct head_fn {
                template <class St>
                constexpr auto operator()(const St& st) const SNICHOLLS_FN_RETURNS(st.next().first)
            };

            struct tail_fn {
                template <class St>
                constexpr St operator()(St st) const
                {
                    st.state = st.next().second;
                    return st;
                }
            };

            struct nth_fn {
                template <class St>
                constexpr auto operator()(const St& st, std::size_t n) const
                {
                    return drop_fn{}(n, st).next().first;
                }
            };

        } // namespace detail

        /// \brief `iterate(f, x)`: `x, f(x), f(f(x)), ...`.
        /// \ingroup core
        inline constexpr detail::iterate_fn iterate{};

        /// \brief `repeat(x)`: `x, x, x, ...`.
        /// \ingroup core
        inline constexpr detail::repeat_fn repeat{};

        /// \brief `cycle(xs)`: the elements of a non-empty `std::array`, over and over.
        /// \ingroup core
        inline constexpr detail::cycle_fn cycle{};

        /// \brief `unfold(seed, step)`: the stream of `step`, `state -> {value, next state}`, from `seed`.
        /// \ingroup core
        inline constexpr detail::unfold_fn unfold{};

        /// \brief `map(f, s)`: `f` of every value.
        /// \ingroup core
        inline constexpr detail::map_fn map{};

        /// \brief `filter(p, s)`: the values for which `p` holds.
        /// \ingroup core
        inline constexpr detail::filter_fn filter{};

        /// \brief `zip_with(f, a, b)`: `f(a0, b0), f(a1, b1), ...`.
        /// \ingroup core
        inline constexpr detail::zip_with_fn zip_with{};

        /// \brief `scanl(f, z, s)`: `z, f(z, s0), f(f(z, s0), s1), ...`.
        /// \ingroup core
        inline constexpr detail::scanl_fn scanl{};

        /// \brief `drop(n, s)`: the stream after its first `n` values (advanced now).
        /// \ingroup core
        inline constexpr detail::drop_fn drop{};

        /// \brief `drop_while(p, s)`: the stream from its first value for which `p` fails (advanced now).
        /// \ingroup core
        inline constexpr detail::drop_while_fn drop_while{};

        /// \brief `head(s)`: the first value.
        /// \ingroup core
        inline constexpr detail::head_fn head{};

        /// \brief `tail(s)`: the stream after its first value.
        /// \ingroup core
        inline constexpr detail::tail_fn tail{};

        /// \brief `nth(s, n)`: the value at position `n`, counting from 0 (Haskell's `s !! n`).
        /// \ingroup core
        inline constexpr detail::nth_fn nth{};

        /// \brief `zip(a, b)`: `std::pair{a0, b0}, std::pair{a1, b1}, ...`.
        /// \ingroup core
        /// \param a The first stream.
        /// \param b The second stream.
        /// \return The stream of pairs.
        template <class A, class B>
        constexpr auto zip(A a, B b)
        {
            return zip_with(detail::make_pair_fn{}, std::move(a), std::move(b));
        }

        /// \brief `take<N>(s)`: the first `N` values, computed now.
        /// \ingroup core
        /// \tparam N How many.
        /// \param s The stream.
        /// \return A `std::array` of the `N` values.
        template <std::size_t N, class St>
        [[nodiscard]] constexpr auto take(const St& s)
        {
            return fn::unfold<N>(s.state, s.step);
        }

        /// \brief `take_while<N>(p, s)`: the values before the first for which `p` fails, at most `N`, computed now.
        /// \ingroup core
        /// \tparam N The most values kept.
        /// \param p The test.
        /// \param s The stream.
        /// \return A `containers::inplace_vector` of the values.
        template <std::size_t N, class P, class St>
        [[nodiscard]] constexpr auto take_while(P p, const St& s)
        {
            containers::inplace_vector<detail::value_t<St>, N> out;
            auto state = s.state;
            while (out.size() < N) {
                auto r = std::invoke(s.step, state);
                if (!std::invoke(p, std::as_const(r.first))) break;
                out.push_back(std::move(r.first));
                state = std::move(r.second);
            }
            return out;
        }

    } // namespace stream

} // namespace snicholls::fn

#endif /* snicholls_core_functional_lazy_hpp */
