//
//  constexpr_for.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Moved from ts-moveables (TSMoveables/utils/constexpr_for.hpp at d7e0f10).
//  The definitions are unchanged; the guard and documentation are core's.
//

/// \file
/// \brief `constexpr_for`, `constexpr_nest`, `unrolled_for`: compile-time loop unrolling.
/// \ingroup core
///
/// **Overview.** `constexpr_for<Start, End, Inc>` expands into exactly the calls it
/// needs and leaves no loop at all: the bounds are template parameters, so there is
/// no counter, no compare and no branch. The body is handed
/// `std::integral_constant<T, I>`, not a bare `T`, so `I.value` is a constant
/// expression inside the body and can index a tuple or be a template argument,
/// while `I` still converts to its value for ordinary arithmetic:
///
/// \snippet constexpr_for_demo.cpp tuple
///
/// `constexpr_nest<E0, E1, ...>` unrolls nested loops to any depth, row-major, with
/// a constant index per level. `unrolled_for<Unroll>(first, last, body)` walks a
/// *run-time* range with the body emitted `Unroll` times per iteration and a tail.
///
/// **Relationships.** For iterating *values* (an index range, a tuple-like by
/// index). To operate on a pack of *types*, use `type_list` (CLAUDE.md "Idioms").
///
/// **Guarantees.** Expansion is a fold over an `index_sequence`, depth 1 whatever
/// the count, so there is no instantiation-depth cliff. The iteration count is
/// computed in `long long`, so a negative step against unsigned bounds cannot wrap.
/// A step of 0 is a compile error. Trailing arguments reach every call as lvalues,
/// never forwarded, so a body cannot move from the same object twice.
///
/// **Performance.** At -O2 compilers already unroll simple counted loops, and doing
/// it by hand can lose. Reach for `constexpr_for` when the body must be instantiated
/// per index; reach for `unrolled_for` only with a benchmark in hand. Not
/// benchmarked in core; ts-moveables' `make bench-parallel` compares them.
///
/// **Tests and demos.** `tests/constexpr_for_test.cpp`, `demos/constexpr_for_demo.cpp`.
///
/// **Dependencies.** `<cstddef>`, `<type_traits>`, `<utility>`.
///
/// **Known issues.** `constexpr_nest<8, 8, 8>` is 512 copies of the body: for small
/// fixed extents, not whole arrays.
///
/// **Status.** Stable.

// Implementation notes, as written in ts-moveables:
//
//      constexpr_for<0, 4>      ([&](auto I) { use<I.value>(); });   // fully unrolled
//      constexpr_for<0, 10, 2>  ([&](auto I) { even(I); });          // with a step
//      constexpr_for<10, 0, -1> ([&](auto I) { down(I); });          // and backwards
//      constexpr_for<0, 3>      (f, a, b);                           // f(I, a, b)
//      unrolled_for<4>(0, n, [&](std::size_t i) { out[i] = g(in[i]); });
//
//  constexpr_for<Start, End, Inc> expands into exactly the calls it needs and
//  leaves no loop at all - the bounds are template parameters, so there is no
//  counter, no compare and no branch. unrolled_for takes a RUN-TIME range and
//  emits the body Unroll times per iteration, with a tail for the remainder.
//
//  ------------------------------------------------------ the index is a type
//
//  The body is handed std::integral_constant<T, I>, not a bare T, and that is
//  the whole reason this is worth having over a plain loop.
//
//  A function parameter is never a constant expression, however constant the
//  value passed in was - so a body taking `std::size_t i` cannot write
//  std::get<i>(tuple) or use i as a template argument, which is usually the
//  entire point of unrolling by hand. Handing it a type carrying the index
//  fixes that: `I.value` is a constant expression inside the body, while `I`
//  still converts implicitly to its underlying type, so the ordinary
//  arithmetic spelling keeps working:
//
//      constexpr_for<0, N>([&](auto I) { sum += v[I]; });                // fine
//      constexpr_for<0, N>([&](auto I) { use(std::get<I.value>(tup)); }); // also fine
//
//  The index type is decltype(Start), so constexpr_for<0, 4> yields int and
//  constexpr_for<std::size_t{0}, std::size_t{4}> yields std::size_t. Say which
//  you want when it matters.
//
//  ------------------------------------------------------------ expansion, not
//                                                                  recursion
//  The obvious implementation recurses - body, then constexpr_for<Start+Inc,
//  End, Inc> - and it is shorter. This expands an index_sequence through a fold
//  instead, because the recursive form instantiates one template per iteration:
//  template depth grows with the count, a few hundred iterations meets the
//  compiler's instantiation-depth limit, and compile time grows superlinearly
//  well before that. The fold is depth 1 whatever the count. Same unrolled
//  output, no cliff to fall off.
//
//  ------------------------------------------------------------ extra arguments
//
//  Trailing arguments are passed through to every call as LVALUES, never
//  forwarded. Forwarding would move the same object once per iteration, which
//  is a use-after-move that compiles quietly; if a body needs to consume
//  something it should capture it instead.
//
//  ------------------------------------------------------------- when to reach
//
//  Be sceptical of unrolled_for as a speed tool. At -O2 the compiler already
//  unrolls simple counted loops, and doing it by hand can lose: more code in
//  the instruction cache, more live registers, and a tail that confuses the
//  vectoriser. `make bench-parallel` prints unrolled against not so the claim
//  stays falsifiable rather than folkloric.
//
//  Where it does earn its place: when the body must be instantiated per index -
//  a tuple element, a template argument, a fixed-size matrix row - which no
//  compiler unroll can do for you, because that is a language requirement
//  rather than an optimisation. Reach for constexpr_for for that; reach for
//  unrolled_for only with a benchmark in hand.
//

#ifndef snicholls_core_constexpr_for_hpp
#define snicholls_core_constexpr_for_hpp

#include <cstddef>
#include <type_traits>
#include <utility>

namespace snicholls
{
    namespace detail
    {
        // How many iterations [Start, End) by Inc actually performs. Computed
        // in long long so a negative Inc against unsigned bounds cannot wrap -
        // the arithmetic that makes a hand-rolled version of this silently
        // produce 18 quintillion iterations.
        constexpr std::size_t constexpr_for_count(long long start, long long end, long long inc)
        {
            const long long span = (inc > 0) ? (end - start) : (start - end);
            const long long step = (inc > 0) ? inc : -inc;
            return span > 0 ? static_cast<std::size_t>((span + step - 1) / step) : 0u;
        }

        template <auto Start, auto Inc, typename F, std::size_t... I, typename... Args>
        constexpr void constexpr_for_impl(F&& f, std::index_sequence<I...>, Args&... args)
        {
            using T = decltype(Start);
            // Fold over the comma operator: one call per index, in order, and
            // nothing at all generated for an empty sequence.
            (static_cast<void>(f(std::integral_constant<
                                     T, static_cast<T>(static_cast<long long>(Start) +
                                                       static_cast<long long>(I) *
                                                           static_cast<long long>(Inc))>{},
                                 args...)),
             ...);
        }
    } // namespace detail

    /// \brief Call `f(I, args...)` for each `I` in [Start, End) stepping by `Inc`, fully unrolled.
    /// \tparam Start The first index; its type is the index type.
    /// \tparam End   The open end of the range.
    /// \tparam Inc   The step; negative counts down; 0 is a compile error.
    /// \tparam F     The body: callable with `std::integral_constant<decltype(Start), I>`, then `args...`.
    /// \tparam Args  Trailing argument types.
    /// \param f      The body.
    /// \param args   Passed to every call as lvalues.
    // Invoke f(I, args...) for each I in [Start, End) stepping by Inc, fully
    // unrolled at compile time. Inc may be negative to count down. I is an
    // integral_constant<decltype(Start), ...>.
    template <auto Start, auto End, auto Inc = 1, typename F, typename... Args>
    constexpr void constexpr_for(F&& f, Args&&... args)
    {
        static_assert(Inc != 0, "constexpr_for with a step of 0 would never terminate");
        constexpr std::size_t n = detail::constexpr_for_count(
            static_cast<long long>(Start), static_cast<long long>(End),
            static_cast<long long>(Inc));
        detail::constexpr_for_impl<Start, Inc>(std::forward<F>(f),
                                               std::make_index_sequence<n>{}, args...);
    }

    /// \brief Call `f(I, args...)` for each `I` in [0, Count), fully unrolled.
    /// \tparam Count The number of iterations; its type is the index type.
    /// \tparam F     The body.
    /// \tparam Args  Trailing argument types.
    /// \param f      The body.
    /// \param args   Passed to every call as lvalues.
    // Invoke f(I, args...) for each I in [0, Count) - the common case, shorter.
    // Disjoint from the overload above: that one cannot deduce End, this one
    // cannot take a second non-type argument.
    template <auto Count, typename F, typename... Args>
    constexpr void constexpr_for(F&& f, Args&&... args)
    {
        constexpr_for<static_cast<decltype(Count)>(0), Count, 1>(std::forward<F>(f), args...);
    }

    namespace detail
    {
        // N-deep nest, peeled one extent at a time. The recursion here is over
        // DEPTH (2, 3, 4...), not over iteration count, so it is three or four
        // instantiations deep rather than thousands - the cliff the flat
        // constexpr_for avoids by folding does not exist on this axis.
        template <std::size_t... Extents>
        struct nest;

        template <>
        struct nest<> {
            template <typename F, typename... Acc>
            static constexpr void run(F& f, Acc... acc) { f(acc...); }
        };

        template <std::size_t E0, std::size_t... Rest>
        struct nest<E0, Rest...> {
            template <typename F, typename... Acc>
            static constexpr void run(F& f, Acc... acc)
            {
                constexpr_for<std::size_t{0}, E0>(
                    [&](auto I) { nest<Rest...>::run(f, acc..., I); });
            }
        };
    } // namespace detail

    /// \brief Nested loops, every level unrolled: `f(I, J, ...)` for every point, row-major.
    /// \tparam Extents The extent of each level, outermost first.
    /// \tparam F       The body: one `integral_constant<std::size_t, ...>` per level.
    /// \param f        The body.
    // Nested loops, every level unrolled, to any depth:
    //
    //     constexpr_nest<2, 3>   ([&](auto I, auto J)         { m[I][J] += a[I]*b[J]; });
    //     constexpr_nest<2, 3, 4>([&](auto I, auto J, auto K) { t[I][J][K] = 0; });
    //
    // The body is called once per point of the cartesian product, in row-major
    // order (last extent varies fastest), with one integral_constant per level
    // - so every index is a constant expression and a fixed-size kernel can be
    // written with no loop and no run-time indexing at all.
    //
    // Mind the multiplication: <8,8,8> is 512 expansions of the body, and the
    // object code grows with it. This is for small fixed extents - a 4x4
    // transform, a tile of a matrix kernel - not for whole arrays.
    template <std::size_t... Extents, typename F>
    constexpr void constexpr_nest(F&& f)
    {
        detail::nest<Extents...>::run(f);
    }

    /// \brief Walk a run-time range [first, last), calling `body(i, args...)`, unrolled `Unroll` times with a tail.
    /// \tparam Unroll Copies of the body per iteration, at least 1.
    /// \tparam Index  An integral index type.
    /// \tparam Body   Callable with `(Index, Args&...)`.
    /// \tparam Args   Trailing argument types.
    /// \param first   The first index.
    /// \param last    The open end; an inverted range does nothing.
    /// \param body    The body.
    /// \param args    Passed to every call as lvalues.
    // Walk a RUN-TIME range [first, last), calling body(i, args...) for each i,
    // with the body emitted Unroll times per iteration and a tail loop for the
    // remainder.
    //
    // body receives a plain Index here, not an integral_constant: the value is
    // genuinely a run-time one, and pretending otherwise would be a lie that
    // only compiles when the bounds happen to be constant.
    template <std::size_t Unroll = 4, typename Index, typename Body, typename... Args>
    constexpr void unrolled_for(Index first, Index last, Body body, Args&... args)
    {
        static_assert(std::is_integral_v<Index>, "unrolled_for needs an integral index");
        static_assert(Unroll >= 1, "unrolled_for needs an unroll factor of at least 1");

        if (!(first < last))
            return;

        const auto count = static_cast<std::size_t>(last - first);
        const auto whole = count - (count % Unroll);
        const Index main_end = first + static_cast<Index>(whole);

        Index i = first;
        for (; i < main_end; i += static_cast<Index>(Unroll))
            constexpr_for<std::size_t{0}, Unroll>(
                [&](auto K) { body(i + static_cast<Index>(K.value), args...); });
        for (; i < last; ++i)                       // the tail Unroll did not divide
            body(i, args...);
    }
} // namespace snicholls

#endif /* snicholls_core_constexpr_for_hpp */
