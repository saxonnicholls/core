//
//  iterate.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief Wolfram's iteration primitives, `constexpr`: `table`, `nest`, `nest_list`, `nest_while`, `fixed_point`, `fold_list`.
/// \ingroup core
///
/// **Overview.**
///
/// | Wolfram                          | here                              |
/// | -------------------------------- | --------------------------------- |
/// | `Table[f[i], {i, 0, n - 1}]`     | `table<n>(f)`                     |
/// | `Nest[f, x, n]`                  | `nest(f, x, n)`                   |
/// | `NestList[f, x, n]`              | `nest_list<n>(f, x)`: n + 1 values |
/// | `NestWhile[f, x, test]`          | `nest_while(f, x, test, limit)`   |
/// | `FixedPoint[f, x]`               | `fixed_point(f, x, limit)`        |
/// | `FoldList[f, x, list]`           | `fold_list(f, x, list)`: n + 1 values |
///
/// Each is `constexpr`: a table of iterates (a recurrence, a schedule of rates, Newton's steps) can be built
/// by the compiler. For types, `meta::nest_list_t` and `meta::fold_list_t`. `Accumulate`, `Differences` and
/// `Range` are C++20's `constexpr` `std::partial_sum`, `std::adjacent_difference` and `std::iota`.
///
/// \snippet iterate_demo.cpp iterate
///
/// **Relationships.** `meta/type_list.hpp`; `thread_apply.hpp`; `combinatorics/subsets.hpp`. Group: \ref core.
///
/// **Guarantees.** `constexpr`, allocation-free; `noexcept` when `f` is. `nest_while` and `fixed_point` stop at
/// `limit` steps whatever happens.
///
/// **Errors.** None of their own: `fixed_point` reports, through `iterations`, whether it converged.
///
/// **Performance.** The calls to `f`.
///
/// **Tests and demos.** `tests/iterate_test.cpp`, `demos/iterate_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<cstddef>`, `<functional>`, `<type_traits>`, `<utility>`.
///
/// **Known issues.** `fixed_point` compares with `==`; for floating point, pass a `same` test with a tolerance.
///
/// **Status.** New in 0.3.0.

#ifndef snicholls_core_iterate_hpp
#define snicholls_core_iterate_hpp

#include <array>
#include <cstddef>
#include <functional>
#include <type_traits>
#include <utility>

namespace snicholls {

    /// \brief `Table[f[i], {i, 0, N - 1}]`: `{f(0), ..., f(N - 1)}`, each called with a `std::size_t`.
    /// \ingroup core
    /// \tparam N The number of values.
    /// \tparam F The callable's type.
    /// \param f The function.
    /// \return The table.
    template <std::size_t N, typename F>
    [[nodiscard]] constexpr auto table(F&& f)
    {
        return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            using R = std::remove_cvref_t<std::invoke_result_t<F&, std::size_t>>;
            return std::array<R, N>{std::invoke(f, Is)...};
        }(std::make_index_sequence<N>{});
    }

    /// \brief `Nest[f, x, n]`: `f` applied `n` times to `x`.
    /// \ingroup core
    /// \tparam F The callable's type.
    /// \tparam T The value's type.
    /// \param f The function, from `T` to `T`.
    /// \param x The start.
    /// \param n How many times.
    /// \return The result.
    template <typename F, typename T>
    [[nodiscard]] constexpr T nest(F&& f, T x, std::size_t n)
    {
        for (std::size_t i = 0; i < n; ++i)
            x = std::invoke(f, std::as_const(x));
        return x;
    }

    /// \brief `NestList[f, x, N]`: `{x, f(x), f(f(x)), ...}`, `N + 1` values.
    /// \ingroup core
    /// \tparam N How many applications.
    /// \tparam F The callable's type.
    /// \tparam T The value's type.
    /// \param f The function, from `T` to `T`.
    /// \param x The start.
    /// \return The iterates.
    template <std::size_t N, typename F, typename T>
    [[nodiscard]] constexpr std::array<T, N + 1> nest_list(F&& f, T x)
    {
        std::array<T, N + 1> out{};
        out[0] = x;
        for (std::size_t i = 1; i <= N; ++i)
            out[i] = std::invoke(f, std::as_const(out[i - 1]));
        return out;
    }

    /// \brief `NestWhile[f, x, test]`: apply `f` while `test` holds of the current value, at most `limit` times.
    /// \ingroup core
    /// \tparam F    The callable's type.
    /// \tparam T    The value's type.
    /// \tparam Test The test's type.
    /// \param f     The function.
    /// \param x     The start.
    /// \param test  Applied to each value before `f`; iteration stops when it fails.
    /// \param limit The most applications.
    /// \return The last value.
    template <typename F, typename T, typename Test>
    [[nodiscard]] constexpr T nest_while(F&& f, T x, Test&& test, std::size_t limit)
    {
        for (std::size_t i = 0; i < limit && std::invoke(test, std::as_const(x)); ++i)
            x = std::invoke(f, std::as_const(x));
        return x;
    }

    /// \brief What `fixed_point` found: the value, and how many applications it took.
    /// \ingroup core
    /// \tparam T The value's type.
    template <typename T>
    struct fixed_point_result {
        T value{};                  ///< The last value.
        std::size_t iterations = 0; ///< Applications made.
        bool converged = false;     ///< Whether two successive values were the same before the limit.
    };

    /// \brief `FixedPoint[f, x]`: apply `f` until two successive values are the same, at most `limit` times.
    /// \ingroup core
    /// \tparam F    The callable's type.
    /// \tparam T    The value's type.
    /// \tparam Same The comparison's type.
    /// \param f     The function.
    /// \param x     The start.
    /// \param limit The most applications.
    /// \param same  Whether two values are the same: `==` by default; a tolerance for floating point.
    /// \return The value, the count and whether it converged.
    template <typename F, typename T, typename Same = std::equal_to<>>
    [[nodiscard]] constexpr fixed_point_result<T> fixed_point(F&& f, T x, std::size_t limit, Same same = {})
    {
        fixed_point_result<T> out{x, 0, false};
        while (out.iterations < limit) {
            T next = std::invoke(f, std::as_const(out.value));
            ++out.iterations;
            if (std::invoke(same, std::as_const(out.value), std::as_const(next))) {
                out.value = std::move(next);
                out.converged = true;
                return out;
            }
            out.value = std::move(next);
        }
        return out;
    }

    /// \brief `FoldList[f, x, list]`: `{x, f(x, a0), f(f(x, a0), a1), ...}`, `N + 1` values.
    /// \ingroup core
    /// \tparam F The callable's type.
    /// \tparam T The accumulator's type.
    /// \tparam U The elements' type.
    /// \tparam N The number of elements.
    /// \param f    The function, `(T, U) -> T`.
    /// \param x    The start.
    /// \param list The elements.
    /// \return The running results.
    template <typename F, typename T, typename U, std::size_t N>
    [[nodiscard]] constexpr std::array<T, N + 1> fold_list(F&& f, T x, const std::array<U, N>& list)
    {
        std::array<T, N + 1> out{};
        out[0] = std::move(x);
        for (std::size_t i = 0; i < N; ++i)
            out[i + 1] = std::invoke(f, std::as_const(out[i]), list[i]);
        return out;
    }

} // namespace snicholls

#endif /* snicholls_core_iterate_hpp */
