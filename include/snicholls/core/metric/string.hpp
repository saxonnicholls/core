//
//  string.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::metric::levenshtein<MaxLength>`: the edit distance between two sequences, allocation-free.
/// \ingroup core
///
/// **Overview.** The minimum number of single-element insertions, deletions and substitutions that turn one sequence
/// into another: names that differ by a typo, call signs, token sequences. `levenshtein<MaxLength>{}(a, b)` works on
/// any two `point`s whose elements compare with `==`: `std::string_view`s, `std::array`s, `std::span`s, `std::vector`s
/// of characters, integers or categories, mixed. Its one row of working storage is an `std::array` of
/// `MaxLength + 1` counts, so it never allocates. A sequence longer than `MaxLength` gives the largest `std::size_t`
/// rather than a wrong answer.
///
/// \snippet metric/string_demo.cpp string
///
/// **Relationships.** `metric/concepts.hpp`, `metric/binary.hpp` (`mismatches`: substitutions only, equal lengths).
/// Group: \ref core.
///
/// **Implementations.** The Wagner-Fischer recurrence over one row, the shorter sequence along it: O(m n) time,
/// O(min(m, n)) space.
///
/// **Guarantees.** `constexpr`, `noexcept` when the element reads and comparisons are, no allocation. A metric:
/// zero only for equal sequences, symmetric, and obeying the triangle inequality.
///
/// **Errors.** None; the largest `std::size_t` for a sequence beyond `MaxLength` (both longer is fine as long as the
/// shorter fits).
///
/// **Performance.** Not benchmarked.
///
/// **Tests and demos.** `tests/metric/string_test.cpp`, `demos/metric/string_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<cstddef>`, `<limits>`, `metric/concepts.hpp`.
///
/// **Known issues.** Elements are compared as code units: two encodings of one character differ.
///
/// **Status.** Experimental, new in 0.20.0.

#ifndef snicholls_core_metric_string_hpp
#define snicholls_core_metric_string_hpp

#include <snicholls/core/metric/concepts.hpp>

#include <array>
#include <cstddef>
#include <limits>

namespace snicholls::metric {

    /// \brief The Levenshtein (edit) distance, for sequences whose shorter is at most `MaxLength` long.
    /// \ingroup core
    /// \tparam MaxLength The longest shorter sequence; the working row holds `MaxLength + 1` counts.
    template <std::size_t MaxLength>
    struct levenshtein {
        static constexpr bool is_metric = true;  ///< It obeys the triangle inequality.

        /// \brief The distance. \tparam A A `point`. \tparam B A `point` whose elements compare with A's.
        /// \param a One sequence. \param b The other. \return The edit distance, or the largest `std::size_t` when
        /// the shorter sequence exceeds `MaxLength`.
        template <point A, point B>
        [[nodiscard]] constexpr std::size_t operator()(const A& a, const B& b) const noexcept(noexcept(coordinate(a, 0)) && noexcept(coordinate(b, 0)))
        {
            const std::size_t na = dimension(a), nb = dimension(b);
            if (na < nb)
                return run(b, nb, a, na);
            return run(a, na, b, nb);
        }

    private:
        // long along the outer loop, short along the row
        template <typename L, typename S>
        static constexpr std::size_t run(const L& lng, std::size_t nl, const S& sht, std::size_t ns) noexcept(noexcept(coordinate(lng, 0)) && noexcept(coordinate(sht, 0)))
        {
            if (ns > MaxLength)
                return std::numeric_limits<std::size_t>::max();
            std::array<std::size_t, MaxLength + 1> row{};
            for (std::size_t j = 0; j <= ns; ++j)
                row[j] = j;
            for (std::size_t i = 1; i <= nl; ++i) {
                std::size_t diagonal = row[0];
                row[0] = i;
                for (std::size_t j = 1; j <= ns; ++j) {
                    const std::size_t above = row[j];
                    const std::size_t substitute = diagonal + (coordinate(lng, i - 1) == coordinate(sht, j - 1) ? 0u : 1u);
                    const std::size_t remove = above + 1, insert = row[j - 1] + 1;
                    std::size_t best = substitute < remove ? substitute : remove;
                    best = insert < best ? insert : best;
                    row[j] = best;
                    diagonal = above;
                }
            }
            return row[ns];
        }
    };

} // namespace snicholls::metric

#endif /* snicholls_core_metric_string_hpp */
