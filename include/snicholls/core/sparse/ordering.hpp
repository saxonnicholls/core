//
//  ordering.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::sparse::amd` and `reverse_cuthill_mckee`: fill-reducing and bandwidth-reducing orderings of a symmetric sparse matrix, with the permutation helpers, all `constexpr` over the caller's storage.
/// \ingroup core
///
/// **Overview.** Factorising a sparse symmetric matrix `A = L D Lᵀ` fills in: `L` has non-zeros where `A` has none,
/// and how many depends on the order of the rows. A good order costs a fraction of the factorisation and can cut
/// `nnz(L)` (and the flops, which go as the sum of the squares of the column counts) by orders of magnitude. Every
/// sparse direct solver for optimisation (OSQP's QDLDL, Clarabel, MOSEK, CHOLMOD) orders its KKT system once, by
/// approximate minimum degree, and reuses the order for every refactorisation.
///
/// - `amd(a, perm, work)`: approximate minimum degree (Amestoy, Davis and Duff, 1996) for the symmetric matrix whose
///   upper triangle is `a`. Writes `perm[new] = old`. Use it before `ldl_symbolic` (`sparse/ldl.hpp`).
/// - `reverse_cuthill_mckee(a, perm, work)`: the bandwidth-reducing order (George and Liu, 1981). Fill is usually
///   worse than AMD's; use it for banded or profile solvers, or as a comparison.
/// - `inverse_permutation`, `is_permutation`, `permute` (`y[k] = x[perm[k]]`), `inverse_permute`
///   (`y[perm[k]] = x[k]`): moving vectors between the original and the permuted numbering.
///
/// The permuted matrix itself is `symmetric_permute_upper` of `sparse/csc.hpp`.
///
/// \snippet sparse/ordering_demo.cpp ordering
///
/// **Relationships.** `sparse/csc.hpp` (`csc_view`, `symmetric_permute_upper`), `sparse/ldl.hpp` (the factorisation
/// that the order is for), `result.hpp`. Group: \ref core.
///
/// **Implementations.** `amd` works on the quotient graph: eliminated variables become *elements* (cliques stored as
/// their variable lists), so the graph never grows beyond the storage of `A + Aᵀ`. On each step it takes a variable
/// of least approximate degree, forms its element `Lme` from the variable's own list and the elements it touches
/// (absorbing them), and updates the degrees of `Lme`'s variables by the bound of Amestoy, Davis and Duff,
/// `d̄ᵢ = min(n - k, d̄ᵢ' + |Lme \ i|, |Aᵢ \ i| + |Lme \ i| + Σ_{e ∈ Eᵢ \ me} |Le \ Lme|)`, with `|Le \ Lme|` for every
/// element at once in one pass over `Lme`. It also does
/// - *aggressive absorption*: an element whose variables all lie in `Lme` is absorbed into it;
/// - *mass elimination*: a variable adjacent to nothing but `Lme` is eliminated with the pivot;
/// - *supervariables*: variables of `Lme` with the same adjacency (found by a hash of their lists, then compared)
///   are merged and eliminated together, so degree updates cost per supervariable;
/// - *dense rows*: a row with more than `max(16, amd_options::dense · √n)` off-diagonal entries (`dense` = 10 by
///   default, as AMD's) is removed before the ordering starts and ordered last, so a budget constraint or an arrow
///   head does not make every degree large.
/// The quotient graph is compacted in place when its free space runs out. Ties go to the variable whose degree last
/// changed, the lowest index first at the start. `reverse_cuthill_mckee` takes, for each connected component, a
/// pseudo-peripheral start node (repeated breadth-first search from a node of least degree in the last level, until
/// the eccentricity stops growing), orders breadth-first with neighbours in ascending degree, and reverses the whole.
///
/// **Guarantees.** `constexpr`, `noexcept`, no allocation, no recursion. Every valid input gives a permutation
/// (`is_permutation(perm)` holds afterwards). `amd` is O(nnz) storage and, in practice, close to O(nnz(A) · α) time;
/// `reverse_cuthill_mckee` is O(nnz + n) storage and O(nnz · d) time for the largest degree `d` (neighbours are sorted
/// by insertion). Fill: on the matrices of `tests/sparse/ordering_test.cpp` (grids, arrows, random SPD and KKT
/// matrices) `amd`'s `nnz(L)` is within a factor 1.25 of an exact minimum degree ordering's (one with true external
/// degrees on the explicit elimination graph), and below natural order's and reverse Cuthill-McKee's on all but
/// the matrices already in their best order.
///
/// **Errors.** `sparse_error::dimension_mismatch`: `a` is not square. `sparse_error::invalid_index`: `a` is not a
/// valid CSC matrix (`is_valid`), or `perm` is not a permutation (the permutation helpers).
/// `sparse_error::insufficient_capacity`: `perm` or `work` is shorter than documented (for `amd`, shorter than the
/// least it accepts).
///
/// **Performance.** Not benchmarked by a Google Benchmark group of its own. Timed by hand (`-O3`, Apple clang 17,
/// Xeon W-3245, 2026-11-23) on the 100 x 100 grid Laplacian: `amd` 2.6 ms, `reverse_cuthill_mckee` 0.31 ms.
/// `benchmarks/sparse/ldl_benchmark.cpp` times the factorisation in natural and in AMD order (on that grid AMD cuts
/// `nnz(L)` from 990,099 to 186,098 and the numeric factorisation from 52 ms to 7.7 ms).
///
/// **Tests and demos.** `tests/sparse/ordering_test.cpp`, `demos/sparse/ordering_demo.cpp`.
///
/// **Dependencies.** `<cstddef>`, `<cstdint>`, `<limits>`, `<span>`, `<type_traits>`, `sparse/csc.hpp`, `result.hpp`.
///
/// **Known issues.** Entries below the diagonal of `a` are ignored, and so is the diagonal: the orderings see the
/// pattern of the strict upper triangle, mirrored. The order is not postordered (the elimination tree's postorder has
/// the same fill; a supernodal code would want it). The constant evaluator's step limit bounds a compile-time `amd`
/// to small problems (a few hundred entries).
///
/// **Status.** Experimental, new in 0.21.0.

#ifndef snicholls_core_sparse_ordering_hpp
#define snicholls_core_sparse_ordering_hpp

#include <snicholls/core/result.hpp>
#include <snicholls/core/sparse/csc.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <type_traits>

namespace snicholls::sparse {

    /// \brief The choices of `amd`.
    /// \ingroup core
    struct amd_options {
        double dense = 10.0;      ///< A row with more than `max(16, dense · √n)` off-diagonal entries is ordered last; negative to treat no row as dense.
        bool aggressive = true;   ///< Absorb every element whose variables all lie in the new element, not only those of the pivot.
    };

    /// \brief The working storage `amd` should have, in indices.
    /// \ingroup core
    /// \param n The order of the matrix.
    /// \param nnz The entries stored in its upper triangle (`a.nnz()`).
    /// \return `14n + 1 + 2·nnz + ⌊2·nnz / 5⌋`: twelve arrays of `n`, one of `n + 1`, and the quotient graph (`A + Aᵀ`
    /// without the diagonal) with a fifth and `n` of elbow room. `amd` accepts down to `13n + 2 + 2m`, `m` the entries
    /// strictly above the diagonal.
    [[nodiscard]] constexpr std::size_t amd_workspace_size(std::size_t n, std::size_t nnz) noexcept { return 14 * n + 1 + 2 * nnz + (2 * nnz) / 5; }

    /// \brief The working storage `reverse_cuthill_mckee` needs, in indices.
    /// \ingroup core
    /// \param n The order of the matrix.
    /// \param nnz The entries stored in its upper triangle (`a.nnz()`).
    /// \return `4n + 1 + 2·nnz`: the adjacency of `A + Aᵀ` and three arrays of `n`.
    [[nodiscard]] constexpr std::size_t rcm_workspace_size(std::size_t n, std::size_t nnz) noexcept { return 4 * n + 1 + 2 * nnz; }

    namespace detail::ordering {
        // Node states of the quotient graph
        inline constexpr int variable = 0;   // a principal variable, not yet eliminated
        inline constexpr int merged = 1;     // merged into a supervariable, or eliminated with a pivot (parent says which)
        inline constexpr int element = 2;    // an eliminated variable, now an element
        inline constexpr int absorbed = 3;   // an element absorbed into another
        inline constexpr int dense = 4;      // a dense row, removed and ordered last

        // √x by Newton's iteration, for the dense-row threshold (constexpr, and exact enough for a threshold)
        constexpr double root(double x) noexcept
        {
            if (!(x > 0))
                return 0;
            double r = x > 1 ? x : 1;
            for (int i = 0; i < 80; ++i)
                r = 0.5 * (r + x / r);
            return r;
        }

        // Calls f(i, j) for every strict upper entry (i < j) of a
        template <typename T, typename I, typename F>
        constexpr void for_each_strict_upper(const csc_view<T, I>& a, F&& f) noexcept
        {
            for (std::size_t j = 0; j < a.cols; ++j)
                for (std::size_t q = detail::at(a.col_ptr[j]); q < detail::at(a.col_ptr[j + 1]); ++q) {
                    const std::size_t i = detail::at(a.row_idx[q]);
                    if (i < j)
                        f(i, j);
                }
        }
    } // namespace detail::ordering

    /// \brief The approximate minimum degree ordering of a symmetric matrix given as its upper triangle.
    /// \ingroup core
    /// \tparam T The value type (unused: only the pattern is read).
    /// \tparam I The index type.
    /// \param a The upper triangle of the matrix, square and valid (`is_valid`); the diagonal and any entries below it
    /// are ignored.
    /// \param perm Receives the order: `perm[new] = old`, `a.cols` entries.
    /// \param work Working storage: `amd_workspace_size(a.cols, a.nnz())` indices. The least accepted is
    /// `13n + 2 + 2m`, for `m` the entries strictly above the diagonal; below the recommended size the quotient graph
    /// compacts itself more often (the order is the same).
    /// \param options The dense-row threshold and aggressive absorption.
    /// \return Nothing, or a `sparse_error`: `dimension_mismatch` (not square), `invalid_index` (not valid CSC),
    /// `insufficient_capacity` (`perm` or `work` too short).
    template <typename T, typename I>
    [[nodiscard]] constexpr result<void, sparse_error> amd(const csc_view<T, I>& a, std::span<std::remove_const_t<I>> perm, std::span<std::remove_const_t<I>> work,
                                                           amd_options options = {}) noexcept
    {
        using J = std::remove_const_t<I>;
        namespace o = detail::ordering;
        const std::size_t n = a.cols;
        if (a.rows != n)
            return err(sparse_error::dimension_mismatch);
        if (!is_valid(a))
            return err(sparse_error::invalid_index);
        if (perm.size() < n || work.size() < 13 * n + 1)
            return err(sparse_error::insufficient_capacity);
        if (n == 0)
            return ok();

        std::size_t offset = 0;
        auto take = [&](std::size_t m) {
            const auto s = work.subspan(offset, m);
            offset += m;
            return s;
        };
        // pe/len: each node's list in iw; elen: the elements at the front of a variable's list; nv: a supervariable's
        // size (negated while in Lme); degree: a variable's approximate degree, an element's weighted size; w: marks;
        // head/next/last: the degree lists; hhead/hnext: the hash buckets; parent: what a node was merged into
        const auto pe = take(n), len = take(n), elen = take(n), state = take(n), nv = take(n), degree = take(n), w = take(n), next = take(n), last = take(n),
                   hhead = take(n), hnext = take(n), parent = take(n), head = take(n + 1);
        const auto iw = work.subspan(offset);
        const J iwlen = iw.size() > static_cast<std::size_t>(std::numeric_limits<J>::max()) ? std::numeric_limits<J>::max() : static_cast<J>(iw.size());
        constexpr J none = -1;
        const J nn = static_cast<J>(n);

        // The graph of A + Aᵀ without the diagonal; rows with more than the threshold's entries are dense
        for (std::size_t i = 0; i < n; ++i)
            len[i] = 0;
        std::size_t entries = 0;
        o::for_each_strict_upper(a, [&](std::size_t i, std::size_t j) {
            ++len[i];
            ++len[j];
            entries += 2;
        });
        if (iw.size() <= entries)
            return err(sparse_error::insufficient_capacity);   // the graph of A + Aᵀ and one free entry, at the least
        std::size_t limit = n;
        if (options.dense >= 0) {
            const double t = options.dense * o::root(static_cast<double>(n));
            limit = t < 16 ? 16 : t >= static_cast<double>(n) ? n : static_cast<std::size_t>(t);
        }
        J ndense = 0;
        for (std::size_t i = 0; i < n; ++i) {
            state[i] = static_cast<std::size_t>(len[i]) > limit ? o::dense : o::variable;
            ndense += state[i] == o::dense ? 1 : 0;
            len[i] = 0;
        }
        o::for_each_strict_upper(a, [&](std::size_t i, std::size_t j) {
            if (state[i] != o::dense && state[j] != o::dense) {
                ++len[i];
                ++len[j];
            }
        });
        J pfree = 0;
        for (std::size_t i = 0; i < n; ++i) {
            pe[i] = pfree;
            degree[i] = pfree;   // the fill cursor, for now
            pfree += len[i];
        }
        o::for_each_strict_upper(a, [&](std::size_t i, std::size_t j) {
            if (state[i] != o::dense && state[j] != o::dense) {
                iw[detail::at(degree[i]++)] = static_cast<J>(j);
                iw[detail::at(degree[j]++)] = static_cast<J>(i);
            }
        });
        for (std::size_t i = 0; i <= n; ++i)
            head[i] = none;
        for (std::size_t i = 0; i < n; ++i) {
            elen[i] = 0;
            nv[i] = 1;
            w[i] = 1;
            parent[i] = none;
            next[i] = last[i] = hhead[i] = hnext[i] = none;
            degree[i] = len[i];
        }

        auto unlink = [&](J i) {
            if (next[i] != none)
                last[next[i]] = last[i];
            if (last[i] != none)
                next[last[i]] = next[i];
            else
                head[degree[i]] = next[i];
        };
        auto link = [&](J i, J d) {
            degree[i] = d;
            last[i] = none;
            next[i] = head[d];
            if (head[d] != none)
                last[head[d]] = i;
            head[d] = i;
        };
        J wflg = 2;
        auto reserve_marks = [&](J need) {
            if (wflg > std::numeric_limits<J>::max() - need) {
                for (std::size_t x = 0; x < n; ++x)
                    w[x] = 1;
                wflg = 2;
            }
        };
        // Compacts the live lists to the front of iw, then the partly built element [lo, hi) after them
        auto compact = [&](J& lo, J& hi) {
            for (J j = 0; j < nn; ++j)
                if ((state[j] == o::variable || state[j] == o::element) && len[j] > 0) {
                    const J first = iw[pe[j]];
                    iw[pe[j]] = -(j + 1);
                    pe[j] = first;
                }
            J dst = 0;
            for (J src = 0; src < lo;) {
                const J v = iw[src];
                if (v >= 0) {
                    ++src;
                    continue;
                }
                const J j = -v - 1;
                iw[src] = 0;
                iw[dst] = pe[j];
                pe[j] = dst;
                for (J t = 1; t < len[j]; ++t)
                    iw[dst + t] = iw[src + t];
                dst += len[j];
                src += len[j];
            }
            for (J t = lo; t < hi; ++t)
                iw[dst + (t - lo)] = iw[t];
            hi = dst + (hi - lo);
            lo = dst;
        };

        for (J i = nn; i-- > 0;)
            if (state[i] == o::variable)
                link(i, degree[i]);

        const J neff = nn - ndense;
        J k = 0, mindeg = 0, lemax = 0;
        std::size_t steps = 0;
        while (k < neff) {
            // 1. The pivot: a variable of least approximate degree
            while (mindeg < nn && head[mindeg] == none)
                ++mindeg;
            if (mindeg >= nn)
                break;   // not reached on a consistent graph
            const J p = head[mindeg];
            unlink(p);
            const J elenp = elen[p], nvpiv = nv[p], me = p;
            k += nvpiv;
            perm[steps++] = p;
            nv[p] = -nvpiv;

            // 2. The new element Lme = (Aₚ ∪ ⋃_{e ∈ Eₚ} Le) \ p, its variables flagged by a negative nv
            J degme = 0, pme1 = 0, pme2 = 0;
            auto add = [&](J j) {
                degme += nv[j];
                nv[j] = -nv[j];
                iw[pme2++] = j;
                unlink(j);
            };
            if (elenp == 0) {
                // Only variables: Lme is p's own list, compacted in place
                pme1 = pme2 = pe[p];
                for (J q = pe[p]; q < pe[p] + len[p]; ++q) {
                    const J j = iw[q];
                    if (state[j] == o::variable && nv[j] > 0)
                        add(j);
                }
            } else {
                pme1 = pme2 = pfree;
                for (J kk = 0; kk < len[p]; ++kk) {
                    const J e = iw[pe[p] + kk];
                    if (kk < elenp) {
                        if (state[e] != o::element)
                            continue;
                        for (J done = 0; done < len[e];) {
                            const J j = iw[pe[e] + done];
                            ++done;
                            if (state[j] != o::variable || nv[j] <= 0)
                                continue;
                            if (pme2 >= iwlen) {
                                pe[e] += done;   // keep only e's unread part through the compaction
                                len[e] -= done;
                                done = 0;
                                compact(pme1, pme2);
                                if (pme2 >= iwlen)
                                    return err(sparse_error::insufficient_capacity);
                            }
                            add(j);
                        }
                        state[e] = o::absorbed;
                        parent[e] = me;
                    } else if (state[e] == o::variable && nv[e] > 0) {
                        if (pme2 >= iwlen) {
                            compact(pme1, pme2);
                            if (pme2 >= iwlen)
                                return err(sparse_error::insufficient_capacity);
                        }
                        add(e);
                    }
                }
                pfree = pme2;
            }
            state[me] = o::element;
            pe[me] = pme1;
            len[me] = pme2 - pme1;
            lemax = degme > lemax ? degme : lemax;

            // 3. |Le \ Lme| for every element touching Lme: w[e] - wflg, from w[e] = wflg + |Le| less each shared variable
            reserve_marks(lemax + (pme2 - pme1) + 4);
            for (J q = pme1; q < pme2; ++q) {
                const J i = iw[q], nvi = -nv[i];
                for (J t = pe[i]; t < pe[i] + elen[i]; ++t) {
                    const J e = iw[t];
                    if (state[e] != o::element)
                        continue;
                    w[e] = w[e] >= wflg ? w[e] - nvi : degree[e] + wflg - nvi;
                }
            }

            // 4. Degree update, pruning, aggressive absorption, mass elimination; me goes to the front of each list
            for (J q = pme1; q < pme2; ++q) {
                const J i = iw[q];
                const J p1 = pe[i], p2 = p1 + elen[i], pend = p1 + len[i];
                J pn = p1, deg = 0;
                for (J t = p1; t < p2; ++t) {
                    const J e = iw[t];
                    if (state[e] != o::element)
                        continue;
                    const J dext = w[e] - wflg;
                    if (dext > 0) {
                        deg += dext;
                        iw[pn++] = e;
                    } else if (options.aggressive) {
                        state[e] = o::absorbed;
                        parent[e] = me;
                    } else
                        iw[pn++] = e;
                }
                J ne = pn - p1;
                for (J t = p2; t < pend; ++t) {
                    const J j = iw[t];
                    if (state[j] == o::variable && nv[j] > 0) {
                        deg += nv[j];
                        iw[pn++] = j;
                    }
                }
                const J nvar = pn - p1 - ne;
                if (ne == 0 && nvar == 0) {
                    // Adjacent to nothing but me: eliminated with the pivot
                    const J nvi = -nv[i];
                    degme -= nvi;
                    k += nvi;
                    nv[i] = 0;
                    state[i] = o::merged;
                    parent[i] = me;
                    len[i] = elen[i] = 0;
                    continue;
                }
                if (pn < pend) {
                    if (nvar > 0)
                        iw[pn] = iw[p1 + ne];
                    if (ne > 0)
                        iw[p1 + ne] = iw[p1];
                    iw[p1] = me;
                    ++ne;
                    ++pn;
                }
                elen[i] = ne;
                len[i] = pn - p1;
                degree[i] = deg < degree[i] ? deg : degree[i];   // the external part, bounded by the previous degree
                std::size_t h = 0;
                for (J t = p1; t < pn; ++t)
                    h = (h + static_cast<std::size_t>(iw[t])) % n;
                last[i] = static_cast<J>(h);
            }
            wflg += lemax + 1;

            // 5. Supervariables: variables of Lme with the same list (hash, then compare) are merged
            for (J q = pme1; q < pme2; ++q) {
                const J i = iw[q];
                if (state[i] != o::variable)
                    continue;
                hnext[i] = hhead[last[i]];
                hhead[last[i]] = i;
            }
            for (J q = pme1; q < pme2; ++q) {
                const J i = iw[q];
                if (state[i] != o::variable || hhead[last[i]] == none)
                    continue;
                const J chain = hhead[last[i]];
                hhead[last[i]] = none;
                for (J x = chain; x != none; x = hnext[x]) {
                    if (state[x] != o::variable)
                        continue;
                    for (J t = pe[x]; t < pe[x] + len[x]; ++t)
                        w[iw[t]] = wflg;
                    J prev = x;
                    for (J y = hnext[x]; y != none; y = hnext[y]) {
                        bool same = state[y] == o::variable && len[y] == len[x] && elen[y] == elen[x];
                        for (J t = pe[y]; same && t < pe[y] + len[y]; ++t)
                            same = w[iw[t]] == wflg;
                        if (same) {
                            nv[x] += nv[y];   // both negative while in Lme
                            nv[y] = 0;
                            state[y] = o::merged;
                            parent[y] = x;
                            degree[x] = degree[y] < degree[x] ? degree[y] : degree[x];
                            len[y] = elen[y] = 0;
                            hnext[prev] = hnext[y];
                        } else
                            prev = y;
                    }
                    ++wflg;
                }
            }

            // 6. Final degrees d = min(external + |Lme \ i|, n - k - |i|), back into the degree lists; Lme compacted
            J pw = pme1;
            for (J q = pme1; q < pme2; ++q) {
                const J i = iw[q];
                if (state[i] != o::variable)
                    continue;
                const J nvi = -nv[i];
                nv[i] = nvi;
                J d = degree[i] + degme - nvi;
                const J dmax = neff - k - nvi;
                d = d > dmax ? dmax : d;
                d = d < 0 ? 0 : d;
                link(i, d);
                mindeg = d < mindeg ? d : mindeg;
                iw[pw++] = i;
            }
            len[me] = pw - pme1;
            degree[me] = degme;
            nv[me] = nvpiv;
            if (elenp != 0)
                pfree = pw;
        }

        // The order: each pivot, then what was merged into it (transitively), then the dense rows
        const auto order = degree;
        for (std::size_t s = 0; s < steps; ++s)
            order[s] = perm[s];
        for (std::size_t i = 0; i < n; ++i)
            hhead[i] = none;
        for (J i = nn; i-- > 0;)
            if (state[i] == o::merged) {
                J r = parent[i];
                while (r != none && state[r] == o::merged)
                    r = parent[r];
                for (J x = i; x != r && state[x] == o::merged;) {
                    const J up = parent[x];
                    parent[x] = r;
                    x = up;
                }
                if (r != none) {
                    hnext[i] = hhead[r];
                    hhead[r] = i;
                }
            }
        for (std::size_t i = 0; i < n; ++i)
            w[i] = 0;
        std::size_t out = 0;
        auto emit = [&](J x) {
            if (w[x] == 0) {
                w[x] = 1;
                perm[out++] = x;
            }
        };
        for (std::size_t s = 0; s < steps; ++s) {
            emit(order[s]);
            for (J m = hhead[order[s]]; m != none; m = hnext[m])
                emit(m);
        }
        for (J i = 0; i < nn; ++i)
            if (state[i] == o::dense)
                emit(i);
        for (J i = 0; i < nn; ++i)
            emit(i);   // anything left (none on a consistent graph): the result is a permutation regardless
        return ok();
    }

    /// \brief The reverse Cuthill-McKee ordering of a symmetric matrix given as its upper triangle.
    /// \ingroup core
    /// \tparam T The value type (unused: only the pattern is read).
    /// \tparam I The index type.
    /// \param a The upper triangle of the matrix, square and valid; the diagonal and entries below it are ignored.
    /// \param perm Receives the order: `perm[new] = old`, `a.cols` entries.
    /// \param work Working storage of at least `rcm_workspace_size(a.cols, a.nnz())` indices.
    /// \return Nothing, or a `sparse_error` (as `amd`).
    template <typename T, typename I>
    [[nodiscard]] constexpr result<void, sparse_error> reverse_cuthill_mckee(const csc_view<T, I>& a, std::span<std::remove_const_t<I>> perm,
                                                                             std::span<std::remove_const_t<I>> work) noexcept
    {
        using J = std::remove_const_t<I>;
        const std::size_t n = a.cols;
        if (a.rows != n)
            return err(sparse_error::dimension_mismatch);
        if (!is_valid(a))
            return err(sparse_error::invalid_index);
        if (perm.size() < n || work.size() < rcm_workspace_size(n, a.nnz()))
            return err(sparse_error::insufficient_capacity);
        if (n == 0)
            return ok();
        const auto xadj = work.first(n + 1), deg = work.subspan(n + 1, n), mark = work.subspan(2 * n + 1, n), queue = work.subspan(3 * n + 1, n);
        const auto adj = work.subspan(4 * n + 1);
        for (std::size_t i = 0; i < n; ++i)
            deg[i] = 0;
        detail::ordering::for_each_strict_upper(a, [&](std::size_t i, std::size_t j) {
            ++deg[i];
            ++deg[j];
        });
        xadj[0] = 0;
        for (std::size_t i = 0; i < n; ++i) {
            xadj[i + 1] = xadj[i] + deg[i];
            mark[i] = xadj[i];
        }
        detail::ordering::for_each_strict_upper(a, [&](std::size_t i, std::size_t j) {
            adj[detail::at(mark[i]++)] = static_cast<J>(j);
            adj[detail::at(mark[j]++)] = static_cast<J>(i);
        });
        constexpr J placed = -1;
        for (std::size_t i = 0; i < n; ++i)
            mark[i] = 0;
        J stamp = 0;
        // Breadth-first levels from s over the unplaced nodes: the number of levels; the last level is queue[first, count)
        auto levels = [&](J s, std::size_t& first, std::size_t& count) {
            ++stamp;
            queue[0] = s;
            mark[s] = stamp;
            std::size_t qh = 0, qt = 1;
            J depth = 0;
            while (qh < qt) {
                first = qh;
                const std::size_t end = qt;
                ++depth;
                for (; qh < end; ++qh) {
                    const J v = queue[qh];
                    for (J t = xadj[v]; t < xadj[v + 1]; ++t) {
                        const J u = adj[t];
                        if (mark[u] != placed && mark[u] != stamp) {
                            mark[u] = stamp;
                            queue[qt++] = u;
                        }
                    }
                }
            }
            count = qt;
            return depth;
        };
        std::size_t pos = 0, scan = 0;
        while (pos < n) {
            while (mark[scan] == placed)
                ++scan;
            J s = static_cast<J>(scan);
            std::size_t first = 0, count = 0;
            J ecc = levels(s, first, count);
            for (std::size_t round = 0; round < n; ++round) {
                J t = queue[first];
                for (std::size_t q = first; q < count; ++q)
                    t = deg[queue[q]] < deg[t] ? queue[q] : t;
                std::size_t f2 = 0, c2 = 0;
                const J e2 = levels(t, f2, c2);
                if (e2 <= ecc)
                    break;
                s = t;
                ecc = e2;
                first = f2;
                count = c2;
            }
            std::size_t h = pos;
            perm[pos++] = s;
            mark[s] = placed;
            while (h < pos) {
                const J v = perm[h++];
                const std::size_t start = pos;
                for (J t = xadj[v]; t < xadj[v + 1]; ++t) {
                    const J u = adj[t];
                    if (mark[u] != placed) {
                        mark[u] = placed;
                        perm[pos++] = u;
                    }
                }
                for (std::size_t x = start + 1; x < pos; ++x)
                    for (std::size_t y = x; y > start && deg[perm[y]] < deg[perm[y - 1]]; --y) {
                        const J tmp = perm[y];
                        perm[y] = perm[y - 1];
                        perm[y - 1] = tmp;
                    }
            }
        }
        for (std::size_t x = 0, y = n - 1; x < y; ++x, --y) {
            const J tmp = perm[x];
            perm[x] = perm[y];
            perm[y] = tmp;
        }
        return ok();
    }

    /// \brief Whether `perm` holds each of `0 .. perm.size() - 1` exactly once.
    /// \ingroup core
    /// \tparam I The index type.
    /// \param perm The candidate.
    /// \param work Optional working storage of `perm.size()` indices, for O(n); without it the check is O(n²).
    /// \return True if it is a permutation.
    template <typename I = std::int32_t>
    [[nodiscard]] constexpr bool is_permutation(std::span<const std::type_identity_t<I>> perm, std::span<std::type_identity_t<I>> work = {}) noexcept
    {
        const std::size_t n = perm.size();
        for (std::size_t k = 0; k < n; ++k)
            if (perm[k] < 0 || detail::at(perm[k]) >= n)
                return false;
        if (work.size() >= n) {
            for (std::size_t k = 0; k < n; ++k)
                work[k] = 0;
            for (std::size_t k = 0; k < n; ++k) {
                if (work[detail::at(perm[k])] != 0)
                    return false;
                work[detail::at(perm[k])] = 1;
            }
            return true;
        }
        for (std::size_t k = 0; k < n; ++k)
            for (std::size_t m = k + 1; m < n; ++m)
                if (perm[k] == perm[m])
                    return false;
        return true;
    }

    /// \brief The inverse permutation: `inv[perm[k]] = k`.
    /// \ingroup core
    /// \tparam I The index type.
    /// \param perm The permutation.
    /// \param inv Receives the inverse, `perm.size()` entries.
    /// \return Nothing, `sparse_error::insufficient_capacity` (`inv` too short) or `sparse_error::invalid_index` (`perm`
    /// is not a permutation; `inv` is then unspecified).
    template <typename I = std::int32_t>
    [[nodiscard]] constexpr result<void, sparse_error> inverse_permutation(std::span<const std::type_identity_t<I>> perm, std::span<std::type_identity_t<I>> inv) noexcept
    {
        const std::size_t n = perm.size();
        if (inv.size() < n)
            return err(sparse_error::insufficient_capacity);
        for (std::size_t k = 0; k < n; ++k)
            inv[k] = -1;
        for (std::size_t k = 0; k < n; ++k) {
            if (perm[k] < 0 || detail::at(perm[k]) >= n || inv[detail::at(perm[k])] != -1)
                return err(sparse_error::invalid_index);
            inv[detail::at(perm[k])] = static_cast<I>(k);
        }
        return ok();
    }

    /// \brief A vector into the permuted numbering: `y[k] = x[perm[k]]` (for a right-hand side).
    /// \ingroup core
    /// \tparam T The value type.
    /// \tparam I The index type.
    /// \param perm The permutation, `perm[new] = old`.
    /// \param x The vector in the original numbering, `perm.size()` values.
    /// \param y Receives it in the new numbering, `perm.size()` values; not `x`'s storage.
    template <typename T, typename I = std::int32_t>
    constexpr void permute(std::span<const std::type_identity_t<I>> perm, std::span<const std::type_identity_t<T>> x, std::span<T> y) noexcept
    {
        for (std::size_t k = 0; k < perm.size(); ++k)
            y[k] = x[detail::at(perm[k])];
    }

    /// \brief A vector back into the original numbering: `y[perm[k]] = x[k]` (for a solution).
    /// \ingroup core
    /// \tparam T The value type.
    /// \tparam I The index type.
    /// \param perm The permutation, `perm[new] = old`.
    /// \param x The vector in the new numbering, `perm.size()` values.
    /// \param y Receives it in the original numbering, `perm.size()` values; not `x`'s storage.
    template <typename T, typename I = std::int32_t>
    constexpr void inverse_permute(std::span<const std::type_identity_t<I>> perm, std::span<const std::type_identity_t<T>> x, std::span<T> y) noexcept
    {
        for (std::size_t k = 0; k < perm.size(); ++k)
            y[detail::at(perm[k])] = x[k];
    }

} // namespace snicholls::sparse

#endif /* snicholls_core_sparse_ordering_hpp */
