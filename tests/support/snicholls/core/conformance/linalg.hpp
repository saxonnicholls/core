//
//  linalg.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  The conformance suite of snicholls::linalg: typed GoogleTest suites that every backend registers in,
//  core's reference backend and any backend from another library alike. Test-only: it is not part of the
//  library, includes GoogleTest, and is exported by CMake as the INTERFACE target
//  snicholls::core_conformance (include directory tests/support), never installed.
//
//  Use, in exactly one translation unit of a test binary that links GTest:
//
//      #include <snicholls/core/conformance/linalg.hpp>
//      #include <my/eigen_backend.hpp>
//
//      SNICHOLLS_LINALG_CONFORMANCE(Eigen, snicholls::conformance::linalg_case<my::eigen_backend, float>,
//                                          snicholls::conformance::linalg_case<my::eigen_backend, double>);
//
//  A backend that models only part of the interface registers the parts it has:
//  SNICHOLLS_LINALG_CONFORMANCE_BLAS, _FACTORISATION, _DECOMPOSITION and _BATCHED take the same arguments.
//  A backend that cannot promise bit-for-bit results across executors says so with
//  linalg_case<B, T, /*deterministic=*/false>, and the determinism test is skipped with that reason.
//
//  Every check is against mathematics, never against another backend's output: exact integer products,
//  residuals and backward errors, reconstructions (P A = L U, A = L L^T, A = L D L^T, A = Q R, Q^T Q = I),
//  the eigen and singular-value equations, and closed forms (the eigenvalues and determinant of the
//  second-difference matrix, matrices built from known singular values, a line fitted through exact
//  points). Tolerances are stated per operation as multiples of the type's epsilon (below), so float
//  and double share the tests and differ only in epsilon. The cases:
//
//      random       well-conditioned random matrices, n = 1, 3, 8, 25 (float to 8)
//      ill          Hilbert matrices: n = 8 (double, condition 1.5e10), n = 3 (float, 5e2); backward error only
//      singular     exactly rank-deficient (an equal row), zero, NaN: reported, never solved
//      non-finite   NaN or infinity anywhere (diagonal, 1 x 1, one column) in a decomposition: reported
//      scale        entries whose squares overflow or underflow: decomposed as correctly as at scale 1
//      indefinite   Cholesky refuses; LDLT solves a bordered (kriging-style) indefinite system
//      layouts      every case but the batched ones runs three times: row-major, column-major (reached through
//                   transposed()) and padded strided (a gap after every row and every element); a write outside
//                   the view fails the case
//      batched      strided batches with gaps, one failure per kind, each reported for its own system;
//                   bit-for-bit identical across executors
//
//  Tolerances (eps = std::numeric_limits<T>::epsilon(); norms are max-norms; k is the inner dimension):
//
//      dot, gemv, gemm, syrk     exact on integer data; |c - c*| <= 2 k eps (|A| |B|) on random data
//      triangular_solve          |T x - b| <= 4 n eps (|T| |x| + |b|)
//      LU, Cholesky, LDLT        backward error |A x - b| / (|A| |x| + |b|) <= 16 n eps;
//                                reconstruction |P A - L U| / |A| <= 8 n eps (and the like)
//      QR, least squares         |Q^T Q - I| <= 8 m eps; |A - Q R| / |A| <= 8 m eps;
//                                |A^T (A x - b)| <= 16 m eps |A| (|A| |x| + |b|)
//      symmetric_eigen           |A v - w v| <= 32 n eps |A|; |V^T V - I| <= 32 n eps; closed forms to 32 n eps |A|
//      svd                       |A v - s u| <= 32 m eps s_max; |U^T U - I|, |V^T V - I| <= 32 m eps;
//                                known singular values to 32 m eps s_max
//      log_det                   |log|det| - exact| <= 16 n eps (second-difference matrix: det = n + 1)
//

#ifndef snicholls_core_conformance_linalg_hpp
#define snicholls_core_conformance_linalg_hpp

#include <snicholls/core/linalg/backend.hpp>
#include <snicholls/core/stats/random.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numbers>
#include <span>
#include <thread>
#include <utility>
#include <vector>

namespace snicholls::conformance {

    /// One registration: a backend type and a number type; `Deterministic` says whether the backend promises the
    /// same bits for every executor and thread count.
    template <typename Backend, typename T, bool Deterministic = true>
    struct linalg_case {
        using backend = Backend;
        using value_type = T;
        static constexpr bool deterministic = Deterministic;
    };

    namespace linalg_conformance {

        namespace la = snicholls::linalg;

        // ---- Matrices the tests own -------------------------------------------------------------------------

        // How a test matrix is stored. Every case runs in each, so a backend that ignores strides, assumes one
        // order, or writes outside its view fails: row-major; column-major (the storage of the transpose, read
        // through transposed()); and padded, a strided sub-view with a gap after every row and every element.
        enum class layout_kind { row_major, column_major, padded };
        inline constexpr layout_kind layouts[] = {layout_kind::row_major, layout_kind::column_major, layout_kind::padded};
        inline const char* name(layout_kind k)
        {
            return k == layout_kind::row_major ? "row-major" : k == layout_kind::column_major ? "column-major" : "padded strided";
        }
        inline thread_local layout_kind current_layout = layout_kind::row_major;

        // The layout every matrix made in its scope takes.
        struct layout_scope {
            layout_kind previous;
            explicit layout_scope(layout_kind k) : previous(current_layout) { current_layout = k; }
            ~layout_scope() { current_layout = previous; }
            layout_scope(const layout_scope&) = delete;
            layout_scope& operator=(const layout_scope&) = delete;
        };

        template <typename T>
        struct dense {
            std::size_t rows = 0, cols = 0;
            layout_kind kind = layout_kind::row_major;
            std::size_t offset = 0, s0 = 0, s1 = 0;
            std::vector<T> v;
            static T sentinel() { return T(-7777); }

            dense() = default;
            dense(std::size_t r, std::size_t c, T fill = T(0)) : rows(r), cols(c), kind(current_layout)
            {
                if (kind == layout_kind::row_major) {
                    s0 = c;
                    s1 = 1;
                    v.assign(r * c, fill);
                } else if (kind == layout_kind::column_major) {
                    s0 = 1;
                    s1 = r;
                    v.assign(r * c, fill);
                } else {
                    offset = 1;
                    s1 = 2;
                    s0 = 2 * c + 3;
                    v.assign(offset + r * s0 + 1, sentinel());
                    for (std::size_t i = 0; i < r; ++i)
                        for (std::size_t j = 0; j < c; ++j)
                            (*this)(i, j) = fill;
                }
            }
            dense(const dense&) = default;
            dense& operator=(const dense&) = default;
            ~dense()
            {
                if (!gaps_intact())
                    ADD_FAILURE() << "a backend wrote outside its view (" << name(kind) << ")";
            }

            T& operator()(std::size_t i, std::size_t j) { return v[offset + i * s0 + j * s1]; }
            const T& operator()(std::size_t i, std::size_t j) const { return v[offset + i * s0 + j * s1]; }
            la::matrix_view<T> view()
            {
                if (kind == layout_kind::column_major)   // column-major storage reached through transposed()
                    return la::transposed(la::matrix_view<T, la::layout_right>(v.data(), {cols, rows}));
                return {v.data() + offset, {rows, cols}, {s0, s1}};
            }
            la::matrix_view<const T> view() const { return const_cast<dense*>(this)->view(); }
            // Column j as a vector, through the matrix's own strides.
            la::vector_view<T> column(std::size_t j = 0) { return {&(*this)(0, j), {rows}, {s0}}; }

            bool gaps_intact() const
            {
                if (kind != layout_kind::padded)
                    return true;
                std::vector<bool> inside(v.size(), false);
                for (std::size_t i = 0; i < rows; ++i)
                    for (std::size_t j = 0; j < cols; ++j)
                        inside[offset + i * s0 + j * s1] = true;
                for (std::size_t k = 0; k < v.size(); ++k)
                    if (!inside[k] && !(v[k] == sentinel()))
                        return false;
                return true;
            }
        };

        template <typename T>
        T eps()
        {
            return std::numeric_limits<T>::epsilon();
        }

        // Uniform in [-1, 1), reproducible on every platform.
        template <typename T>
        dense<T> random(std::size_t r, std::size_t c, std::uint64_t seed)
        {
            stats::wyrand g(seed);
            dense<T> m(r, c);
            for (std::size_t i = 0; i < r; ++i)
                for (std::size_t j = 0; j < c; ++j)
                    m(i, j) = static_cast<T>(2.0 * stats::uniform_real(g) - 1.0);
            return m;
        }

        // Random, made strongly nonsingular by a diagonal shift.
        template <typename T>
        dense<T> well_conditioned(std::size_t n, std::uint64_t seed)
        {
            dense<T> m = random<T>(n, n, seed);
            for (std::size_t i = 0; i < n; ++i)
                m(i, i) += static_cast<T>(n);
            return m;
        }

        // G G^T + n I: symmetric positive definite.
        template <typename T>
        dense<T> spd(std::size_t n, std::uint64_t seed)
        {
            const dense<T> g = random<T>(n, n, seed);
            dense<T> a(n, n);
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t j = 0; j < n; ++j) {
                    long double s = 0;
                    for (std::size_t k = 0; k < n; ++k)
                        s += static_cast<long double>(g(i, k)) * g(j, k);
                    a(i, j) = static_cast<T>(s) + (i == j ? static_cast<T>(n) : T(0));
                }
            return a;
        }

        template <typename T>
        dense<T> hilbert(std::size_t n)
        {
            dense<T> h(n, n);
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t j = 0; j < n; ++j)
                    h(i, j) = T(1) / static_cast<T>(i + j + 1);
            return h;
        }

        // The second-difference matrix tridiag(-1, 2, -1): eigenvalues 2 - 2 cos(k pi / (n + 1)), det n + 1.
        template <typename T>
        dense<T> second_difference(std::size_t n)
        {
            dense<T> a(n, n);
            for (std::size_t i = 0; i < n; ++i) {
                a(i, i) = T(2);
                if (i + 1 < n)
                    a(i, i + 1) = a(i + 1, i) = T(-1);
            }
            return a;
        }

        // An orthogonal n x n matrix: a product of Householder reflections from random vectors (built in long double).
        template <typename T>
        dense<long double> orthogonal(std::size_t n, std::uint64_t seed)
        {
            dense<long double> q(n, n);
            for (std::size_t i = 0; i < n; ++i)
                q(i, i) = 1;
            const dense<T> vs = random<T>(3, n, seed);
            for (std::size_t r = 0; r < 3; ++r) {
                long double norm2 = 0;
                for (std::size_t i = 0; i < n; ++i)
                    norm2 += static_cast<long double>(vs(r, i)) * vs(r, i);
                if (norm2 == 0)
                    continue;
                for (std::size_t i = 0; i < n; ++i) {   // q <- q (I - 2 v v^T / v^T v)
                    long double w = 0;
                    for (std::size_t k = 0; k < n; ++k)
                        w += q(i, k) * vs(r, k);
                    for (std::size_t k = 0; k < n; ++k)
                        q(i, k) -= 2 * w * vs(r, k) / norm2;
                }
            }
            return q;
        }

        template <typename T>
        long double max_norm(const dense<T>& a)
        {
            long double m = 0;
            for (std::size_t i = 0; i < a.rows; ++i)
                for (std::size_t j = 0; j < a.cols; ++j)
                    m = std::max(m, std::fabs(static_cast<long double>(a(i, j))));
            return m;
        }

        template <typename T>
        long double inf_norm(const dense<T>& a)
        {
            long double m = 0;
            for (std::size_t i = 0; i < a.rows; ++i) {
                long double s = 0;
                for (std::size_t j = 0; j < a.cols; ++j)
                    s += std::fabs(static_cast<long double>(a(i, j)));
                m = std::max(m, s);
            }
            return m;
        }

        // |A x - b|_inf / (|A|_inf |x|_inf + |b|_inf) for each column of x, the largest; computed in long double.
        template <typename T>
        long double backward_error(const dense<T>& a, const dense<T>& x, const dense<T>& b)
        {
            const long double na = inf_norm(a);
            long double worst = 0;
            for (std::size_t c = 0; c < b.cols; ++c) {
                long double r = 0, nx = 0, nb = 0;
                for (std::size_t i = 0; i < a.rows; ++i) {
                    long double s = -static_cast<long double>(b(i, c));
                    for (std::size_t j = 0; j < a.cols; ++j)
                        s += static_cast<long double>(a(i, j)) * x(j, c);
                    r = std::max(r, std::fabs(s));
                    nb = std::max(nb, std::fabs(static_cast<long double>(b(i, c))));
                }
                for (std::size_t j = 0; j < a.cols; ++j)
                    nx = std::max(nx, std::fabs(static_cast<long double>(x(j, c))));
                const long double denominator = na * nx + nb;
                worst = std::max(worst, denominator > 0 ? r / denominator : r);
            }
            return worst;
        }

        template <typename T>
        long double bound(long double multiple, std::size_t n)
        {
            return multiple * static_cast<long double>(std::max<std::size_t>(n, 1)) * eps<T>();
        }

        template <typename T>
        dense<T> rhs_for(std::size_t n, std::size_t k, std::uint64_t seed)
        {
            return random<T>(n, k, seed);
        }

        // ---- What a backend must report -----------------------------------------------------------------------

        template <typename R>
        bool is_error(const R& r, la::error e)
        {
            return !r && r.err() == e;
        }

        template <typename T>
        std::vector<std::size_t> sizes()
        {
            if constexpr (sizeof(T) <= 4)
                return {1, 3, 8};
            else
                return {1, 3, 8, 25};
        }

        template <typename T>
        dense<T> ill_conditioned()
        {
            return hilbert<T>(sizeof(T) <= 4 ? 3 : 8);
        }

        // An n x n matrix with two equal rows, exactly singular.
        template <typename T>
        dense<T> rank_deficient(std::size_t n, std::uint64_t seed)
        {
            dense<T> a = well_conditioned<T>(n, seed);
            for (std::size_t j = 0; j < n; ++j)
                a(n - 1, j) = a(0, j);
            return a;
        }

        // The symmetric version: an equal row and column.
        template <typename T>
        dense<T> symmetric_rank_deficient(std::size_t n, std::uint64_t seed)
        {
            dense<T> a = spd<T>(n, seed);
            for (std::size_t j = 0; j < n; ++j)
                a(n - 1, j) = a(0, j);
            for (std::size_t i = 0; i < n; ++i)
                a(i, n - 1) = a(i, 0);
            return a;
        }

        // A bordered system [[C, 1], [1^T, 0]] with C an exponential covariance of points on a line: symmetric,
        // indefinite, nonsingular for distinct points; singular when two points coincide.
        template <typename T>
        dense<T> bordered(std::size_t k, std::uint64_t seed, bool repeat_a_point)
        {
            stats::wyrand g(seed);
            std::vector<double> x(k);
            for (std::size_t i = 0; i < k; ++i)
                x[i] = static_cast<double>(i) + 0.4 * stats::uniform_real(g);
            if (repeat_a_point && k > 1)
                x[1] = x[0];
            dense<T> a(k + 1, k + 1);
            for (std::size_t i = 0; i <= k; ++i)
                for (std::size_t j = 0; j <= k; ++j)
                    a(i, j) = i < k && j < k ? static_cast<T>(std::exp(-std::fabs(x[i] - x[j]) / 2.0)) : (i == k && j == k ? T(0) : T(1));
            return a;
        }

        // ---- Executors for the determinism test -----------------------------------------------------------

        struct reversed_uneven_chunks {
            template <typename F>
            void operator()(std::size_t n, F&& f) const
            {
                std::vector<std::pair<std::size_t, std::size_t>> ranges;
                for (std::size_t b = 0, k = 0; b < n; ++k) {
                    const std::size_t e = std::min(n, b + 1 + k % 5);
                    ranges.emplace_back(b, e);
                    b = e;
                }
                for (auto it = ranges.rbegin(); it != ranges.rend(); ++it)
                    f(it->first, it->second);
            }
        };

        struct three_threads {
            template <typename F>
            void operator()(std::size_t n, F&& f) const
            {
                std::thread a([&] { f(0, n / 3); });
                std::thread b([&] { f(n / 3, 2 * n / 3); });
                f(2 * n / 3, n);
                a.join();
                b.join();
            }
        };

        template <typename T>
        bool same_bits(const std::vector<T>& x, const std::vector<T>& y)
        {
            return x.size() == y.size() && std::memcmp(x.data(), y.data(), x.size() * sizeof(T)) == 0;
        }

        // A batch: n systems of order m, rows padded to m + 1, a gap of 3 between systems; right-hand sides
        // n x m x 2 with a gap of 1.
        template <typename T>
        struct batch {
            std::size_t n, m, ld, slot, rhs_slot;
            std::vector<T> a, b;
            batch(std::size_t n_, std::size_t m_)
                : n(n_), m(m_), ld(m_ + 1), slot((m_ + 1) * m_ + 3), rhs_slot(2 * m_ + 1), a(n_ * slot, T(-9)), b(n_ * rhs_slot, T(-9))
            {
            }
            void set(std::size_t s, const dense<T>& sys, const dense<T>& r)
            {
                for (std::size_t i = 0; i < m; ++i) {
                    for (std::size_t j = 0; j < m; ++j)
                        a[s * slot + i * ld + j] = sys(i, j);
                    for (std::size_t c = 0; c < 2; ++c)
                        b[s * rhs_slot + i * 2 + c] = r(i, c);
                }
            }
            la::batch_view<T> matrices() { return {a.data(), {n, m, m}, {slot, ld, 1}}; }
            la::batch_view<T> rhs() { return {b.data(), {n, m, 2}, {rhs_slot, 2, 1}}; }
            dense<T> solution(std::size_t s) const
            {
                dense<T> x(m, 2);
                for (std::size_t i = 0; i < m; ++i)
                    for (std::size_t c = 0; c < 2; ++c)
                        x(i, c) = b[s * rhs_slot + i * 2 + c];
                return x;
            }
        };

    } // namespace linalg_conformance

    // =========================================================================================================
    // BLAS
    // =========================================================================================================

    template <typename Case>
    class LinalgBlasConformance : public ::testing::Test {};
    TYPED_TEST_SUITE_P(LinalgBlasConformance);

    TYPED_TEST_P(LinalgBlasConformance, ProductsOfSmallIntegersAreExact)
    {
        for (const auto kind : linalg_conformance::layouts) {
            const linalg_conformance::layout_scope scope(kind);
            SCOPED_TRACE(linalg_conformance::name(kind));
            using T = typename TypeParam::value_type;
            namespace lc = linalg_conformance;
            namespace la = snicholls::linalg;
            const typename TypeParam::backend b{};
            // Integers in [-4, 4], inner dimension 9: every product and sum is exact in float and double.
            lc::dense<T> a(5, 9), bm(9, 4), c(5, 4, T(3)), x(9, 1), y(5, 1, T(1)), cs(5, 5, T(7));
            for (std::size_t i = 0; i < 5 * 9; ++i)
                a(i / 9, i % 9) = static_cast<T>(static_cast<int>((i * 7) % 9) - 4);
            for (std::size_t i = 0; i < 9 * 4; ++i)
                bm(i / 4, i % 4) = static_cast<T>(static_cast<int>((i * 5) % 9) - 4);
            for (std::size_t i = 0; i < 9; ++i)
                x(i, 0) = static_cast<T>(static_cast<int>(i % 3) - 1);
            const auto c0 = c;
            const auto y0 = y;
            const la::vector_view<const T> xv = x.column();
            ASSERT_TRUE(la::gemm(b, T(2), a.view(), bm.view(), T(-1), c.view()));
            ASSERT_TRUE(la::gemv(b, T(1), a.view(), xv, T(3), y.column()));
            ASSERT_TRUE(la::syrk(b, T(1), a.view(), T(0), cs.view(), la::triangle::lower));
            for (std::size_t i = 0; i < 5; ++i) {
                long long gv = 0;
                for (std::size_t k = 0; k < 9; ++k)
                    gv += static_cast<long long>(a(i, k)) * static_cast<long long>(x(k, 0));
                EXPECT_EQ(y(i, 0), static_cast<T>(gv + 3 * static_cast<long long>(y0(i, 0))));
                for (std::size_t j = 0; j < 4; ++j) {
                    long long s = 0;
                    for (std::size_t k = 0; k < 9; ++k)
                        s += static_cast<long long>(a(i, k)) * static_cast<long long>(bm(k, j));
                    EXPECT_EQ(c(i, j), static_cast<T>(2 * s - static_cast<long long>(c0(i, j))));
                }
                for (std::size_t j = 0; j < 5; ++j) {
                    long long s = 0;
                    for (std::size_t k = 0; k < 9; ++k)
                        s += static_cast<long long>(a(i, k)) * static_cast<long long>(a(j, k));
                    if (j <= i) {
                        EXPECT_EQ(cs(i, j), static_cast<T>(s));
                    } else {
                        EXPECT_EQ(cs(i, j), T(7));   // the other triangle is untouched
                    }
                }
            }
            const auto d = la::dot(b, xv, xv);
            ASSERT_TRUE(d);
            EXPECT_EQ(*d, T(6));   // three -1s and three 1s
            lc::dense<T> z(9, 1, T(1));
            ASSERT_TRUE(la::axpy(b, T(-2), xv, z.column()));
            for (std::size_t i = 0; i < 9; ++i)
                EXPECT_EQ(z(i, 0), T(1) - T(2) * x(i, 0));
            // x and y the same vector, as BLAS allows: y = (1 + alpha) y.
            lc::dense<T> u(9, 1, T(3));
            ASSERT_TRUE(la::axpy(b, T(2), u.column(), u.column()));
            for (std::size_t i = 0; i < 9; ++i)
                EXPECT_EQ(u(i, 0), T(9));
        }
    }

    TYPED_TEST_P(LinalgBlasConformance, RandomProductsAreWithinTheirErrorBound)
    {
        for (const auto kind : linalg_conformance::layouts) {
            const linalg_conformance::layout_scope scope(kind);
            SCOPED_TRACE(linalg_conformance::name(kind));
            using T = typename TypeParam::value_type;
            namespace lc = linalg_conformance;
            namespace la = snicholls::linalg;
            const typename TypeParam::backend b{};
            for (const std::size_t n : lc::sizes<T>()) {
                const auto a = lc::random<T>(n, n + 2, 11 + n), bm = lc::random<T>(n + 2, n, 12 + n);
                lc::dense<T> c(n, n);
                ASSERT_TRUE(la::gemm(b, T(1), a.view(), bm.view(), T(0), c.view()));
                for (std::size_t i = 0; i < n; ++i)
                    for (std::size_t j = 0; j < n; ++j) {
                        long double exact = 0, magnitude = 0;
                        for (std::size_t k = 0; k < n + 2; ++k) {
                            exact += static_cast<long double>(a(i, k)) * bm(k, j);
                            magnitude += std::fabs(static_cast<long double>(a(i, k)) * bm(k, j));
                        }
                        EXPECT_LE(std::fabs(c(i, j) - exact), lc::bound<T>(2, n + 2) * magnitude + std::numeric_limits<T>::denorm_min()) << n;
                    }
                // Strided operands (a transpose and a column) give the same products.
                lc::dense<T> ct(n, n);
                ASSERT_TRUE(la::gemm(b, T(1), la::transposed(bm.view()), la::transposed(a.view()), T(0), ct.view()));   // (A B)^T
                for (std::size_t i = 0; i < n; ++i)
                    for (std::size_t j = 0; j < n; ++j)
                        EXPECT_NEAR(static_cast<double>(ct(j, i)), static_cast<double>(c(i, j)), static_cast<double>(lc::bound<T>(4, n + 2) * (n + 2)));
            }
        }
    }

    TYPED_TEST_P(LinalgBlasConformance, TriangularSolvesHaveSmallResiduals)
    {
        for (const auto kind : linalg_conformance::layouts) {
            const linalg_conformance::layout_scope scope(kind);
            SCOPED_TRACE(linalg_conformance::name(kind));
            using T = typename TypeParam::value_type;
            namespace lc = linalg_conformance;
            namespace la = snicholls::linalg;
            const typename TypeParam::backend b{};
            for (const std::size_t n : lc::sizes<T>()) {
                for (const auto which : {la::triangle::lower, la::triangle::upper})
                    for (const auto diag : {la::diagonal::explicit_diagonal, la::diagonal::implicit_unit}) {
                        lc::dense<T> t = lc::well_conditioned<T>(n, 21 + n);
                        lc::dense<T> tri(n, n);   // the triangle the solve reads, the other zero, unit diagonal if asked
                        for (std::size_t i = 0; i < n; ++i)
                            for (std::size_t j = 0; j < n; ++j) {
                                const bool inside = which == la::triangle::lower ? j <= i : j >= i;
                                tri(i, j) = i == j && diag == la::diagonal::implicit_unit ? T(1) : (inside ? t(i, j) : T(0));
                            }
                        const lc::dense<T> rhs = lc::rhs_for<T>(n, 3, 22 + n);
                        lc::dense<T> x = rhs;
                        ASSERT_TRUE(la::triangular_solve(b, t.view(), x.view(), which, diag));
                        EXPECT_LE(lc::backward_error(tri, x, rhs), lc::bound<T>(4, n)) << n;
                    }
            }
        }
    }

    TYPED_TEST_P(LinalgBlasConformance, TriangularSolveReportsAZeroDiagonal)
    {
        for (const auto kind : linalg_conformance::layouts) {
            const linalg_conformance::layout_scope scope(kind);
            SCOPED_TRACE(linalg_conformance::name(kind));
            using T = typename TypeParam::value_type;
            namespace lc = linalg_conformance;
            namespace la = snicholls::linalg;
            lc::dense<T> t = lc::well_conditioned<T>(4, 31);
            t(2, 2) = T(0);
            lc::dense<T> x = lc::rhs_for<T>(4, 1, 32);
            EXPECT_TRUE(lc::is_error(la::triangular_solve(typename TypeParam::backend{}, t.view(), x.view(), la::triangle::upper), la::error::singular));
        }
    }

    REGISTER_TYPED_TEST_SUITE_P(LinalgBlasConformance, ProductsOfSmallIntegersAreExact, RandomProductsAreWithinTheirErrorBound,
                                TriangularSolvesHaveSmallResiduals, TriangularSolveReportsAZeroDiagonal);

    // =========================================================================================================
    // Factorisations
    // =========================================================================================================

    template <typename Case>
    class LinalgFactorisationConformance : public ::testing::Test {};
    TYPED_TEST_SUITE_P(LinalgFactorisationConformance);

    TYPED_TEST_P(LinalgFactorisationConformance, LuReconstructsAndSolves)
    {
        for (const auto kind : linalg_conformance::layouts) {
            const linalg_conformance::layout_scope scope(kind);
            SCOPED_TRACE(linalg_conformance::name(kind));
            using T = typename TypeParam::value_type;
            namespace lc = linalg_conformance;
            namespace la = snicholls::linalg;
            const typename TypeParam::backend b{};
            for (const std::size_t n : lc::sizes<T>()) {
                const lc::dense<T> a = lc::random<T>(n, n, 41 + n);   // random, not shifted: pivoting matters
                lc::dense<T> f = a;
                std::vector<la::pivot_index> p(n);
                ASSERT_TRUE(la::lu_factor(b, f.view(), std::span<la::pivot_index>(p))) << n;
                // P A = L U: apply the interchanges to A in order, then compare with the product of the factors.
                lc::dense<long double> pa(n, n);
                for (std::size_t i = 0; i < n; ++i)
                    for (std::size_t j = 0; j < n; ++j)
                        pa(i, j) = a(i, j);
                for (std::size_t k = 0; k < n; ++k) {
                    ASSERT_GE(p[k], static_cast<la::pivot_index>(k));
                    ASSERT_LT(p[k], static_cast<la::pivot_index>(n));
                    for (std::size_t j = 0; j < n; ++j)
                        std::swap(pa(k, j), pa(static_cast<std::size_t>(p[k]), j));
                }
                long double worst = 0;
                for (std::size_t i = 0; i < n; ++i)
                    for (std::size_t j = 0; j < n; ++j) {
                        long double s = 0;
                        for (std::size_t k = 0; k <= std::min(i, j); ++k)
                            s += (k == i ? 1.0L : static_cast<long double>(f(i, k))) * f(k, j);
                        worst = std::max(worst, std::fabs(s - pa(i, j)));
                    }
                EXPECT_LE(worst, lc::bound<T>(8, n) * lc::max_norm(a)) << n;
                const lc::dense<T> rhs = lc::rhs_for<T>(n, 2, 42 + n);
                lc::dense<T> x = rhs;
                ASSERT_TRUE(la::lu_solve(b, f.view(), std::span<const la::pivot_index>(p), x.view()));
                EXPECT_LE(lc::backward_error(a, x, rhs), lc::bound<T>(16, n)) << n;
                // The inverse: A A^-1 = I.
                lc::dense<T> inv(n, n);
                ASSERT_TRUE(la::lu_inverse(b, f.view(), std::span<const la::pivot_index>(p), inv.view()));
                lc::dense<T> identity(n, n);
                for (std::size_t i = 0; i < n; ++i)
                    identity(i, i) = T(1);
                EXPECT_LE(lc::backward_error(a, inv, identity), lc::bound<T>(16, n)) << n;
            }
        }
    }

    TYPED_TEST_P(LinalgFactorisationConformance, CholeskyAndLdltReconstructAndSolve)
    {
        for (const auto kind : linalg_conformance::layouts) {
            const linalg_conformance::layout_scope scope(kind);
            SCOPED_TRACE(linalg_conformance::name(kind));
            using T = typename TypeParam::value_type;
            namespace lc = linalg_conformance;
            namespace la = snicholls::linalg;
            const typename TypeParam::backend b{};
            for (const std::size_t n : lc::sizes<T>()) {
                const lc::dense<T> a = lc::spd<T>(n, 51 + n);
                lc::dense<T> l = a, ld = a;
                ASSERT_TRUE(la::cholesky_factor(b, l.view())) << n;
                ASSERT_TRUE(la::ldlt_factor(b, ld.view())) << n;
                long double worst_l = 0, worst_ld = 0;
                for (std::size_t i = 0; i < n; ++i)
                    for (std::size_t j = 0; j <= i; ++j) {
                        long double s = 0, t = 0;
                        for (std::size_t k = 0; k <= j; ++k) {
                            s += static_cast<long double>(l(i, k)) * l(j, k);
                            const long double li = k == i ? 1.0L : ld(i, k), lj = k == j ? 1.0L : ld(j, k);
                            t += li * ld(k, k) * lj;
                        }
                        worst_l = std::max(worst_l, std::fabs(s - a(i, j)));
                        worst_ld = std::max(worst_ld, std::fabs(t - a(i, j)));
                    }
                EXPECT_LE(worst_l, lc::bound<T>(8, n) * lc::max_norm(a)) << n;
                EXPECT_LE(worst_ld, lc::bound<T>(8, n) * lc::max_norm(a)) << n;
                const lc::dense<T> rhs = lc::rhs_for<T>(n, 2, 52 + n);
                lc::dense<T> x = rhs, y = rhs;
                ASSERT_TRUE(la::cholesky_solve(b, l.view(), x.view()));
                ASSERT_TRUE(la::ldlt_solve(b, ld.view(), y.view()));
                EXPECT_LE(lc::backward_error(a, x, rhs), lc::bound<T>(16, n)) << n;
                EXPECT_LE(lc::backward_error(a, y, rhs), lc::bound<T>(16, n)) << n;
            }
        }
    }

    TYPED_TEST_P(LinalgFactorisationConformance, IllConditionedSystemsHaveSmallBackwardError)
    {
        for (const auto kind : linalg_conformance::layouts) {
            const linalg_conformance::layout_scope scope(kind);
            SCOPED_TRACE(linalg_conformance::name(kind));
            using T = typename TypeParam::value_type;
            namespace lc = linalg_conformance;
            namespace la = snicholls::linalg;
            const typename TypeParam::backend b{};
            const lc::dense<T> h = lc::ill_conditioned<T>();
            const std::size_t n = h.rows;
            const lc::dense<T> rhs = lc::rhs_for<T>(n, 1, 61);
            lc::dense<T> lu = h, ch = h, ld = h, x1 = rhs, x2 = rhs, x3 = rhs;
            std::vector<la::pivot_index> p(n);
            ASSERT_TRUE(la::lu_factor(b, lu.view(), std::span<la::pivot_index>(p)));
            ASSERT_TRUE(la::lu_solve(b, lu.view(), std::span<const la::pivot_index>(p), x1.view()));
            ASSERT_TRUE(la::cholesky_factor(b, ch.view()));
            ASSERT_TRUE(la::cholesky_solve(b, ch.view(), x2.view()));
            ASSERT_TRUE(la::ldlt_factor(b, ld.view()));
            ASSERT_TRUE(la::ldlt_solve(b, ld.view(), x3.view()));
            EXPECT_LE(lc::backward_error(h, x1, rhs), lc::bound<T>(16, n));
            EXPECT_LE(lc::backward_error(h, x2, rhs), lc::bound<T>(16, n));
            EXPECT_LE(lc::backward_error(h, x3, rhs), lc::bound<T>(16, n));
        }
    }

    TYPED_TEST_P(LinalgFactorisationConformance, SingularAndIndefiniteAreReported)
    {
        for (const auto kind : linalg_conformance::layouts) {
            const linalg_conformance::layout_scope scope(kind);
            SCOPED_TRACE(linalg_conformance::name(kind));
            using T = typename TypeParam::value_type;
            namespace lc = linalg_conformance;
            namespace la = snicholls::linalg;
            const typename TypeParam::backend b{};
            const std::size_t n = 6;
            std::vector<la::pivot_index> p(n);
            auto rd = lc::rank_deficient<T>(n, 71);
            EXPECT_TRUE(lc::is_error(la::lu_factor(b, rd.view(), std::span<la::pivot_index>(p)), la::error::singular));
            lc::dense<T> zero(n, n);
            EXPECT_TRUE(lc::is_error(la::lu_factor(b, zero.view(), std::span<la::pivot_index>(p)), la::error::singular));
            auto nan = lc::well_conditioned<T>(n, 72);
            nan(3, 2) = std::numeric_limits<T>::quiet_NaN();
            EXPECT_TRUE(lc::is_error(la::lu_factor(b, nan.view(), std::span<la::pivot_index>(p)), la::error::singular));

            // Symmetric: Cholesky refuses an indefinite, a semidefinite and a NaN matrix; LDLT refuses a singular one.
            auto indefinite = lc::spd<T>(n, 73);
            indefinite(4, 4) = -indefinite(4, 4);
            auto semidefinite = lc::symmetric_rank_deficient<T>(n, 74);
            auto snan = lc::spd<T>(n, 75);
            snan(2, 2) = std::numeric_limits<T>::quiet_NaN();
            EXPECT_TRUE(lc::is_error(la::cholesky_factor(b, indefinite.view()), la::error::not_positive_definite));
            EXPECT_TRUE(lc::is_error(la::cholesky_factor(b, semidefinite.view()), la::error::not_positive_definite));
            EXPECT_TRUE(lc::is_error(la::cholesky_factor(b, snan.view()), la::error::not_positive_definite));
            auto sd2 = lc::symmetric_rank_deficient<T>(n, 74);
            EXPECT_TRUE(lc::is_error(la::ldlt_factor(b, sd2.view()), la::error::singular));
            lc::dense<T> zero2(n, n);
            EXPECT_TRUE(lc::is_error(la::ldlt_factor(b, zero2.view()), la::error::singular));
        }
    }

    TYPED_TEST_P(LinalgFactorisationConformance, LdltSolvesABorderedIndefiniteSystem)
    {
        for (const auto kind : linalg_conformance::layouts) {
            const linalg_conformance::layout_scope scope(kind);
            SCOPED_TRACE(linalg_conformance::name(kind));
            using T = typename TypeParam::value_type;
            namespace lc = linalg_conformance;
            namespace la = snicholls::linalg;
            const typename TypeParam::backend b{};
            for (const std::size_t k : {2u, 6u, 16u}) {
                const lc::dense<T> a = lc::bordered<T>(k, 81 + k, false);
                lc::dense<T> f = a, ch = a;
                ASSERT_TRUE(la::ldlt_factor(b, f.view())) << k;
                EXPECT_TRUE(lc::is_error(la::cholesky_factor(b, ch.view()), la::error::not_positive_definite));
                lc::dense<T> rhs(k + 1, 1, T(1));
                for (std::size_t i = 0; i < k; ++i)
                    rhs(i, 0) = static_cast<T>(std::exp(-static_cast<double>(i) / 3.0));
                lc::dense<T> x = rhs;
                ASSERT_TRUE(la::ldlt_solve(b, f.view(), x.view()));
                EXPECT_LE(lc::backward_error(a, x, rhs), lc::bound<T>(16, k + 1)) << k;
                long double weights = 0;   // the constraint row: the weights sum to one
                for (std::size_t i = 0; i < k; ++i)
                    weights += x(i, 0);
                EXPECT_NEAR(static_cast<double>(weights), 1.0, static_cast<double>(lc::bound<T>(64, k + 1) * lc::inf_norm(x)));
                auto repeated = lc::bordered<T>(k, 81 + k, true);
                EXPECT_TRUE(lc::is_error(la::ldlt_factor(b, repeated.view()), la::error::singular)) << k;
            }
        }
    }

    TYPED_TEST_P(LinalgFactorisationConformance, LdltWithASmallLeadingPivotIsAccurateOrReported)
    {
        for (const auto kind : linalg_conformance::layouts) {
            const linalg_conformance::layout_scope scope(kind);
            SCOPED_TRACE(linalg_conformance::name(kind));
            using T = typename TypeParam::value_type;
            namespace lc = linalg_conformance;
            namespace la = snicholls::linalg;
            const typename TypeParam::backend b{};
            // Well conditioned and symmetric, but its leading pivot is 1e-6: unpivoted elimination grows the elements
            // by 1e6. A backend may pivot and solve it, or report it; it may not return an inaccurate answer.
            lc::dense<T> a(3, 3);
            a(0, 0) = static_cast<T>(1e-6);
            a(0, 1) = a(1, 0) = a(1, 2) = a(2, 1) = a(2, 2) = T(1);
            lc::dense<T> rhs(3, 1);   // A (1, 2, 3)
            rhs(0, 0) = static_cast<T>(1e-6 + 2.0);
            rhs(1, 0) = T(4);
            rhs(2, 0) = T(5);
            lc::dense<T> f = a, x = rhs;
            const auto r = la::ldlt_factor(b, f.view());
            if (r) {
                ASSERT_TRUE(la::ldlt_solve(b, f.view(), x.view()));
                EXPECT_LE(lc::backward_error(a, x, rhs), lc::bound<T>(16, 3));
            } else {
                EXPECT_EQ(r.err(), la::error::singular);
            }
        }
    }

    TYPED_TEST_P(LinalgFactorisationConformance, QrIsOrthogonalTimesTriangular)
    {
        for (const auto kind : linalg_conformance::layouts) {
            const linalg_conformance::layout_scope scope(kind);
            SCOPED_TRACE(linalg_conformance::name(kind));
            using T = typename TypeParam::value_type;
            namespace lc = linalg_conformance;
            namespace la = snicholls::linalg;
            const typename TypeParam::backend b{};
            for (const std::size_t n : lc::sizes<T>()) {
                const std::size_t m = n + 3;
                const lc::dense<T> a = lc::random<T>(m, n, 91 + n);
                lc::dense<T> f = a, q(m, n);
                std::vector<T> tau(n);
                ASSERT_TRUE(la::qr_factor(b, f.view(), std::span<T>(tau)));
                ASSERT_TRUE(la::qr_q(b, f.view(), std::span<const T>(tau), q.view()));
                long double orth = 0, recon = 0;
                for (std::size_t i = 0; i < n; ++i)
                    for (std::size_t j = 0; j < n; ++j) {
                        long double s = 0;
                        for (std::size_t r = 0; r < m; ++r)
                            s += static_cast<long double>(q(r, i)) * q(r, j);
                        orth = std::max(orth, std::fabs(s - (i == j ? 1.0L : 0.0L)));
                    }
                for (std::size_t i = 0; i < m; ++i)
                    for (std::size_t j = 0; j < n; ++j) {
                        long double s = 0;
                        for (std::size_t k = 0; k <= j; ++k)
                            s += static_cast<long double>(q(i, k)) * f(k, j);   // R is f's upper triangle
                        recon = std::max(recon, std::fabs(s - a(i, j)));
                    }
                EXPECT_LE(orth, lc::bound<T>(8, m)) << n;
                EXPECT_LE(recon, lc::bound<T>(8, m) * lc::max_norm(a)) << n;
            }
        }
    }

    TYPED_TEST_P(LinalgFactorisationConformance, LeastSquaresSatisfiesTheNormalEquations)
    {
        for (const auto kind : linalg_conformance::layouts) {
            const linalg_conformance::layout_scope scope(kind);
            SCOPED_TRACE(linalg_conformance::name(kind));
            using T = typename TypeParam::value_type;
            namespace lc = linalg_conformance;
            namespace la = snicholls::linalg;
            const typename TypeParam::backend b{};
            // Exact data: the line 3 - 2 t through 7 points is recovered.
            lc::dense<T> design(7, 2), obs(7, 1);
            for (std::size_t i = 0; i < 7; ++i) {
                design(i, 0) = T(1);
                design(i, 1) = static_cast<T>(i);
                obs(i, 0) = static_cast<T>(3.0 - 2.0 * static_cast<double>(i));
            }
            std::vector<T> tau(2);
            ASSERT_TRUE(la::least_squares(b, design.view(), std::span<T>(tau), obs.view()));
            EXPECT_NEAR(static_cast<double>(obs(0, 0)), 3.0, static_cast<double>(lc::bound<T>(64, 7)) * 10);
            EXPECT_NEAR(static_cast<double>(obs(1, 0)), -2.0, static_cast<double>(lc::bound<T>(64, 7)) * 10);

            for (const std::size_t n : lc::sizes<T>()) {
                const std::size_t m = 2 * n + 1;
                const lc::dense<T> a = lc::random<T>(m, n, 101 + n), rhs = lc::rhs_for<T>(m, 2, 102 + n);
                lc::dense<T> f = a, x = rhs;
                std::vector<T> t(n);
                ASSERT_TRUE(la::least_squares(b, f.view(), std::span<T>(t), x.view()));
                const long double na = lc::inf_norm(a);
                for (std::size_t c = 0; c < 2; ++c) {
                    long double nx = 0, nb = 0, worst = 0;
                    std::vector<long double> r(m);
                    for (std::size_t i = 0; i < m; ++i) {
                        long double s = -static_cast<long double>(rhs(i, c));
                        for (std::size_t j = 0; j < n; ++j)
                            s += static_cast<long double>(a(i, j)) * x(j, c);
                        r[i] = s;
                        nb = std::max(nb, std::fabs(static_cast<long double>(rhs(i, c))));
                    }
                    for (std::size_t j = 0; j < n; ++j) {
                        nx = std::max(nx, std::fabs(static_cast<long double>(x(j, c))));
                        long double s = 0;
                        for (std::size_t i = 0; i < m; ++i)
                            s += static_cast<long double>(a(i, j)) * r[i];
                        worst = std::max(worst, std::fabs(s));
                    }
                    EXPECT_LE(worst, lc::bound<T>(16, m) * na * (na * nx + nb)) << n;
                }
            }
            // Rank deficient: two equal columns.
            lc::dense<T> collinear = lc::random<T>(6, 3, 103);
            for (std::size_t i = 0; i < 6; ++i)
                collinear(i, 2) = collinear(i, 0);
            lc::dense<T> y = lc::rhs_for<T>(6, 1, 104);
            std::vector<T> t3(3);
            EXPECT_TRUE(lc::is_error(la::least_squares(b, collinear.view(), std::span<T>(t3), y.view()), la::error::singular));
        }
    }

    TYPED_TEST_P(LinalgFactorisationConformance, SubnormalMatricesSolve)
    {
        for (const auto kind : linalg_conformance::layouts) {
            const linalg_conformance::layout_scope scope(kind);
            SCOPED_TRACE(linalg_conformance::name(kind));
            using T = typename TypeParam::value_type;
            namespace lc = linalg_conformance;
            namespace la = snicholls::linalg;
            const typename TypeParam::backend b{};
            // diag(t, 2t) with t subnormal: nonsingular, a multiple of a perfectly conditioned matrix. A factorisation that
            // multiplies by 1 / pivot overflows to infinity and reports it singular or returns infinities.
            const T t = std::numeric_limits<T>::min() / T(16);
            const auto make = [&] {
                lc::dense<T> a(2, 2);
                a(0, 0) = t;
                a(1, 1) = T(2) * t;
                return a;
            };
            const auto rhs = [&] {
                lc::dense<T> y(2, 1);
                y(0, 0) = t;
                y(1, 0) = T(2) * t;
                return y;
            };
            const double tol = 1e-3;   // a subnormal carries fewer digits
            lc::dense<T> lu = make(), x = rhs();
            std::vector<la::pivot_index> p(2);
            ASSERT_TRUE(la::lu_factor(b, lu.view(), std::span<la::pivot_index>(p)));
            ASSERT_TRUE(la::lu_solve(b, lu.view(), std::span<const la::pivot_index>(p), x.view()));
            EXPECT_NEAR(static_cast<double>(x(0, 0)), 1.0, tol);
            EXPECT_NEAR(static_cast<double>(x(1, 0)), 1.0, tol);
            lc::dense<T> ch = make(), y = rhs();
            ASSERT_TRUE(la::cholesky_factor(b, ch.view()));
            ASSERT_TRUE(la::cholesky_solve(b, ch.view(), y.view()));
            EXPECT_NEAR(static_cast<double>(y(1, 0)), 1.0, tol);
            lc::dense<T> ld = make(), z = rhs();
            ASSERT_TRUE(la::ldlt_factor(b, ld.view()));
            ASSERT_TRUE(la::ldlt_solve(b, ld.view(), z.view()));
            EXPECT_NEAR(static_cast<double>(z(1, 0)), 1.0, tol);
            // Least squares on [[0], [d]] with d the smallest subnormal: x = 1 exactly fits [0, d].
            lc::dense<T> column(2, 1), obs(2, 1);
            column(1, 0) = obs(1, 0) = std::numeric_limits<T>::denorm_min();
            std::vector<T> tau(1);
            ASSERT_TRUE(la::least_squares(b, column.view(), std::span<T>(tau), obs.view()));
            EXPECT_NEAR(static_cast<double>(obs(0, 0)), 1.0, tol);
        }
    }

    TYPED_TEST_P(LinalgFactorisationConformance, LeastSquaresNearTheTopOfTheRangeIsExact)
    {
        for (const auto kind : linalg_conformance::layouts) {
            const linalg_conformance::layout_scope scope(kind);
            SCOPED_TRACE(linalg_conformance::name(kind));
            using T = typename TypeParam::value_type;
            namespace lc = linalg_conformance;
            namespace la = snicholls::linalg;
            const typename TypeParam::backend b{};
            // [[h], [h]] x = [h, h] with h near the largest finite value: x = 1. Householder's beta - alpha and
            // tau (v^T b) both exceed the range when formed directly, and gave x = inf.
            const T h = std::numeric_limits<T>::max() / T(2);
            lc::dense<T> a(2, 1, h), y(2, 1, h);
            std::vector<T> tau(1);
            ASSERT_TRUE(la::least_squares(b, a.view(), std::span<T>(tau), y.view()));
            EXPECT_NEAR(static_cast<double>(y(0, 0)), 1.0, static_cast<double>(lc::bound<T>(16, 2)));
            EXPECT_TRUE(std::isfinite(static_cast<double>(tau[0])));
        }
    }

    TYPED_TEST_P(LinalgFactorisationConformance, QrReportsNonFiniteInput)
    {
        for (const auto kind : linalg_conformance::layouts) {
            const linalg_conformance::layout_scope scope(kind);
            SCOPED_TRACE(linalg_conformance::name(kind));
            using T = typename TypeParam::value_type;
            namespace lc = linalg_conformance;
            namespace la = snicholls::linalg;
            const typename TypeParam::backend b{};
            const T nan = std::numeric_limits<T>::quiet_NaN(), inf = std::numeric_limits<T>::infinity();
            // Each matrix has one NaN or infinity: below the first diagonal (where a scaled norm can skip it), in the
            // last row, in R above the diagonal, or on it.
            const std::vector<std::vector<T>> columns_of_three{{1, nan, nan}, {1, 0, nan}, {1, 0, inf}};
            for (const auto& c : columns_of_three) {
                lc::dense<T> a(3, 1), y(3, 1, T(1));
                for (std::size_t i = 0; i < 3; ++i)
                    a(i, 0) = c[i];
                std::vector<T> tau(1);
                EXPECT_FALSE(la::least_squares(b, a.view(), std::span<T>(tau), y.view()));
            }
            lc::dense<T> upper(3, 2, T(1));
            upper(0, 1) = nan;
            lc::dense<T> y(3, 1, T(1));
            std::vector<T> tau(2);
            EXPECT_FALSE(la::least_squares(b, upper.view(), std::span<T>(tau), y.view()));
            lc::dense<T> diagonal(3, 2, T(1));
            diagonal(1, 1) = inf;
            lc::dense<T> y2(3, 1, T(1));
            EXPECT_FALSE(la::least_squares(b, diagonal.view(), std::span<T>(tau), y2.view()));
        }
    }

    TYPED_TEST_P(LinalgFactorisationConformance, LogDeterminantsMatchTheClosedForm)
    {
        for (const auto kind : linalg_conformance::layouts) {
            const linalg_conformance::layout_scope scope(kind);
            SCOPED_TRACE(linalg_conformance::name(kind));
            using T = typename TypeParam::value_type;
            namespace lc = linalg_conformance;
            namespace la = snicholls::linalg;
            const typename TypeParam::backend b{};
            for (const std::size_t n : lc::sizes<T>()) {
                const lc::dense<T> a = lc::second_difference<T>(n);   // det = n + 1
                const double exact = std::log(static_cast<double>(n + 1));
                const double tol = static_cast<double>(lc::bound<T>(16, n));
                lc::dense<T> lu = a, ch = a, ld = a;
                std::vector<la::pivot_index> p(n);
                ASSERT_TRUE(la::lu_factor(b, lu.view(), std::span<la::pivot_index>(p)));
                ASSERT_TRUE(la::cholesky_factor(b, ch.view()));
                ASSERT_TRUE(la::ldlt_factor(b, ld.view()));
                const auto d1 = la::lu_log_det(lu.view(), std::span<const la::pivot_index>(p));
                const auto d2 = la::cholesky_log_det(ch.view());
                const auto d3 = la::ldlt_log_det(ld.view());
                ASSERT_TRUE(d1 && d2 && d3);
                EXPECT_EQ(d1->sign, 1);
                EXPECT_EQ(d3->sign, 1);
                EXPECT_NEAR(static_cast<double>(d1->log_abs), exact, tol) << n;
                EXPECT_NEAR(static_cast<double>(d2->log_abs), exact, tol) << n;
                EXPECT_NEAR(static_cast<double>(d3->log_abs), exact, tol) << n;
                // Negating the matrix multiplies the determinant by (-1)^n.
                lc::dense<T> neg = a;
                for (std::size_t i = 0; i < n; ++i)
                    for (std::size_t j = 0; j < n; ++j)
                        neg(i, j) = -neg(i, j);
                ASSERT_TRUE(la::lu_factor(b, neg.view(), std::span<la::pivot_index>(p)));
                const auto d4 = la::lu_log_det(neg.view(), std::span<const la::pivot_index>(p));
                ASSERT_TRUE(d4);
                EXPECT_EQ(d4->sign, n % 2 == 0 ? 1 : -1) << n;
                EXPECT_NEAR(static_cast<double>(d4->log_abs), exact, tol) << n;
            }
            // An indefinite symmetric matrix: LDLT's sign. det [[1, 2], [2, 1]] = -3.
            lc::dense<T> m(2, 2, T(2));
            m(0, 0) = m(1, 1) = T(1);
            ASSERT_TRUE(la::ldlt_factor(b, m.view()));
            const auto d = la::ldlt_log_det(m.view());
            ASSERT_TRUE(d);
            EXPECT_EQ(d->sign, -1);
            EXPECT_NEAR(static_cast<double>(d->value()), -3.0, 1e-5);
        }
    }

    REGISTER_TYPED_TEST_SUITE_P(LinalgFactorisationConformance, LuReconstructsAndSolves, CholeskyAndLdltReconstructAndSolve,
                                IllConditionedSystemsHaveSmallBackwardError, SingularAndIndefiniteAreReported, LdltSolvesABorderedIndefiniteSystem,
                                LdltWithASmallLeadingPivotIsAccurateOrReported,
                                QrIsOrthogonalTimesTriangular, LeastSquaresSatisfiesTheNormalEquations, SubnormalMatricesSolve, LeastSquaresNearTheTopOfTheRangeIsExact,
                                QrReportsNonFiniteInput,
                                LogDeterminantsMatchTheClosedForm);

    // =========================================================================================================
    // Decompositions
    // =========================================================================================================

    template <typename Case>
    class LinalgDecompositionConformance : public ::testing::Test {};
    TYPED_TEST_SUITE_P(LinalgDecompositionConformance);

    TYPED_TEST_P(LinalgDecompositionConformance, EigenpairsSatisfyTheEigenEquation)
    {
        for (const auto kind : linalg_conformance::layouts) {
            const linalg_conformance::layout_scope scope(kind);
            SCOPED_TRACE(linalg_conformance::name(kind));
            using T = typename TypeParam::value_type;
            namespace lc = linalg_conformance;
            namespace la = snicholls::linalg;
            const typename TypeParam::backend b{};
            for (const std::size_t n : lc::sizes<T>()) {
                const lc::dense<T> a = lc::spd<T>(n, 111 + n);   // any symmetric matrix; shifted so eigenvalues spread
                lc::dense<T> work = a, v(n, n);
                std::vector<T> w(n);
                ASSERT_TRUE(la::symmetric_eigen(b, work.view(), std::span<T>(w), v.view())) << n;
                const long double na = lc::inf_norm(a);
                long double residual = 0, orth = 0;
                for (std::size_t c = 0; c < n; ++c) {
                    if (c > 0) {
                        EXPECT_LE(w[c - 1], w[c]);   // ascending
                    }
                    for (std::size_t i = 0; i < n; ++i) {
                        long double s = -static_cast<long double>(w[c]) * v(i, c);
                        for (std::size_t j = 0; j < n; ++j)
                            s += static_cast<long double>(a(i, j)) * v(j, c);
                        residual = std::max(residual, std::fabs(s));
                    }
                    for (std::size_t d = 0; d < n; ++d) {
                        long double s = 0;
                        for (std::size_t i = 0; i < n; ++i)
                            s += static_cast<long double>(v(i, c)) * v(i, d);
                        orth = std::max(orth, std::fabs(s - (c == d ? 1.0L : 0.0L)));
                    }
                }
                EXPECT_LE(residual, lc::bound<T>(32, n) * na) << n;
                EXPECT_LE(orth, lc::bound<T>(32, n)) << n;
            }
        }
    }

    TYPED_TEST_P(LinalgDecompositionConformance, EigenvaluesOfTheSecondDifferenceMatrix)
    {
        for (const auto kind : linalg_conformance::layouts) {
            const linalg_conformance::layout_scope scope(kind);
            SCOPED_TRACE(linalg_conformance::name(kind));
            using T = typename TypeParam::value_type;
            namespace lc = linalg_conformance;
            namespace la = snicholls::linalg;
            const typename TypeParam::backend b{};
            for (const std::size_t n : lc::sizes<T>()) {
                lc::dense<T> a = lc::second_difference<T>(n), v(n, n);
                std::vector<T> w(n);
                ASSERT_TRUE(la::symmetric_eigen(b, a.view(), std::span<T>(w), v.view()));
                for (std::size_t k = 0; k < n; ++k) {
                    const double exact = 2.0 - 2.0 * std::cos(static_cast<double>(k + 1) * std::numbers::pi / static_cast<double>(n + 1));
                    EXPECT_NEAR(static_cast<double>(w[k]), exact, static_cast<double>(lc::bound<T>(32, n) * 4)) << n << " " << k;
                }
            }
        }
    }

    TYPED_TEST_P(LinalgDecompositionConformance, SvdRecoversKnownSingularValues)
    {
        for (const auto kind : linalg_conformance::layouts) {
            const linalg_conformance::layout_scope scope(kind);
            SCOPED_TRACE(linalg_conformance::name(kind));
            using T = typename TypeParam::value_type;
            namespace lc = linalg_conformance;
            namespace la = snicholls::linalg;
            const typename TypeParam::backend b{};
            for (const std::size_t n : lc::sizes<T>()) {
                const std::size_t m = n + 2;
                // A = Q1[:, :n] diag(sigma) Q2^T with sigma = n, n - 1, ..., 1 (and one zero for n > 2).
                const auto q1 = lc::orthogonal<T>(m, 121 + n), q2 = lc::orthogonal<T>(n, 122 + n);
                std::vector<long double> sigma(n);
                for (std::size_t i = 0; i < n; ++i)
                    sigma[i] = static_cast<long double>(n - i);
                if (n > 2)
                    sigma[n - 1] = 0;
                lc::dense<T> a(m, n);
                for (std::size_t i = 0; i < m; ++i)
                    for (std::size_t j = 0; j < n; ++j) {
                        long double s = 0;
                        for (std::size_t k = 0; k < n; ++k)
                            s += q1(i, k) * sigma[k] * q2(j, k);
                        a(i, j) = static_cast<T>(s);
                    }
                lc::dense<T> u = a, v(n, n);
                std::vector<T> s(n);
                ASSERT_TRUE(la::svd(b, u.view(), std::span<T>(s), v.view())) << n;
                const long double smax = static_cast<long double>(n);
                const long double tol = lc::bound<T>(32, m) * smax;
                for (std::size_t i = 0; i < n; ++i)
                    EXPECT_NEAR(static_cast<double>(s[i]), static_cast<double>(sigma[i]), static_cast<double>(tol)) << n << " " << i;
                long double residual = 0, orth_u = 0, orth_v = 0;
                for (std::size_t c = 0; c < n; ++c) {
                    for (std::size_t i = 0; i < m; ++i) {
                        long double t = -static_cast<long double>(s[c]) * u(i, c);
                        for (std::size_t j = 0; j < n; ++j)
                            t += static_cast<long double>(a(i, j)) * v(j, c);
                        residual = std::max(residual, std::fabs(t));
                    }
                    for (std::size_t d = 0; d < n; ++d) {
                        long double tv = 0, tu = 0;
                        for (std::size_t i = 0; i < n; ++i)
                            tv += static_cast<long double>(v(i, c)) * v(i, d);
                        for (std::size_t i = 0; i < m; ++i)
                            tu += static_cast<long double>(u(i, c)) * u(i, d);
                        orth_v = std::max(orth_v, std::fabs(tv - (c == d ? 1.0L : 0.0L)));
                        if (sigma[c] > 0 && sigma[d] > 0)   // U is orthonormal where the singular values are not zero
                            orth_u = std::max(orth_u, std::fabs(tu - (c == d ? 1.0L : 0.0L)));
                    }
                }
                EXPECT_LE(residual, tol) << n;
                EXPECT_LE(orth_v, lc::bound<T>(32, m)) << n;
                EXPECT_LE(orth_u, lc::bound<T>(32, m)) << n;
            }
        }
    }

    TYPED_TEST_P(LinalgDecompositionConformance, ExtremeScalesDecomposeCorrectly)
    {
        for (const auto kind : linalg_conformance::layouts) {
            const linalg_conformance::layout_scope scope(kind);
            SCOPED_TRACE(linalg_conformance::name(kind));
            using T = typename TypeParam::value_type;
            namespace lc = linalg_conformance;
            namespace la = snicholls::linalg;
            const typename TypeParam::backend b{};
            // Scales whose squares overflow (huge) or underflow (tiny): a decomposition that sums unscaled squares
            // sees an infinite or zero off-diagonal norm and stops before rotating.
            const T huge = std::sqrt(std::numeric_limits<T>::max()) * T(16);
            const T tiny = std::sqrt(std::numeric_limits<T>::min()) / T(16);
            const double golden = (1.0 + std::sqrt(5.0)) / 2.0;
            for (const T scale : {huge, tiny}) {
                // [[0, x], [x, 0]] has eigenvalues -x and x.
                lc::dense<T> a(2, 2), v(2, 2);
                a(0, 1) = a(1, 0) = scale;
                std::vector<T> w(2);
                ASSERT_TRUE(la::symmetric_eigen(b, a.view(), std::span<T>(w), v.view())) << scale;
                EXPECT_NEAR(static_cast<double>(w[0] / scale), -1.0, static_cast<double>(lc::bound<T>(32, 2))) << scale;
                EXPECT_NEAR(static_cast<double>(w[1] / scale), 1.0, static_cast<double>(lc::bound<T>(32, 2))) << scale;
                // [[x, x], [0, x]] has singular values x * golden and x / golden.
                lc::dense<T> m(2, 2, scale), vv(2, 2);
                m(1, 0) = T(0);
                std::vector<T> s(2);
                ASSERT_TRUE(la::svd(b, m.view(), std::span<T>(s), vv.view())) << scale;
                EXPECT_NEAR(static_cast<double>(s[0] / scale), golden, static_cast<double>(lc::bound<T>(32, 2)) * 2) << scale;
                EXPECT_NEAR(static_cast<double>(s[1] / scale), 1.0 / golden, static_cast<double>(lc::bound<T>(32, 2)) * 2) << scale;
            }
        }
    }

    TYPED_TEST_P(LinalgDecompositionConformance, NonFiniteInputIsReportedNeverDecomposed)
    {
        for (const auto kind : linalg_conformance::layouts) {
            const linalg_conformance::layout_scope scope(kind);
            SCOPED_TRACE(linalg_conformance::name(kind));
            using T = typename TypeParam::value_type;
            namespace lc = linalg_conformance;
            namespace la = snicholls::linalg;
            const typename TypeParam::backend b{};
            const T nan = std::numeric_limits<T>::quiet_NaN(), inf = std::numeric_limits<T>::infinity();
            const auto eigen_fails = [&](lc::dense<T> a) {
                lc::dense<T> v(a.rows, a.rows);
                std::vector<T> w(a.rows);
                return !la::symmetric_eigen(b, a.view(), std::span<T>(w), v.view());
            };
            const auto svd_fails = [&](lc::dense<T> a) {
                lc::dense<T> v(a.cols, a.cols);
                std::vector<T> s(a.cols);
                return !la::svd(b, a.view(), std::span<T>(s), v.view());
            };
            lc::dense<T> one(1, 1, nan);
            EXPECT_TRUE(eigen_fails(one));                       // NaN, 1 x 1
            lc::dense<T> diagonal_nan(2, 2);
            diagonal_nan(0, 0) = nan;                            // NaN only on the diagonal: the off-diagonal norm stays 0
            diagonal_nan(1, 1) = T(1);
            EXPECT_TRUE(eigen_fails(diagonal_nan));
            lc::dense<T> infinite(2, 2, T(1));
            infinite(1, 0) = infinite(0, 1) = inf;
            EXPECT_TRUE(eigen_fails(infinite));
            lc::dense<T> column(2, 1, T(1));
            column(1, 0) = nan;                                  // one column: no pair is ever rotated
            EXPECT_TRUE(svd_fails(column));
            EXPECT_TRUE(svd_fails(diagonal_nan));
            EXPECT_TRUE(svd_fails(infinite));
        }
    }

    REGISTER_TYPED_TEST_SUITE_P(LinalgDecompositionConformance, EigenpairsSatisfyTheEigenEquation, EigenvaluesOfTheSecondDifferenceMatrix,
                                SvdRecoversKnownSingularValues, ExtremeScalesDecomposeCorrectly, NonFiniteInputIsReportedNeverDecomposed);

    // =========================================================================================================
    // Batched
    // =========================================================================================================

    template <typename Case>
    class LinalgBatchedConformance : public ::testing::Test {};
    TYPED_TEST_SUITE_P(LinalgBatchedConformance);

    TYPED_TEST_P(LinalgBatchedConformance, LuBatchReportsEachFailureForItsOwnSystem)
    {
        using T = typename TypeParam::value_type;
        namespace lc = linalg_conformance;
        namespace la = snicholls::linalg;
        const typename TypeParam::backend b{};
        constexpr std::size_t n = 23, m = 9;
        lc::batch<T> x(n, m);
        std::vector<lc::dense<T>> systems, rhs;
        for (std::size_t s = 0; s < n; ++s) {
            systems.push_back(s == 4 ? lc::rank_deficient<T>(m, 131 + s) : lc::random<T>(m, m, 131 + s));
            if (s == 11)
                systems.back()(5, 5) = std::numeric_limits<T>::quiet_NaN();
            if (s == 17)
                systems.back() = lc::dense<T>(m, m);
            rhs.push_back(lc::rhs_for<T>(m, 2, 231 + s));
            x.set(s, systems.back(), rhs.back());
        }
        std::vector<la::pivot_index> p(n * m);
        std::vector<la::status> status(n);
        const auto r = la::lu_solve_batched(b, x.matrices(), std::span<la::pivot_index>(p), x.rhs(), std::span<la::status>(status));
        ASSERT_TRUE(r);
        EXPECT_EQ(*r, 3u);
        for (std::size_t s = 0; s < n; ++s) {
            const bool fails = s == 4 || s == 11 || s == 17;
            EXPECT_EQ(status[s], fails ? la::status::singular : la::status::ok) << s;
            if (!fails) {
                EXPECT_LE(lc::backward_error(systems[s], x.solution(s), rhs[s]), lc::bound<T>(16, m)) << s;
            }
        }
        EXPECT_EQ(x.a[m], T(-9));   // a gap
    }

    TYPED_TEST_P(LinalgBatchedConformance, CholeskyAndLdltBatchesReportEachFailure)
    {
        using T = typename TypeParam::value_type;
        namespace lc = linalg_conformance;
        namespace la = snicholls::linalg;
        const typename TypeParam::backend b{};
        constexpr std::size_t n = 19, k = 7, m = k + 1;
        lc::batch<T> chol(n, m), ldlt(n, m);
        std::vector<lc::dense<T>> spds, bordered, rhs;
        for (std::size_t s = 0; s < n; ++s) {
            spds.push_back(lc::spd<T>(m, 141 + s));
            if (s == 5)
                spds.back()(2, 2) = -spds.back()(2, 2);   // indefinite
            bordered.push_back(lc::bordered<T>(k, 151 + s, s == 8));
            rhs.push_back(lc::rhs_for<T>(m, 2, 161 + s));
            chol.set(s, spds.back(), rhs.back());
            ldlt.set(s, bordered.back(), rhs.back());
        }
        std::vector<la::status> sc(n), sl(n);
        const auto rc = la::cholesky_solve_batched(b, chol.matrices(), chol.rhs(), std::span<la::status>(sc));
        const auto rl = la::ldlt_solve_batched(b, ldlt.matrices(), ldlt.rhs(), std::span<la::status>(sl));
        ASSERT_TRUE(rc && rl);
        EXPECT_EQ(*rc, 1u);
        EXPECT_EQ(*rl, 1u);
        for (std::size_t s = 0; s < n; ++s) {
            EXPECT_EQ(sc[s], s == 5 ? la::status::not_positive_definite : la::status::ok) << s;
            EXPECT_EQ(sl[s], s == 8 ? la::status::singular : la::status::ok) << s;
            if (s != 5) {
                EXPECT_LE(lc::backward_error(spds[s], chol.solution(s), rhs[s]), lc::bound<T>(16, m)) << s;
            }
            if (s != 8) {
                EXPECT_LE(lc::backward_error(bordered[s], ldlt.solution(s), rhs[s]), lc::bound<T>(16, m)) << s;
            }
        }
    }

    TYPED_TEST_P(LinalgBatchedConformance, BatchesAreBitIdenticalAcrossExecutors)
    {
        using T = typename TypeParam::value_type;
        namespace lc = linalg_conformance;
        namespace la = snicholls::linalg;
        if constexpr (!TypeParam::deterministic) {
            GTEST_SKIP() << "the backend does not promise the same bits for every executor";
        } else {
            const typename TypeParam::backend b{};
            constexpr std::size_t n = 37, m = 6;
            lc::batch<T> x(n, m);
            for (std::size_t s = 0; s < n; ++s)
                x.set(s, s == 9 ? lc::rank_deficient<T>(m, 171 + s) : lc::spd<T>(m, 171 + s), lc::rhs_for<T>(m, 2, 181 + s));
            auto y = x, z = x, x2 = x, y2 = x;
            std::vector<la::status> s1(n), s2(n), s3(n);
            std::vector<la::pivot_index> p1(n * m), p2(n * m), p3(n * m);
            ASSERT_TRUE(la::lu_solve_batched(b, x.matrices(), std::span<la::pivot_index>(p1), x.rhs(), std::span<la::status>(s1)));
            ASSERT_TRUE(la::lu_solve_batched(b, y.matrices(), std::span<la::pivot_index>(p2), y.rhs(), std::span<la::status>(s2),
                                             lc::reversed_uneven_chunks{}));
            ASSERT_TRUE(la::lu_solve_batched(b, z.matrices(), std::span<la::pivot_index>(p3), z.rhs(), std::span<la::status>(s3), lc::three_threads{}));
            EXPECT_TRUE(lc::same_bits(x.b, y.b) && lc::same_bits(x.b, z.b));
            EXPECT_TRUE(lc::same_bits(x.a, y.a) && lc::same_bits(x.a, z.a));
            EXPECT_EQ(p1, p2);
            EXPECT_EQ(p1, p3);
            EXPECT_EQ(s1, s2);
            EXPECT_EQ(s1, s3);
            ASSERT_TRUE(la::ldlt_solve_batched(b, x2.matrices(), x2.rhs(), std::span<la::status>(s1)));
            ASSERT_TRUE(la::ldlt_solve_batched(b, y2.matrices(), y2.rhs(), std::span<la::status>(s2), lc::three_threads{}));
            EXPECT_TRUE(lc::same_bits(x2.b, y2.b));
            EXPECT_EQ(s1, s2);
        }
    }

    REGISTER_TYPED_TEST_SUITE_P(LinalgBatchedConformance, LuBatchReportsEachFailureForItsOwnSystem, CholeskyAndLdltBatchesReportEachFailure,
                                BatchesAreBitIdenticalAcrossExecutors);

} // namespace snicholls::conformance

/// Register the BLAS suite: `SNICHOLLS_LINALG_CONFORMANCE_BLAS(Prefix, linalg_case<B, T>...)`, at namespace scope.
#define SNICHOLLS_LINALG_CONFORMANCE_BLAS(prefix, ...)                                                                                     \
    namespace snicholls::conformance {                                                                                                     \
        using prefix##_BlasTypes = ::testing::Types<__VA_ARGS__>;                                                                      \
        INSTANTIATE_TYPED_TEST_SUITE_P(prefix, LinalgBlasConformance, prefix##_BlasTypes);                                             \
    }                                                                                                                                  \
    static_assert(true, "")

/// Register the factorisation suite.
#define SNICHOLLS_LINALG_CONFORMANCE_FACTORISATION(prefix, ...)                                                                            \
    namespace snicholls::conformance {                                                                                                     \
        using prefix##_FactorisationTypes = ::testing::Types<__VA_ARGS__>;                                                             \
        INSTANTIATE_TYPED_TEST_SUITE_P(prefix, LinalgFactorisationConformance, prefix##_FactorisationTypes);                           \
    }                                                                                                                                  \
    static_assert(true, "")

/// Register the decomposition suite.
#define SNICHOLLS_LINALG_CONFORMANCE_DECOMPOSITION(prefix, ...)                                                                            \
    namespace snicholls::conformance {                                                                                                     \
        using prefix##_DecompositionTypes = ::testing::Types<__VA_ARGS__>;                                                             \
        INSTANTIATE_TYPED_TEST_SUITE_P(prefix, LinalgDecompositionConformance, prefix##_DecompositionTypes);                           \
    }                                                                                                                                  \
    static_assert(true, "")

/// Register the batched suite.
#define SNICHOLLS_LINALG_CONFORMANCE_BATCHED(prefix, ...)                                                                                  \
    namespace snicholls::conformance {                                                                                                     \
        using prefix##_BatchedTypes = ::testing::Types<__VA_ARGS__>;                                                                   \
        INSTANTIATE_TYPED_TEST_SUITE_P(prefix, LinalgBatchedConformance, prefix##_BatchedTypes);                                       \
    }                                                                                                                                  \
    static_assert(true, "")

/// Register every suite, for a backend that models `snicholls::linalg::backend`.
#define SNICHOLLS_LINALG_CONFORMANCE(prefix, ...)                                                                                          \
    SNICHOLLS_LINALG_CONFORMANCE_BLAS(prefix, __VA_ARGS__);                                                                            \
    SNICHOLLS_LINALG_CONFORMANCE_FACTORISATION(prefix, __VA_ARGS__);                                                                   \
    SNICHOLLS_LINALG_CONFORMANCE_DECOMPOSITION(prefix, __VA_ARGS__);                                                                   \
    SNICHOLLS_LINALG_CONFORMANCE_BATCHED(prefix, __VA_ARGS__)

#endif /* snicholls_core_conformance_linalg_hpp */
