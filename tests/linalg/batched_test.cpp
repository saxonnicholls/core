//
//  batched_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/linalg/batched.hpp>: the executor concept and for_each_system, the extent
//  checks, a failure reported for its own system only, vector and matrix right-hand sides, systems with
//  gaps between them, and determinism: the same bits for every executor, chunking and thread count. The
//  numerical identities of batched solves are the conformance suite's.
//

#include <snicholls/core/linalg/reference.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <span>
#include <thread>
#include <vector>

namespace {

namespace la = snicholls::linalg;
using la::batch_view;
using la::error;
using la::layout_right;
using la::pivot_index;
using la::status;

// ---- Executors ------------------------------------------------------------------------------------------

// Uneven chunks, run last first: what a work-stealing pool might do.
struct reversed_chunks {
    std::size_t chunk = 3;
    template <typename F>
    void operator()(std::size_t n, F&& f) const
    {
        std::vector<std::pair<std::size_t, std::size_t>> ranges;
        for (std::size_t b = 0, k = 0; b < n; ++k) {   // chunks of chunk, chunk + 1, ..., chunk + 3 systems
            const std::size_t e = std::min(n, b + chunk + k % 4);
            ranges.emplace_back(b, e);
            b = e;
        }
        std::reverse(ranges.begin(), ranges.end());
        for (const auto& [b, e] : ranges)
            f(b, e);
    }
};

// Real threads, as a pool would run them.
struct threads {
    unsigned count = 4;
    template <typename F>
    void operator()(std::size_t n, F&& f) const
    {
        std::vector<std::thread> pool;
        for (unsigned t = 0; t < count; ++t)
            pool.emplace_back([&, t] { f(n * t / count, n * (t + 1) / count); });
        for (auto& th : pool)
            th.join();
    }
};

struct not_an_executor {};

static_assert(la::range_executor<la::inline_executor> && la::range_executor<reversed_chunks> && la::range_executor<threads>);
static_assert(!la::range_executor<not_an_executor> && !la::range_executor<int>);
static_assert(la::batched_backend<la::reference_backend, double> && la::batched_backend<la::reference_backend, float>);

TEST(LinalgBatchedExecutors, ForEachSystemVisitsEveryIndexOnce)
{
    for (std::size_t n : {0u, 1u, 7u, 100u}) {
        std::vector<int> seen(n, 0);
        reversed_chunks ex;
        la::for_each_system(ex, n, [&](std::size_t i) { ++seen[i]; });
        EXPECT_TRUE(std::all_of(seen.begin(), seen.end(), [](int c) { return c == 1; })) << n;
        la::inline_executor in;
        la::for_each_system(in, n, [&](std::size_t i) { ++seen[i]; });
        EXPECT_TRUE(std::all_of(seen.begin(), seen.end(), [](int c) { return c == 2; })) << n;
    }
}

TEST(LinalgBatchedExecutors, CountFailuresCountsNonOk)
{
    const std::array<status, 5> s{status::ok, status::singular, status::ok, status::not_positive_definite, status::ok};
    EXPECT_EQ(la::count_failures(s), 2u);
    EXPECT_EQ(la::count_failures({}), 0u);
    static_assert(la::count_failures(std::array<status, 2>{status::ok, status::ok}) == 0);
}

// ---- A batch of small systems ---------------------------------------------------------------------------

// n systems of order m, each in a slot of `slot` elements with rows of `ld` (gaps between rows and systems).
struct batch {
    std::size_t n, m, ld, slot;
    std::vector<double> a, b;
    batch(std::size_t n_, std::size_t m_) : n(n_), m(m_), ld(m_ + 2), slot((m_ + 2) * m_ + 5), a(n_ * slot, -7.0), b(n_ * (m_ + 1), -7.0)
    {
        for (std::size_t s = 0; s < n; ++s)
            for (std::size_t i = 0; i < m; ++i) {
                for (std::size_t j = 0; j <= i; ++j) {
                    const double v = std::cos(static_cast<double>(s * 31 + i * 7 + j * 3)) / static_cast<double>(1 + i + j);
                    a[s * slot + i * ld + j] = v;
                    a[s * slot + j * ld + i] = v;
                }
                a[s * slot + i * ld + i] += static_cast<double>(m);   // symmetric, positive definite
                b[s * (m + 1) + i] = static_cast<double>(i + s % 5);
            }
    }
    [[nodiscard]] batch_view<double> matrices() { return {a.data(), {n, m, m}, {slot, ld, 1}}; }
    [[nodiscard]] la::md_span<double, 2, la::layout_stride> rhs() { return {b.data(), {n, m}, {m + 1, 1}}; }
    double& at(std::size_t s, std::size_t i, std::size_t j) { return a[s * slot + i * ld + j]; }
};

TEST(LinalgBatched, ExtentsThatDoNotFitWriteNothing)
{
    batch x(4, 3);
    const auto before = x.a;
    std::vector<pivot_index> p(12);
    std::vector<status> s(4, status::did_not_converge);
    const auto short_status = std::span<status>(s.data(), 3);
    const auto r1 = la::lu_solve_batched(la::reference, x.matrices(), std::span<pivot_index>(p.data(), 11), x.rhs(), std::span<status>(s));
    const auto r2 = la::lu_solve_batched(la::reference, x.matrices(), std::span<pivot_index>(p), x.rhs(), short_status);
    const auto r3 = la::cholesky_solve_batched(la::reference, x.matrices(), x.rhs(), short_status);
    const auto r4 = la::ldlt_solve_batched(la::reference, x.matrices(), x.rhs(), short_status);
    const batch_view<double> not_square(x.a.data(), {4, 3, 2}, {x.slot, x.ld, 1});
    const auto r5 = la::ldlt_solve_batched(la::reference, not_square, x.rhs(), std::span<status>(s));
    const la::md_span<double, 2, la::layout_stride> wrong_rows(x.b.data(), {4, 2}, {4, 1});
    const auto r6 = la::cholesky_solve_batched(la::reference, x.matrices(), wrong_rows, std::span<status>(s));
    for (const auto* r : {&r1, &r2, &r3, &r4, &r5, &r6})
        EXPECT_EQ(r->err(), error::dimension_mismatch);
    EXPECT_EQ(x.a, before);
    EXPECT_EQ(s[0], status::did_not_converge);
}

TEST(LinalgBatched, RightHandSidesThatStartAtTheMatricesAreReported)
{
    batch x(3, 2);
    const auto before = x.a;
    std::vector<pivot_index> p(6);
    std::vector<status> s(3, status::did_not_converge);
    const la::md_span<double, 2, la::layout_stride> rhs_on_a(x.a.data(), {3, 2}, {x.slot, x.ld});
    EXPECT_EQ(la::lu_solve_batched(la::reference, x.matrices(), std::span<pivot_index>(p), rhs_on_a, std::span<status>(s)).err(), error::aliased);
    EXPECT_EQ(la::cholesky_solve_batched(la::reference, x.matrices(), rhs_on_a, std::span<status>(s)).err(), error::aliased);
    EXPECT_EQ(la::ldlt_solve_batched(la::reference, x.matrices(), rhs_on_a, std::span<status>(s)).err(), error::aliased);
    // A status span on the right-hand sides' storage.
    std::vector<double> rhs(3 * 2, 1.0);
    const la::md_span<double, 2, la::layout_stride> r(rhs.data(), {3, 2}, {2, 1});
    const std::span<status> on_rhs(reinterpret_cast<status*>(rhs.data()), 3);
    EXPECT_EQ(la::ldlt_solve_batched(la::reference, x.matrices(), r, on_rhs).err(), error::aliased);
    EXPECT_EQ(x.a, before);
    EXPECT_EQ(s[0], status::did_not_converge);
}

TEST(LinalgBatched, AFailureIsReportedForItsOwnSystemOnly)
{
    batch x(9, 5);
    for (std::size_t j = 0; j < 5; ++j)   // system 2: two equal rows (and columns), singular
        x.at(2, 1, j) = x.at(2, 0, j);
    for (std::size_t i = 0; i < 5; ++i)
        x.at(2, i, 1) = x.at(2, i, 0);
    x.at(6, 3, 3) = std::numeric_limits<double>::quiet_NaN();   // system 6: NaN
    batch reference = x;
    std::vector<pivot_index> p(9 * 5);
    std::vector<status> s(9);
    const auto r = la::lu_solve_batched(la::reference, x.matrices(), std::span<pivot_index>(p), x.rhs(), std::span<status>(s));
    ASSERT_TRUE(r);
    EXPECT_EQ(*r, 2u);
    for (std::size_t i = 0; i < 9; ++i)
        EXPECT_EQ(s[i], i == 2 || i == 6 ? status::singular : status::ok) << i;

    // Every other system's solution is the single-system solve's, bit for bit.
    for (std::size_t sys : {0u, 1u, 3u, 8u}) {
        std::vector<pivot_index> ps(5);
        const auto a = snicholls::containers::submdspan(reference.matrices(), sys, snicholls::containers::full_extent, snicholls::containers::full_extent);
        const auto b = snicholls::containers::submdspan(reference.rhs(), sys, snicholls::containers::full_extent);
        ASSERT_TRUE(la::lu_factor(la::reference, a, std::span<pivot_index>(ps)));
        ASSERT_TRUE(la::lu_solve(la::reference, a, std::span<const pivot_index>(ps), b));
        for (std::size_t i = 0; i < 5; ++i)
            EXPECT_EQ(x.b[sys * 6 + i], reference.b[sys * 6 + i]);
    }
    // The gaps between rows and systems are untouched.
    EXPECT_EQ(x.a[5], -7.0);
    EXPECT_EQ(x.b[5], -7.0);
}

TEST(LinalgBatched, CholeskyAndLdltReportTheirOwnOutcomes)
{
    batch x(5, 4);
    x.at(1, 2, 2) = -50.0;   // indefinite: Cholesky refuses, LDLT solves
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            x.at(3, i, j) = 0.0;   // zero: both refuse
    batch y = x;
    std::vector<status> s(5);
    const auto rc = la::cholesky_solve_batched(la::reference, x.matrices(), x.rhs(), std::span<status>(s));
    ASSERT_TRUE(rc);
    EXPECT_EQ(*rc, 2u);
    EXPECT_EQ(s[1], status::not_positive_definite);
    EXPECT_EQ(s[3], status::not_positive_definite);
    EXPECT_EQ(s[0], status::ok);
    const auto rl = la::ldlt_solve_batched(la::reference, y.matrices(), y.rhs(), std::span<status>(s));
    ASSERT_TRUE(rl);
    EXPECT_EQ(*rl, 1u);
    EXPECT_EQ(s[1], status::ok);
    EXPECT_EQ(s[3], status::singular);
}

TEST(LinalgBatched, VectorAndMatrixRightHandSidesAgree)
{
    batch x(3, 4), y(3, 4);
    std::vector<status> s(3);
    std::vector<pivot_index> p(12);
    ASSERT_TRUE(la::lu_solve_batched(la::reference, x.matrices(), std::span<pivot_index>(p), x.rhs(), std::span<status>(s)));
    const batch_view<double> as_matrix(y.b.data(), {3, 4, 1}, {5, 1, 1});
    ASSERT_TRUE(la::lu_solve_batched(la::reference, y.matrices(), std::span<pivot_index>(p), as_matrix, std::span<status>(s)));
    EXPECT_EQ(x.b, y.b);
}

// Equal bit for bit (NaN included).
template <typename T>
bool same_bits(const std::vector<T>& x, const std::vector<T>& y)
{
    return x.size() == y.size() && std::memcmp(x.data(), y.data(), x.size() * sizeof(T)) == 0;
}

TEST(LinalgBatched, SameBitsForEveryExecutorAndThreadCount)
{
    constexpr std::size_t n = 61, m = 7;
    batch serial(n, m);
    serial.at(13, 0, 0) = std::numeric_limits<double>::quiet_NaN();
    batch chunked = serial, threaded = serial, lu_serial = serial, lu_threaded = serial;
    std::vector<status> s1(n), s2(n), s3(n), s4(n), s5(n);
    ASSERT_TRUE(la::ldlt_solve_batched(la::reference, serial.matrices(), serial.rhs(), std::span<status>(s1)));
    ASSERT_TRUE(la::ldlt_solve_batched(la::reference, chunked.matrices(), chunked.rhs(), std::span<status>(s2), reversed_chunks{}));
    ASSERT_TRUE(la::ldlt_solve_batched(la::reference, threaded.matrices(), threaded.rhs(), std::span<status>(s3), threads{5}));
    std::vector<pivot_index> p4(n * m), p5(n * m);
    ASSERT_TRUE(la::lu_solve_batched(la::reference, lu_serial.matrices(), std::span<pivot_index>(p4), lu_serial.rhs(), std::span<status>(s4)));
    ASSERT_TRUE(la::lu_solve_batched(la::reference, lu_threaded.matrices(), std::span<pivot_index>(p5), lu_threaded.rhs(), std::span<status>(s5),
                                     threads{3}));
    EXPECT_TRUE(same_bits(serial.b, chunked.b));
    EXPECT_TRUE(same_bits(serial.b, threaded.b));
    EXPECT_TRUE(same_bits(serial.a, chunked.a));
    EXPECT_TRUE(same_bits(serial.a, threaded.a));
    EXPECT_EQ(s1, s2);
    EXPECT_EQ(s1, s3);
    EXPECT_EQ(s1[13], status::singular);
    EXPECT_TRUE(same_bits(lu_serial.b, lu_threaded.b));
    EXPECT_TRUE(same_bits(lu_serial.a, lu_threaded.a));
    EXPECT_EQ(p4, p5);
    EXPECT_EQ(s4, s5);
}

} // namespace
