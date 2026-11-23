//
//  csc_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/sparse/csc.hpp>: the view and the fixed matrix, is_valid on every malformation,
//  from_triplets (any order, duplicates summed, invalid indices, short spans, mismatched triplets), every product
//  against the dense product of to_dense, transpose and its transpose, symmetric_permute_upper against the dense
//  P A Pᵀ (with its map, and refilling through the map), scale, the three norms, and a matrix built and multiplied
//  in a constant expression.
//

#include <snicholls/core/sparse/csc.hpp>

#include <snicholls/core/stats/random.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

namespace sp = snicholls::sparse;
namespace st = snicholls::stats;
using I = std::int32_t;

struct owned {
    std::size_t rows = 0, cols = 0;
    std::vector<I> cp, ri;
    std::vector<double> vx;
    [[nodiscard]] sp::csc_view<double, I> view() { return {rows, cols, cp, ri, vx}; }
    [[nodiscard]] sp::csc_view<const double, const I> cview() const { return {rows, cols, cp, ri, vx}; }
};

// A random m x n matrix with about `density` of its entries stored, from shuffled triplets with some duplicates
owned random_matrix(std::size_t m, std::size_t n, double density, std::uint64_t seed, std::vector<double>* dense = nullptr)
{
    st::wyrand g(seed);
    std::vector<I> ti, tj;
    std::vector<double> tv;
    if (dense)
        dense->assign(m * n, 0.0);
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j)
            if (st::uniform_real(g) < density) {
                const double v = st::uniform_real(g) - 0.5, half = v / 2;
                ti.insert(ti.end(), {static_cast<I>(i), static_cast<I>(i)});
                tj.insert(tj.end(), {static_cast<I>(j), static_cast<I>(j)});
                tv.insert(tv.end(), {half, v - half});   // a duplicate pair summing to v
                if (dense)
                    (*dense)[i * n + j] += half + (v - half);
            }
    std::vector<std::size_t> order(ti.size());
    for (std::size_t k = 0; k < order.size(); ++k)
        order[k] = k;
    st::shuffle(std::span<std::size_t>(order), g);
    std::vector<I> si(ti.size()), sj(ti.size());
    std::vector<double> sv(ti.size());
    for (std::size_t k = 0; k < order.size(); ++k) {
        si[k] = ti[order[k]];
        sj[k] = tj[order[k]];
        sv[k] = tv[order[k]];
    }
    owned a;
    a.cp.resize(n + 1);
    a.ri.resize(si.size());
    a.vx.resize(si.size());
    std::vector<I> work(std::max(m, n));
    auto v = a.view();
    const auto r = sp::from_triplets(m, n, std::span<const I>(si), std::span<const I>(sj), std::span<const double>(sv), v, std::span<I>(work));
    EXPECT_TRUE(r.is_ok());
    a.rows = m;
    a.cols = n;
    a.ri.resize(v.nnz());
    a.vx.resize(v.nnz());
    return a;
}

std::vector<double> dense_of(const owned& a)
{
    std::vector<double> d(a.rows * a.cols);
    sp::to_dense(a.cview(), std::span<double>(d));
    return d;
}

std::vector<double> random_vector(std::size_t n, std::uint64_t seed)
{
    st::wyrand g(seed);
    std::vector<double> x(n);
    for (double& v : x)
        v = st::uniform_real(g) * 2 - 1;
    return x;
}

TEST(SparseCscView, CountsEntriesAndConvertsToReadOnly)
{
    std::vector<I> cp{0, 1, 3}, ri{0, 0, 1};
    std::vector<double> vx{1, 2, 3};
    sp::csc_view<double, I> v{2, 2, cp, ri, vx};
    EXPECT_EQ(v.nnz(), 3u);
    EXPECT_EQ((sp::csc_view<double, I>{}.nnz()), 0u);
    const sp::csc_view<const double, const I> c = v;   // implicit
    EXPECT_EQ(c.nnz(), 3u);
    EXPECT_EQ(v.as_const().values.data(), vx.data());
    static_assert(std::is_convertible_v<sp::csc_view<double, I>, sp::csc_view<const double, const I>>);
    static_assert(noexcept(v.nnz()) && noexcept(v.as_const()));
}

TEST(SparseCscMatrix, HoldsItsStorageAndViewsIt)
{
    sp::csc_matrix<double, 3, 2, 4> m{};
    EXPECT_EQ(m.nnz(), 0u);
    m.col_ptr = {0, 2, 3};
    m.row_idx = {0, 2, 1, 0};
    m.values = {1, 2, 3, 0};
    EXPECT_EQ(m.nnz(), 3u);
    EXPECT_TRUE(sp::is_valid(m.view()));
    EXPECT_TRUE(sp::is_valid(std::as_const(m).view()));
    EXPECT_EQ(m.view().rows, 3u);
    EXPECT_EQ(m.view().cols, 2u);
}

TEST(SparseCscIsValid, RejectsEveryMalformation)
{
    std::vector<I> cp{0, 2, 3}, ri{0, 2, 1};
    std::vector<double> vx{1, 2, 3};
    auto make = [&](std::size_t rows) { return sp::csc_view<const double, const I>{rows, 2, cp, ri, vx}; };
    EXPECT_TRUE(sp::is_valid(make(3)));
    EXPECT_FALSE(sp::is_valid(make(2)));   // row 2 out of range
    cp = {1, 2, 3};
    EXPECT_FALSE(sp::is_valid(make(3)));   // not from 0
    cp = {0, 2, 1};
    EXPECT_FALSE(sp::is_valid(make(3)));   // decreasing
    cp = {0, 2, 4};
    EXPECT_FALSE(sp::is_valid(make(3)));   // spans too short
    cp = {0, 2, 3};
    ri = {2, 0, 1};
    EXPECT_FALSE(sp::is_valid(make(3)));   // not ascending
    ri = {0, 0, 1};
    EXPECT_FALSE(sp::is_valid(make(3)));   // repeated
    ri = {0, -1, 1};
    EXPECT_FALSE(sp::is_valid(make(3)));   // negative
    EXPECT_FALSE(sp::is_valid(sp::csc_view<const double, const I>{3, 2, std::span<const I>(cp).first(2), ri, vx}));   // col_ptr short
    EXPECT_TRUE(sp::is_valid(sp::csc_view<const double, const I>{0, 0, std::span<const I>(cp).first(1), {}, {}}));
}

TEST(SparseCscFromTriplets, SortsSumsDuplicatesAndAcceptsAnyOrder)
{
    // (row, col, value), out of order, (1, 0) three times and (0, 2) twice
    const std::vector<I> ti{2, 1, 0, 1, 0, 1, 0}, tj{2, 0, 2, 0, 0, 0, 2};
    const std::vector<double> tv{5, 1, 2, 10, 7, 100, 3};
    std::vector<I> cp(4), ri(7), work(3);
    std::vector<double> vx(7);
    sp::csc_view<double, I> out{0, 0, cp, ri, vx};
    const auto r = sp::from_triplets(3, 3, std::span<const I>(ti), std::span<const I>(tj), std::span<const double>(tv), out, std::span<I>(work));
    ASSERT_TRUE(r.is_ok());
    EXPECT_EQ(*r, 4u);
    EXPECT_EQ(out.rows, 3u);
    EXPECT_EQ(out.cols, 3u);
    EXPECT_EQ(cp, (std::vector<I>{0, 2, 2, 4}));
    EXPECT_EQ((std::vector<I>(ri.begin(), ri.begin() + 4)), (std::vector<I>{0, 1, 0, 2}));
    EXPECT_EQ((std::vector<double>(vx.begin(), vx.begin() + 4)), (std::vector<double>{7, 111, 5, 5}));
    EXPECT_TRUE(sp::is_valid(out));
    // Random shapes agree with dense accumulation, including empty rows and columns
    for (std::uint64_t seed = 1; seed <= 20; ++seed) {
        std::vector<double> dense;
        const owned a = random_matrix(1 + seed % 7, 1 + seed % 5, 0.4, seed, &dense);
        EXPECT_TRUE(sp::is_valid(a.cview()));
        EXPECT_EQ(dense_of(a), dense) << seed;
    }
    // Nothing at all
    std::vector<I> cp0(3), w0(2);
    sp::csc_view<double, I> empty{0, 0, cp0, {}, {}};
    ASSERT_TRUE(sp::from_triplets(2, 2, std::span<const I>(), std::span<const I>(), std::span<const double>(), empty, std::span<I>(w0)).is_ok());
    EXPECT_EQ(empty.nnz(), 0u);
}

TEST(SparseCscFromTriplets, ReportsInvalidIndicesShortSpansAndMismatchedTriplets)
{
    std::vector<I> cp(4), ri(3), work(3);
    std::vector<double> vx(3);
    sp::csc_view<double, I> out{0, 0, cp, ri, vx};
    const std::vector<I> good{0, 1, 2};
    const std::vector<double> v{1, 2, 3};
    auto call = [&](std::vector<I> ti, std::vector<I> tj, std::size_t rows, std::size_t cols, std::span<I> w) {
        return sp::from_triplets(rows, cols, std::span<const I>(ti), std::span<const I>(tj), std::span<const double>(v).first(ti.size() < 3 ? ti.size() : 3), out, w);
    };
    EXPECT_EQ(call({0, 1, 3}, good, 3, 3, work).unwrap_err(), sp::sparse_error::invalid_index);
    EXPECT_EQ(call({0, -1, 2}, good, 3, 3, work).unwrap_err(), sp::sparse_error::invalid_index);
    EXPECT_EQ(call(good, {0, 1, 3}, 3, 3, work).unwrap_err(), sp::sparse_error::invalid_index);
    EXPECT_EQ(call(good, {0, 1, 2}, 3, 4, work).unwrap_err(), sp::sparse_error::insufficient_capacity);   // col_ptr short
    EXPECT_EQ(call(good, good, 3, 3, std::span<I>(work).first(2)).unwrap_err(), sp::sparse_error::insufficient_capacity);
    EXPECT_EQ(sp::from_triplets(3, 3, std::span<const I>(good), std::span<const I>(good).first(2), std::span<const double>(v), out, std::span<I>(work)).unwrap_err(),
              sp::sparse_error::dimension_mismatch);
    sp::csc_view<double, I> small{0, 0, cp, std::span<I>(ri).first(2), vx};
    EXPECT_EQ(sp::from_triplets(3, 3, std::span<const I>(good), std::span<const I>(good), std::span<const double>(v), small, std::span<I>(work)).unwrap_err(),
              sp::sparse_error::insufficient_capacity);
}

TEST(SparseCscProducts, EverySpmvMatchesTheDenseProduct)
{
    for (std::uint64_t seed = 1; seed <= 12; ++seed) {
        const std::size_t m = 3 + seed % 6, n = 2 + seed % 5;
        const owned a = random_matrix(m, n, 0.35, seed);
        const auto d = dense_of(a);
        const auto x = random_vector(n, seed + 100), xt = random_vector(m, seed + 200);
        const double alpha = 1.5, beta = -0.25;
        // y ← αAx + βy
        auto y = random_vector(m, seed + 300), expect = y;
        for (std::size_t i = 0; i < m; ++i) {
            double s = 0;
            for (std::size_t j = 0; j < n; ++j)
                s += d[i * n + j] * x[j];
            expect[i] = alpha * s + beta * expect[i];
        }
        sp::spmv(a.cview(), std::span<const double>(x), std::span<double>(y), alpha, beta);
        for (std::size_t i = 0; i < m; ++i)
            EXPECT_NEAR(y[i], expect[i], 1e-14) << seed;
        // y ← αAᵀx + βy
        auto yt = random_vector(n, seed + 400), expect_t = yt;
        for (std::size_t j = 0; j < n; ++j) {
            double s = 0;
            for (std::size_t i = 0; i < m; ++i)
                s += d[i * n + j] * xt[i];
            expect_t[j] = alpha * s + beta * expect_t[j];
        }
        sp::spmv_transpose(a.cview(), std::span<const double>(xt), std::span<double>(yt), alpha, beta);
        for (std::size_t j = 0; j < n; ++j)
            EXPECT_NEAR(yt[j], expect_t[j], 1e-14) << seed;
    }
    // β = 0 overwrites, NaN included; the defaults are α = 1, β = 0
    const owned a = random_matrix(4, 4, 0.5, 9);
    const auto x = random_vector(4, 1);
    std::vector<double> y(4, std::nan("")), z(4);
    sp::spmv(a.cview(), std::span<const double>(x), std::span<double>(y));
    sp::spmv(a.cview(), std::span<const double>(x), std::span<double>(z), 1.0, 0.0);
    EXPECT_EQ(y, z);
    std::fill(y.begin(), y.end(), std::nan(""));
    sp::spmv_transpose(a.cview(), std::span<const double>(x), std::span<double>(y));
    for (double v : y)
        EXPECT_FALSE(std::isnan(v));
}

TEST(SparseCscProducts, SymmetricUpperMatchesTheFullDenseProductAndIgnoresTheLowerPart)
{
    for (std::uint64_t seed = 1; seed <= 10; ++seed) {
        const std::size_t n = 3 + seed % 7;
        const owned full = random_matrix(n, n, 0.4, seed);
        auto d = dense_of(full);
        // The symmetric matrix whose upper triangle is full's; full also has lower entries, which must be ignored
        std::vector<double> s(n * n);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
                s[i * n + j] = i <= j ? d[i * n + j] : d[j * n + i];
        const auto x = random_vector(n, seed + 7);
        auto y = random_vector(n, seed + 8), expect = y;
        for (std::size_t i = 0; i < n; ++i) {
            double t = 0;
            for (std::size_t j = 0; j < n; ++j)
                t += s[i * n + j] * x[j];
            expect[i] = 2.0 * t + 0.5 * expect[i];
        }
        sp::spmv_symmetric_upper(full.cview(), std::span<const double>(x), std::span<double>(y), 2.0, 0.5);
        for (std::size_t i = 0; i < n; ++i)
            EXPECT_NEAR(y[i], expect[i], 1e-14) << seed;
    }
}

TEST(SparseCscTranspose, MatchesTheDenseTransposeAndTwiceIsTheIdentity)
{
    for (std::uint64_t seed = 1; seed <= 12; ++seed) {
        const std::size_t m = 1 + seed % 6, n = 1 + seed % 4;
        const owned a = random_matrix(m, n, 0.4, seed);
        const auto d = dense_of(a);
        owned t;
        t.cp.resize(m + 1);
        t.ri.resize(a.cview().nnz());
        t.vx.resize(a.cview().nnz());
        std::vector<I> work(m);
        auto tv = t.view();
        ASSERT_TRUE(sp::transpose(a.cview(), tv, std::span<I>(work)).is_ok());
        t.rows = tv.rows;
        t.cols = tv.cols;
        EXPECT_EQ(t.rows, n);
        EXPECT_EQ(t.cols, m);
        EXPECT_TRUE(sp::is_valid(t.cview()));
        const auto dt = dense_of(t);
        for (std::size_t i = 0; i < m; ++i)
            for (std::size_t j = 0; j < n; ++j)
                EXPECT_EQ(dt[j * m + i], d[i * n + j]);
        owned tt;
        tt.cp.resize(n + 1);
        tt.ri.resize(a.cview().nnz());
        tt.vx.resize(a.cview().nnz());
        std::vector<I> work2(n);
        auto ttv = tt.view();
        ASSERT_TRUE(sp::transpose(t.cview(), ttv, std::span<I>(work2)).is_ok());
        EXPECT_EQ(tt.cp, a.cp);
        EXPECT_EQ(tt.ri, a.ri);
        EXPECT_EQ(tt.vx, a.vx);
    }
    const owned a = random_matrix(3, 3, 0.5, 4);
    std::vector<I> cp(4), ri(1), w(3);
    std::vector<double> vx(1);
    sp::csc_view<double, I> small{0, 0, cp, ri, vx};
    EXPECT_EQ(sp::transpose(a.cview(), small, std::span<I>(w)).unwrap_err(), sp::sparse_error::insufficient_capacity);
}

TEST(SparseCscSymmetricPermuteUpper, MatchesTheDensePermutationAndItsMapRefills)
{
    for (std::uint64_t seed = 1; seed <= 15; ++seed) {
        const std::size_t n = 2 + seed % 9;
        // An upper triangle (with a few lower entries, which are dropped)
        const owned full = random_matrix(n, n, 0.45, seed);
        const auto d = dense_of(full);
        std::vector<I> perm(n);
        for (std::size_t k = 0; k < n; ++k)
            perm[k] = static_cast<I>(k);
        st::wyrand g(seed);
        st::shuffle(std::span<I>(perm), g);
        const std::size_t nnz = full.cview().nnz();
        owned c;
        c.cp.resize(n + 1);
        c.ri.resize(nnz);
        c.vx.resize(nnz);
        std::vector<I> work(2 * n), map(nnz);
        auto cv = c.view();
        ASSERT_TRUE(sp::symmetric_permute_upper(full.cview(), std::span<const I>(perm), cv, std::span<I>(work), std::span<I>(map)).is_ok());
        c.rows = c.cols = n;
        c.ri.resize(cv.nnz());
        c.vx.resize(cv.nnz());
        EXPECT_TRUE(sp::is_valid(c.cview()));
        // C(a, b) = S(perm[a], perm[b]) for the symmetric S of full's upper triangle; only C's upper triangle stored
        const auto dc = dense_of(c);
        for (std::size_t r = 0; r < n; ++r)
            for (std::size_t s = 0; s < n; ++s) {
                const std::size_t i = static_cast<std::size_t>(perm[r]), j = static_cast<std::size_t>(perm[s]);
                const double expect = i <= j ? d[i * n + j] : d[j * n + i];
                EXPECT_EQ(r <= s ? dc[r * n + s] : dc[s * n + r], expect) << seed;
                if (r > s) {
                    EXPECT_EQ(dc[r * n + s], 0.0);
                }
            }
        // The map: each upper entry's position in C, -1 for a lower one; refilling through it gives the same values
        std::vector<double> refill(c.vx.size(), -99);
        for (std::size_t j = 0; j < n; ++j)
            for (std::size_t q = static_cast<std::size_t>(full.cp[j]); q < static_cast<std::size_t>(full.cp[j + 1]); ++q) {
                if (static_cast<std::size_t>(full.ri[q]) > j) {
                    EXPECT_EQ(map[q], -1);
                } else {
                    refill[static_cast<std::size_t>(map[q])] = full.vx[q];
                }
            }
        EXPECT_EQ(refill, c.vx) << seed;
        // Without the map, the same matrix
        owned c2;
        c2.cp.resize(n + 1);
        c2.ri.resize(nnz);
        c2.vx.resize(nnz);
        auto cv2 = c2.view();
        ASSERT_TRUE(sp::symmetric_permute_upper(full.cview(), std::span<const I>(perm), cv2, std::span<I>(work)).is_ok());
        EXPECT_EQ(std::vector<I>(c2.cp), c.cp);
    }
}

TEST(SparseCscSymmetricPermuteUpper, ReportsErrors)
{
    const owned a = random_matrix(3, 3, 0.6, 2);
    const owned rect = random_matrix(3, 2, 0.6, 2);
    std::vector<I> cp(4), ri(9), work(6), map(9);
    std::vector<double> vx(9);
    sp::csc_view<double, I> out{0, 0, cp, ri, vx};
    const std::vector<I> good{2, 0, 1}, repeat{0, 0, 1}, range{0, 1, 3};
    auto call = [&](const owned& m, const std::vector<I>& p, std::span<I> w, std::span<I> mp = {}) {
        return sp::symmetric_permute_upper(m.cview(), std::span<const I>(p), out, w, mp);
    };
    EXPECT_TRUE(call(a, good, work).is_ok());
    EXPECT_EQ(call(rect, std::vector<I>{1, 0}, work).unwrap_err(), sp::sparse_error::dimension_mismatch);
    EXPECT_EQ(call(a, std::vector<I>{1, 0}, work).unwrap_err(), sp::sparse_error::dimension_mismatch);
    EXPECT_EQ(call(a, repeat, work).unwrap_err(), sp::sparse_error::invalid_index);
    EXPECT_EQ(call(a, range, work).unwrap_err(), sp::sparse_error::invalid_index);
    EXPECT_EQ(call(a, good, std::span<I>(work).first(5)).unwrap_err(), sp::sparse_error::insufficient_capacity);
    EXPECT_EQ(call(a, good, work, std::span<I>(map).first(a.cview().nnz() - 1)).unwrap_err(), sp::sparse_error::insufficient_capacity);
}

TEST(SparseCscScale, ScalesRowsAndColumnsOrEither)
{
    owned a = random_matrix(3, 4, 0.7, 5);
    const auto d = dense_of(a);
    const std::vector<double> r{2, 3, 5}, c{7, 11, 13, 17};
    sp::scale(a.view(), std::span<const double>(r), std::span<const double>(c));
    auto ds = dense_of(a);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            EXPECT_DOUBLE_EQ(ds[i * 4 + j], d[i * 4 + j] * r[i] * c[j]);
    owned b = random_matrix(3, 4, 0.7, 5);
    sp::scale(b.view(), std::span<const double>(r), std::span<const double>());
    ds = dense_of(b);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            EXPECT_DOUBLE_EQ(ds[i * 4 + j], d[i * 4 + j] * r[i]);
    owned e = random_matrix(3, 4, 0.7, 5);
    sp::scale(e.view(), std::span<const double>(), std::span<const double>(c));
    ds = dense_of(e);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            EXPECT_DOUBLE_EQ(ds[i * 4 + j], d[i * 4 + j] * c[j]);
}

TEST(SparseCscNorms, ColumnRowAndSymmetricInfinityNormsMatchDense)
{
    for (std::uint64_t seed = 1; seed <= 10; ++seed) {
        const std::size_t m = 2 + seed % 5, n = 2 + seed % 4;
        const owned a = random_matrix(m, n, 0.4, seed);
        const auto d = dense_of(a);
        std::vector<double> cn(n), rn(m);
        sp::column_inf_norms(a.cview(), std::span<double>(cn));
        sp::row_inf_norms(a.cview(), std::span<double>(rn));
        for (std::size_t j = 0; j < n; ++j) {
            double e = 0;
            for (std::size_t i = 0; i < m; ++i)
                e = std::max(e, std::abs(d[i * n + j]));
            EXPECT_EQ(cn[j], e);
        }
        for (std::size_t i = 0; i < m; ++i) {
            double e = 0;
            for (std::size_t j = 0; j < n; ++j)
                e = std::max(e, std::abs(d[i * n + j]));
            EXPECT_EQ(rn[i], e);
        }
        const owned s = random_matrix(n, n, 0.5, seed + 50);
        const auto ds = dense_of(s);
        std::vector<double> sn(n);
        sp::symmetric_upper_inf_norms(s.cview(), std::span<double>(sn));
        for (std::size_t j = 0; j < n; ++j) {
            double e = 0;
            for (std::size_t i = 0; i < n; ++i)
                e = std::max(e, std::abs(i <= j ? ds[i * n + j] : ds[j * n + i]));
            EXPECT_EQ(sn[j], e) << seed;
        }
    }
}

TEST(SparseCscToDense, SumsExplicitEntriesAndZeroesTheRest)
{
    std::vector<I> cp{0, 2, 2, 3}, ri{0, 1, 1};
    std::vector<double> vx{1, 2, 0};   // an explicitly stored zero
    std::vector<double> d(6, -1);
    sp::to_dense(sp::csc_view<const double, const I>{2, 3, cp, ri, vx}, std::span<double>(d));
    EXPECT_EQ(d, (std::vector<double>{1, 0, 0, 2, 0, 0}));
}

// A 3 x 3 matrix assembled, transposed and multiplied in a constant expression
consteval double built_at_compile_time()
{
    const std::array<I, 4> ti{2, 0, 1, 0}, tj{0, 0, 2, 0};
    const std::array<double, 4> tv{3, 1, 4, 1};
    sp::csc_matrix<double, 3, 3, 4> a{}, t{};
    auto av = a.view();
    std::array<I, 3> work{};
    if (!sp::from_triplets(3, 3, std::span<const I>(ti), std::span<const I>(tj), std::span<const double>(tv), av, std::span<I>(work)))
        return -1;
    auto tview = t.view();
    if (!sp::transpose(std::as_const(a).view(), tview, std::span<I>(work)))
        return -1;
    const std::array<double, 3> x{1, 2, 3};
    std::array<double, 3> y{};
    sp::spmv(std::as_const(t).view(), std::span<const double>(x), std::span<double>(y));   // Aᵀx = (2 + 9, 0, 8)
    return y[0] * 100 + y[1] * 10 + y[2];
}

TEST(SparseCsc, BuildsTransposesAndMultipliesInAConstantExpression)
{
    static_assert(built_at_compile_time() == 1108.0);
    static_assert(noexcept(sp::spmv(std::declval<sp::csc_view<const double, const I>>(), std::declval<std::span<const double>>(), std::declval<std::span<double>>())));
    static_assert(noexcept(sp::from_triplets(3, 3, std::declval<std::span<const I>>(), std::declval<std::span<const I>>(), std::declval<std::span<const double>>(),
                                             std::declval<sp::csc_view<double, I>&>(), std::declval<std::span<I>>())));
    EXPECT_EQ(built_at_compile_time(), 1108.0);
}

} // namespace
