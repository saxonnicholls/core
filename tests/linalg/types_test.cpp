//
//  types_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/linalg/types.hpp>: the scalar concept, the error and status codes, the
//  view helpers (transposed, as_vector, as_column), the pivot tolerance and the signed log-determinant.
//

#include <snicholls/core/linalg/types.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <limits>
#include <span>
#include <type_traits>

namespace {

namespace la = snicholls::linalg;
using la::layout_left;
using la::layout_right;
using la::layout_stride;
using la::matrix_view;

static_assert(la::scalar<float> && la::scalar<double> && la::scalar<long double>);
static_assert(!la::scalar<int> && !la::scalar<std::uint8_t> && !la::scalar<bool>);

// Every code fits a byte and is trivially copyable, so a status span is one byte per system on any device.
static_assert(sizeof(la::error) == 1 && sizeof(la::status) == 1 && std::is_trivially_copyable_v<la::signed_log_det<double>>);
static_assert(la::to_status(la::error::singular) == la::status::singular);
static_assert(la::to_status(la::error::not_positive_definite) == la::status::not_positive_definite);
static_assert(la::to_status(la::error::did_not_converge) == la::status::did_not_converge);
static_assert(la::to_status(la::error::dimension_mismatch) == la::status::dimension_mismatch);
static_assert(la::to_status(la::error::aliased) == la::status::aliased);
static_assert(static_cast<int>(la::status::ok) == 0);

// The tolerance is relative: epsilon * n * scale, in the argument's own type.
static_assert(la::pivot_tolerance<double>(10, 2.0) == std::numeric_limits<double>::epsilon() * 20.0);
static_assert(la::pivot_tolerance<float>(4, 1.0f) == std::numeric_limits<float>::epsilon() * 4.0f);
static_assert(la::pivot_tolerance<double>(10, 0.0) == 0.0);

// The LDLT growth limit is n / epsilon^(1/4), linear in n, and wider for double than for float.
static_assert(la::ldlt_growth_limit<double>(2) == 2 * la::ldlt_growth_limit<double>(1));
static_assert(la::ldlt_growth_limit<double>(1) > 8000.0 && la::ldlt_growth_limit<double>(1) < 8300.0);
static_assert(la::ldlt_growth_limit<float>(1) > 50.0f && la::ldlt_growth_limit<float>(1) < 60.0f);

// transposed keeps unit strides visible: row-major becomes column-major and back.
static_assert(std::is_same_v<decltype(la::transposed(std::declval<matrix_view<double, layout_right>>())), matrix_view<double, layout_left>>);
static_assert(std::is_same_v<decltype(la::transposed(std::declval<matrix_view<double, layout_left>>())), matrix_view<double, layout_right>>);
static_assert(std::is_same_v<decltype(la::transposed(std::declval<matrix_view<double>>())), matrix_view<double>>);

consteval bool transposes_agree()
{
    std::array<int, 6> s{1, 2, 3, 4, 5, 6};
    const matrix_view<int, layout_right> r(s.data(), {2, 3});
    const matrix_view<int, layout_left> l(s.data(), {2, 3});
    const matrix_view<int> st(s.data(), {2, 3}, {3, 1});
    const auto rt = la::transposed(r);
    const auto lt = la::transposed(l);
    const auto stt = la::transposed(st);
    bool ok = rt.extent(0) == 3 && lt.extent(1) == 2 && stt.extent(0) == 3;
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            ok = ok && rt(j, i) == r(i, j) && lt(j, i) == l(i, j) && stt(j, i) == st(i, j);
    return ok;
}
static_assert(transposes_agree());

TEST(LinalgTypes, AsVectorAndAsColumnViewTheSameElements)
{
    std::array<double, 4> v{1, 2, 3, 4};
    const auto x = la::as_vector(std::span<double>(v));
    EXPECT_EQ(x.extent(0), 4u);
    EXPECT_EQ(&x(2), &v[2]);
    const auto c = la::as_column(x);
    EXPECT_EQ(c.extent(0), 4u);
    EXPECT_EQ(c.extent(1), 1u);
    EXPECT_EQ(&c(3, 0), &v[3]);
    // A strided vector (every other element) stays strided as a column.
    const la::vector_view<double> every_other(v.data(), {2}, {2});
    const auto c2 = la::as_column(every_other);
    EXPECT_EQ(&c2(1, 0), &v[2]);
}

TEST(LinalgTypes, SignedLogDetValueIsSignTimesExp)
{
    EXPECT_NEAR((la::signed_log_det<double>{-1, std::log(6.0)}.value()), -6.0, 1e-12);
    EXPECT_EQ((la::signed_log_det<double>{0, -std::numeric_limits<double>::infinity()}.value()), 0.0);
    EXPECT_NEAR((la::signed_log_det<float>{1, 0.0f}.value()), 1.0f, 0.0f);
    static_assert(la::signed_log_det<double>{}.sign == 1);
}

} // namespace
