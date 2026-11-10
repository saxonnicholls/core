//
//  minifloat_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/ml/minifloat.hpp>: the published values of each format, every code of every
//  format decoded and re-encoded, ties to even at every midpoint of the 8-bit formats, overflow under both
//  policies, NaN and infinity, the bfloat16 fast path against the generic conversion, stochastic rounding's
//  mean, layout, and compile-time use.
//

#include <snicholls/core/ml/minifloat.hpp>

#include <gtest/gtest.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace {

namespace ml = snicholls::ml;
constexpr double inf = std::numeric_limits<double>::infinity();

// Layout: a word of 8 or 16 bits, trivially copyable, structural
static_assert(sizeof(ml::bfloat16) == 2 && sizeof(ml::float16) == 2 && sizeof(ml::float8_e4m3) == 1 && sizeof(ml::float4_e2m1) == 1);
static_assert(std::is_trivially_copyable_v<ml::bfloat16> && std::is_trivially_copyable_v<ml::float8_e5m2>);
template <ml::float8_e4m3 V> struct structural {};
static_assert(sizeof(structural<ml::float8_e4m3{0x38}>) == 1);
static_assert(ml::is_minifloat_v<ml::float16> && !ml::is_minifloat_v<float>);

// The published values: largest finite, one, smallest subnormal, and what lies beyond
static_assert(ml::to_double(ml::minifloat_max<ml::float16>()) == 65504.0 && ml::minifloat_max<ml::float16>().bits == 0x7bff);
static_assert(ml::to_double(ml::minifloat_max<ml::bfloat16>()) == 0x1.fep127 && ml::minifloat_max<ml::bfloat16>().bits == 0x7f7f);
static_assert(ml::to_double(ml::minifloat_max<ml::float8_e4m3>()) == 448.0 && ml::minifloat_max<ml::float8_e4m3>().bits == 0x7e);
static_assert(ml::to_double(ml::minifloat_max<ml::float8_e5m2>()) == 57344.0 && ml::minifloat_max<ml::float8_e5m2>().bits == 0x7b);
static_assert(ml::to_double(ml::minifloat_max<ml::float6_e3m2>()) == 28.0 && ml::to_double(ml::minifloat_max<ml::float6_e2m3>()) == 7.5);
static_assert(ml::to_double(ml::minifloat_max<ml::float4_e2m1>()) == 6.0);
static_assert(ml::to_minifloat<ml::float16>(1.0).bits == 0x3c00 && ml::to_minifloat<ml::bfloat16>(1.0).bits == 0x3f80);
static_assert(ml::to_minifloat<ml::float8_e4m3>(1.0).bits == 0x38 && ml::to_minifloat<ml::float8_e5m2>(1.0).bits == 0x3c);
static_assert(ml::to_minifloat<ml::float16>(0x1p-24).bits == 0x0001 && ml::to_double(ml::float16{0x0001}) == 0x1p-24);
static_assert(ml::to_double(ml::float8_e4m3{0x01}) == 0x1p-9 && ml::to_double(ml::float8_e5m2{0x01}) == 0x1p-16);
static_assert(ml::to_minifloat<ml::float16>(65520.0).bits == 0x7c00);   // the tie above the largest rounds to infinity
static_assert(ml::to_minifloat<ml::float16>(65519.0).bits == 0x7bff);
static_assert(ml::to_minifloat<ml::float16>(1e6, ml::overflow_policy::saturate).bits == 0x7bff);
static_assert(ml::to_minifloat<ml::float8_e4m3>(1000.0).bits == 0x7e);                                    // saturates by default
static_assert(ml::to_minifloat<ml::float8_e4m3>(1000.0, ml::overflow_policy::infinity).bits == 0x7f);     // NaN: no infinity
static_assert(ml::to_minifloat<ml::float8_e5m2>(1e6).bits == 0x7c && ml::to_double(ml::float8_e5m2{0x7c}) == inf);
static_assert(ml::to_minifloat<ml::float4_e2m1>(100.0).bits == 0x7 && ml::to_minifloat<ml::float4_e2m1>(-100.0).bits == 0xf);
static_assert(ml::to_minifloat<ml::float8_e4m3>(-inf).bits == 0xfe && ml::to_minifloat<ml::float16>(-inf).bits == 0xfc00);
static_assert(ml::to_minifloat<ml::float16>(-0.0).bits == 0x8000 && ml::to_minifloat<ml::float16>(1e-30).bits == 0x0000);
static_assert(ml::to_float(ml::to_bfloat16(3.14159265f)) == 3.140625f);

constexpr std::array<double, 8> fp4_values{0, 0.5, 1, 1.5, 2, 3, 4, 6};

TEST(MlMinifloat, Fp4HasExactlyTheMxValues)
{
    for (std::uint32_t c = 0; c < 16; ++c) {
        const double v = ml::to_double(ml::float4_e2m1::from_bits(c));
        EXPECT_EQ(v, (c & 8 ? -1 : 1) * fp4_values[c & 7]) << c;
        EXPECT_EQ(ml::decode_table<ml::float4_e2m1>[c], static_cast<float>(v));
    }
    EXPECT_EQ(ml::to_minifloat<ml::float4_e2m1>(2.5).bits, 0x4);   // a tie between 2 and 3: to even (2)
    EXPECT_EQ(ml::to_minifloat<ml::float4_e2m1>(5.0).bits, 0x6);   // a tie between 4 and 6: to even (4)
    EXPECT_EQ(ml::to_minifloat<ml::float4_e2m1>(0.25).bits, 0x0);  // a tie between 0 and 0.5: to even (0)
}

// Every code decodes and re-encodes to itself (NaNs to the canonical NaN), and the table agrees
template <typename F>
void round_trips()
{
    for (std::uint32_t c = 0; c < (1u << F::total_bits); ++c) {
        const F f = F::from_bits(c);
        const double v = ml::to_double(f);
        if (std::isnan(v)) {
            EXPECT_TRUE(F::has_nan) << c;
            EXPECT_TRUE(std::isnan(ml::to_double(ml::to_minifloat<F>(v))));
            continue;
        }
        EXPECT_EQ(ml::to_minifloat<F>(v).bits, c) << c;
        EXPECT_EQ(ml::to_minifloat_stochastic<F>(v, 0xffffffffu).bits, c) << c;   // exact values never move
        EXPECT_EQ(ml::to_float(f), static_cast<float>(v));
    }
}

TEST(MlMinifloat, EveryCodeOfEveryFormatRoundTrips)
{
    round_trips<ml::bfloat16>();
    round_trips<ml::float16>();
    round_trips<ml::float8_e5m2>();
    round_trips<ml::float8_e4m3>();
    round_trips<ml::float6_e3m2>();
    round_trips<ml::float6_e2m3>();
    round_trips<ml::float4_e2m1>();
    round_trips<ml::minifloat<3, 4>>();
}

// Between consecutive positive codes: below the midpoint rounds down, above rounds up, the midpoint to the even code
template <typename F>
void ties_to_even()
{
    const std::uint32_t top = ml::minifloat_max<F>().bits;
    for (std::uint32_t c = 0; c < top; ++c) {
        const double lo = ml::to_double(F::from_bits(c)), hi = ml::to_double(F::from_bits(c + 1));
        const double mid = (lo + hi) / 2;
        EXPECT_EQ(ml::to_minifloat<F>(mid).bits, (c % 2 == 0) ? c : c + 1) << c;
        EXPECT_EQ(ml::to_minifloat<F>(std::nextafter(mid, 0.0)).bits, c) << c;
        EXPECT_EQ(ml::to_minifloat<F>(std::nextafter(mid, inf)).bits, c + 1) << c;
        EXPECT_EQ(ml::to_minifloat<F>(-mid).bits, ((c % 2 == 0) ? c : c + 1) | (1u << (F::total_bits - 1))) << c;
    }
}

TEST(MlMinifloat, TiesGoToEvenAtEveryMidpoint)
{
    ties_to_even<ml::float8_e5m2>();
    ties_to_even<ml::float8_e4m3>();
    ties_to_even<ml::float6_e3m2>();
    ties_to_even<ml::float6_e2m3>();
    ties_to_even<ml::float16>();
}

TEST(MlMinifloat, NanAndInfinityFollowTheFormat)
{
    const double nan = std::numeric_limits<double>::quiet_NaN();
    EXPECT_TRUE(std::isnan(ml::to_double(ml::to_minifloat<ml::float16>(nan))));
    EXPECT_TRUE(std::isnan(ml::to_double(ml::to_minifloat<ml::bfloat16>(nan))));
    EXPECT_EQ(ml::to_minifloat<ml::float8_e4m3>(nan).bits, 0x7f);
    EXPECT_EQ(ml::to_minifloat<ml::float8_e4m3>(-nan).bits, 0xff);
    EXPECT_EQ(ml::to_minifloat<ml::float4_e2m1>(nan).bits, 0x7);   // no NaN: the largest magnitude
    EXPECT_EQ(ml::to_double(ml::to_minifloat<ml::float8_e5m2>(-inf)), -inf);
    EXPECT_EQ(ml::to_minifloat<ml::float8_e5m2>(inf, ml::overflow_policy::saturate).bits, 0x7b);
    EXPECT_EQ(ml::to_minifloat<ml::float8_e4m3>(inf).bits, 0x7e);
    EXPECT_EQ(ml::to_minifloat<ml::float8_e4m3>(inf, ml::overflow_policy::infinity).bits, 0x7f);
    EXPECT_EQ(ml::to_double(ml::to_minifloat<ml::bfloat16>(1e300)), inf);
    EXPECT_EQ(ml::to_minifloat<ml::bfloat16>(1e-300).bits, 0);   // below the smallest subnormal
    EXPECT_EQ(ml::to_minifloat<ml::bfloat16>(std::numeric_limits<double>::denorm_min()).bits, 0);
}

TEST(MlMinifloat, TheBfloat16FastPathGivesTheGenericBits)
{
    std::uint32_t x = 0x9e3779b9u;
    for (int i = 0; i < 2000000; ++i) {
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        const float f = std::bit_cast<float>(x);
        if (std::isnan(f)) {
            EXPECT_TRUE(std::isnan(ml::to_float(ml::to_bfloat16(f))));
            continue;
        }
        ASSERT_EQ(ml::to_bfloat16(f).bits, ml::to_minifloat<ml::bfloat16>(f).bits) << f;
    }
    for (const float f : {0.0f, -0.0f, 1.0f, 3.4e38f, -3.4e38f, std::numeric_limits<float>::max(), std::numeric_limits<float>::denorm_min(),
                          std::numeric_limits<float>::infinity(), 1.00390625f, 1.01171875f})   // the last two are ties
        EXPECT_EQ(ml::to_bfloat16(f).bits, ml::to_minifloat<ml::bfloat16>(f).bits) << f;
}

TEST(MlMinifloat, StochasticRoundingIsUnbiased)
{
    // 1 + a third of bfloat16's last place: rounds up a third of the time
    const double x = 1.0 + 0x1p-7 / 3;
    std::uint32_t r = 12345;
    double sum = 0;
    int up = 0;
    const int n = 300000;
    for (int i = 0; i < n; ++i) {
        r = r * 1664525u + 1013904223u;
        const auto b = ml::to_minifloat_stochastic<ml::bfloat16>(x, r);
        sum += ml::to_double(b);
        up += b.bits == 0x3f81;
        ASSERT_TRUE(b.bits == 0x3f80 || b.bits == 0x3f81);
    }
    EXPECT_NEAR(static_cast<double>(up) / n, 1.0 / 3, 0.005);
    EXPECT_NEAR(sum / n, x, 0x1p-7 * 0.01);
    // Magnitudes round away from zero when random / 2^32 is below the dropped fraction: random 0 always does (if
    // anything is dropped), the largest value never does; negatives the same in magnitude
    EXPECT_EQ(ml::to_minifloat_stochastic<ml::float8_e4m3>(-1.03, 0).bits, ml::to_minifloat<ml::float8_e4m3>(-1.125).bits);
    EXPECT_EQ(ml::to_minifloat_stochastic<ml::float8_e4m3>(-1.03, 0xffffffffu).bits, ml::to_minifloat<ml::float8_e4m3>(-1.0).bits);
    EXPECT_EQ(ml::to_minifloat_stochastic<ml::float8_e4m3>(447.9, 0).bits, 0x7e);
    EXPECT_EQ(ml::to_minifloat_stochastic<ml::float8_e4m3>(460.0, 0).bits, 0x7e);   // beyond 448: saturates, never NaN by default
}

TEST(MlMinifloat, DecodeTablesHoldEveryValue)
{
    static_assert(ml::decode_table<ml::float8_e4m3>.size() == 256 && ml::decode_table<ml::float4_e2m1>.size() == 16);
    static_assert(ml::decode_table<ml::float8_e4m3>[0x38] == 1.0f && ml::decode_table<ml::float8_e5m2>[0x7b] == 57344.0f);
    EXPECT_TRUE(std::isnan(ml::decode_table<ml::float8_e4m3>[0x7f]));
    EXPECT_EQ(ml::decode_table<ml::float8_e5m2>[0xfc], -std::numeric_limits<float>::infinity());
}

} // namespace
