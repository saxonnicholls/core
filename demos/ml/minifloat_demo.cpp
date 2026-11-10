//
//  minifloat_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/ml/minifloat.hpp>: one value in every format, FP8 saturation, an 8-bit decode
//  table, and why training rounds bfloat16 weights stochastically. Exits non-zero if a result is not what
//  it documents.
//

#include <snicholls/core/ml/minifloat.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>

int main()
{
    std::printf("snicholls-core minifloat demo\n");
    namespace ml = snicholls::ml;
    bool ok = true;

    //! [minifloat]
    // pi in each format: the bits, and the value they hold. Conversions are constexpr.
    constexpr double pi = 3.14159265358979;
    constexpr ml::bfloat16 b = ml::to_minifloat<ml::bfloat16>(pi);
    constexpr ml::float16 h = ml::to_minifloat<ml::float16>(pi);
    constexpr ml::float8_e4m3 e4 = ml::to_minifloat<ml::float8_e4m3>(pi);
    constexpr ml::float8_e5m2 e5 = ml::to_minifloat<ml::float8_e5m2>(pi);
    constexpr ml::float4_e2m1 f4 = ml::to_minifloat<ml::float4_e2m1>(pi);
    static_assert(ml::to_double(e4) == 3.25 && ml::to_double(f4) == 3.0);

    // FP8 E4M3 has no infinity: an activation of 1000 saturates to 448 by default (as FP8 training wants), or
    // becomes NaN under the non-saturating policy. The MX FP6 formats saturate too.
    constexpr ml::float8_e4m3 big = ml::to_minifloat<ml::float8_e4m3>(1000.0);
    constexpr ml::float8_e4m3 big_nan = ml::to_minifloat<ml::float8_e4m3>(1000.0, ml::overflow_policy::infinity);
    constexpr double e3m2 = ml::to_double(ml::to_minifloat<ml::float6_e3m2>(pi)), e2m3 = ml::to_double(ml::to_minifloat<ml::float6_e2m3>(pi));

    // Decoding an 8-bit format is one load from a table the compiler built.
    constexpr float one = ml::decode_table<ml::float8_e4m3>[0x38];

    // A weight of 1 in bfloat16 updated 1,000 times by 1e-4: rounding to nearest loses every update, because
    // 1e-4 is below half of bfloat16's last place at 1 (2^-8); stochastic rounding keeps them on average (each
    // step rounds up with probability 1e-4 / 2^-7, so the total is within about 0.03 of 1.1).
    float nearest = 1.0f, stochastic = 1.0f;
    std::uint32_t r = 2463534242u;
    for (int i = 0; i < 1000; ++i) {
        nearest = ml::to_float(ml::to_bfloat16(nearest + 1e-4f));
        r ^= r << 13; r ^= r >> 17; r ^= r << 5;   // the caller's random bits
        stochastic = ml::to_float(ml::to_minifloat_stochastic<ml::bfloat16>(stochastic + 1e-4f, r));
    }
    //! [minifloat]

    std::printf("  pi: bfloat16 %#06x = %.6f, float16 %#06x = %.6f, e4m3 %#04x = %.4f, e5m2 %#04x = %.4f, fp4 %#03x = %.1f\n", b.bits,
                ml::to_double(b), h.bits, ml::to_double(h), e4.bits, ml::to_double(e4), e5.bits, ml::to_double(e5), f4.bits, ml::to_double(f4));
    ok &= b.bits == 0x4049 && h.bits == 0x4248 && e4.bits == 0x45 && e5.bits == 0x42 && ml::to_float(b) == 3.140625f;
    std::printf("  fp6: e3m2 %.2f, e2m3 %.3f\n", e3m2, e2m3);
    ok &= e3m2 == 3.0 && e2m3 == 3.25;
    std::printf("  e4m3 of 1000: %.0f saturating, %s non-saturating; largest e4m3 %.0f, largest e5m2 %.0f\n", ml::to_double(big),
                std::isnan(ml::to_double(big_nan)) ? "NaN" : "?", ml::to_double(ml::minifloat_max<ml::float8_e4m3>()),
                ml::to_double(ml::minifloat_max<ml::float8_e5m2>()));
    ok &= ml::to_double(big) == 448.0 && std::isnan(ml::to_double(big_nan)) && one == 1.0f;
    std::printf("  1 + 1000 x 1e-4 in bfloat16: nearest %.4f, stochastic %.4f (exact 1.1)\n", static_cast<double>(nearest),
                static_cast<double>(stochastic));
    ok &= nearest == 1.0f && std::abs(stochastic - 1.1f) < 0.06f;

    std::printf(ok ? "minifloat demo: ok\n" : "minifloat demo: FAILED\n");
    return ok ? 0 : 1;
}
