//
//  quant_ternary_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/ml/quant_ternary.hpp>: a row of ternary-looking weights stored as ggml's TQ1_0 and TQ2_0,
//  their sizes, the round trip, and the dot product with Q8_K activations against the exact one. Exits non-zero if
//  a result is not what it documents.
//

#include <snicholls/core/ml/quant_ternary.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <span>

int main()
{
    std::printf("snicholls-core quant_ternary demo\n");
    namespace ml = snicholls::ml;
    bool ok = true;

    //! [ternary]
    // 512 weights already ternary times 1/32 (as a BitNet b1.58 checkpoint holds them), and 512 activations
    std::array<float, 512> w{}, a{};
    for (std::size_t i = 0; i < 512; ++i) {
        w[i] = 0.03125f * float(int(i * 7 % 5 % 3) - 1);
        a[i] = std::sin(0.1f * float(i));
    }

    // Two layouts of the same trits: five a byte, or four
    std::array<ml::block_tq1_0, 2> t1{};
    std::array<ml::block_tq2_0, 2> t2{};
    ml::quantize(w, std::span(t1));
    ml::quantize(w, std::span(t2));

    // The activations as Q8_K, then integer dot products with one multiplication per 256 weights
    std::array<ml::block_q8_k, 2> a8{};
    ml::quantize(a, std::span(a8));
    const float d1 = ml::dot(std::span<const ml::block_tq1_0>(t1), std::span<const ml::block_q8_k>(a8));
    const float d2 = ml::dot(std::span<const ml::block_tq2_0>(t2), std::span<const ml::block_q8_k>(a8));
    //! [ternary]

    std::array<float, 512> back{};
    ml::dequantize(std::span<const ml::block_tq1_0>(t1), back);
    ok &= back == w;   // ternary times a float16-exact scale comes back exactly
    double exact = 0;
    for (std::size_t i = 0; i < 512; ++i)
        exact += double(w[i]) * double(a[i]);
    std::printf("  bytes per 256 weights: TQ1_0 %zu (%.4f bits each), TQ2_0 %zu (%.4f bits each)\n", sizeof(ml::block_tq1_0),
                ml::block_traits<ml::block_tq1_0>::bits_per_weight, sizeof(ml::block_tq2_0), ml::block_traits<ml::block_tq2_0>::bits_per_weight);
    std::printf("  w . a: exact %.6f, TQ1_0 %.6f, TQ2_0 %.6f; round trip %s\n", exact, double(d1), double(d2), back == w ? "exact" : "INEXACT");
    ok &= d1 == d2 && std::fabs(d1 - exact) < 0.01;

    std::printf(ok ? "quant_ternary demo: ok\n" : "quant_ternary demo: FAILED\n");
    return ok ? 0 : 1;
}
