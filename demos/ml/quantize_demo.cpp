//
//  quantize_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/ml/quantize.hpp>: a row of weights quantised to Q8_0, Q4_0 and MXFP4 as llama.cpp
//  stores them, its dot product with activations quantised to Q8_0 against the exact one, the error of each format,
//  and a symmetric int8 row. Exits non-zero if a result is not what it documents.
//

#include <snicholls/core/ml/quantize.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <utility>

int main()
{
    std::printf("snicholls-core quantize demo\n");
    namespace ml = snicholls::ml;
    bool ok = true;

    //! [quantize]
    // One row of 64 weights and 64 activations: two blocks of 32 each
    std::array<float, 64> w{}, a{};
    for (std::size_t i = 0; i < 64; ++i) {
        w[i] = 0.05f * std::sin(0.7f * static_cast<float>(i));
        a[i] = std::cos(0.3f * static_cast<float>(i));
    }

    // The weights in three formats, byte for byte as a GGUF file stores them
    std::array<ml::block_q8_0, 2> w8{};
    std::array<ml::block_q4_0, 2> w4{};
    std::array<ml::block_mxfp4, 2> wm{};
    ml::quantize(w, std::span(w8));
    ml::quantize(w, std::span(w4));
    ml::quantize(w, std::span(wm));

    // The activations quantised on the fly to Q8_0, then integer dot products, one multiplication per block
    std::array<ml::block_q8_0, 2> a8{};
    ml::quantize(a, std::span(a8));
    const float d8 = ml::dot(std::span<const ml::block_q8_0>(w8), a8);
    const float d4 = ml::dot(std::span<const ml::block_q4_0>(w4), a8);
    const float dm = ml::dot(std::span<const ml::block_mxfp4>(wm), a8);
    //! [quantize]

    double exact = 0;
    for (std::size_t i = 0; i < 64; ++i)
        exact += double(w[i]) * double(a[i]);
    std::printf("  bytes per 32 weights: Q8_0 %zu, Q4_0 %zu, MXFP4 %zu (float %zu)\n", sizeof(ml::block_q8_0), sizeof(ml::block_q4_0),
                sizeof(ml::block_mxfp4), 32 * sizeof(float));
    std::printf("  w . a: exact %.6f, Q8_0 %.6f, Q4_0 %.6f, MXFP4 %.6f\n", exact, double(d8), double(d4), double(dm));
    ok &= std::fabs(d8 - exact) < 0.01 && std::fabs(d4 - exact) < 0.05 && std::fabs(dm - exact) < 0.1;

    // Each format's largest error on the weights
    const auto check = [&](const char* name, const auto& blocks, float worst) {
        std::array<float, 64> back{};
        ml::dequantize(std::span(blocks.data(), blocks.size()), back);
        float err = 0.0f;
        for (std::size_t i = 0; i < 64; ++i)
            err = std::fmax(err, std::fabs(back[i] - w[i]));
        std::printf("  %-5s largest error %.5f\n", name, double(err));
        ok &= err < worst;
    };
    check("Q8_0", std::as_const(w8), 0.002f);
    check("Q4_0", std::as_const(w4), 0.01f);
    check("MXFP4", std::as_const(wm), 0.01f);

    // One scale for a whole row: int8 per channel
    std::array<std::int8_t, 64> q{};
    const float scale = ml::quantize_int8(a, q);
    std::array<float, 64> a_back{};
    ml::dequantize_int8(q, scale, a_back);
    std::printf("  int8 row: scale %.6f, a[0] %.4f -> %d -> %.4f\n", double(scale), double(a[0]), int(q[0]), double(a_back[0]));
    ok &= q[0] == 127 && std::fabs(a_back[0] - a[0]) <= scale / 2;

    // The type table a GGUF reader needs, and the newer formats: importance weighting, NVFP4, Q1_0, Q2_0, FP8 rows
    //! [formats]
    static_assert(ml::ggml_type_traits(ml::ggml_type::q4_1).block_bytes == sizeof(ml::block_q4_1));
    static_assert(ml::ggml_row_bytes(ml::ggml_type::q5_0, 4096) == 128 * sizeof(ml::block_q5_0));
    std::array<float, 128> row{}, importance{};
    for (std::size_t i = 0; i < 128; ++i) {
        row[i] = 0.05f * std::sin(0.7f * static_cast<float>(i));
        importance[i] = i % 4 == 0 ? 10.0f : 1.0f;   // one weight per column, from an importance matrix
    }
    std::array<ml::block_q4_1, 4> r41{}, r41w{};
    std::array<ml::block_q8_1, 4> a81{};
    ml::quantize(row, std::span(r41));
    ml::quantize(row, importance, std::span(r41w));
    ml::quantize(row, std::span(a81));
    std::array<ml::block_q5_0, 4> r50{};
    std::array<ml::block_q5_1, 4> r51{};
    std::array<ml::block_nvfp4, 2> rnv{};
    std::array<ml::block_q1_0, 1> r10{};
    std::array<ml::block_q2_0, 2> r20{};
    std::array<ml::block_q8_0, 4> a80{};
    ml::quantize(row, std::span(r50));
    ml::quantize(row, std::span(r51));
    ml::quantize(row, std::span(rnv));
    ml::quantize(row, std::span(r10));
    ml::quantize(row, std::span(r20));
    ml::quantize(row, std::span(a80));
    std::array<ml::float8_e4m3, 128> r8{};
    ml::quantize_row(row, r8);
    //! [formats]

    double self = 0;
    for (const float v : row)
        self += double(v) * double(v);
    const auto report = [&](const char* name, double bits, float d) {
        std::printf("  %-8s %6.3f bits/weight  row . row %.6f (exact %.6f)\n", name, bits, double(d), self);
        ok &= std::fabs(d - self) < 0.05 * self + 1e-3;
    };
    report("Q4_1", ml::block_traits<ml::block_q4_1>::bits_per_weight, ml::dot(std::span<const ml::block_q4_1>(r41), std::span<const ml::block_q8_1>(a81)));
    report("Q4_1 imx", ml::block_traits<ml::block_q4_1>::bits_per_weight, ml::dot(std::span<const ml::block_q4_1>(r41w), std::span<const ml::block_q8_1>(a81)));
    report("Q5_0", ml::block_traits<ml::block_q5_0>::bits_per_weight, ml::dot(std::span<const ml::block_q5_0>(r50), std::span<const ml::block_q8_0>(a80)));
    report("Q5_1", ml::block_traits<ml::block_q5_1>::bits_per_weight, ml::dot(std::span<const ml::block_q5_1>(r51), std::span<const ml::block_q8_1>(a81)));
    report("NVFP4", ml::block_traits<ml::block_nvfp4>::bits_per_weight, ml::dot(std::span<const ml::block_nvfp4>(rnv), std::span<const ml::block_q8_0>(a80)));
    report("FP8", 8.0, ml::dot(std::span<const ml::float8_e4m3>(r8), std::span<const float>(row)));
    // One and two bits: the sign or {-1, 0, 1, 2} times one scale; a coarse match, so only the sign is checked
    const float d10 = ml::dot(std::span<const ml::block_q1_0>(r10), std::span<const ml::block_q8_0>(a80));
    const float d20 = ml::dot(std::span<const ml::block_q2_0>(r20), std::span<const ml::block_q8_0>(a80));
    std::printf("  Q1_0 row . row %.4f, Q2_0 %.4f (exact %.4f)\n", double(d10), double(d20), self);
    ok &= d10 > 0 && d20 > 0;
    std::array<float, 128> back{};
    ml::dequantize(std::span<const ml::block_q8_1>(a81), back);
    ml::dequantize_row(r8, back);
    ok &= std::fabs(back[1] - row[1]) <= std::fabs(row[1]) / 16;   // E4M3: three mantissa bits, within half a step

    std::printf(ok ? "quantize demo: ok\n" : "quantize demo: FAILED\n");
    return ok ? 0 : 1;
}
