//
//  bitnet_kernels_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/ml/bitnet_kernels.hpp>: a ternary matrix packed as bitnet.cpp's I2_S, its multiply-add
//  dot product; the same matrix as TL1 and TL2 lookup-table codes, their GEMVs, and the tiled bytes the converter
//  writes for a model's kernel configuration. Every product equals the plain ternary one. Exits non-zero if a result
//  is not what it documents.
//

#include <snicholls/core/ml/bitnet_kernels.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <vector>

int main()
{
    std::printf("snicholls-core bitnet_kernels demo\n");
    namespace ml = snicholls::ml;
    bool ok = true;
    constexpr std::size_t rows = 64, cols = 384;

    //! [kernels]
    // A ternary matrix (absmean_quantize's codes) with its scale, and a token's activations
    std::vector<std::int8_t> t(rows * cols);
    for (std::size_t i = 0; i < t.size(); ++i)
        t[i] = static_cast<std::int8_t>(int((i * 2654435761u >> 7) % 3) - 1);
    std::vector<float> x(cols);
    for (std::size_t j = 0; j < cols; ++j)
        x[j] = std::sin(0.37f * float(j));

    // The kernels' activations: one scale for the tensor
    std::vector<std::int8_t> a(cols);
    const float s = ml::lut_activation_quantize(x, a);

    // I2_S: two bits a weight, 128 to a block on x86, the scale after the codes; one row's dot product
    std::vector<std::uint8_t> i2(ml::i2_s_bytes(rows * cols));
    ml::pack_i2_s(t, 0.02f, i2);
    const std::int32_t row0 = ml::i2_s_dot(std::span<const std::uint8_t>(i2).first(cols / 4), a);

    // TL1 (pairs) and TL2 (triples): tables of activation sums, one lookup per weight group
    std::vector<std::uint8_t> c1(rows * cols / 2);
    std::vector<ml::tl2_code_t> c2(rows * cols / 3);
    ml::tl1_encode(t, c1);
    ml::tl2_encode(t, c2);
    std::vector<std::array<std::int16_t, 16>> l1(cols / 2), l2(cols / 3);
    std::vector<std::int32_t> y1(rows), y2(rows);
    ml::tl1_gemv(c1, rows, cols, a, l1, y1);
    ml::tl2_gemv(c2, rows, cols, a, l2, y2);
    const float out0 = float(y1[0]) / s * ml::i2_s_scale(i2, rows * cols);   // the kernels' epilogue

    // The converter's tiled bytes for a kernel configuration (BM, BK, bm)
    std::vector<std::uint8_t> tl1(ml::tl1_packed_bytes(rows, cols));
    ml::tl1_pack(t, rows, cols, {32, 128, 32}, tl1);
    //! [kernels]

    bool same = row0 == y1[0];
    for (std::size_t i = 0; i < rows; ++i)
        same &= y1[i] == ml::ternary_dot(std::span<const std::int8_t>(t).subspan(i * cols, cols), a) && y2[i] == y1[i];
    std::printf("  %zu x %zu ternary: I2_S %zu bytes, TL1 tiled %zu bytes; row 0 sum %d, output %.5f\n", rows, cols, i2.size(), tl1.size(),
                int(y1[0]), double(out0));
    std::printf("  I2_S, TL1 and TL2 equal the ternary product on every row: %s\n", same ? "yes" : "NO");
    ok &= same && i2.size() == rows * cols / 4 + 32 && tl1.size() == rows * cols / 4;

    std::printf(ok ? "bitnet_kernels demo: ok\n" : "bitnet_kernels demo: FAILED\n");
    return ok ? 0 : 1;
}
