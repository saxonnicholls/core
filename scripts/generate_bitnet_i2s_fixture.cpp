//
//  generate_bitnet_i2s_fixture.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Writes tests/fixtures/ml/bitnet_i2s_fixture.txt from BitNet's own code (microsoft/BitNet 0b341e5,
//  src/ggml-bitnet-mad.cpp): for every row kind of tests/ml/quant_rows.hpp, and for rows already ternary times a
//  scale (what BitNet's converter hands quantize_i2_s), a hash of the input, of quantize_i2_s's bytes (the packed
//  codes and the float scale after them) and the value of ggml_vec_dot_i2_i8_s_1x1 against int8 activations, in the
//  x86 configuration (ACT_PARALLEL, QK_I2_S 128, AVX2). Not part of the build: core has no dependencies.
//
//  BitNet's file needs the rest of its llama.cpp fork to compile; the two functions do not, so they are cut out:
//
//    R=<BitNet>/src/ggml-bitnet-mad.cpp
//    { printf '#include <cassert>\n#include <cmath>\n#include <cstdint>\n#include <cstdlib>\n#include <cstring>\n'
//      printf '#define ACT_PARALLEL\n#define GGML_TYPE_I2_S 0\n'
//      printf 'static size_t ggml_row_size(int, int64_t n) { return size_t(n); }\n'
//      sed -n 10,49p $R; sed -n 51,196p $R; sed -n 198,412p $R; } > bitnet_mad.cpp
//    c++ -std=c++17 -O1 -mavx2 -c bitnet_mad.cpp
//    c++ -std=c++20 -O1 -ffp-contract=off scripts/generate_bitnet_i2s_fixture.cpp bitnet_mad.o -o gen && ./gen \
//        > tests/fixtures/ml/bitnet_i2s_fixture.txt
//
//  (ggml_row_size only sizes quantize_i2_s's return value.)
//

#include "../tests/ml/quant_rows.hpp"

#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

std::size_t quantize_i2_s(const float* src, void* dst, std::int64_t nrow, std::int64_t n_per_row, const float* quant_weights);
void ggml_vec_dot_i2_i8_s_1x1(int n, float* s, std::size_t bs, const void* vx, std::size_t bx, const void* vy, std::size_t by, int nrc);

namespace {

constexpr std::size_t n = 128 * 37;   // 37 blocks: one group of 32 and a tail of 5 in the AVX2 kernel

// ternary codes times 0.37, as BitNet's weight_quant leaves a matrix; kind 16 has no zeros
std::vector<float> ternary_row(int k)
{
    quant_rows::stream s(1000 + std::uint64_t(k));
    std::vector<float> r(n);
    for (auto& v : r) {
        const int t = k == 16 ? (s.next() >> 40) % 2 == 0 ? -1 : 1 : int((s.next() >> 40) % 3) - 1;
        v = float(t) * 0.37f;
    }
    return r;
}

std::vector<std::int8_t> activations(std::uint64_t seed)
{
    quant_rows::stream s(seed);
    std::vector<std::int8_t> a(n);
    for (auto& v : a)
        v = static_cast<std::int8_t>(s.next() >> 56);
    return a;
}

} // namespace

int main()
{
    std::printf("# BitNet's I2_S bytes and ggml_vec_dot_i2_i8_s_1x1 (microsoft/BitNet 0b341e5, x86, QK_I2_S 128), from\n");
    std::printf("# scripts/generate_bitnet_i2s_fixture.cpp; rows of %zu: kinds 0-15 are tests/ml/quant_rows.hpp's, 16 and 17\n", n);
    std::printf("# ternary times 0.37 (17 without zeros); <kind> x=<input> a=<activations> bytes=<codes and scale> dot=<int32>\n");
    for (int k = 0; k < quant_rows::kinds + 2; ++k) {
        const auto x = k < quant_rows::kinds ? quant_rows::row(k, n, 4) : ternary_row(k - quant_rows::kinds + 15);
        const auto a = activations(7 + std::uint64_t(k));
        std::vector<unsigned char> out(n / 4 + 32, 0);
        quantize_i2_s(x.data(), out.data(), 1, n, nullptr);
        float d = 0;
        ggml_vec_dot_i2_i8_s_1x1(int(n), &d, 0, out.data(), 0, a.data(), 0, 1);
        out.resize(n / 4 + 4);   // the codes and the scale; the rest of the 32-byte tail is not written
        std::printf("i2_s %d x=%016" PRIx64 " a=%016" PRIx64 " bytes=%016" PRIx64 " dot=%08" PRIx32 "\n", k, quant_rows::fnv_floats(x),
                    quant_rows::fnv(a), quant_rows::fnv(out), static_cast<std::uint32_t>(static_cast<std::int32_t>(d)));
    }
    return 0;
}
