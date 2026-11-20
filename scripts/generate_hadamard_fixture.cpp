//
//  generate_hadamard_fixture.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Writes tests/fixtures/ml/hadamard_fixture.txt from ggml's own code (llama.cpp commit 39039aa): the hash of
//  ggml's CPU fast Walsh-Hadamard transform (the GGML_HINT_SRC0_IS_HADAMARD matrix product,
//  ggml_compute_forward_fwht_impl in ggml-cpu/ops.cpp) of each row kind of tests/ml/quant_rows.hpp at every
//  power-of-two length to 4096, of block-diagonal transforms of 384- and 768-wide rows, and of the rotation
//  matrix llama.cpp builds for its KV cache (ggml_gen_hadamard in src/llama-kv-cache.cpp). Not part of the build:
//  core has no dependencies. ggml_fwht.cpp is the row loop cut verbatim from ops.cpp, scalar path:
//
//    R=<llama.cpp>/ggml/src/ggml-cpu/ops.cpp
//    { echo '#include <cmath>'; echo '#include <cstdint>'; echo 'static float ggml_fwht_load(float v) { return v; }';
//      echo 'void ggml_fwht_row(const float * src_row, float * dst_row, int64_t n) {'; sed -n 12327p $R;
//      sed -n 12341,12343p $R; echo 'const int step = n;'; sed -n 12355,12364p $R; echo '}'; } > ggml_fwht.cpp
//    c++ -std=c++20 -O1 -ffp-contract=off scripts/generate_hadamard_fixture.cpp ggml_fwht.cpp -o gen
//    ./gen > tests/fixtures/ml/hadamard_fixture.txt
//
//  ggml's SIMD passes give the same bits as the scalar one: u - v is written FMA(u, v, -1), which rounds once, as
//  u - v does.
//

#include "../tests/ml/quant_rows.hpp"

#include <cinttypes>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

void ggml_fwht_row(const float* src_row, float* dst_row, std::int64_t n);

// ggml_gen_hadamard (src/llama-kv-cache.cpp), the n x n orthonormal matrix, as written there
static std::vector<float> gen_hadamard(int n)
{
    std::vector<float> data(std::size_t(n) * std::size_t(n));
    data[0] = 1.0 / sqrtf(n);
    for (int s = 1; s < n; s *= 2)
        for (int i = 0; i < s; i++)
            for (int j = 0; j < s; j++) {
                const float val = data[std::size_t(i * n + j)];
                data[std::size_t((i + s) * n + j)] = val;
                data[std::size_t(i * n + j + s)] = val;
                data[std::size_t((i + s) * n + j + s)] = -val;
            }
    return data;
}

int main()
{
    std::printf("# ggml's CPU Walsh-Hadamard transform (llama.cpp 39039aa), from scripts/generate_hadamard_fixture.cpp\n");
    std::printf("# row <kind> <n> <block> x=<input> h=<output>: each block of the row transformed; FNV-1a 64 of float bits\n");
    std::printf("# matrix <n> m=<ggml_gen_hadamard>\n");
    for (std::size_t n = 1; n <= 4096; n *= 2)
        for (int k = 0; k < quant_rows::kinds; ++k) {
            const auto x = quant_rows::row(k, n, 7);
            std::vector<float> h(n);
            ggml_fwht_row(x.data(), h.data(), std::int64_t(n));
            std::printf("row %d %zu %zu x=%016" PRIx64 " h=%016" PRIx64 "\n", k, n, n, quant_rows::fnv_floats(x), quant_rows::fnv_floats(h));
        }
    for (const std::size_t width : {std::size_t{384}, std::size_t{768}})
        for (const std::size_t block : {std::size_t{64}, std::size_t{128}})
            for (int k = 0; k < quant_rows::kinds; ++k) {
                const auto x = quant_rows::row(k, width, 8);
                std::vector<float> h(width);
                for (std::size_t b = 0; b < width; b += block)
                    ggml_fwht_row(x.data() + b, h.data() + b, std::int64_t(block));
                std::printf("row %d %zu %zu x=%016" PRIx64 " h=%016" PRIx64 "\n", k, width, block, quant_rows::fnv_floats(x),
                            quant_rows::fnv_floats(h));
            }
    for (int n = 1; n <= 512; n *= 2)
        std::printf("matrix %d m=%016" PRIx64 "\n", n, quant_rows::fnv_floats(gen_hadamard(n)));
    return 0;
}
