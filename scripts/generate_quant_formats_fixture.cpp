//
//  generate_quant_formats_fixture.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Writes tests/fixtures/ml/quant_formats_fixture.txt from ggml itself (llama.cpp commit 39039aa): for every row
//  kind of tests/ml/quant_rows.hpp and every format here, a hash of the input row, of ggml's quantised bytes (the
//  reference, quantize_<type>(..., NULL), and the importance-matrix path, quantize_<type>(..., imatrix)), of the
//  dequantised floats, of the activations quantised to the format's vec_dot_type, and the bits of ggml's scalar
//  dot product ggml_vec_dot_<type>_<dot>_generic. Not part of the build: core has no dependencies.
//
//  Built against ggml-quants.c and ggml-cpu/quants.c compiled alone, scalar paths, no FMA, asserts off:
//
//    R=<llama.cpp>/ggml; F="-std=c11 -O1 -DNDEBUG -ffp-contract=off -I$R/include -I$R/src -I$R/src/ggml-cpu"
//    cc $F -c $R/src/ggml-quants.c && cc $F -c $R/src/ggml-cpu/quants.c && cc $F -c ggml_ref_stubs.c
//    c++ -std=c++20 -O1 -ffp-contract=off scripts/generate_quant_formats_fixture.cpp ggml-quants.o quants.o \
//        ggml_ref_stubs.o -o gen && ./gen > tests/fixtures/ml/quant_formats_fixture.txt
//
//  ggml_ref_stubs.c supplies what those two files take from the rest of ggml: ggml_abort, ggml_row_size and the
//  type table, and the float16 / E4M3 decode tables that ggml_cpu_init fills (ggml_ref_init here fills them the
//  same way, from GGML_COMPUTE_FP16_TO_FP32 and ggml_f8_e4m3_to_fp32).
//

#include "../tests/ml/quant_rows.hpp"

#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

extern "C" {
void ggml_ref_init(void);
std::size_t ggml_row_size(int type, std::int64_t ne);

#define GGML_QUANT(t) std::size_t quantize_##t(const float*, void*, std::int64_t, std::int64_t, const float*); \
                      void dequantize_row_##t(const void*, float*, std::int64_t);
#define GGML_REF(t) void quantize_row_##t##_ref(const float*, void*, std::int64_t);
#define GGML_DOT(t, d) void ggml_vec_dot_##t##_##d##_generic(int, float*, std::size_t, const void*, std::size_t, const void*, std::size_t, int);
GGML_QUANT(q4_0) GGML_QUANT(q4_1) GGML_QUANT(q5_0) GGML_QUANT(q5_1) GGML_QUANT(q8_0) GGML_QUANT(mxfp4) GGML_QUANT(nvfp4)
GGML_QUANT(q1_0) GGML_QUANT(q2_0) GGML_QUANT(q2_K) GGML_QUANT(q3_K) GGML_QUANT(q4_K) GGML_QUANT(q5_K) GGML_QUANT(q6_K)
GGML_REF(q8_0) GGML_REF(q8_1) GGML_REF(q8_K)
void dequantize_row_q8_K(const void*, float*, std::int64_t);
std::size_t quantize_f8_e4m3(const float*, void*, std::int64_t, std::int64_t, const float*);
void dequantize_row_f8_e4m3(const void*, float*, std::int64_t);
void ggml_vec_dot_f8_e4m3_f32(int, float*, std::size_t, const void*, std::size_t, const void*, std::size_t, int);
GGML_DOT(q4_0, q8_0) GGML_DOT(q4_1, q8_1) GGML_DOT(q5_0, q8_0) GGML_DOT(q5_1, q8_1) GGML_DOT(q8_0, q8_0) GGML_DOT(mxfp4, q8_0)
GGML_DOT(nvfp4, q8_0) GGML_DOT(q1_0, q8_0) GGML_DOT(q2_0, q8_0) GGML_DOT(q2_K, q8_K) GGML_DOT(q3_K, q8_K) GGML_DOT(q4_K, q8_K)
GGML_DOT(q5_K, q8_K) GGML_DOT(q6_K, q8_K)
}

namespace {

using quantize_fn = std::size_t (*)(const float*, void*, std::int64_t, std::int64_t, const float*);
using dequantize_fn = void (*)(const void*, float*, std::int64_t);
using act_fn = void (*)(const float*, void*, std::int64_t);
using dot_fn = void (*)(int, float*, std::size_t, const void*, std::size_t, const void*, std::size_t, int);

struct format {
    const char* name;
    int type;          // ggml_type
    int act_type;      // the vec_dot_type
    quantize_fn quantize;
    dequantize_fn dequantize;
    act_fn act;
    dot_fn dot;
};

constexpr std::size_t n = 512;

void f8_act(const float* x, void* y, std::int64_t k)
{
    std::memcpy(y, x, std::size_t(k) * sizeof(float));
}

const format formats[] = {
    {"q4_0", 2, 8, quantize_q4_0, dequantize_row_q4_0, quantize_row_q8_0_ref, ggml_vec_dot_q4_0_q8_0_generic},
    {"q4_1", 3, 9, quantize_q4_1, dequantize_row_q4_1, quantize_row_q8_1_ref, ggml_vec_dot_q4_1_q8_1_generic},
    {"q5_0", 6, 8, quantize_q5_0, dequantize_row_q5_0, quantize_row_q8_0_ref, ggml_vec_dot_q5_0_q8_0_generic},
    {"q5_1", 7, 9, quantize_q5_1, dequantize_row_q5_1, quantize_row_q8_1_ref, ggml_vec_dot_q5_1_q8_1_generic},
    {"q8_0", 8, 8, quantize_q8_0, dequantize_row_q8_0, quantize_row_q8_0_ref, ggml_vec_dot_q8_0_q8_0_generic},
    {"mxfp4", 39, 8, quantize_mxfp4, dequantize_row_mxfp4, quantize_row_q8_0_ref, ggml_vec_dot_mxfp4_q8_0_generic},
    {"nvfp4", 40, 8, quantize_nvfp4, dequantize_row_nvfp4, quantize_row_q8_0_ref, ggml_vec_dot_nvfp4_q8_0_generic},
    {"q1_0", 41, 8, quantize_q1_0, dequantize_row_q1_0, quantize_row_q8_0_ref, ggml_vec_dot_q1_0_q8_0_generic},
    {"q2_0", 42, 8, quantize_q2_0, dequantize_row_q2_0, quantize_row_q8_0_ref, ggml_vec_dot_q2_0_q8_0_generic},
    {"q2_K", 10, 15, quantize_q2_K, dequantize_row_q2_K, quantize_row_q8_K_ref, ggml_vec_dot_q2_K_q8_K_generic},
    {"q3_K", 11, 15, quantize_q3_K, dequantize_row_q3_K, quantize_row_q8_K_ref, ggml_vec_dot_q3_K_q8_K_generic},
    {"q4_K", 12, 15, quantize_q4_K, dequantize_row_q4_K, quantize_row_q8_K_ref, ggml_vec_dot_q4_K_q8_K_generic},
    {"q5_K", 13, 15, quantize_q5_K, dequantize_row_q5_K, quantize_row_q8_K_ref, ggml_vec_dot_q5_K_q8_K_generic},
    {"q6_K", 14, 15, quantize_q6_K, dequantize_row_q6_K, quantize_row_q8_K_ref, ggml_vec_dot_q6_K_q8_K_generic},
    {"f8_e4m3", 43, 0, quantize_f8_e4m3, dequantize_row_f8_e4m3, f8_act, ggml_vec_dot_f8_e4m3_f32},
};

} // namespace

int main()
{
    ggml_ref_init();
    std::printf("# ggml's bytes and scalar dot products (llama.cpp 39039aa), from scripts/generate_quant_formats_fixture.cpp\n");
    std::printf("# <format> <row kind> x=<input> w=<importance> ref=<bytes> imat=<bytes> deq=<floats> act=<activation bytes> dot=<bits>\n");
    std::printf("# hashes are FNV-1a 64 (tests/ml/quant_rows.hpp); rows %zu numbers, the activations row kind (kind + 1) %% 15\n", n);
    for (const format& f : formats) {
        for (int k = 0; k < quant_rows::kinds; ++k) {
            const auto x = quant_rows::row(k, n, 1);
            const auto y = quant_rows::row((k + 1) % 15, n, 2);
            const auto w = quant_rows::importance(n, 3);
            const std::size_t bytes = ggml_row_size(f.type, n);
            const std::size_t act_bytes = f.act_type == 0 ? n * sizeof(float) : ggml_row_size(f.act_type, n);
            std::vector<unsigned char> ref(bytes, 0), imat(bytes, 0), act(act_bytes, 0);
            std::vector<float> deq(n);
            f.quantize(x.data(), ref.data(), 1, n, nullptr);
            f.quantize(x.data(), imat.data(), 1, n, w.data());
            f.dequantize(ref.data(), deq.data(), n);
            f.act(y.data(), act.data(), n);
            float d = 0;
            f.dot(int(n), &d, 0, ref.data(), 0, act.data(), 0, 1);
            std::printf("%s %d x=%016" PRIx64 " w=%016" PRIx64 " ref=%016" PRIx64 " imat=%016" PRIx64 " deq=%016" PRIx64
                        " act=%016" PRIx64 " dot=%08" PRIx32 "\n",
                        f.name, k, quant_rows::fnv_floats(x), quant_rows::fnv(w), quant_rows::fnv(ref), quant_rows::fnv(imat),
                        quant_rows::fnv_floats(deq), quant_rows::fnv(act), quant_rows::float_bits(d));
        }
    }
    return 0;
}
