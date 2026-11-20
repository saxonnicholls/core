//
//  generate_quant_ternary_fixture.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Writes tests/fixtures/ml/quant_ternary_fixture.txt from ggml itself (llama.cpp commit 39039aa): for every row
//  kind of tests/ml/quant_rows.hpp and the ternary formats TQ1_0 and TQ2_0, a hash of the input row, of ggml's
//  quantised bytes (the reference, quantize_<type>(..., NULL), and with an importance matrix, which ggml ignores
//  for these), of the dequantised floats, of the activations quantised to the format's vec_dot_type, and the bits of ggml's scalar
//  dot product ggml_vec_dot_<type>_<dot>_generic. Not part of the build: core has no dependencies.
//
//  Built against ggml-quants.c and ggml-cpu/quants.c compiled alone, scalar paths, no FMA, asserts off:
//
//    R=<llama.cpp>/ggml; F="-std=c11 -O1 -DNDEBUG -ffp-contract=off -I$R/include -I$R/src -I$R/src/ggml-cpu"
//    cc $F -c $R/src/ggml-quants.c && cc $F -c $R/src/ggml-cpu/quants.c && cc $F -c ggml_ref_stubs.c
//    c++ -std=c++20 -O1 -ffp-contract=off scripts/generate_quant_ternary_fixture.cpp ggml-quants.o quants.o \
//        ggml_ref_stubs.o -o gen && ./gen > tests/fixtures/ml/quant_ternary_fixture.txt
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
std::size_t quantize_tq1_0(const float*, void*, std::int64_t, std::int64_t, const float*);
std::size_t quantize_tq2_0(const float*, void*, std::int64_t, std::int64_t, const float*);
void dequantize_row_tq1_0(const void*, float*, std::int64_t);
void dequantize_row_tq2_0(const void*, float*, std::int64_t);
void quantize_row_q8_K_ref(const float*, void*, std::int64_t);
void ggml_vec_dot_tq1_0_q8_K_generic(int, float*, std::size_t, const void*, std::size_t, const void*, std::size_t, int);
void ggml_vec_dot_tq2_0_q8_K_generic(int, float*, std::size_t, const void*, std::size_t, const void*, std::size_t, int);
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

const format formats[] = {
    {"tq1_0", 34, 15, quantize_tq1_0, dequantize_row_tq1_0, quantize_row_q8_K_ref, ggml_vec_dot_tq1_0_q8_K_generic},
    {"tq2_0", 35, 15, quantize_tq2_0, dequantize_row_tq2_0, quantize_row_q8_K_ref, ggml_vec_dot_tq2_0_q8_K_generic},
};

} // namespace

int main()
{
    ggml_ref_init();
    std::printf("# ggml's TQ1_0 and TQ2_0 bytes and scalar dot products (llama.cpp 39039aa), from scripts/generate_quant_ternary_fixture.cpp\n");
    std::printf("# <format> <row kind> x=<input> w=<importance> ref=<bytes> imat=<bytes> deq=<floats> act=<activation bytes> dot=<bits>\n");
    std::printf("# hashes are FNV-1a 64 (tests/ml/quant_rows.hpp); rows %zu numbers, the activations row kind (kind + 1) %% 15\n", n);
    for (const format& f : formats) {
        for (int k = 0; k < quant_rows::kinds; ++k) {
            const auto x = quant_rows::row(k, n, 1);
            const auto y = quant_rows::row((k + 1) % 15, n, 2);
            const auto w = quant_rows::importance(n, 3);
            const std::size_t bytes = ggml_row_size(f.type, n);
            const std::size_t act_bytes = ggml_row_size(f.act_type, n);
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
