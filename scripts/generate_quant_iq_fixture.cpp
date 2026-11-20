//
//  generate_quant_iq_fixture.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Writes tests/fixtures/ml/quant_iq_fixture.txt from ggml itself (llama.cpp commit 39039aa): the IQ family, as
//  scripts/generate_quant_formats_fixture.cpp does for the other formats. For every row kind of
//  tests/ml/quant_rows.hpp and every IQ format, a hash of the input row, of ggml's quantised bytes
//  (quantize_<type>(..., NULL); for IQ2_XXS, IQ2_XS and IQ1_S, which ggml refuses to quantise without an importance
//  matrix, with an importance row of ones), of the importance-matrix path, of the dequantised floats, of the
//  activations (Q8_K, or Q8_0 for IQ4_NL) and the bits of ggml's scalar dot product. IQ4_NL also has
//  quantize_row_iq4_nl_ref ("rref"), which searches no scales. And the FNV-1a hash of each of ggml-common.h's
//  grids and tables, which core's tables, built at compile time, are checked against.
//
//  Each case runs in a child process, so a case where ggml aborts (a NaN reaches an assertion) is recorded as
//  "abort" instead of ending the file. Not part of the build: core has no dependencies. Built as
//  scripts/generate_quant_formats_fixture.cpp is, with ggml-common.h's tables included:
//
//    R=<llama.cpp>/ggml; F="-std=c11 -O1 -DNDEBUG -ffp-contract=off -I$R/include -I$R/src -I$R/src/ggml-cpu"
//    cc $F -c $R/src/ggml-quants.c && cc $F -c $R/src/ggml-cpu/quants.c && cc $F -c scripts/ggml_ref_stubs.c
//    c++ -std=c++20 -O1 -ffp-contract=off -I$R/src scripts/generate_quant_iq_fixture.cpp ggml-quants.o quants.o \
//        ggml_ref_stubs.o -o gen && ./gen > tests/fixtures/ml/quant_iq_fixture.txt
//

#include "../tests/ml/quant_rows.hpp"

#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

extern "C" {
void ggml_ref_init(void);
std::size_t ggml_row_size(int type, std::int64_t ne);
void iq2xs_init_impl(int type);
void iq3xs_init_impl(int grid_size);

#define GGML_QUANT(t) std::size_t quantize_##t(const float*, void*, std::int64_t, std::int64_t, const float*); \
                      void dequantize_row_##t(const void*, float*, std::int64_t);
#define GGML_DOT(t, d) void ggml_vec_dot_##t##_##d##_generic(int, float*, std::size_t, const void*, std::size_t, const void*, std::size_t, int);
GGML_QUANT(iq1_s) GGML_QUANT(iq1_m) GGML_QUANT(iq2_xxs) GGML_QUANT(iq2_xs) GGML_QUANT(iq2_s) GGML_QUANT(iq3_xxs)
GGML_QUANT(iq3_s) GGML_QUANT(iq4_nl) GGML_QUANT(iq4_xs)
void quantize_row_q8_0_ref(const float*, void*, std::int64_t);
void quantize_row_q8_K_ref(const float*, void*, std::int64_t);
void quantize_row_iq4_nl_ref(const float*, void*, std::int64_t);
GGML_DOT(iq1_s, q8_K) GGML_DOT(iq1_m, q8_K) GGML_DOT(iq2_xxs, q8_K) GGML_DOT(iq2_xs, q8_K) GGML_DOT(iq2_s, q8_K)
GGML_DOT(iq3_xxs, q8_K) GGML_DOT(iq3_s, q8_K) GGML_DOT(iq4_nl, q8_0) GGML_DOT(iq4_xs, q8_K)
}

namespace {

using quantize_fn = std::size_t (*)(const float*, void*, std::int64_t, std::int64_t, const float*);
using dequantize_fn = void (*)(const void*, float*, std::int64_t);
using act_fn = void (*)(const float*, void*, std::int64_t);
using dot_fn = void (*)(int, float*, std::size_t, const void*, std::size_t, const void*, std::size_t, int);

struct format {
    const char* name;
    int type;
    int act_type;
    bool needs_importance;   // ggml asserts an importance matrix: "ref" is quantised with ones
    quantize_fn quantize;
    dequantize_fn dequantize;
    act_fn act;
    dot_fn dot;
};

constexpr std::size_t n = 512;

const format formats[] = {
    {"iq1_s", 19, 15, true, quantize_iq1_s, dequantize_row_iq1_s, quantize_row_q8_K_ref, ggml_vec_dot_iq1_s_q8_K_generic},
    {"iq1_m", 29, 15, false, quantize_iq1_m, dequantize_row_iq1_m, quantize_row_q8_K_ref, ggml_vec_dot_iq1_m_q8_K_generic},
    {"iq2_xxs", 16, 15, true, quantize_iq2_xxs, dequantize_row_iq2_xxs, quantize_row_q8_K_ref, ggml_vec_dot_iq2_xxs_q8_K_generic},
    {"iq2_xs", 17, 15, true, quantize_iq2_xs, dequantize_row_iq2_xs, quantize_row_q8_K_ref, ggml_vec_dot_iq2_xs_q8_K_generic},
    {"iq2_s", 22, 15, false, quantize_iq2_s, dequantize_row_iq2_s, quantize_row_q8_K_ref, ggml_vec_dot_iq2_s_q8_K_generic},
    {"iq3_xxs", 18, 15, false, quantize_iq3_xxs, dequantize_row_iq3_xxs, quantize_row_q8_K_ref, ggml_vec_dot_iq3_xxs_q8_K_generic},
    {"iq3_s", 21, 15, false, quantize_iq3_s, dequantize_row_iq3_s, quantize_row_q8_K_ref, ggml_vec_dot_iq3_s_q8_K_generic},
    {"iq4_nl", 20, 8, false, quantize_iq4_nl, dequantize_row_iq4_nl, quantize_row_q8_0_ref, ggml_vec_dot_iq4_nl_q8_0_generic},
    {"iq4_xs", 23, 15, false, quantize_iq4_xs, dequantize_row_iq4_xs, quantize_row_q8_K_ref, ggml_vec_dot_iq4_xs_q8_K_generic},
};

void one_case(const format& f, int k)
{
    const auto x = quant_rows::row(k, n, 1);
    const auto y = quant_rows::row((k + 1) % 15, n, 2);
    const auto w = quant_rows::importance(n, 3);
    const std::vector<float> ones(n, 1.0f);
    const std::size_t bytes = ggml_row_size(f.type, n);
    const std::size_t act_bytes = ggml_row_size(f.act_type, n);
    std::vector<unsigned char> ref(bytes, 0), imat(bytes, 0), act(act_bytes, 0), rref(bytes, 0);
    std::vector<float> deq(n);
    f.quantize(x.data(), ref.data(), 1, n, f.needs_importance ? ones.data() : nullptr);
    f.quantize(x.data(), imat.data(), 1, n, w.data());
    f.dequantize(ref.data(), deq.data(), n);
    f.act(y.data(), act.data(), n);
    float d = 0;
    f.dot(int(n), &d, 0, ref.data(), 0, act.data(), 0, 1);
    std::printf("%s %d x=%016" PRIx64 " w=%016" PRIx64 " ref=%016" PRIx64 " imat=%016" PRIx64 " deq=%016" PRIx64
                " act=%016" PRIx64 " dot=%08" PRIx32,
                f.name, k, quant_rows::fnv_floats(x), quant_rows::fnv(w), quant_rows::fnv(ref), quant_rows::fnv(imat),
                quant_rows::fnv_floats(deq), quant_rows::fnv(act), quant_rows::float_bits(d));
    if (f.type == 20) {
        quantize_row_iq4_nl_ref(x.data(), rref.data(), n);
        std::printf(" rref=%016" PRIx64, quant_rows::fnv(rref));
    }
    std::printf("\n");
}

template <typename T, std::size_t N>
void table(const char* name, const T (&t)[N])
{
    std::printf("table %s %zu fnv=%016" PRIx64 "\n", name, N, quant_rows::fnv(t, sizeof(t)));
}

} // namespace

int main()
{
    ggml_ref_init();
    for (const int t : {16, 17, 19, 29, 22})
        iq2xs_init_impl(t);
    iq3xs_init_impl(256);
    iq3xs_init_impl(512);
    std::printf("# ggml's IQ bytes and scalar dot products (llama.cpp 39039aa), from scripts/generate_quant_iq_fixture.cpp\n");
    std::printf("# <format> <row kind> x=<input> w=<importance> ref=<bytes> imat=<bytes> deq=<floats> act=<activation bytes> dot=<bits>\n");
    std::printf("# hashes are FNV-1a 64 (tests/ml/quant_rows.hpp); rows %zu numbers, the activations row kind (kind + 1) %% 15;\n", n);
    std::printf("# ref of iq1_s, iq2_xxs, iq2_xs with an importance row of ones; \"abort\" where ggml aborted\n");
    table("iq2xxs_grid", iq2xxs_grid);
    table("iq2xs_grid", iq2xs_grid);
    table("iq2s_grid", iq2s_grid);
    table("iq3xxs_grid", iq3xxs_grid);
    table("iq3s_grid", iq3s_grid);
    table("iq1s_grid", iq1s_grid);
    table("ksigns_iq2xs", ksigns_iq2xs);
    table("kmask_iq2xs", kmask_iq2xs);
    table("kvalues_iq4nl", kvalues_iq4nl);
    std::fflush(stdout);
    for (const format& f : formats) {
        for (int k = 0; k < quant_rows::kinds; ++k) {
            const pid_t pid = fork();
            if (pid == 0) {
                std::freopen("/dev/null", "w", stderr);
                one_case(f, k);
                std::fflush(stdout);
                _exit(0);
            }
            int status = 0;
            waitpid(pid, &status, 0);
            if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
                std::printf("%s %d abort\n", f.name, k);
            std::fflush(stdout);
        }
    }
    return 0;
}
