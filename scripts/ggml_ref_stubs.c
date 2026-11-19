// Test-only harness support: what ggml-quants.c and ggml-cpu/quants.c need from the rest of ggml.
#define GGML_COMMON_DECL_C
#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-common.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
float ggml_table_f32_f16[1 << 16];
float ggml_table_f32_f8_e4m3[1 << 8];
void ggml_abort(const char * file, int line, const char * fmt, ...) {
    fprintf(stderr, "ggml_abort %s:%d ", file, line);
    va_list a; va_start(a, fmt); vfprintf(stderr, fmt, a); va_end(a);
    fprintf(stderr, "\n"); abort();
}
static const struct { const char * name; int64_t blck; size_t size; } traits[GGML_TYPE_COUNT] = {
    [GGML_TYPE_Q1_0] = {"q1_0", QK1_0, sizeof(block_q1_0)}, [GGML_TYPE_Q2_0] = {"q2_0", QK2_0, sizeof(block_q2_0)},
    [GGML_TYPE_Q4_0] = {"q4_0", 32, sizeof(block_q4_0)}, [GGML_TYPE_Q4_1] = {"q4_1", 32, sizeof(block_q4_1)},
    [GGML_TYPE_Q5_0] = {"q5_0", 32, sizeof(block_q5_0)}, [GGML_TYPE_Q5_1] = {"q5_1", 32, sizeof(block_q5_1)},
    [GGML_TYPE_Q8_0] = {"q8_0", 32, sizeof(block_q8_0)}, [GGML_TYPE_Q8_1] = {"q8_1", 32, sizeof(block_q8_1)},
    [GGML_TYPE_Q2_K] = {"q2_K", QK_K, sizeof(block_q2_K)}, [GGML_TYPE_Q3_K] = {"q3_K", QK_K, sizeof(block_q3_K)},
    [GGML_TYPE_Q4_K] = {"q4_K", QK_K, sizeof(block_q4_K)}, [GGML_TYPE_Q5_K] = {"q5_K", QK_K, sizeof(block_q5_K)},
    [GGML_TYPE_Q6_K] = {"q6_K", QK_K, sizeof(block_q6_K)}, [GGML_TYPE_Q8_K] = {"q8_K", QK_K, sizeof(block_q8_K)},
    [GGML_TYPE_IQ2_XXS] = {"iq2_xxs", QK_K, sizeof(block_iq2_xxs)}, [GGML_TYPE_IQ2_XS] = {"iq2_xs", QK_K, sizeof(block_iq2_xs)},
    [GGML_TYPE_IQ2_S] = {"iq2_s", QK_K, sizeof(block_iq2_s)}, [GGML_TYPE_IQ3_XXS] = {"iq3_xxs", QK_K, sizeof(block_iq3_xxs)},
    [GGML_TYPE_IQ3_S] = {"iq3_s", QK_K, sizeof(block_iq3_s)}, [GGML_TYPE_IQ1_S] = {"iq1_s", QK_K, sizeof(block_iq1_s)},
    [GGML_TYPE_IQ1_M] = {"iq1_m", QK_K, sizeof(block_iq1_m)}, [GGML_TYPE_IQ4_NL] = {"iq4_nl", 32, sizeof(block_iq4_nl)},
    [GGML_TYPE_IQ4_XS] = {"iq4_xs", QK_K, sizeof(block_iq4_xs)}, [GGML_TYPE_TQ1_0] = {"tq1_0", QK_K, sizeof(block_tq1_0)},
    [GGML_TYPE_TQ2_0] = {"tq2_0", QK_K, sizeof(block_tq2_0)}, [GGML_TYPE_MXFP4] = {"mxfp4", 32, sizeof(block_mxfp4)},
    [GGML_TYPE_NVFP4] = {"nvfp4", QK_NVFP4, sizeof(block_nvfp4)}, [GGML_TYPE_F8_E4M3] = {"f8_e4m3", 1, 1},
};
const char * ggml_type_name(enum ggml_type t) { return traits[t].name; }
size_t ggml_type_size(enum ggml_type t) { return traits[t].size; }
int64_t ggml_blck_size(enum ggml_type t) { return traits[t].blck; }
size_t ggml_row_size(enum ggml_type t, int64_t ne) { return traits[t].size * (size_t)(ne / traits[t].blck); }
void ggml_ref_init(void) {
    for (int i = 0; i < (1 << 16); ++i) { ggml_fp16_t h = (ggml_fp16_t)i; ggml_table_f32_f16[i] = GGML_COMPUTE_FP16_TO_FP32(h); }
    for (int i = 0; i < 256; ++i) ggml_table_f32_f8_e4m3[i] = ggml_f8_e4m3_to_fp32((uint8_t)i);
}
