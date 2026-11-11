//
//  generate_quant_fixture.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Writes tests/fixtures/ml/quant_fixture.txt from ggml itself: for each case, two rows of floats, ggml's
//  quantize_row_q8_0_ref / q4_0_ref / mxfp4_ref bytes for the first and Q8_0 bytes for the second, and ggml's
//  scalar dot products ggml_vec_dot_{q8_0,q4_0,mxfp4}_q8_0_generic. Not part of the build: core has no
//  dependencies. Built once against a ggml checkout, its scalar paths only, no FMA:
//
//    git clone --depth 1 https://github.com/ggml-org/ggml.git && cd ggml
//    cmake -S . -B build -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF -DGGML_AVX=OFF -DGGML_AVX2=OFF -DGGML_FMA=OFF \
//          -DGGML_F16C=OFF -DGGML_METAL=OFF -DGGML_OPENMP=OFF -DCMAKE_C_FLAGS=-ffp-contract=off
//    cmake --build build --target ggml-base ggml-cpu
//    c++ -std=c++20 -O1 -ffp-contract=off <core>/scripts/generate_quant_fixture.cpp build/src/libggml-cpu.a \
//        build/src/libggml-base.a -o gen && ./gen > <core>/tests/fixtures/ml/quant_fixture.txt
//
//  The rows are fixed: an xorshift stream and hand-picked edge cases (zeros, ties at .5, subnormal and large
//  scales, a negative extreme, outliers), so the file is the same on every run.
//

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

extern "C" {
// ggml's block layouts (ggml-common.h) and the reference functions this fixture records
struct g_q8_0 { std::uint16_t d; std::int8_t qs[32]; };
struct g_q4_0 { std::uint16_t d; std::uint8_t qs[16]; };
struct g_mxfp4 { std::uint8_t e; std::uint8_t qs[16]; };
void quantize_row_q8_0_ref(const float* x, g_q8_0* y, std::int64_t k);
void quantize_row_q4_0_ref(const float* x, g_q4_0* y, std::int64_t k);
void quantize_row_mxfp4_ref(const float* x, g_mxfp4* y, std::int64_t k);
void ggml_vec_dot_q8_0_q8_0_generic(int n, float* s, std::size_t bs, const void* vx, std::size_t bx, const void* vy, std::size_t by, int nrc);
void ggml_vec_dot_q4_0_q8_0_generic(int n, float* s, std::size_t bs, const void* vx, std::size_t bx, const void* vy, std::size_t by, int nrc);
void ggml_vec_dot_mxfp4_q8_0_generic(int n, float* s, std::size_t bs, const void* vx, std::size_t bx, const void* vy, std::size_t by, int nrc);
void ggml_cpu_init(void);
}

namespace {

constexpr int n = 64;   // two blocks per row
std::uint64_t state = 0x243f6a8885a308d3ULL;

double uniform()
{
    state ^= state << 13; state ^= state >> 7; state ^= state << 17;
    return double(state >> 11) * 0x1.0p-53;
}

std::vector<float> row(int kind)
{
    std::vector<float> r(n);
    for (int i = 0; i < n; ++i) {
        const double u = 2 * uniform() - 1;
        switch (kind % 8) {
        case 0: r[i] = float(u); break;                                              // uniform in [-1, 1)
        case 1: r[i] = float(0.02 * (u + 2 * uniform() - 1 + 2 * uniform() - 1)); break;   // weight-like
        case 2: r[i] = float(u) * (i == 5 || i == 40 ? 100.0f : 1.0f); break;           // outliers
        case 3: r[i] = i < 32 ? 0.0f : float(u); break;                                // a block of zeros
        case 4: r[i] = float(u * (i < 32 ? 1e-4 : 1e-30)); break;                       // a subnormal float16 scale; one that rounds to 0
        case 5: r[i] = float(u * 6e6); break;                                           // scale near float16's top
        case 6: r[i] = float(std::round(u * 254) / 2); break;                           // halves: ties in rounding
        case 7: r[i] = i == 9 || i == 50 ? -3.0f : float(u); break;                    // a negative extreme
        }
    }
    return r;
}

template <typename B>
void bytes(const char* name, const B* b, int count)
{
    std::printf("%s", name);
    const auto* p = reinterpret_cast<const unsigned char*>(b);
    std::printf(" ");
    for (std::size_t i = 0; i < sizeof(B) * std::size_t(count); ++i)
        std::printf("%02x", p[i]);
    std::printf("\n");
}

void floats(const char* name, const std::vector<float>& r)
{
    std::printf("%s", name);
    for (const float v : r)
        std::printf(" %08x", std::bit_cast<std::uint32_t>(v));
    std::printf("\n");
}

} // namespace

int main()
{
    ggml_cpu_init();
    std::printf("# ggml's Q8_0, Q4_0 and MXFP4 bytes and scalar dot products, from scripts/generate_quant_fixture.cpp\n");
    std::printf("# case: x, y (float bits); x as q8_0, q4_0, mxfp4; y as q8_0; dot_q8, dot_q4, dot_mx (float bits)\n");
    for (int c = 0; c < 32; ++c) {
        const auto x = row(c), y = row(c / 8 + c);
        g_q8_0 xq8[n / 32], yq8[n / 32];
        g_q4_0 xq4[n / 32];
        g_mxfp4 xmx[n / 32];
        quantize_row_q8_0_ref(x.data(), xq8, n);
        quantize_row_q4_0_ref(x.data(), xq4, n);
        quantize_row_mxfp4_ref(x.data(), xmx, n);
        quantize_row_q8_0_ref(y.data(), yq8, n);
        float d8 = 0, d4 = 0, dm = 0;
        ggml_vec_dot_q8_0_q8_0_generic(n, &d8, 0, xq8, 0, yq8, 0, 1);
        ggml_vec_dot_q4_0_q8_0_generic(n, &d4, 0, xq4, 0, yq8, 0, 1);
        ggml_vec_dot_mxfp4_q8_0_generic(n, &dm, 0, xmx, 0, yq8, 0, 1);
        std::printf("case %d\n", c);
        floats("x", x);
        floats("y", y);
        bytes("q8_0", xq8, n / 32);
        bytes("q4_0", xq4, n / 32);
        bytes("mxfp4", xmx, n / 32);
        bytes("y_q8_0", yq8, n / 32);
        std::printf("dot %08x %08x %08x\n", std::bit_cast<std::uint32_t>(d8), std::bit_cast<std::uint32_t>(d4), std::bit_cast<std::uint32_t>(dm));
    }
    return 0;
}
