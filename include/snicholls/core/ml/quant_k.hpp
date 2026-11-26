//
//  quant_k.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::ml` K-quants: ggml's super-block formats Q2_K, Q3_K, Q4_K, Q5_K and Q6_K, byte for byte, with their reference and importance-weighted quantisation, dequantisation and the dot products with Q8_K, all `constexpr`.
/// \ingroup core
///
/// **Overview.** A K-quant block holds 256 numbers in sub-blocks of 16 or 32, each sub-block with its own scale
/// (and minimum), the sub-block scales themselves quantised against one or two `float16` super-block scales. These
/// are llama.cpp's "Q4_K_M"-style formats, so a GGUF tensor of one of them is an array of these structs, read in
/// place:
///
/// | Block | Sub-blocks | Weight | Sub-block scales | Bytes | Bits per weight |
/// |---|---|---|---|---|---|
/// | `block_q2_k` | 16 of 16 | 2-bit, `x ≈ d·s·q - dmin·m` | 4-bit scale and min, in `scales` | 84 | 2.625 |
/// | `block_q3_k` | 16 of 16 | 3-bit (2 in `qs`, the third in `hmask`), `x ≈ d·(s - 32)·(q - 4)` | 6-bit, packed in 12 bytes | 110 | 3.4375 |
/// | `block_q4_k` | 8 of 32 | 4-bit, `x ≈ d·s·q - dmin·m` | 6-bit scale and min, packed in 12 bytes | 144 | 4.5 |
/// | `block_q5_k` | 8 of 32 | 5-bit (the fifth in `qh`) | as Q4_K | 176 | 5.5 |
/// | `block_q6_k` | 16 of 16 | 6-bit (4 in `ql`, 2 in `qh`), `x ≈ d·s·(q - 32)` | `int8` | 210 | 6.5625 |
///
/// The activations are `block_q8_k` (`ml/quantize.hpp`). The functions follow `ml/quantize.hpp`'s contract:
/// `quantize_row(x, importance, blocks)` is ggml's `quantize_q<n>_K` for one row (an empty `importance` is the
/// reference `quantize_row_q<n>_K_ref`; a non-empty one the importance-matrix path, `quantize_row_q<n>_K_impl`),
/// `dequantize_row` is `dequantize_row_q<n>_K`, and `dot` is `ggml_vec_dot_q<n>_K_q8_K_generic`, with its
/// summation order (eight lanes of integer sums for Q3_K to Q6_K), so the result is ggml's to the bit.
///
/// \snippet ml/quant_k_demo.cpp quant_k
///
/// **Relationships.** `ml/quantize.hpp` (the type ids, `block_traits`, Q8_K, and the scale searches
/// `make_qx_quants`, `make_qkx_quants`, `make_qp_quants` this header shares with Q4_0 to Q5_1 and the IQ formats);
/// `ml/quant_iq.hpp`, `ml/quant_ternary.hpp`, the other super-block formats. Group: \ref core.
///
/// **Implementations.** One per block. The quantisers search scales as ggml's do (`make_qkx2_quants` for the
/// reference Q2_K, Q4_K and Q5_K, `make_q3_quants` for Q3_K, `make_qx_quants` for Q6_K and the weighted Q3_K;
/// `make_qkx3_quants` and `make_qp_quants` for the weighted Q2_K, Q4_K and Q5_K), so they cost tens of operations
/// per weight; dequantisation and the dot products are the scalar reference a SIMD or GPU kernel reorders.
///
/// **Guarantees.** `constexpr`, no allocation, linear in the row. The blocks are trivially copyable, have ggml's
/// sizes and no padding (`static_assert`ed against `ggml_type_traits`). Bytes, dequantised numbers, Q8_K activations
/// and dot products equal ggml's on every row of the committed fixture (`tests/fixtures/ml/quant_formats_fixture.txt`,
/// from `scripts/generate_quant_formats_fixture.cpp` built against ggml, llama.cpp 39039aa), with and without an
/// importance row. Each multiplication and addition is its own statement, so no compiler fuses them.
///
/// **Errors.** Preconditions, asserted: the row has `256 · blocks.size()` numbers; `importance` is empty or as long
/// as the row; the rows of a dot product have the same number of blocks. NaN and infinity are not representable;
/// the result is ggml's (deterministic, not meaningful).
///
/// **Performance.** Against ggml's scalar loops written out by hand (`benchmarks/ml/quant_k_benchmark.cpp`), rows of
/// 4,096, medians of five: Q6_K dot 2.1 against 2.2 billion weights per second; Q4_K dot 2.2 against 2.8 (core
/// unpacks the nibbles in element order, ggml in two passes); Q4_K dequantisation 3.9 against 1.4 (Xeon W-3245,
/// Release, Apple clang 17, 2026-11-20, one thread, scalar code). Quantisation is a conversion step, tens of
/// operations per weight (the scale searches), not benchmarked.
///
/// **Tests and demos.** `tests/ml/quant_k_test.cpp`, `demos/ml/quant_k_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<cassert>`, `<cstddef>`, `<cstdint>`, `<span>`, `ml/quantize.hpp`.
///
/// **Known issues.** As `ml/quantize.hpp`: build with `-ffp-contract=off` on GCC for ggml's last bit, and a float
/// converted out of an integer's range gives x86's result, where the fixture was made. Whole-block quantisation is
/// thousands of operations per block, so compile-time use is for a block or two within the compiler's limits.
///
/// **Status.** Experimental, new in 0.17.0.

#ifndef snicholls_core_ml_quant_k_hpp
#define snicholls_core_ml_quant_k_hpp

#include <snicholls/core/ml/quantize.hpp>

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <span>

namespace snicholls::ml
{

/// \brief The numbers in a K-quant super-block.
inline constexpr std::size_t qk_k = 256;

/// \brief ggml's Q2_K: 16 sub-blocks of 16 two-bit numbers, each with a 4-bit scale and minimum.
/// \ingroup core
struct block_q2_k
{
    static constexpr ggml_type type = ggml_type::q2_k;   ///< The ggml id.
    static constexpr std::size_t elements = qk_k;        ///< Numbers per block.
    /// \brief ggml's `vec_dot_type`.
    using dot_block = block_q8_k;

    std::array<std::uint8_t, 16> scales;   ///< Sub-block `j`: scale in the low nibble, minimum in the high.
    std::array<std::uint8_t, 64> qs;       ///< Element `128 h + 32 g + l` in bits `2g` of byte `32 h + l`.
    float16 d;                             ///< The super-block scale of the scales.
    float16 dmin;                          ///< The super-block scale of the minimums.
};

/// \brief ggml's Q3_K: 16 sub-blocks of 16 three-bit numbers, each with a 6-bit signed scale.
/// \ingroup core
struct block_q3_k
{
    static constexpr ggml_type type = ggml_type::q3_k;   ///< The ggml id.
    static constexpr std::size_t elements = qk_k;        ///< Numbers per block.
    /// \brief ggml's `vec_dot_type`.
    using dot_block = block_q8_k;

    std::array<std::uint8_t, 32> hmask;    ///< The third bit, set when the code is at least 4: element `e` in bit `e / 32` of byte `e % 32`.
    std::array<std::uint8_t, 64> qs;       ///< The low two bits, packed as Q2_K's.
    std::array<std::uint8_t, 12> scales;   ///< The 6-bit scales plus 32: low nibbles in bytes 0 to 7, high bits in 8 to 11.
    float16 d;                             ///< The super-block scale.
};

/// \brief ggml's Q4_K: 8 sub-blocks of 32 four-bit numbers, each with a 6-bit scale and minimum.
/// \ingroup core
struct block_q4_k
{
    static constexpr ggml_type type = ggml_type::q4_k;   ///< The ggml id.
    static constexpr std::size_t elements = qk_k;        ///< Numbers per block.
    /// \brief ggml's `vec_dot_type`.
    using dot_block = block_q8_k;

    float16 d;                             ///< The super-block scale of the scales.
    float16 dmin;                          ///< The super-block scale of the minimums.
    std::array<std::uint8_t, 12> scales;   ///< The 6-bit scales and minimums, as ggml's `get_scale_min_k4` reads them.
    std::array<std::uint8_t, 128> qs;      ///< Element `64 h + l` in the low nibble of byte `32 h + l`, `64 h + 32 + l` in the high.
};

/// \brief ggml's Q5_K: Q4_K with a fifth bit per number.
/// \ingroup core
struct block_q5_k
{
    static constexpr ggml_type type = ggml_type::q5_k;   ///< The ggml id.
    static constexpr std::size_t elements = qk_k;        ///< Numbers per block.
    /// \brief ggml's `vec_dot_type`.
    using dot_block = block_q8_k;

    float16 d;                             ///< The super-block scale of the scales.
    float16 dmin;                          ///< The super-block scale of the minimums.
    std::array<std::uint8_t, 12> scales;   ///< As Q4_K's.
    std::array<std::uint8_t, 32> qh;       ///< The fifth bits: element `e` in bit `e / 32` of byte `e % 32`.
    std::array<std::uint8_t, 128> qs;      ///< The low four bits, packed as Q4_K's.
};

/// \brief ggml's Q6_K: 16 sub-blocks of 16 six-bit numbers, each with an `int8` scale.
/// \ingroup core
struct block_q6_k
{
    static constexpr ggml_type type = ggml_type::q6_k;   ///< The ggml id.
    static constexpr std::size_t elements = qk_k;        ///< Numbers per block.
    /// \brief ggml's `vec_dot_type`.
    using dot_block = block_q8_k;

    std::array<std::uint8_t, 128> ql;      ///< The low four bits.
    std::array<std::uint8_t, 64> qh;       ///< The high two bits.
    std::array<std::int8_t, 16> scales;    ///< The sub-block scales.
    float16 d;                             ///< The super-block scale.
};

static_assert(quantized_block<block_q2_k> && quantized_block<block_q3_k> && quantized_block<block_q4_k> && quantized_block<block_q5_k>
                  && quantized_block<block_q6_k>,
              "snicholls::ml: the K-quant blocks must have ggml's sizes");

namespace detail::quant_k
{
    using namespace detail::quantize;

    // ggml's make_q3_quants: a symmetric scale for n numbers, codes in [0, 2 nmax), refined for the x^2-weighted
    // error when do_rmse
    constexpr float make_q3_quants(int n, int nmax, const float* x, std::int8_t* L, bool do_rmse)
    {
        float max = 0, amax = 0;
        for (int i = 0; i < n; ++i) {
            const float ax = abs(x[i]);
            if (ax > amax) {
                amax = ax;
                max = x[i];
            }
        }
        if (amax < group_max_eps) {
            for (int i = 0; i < n; ++i)
                L[i] = 0;
            return 0.0f;
        }
        const float iscale = float(-nmax) / max;
        if (do_rmse) {
            float sumlx = 0, suml2 = 0;
            for (int i = 0; i < n; ++i) {
                const float v = iscale * x[i];
                const int l = imax(-nmax, imin(nmax - 1, nearest_int(v)));
                L[i] = static_cast<std::int8_t>(l);
                const float w = x[i] * x[i];
                const float wx = w * x[i];
                const float wxl = wx * float(l);
                sumlx += wxl;
                const float wl = w * float(l);
                const float wll = wl * float(l);
                suml2 += wll;
            }
            for (int itry = 0; itry < 5; ++itry) {
                int n_changed = 0;
                for (int i = 0; i < n; ++i) {
                    const float w = x[i] * x[i];
                    const float wx = w * x[i];
                    const float wxl = wx * float(L[i]);
                    float slx = sumlx - wxl;
                    if (slx > 0) {
                        const float wl = w * float(L[i]);
                        const float wll = wl * float(L[i]);
                        float sl2 = suml2 - wll;
                        const float xs = x[i] * sl2;
                        const float v = xs / slx;
                        const int new_l = imax(-nmax, imin(nmax - 1, nearest_int(v)));
                        if (new_l != L[i]) {
                            const float wxn = wx * float(new_l);
                            slx += wxn;
                            const float wn = w * float(new_l);
                            const float wnn = wn * float(new_l);
                            sl2 += wnn;
                            const float s1 = slx * slx;
                            const float lhs = s1 * suml2;
                            const float s2 = sumlx * sumlx;
                            const float rhs = s2 * sl2;
                            if (sl2 > 0 && lhs > rhs) {
                                L[i] = static_cast<std::int8_t>(new_l);
                                sumlx = slx;
                                suml2 = sl2;
                                ++n_changed;
                            }
                        }
                    }
                }
                if (n_changed == 0)
                    break;
            }
            for (int i = 0; i < n; ++i)
                L[i] = static_cast<std::int8_t>(L[i] + nmax);
            return suml2 > 0.0f ? sumlx / suml2 : 0.0f;
        }
        for (int i = 0; i < n; ++i) {
            const float v = iscale * x[i];
            L[i] = static_cast<std::int8_t>(imax(-nmax, imin(nmax - 1, nearest_int(v))) + nmax);
        }
        return 1 / iscale;
    }

    // ggml's get_scale_min_k4: the 6-bit scale and minimum of sub-block j of Q4_K and Q5_K
    constexpr void scale_min_k4(int j, const std::array<std::uint8_t, 12>& q, std::uint8_t& d, std::uint8_t& m)
    {
        if (j < 4) {
            d = static_cast<std::uint8_t>(q[j] & 63);
            m = static_cast<std::uint8_t>(q[j + 4] & 63);
        } else {
            d = static_cast<std::uint8_t>((q[j + 4] & 0xf) | ((q[j - 4] >> 6) << 4));
            m = static_cast<std::uint8_t>((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
        }
    }

    // Q4_K's and Q5_K's packing of eight 6-bit scales and minimums into 12 bytes
    constexpr void pack_scale_min_k4(const std::uint8_t* ls, const std::uint8_t* lm, std::array<std::uint8_t, 12>& s)
    {
        s = {};
        for (int j = 0; j < 8; ++j) {
            if (j < 4) {
                s[j] = ls[j];
                s[j + 4] = lm[j];
            } else {
                s[j + 4] = static_cast<std::uint8_t>((ls[j] & 0xf) | ((lm[j] & 0xf) << 4));
                s[j - 4] = static_cast<std::uint8_t>(s[j - 4] | ((ls[j] >> 4) << 6));
                s[j] = static_cast<std::uint8_t>(s[j] | ((lm[j] >> 4) << 6));
            }
        }
    }

    // Q3_K's 6-bit scale of sub-block j, minus 32
    constexpr int q3_scale(const std::array<std::uint8_t, 12>& s, int j)
    {
        const int low = j < 8 ? s[j] & 0xf : s[j - 8] >> 4;
        const int high = (s[8 + j % 4] >> (2 * (j / 4))) & 3;
        return static_cast<std::int8_t>(low | (high << 4)) - 32;
    }

    // Q3_K's packing of sixteen 6-bit scales (0 to 63)
    constexpr void pack_q3_scales(const int* l6, std::array<std::uint8_t, 12>& s)
    {
        s = {};
        for (int j = 0; j < 16; ++j) {
            int l = l6[j];
            if (j < 8)
                s[j] = static_cast<std::uint8_t>(l & 0xf);
            else
                s[j - 8] = static_cast<std::uint8_t>(s[j - 8] | ((l & 0xf) << 4));
            l >>= 4;
            s[j % 4 + 8] = static_cast<std::uint8_t>(s[j % 4 + 8] | (l << (2 * (j / 4))));
        }
    }

    // The 2-bit packing of Q2_K and Q3_K: element 128 h + 32 g + l in bits 2g of byte 32 h + l
    template <typename L>
    constexpr void pack2(const L* codes, std::array<std::uint8_t, 64>& qs)
    {
        for (int j = 0; j < 256; j += 128)
            for (int l = 0; l < 32; ++l)
                qs[j / 4 + l] = static_cast<std::uint8_t>(codes[j + l] | (codes[j + l + 32] << 2) | (codes[j + l + 64] << 4) | (codes[j + l + 96] << 6));
    }

    // A block's codes, element by element, unpacked the way ggml's scalar dot products unpack them
    constexpr void unpack(const block_q2_k& b, std::array<std::int8_t, 256>& a)
    {
        for (std::size_t h = 0; h < 2; ++h)
            for (std::size_t g = 0; g < 4; ++g)
                for (std::size_t l = 0; l < 32; ++l)
                    a[128 * h + 32 * g + l] = static_cast<std::int8_t>((b.qs[32 * h + l] >> (2 * g)) & 3);
    }

    constexpr void unpack(const block_q3_k& b, std::array<std::int8_t, 256>& a)
    {
        for (std::size_t h = 0; h < 2; ++h)
            for (std::size_t g = 0; g < 4; ++g) {
                const auto m = static_cast<std::uint8_t>(1u << (4 * h + g));
                for (std::size_t l = 0; l < 32; ++l)
                    a[128 * h + 32 * g + l] = static_cast<std::int8_t>(((b.qs[32 * h + l] >> (2 * g)) & 3) - (b.hmask[l] & m ? 0 : 4));
            }
    }

    constexpr void unpack(const block_q4_k& b, std::array<std::int8_t, 256>& a)
    {
        for (std::size_t h = 0; h < 4; ++h)
            for (std::size_t l = 0; l < 32; ++l) {
                a[64 * h + l] = static_cast<std::int8_t>(b.qs[32 * h + l] & 0xf);
                a[64 * h + 32 + l] = static_cast<std::int8_t>(b.qs[32 * h + l] >> 4);
            }
    }

    constexpr void unpack(const block_q5_k& b, std::array<std::int8_t, 256>& a)
    {
        for (std::size_t h = 0; h < 4; ++h) {
            const auto m1 = static_cast<std::uint8_t>(1u << (2 * h)), m2 = static_cast<std::uint8_t>(2u << (2 * h));
            for (std::size_t l = 0; l < 32; ++l) {
                a[64 * h + l] = static_cast<std::int8_t>((b.qs[32 * h + l] & 0xf) + (b.qh[l] & m1 ? 16 : 0));
                a[64 * h + 32 + l] = static_cast<std::int8_t>((b.qs[32 * h + l] >> 4) + (b.qh[l] & m2 ? 16 : 0));
            }
        }
    }

    constexpr void unpack(const block_q6_k& b, std::array<std::int8_t, 256>& a)
    {
        for (std::size_t n = 0; n < 2; ++n)
            for (std::size_t l = 0; l < 32; ++l) {
                const std::uint8_t lo = b.ql[64 * n + l], hi = b.ql[64 * n + 32 + l], h = b.qh[32 * n + l];
                a[128 * n + l] = static_cast<std::int8_t>(((lo & 0xf) | ((h & 3) << 4)) - 32);
                a[128 * n + 32 + l] = static_cast<std::int8_t>(((hi & 0xf) | (((h >> 2) & 3) << 4)) - 32);
                a[128 * n + 64 + l] = static_cast<std::int8_t>(((lo >> 4) | (((h >> 4) & 3) << 4)) - 32);
                a[128 * n + 96 + l] = static_cast<std::int8_t>(((hi >> 4) | (((h >> 6) & 3) << 4)) - 32);
            }
    }

    // The sum over a block's 256 products scale[e / 16] · q8[e] · a[e] in ggml's eight lanes (lane e % 8), then
    // sums[lane] += d · lane: the order of ggml_vec_dot_q{3,4,5,6}_K_q8_K_generic (integer sums, so exact)
    constexpr void lanes(const block_q8_k& y, float d, const std::array<std::int8_t, 256>& a, const std::array<int, 16>& scale, std::array<float, 8>& sums)
    {
        std::array<int, 8> aux32{};
        for (std::size_t j = 0; j < 32; ++j) {
            const int s = scale[j / 2];
            for (std::size_t l = 0; l < 8; ++l)
                aux32[l] += s * (int(y.qs[8 * j + l]) * int(a[8 * j + l]));
        }
        for (std::size_t l = 0; l < 8; ++l) {
            const float term = d * float(aux32[l]);
            sums[l] += term;
        }
    }

    constexpr float sum_lanes(float sumf, const std::array<float, 8>& sums)
    {
        for (const float s : sums)
            sumf += s;
        return sumf;
    }

    // Q4_K's and Q5_K's reference: an affine fit per 32 (make_qkx2_quants), weighted av_x + |x|
    template <int Nmax>
    constexpr void reference_k4(const float* x, std::array<std::uint8_t, 256>& L, float16& d16, float16& dmin16, std::array<std::uint8_t, 12>& packed,
                                float rmin, float rdelta, int nstep)
    {
        std::array<float, 8> scales{}, mins{};
        std::array<float, 32> weights{};
        std::array<std::uint8_t, 32> Laux{};
        float largest_scale = 0, max_min = 0;
        for (int j = 0; j < 8; ++j) {
            float sum_x2 = 0;
            for (int l = 0; l < 32; ++l) {
                const float v2 = x[32 * j + l] * x[32 * j + l];
                sum_x2 += v2;
            }
            const float av_x = sqrt(sum_x2 / 32);
            for (int l = 0; l < 32; ++l)
                weights[std::size_t(l)] = av_x + abs(x[32 * j + l]);
            scales[std::size_t(j)] = make_qkx_quants<false>(32, Nmax, x + 32 * j, weights.data(), L.data() + 32 * j, &mins[std::size_t(j)], Laux.data(), rmin, rdelta, nstep, false);
            if (scales[std::size_t(j)] > largest_scale)
                largest_scale = scales[std::size_t(j)];
            if (mins[std::size_t(j)] > max_min)
                max_min = mins[std::size_t(j)];
        }
        const float inv_scale = largest_scale > 0 ? 63.0f / largest_scale : 0.0f;
        const float inv_min = max_min > 0 ? 63.0f / max_min : 0.0f;
        std::array<std::uint8_t, 8> ls{}, lm{};
        for (std::size_t j = 0; j < 8; ++j) {
            const float vs = inv_scale * scales[j];
            const float vm = inv_min * mins[j];
            const auto a = static_cast<std::uint8_t>(nearest_int(vs));
            const auto b = static_cast<std::uint8_t>(nearest_int(vm));
            ls[j] = a < 63 ? a : std::uint8_t{63};
            lm[j] = b < 63 ? b : std::uint8_t{63};
        }
        pack_scale_min_k4(ls.data(), lm.data(), packed);
        d16 = fp16(largest_scale / 63.0f);
        dmin16 = fp16(max_min / 63.0f);
    }

    // Q4_K's and Q5_K's importance-weighted path: make_qkx3_quants per 32, make_qp_quants for the scales
    template <int Nmax>
    constexpr void weighted_k4(const float* x, const float* qw, std::array<std::uint8_t, 256>& L, float16& d16, float16& dmin16,
                               std::array<std::uint8_t, 12>& packed, bool clamp)
    {
        std::array<float, 8> scales{}, mins{}, sw{};
        std::array<float, 32> weights{};
        std::array<std::uint8_t, 32> Laux{};
        std::array<std::uint8_t, 8> Ls{}, Lm{};
        float sum_x2 = 0;
        for (int l = 0; l < 256; ++l) {
            const float v2 = x[l] * x[l];
            sum_x2 += v2;
        }
        const float sigma2 = 2 * sum_x2 / 256;
        for (int j = 0; j < 8; ++j) {
            for (int l = 0; l < 32; ++l)
                weights[std::size_t(l)] = importance_weight(qw[32 * j + l], sigma2, x[32 * j + l]);
            float sumw = 0;
            for (const float w : weights)
                sumw += w;
            sw[std::size_t(j)] = sumw;
            scales[std::size_t(j)] = make_qkx_quants<true>(32, Nmax, x + 32 * j, weights.data(), L.data() + 32 * j, &mins[std::size_t(j)], Laux.data(), -0.9f, 0.05f, 36, false);
        }
        const float d_block = make_qp_quants(8, 63, scales.data(), Ls.data(), sw.data());
        const float m_block = make_qp_quants(8, 63, mins.data(), Lm.data(), sw.data());
        if (clamp)
            for (std::size_t j = 0; j < 8; ++j) {
                Ls[j] = Ls[j] < 63 ? Ls[j] : std::uint8_t{63};
                Lm[j] = Lm[j] < 63 ? Lm[j] : std::uint8_t{63};
            }
        pack_scale_min_k4(Ls.data(), Lm.data(), packed);
        d16 = fp16(d_block);
        dmin16 = fp16(m_block);
    }

    // The codes of Q4_K and Q5_K from the packed scales: round((x + dmin·m) / (d·s)) clamped to [0, top]
    constexpr void requantize_k4(const float* x, const float16& d16, const float16& dmin16, const std::array<std::uint8_t, 12>& packed, int top,
                                 std::array<std::uint8_t, 256>& L)
    {
        for (int j = 0; j < 8; ++j) {
            std::uint8_t sc = 0, m = 0;
            scale_min_k4(j, packed, sc, m);
            const float d = f32(d16) * float(sc);
            if (d == 0.0f)
                continue;
            const float dm = f32(dmin16) * float(m);
            for (int ii = 0; ii < 32; ++ii) {
                const float v = (x[32 * j + ii] + dm) / d;
                L[std::size_t(32 * j + ii)] = static_cast<std::uint8_t>(imax(0, imin(top, nearest_int(v))));
            }
        }
    }
} // namespace detail::quant_k

/// \brief Quantises one row to Q2_K, as ggml's `quantize_q2_K`.
/// \param x The row, `256 · y.size()` floats.
/// \param importance Empty (the reference: `make_qkx2_quants` per 16, weighted |x|), or one weight per number.
/// \param y Receives the blocks.
constexpr void quantize_row(std::span<const float> x, std::span<const float> importance, std::span<block_q2_k> y)
{
    using namespace detail::quant_k;
    check(256, y.size(), x.size(), importance.size());
    for (std::size_t i = 0; i < y.size(); ++i) {
        const float* xb = x.data() + 256 * i;
        std::array<std::uint8_t, 256> L{};
        std::array<std::uint8_t, 16> Laux{};
        std::array<float, 16> weight{}, mins{}, scales{};
        block_q2_k& b = y[i];
        if (importance.empty()) {
            float largest_scale = 0, max_min = 0;
            for (int j = 0; j < 16; ++j) {
                for (int l = 0; l < 16; ++l)
                    weight[std::size_t(l)] = abs(xb[16 * j + l]);
                scales[std::size_t(j)] = make_qkx_quants<false>(16, 3, xb + 16 * j, weight.data(), L.data() + 16 * j, &mins[std::size_t(j)], Laux.data(), -0.5f, 0.1f, 15, true);
                if (scales[std::size_t(j)] > largest_scale)
                    largest_scale = scales[std::size_t(j)];
                if (mins[std::size_t(j)] > max_min)
                    max_min = mins[std::size_t(j)];
            }
            b.scales = {};
            if (largest_scale > 0) {
                const float iscale = 15.0f / largest_scale;
                for (std::size_t j = 0; j < 16; ++j) {
                    const float v = iscale * scales[j];
                    b.scales[j] = static_cast<std::uint8_t>(nearest_int(v));
                }
                b.d = fp16(largest_scale / 15.0f);
            } else {
                b.d = fp16(0.0f);
            }
            if (max_min > 0) {
                const float iscale = 15.0f / max_min;
                for (std::size_t j = 0; j < 16; ++j) {
                    const float v = iscale * mins[j];
                    b.scales[j] = static_cast<std::uint8_t>(b.scales[j] | (nearest_int(v) << 4));
                }
                b.dmin = fp16(max_min / 15.0f);
            } else {
                b.dmin = fp16(0.0f);
            }
        } else {
            const float* qw = importance.data() + 256 * i;
            std::array<float, 16> sw{};
            std::array<std::uint8_t, 16> Ls{}, Lm{};
            float sumx2 = 0;
            for (int j = 0; j < 256; ++j) {
                const float v2 = xb[j] * xb[j];
                sumx2 += v2;
            }
            const float sigma2 = sumx2 / 256;
            for (int j = 0; j < 16; ++j) {
                for (int l = 0; l < 16; ++l)
                    weight[std::size_t(l)] = importance_weight(qw[16 * j + l], sigma2, xb[16 * j + l]);
                for (const float w : weight)
                    sw[std::size_t(j)] += w;
                scales[std::size_t(j)] = make_qkx_quants<true>(16, 3, xb + 16 * j, weight.data(), L.data() + 16 * j, &mins[std::size_t(j)], Laux.data(), -0.9f, 0.05f, 36, false);
            }
            b.d = fp16(make_qp_quants(16, 15, scales.data(), Ls.data(), sw.data()));
            b.dmin = fp16(make_qp_quants(16, 15, mins.data(), Lm.data(), sw.data()));
            for (std::size_t j = 0; j < 16; ++j)
                b.scales[j] = static_cast<std::uint8_t>(Ls[j] | (Lm[j] << 4));
        }
        for (int j = 0; j < 16; ++j) {
            const float d = f32(b.d) * float(b.scales[std::size_t(j)] & 0xf);
            if (d == 0.0f)
                continue;
            const float dm = f32(b.dmin) * float(b.scales[std::size_t(j)] >> 4);
            for (int ii = 0; ii < 16; ++ii) {
                const float v = (xb[16 * j + ii] + dm) / d;
                L[std::size_t(16 * j + ii)] = static_cast<std::uint8_t>(imax(0, imin(3, nearest_int(v))));
            }
        }
        pack2(L.data(), b.qs);
    }
}

/// \brief Quantises one row to Q3_K, as ggml's `quantize_q3_K`.
/// \param x The row, `256 · y.size()` floats.
/// \param importance Empty (the reference: `make_q3_quants` per 16), or one weight per number.
/// \param y Receives the blocks.
constexpr void quantize_row(std::span<const float> x, std::span<const float> importance, std::span<block_q3_k> y)
{
    using namespace detail::quant_k;
    check(256, y.size(), x.size(), importance.size());
    for (std::size_t i = 0; i < y.size(); ++i) {
        const float* xb = x.data() + 256 * i;
        std::array<std::int8_t, 256> L{};
        std::array<float, 16> scales{};
        std::array<int, 16> l6{};
        block_q3_k& b = y[i];
        if (importance.empty()) {
            float largest_scale = 0, amax = 0;
            for (int j = 0; j < 16; ++j) {
                scales[std::size_t(j)] = make_q3_quants(16, 4, xb + 16 * j, L.data() + 16 * j, true);
                const float scale = abs(scales[std::size_t(j)]);
                if (scale > amax) {
                    amax = scale;
                    largest_scale = scales[std::size_t(j)];
                }
            }
            if (largest_scale != 0.0f) {
                const float iscale = -32.0f / largest_scale;
                for (std::size_t j = 0; j < 16; ++j) {
                    const float v = iscale * scales[j];
                    const int l = static_cast<std::int8_t>(nearest_int(v));
                    l6[j] = imax(-32, imin(31, l)) + 32;
                }
                pack_q3_scales(l6.data(), b.scales);
                b.d = fp16(1 / iscale);
            } else {
                b.scales = {};
                b.d = fp16(0.0f);
            }
        } else {
            const float* qw = importance.data() + 256 * i;
            std::array<float, 16> weight{}, sw{};
            std::array<std::int8_t, 16> Ls{};
            float sumx2 = 0;
            for (int j = 0; j < 256; ++j) {
                const float v2 = xb[j] * xb[j];
                sumx2 += v2;
            }
            const float sigma2 = 2 * sumx2 / 256;
            for (int j = 0; j < 16; ++j) {
                for (int l = 0; l < 16; ++l)
                    weight[std::size_t(l)] = importance_weight(qw[16 * j + l], sigma2, xb[16 * j + l]);
                float sumw = 0;
                for (const float w : weight)
                    sumw += w;
                sw[std::size_t(j)] = sumw;
                scales[std::size_t(j)] = make_qx_quants(16, 4, xb + 16 * j, L.data() + 16 * j, 1, weight.data());
            }
            const float d_block = make_qx_quants(16, 32, scales.data(), Ls.data(), 1, sw.data());
            for (std::size_t j = 0; j < 16; ++j)
                l6[j] = Ls[j];
            pack_q3_scales(l6.data(), b.scales);
            b.d = fp16(d_block);
        }
        for (int j = 0; j < 16; ++j) {
            const float d = f32(b.d) * float(q3_scale(b.scales, j));
            if (d == 0.0f)
                continue;
            for (int ii = 0; ii < 16; ++ii) {
                const float v = xb[16 * j + ii] / d;
                L[std::size_t(16 * j + ii)] = static_cast<std::int8_t>(imax(-4, imin(3, nearest_int(v))) + 4);
            }
        }
        b.hmask = {};
        for (std::size_t j = 0; j < 256; ++j)
            if (L[j] > 3) {
                b.hmask[j % 32] = static_cast<std::uint8_t>(b.hmask[j % 32] | (1u << (j / 32)));
                L[j] = static_cast<std::int8_t>(L[j] - 4);
            }
        pack2(L.data(), b.qs);
    }
}

/// \brief Quantises one row to Q4_K, as ggml's `quantize_q4_K`.
/// \param x The row, `256 · y.size()` floats.
/// \param importance Empty (the reference: `make_qkx2_quants` per 32), or one weight per number.
/// \param y Receives the blocks.
constexpr void quantize_row(std::span<const float> x, std::span<const float> importance, std::span<block_q4_k> y)
{
    using namespace detail::quant_k;
    check(256, y.size(), x.size(), importance.size());
    for (std::size_t i = 0; i < y.size(); ++i) {
        const float* xb = x.data() + 256 * i;
        std::array<std::uint8_t, 256> L{};
        block_q4_k& b = y[i];
        if (importance.empty())
            reference_k4<15>(xb, L, b.d, b.dmin, b.scales, -1.0f, 0.1f, 20);
        else
            weighted_k4<15>(xb, importance.data() + 256 * i, L, b.d, b.dmin, b.scales, false);
        requantize_k4(xb, b.d, b.dmin, b.scales, 15, L);
        for (std::size_t j = 0; j < 256; j += 64)
            for (std::size_t l = 0; l < 32; ++l)
                b.qs[j / 2 + l] = static_cast<std::uint8_t>(L[j + l] | (L[j + l + 32] << 4));
    }
}

/// \brief Quantises one row to Q5_K, as ggml's `quantize_q5_K`.
/// \param x The row, `256 · y.size()` floats.
/// \param importance Empty (the reference: `make_qkx2_quants` per 32), or one weight per number.
/// \param y Receives the blocks.
constexpr void quantize_row(std::span<const float> x, std::span<const float> importance, std::span<block_q5_k> y)
{
    using namespace detail::quant_k;
    check(256, y.size(), x.size(), importance.size());
    for (std::size_t i = 0; i < y.size(); ++i) {
        const float* xb = x.data() + 256 * i;
        std::array<std::uint8_t, 256> L{};
        block_q5_k& b = y[i];
        if (importance.empty())
            reference_k4<31>(xb, L, b.d, b.dmin, b.scales, -0.5f, 0.1f, 15);
        else
            weighted_k4<31>(xb, importance.data() + 256 * i, L, b.d, b.dmin, b.scales, true);
        requantize_k4(xb, b.d, b.dmin, b.scales, 31, L);
        b.qh = {};
        for (std::size_t n = 0; n < 256; n += 64)
            for (std::size_t j = 0; j < 32; ++j) {
                int l1 = L[n + j], l2 = L[n + j + 32];
                if (l1 > 15) {
                    l1 -= 16;
                    b.qh[j] = static_cast<std::uint8_t>(b.qh[j] | (1u << (n / 32)));
                }
                if (l2 > 15) {
                    l2 -= 16;
                    b.qh[j] = static_cast<std::uint8_t>(b.qh[j] | (2u << (n / 32)));
                }
                b.qs[n / 2 + j] = static_cast<std::uint8_t>(l1 | (l2 << 4));
            }
    }
}

/// \brief Quantises one row to Q6_K, as ggml's `quantize_q6_K`.
/// \param x The row, `256 · y.size()` floats.
/// \param importance Empty (the reference: `make_qx_quants` per 16, weighted x^2), or one weight per number (used as the weights directly, as ggml does).
/// \param y Receives the blocks.
constexpr void quantize_row(std::span<const float> x, std::span<const float> importance, std::span<block_q6_k> y)
{
    using namespace detail::quant_k;
    check(256, y.size(), x.size(), importance.size());
    for (std::size_t i = 0; i < y.size(); ++i) {
        const float* xb = x.data() + 256 * i;
        std::array<std::int8_t, 256> L{};
        std::array<float, 16> scales{};
        block_q6_k& b = y[i];
        float largest_scale = 0, max_abs_scale = 0;
        for (int ib = 0; ib < 16; ++ib) {
            const float* qw = importance.empty() ? nullptr : importance.data() + 256 * i + 16 * std::size_t(ib);
            const float scale = make_qx_quants(16, 32, xb + 16 * ib, L.data() + 16 * ib, 1, qw);
            scales[std::size_t(ib)] = scale;
            if (abs(scale) > max_abs_scale) {
                max_abs_scale = abs(scale);
                largest_scale = scale;
            }
        }
        if (max_abs_scale < group_max_eps) {
            b = block_q6_k{};
            b.d = fp16(0.0f);
            continue;
        }
        const float iscale = -128.0f / largest_scale;
        b.d = fp16(1 / iscale);
        for (std::size_t ib = 0; ib < 16; ++ib) {
            const float v = iscale * scales[ib];
            b.scales[ib] = static_cast<std::int8_t>(imin(127, nearest_int(v)));
        }
        for (int j = 0; j < 16; ++j) {
            const float d = f32(b.d) * float(b.scales[std::size_t(j)]);
            if (d == 0.0f)
                continue;
            for (int ii = 0; ii < 16; ++ii) {
                const float v = xb[16 * j + ii] / d;
                L[std::size_t(16 * j + ii)] = static_cast<std::int8_t>(imax(-32, imin(31, nearest_int(v))) + 32);
            }
        }
        for (std::size_t j = 0; j < 256; j += 128)
            for (std::size_t l = 0; l < 32; ++l) {
                const int q1 = L[j + l] & 0xf, q2 = L[j + l + 32] & 0xf, q3 = L[j + l + 64] & 0xf, q4 = L[j + l + 96] & 0xf;
                b.ql[j / 2 + l] = static_cast<std::uint8_t>(q1 | (q3 << 4));
                b.ql[j / 2 + l + 32] = static_cast<std::uint8_t>(q2 | (q4 << 4));
                b.qh[j / 4 + l] = static_cast<std::uint8_t>((L[j + l] >> 4) | ((L[j + l + 32] >> 4) << 2) | ((L[j + l + 64] >> 4) << 4) | ((L[j + l + 96] >> 4) << 6));
            }
    }
}

/// \brief Dequantises Q2_K blocks: `x = d·s·q - dmin·m`, as ggml's `dequantize_row_q2_K`.
/// \param x The blocks.
/// \param y Receives `256 · x.size()` floats.
constexpr void dequantize_row(std::span<const block_q2_k> x, std::span<float> y)
{
    assert(y.size() == 256 * x.size());
    std::array<std::int8_t, 256> a{};
    for (std::size_t i = 0; i < x.size(); ++i) {
        detail::quant_k::unpack(x[i], a);
        const float d = to_float(x[i].d), min = to_float(x[i].dmin);
        for (std::size_t j = 0; j < 16; ++j) {
            const std::uint8_t sc = x[i].scales[j];
            const float dl = d * float(sc & 0xf);
            const float ml = min * float(sc >> 4);
            for (std::size_t l = 0; l < 16; ++l) {
                const float v = dl * float(a[16 * j + l]);
                y[256 * i + 16 * j + l] = v - ml;
            }
        }
    }
}

/// \brief Dequantises Q3_K blocks: `x = d·(s - 32)·(q - 4)`, as ggml's `dequantize_row_q3_K`.
/// \param x The blocks.
/// \param y Receives `256 · x.size()` floats.
constexpr void dequantize_row(std::span<const block_q3_k> x, std::span<float> y)
{
    assert(y.size() == 256 * x.size());
    std::array<std::int8_t, 256> a{};
    for (std::size_t i = 0; i < x.size(); ++i) {
        detail::quant_k::unpack(x[i], a);
        const float d_all = to_float(x[i].d);
        for (std::size_t j = 0; j < 16; ++j) {
            const float dl = d_all * float(detail::quant_k::q3_scale(x[i].scales, int(j)));
            for (std::size_t l = 0; l < 16; ++l)
                y[256 * i + 16 * j + l] = dl * float(a[16 * j + l]);
        }
    }
}

namespace detail::quant_k
{
    // Q4_K's and Q5_K's dequantisation: d · s · q - dmin · m per 32
    template <typename B>
    constexpr void dequantize_k4(std::span<const B> x, std::span<float> y)
    {
        assert(y.size() == 256 * x.size());
        std::array<std::int8_t, 256> a{};
        for (std::size_t i = 0; i < x.size(); ++i) {
            unpack(x[i], a);
            const float d = to_float(x[i].d), min = to_float(x[i].dmin);
            for (std::size_t j = 0; j < 8; ++j) {
                std::uint8_t sc = 0, m = 0;
                scale_min_k4(int(j), x[i].scales, sc, m);
                const float d1 = d * float(sc);
                const float m1 = min * float(m);
                for (std::size_t l = 0; l < 32; ++l) {
                    const float v = d1 * float(a[32 * j + l]);
                    y[256 * i + 32 * j + l] = v - m1;
                }
            }
        }
    }

    // Q4_K's and Q5_K's dot product: the lanes with a scale per 32, minus dmin · (mins · bsums) per block
    template <typename B>
    constexpr float dot_k4(std::span<const B> x, std::span<const block_q8_k> y)
    {
        assert(x.size() == y.size());
        std::array<float, 8> sums{};
        std::array<std::int8_t, 256> a{};
        float sumf = 0;
        for (std::size_t i = 0; i < x.size(); ++i) {
            const B& b = x[i];
            unpack(b, a);
            std::array<int, 16> sc{};
            int sumi = 0;
            for (std::size_t j = 0; j < 8; ++j) {
                std::uint8_t s = 0, m = 0;
                scale_min_k4(int(j), b.scales, s, m);
                sc[2 * j] = sc[2 * j + 1] = s;
                sumi += y[i].bsums[2 * j] * m;
                sumi += y[i].bsums[2 * j + 1] * m;
            }
            const float d = to_float(b.d) * y[i].d;
            lanes(y[i], d, a, sc, sums);
            const float dmin = to_float(b.dmin) * y[i].d;
            const float term = dmin * float(sumi);
            sumf -= term;
        }
        return sum_lanes(sumf, sums);
    }
} // namespace detail::quant_k

/// \brief Dequantises Q4_K blocks: `x = d·s·q - dmin·m`, as ggml's `dequantize_row_q4_K`.
/// \param x The blocks.
/// \param y Receives `256 · x.size()` floats.
constexpr void dequantize_row(std::span<const block_q4_k> x, std::span<float> y) { detail::quant_k::dequantize_k4(x, y); }

/// \brief Dequantises Q5_K blocks: `x = d·s·q - dmin·m`, as ggml's `dequantize_row_q5_K`.
/// \param x The blocks.
/// \param y Receives `256 · x.size()` floats.
constexpr void dequantize_row(std::span<const block_q5_k> x, std::span<float> y) { detail::quant_k::dequantize_k4(x, y); }

/// \brief Dequantises Q6_K blocks: `x = d·s·(q - 32)`, as ggml's `dequantize_row_q6_K`.
/// \param x The blocks.
/// \param y Receives `256 · x.size()` floats.
constexpr void dequantize_row(std::span<const block_q6_k> x, std::span<float> y)
{
    assert(y.size() == 256 * x.size());
    std::array<std::int8_t, 256> a{};
    for (std::size_t i = 0; i < x.size(); ++i) {
        detail::quant_k::unpack(x[i], a);
        const float d = to_float(x[i].d);
        for (std::size_t j = 0; j < 16; ++j) {
            const float ds = d * float(x[i].scales[j]);
            for (std::size_t l = 0; l < 16; ++l)
                y[256 * i + 16 * j + l] = ds * float(a[16 * j + l]);
        }
    }
}

/// \brief The dot product of a Q2_K row with a Q8_K row, as ggml's `ggml_vec_dot_q2_K_q8_K`.
/// \param x The weights.
/// \param y The activations.
/// \return `sum over blocks of d_x d_y (sum s · q_x · q_y) - dmin_x d_y (sum m · bsum)`.
constexpr float dot(std::span<const block_q2_k> x, std::span<const block_q8_k> y)
{
    assert(x.size() == y.size());
    std::array<std::int8_t, 256> a{};
    float sumf = 0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        detail::quant_k::unpack(x[i], a);
        int summs = 0;
        for (std::size_t j = 0; j < 16; ++j)
            summs += y[i].bsums[j] * (x[i].scales[j] >> 4);
        const float dall = y[i].d * to_float(x[i].d);
        const float dmin = y[i].d * to_float(x[i].dmin);
        int isum = 0;
        for (std::size_t j = 0; j < 16; ++j) {
            int isuml = 0;
            for (std::size_t l = 0; l < 16; ++l)
                isuml += int(y[i].qs[16 * j + l]) * int(a[16 * j + l]);
            isum += (x[i].scales[j] & 0xf) * isuml;
        }
        const float p = dall * float(isum);
        const float q = dmin * float(summs);
        const float term = p - q;
        sumf += term;
    }
    return sumf;
}

/// \brief The dot product of a Q3_K row with a Q8_K row, as ggml's `ggml_vec_dot_q3_K_q8_K`, in its eight lanes.
/// \param x The weights.
/// \param y The activations.
/// \return `sum over blocks of d_x d_y (sum (s - 32) · (q_x - 4) · q_y)`.
constexpr float dot(std::span<const block_q3_k> x, std::span<const block_q8_k> y)
{
    assert(x.size() == y.size());
    std::array<float, 8> sums{};
    std::array<std::int8_t, 256> a{};
    for (std::size_t i = 0; i < x.size(); ++i) {
        detail::quant_k::unpack(x[i], a);
        std::array<int, 16> sc{};
        for (std::size_t j = 0; j < 16; ++j)
            sc[j] = detail::quant_k::q3_scale(x[i].scales, int(j));
        const float d = to_float(x[i].d) * y[i].d;
        detail::quant_k::lanes(y[i], d, a, sc, sums);
    }
    return detail::quant_k::sum_lanes(0.0f, sums);
}

/// \brief The dot product of a Q4_K row with a Q8_K row, as ggml's `ggml_vec_dot_q4_K_q8_K`, in its eight lanes.
/// \param x The weights.
/// \param y The activations.
/// \return `sum over blocks of d_x d_y (sum s · q_x · q_y) - dmin_x d_y (sum m · bsum)`.
constexpr float dot(std::span<const block_q4_k> x, std::span<const block_q8_k> y) { return detail::quant_k::dot_k4(x, y); }

/// \brief The dot product of a Q5_K row with a Q8_K row, as ggml's `ggml_vec_dot_q5_K_q8_K`, in its eight lanes.
/// \param x The weights.
/// \param y The activations.
/// \return `sum over blocks of d_x d_y (sum s · q_x · q_y) - dmin_x d_y (sum m · bsum)`.
constexpr float dot(std::span<const block_q5_k> x, std::span<const block_q8_k> y) { return detail::quant_k::dot_k4(x, y); }

/// \brief The dot product of a Q6_K row with a Q8_K row, as ggml's `ggml_vec_dot_q6_K_q8_K`, in its eight lanes.
/// \param x The weights.
/// \param y The activations.
/// \return `sum over blocks of d_x d_y (sum s · (q_x - 32) · q_y)`.
constexpr float dot(std::span<const block_q6_k> x, std::span<const block_q8_k> y)
{
    assert(x.size() == y.size());
    std::array<float, 8> sums{};
    std::array<std::int8_t, 256> a{};
    for (std::size_t i = 0; i < x.size(); ++i) {
        detail::quant_k::unpack(x[i], a);
        std::array<int, 16> sc{};
        for (std::size_t j = 0; j < 16; ++j)
            sc[j] = x[i].scales[j];
        const float d = to_float(x[i].d) * y[i].d;
        detail::quant_k::lanes(y[i], d, a, sc, sums);
    }
    return detail::quant_k::sum_lanes(0.0f, sums);
}

} // namespace snicholls::ml

#endif // snicholls_core_ml_quant_k_hpp
