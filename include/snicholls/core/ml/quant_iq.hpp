//
//  quant_iq.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::ml` ggml's IQ quantisation family, byte for byte: IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S (lattice codebooks with sign bits), IQ4_NL and IQ4_XS (non-linear 4-bit values), with ggml's importance-weighted quantisers, the decoders and the integer dot products with Q8_K (Q8_0 for IQ4_NL), all `constexpr`.
/// \ingroup core
///
/// **Overview.** The IQ formats quantise a group of 8 (IQ1, IQ2) or 4 (IQ3) magnitudes to the index of the nearest
/// point of a fixed grid (`ml/quant_iq_grids.hpp`), the signs to 7 or 8 bits, and a sub-block scale to 3 or 4
/// bits; IQ4 maps each weight to one of 16 non-linear values. They are what llama.cpp ships below 4 bits per
/// weight:
///
/// | Block | Elements | Bytes | Bits per weight | Groups | Importance matrix |
/// |---|---|---|---|---|---|
/// | `block_iq1_s` | 256 | 50 | 1.5625 | 8 of the 2048-point ternary grid, shift ±1/8 per 32 | required |
/// | `block_iq1_m` | 256 | 56 | 1.75 | as IQ1_S, a shift per 8 and a scale per 16 | optional |
/// | `block_iq2_xxs` | 256 | 66 | 2.0625 | 8 of 256 points, 7 sign bits (even parity) | required |
/// | `block_iq2_xs` | 256 | 74 | 2.3125 | 8 of 512 points, a scale per 16 | required |
/// | `block_iq2_s` | 256 | 82 | 2.5625 | 8 of 1024 points, 8 sign bits | optional |
/// | `block_iq3_xxs` | 256 | 98 | 3.0625 | 4 of 256 points, 7 sign bits per 8 | optional |
/// | `block_iq3_s` | 256 | 110 | 3.4375 | 4 of 512 points, 8 sign bits | optional |
/// | `block_iq4_nl` | 32 | 18 | 4.5 | 16 non-linear values | optional |
/// | `block_iq4_xs` | 256 | 136 | 4.25 | as IQ4_NL, 6-bit scales per 32 | optional |
///
/// Each follows `ml/quantize.hpp`'s contract: `quantize_row(x, importance, blocks)` is ggml's
/// `quantize_<type>(x, blocks, 1, n, importance)` (an empty importance is ggml's null; IQ1_S, IQ2_XXS and IQ2_XS,
/// which ggml refuses to quantise without one, assert it), `dequantize_row` is `dequantize_row_<type>`, and `dot`
/// is `ggml_vec_dot_<type>_q8_K_generic`. `quantize_row_reference` is IQ4_NL's `quantize_row_iq4_nl_ref` (no scale
/// search, as ggml's CPU backend quantises IQ4_NL activations and caches).
///
/// \snippet ml/quant_iq_demo.cpp iq
///
/// **Relationships.** `ml/quantize.hpp` (the type ids, `block_q8_k`, `block_q8_0`, the scale searches
/// `make_qp_quants`, `nearest_int`), `ml/quant_iq_grids.hpp` (the codebooks). `block_iq4_nl` is a `quant_block`, so
/// `ml/linear.hpp`'s `quantized_weights` takes it. Group: \ref core.
///
/// **Implementations.** ggml keeps, per grid, a table of each off-grid point's nearest grid points (`kneighbors`,
/// built by `iq2xs_init_impl` / `iq3xs_init_impl` over 43,692 and 4,096 points). Core computes a point's
/// neighbours when the quantiser asks for them: every grid point's distance, then the points of the `nwant`
/// smallest distances in ggml's sorted order (distance, then index), so the choice is ggml's to the bit and the
/// function stays `constexpr` without a 100 KB table. That costs one pass over the grid per off-grid group.
///
/// **Guarantees.** `constexpr` (quantising a whole IQ1, IQ2 or IQ3 block at compile time takes more than Clang's
/// default 1,048,576 evaluation steps, `-fconstexpr-steps`; GCC's limit is far higher), no allocation (the largest buffer is 8 KB on the stack, IQ1's distances), linear in
/// the row. The blocks are trivially copyable with ggml's sizes (`static_assert`ed). Bytes, dequantised floats and
/// dot products equal ggml's on every row kind of the committed fixture (`tests/fixtures/ml/quant_iq_fixture.txt`,
/// from `scripts/generate_quant_iq_fixture.cpp` built against ggml), except where noted under Known issues. Each
/// multiplication and addition is its own statement, so no FMA is formed (compile GCC with `-ffp-contract=off`).
///
/// **Errors.** Preconditions, asserted: the row has `elements · blocks.size()` numbers; `importance` is empty or as
/// long as the row (and not empty for IQ1_S, IQ2_XXS, IQ2_XS); dot products' rows have the same number of blocks.
///
/// **Performance.** Dot products and dequantisation against ggml's scalar loops written out by hand
/// (`benchmarks/ml/quant_iq_benchmark.cpp`); see BENCHMARKS.md. Quantisation is a model-conversion step: the
/// on-demand neighbour search makes it slower than ggml's table lookups (not benchmarked against them).
///
/// **Tests and demos.** `tests/ml/quant_iq_test.cpp`, `demos/ml/quant_iq_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<cassert>`, `<cstddef>`, `<cstdint>`, `<limits>`, `<span>`, `ml/quantize.hpp`,
/// `ml/quant_iq_grids.hpp`.
///
/// **Known issues.** IQ1_S and IQ1_M sort each block's weights with C's `qsort` on the value alone, so ggml's order
/// of equal weights is its C library's; core sorts stably (equal values in their original order). Where a block
/// has equal weights the bytes can differ from ggml's on some C libraries; the committed fixture (macOS) agrees on
/// every row kind except where a tie changes the prefix sums (listed in the test). With a NaN in a row ggml aborts
/// for IQ1_M, IQ2_XS, IQ2_S, IQ3_XXS and IQ3_S (its "point not on grid" assertion) and core asserts likewise;
/// IQ1_S, IQ2_XXS and the IQ4 formats give ggml's bytes. Where ggml's IQ1 neighbour search finds no candidate (only
/// with infinities) its fallback reads past the group; core searches the whole grid instead. IQ4_NL's
/// `quantize_row_reference` leaves a block of zeros with the codes of the block before it, as ggml does (the first
/// block's codes are then 0; ggml's are uninitialised memory).
///
/// **Status.** Experimental, new in 0.17.0.

#ifndef snicholls_core_ml_quant_iq_hpp
#define snicholls_core_ml_quant_iq_hpp

#include <snicholls/core/ml/quant_iq_grids.hpp>
#include <snicholls/core/ml/quantize.hpp>

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>

namespace snicholls::ml
{

/// \brief ggml's IQ1_S: 256 weights in 8 groups of 32, each 4 points of the 2048-point ternary grid, a 3-bit scale and a shift of ±1/8: about 1.56 bits a weight.
/// \ingroup core
struct block_iq1_s
{
    static constexpr ggml_type type = ggml_type::iq1_s;  ///< The ggml id.
    static constexpr std::size_t elements = 256;         ///< Numbers per block.
    /// \brief ggml's `vec_dot_type`.
    using dot_block = block_q8_k;

    float16 d;                            ///< The super-block scale.
    std::array<std::uint8_t, 32> qs;      ///< Grid index, low 8 bits, per 8 weights.
    std::array<std::uint16_t, 8> qh;      ///< Per 32: grid index high 3 bits of each 8, the 3-bit scale (bits 12 to 14), the shift's sign (bit 15).
};

/// \brief ggml's IQ1_M: as IQ1_S with a shift per 8 weights and a 3-bit scale per 16, the super-block scale spread over the scales' top nibbles: 1.75 bits a weight.
/// \ingroup core
struct block_iq1_m
{
    static constexpr ggml_type type = ggml_type::iq1_m;  ///< The ggml id.
    static constexpr std::size_t elements = 256;         ///< Numbers per block.
    /// \brief ggml's `vec_dot_type`.
    using dot_block = block_q8_k;

    std::array<std::uint8_t, 32> qs;      ///< Grid index, low 8 bits, per 8 weights.
    std::array<std::uint8_t, 16> qh;      ///< Per 16: two grid indices' high 3 bits and their shifts' signs (bits 3, 7).
    std::array<std::uint8_t, 8> scales;   ///< Four little-endian 16-bit words: 3-bit scales, and the float16 scale's nibbles on top.
};

/// \brief ggml's IQ2_XXS: 256 weights, per 8 an index into the 256-point grid and 7 sign bits (even parity), a 4-bit scale per 32: 2.06 bits a weight.
/// \ingroup core
struct block_iq2_xxs
{
    static constexpr ggml_type type = ggml_type::iq2_xxs;  ///< The ggml id.
    static constexpr std::size_t elements = 256;           ///< Numbers per block.
    /// \brief ggml's `vec_dot_type`.
    using dot_block = block_q8_k;

    float16 d;                             ///< The super-block scale.
    std::array<std::uint16_t, 32> qs;      ///< Per 32, two 32-bit words: four grid indices; four 7-bit signs and the scale (bits 28 to 31).
};

/// \brief ggml's IQ2_XS: 256 weights, per 8 an index into the 512-point grid and 7 sign bits, a 4-bit scale per 16: 2.31 bits a weight.
/// \ingroup core
struct block_iq2_xs
{
    static constexpr ggml_type type = ggml_type::iq2_xs;  ///< The ggml id.
    static constexpr std::size_t elements = 256;          ///< Numbers per block.
    /// \brief ggml's `vec_dot_type`.
    using dot_block = block_q8_k;

    float16 d;                             ///< The super-block scale.
    std::array<std::uint16_t, 32> qs;      ///< Per 8: the grid index (9 bits) and the signs (7 bits).
    std::array<std::uint8_t, 8> scales;    ///< Two 4-bit scales a byte, one per 16.
};

/// \brief ggml's IQ2_S: 256 weights, per 8 an index into the 1024-point grid and 8 sign bits, a 4-bit scale per 16: 2.56 bits a weight.
/// \ingroup core
struct block_iq2_s
{
    static constexpr ggml_type type = ggml_type::iq2_s;  ///< The ggml id.
    static constexpr std::size_t elements = 256;         ///< Numbers per block.
    /// \brief ggml's `vec_dot_type`.
    using dot_block = block_q8_k;

    float16 d;                             ///< The super-block scale.
    std::array<std::uint8_t, 64> qs;       ///< The grid indices' low 8 bits (32), then the signs (32).
    std::array<std::uint8_t, 8> qh;        ///< The grid indices' high 2 bits, four a byte.
    std::array<std::uint8_t, 8> scales;    ///< Two 4-bit scales a byte, one per 16.
};

/// \brief ggml's IQ3_XXS: 256 weights, per 4 an index into the 256-point grid, per 8 seven sign bits, a 4-bit scale per 32: 3.06 bits a weight.
/// \ingroup core
struct block_iq3_xxs
{
    static constexpr ggml_type type = ggml_type::iq3_xxs;  ///< The ggml id.
    static constexpr std::size_t elements = 256;           ///< Numbers per block.
    /// \brief ggml's `vec_dot_type`.
    using dot_block = block_q8_k;

    float16 d;                             ///< The super-block scale.
    std::array<std::uint8_t, 96> qs;       ///< 64 grid indices, then per 32 a little-endian word: four 7-bit signs and the scale (bits 28 to 31).
};

/// \brief ggml's IQ3_S: 256 weights, per 4 an index into the 512-point grid, 8 sign bits per 8, a 4-bit scale per 32: 3.44 bits a weight.
/// \ingroup core
struct block_iq3_s
{
    static constexpr ggml_type type = ggml_type::iq3_s;  ///< The ggml id.
    static constexpr std::size_t elements = 256;         ///< Numbers per block.
    /// \brief ggml's `vec_dot_type`.
    using dot_block = block_q8_k;

    float16 d;                             ///< The super-block scale.
    std::array<std::uint8_t, 64> qs;       ///< The grid indices' low 8 bits.
    std::array<std::uint8_t, 8> qh;        ///< The grid indices' 9th bits, one byte per 32.
    std::array<std::uint8_t, 32> signs;    ///< The signs, one byte per 8.
    std::array<std::uint8_t, 4> scales;    ///< Two 4-bit scales a byte, one per 32.
};

/// \brief ggml's IQ4_NL: 32 weights, each one of 16 non-linear values (`iq4nl_values`) times a `float16` scale: 4.5 bits a weight.
/// \ingroup core
struct block_iq4_nl
{
    static constexpr ggml_type type = ggml_type::iq4_nl;  ///< The ggml id.
    static constexpr std::size_t elements = 32;           ///< Numbers per block.
    /// \brief ggml's `vec_dot_type`.
    using dot_block = block_q8_0;

    float16 d;                             ///< The scale.
    std::array<std::uint8_t, 16> qs;       ///< Value indices, packed as Q4_0's.
};

/// \brief ggml's IQ4_XS: 256 weights as IQ4_NL's values, with a 6-bit scale per 32 under one `float16` scale: 4.25 bits a weight.
/// \ingroup core
struct block_iq4_xs
{
    static constexpr ggml_type type = ggml_type::iq4_xs;  ///< The ggml id.
    static constexpr std::size_t elements = 256;          ///< Numbers per block.
    /// \brief ggml's `vec_dot_type`.
    using dot_block = block_q8_k;

    float16 d;                             ///< The super-block scale.
    std::uint16_t scales_h;                ///< The scales' high 2 bits, two bits per 32.
    std::array<std::uint8_t, 4> scales_l;  ///< The scales' low 4 bits, two a byte.
    std::array<std::uint8_t, 128> qs;      ///< Value indices, per 32 packed as Q4_0's.
};

static_assert(quantized_block<block_iq1_s> && quantized_block<block_iq1_m> && quantized_block<block_iq2_xxs>
                  && quantized_block<block_iq2_xs> && quantized_block<block_iq2_s> && quantized_block<block_iq3_xxs>
                  && quantized_block<block_iq3_s> && quantized_block<block_iq4_nl> && quantized_block<block_iq4_xs>,
              "snicholls::ml: the IQ blocks must have ggml's sizes");

namespace detail::iq
{
    using detail::quantize::fmax;
    using detail::quantize::imax;
    using detail::quantize::imin;
    using detail::quantize::nearest_int;

    inline constexpr float flt_max = std::numeric_limits<float>::max();

    // A grid as the quantiser sees it: its index list, bits per coordinate, coordinates, and how many distinct
    // distances of neighbours ggml keeps for a point off the grid (nwant)
    template <std::size_t N>
    struct lattice
    {
        const std::array<std::uint16_t, N>& index;
        int bits;
        int dims;
        int nwant;

        constexpr int coordinate(std::size_t g, int k) const { return iq_grid_coordinate(index, bits, g, k); }
        constexpr int level(std::size_t g, int k) const { return int((index[g] >> (bits * k)) & ((1u << bits) - 1)); }
    };

    inline constexpr lattice<256> iq2xxs{iq2xxs_grid_index, 2, 8, 2};
    inline constexpr lattice<512> iq2xs{iq2xs_grid_index, 2, 8, 2};
    inline constexpr lattice<1024> iq2s{iq2s_grid_index, 2, 8, 1};
    inline constexpr lattice<2048> iq1s{iq1s_grid_index, 2, 8, 3};
    inline constexpr lattice<256> iq3xxs{iq3xxs_grid_index, 3, 4, 2};
    inline constexpr lattice<512> iq3s{iq3s_grid_index, 3, 4, 3};

    // ggml's neighbour list of the off-grid point u, visited in its order: every grid point whose squared distance
    // to u is one of the nwant smallest, by distance then index. score(g) is a candidate's error; the first with
    // the least error wins (ggml's strict <). Returns -1 when no score is below FLT_MAX.
    template <std::size_t N, typename Score>
    constexpr int best_neighbour(const lattice<N>& grid, std::uint16_t u, Score score)
    {
        std::array<int, N> d2{};
        int level[3] = {std::numeric_limits<int>::max(), std::numeric_limits<int>::max(), std::numeric_limits<int>::max()};
        for (std::size_t g = 0; g < N; ++g) {
            int s = 0;
            for (int k = 0; k < grid.dims; ++k) {
                const int pos = 2 * int((u >> (grid.bits * k)) & ((1u << grid.bits) - 1)) + 1;
                const int diff = grid.coordinate(g, k) - pos;
                s += diff * diff;
            }
            d2[g] = s;
            // keep the nwant smallest distinct distances, ascending
            for (int l = 0; l < grid.nwant; ++l) {
                if (s == level[l])
                    break;
                if (s < level[l]) {
                    for (int m = grid.nwant - 1; m > l; --m)
                        level[m] = level[m - 1];
                    level[l] = s;
                    break;
                }
            }
        }
        float best = flt_max;
        int best_g = -1;
        for (int l = 0; l < grid.nwant && level[l] != std::numeric_limits<int>::max(); ++l)
            for (std::size_t g = 0; g < N; ++g)
                if (d2[g] == level[l]) {
                    const float e = score(g);
                    if (e < best) {
                        best = e;
                        best_g = int(g);
                    }
                }
        return best_g;
    }

    // ggml's iq2_find_best_neighbour / iq3_find_best_neighbour: the neighbour nearest scale · grid in the weighted
    // norm; writes its levels to L
    template <std::size_t N>
    constexpr int find_best_neighbour(const lattice<N>& grid, std::uint16_t u, const float* xval, const float* weight, float scale, std::int8_t* L)
    {
        const int g = best_neighbour(grid, u, [&](std::size_t j) {
            float d2 = 0;
            for (int i = 0; i < grid.dims; ++i) {
                const float q = float(grid.coordinate(j, i));
                const float sq = scale * q;
                const float diff = sq - xval[i];
                const float wd = weight[i] * diff;
                const float wdd = wd * diff;
                d2 += wdd;
            }
            return d2;
        });
        assert(g >= 0 && "snicholls::ml: no grid point near a group (a NaN in the row?)");
        for (int i = 0; i < grid.dims; ++i)
            L[i] = static_cast<std::int8_t>(grid.level(std::size_t(g), i));
        return g;
    }

    template <std::size_t N>
    constexpr std::uint16_t pack(const lattice<N>& grid, const std::int8_t* L)
    {
        std::uint16_t u = 0;
        for (int i = 0; i < grid.dims; ++i)
            u = static_cast<std::uint16_t>(u | (L[i] << (grid.bits * i)));
        return u;
    }

    // nearest_int(0.5 · (id · x - 1)): the level of x at inverse scale id, before clamping
    constexpr int level_of(float id, float x)
    {
        const float t = id * x;
        const float t1 = t - 1.0f;
        const float h = 0.5f * t1;
        return nearest_int(h);
    }

    // The sum of x^2 over a super-block
    constexpr float sum_squares(const float* x, int n)
    {
        float s = 0;
        for (int i = 0; i < n; ++i) {
            const float x2 = x[i] * x[i];
            s += x2;
        }
        return s;
    }

    // sumqx and sumq2 of levels L (q = 2l + 1) against xval with weights
    constexpr void sums(int n, const float* weight, const float* xval, const std::int8_t* L, float& sumqx, float& sumq2)
    {
        sumqx = 0;
        sumq2 = 0;
        for (int i = 0; i < n; ++i) {
            const float w = weight[i];
            const float q = float(2 * L[i] + 1);
            const float wx = w * xval[i];
            const float wxq = wx * q;
            sumqx += wxq;
            const float wq = w * q;
            const float wqq = wq * q;
            sumq2 += wqq;
        }
    }

    // |x| into xval with the signs in s, and with even parity: when an odd number are negative, the one of least
    // weight · x^2 is flipped (IQ2_XXS, IQ2_XS, IQ3_XXS store 7 sign bits)
    constexpr std::uint8_t even_signs(const float* xb, const float* weight, float* xval)
    {
        int nflip = 0;
        std::uint8_t s = 0;
        for (int i = 0; i < 8; ++i) {
            if (xb[i] >= 0) {
                xval[i] = xb[i];
            } else {
                xval[i] = -xb[i];
                ++nflip;
                s = static_cast<std::uint8_t>(s | (1 << i));
            }
        }
        if (nflip % 2) {
            int imin_ = 0;
            const float w0 = weight[0] * xb[0];
            float min = w0 * xb[0];
            for (int i = 1; i < 8; ++i) {
                const float wi = weight[i] * xb[i];
                const float ax = wi * xb[i];
                if (ax < min) {
                    min = ax;
                    imin_ = i;
                }
            }
            xval[imin_] = -xval[imin_];
            s = static_cast<std::uint8_t>(s ^ (1 << imin_));
        }
        return static_cast<std::uint8_t>(s & 127);
    }

    constexpr std::uint8_t plain_signs(const float* xb, float* xval)
    {
        std::uint8_t s = 0;
        for (int i = 0; i < 8; ++i) {
            if (xb[i] >= 0) {
                xval[i] = xb[i];
            } else {
                xval[i] = -xb[i];
                s = static_cast<std::uint8_t>(s | (1 << i));
            }
        }
        return s;
    }

    constexpr std::uint8_t grid_byte(std::uint64_t point, int j) { return static_cast<std::uint8_t>(point >> (8 * j)); }
    constexpr std::int8_t grid_sbyte(std::uint64_t point, int j) { return static_cast<std::int8_t>(static_cast<std::uint8_t>(point >> (8 * j))); }
    constexpr bool negative(std::uint8_t signs, int j) { return (signs >> j) & 1; }
    constexpr int sign(std::uint8_t signs, int j) { return (signs >> j) & 1 ? -1 : 1; }

    constexpr std::uint32_t load32(const std::uint8_t* p)
    {
        return p[0] | (std::uint32_t(p[1]) << 8) | (std::uint32_t(p[2]) << 16) | (std::uint32_t(p[3]) << 24);
    }
    constexpr void store32(std::uint8_t* p, std::uint32_t v)
    {
        for (int k = 0; k < 4; ++k)
            p[k] = static_cast<std::uint8_t>(v >> (8 * k));
    }

    // The per-32 scale of IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S: nearest_int(0.5 (id s - 1)) in [0, 15]
    constexpr int scale_code(float id, float s) { return imax(0, imin(15, level_of(id, s))); }

    // IQ2_XXS (sub-blocks of 32, kMaxQ 3): ggml's quantize_row_iq2_xxs_impl
    constexpr void quantize(std::span<const float> x, std::span<const float> qw_row, std::span<block_iq2_xxs> y)
    {
        constexpr int kMaxQ = 3;
        std::array<float, 8> scales{};
        std::array<float, 32> weight{}, xval{}, waux{};
        std::array<std::int8_t, 32> L{}, Laux{};
        std::array<std::uint8_t, 4> block_signs{};
        for (std::size_t ibl = 0; ibl < y.size(); ++ibl) {
            std::array<std::uint32_t, 16> q2{};
            y[ibl].d = detail::quantize::fp16(0.0f);
            float max_scale = 0;
            const float* xbl = x.data() + 256 * ibl;
            const float sigma2 = sum_squares(xbl, 256) / 256.0f;
            for (int ib = 0; ib < 8; ++ib) {
                const float* xb = xbl + 32 * ib;
                const float* qw = qw_row.data() + 256 * ibl + 32 * ib;
                for (int i = 0; i < 32; ++i)
                    weight[i] = detail::quantize::importance_weight(qw[i], sigma2, xb[i]);
                for (int i = 0; i < 32; ++i)
                    waux[i] = detail::quantize::sqrt(weight[i]);
                for (int k = 0; k < 4; ++k)
                    block_signs[k] = even_signs(xb + 8 * k, weight.data() + 8 * k, xval.data() + 8 * k);
                float max = xval[0];
                for (int i = 1; i < 32; ++i)
                    max = fmax(max, xval[i]);
                if (max < detail::quantize::group_max_eps) {
                    scales[ib] = 0;
                    L = {};
                    continue;
                }
                std::array<std::uint8_t, 32> Lq{};
                float scale = detail::quantize::make_qp_quants(32, kMaxQ + 1, xval.data(), Lq.data(), weight.data());
                for (int i = 0; i < 32; ++i)
                    L[i] = static_cast<std::int8_t>(Lq[i]);
                const float eff_max = scale * float(kMaxQ);
                if (eff_max <= 0) {
                    scales[ib] = 0;
                    L = {};
                    continue;
                }
                float best = 0;
                for (int is = -6; is <= 6; ++is) {
                    const float step = float(is) * 0.1f;
                    const float top = float(2 * kMaxQ - 1) + step;
                    const float id = top / eff_max;
                    const float this_scale = 1 / id;
                    for (int k = 0; k < 4; ++k) {
                        for (int i = 0; i < 8; ++i)
                            Laux[8 * k + i] = static_cast<std::int8_t>(imax(0, imin(kMaxQ - 1, level_of(id, xval[8 * k + i]))));
                        const std::uint16_t u = pack(iq2xxs, Laux.data() + 8 * k);
                        if (iq_grid_find(iq2xxs.index, u) < 0)
                            find_best_neighbour(iq2xxs, u, xval.data() + 8 * k, waux.data() + 8 * k, this_scale, Laux.data() + 8 * k);
                    }
                    float sumqx = 0, sumq2 = 0;
                    sums(32, weight.data(), xval.data(), Laux.data(), sumqx, sumq2);
                    if (sumq2 > 0 && sumqx * sumqx > best * sumq2) {
                        scale = sumqx / sumq2;
                        best = scale * sumqx;
                        L = Laux;
                    }
                }
                if (scale > 0) {
                    const float id = 1 / scale;
                    for (int k = 0; k < 4; ++k) {
                        std::array<std::int8_t, 8> l8{};
                        for (int i = 0; i < 8; ++i)
                            l8[i] = static_cast<std::int8_t>(imax(0, imin(kMaxQ - 1, level_of(id, xval[8 * k + i]))));
                        const std::uint16_t u = pack(iq2xxs, l8.data());
                        int g = iq_grid_find(iq2xxs.index, u);
                        if (g < 0)
                            g = find_best_neighbour(iq2xxs, u, xval.data() + 8 * k, waux.data() + 8 * k, scale, L.data() + 8 * k);
                        for (int i = 0; i < 8; ++i)
                            L[8 * k + i] = static_cast<std::int8_t>(iq2xxs.level(std::size_t(g), i));
                    }
                    float sumqx = 0, sumq2 = 0;
                    sums(32, weight.data(), xval.data(), L.data(), sumqx, sumq2);
                    if (sumq2 > 0)
                        scale = sumqx / sumq2;
                }
                if (scale < 0) {
                    scale = -scale;
                    for (int k = 0; k < 4; ++k)
                        block_signs[k] = static_cast<std::uint8_t>((~block_signs[k]) & 127);
                }
                for (int k = 0; k < 4; ++k) {
                    const int g = iq_grid_find(iq2xxs.index, pack(iq2xxs, L.data() + 8 * k));
                    assert(g >= 0 && "snicholls::ml: IQ2_XXS found a point not on the grid");
                    q2[2 * ib] |= std::uint32_t(g) << (8 * k);
                    q2[2 * ib + 1] |= std::uint32_t(block_signs[k]) << (7 * k);
                }
                scales[ib] = scale;
                max_scale = fmax(max_scale, scale);
            }
            if (max_scale == 0.0f) {
                y[ibl].qs = {};
                continue;
            }
            const float d = max_scale / 31;
            y[ibl].d = detail::quantize::fp16(d);
            const float id = 1 / d;
            for (int ib = 0; ib < 8; ++ib)
                q2[2 * ib + 1] |= std::uint32_t(scale_code(id, scales[ib])) << 28;
            for (int i = 0; i < 16; ++i) {
                y[ibl].qs[2 * i] = static_cast<std::uint16_t>(q2[i]);
                y[ibl].qs[2 * i + 1] = static_cast<std::uint16_t>(q2[i] >> 16);
            }
        }
    }

    // IQ2_XS and IQ2_S (sub-blocks of 16, kMaxQ 3): ggml's quantize_row_iq2_xs_impl and quantize_row_iq2_s_impl,
    // which differ in their grid, signs (7 with parity, or 8), the weights without an importance row, the group
    // threshold, the scale's fudge factor and the layout
    template <bool S, typename Block>
    constexpr void quantize_iq2_16(std::span<const float> x, std::span<const float> qw_row, std::span<Block> y)
    {
        constexpr int kMaxQ = 3;
        constexpr auto& grid = []() -> const auto& {
            if constexpr (S)
                return iq2s;
            else
                return iq2xs;
        }();
        std::array<float, 16> scales{};
        std::array<float, 16> weight{}, xval{}, waux{};
        std::array<std::int8_t, 16> L{}, Laux{};
        std::array<bool, 2> on_grid{}, on_grid_aux{};
        std::array<std::uint8_t, 2> block_signs{};
        for (std::size_t ibl = 0; ibl < y.size(); ++ibl) {
            std::array<std::uint16_t, 32> q2{};
            y[ibl] = Block{};
            y[ibl].d = detail::quantize::fp16(0.0f);
            float max_scale = 0;
            const float* xbl = x.data() + 256 * ibl;
            const float sumx2 = sum_squares(xbl, 256);
            float sigma2 = 0;
            if constexpr (S) {
                const float two = 2 * sumx2;
                sigma2 = two / 256.0f;
            } else {
                sigma2 = sumx2 / 256.0f;
            }
            for (int ib = 0; ib < 16; ++ib) {
                const float* xb = xbl + 16 * ib;
                if (S && qw_row.empty()) {
                    const float quarter = 0.25f * sigma2;
                    for (int i = 0; i < 16; ++i) {
                        const float x2 = xb[i] * xb[i];
                        weight[i] = quarter + x2;
                    }
                } else {
                    const float* qw = qw_row.data() + 256 * ibl + 16 * ib;
                    for (int i = 0; i < 16; ++i)
                        weight[i] = detail::quantize::importance_weight(qw[i], sigma2, xb[i]);
                }
                for (int i = 0; i < 16; ++i)
                    waux[i] = detail::quantize::sqrt(weight[i]);
                for (int k = 0; k < 2; ++k)
                    block_signs[k] = S ? plain_signs(xb + 8 * k, xval.data() + 8 * k) : even_signs(xb + 8 * k, weight.data() + 8 * k, xval.data() + 8 * k);
                float max = xval[0];
                for (int i = 1; i < 16; ++i)
                    max = fmax(max, xval[i]);
                L = {};
                if (max < (S ? 1e-8f : detail::quantize::group_max_eps)) {
                    scales[ib] = 0;
                    continue;
                }
                float best = 0;
                float scale = max / float(2 * kMaxQ - 1);
                on_grid = {true, true};
                for (int is = -9; is <= 9; ++is) {
                    const float step = float(is) * 0.1f;
                    const float top = float(2 * kMaxQ - 1) + step;
                    const float id = top / max;
                    const float this_scale = 1 / id;
                    for (int k = 0; k < 2; ++k) {
                        for (int i = 0; i < 8; ++i)
                            Laux[8 * k + i] = static_cast<std::int8_t>(imax(0, imin(kMaxQ - 1, level_of(id, xval[8 * k + i]))));
                        const std::uint16_t u = pack(grid, Laux.data() + 8 * k);
                        on_grid_aux[k] = true;
                        if (iq_grid_find(grid.index, u) < 0) {
                            on_grid_aux[k] = false;
                            find_best_neighbour(grid, u, xval.data() + 8 * k, waux.data() + 8 * k, this_scale, Laux.data() + 8 * k);
                        }
                    }
                    float sumqx = 0, sumq2 = 0;
                    sums(16, weight.data(), xval.data(), Laux.data(), sumqx, sumq2);
                    if (sumq2 > 0 && sumqx * sumqx > best * sumq2) {
                        scale = sumqx / sumq2;
                        best = scale * sumqx;
                        L = Laux;
                        on_grid = on_grid_aux;
                    }
                }
                const int n_not_ongrid = int(!on_grid[0]) + int(!on_grid[1]);
                if (n_not_ongrid > 0 && scale > 0) {
                    const float id = 1 / scale;
                    for (int k = 0; k < 2; ++k) {
                        if (on_grid[k])
                            continue;
                        for (int i = 0; i < 8; ++i)
                            L[8 * k + i] = static_cast<std::int8_t>(imax(0, imin(kMaxQ - 1, level_of(id, xval[8 * k + i]))));
                        const std::uint16_t u = pack(grid, L.data() + 8 * k);
                        if (iq_grid_find(grid.index, u) < 0)
                            find_best_neighbour(grid, u, xval.data() + 8 * k, waux.data() + 8 * k, scale, L.data() + 8 * k);
                    }
                    float sumqx = 0, sumq2 = 0;
                    sums(16, weight.data(), xval.data(), L.data(), sumqx, sumq2);
                    if (sumq2 > 0)
                        scale = sumqx / sumq2;
                }
                if (scale < 0) {
                    scale = -scale;
                    for (int k = 0; k < 2; ++k)
                        block_signs[k] = static_cast<std::uint8_t>(S ? ~block_signs[k] : (~block_signs[k]) & 127);
                }
                for (int k = 0; k < 2; ++k) {
                    const int g = iq_grid_find(grid.index, pack(grid, L.data() + 8 * k));
                    assert(g >= 0 && "snicholls::ml: IQ2 found a point not on the grid");
                    if constexpr (S) {
                        const int i8 = 2 * ib + k;
                        y[ibl].qs[i8] = static_cast<std::uint8_t>(g & 255);
                        y[ibl].qh[i8 / 4] = static_cast<std::uint8_t>(y[ibl].qh[i8 / 4] | ((g >> 8) << (2 * (i8 % 4))));
                        y[ibl].qs[32 + i8] = block_signs[k];
                    } else {
                        q2[2 * ib + k] = static_cast<std::uint16_t>(g | (block_signs[k] << 9));
                    }
                }
                scales[ib] = scale;
                max_scale = fmax(max_scale, scale);
            }
            if (max_scale == 0.0f) {
                if constexpr (!S)
                    y[ibl].qs = {};
                continue;
            }
            const float d = max_scale / 31;
            if constexpr (S) {
                const float fudged = d * 0.9875f;
                y[ibl].d = detail::quantize::fp16(fudged);
            } else {
                y[ibl].d = detail::quantize::fp16(d);
            }
            const float id = 1 / d;
            for (int ib = 0; ib < 16; ++ib) {
                const int l = scale_code(id, scales[ib]);
                if (ib % 2 == 0)
                    y[ibl].scales[ib / 2] = static_cast<std::uint8_t>(l);
                else
                    y[ibl].scales[ib / 2] = static_cast<std::uint8_t>(y[ibl].scales[ib / 2] | (l << 4));
            }
            if constexpr (!S)
                y[ibl].qs = q2;
        }
    }

    // IQ3_XXS and IQ3_S (groups of 4, kMaxQ 8): ggml's quantize_row_iq3_xxs_impl(256, ...) and
    // quantize_row_iq3_s_impl(32, ...)
    template <bool S, typename Block>
    constexpr void quantize_iq3(std::span<const float> x, std::span<const float> qw_row, std::span<Block> y)
    {
        constexpr int kMaxQ = 8;
        constexpr auto& grid = []() -> const auto& {
            if constexpr (S)
                return iq3s;
            else
                return iq3xxs;
        }();
        std::array<float, 8> scales{};
        std::array<float, 32> weight{}, xval{}, waux{};
        std::array<std::int8_t, 32> L{}, Laux{};
        std::array<bool, 8> on_grid{}, on_grid_aux{};
        std::array<std::uint8_t, 4> block_signs{};
        for (std::size_t ibl = 0; ibl < y.size(); ++ibl) {
            std::array<std::uint8_t, 104> q3{};   // IQ3_XXS: 64 grid indices, 8 words of scales and signs
            y[ibl] = Block{};
            y[ibl].d = detail::quantize::fp16(0.0f);
            float max_scale = 0;
            std::size_t qs_off = 0, signs_off = 0;
            (void)qs_off, (void)signs_off;
            const float* xbl = x.data() + 256 * ibl;
            const float two = 2 * sum_squares(xbl, 256);
            const float sigma2 = two / 256.0f;
            for (int ib = 0; ib < 8; ++ib) {
                const float* xb = xbl + 32 * ib;
                if (!qw_row.empty()) {
                    const float* qw = qw_row.data() + 256 * ibl + 32 * ib;
                    for (int i = 0; i < 32; ++i)
                        weight[i] = detail::quantize::importance_weight(qw[i], sigma2, xb[i]);
                } else {
                    for (int i = 0; i < 32; ++i)
                        weight[i] = xb[i] * xb[i];
                }
                for (int i = 0; i < 32; ++i)
                    waux[i] = detail::quantize::sqrt(weight[i]);
                for (int k = 0; k < 4; ++k)
                    block_signs[k] = S ? plain_signs(xb + 8 * k, xval.data() + 8 * k) : even_signs(xb + 8 * k, weight.data() + 8 * k, xval.data() + 8 * k);
                float max = xval[0];
                for (int i = 1; i < 32; ++i)
                    max = fmax(max, xval[i]);
                L = {};
                if (S ? max == 0.0f : max < 1e-8f) {
                    scales[ib] = 0;
                    continue;   // IQ3_S: ggml's qs and signs pointers do not advance, so the next sub-block takes these bytes
                }
                float best = 0;
                float scale = max / float(2 * kMaxQ - 1);
                on_grid.fill(!S);
                for (int is = S ? -9 : -15; is <= (S ? 9 : 15); ++is) {
                    const float step = float(is) * 0.2f;
                    const float top = float(2 * kMaxQ - 1) + step;
                    const float id = top / max;
                    const float this_scale = 1 / id;
                    for (int k = 0; k < 8; ++k) {
                        for (int i = 0; i < 4; ++i)
                            Laux[4 * k + i] = static_cast<std::int8_t>(imax(0, imin(kMaxQ - 1, level_of(id, xval[4 * k + i]))));
                        const std::uint16_t u = pack(grid, Laux.data() + 4 * k);
                        on_grid_aux[k] = true;
                        if (iq_grid_find(grid.index, u) < 0) {
                            on_grid_aux[k] = false;
                            find_best_neighbour(grid, u, xval.data() + 4 * k, waux.data() + 4 * k, this_scale, Laux.data() + 4 * k);
                        }
                    }
                    float sumqx = 0, sumq2 = 0;
                    sums(32, weight.data(), xval.data(), Laux.data(), sumqx, sumq2);
                    if (sumq2 > 0 && sumqx * sumqx > best * sumq2) {
                        scale = sumqx / sumq2;
                        best = scale * sumqx;
                        L = Laux;
                        on_grid = on_grid_aux;
                    }
                }
                int n_not_ongrid = 0;
                for (int k = 0; k < 8; ++k)
                    n_not_ongrid += on_grid[k] ? 0 : 1;
                if (n_not_ongrid > 0 && scale > 0) {
                    const float id = 1 / scale;
                    for (int k = 0; k < 8; ++k) {
                        if (!S && on_grid[k])   // IQ3_S requantises every group
                            continue;
                        std::array<std::int8_t, 4> l4{};
                        for (int i = 0; i < 4; ++i)
                            l4[i] = static_cast<std::int8_t>(imax(0, imin(kMaxQ - 1, level_of(id, xval[4 * k + i]))));
                        const std::uint16_t u = pack(grid, l4.data());
                        int g = iq_grid_find(grid.index, u);
                        if (g < 0)
                            g = find_best_neighbour(grid, u, xval.data() + 4 * k, waux.data() + 4 * k, scale, L.data() + 4 * k);
                        for (int i = 0; i < 4; ++i)
                            L[4 * k + i] = static_cast<std::int8_t>(grid.level(std::size_t(g), i));
                    }
                    float sumqx = 0, sumq2 = 0;
                    sums(32, weight.data(), xval.data(), L.data(), sumqx, sumq2);
                    if (sumq2 > 0)
                        scale = sumqx / sumq2;
                }
                if (scale < 0) {
                    scale = -scale;
                    for (int k = 0; k < 4; ++k)
                        block_signs[k] = static_cast<std::uint8_t>(S ? ~block_signs[k] : (~block_signs[k]) & 127);
                }
                for (int k = 0; k < 8; ++k) {
                    const int g = iq_grid_find(grid.index, pack(grid, L.data() + 4 * k));
                    assert(g >= 0 && "snicholls::ml: IQ3 found a point not on the grid");
                    if constexpr (S) {
                        y[ibl].qs[qs_off + std::size_t(k)] = static_cast<std::uint8_t>(g & 255);
                        y[ibl].qh[ib] = static_cast<std::uint8_t>(y[ibl].qh[ib] | ((g >> 8) << k));
                    } else {
                        q3[8 * ib + k] = static_cast<std::uint8_t>(g);
                    }
                }
                if constexpr (S) {
                    qs_off += 8;
                    for (int k = 0; k < 4; ++k)
                        y[ibl].signs[signs_off + std::size_t(k)] = block_signs[k];
                    signs_off += 4;
                } else {
                    store32(q3.data() + 64 + 4 * ib, block_signs[0] | (std::uint32_t(block_signs[1]) << 7) | (std::uint32_t(block_signs[2]) << 14)
                                                         | (std::uint32_t(block_signs[3]) << 21));
                }
                scales[ib] = scale;
                max_scale = fmax(max_scale, scale);
            }
            if (max_scale == 0.0f)
                continue;   // the block stays zero
            const float d = max_scale / 31;
            const float fudged = d * (S ? 1.033f : 1.0125f);
            y[ibl].d = detail::quantize::fp16(fudged);
            const float id = 1 / d;
            if constexpr (S) {
                for (int ib = 0; ib < 8; ib += 2)
                    y[ibl].scales[ib / 2] = static_cast<std::uint8_t>(scale_code(id, scales[ib]) | (scale_code(id, scales[ib + 1]) << 4));
            } else {
                for (int ib = 0; ib < 8; ++ib)
                    store32(q3.data() + 64 + 4 * ib, load32(q3.data() + 64 + 4 * ib) | (std::uint32_t(scale_code(id, scales[ib])) << 28));
                for (int i = 0; i < 96; ++i)
                    y[ibl].qs[i] = q3[i];
            }
        }
    }

    // IQ1's three values with the shift: -1 + δ, δ, 1 + δ and -1 - δ, -δ, 1 - δ, δ = 1/8
    inline constexpr float iq1_delta = 0.125f;
    inline constexpr std::array<float, 3> x_p{-1 + iq1_delta, iq1_delta, 1 + iq1_delta};
    inline constexpr std::array<float, 3> x_m{-1 - iq1_delta, -iq1_delta, 1 - iq1_delta};

    // The order qsort with ggml's iq1_sort_helper gives: ascending by value; equal values stay in their order
    template <std::size_t N>
    constexpr std::array<int, N> sorted_order(const float* xb)
    {
        std::array<int, N> idx{};
        for (std::size_t j = 0; j < N; ++j)
            idx[j] = int(j);
        for (std::size_t j = 1; j < N; ++j) {
            const int v = idx[j];
            std::size_t k = j;
            while (k > 0 && xb[v] < xb[idx[k - 1]]) {
                idx[k] = idx[k - 1];
                --k;
            }
            idx[k] = v;
        }
        return idx;
    }

    // ggml's iq1_find_best_neighbour2: the neighbour nearest scale · xx[level] in the weighted norm
    constexpr int iq1_neighbour(std::uint16_t u, const float* xb, const float* weight, float scale, const std::array<float, 3>& xx, std::int8_t* L)
    {
        const auto score = [&](std::size_t j) {
            float d2 = 0;
            for (int i = 0; i < 8; ++i) {
                const float q = xx[std::size_t(iq1s.level(j, i))];
                const float sq = scale * q;
                const float diff = sq - xb[i];
                const float wd = weight[i] * diff;
                const float wdd = wd * diff;
                d2 += wdd;
            }
            return d2;
        };
        int g = best_neighbour(iq1s, u, score);
        if (g < 0) {   // ggml's fallback reads past the group here; core searches the whole grid
            float best = flt_max;
            for (std::size_t j = 0; j < 2048; ++j) {
                const float e = score(j);
                if (e < best) {
                    best = e;
                    g = int(j);
                }
            }
        }
        assert(g >= 0 && "snicholls::ml: IQ1 found no grid point");
        for (int i = 0; i < 8; ++i)
            L[i] = static_cast<std::int8_t>(iq1s.level(std::size_t(g), i));
        return g;
    }

    // IQ1_S: ggml's quantize_row_iq1_s_impl
    constexpr void quantize(std::span<const float> x, std::span<const float> qw_row, std::span<block_iq1_s> y)
    {
        constexpr int bs = 32;
        std::array<float, 8> scales{};
        std::array<float, bs> weight{};
        std::array<std::int8_t, bs> L{};
        std::array<std::int8_t, 8> shifts{};
        for (std::size_t ibl = 0; ibl < y.size(); ++ibl) {
            y[ibl] = block_iq1_s{};
            y[ibl].d = detail::quantize::fp16(0.0f);
            float max_scale = 0;
            const float* xbl = x.data() + 256 * ibl;
            const float two = 2 * sum_squares(xbl, 256);
            const float sigma2 = two / 256.0f;
            for (int ib = 0; ib < 8; ++ib) {
                const float* xb = xbl + bs * ib;
                const float* qw = qw_row.data() + 256 * ibl + bs * ib;
                for (int i = 0; i < bs; ++i)
                    weight[i] = detail::quantize::importance_weight(qw[i], sigma2, xb[i]);
                float max = detail::quantize::abs(xb[0]);
                for (int i = 1; i < bs; ++i)
                    max = fmax(max, detail::quantize::abs(xb[i]));
                if (max < 1e-12f) {
                    scales[ib] = 0;
                    shifts[ib] = 1;
                    L.fill(1);
                    continue;
                }
                const auto idx = sorted_order<bs>(xb);
                std::array<float, bs + 1> sumx{}, sumw{};
                for (int j = 0; j < bs; ++j) {
                    const int i = idx[std::size_t(j)];
                    const float wx = weight[i] * xb[i];
                    sumx[j + 1] = sumx[j] + wx;
                    sumw[j + 1] = sumw[j] + weight[i];
                }
                float best_score = -flt_max, scale = max;
                int besti1 = -1, besti2 = -1, best_shift = 0;
                const auto try_split = [&](int i1, int i2, const std::array<float, 3>& v, int shift) {
                    const float a = sumx[i1] - sumx[0], b = sumx[i2] - sumx[i1], c = sumx[bs] - sumx[i2];
                    const float av = a * v[0], bv = b * v[1], cv = c * v[2];
                    const float ab = av + bv;
                    const float sumqx = ab + cv;
                    const float A = sumw[i1] - sumw[0], B = sumw[i2] - sumw[i1], C = sumw[bs] - sumw[i2];
                    const float Av = A * v[0], Bv = B * v[1], Cv = C * v[2];
                    const float Avv = Av * v[0], Bvv = Bv * v[1], Cvv = Cv * v[2];
                    const float AB = Avv + Bvv;
                    const float sumq2 = AB + Cvv;
                    if (sumq2 > 0 && sumqx * sumqx > best_score * sumq2) {
                        scale = sumqx / sumq2;
                        best_score = scale * sumqx;
                        besti1 = i1;
                        besti2 = i2;
                        best_shift = shift;
                    }
                };
                for (int i1 = 0; i1 <= bs; ++i1)
                    for (int i2 = i1; i2 <= bs; ++i2) {
                        try_split(i1, i2, x_p, 1);
                        try_split(i1, i2, x_m, -1);
                    }
                if (besti1 < 0 || besti2 < 0 || best_shift == 0) {
                    scales[ib] = 0;
                    shifts[ib] = 1;
                    L.fill(1);
                    continue;
                }
                for (int j = 0; j < bs; ++j)
                    L[std::size_t(idx[std::size_t(j)])] = static_cast<std::int8_t>(j < besti1 ? 0 : j < besti2 ? 1 : 2);
                if (scale < 0) {
                    for (int j = 0; j < bs; ++j)
                        L[j] = static_cast<std::int8_t>(2 - L[j]);
                    scale = -scale;
                    best_shift = -best_shift;
                }
                bool all_on_grid = true;
                const auto& xx = best_shift == 1 ? x_p : x_m;
                std::array<int, 4> index{};
                for (int k = 0; k < 4; ++k) {
                    const std::uint16_t u = pack(iq1s, L.data() + 8 * k);
                    int g = iq_grid_find(iq1s.index, u);
                    if (g < 0) {
                        all_on_grid = false;
                        g = iq1_neighbour(u, xb + 8 * k, weight.data() + 8 * k, scale, xx, L.data() + 8 * k);
                    }
                    index[k] = g;
                }
                if (!all_on_grid) {
                    float sumqx = 0, sumq2 = 0;
                    for (int k = 0; k < 4; ++k)
                        for (int j = 0; j < 8; ++j) {
                            const float w = weight[8 * k + j];
                            const float q = xx[std::size_t(iq1s.level(std::size_t(index[k]), j))];
                            const float wq = w * q;
                            const float wqx = wq * xb[8 * k + j];
                            sumqx += wqx;
                            const float wqq = wq * q;
                            sumq2 += wqq;
                        }
                    if (sumqx > 0 && sumq2 > 0)
                        scale = sumqx / sumq2;
                }
                std::uint16_t h = 0;
                for (int k = 0; k < 4; ++k) {
                    y[ibl].qs[4 * ib + k] = static_cast<std::uint8_t>(index[k] & 255);
                    h = static_cast<std::uint16_t>(h | ((index[k] >> 8) << (3 * k)));
                }
                y[ibl].qh[ib] = h;
                scales[ib] = scale;
                shifts[ib] = static_cast<std::int8_t>(best_shift);
                max_scale = fmax(max_scale, scale);
            }
            if (max_scale == 0.0f)
                continue;
            const float d = max_scale / 15;
            const float fudged = d * 1.125f;
            y[ibl].d = detail::quantize::fp16(fudged);
            const float id = 1 / d;
            for (int ib = 0; ib < 8; ++ib) {
                int l = imax(0, imin(7, level_of(id, scales[ib])));
                if (shifts[ib] == -1)
                    l |= 8;
                y[ibl].qh[ib] = static_cast<std::uint16_t>(y[ibl].qh[ib] | (l << 12));
            }
        }
    }

    // IQ1_M: ggml's quantize_row_iq1_m_impl
    constexpr void quantize(std::span<const float> x, std::span<const float> qw_row, std::span<block_iq1_m> y)
    {
        constexpr int bs = 16;
        constexpr std::array<std::uint8_t, 4> masks{0x00, 0x80, 0x08, 0x88};
        std::array<float, 16> scales{};
        std::array<float, bs> weight{};
        std::array<std::int8_t, bs> L{};
        std::array<std::int8_t, 16> shifts{};
        const auto weights_of = [&](const float* xb, std::size_t offset, float sigma2) {
            if (!qw_row.empty()) {
                for (int i = 0; i < bs; ++i)
                    weight[i] = detail::quantize::importance_weight(qw_row[offset + std::size_t(i)], sigma2, xb[i]);
            } else {
                for (int i = 0; i < bs; ++i)
                    weight[i] = xb[i] * xb[i];
            }
        };
        const auto shift_values = [](int best_k, int k) -> const std::array<float, 3>& {
            if (k == 0)
                return best_k < 2 ? x_p : x_m;
            return best_k % 2 == 0 ? x_p : x_m;
        };
        for (std::size_t ibl = 0; ibl < y.size(); ++ibl) {
            y[ibl] = block_iq1_m{};
            float max_scale = 0;
            const float* xbl = x.data() + 256 * ibl;
            const float two = 2 * sum_squares(xbl, 256);
            const float sigma2 = two / 256.0f;
            for (int ib = 0; ib < 16; ++ib) {
                const float* xb = xbl + bs * ib;
                weights_of(xb, 256 * ibl + std::size_t(bs * ib), sigma2);
                float max = detail::quantize::abs(xb[0]);
                for (int i = 1; i < bs; ++i)
                    max = fmax(max, detail::quantize::abs(xb[i]));
                if (max < 1e-7f) {
                    scales[ib] = 0;
                    shifts[ib] = 0;
                    L.fill(1);
                    continue;
                }
                const auto idx = sorted_order<bs>(xb);
                float best_score = -flt_max, scale = max;
                int besti1 = -1, besti2 = -1, best_k = -1;
                std::array<std::array<float, bs + 1>, 2> px{}, pw{};
                for (int j = 0; j < bs; ++j) {
                    const int i = idx[std::size_t(j)];
                    const int h = i < bs / 2 ? 0 : 1;
                    const float wx = weight[i] * xb[i];
                    px[h][j + 1] = px[h][j] + wx;
                    px[1 - h][j + 1] = px[1 - h][j];
                    pw[h][j + 1] = pw[h][j] + weight[i];
                    pw[1 - h][j + 1] = pw[1 - h][j];
                }
                const std::array<float, 2> txs{px[0][bs], px[1][bs]};
                const std::array<float, 2> tws{pw[0][bs], pw[1][bs]};
                const float p2 = x_p[2], m2 = x_m[2];
                const auto diff_sq = [](float a, float b) {
                    const float a2 = a * a;
                    const float b2 = b * b;
                    return a2 - b2;
                };
                const float cp1 = diff_sq(x_p[0], x_p[1]), cp2 = diff_sq(x_p[1], x_p[2]);
                const float cm1 = diff_sq(x_m[0], x_m[1]), cm2 = diff_sq(x_m[1], x_m[2]);
                for (int i1 = 0; i1 <= bs; ++i1)
                    for (int i2 = i1; i2 <= bs; ++i2) {
                        std::array<float, 2> qx_p{}, qx_m{}, q2_p{}, q2_m{};
                        for (int h = 0; h < 2; ++h) {
                            const float sx = px[h][i1] + px[h][i2];
                            const float tp = txs[h] * p2;
                            qx_p[h] = tp - sx;
                            const float tm = txs[h] * m2;
                            qx_m[h] = tm - sx;
                            const auto q2 = [&](float v2, float c1, float c2) {
                                const float t = tws[h] * v2;
                                const float tt = t * v2;
                                const float a = pw[h][i1] * c1;
                                const float s1 = tt + a;
                                const float b = pw[h][i2] * c2;
                                return s1 + b;
                            };
                            q2_p[h] = q2(p2, cp1, cp2);
                            q2_m[h] = q2(m2, cm1, cm2);
                        }
                        const std::array<float, 4> sumqx{qx_p[0] + qx_p[1], qx_p[0] + qx_m[1], qx_m[0] + qx_p[1], qx_m[0] + qx_m[1]};
                        const std::array<float, 4> sumq2{q2_p[0] + q2_p[1], q2_p[0] + q2_m[1], q2_m[0] + q2_p[1], q2_m[0] + q2_m[1]};
                        for (int k = 0; k < 4; ++k)
                            if (sumq2[k] > 0 && sumqx[k] * sumqx[k] > best_score * sumq2[k]) {
                                scale = sumqx[k] / sumq2[k];
                                best_score = scale * sumqx[k];
                                besti1 = i1;
                                besti2 = i2;
                                best_k = k;
                            }
                    }
                if (besti1 < 0 || besti2 < 0 || best_k < 0) {
                    scales[ib] = 0;
                    shifts[ib] = 0;
                    L.fill(1);
                    continue;
                }
                for (int j = 0; j < bs; ++j)
                    L[std::size_t(idx[std::size_t(j)])] = static_cast<std::int8_t>(j < besti1 ? 0 : j < besti2 ? 1 : 2);
                if (scale < 0) {
                    for (int j = 0; j < bs; ++j)
                        L[j] = static_cast<std::int8_t>(2 - L[j]);
                    scale = -scale;
                    best_k = 3 - best_k;
                }
                bool all_on_grid = true;
                std::array<int, 2> index{};
                for (int k = 0; k < 2; ++k) {
                    const std::uint16_t u = pack(iq1s, L.data() + 8 * k);
                    int g = iq_grid_find(iq1s.index, u);
                    if (g < 0) {
                        all_on_grid = false;
                        g = iq1_neighbour(u, xb + 8 * k, weight.data() + 8 * k, scale, shift_values(best_k, k), L.data() + 8 * k);
                    }
                    index[k] = g;
                }
                if (!all_on_grid) {
                    float sumqx = 0, sumq2 = 0;
                    for (int k = 0; k < 2; ++k) {
                        const auto& xx = shift_values(best_k, k);
                        for (int j = 0; j < 8; ++j) {
                            const float w = weight[8 * k + j];
                            const float q = xx[std::size_t(iq1s.level(std::size_t(index[k]), j))];
                            const float wq = w * q;
                            const float wqx = wq * xb[8 * k + j];
                            sumqx += wqx;
                            const float wqq = wq * q;
                            sumq2 += wqq;
                        }
                    }
                    if (sumqx > 0 && sumq2 > 0)
                        scale = sumqx / sumq2;
                }
                y[ibl].qs[2 * ib] = static_cast<std::uint8_t>(index[0] & 255);
                y[ibl].qs[2 * ib + 1] = static_cast<std::uint8_t>(index[1] & 255);
                y[ibl].qh[ib] = static_cast<std::uint8_t>((index[0] >> 8) | ((index[1] >> 8) << 4));
                scales[ib] = scale;
                shifts[ib] = static_cast<std::int8_t>(best_k);
                max_scale = fmax(max_scale, scale);
            }
            if (max_scale == 0.0f)
                continue;
            std::array<std::uint16_t, 4> sc{};
            float d = max_scale / 15;
            const float id = 1 / d;
            float sumqx = 0, sumq2 = 0;
            for (int ib = 0; ib < 16; ++ib) {
                const int l = imax(0, imin(7, level_of(id, scales[ib])));
                sc[ib / 4] = static_cast<std::uint16_t>(sc[ib / 4] | (l << (3 * (ib % 4))));
                y[ibl].qh[ib] = static_cast<std::uint8_t>(y[ibl].qh[ib] | masks[std::size_t(shifts[ib])]);
                const float* xb = xbl + bs * ib;
                weights_of(xb, 256 * ibl + std::size_t(bs * ib), sigma2);
                for (int k = 0; k < 2; ++k) {
                    const auto& xx = shift_values(shifts[ib], k);
                    const int g = y[ibl].qs[2 * ib + k] + ((y[ibl].qh[ib] << (8 - 4 * k)) & 0x700);
                    for (int j = 0; j < 8; ++j) {
                        const float w = weight[8 * k + j];
                        const float v = xx[std::size_t(iq1s.level(std::size_t(g), j))];
                        const float q = v * float(2 * l + 1);
                        const float wq = w * q;
                        const float wqx = wq * xb[8 * k + j];
                        sumqx += wqx;
                        const float wqq = wq * q;
                        sumq2 += wqq;
                    }
                }
            }
            if (sumq2 > 0)
                d = sumqx / sumq2;
            const float fudged = d * 1.1125f;
            const std::uint16_t s = detail::quantize::fp16(fudged).bits;
            sc[0] = static_cast<std::uint16_t>(sc[0] | ((s & 0x000f) << 12));
            sc[1] = static_cast<std::uint16_t>(sc[1] | ((s & 0x00f0) << 8));
            sc[2] = static_cast<std::uint16_t>(sc[2] | ((s & 0x0f00) << 4));
            sc[3] = static_cast<std::uint16_t>(sc[3] | (s & 0xf000));
            for (int k = 0; k < 4; ++k) {
                y[ibl].scales[2 * k] = static_cast<std::uint8_t>(sc[k]);
                y[ibl].scales[2 * k + 1] = static_cast<std::uint8_t>(sc[k] >> 8);
            }
        }
    }

    // ggml's best_index_iq4nl: the nearest IQ4_NL value's index, with mid the midpoints
    constexpr int best_index_iq4nl(const std::array<float, 15>& mid, float x)
    {
        int l = 15;
        for (int k = 0; k < 15; ++k)
            l -= x < mid[k] ? 1 : 0;
        return l;
    }

    inline constexpr std::array<float, 15> iq4nl_mid = [] {
        std::array<float, 15> m{};
        for (std::size_t k = 0; k < 15; ++k)
            m[k] = 0.5f * float(iq4nl_values[k] + iq4nl_values[k + 1]);
        return m;
    }();

    // ggml's quantize_row_iq4_nl_impl for one super-block (32 for IQ4_NL, 256 for IQ4_XS); L persists between
    // calls, as ggml's buffer does
    constexpr void iq4(int super_block_size, const float* x, float16& dh, std::uint8_t* q4, std::uint16_t* scales_h, std::uint8_t* scales_l,
                       std::uint8_t* L, const float* quant_weights, int ntry)
    {
        constexpr int block_size = 32;
        float sigma2 = sum_squares(x, super_block_size);
        sigma2 *= 2.0f / float(super_block_size);
        for (int i = 0; i < super_block_size / 2; ++i)
            q4[i] = 0;
        dh = detail::quantize::fp16(0.0f);
        const auto& mid = iq4nl_mid;
        std::array<float, 8> scales{};
        std::array<float, block_size> weight{};
        float max_scale = 0, amax_scale = 0;
        const int nb = super_block_size / block_size;
        for (int ib = 0; ib < nb; ++ib) {
            const float* xb = x + ib * block_size;
            std::uint8_t* Lb = L + ib * block_size;
            if (quant_weights) {
                const float* qw = quant_weights + ib * block_size;
                for (int j = 0; j < block_size; ++j)
                    weight[j] = detail::quantize::importance_weight(qw[j], sigma2, xb[j]);
            } else {
                for (int j = 0; j < block_size; ++j)
                    weight[j] = xb[j] * xb[j];
            }
            float amax = 0, max = 0;
            for (int j = 0; j < block_size; ++j) {
                const float ax = detail::quantize::abs(xb[j]);
                if (ax > amax) {
                    amax = ax;
                    max = xb[j];
                }
            }
            if (amax < detail::quantize::group_max_eps) {
                scales[ib] = 0;
                continue;
            }
            const float v0 = float(iq4nl_values[0]);
            float d = ntry > 0 ? -max / v0 : max / v0;
            const int nt = ntry >= 0 ? 2 * ntry + 2 : 1;
            std::array<float, 16> idt{}, sqx{}, sq2{};
            idt[0] = 1 / d;
            for (int t = 1; t < nt; ++t)
                idt[t] = float(t - 1 - ntry + iq4nl_values[0]) / max;
            for (int j = 0; j < block_size; ++j) {
                const float v = idt[0] * xb[j];
                Lb[j] = static_cast<std::uint8_t>(best_index_iq4nl(mid, v));
            }
            for (int j = 0; j < block_size; ++j) {
                const float w = weight[j];
                for (int t = 0; t < nt; ++t) {
                    const float v = idt[t] * xb[j];
                    const float q = float(iq4nl_values[std::size_t(best_index_iq4nl(mid, v))]);
                    const float wq = w * q;
                    const float wqx = wq * xb[j];
                    sqx[t] += wqx;
                    const float wqq = wq * q;
                    sq2[t] += wqq;
                }
            }
            d = sq2[0] > 0 ? sqx[0] / sq2[0] : 0.0f;
            float best = d * sqx[0];
            for (int t = 1; t < nt; ++t)
                if (sq2[t] > 0 && sqx[t] * sqx[t] > best * sq2[t]) {
                    d = sqx[t] / sq2[t];
                    best = d * sqx[t];
                }
            scales[ib] = d;
            const float abs_d = detail::quantize::abs(d);
            if (abs_d > amax_scale) {
                amax_scale = abs_d;
                max_scale = d;
            }
        }
        if (nb > 1) {
            *scales_h = 0;
            const float d = -max_scale / 32;
            dh = detail::quantize::fp16(d);
            const float id = d != 0.0f ? 1 / d : 0.0f;
            for (int ib = 0; ib < nb; ++ib) {
                const float v = id * scales[ib];
                int l = imax(-32, imin(31, nearest_int(v)));
                const float dl = d * float(l);
                const float idl = dl != 0.0f ? 1 / dl : 0.0f;
                std::uint8_t* Lb = L + ib * block_size;
                const float* xb = x + ib * block_size;
                for (int j = 0; j < block_size; ++j) {
                    const float s = idl * xb[j];
                    Lb[j] = static_cast<std::uint8_t>(best_index_iq4nl(mid, s));
                }
                l += 32;
                const auto l_l = static_cast<std::uint8_t>(l & 0xf);
                const auto l_h = static_cast<std::uint8_t>(l >> 4);
                if (ib % 2 == 0)
                    scales_l[ib / 2] = l_l;
                else
                    scales_l[ib / 2] = static_cast<std::uint8_t>(scales_l[ib / 2] | (l_l << 4));
                *scales_h = static_cast<std::uint16_t>(*scales_h | (l_h << (2 * (ib % 8))));
            }
        } else {
            dh = detail::quantize::fp16(scales[0]);
            if (ntry > 0) {
                const float id = scales[0] != 0.0f ? 1 / scales[0] : 0.0f;
                for (int j = 0; j < super_block_size; ++j) {
                    const float s = id * x[j];
                    L[j] = static_cast<std::uint8_t>(best_index_iq4nl(mid, s));
                }
            }
        }
        for (int i = 0; i < super_block_size / 32; ++i)
            for (int j = 0; j < 16; ++j)
                q4[16 * i + j] = static_cast<std::uint8_t>(L[32 * i + j] | (L[32 * i + 16 + j] << 4));
    }

    constexpr void check(std::size_t elements, std::size_t blocks, std::size_t n, std::size_t importance, bool required)
    {
        detail::quantize::check(elements, blocks, n, importance);
        assert((!required || importance != 0) && "snicholls::ml: ggml quantises this IQ format only with an importance matrix");
        (void)required;
    }
} // namespace detail::iq

/// \brief Quantises one row to IQ2_XXS, as ggml's `quantize_iq2_xxs`: per 8 magnitudes the best of the 256-point grid, searched over 13 scales.
/// \param x The row, `256 · y.size()` floats.
/// \param importance One weight per number: required (ggml refuses to quantise IQ2_XXS without an importance matrix).
/// \param y Receives the blocks.
constexpr void quantize_row(std::span<const float> x, std::span<const float> importance, std::span<block_iq2_xxs> y)
{
    detail::iq::check(256, y.size(), x.size(), importance.size(), true);
    detail::iq::quantize(x, importance, y);
}

/// \brief Quantises one row to IQ2_XS, as ggml's `quantize_iq2_xs`.
/// \param x The row, `256 · y.size()` floats.
/// \param importance One weight per number: required, as in ggml.
/// \param y Receives the blocks.
constexpr void quantize_row(std::span<const float> x, std::span<const float> importance, std::span<block_iq2_xs> y)
{
    detail::iq::check(256, y.size(), x.size(), importance.size(), true);
    detail::iq::quantize_iq2_16<false>(x, importance, y);
}

/// \brief Quantises one row to IQ2_S, as ggml's `quantize_iq2_s`.
/// \param x The row, `256 · y.size()` floats.
/// \param importance Empty, or one weight per number.
/// \param y Receives the blocks.
constexpr void quantize_row(std::span<const float> x, std::span<const float> importance, std::span<block_iq2_s> y)
{
    detail::iq::check(256, y.size(), x.size(), importance.size(), false);
    detail::iq::quantize_iq2_16<true>(x, importance, y);
}

/// \brief Quantises one row to IQ3_XXS, as ggml's `quantize_iq3_xxs` (and, with no importance row, `quantize_row_iq3_xxs_ref`).
/// \param x The row, `256 · y.size()` floats.
/// \param importance Empty, or one weight per number.
/// \param y Receives the blocks.
constexpr void quantize_row(std::span<const float> x, std::span<const float> importance, std::span<block_iq3_xxs> y)
{
    detail::iq::check(256, y.size(), x.size(), importance.size(), false);
    detail::iq::quantize_iq3<false>(x, importance, y);
}

/// \brief Quantises one row to IQ3_S, as ggml's `quantize_iq3_s`.
/// \param x The row, `256 · y.size()` floats.
/// \param importance Empty, or one weight per number.
/// \param y Receives the blocks.
constexpr void quantize_row(std::span<const float> x, std::span<const float> importance, std::span<block_iq3_s> y)
{
    detail::iq::check(256, y.size(), x.size(), importance.size(), false);
    detail::iq::quantize_iq3<true>(x, importance, y);
}

/// \brief Quantises one row to IQ1_S, as ggml's `quantize_iq1_s`: per 32 the exact best split into the three values, then the nearest grid points.
/// \param x The row, `256 · y.size()` floats.
/// \param importance One weight per number: required, as in ggml.
/// \param y Receives the blocks.
constexpr void quantize_row(std::span<const float> x, std::span<const float> importance, std::span<block_iq1_s> y)
{
    detail::iq::check(256, y.size(), x.size(), importance.size(), true);
    detail::iq::quantize(x, importance, y);
}

/// \brief Quantises one row to IQ1_M, as ggml's `quantize_iq1_m`.
/// \param x The row, `256 · y.size()` floats.
/// \param importance Empty, or one weight per number.
/// \param y Receives the blocks.
constexpr void quantize_row(std::span<const float> x, std::span<const float> importance, std::span<block_iq1_m> y)
{
    detail::iq::check(256, y.size(), x.size(), importance.size(), false);
    detail::iq::quantize(x, importance, y);
}

/// \brief Quantises one row to IQ4_NL, as ggml's `quantize_iq4_nl`: per 32 a scale searched over 16 candidates.
/// \param x The row, `32 · y.size()` floats.
/// \param importance Empty, or one weight per number.
/// \param y Receives the blocks.
constexpr void quantize_row(std::span<const float> x, std::span<const float> importance, std::span<block_iq4_nl> y)
{
    detail::iq::check(32, y.size(), x.size(), importance.size(), false);
    std::array<std::uint8_t, 32> L{};
    std::uint16_t unused_h = 0;
    for (std::size_t i = 0; i < y.size(); ++i)
        detail::iq::iq4(32, x.data() + 32 * i, y[i].d, y[i].qs.data(), &unused_h, nullptr, L.data(),
                        importance.empty() ? nullptr : importance.data() + 32 * i, 7);
}

/// \brief Quantises one row to IQ4_NL as ggml's `quantize_row_iq4_nl_ref`: the scale amax / -127 refined once, no search, as ggml quantises IQ4_NL activations and caches.
/// \param x The row, `32 · y.size()` floats.
/// \param y Receives the blocks.
constexpr void quantize_row_reference(std::span<const float> x, std::span<block_iq4_nl> y)
{
    detail::quantize::check(32, y.size(), x.size(), 0);
    std::array<std::uint8_t, 32> L{};
    std::uint16_t unused_h = 0;
    for (std::size_t i = 0; i < y.size(); ++i)
        detail::iq::iq4(32, x.data() + 32 * i, y[i].d, y[i].qs.data(), &unused_h, nullptr, L.data(), nullptr, -1);
}

/// \brief Quantises one row to IQ4_XS, as ggml's `quantize_iq4_xs`.
/// \param x The row, `256 · y.size()` floats.
/// \param importance Empty, or one weight per number.
/// \param y Receives the blocks.
constexpr void quantize_row(std::span<const float> x, std::span<const float> importance, std::span<block_iq4_xs> y)
{
    detail::iq::check(256, y.size(), x.size(), importance.size(), false);
    std::array<std::uint8_t, 256> L{};
    for (std::size_t i = 0; i < y.size(); ++i)
        detail::iq::iq4(256, x.data() + 256 * i, y[i].d, y[i].qs.data(), &y[i].scales_h, y[i].scales_l.data(), L.data(),
                        importance.empty() ? nullptr : importance.data() + 256 * i, 7);
}

/// \brief Dequantises IQ2_XXS blocks, as ggml's `dequantize_row_iq2_xxs`.
/// \param x The blocks.
/// \param y Receives `256 · x.size()` floats.
constexpr void dequantize_row(std::span<const block_iq2_xxs> x, std::span<float> y)
{
    assert(y.size() == 256 * x.size());
    std::size_t o = 0;
    for (const auto& b : x) {
        const float d = to_float(b.d);
        for (std::size_t ib = 0; ib < 8; ++ib) {
            const std::uint32_t a0 = b.qs[4 * ib] | (std::uint32_t(b.qs[4 * ib + 1]) << 16);
            const std::uint32_t a1 = b.qs[4 * ib + 2] | (std::uint32_t(b.qs[4 * ib + 3]) << 16);
            const float s = 0.5f + float(a1 >> 28);
            const float ds = d * s;
            const float db = ds * 0.25f;
            for (int l = 0; l < 4; ++l) {
                const std::uint64_t grid = iq2xxs_grid[(a0 >> (8 * l)) & 255];
                const std::uint8_t signs = iq2_signs[(a1 >> (7 * l)) & 127];
                for (int j = 0; j < 8; ++j) {
                    const float g = db * float(detail::iq::grid_byte(grid, j));
                    y[o++] = g * (detail::iq::negative(signs, j) ? -1.0f : 1.0f);
                }
            }
        }
    }
}

/// \brief Dequantises IQ2_XS blocks, as ggml's `dequantize_row_iq2_xs`.
/// \param x The blocks.
/// \param y Receives `256 · x.size()` floats.
constexpr void dequantize_row(std::span<const block_iq2_xs> x, std::span<float> y)
{
    assert(y.size() == 256 * x.size());
    std::size_t o = 0;
    for (const auto& b : x) {
        const float d = to_float(b.d);
        for (std::size_t ib = 0; ib < 8; ++ib) {
            float db[2];
            for (int h = 0; h < 2; ++h) {
                const float s = 0.5f + float((b.scales[ib] >> (4 * h)) & 0xf);
                const float ds = d * s;
                db[h] = ds * 0.25f;
            }
            for (std::size_t l = 0; l < 4; ++l) {
                const std::uint16_t q = b.qs[4 * ib + l];
                const std::uint64_t grid = iq2xs_grid[q & 511];
                const std::uint8_t signs = iq2_signs[q >> 9];
                for (int j = 0; j < 8; ++j) {
                    const float g = db[l / 2] * float(detail::iq::grid_byte(grid, j));
                    y[o++] = g * (detail::iq::negative(signs, j) ? -1.0f : 1.0f);
                }
            }
        }
    }
}

/// \brief Dequantises IQ2_S blocks, as ggml's `dequantize_row_iq2_s`.
/// \param x The blocks.
/// \param y Receives `256 · x.size()` floats.
constexpr void dequantize_row(std::span<const block_iq2_s> x, std::span<float> y)
{
    assert(y.size() == 256 * x.size());
    std::size_t o = 0;
    for (const auto& b : x) {
        const float d = to_float(b.d);
        for (std::size_t ib = 0; ib < 8; ++ib) {
            float db[2];
            for (int h = 0; h < 2; ++h) {
                const float s = 0.5f + float((b.scales[ib] >> (4 * h)) & 0xf);
                const float ds = d * s;
                db[h] = ds * 0.25f;
            }
            for (std::size_t l = 0; l < 4; ++l) {
                const std::uint64_t grid = iq2s_grid[b.qs[4 * ib + l] | ((b.qh[ib] << (8 - 2 * l)) & 0x300)];
                const std::uint8_t signs = b.qs[32 + 4 * ib + l];
                for (int j = 0; j < 8; ++j) {
                    const float g = db[l / 2] * float(detail::iq::grid_byte(grid, j));
                    y[o++] = g * (detail::iq::negative(signs, j) ? -1.0f : 1.0f);
                }
            }
        }
    }
}

/// \brief Dequantises IQ3_XXS blocks, as ggml's `dequantize_row_iq3_xxs`.
/// \param x The blocks.
/// \param y Receives `256 · x.size()` floats.
constexpr void dequantize_row(std::span<const block_iq3_xxs> x, std::span<float> y)
{
    assert(y.size() == 256 * x.size());
    std::size_t o = 0;
    for (const auto& b : x) {
        const float d = to_float(b.d);
        for (std::size_t ib = 0; ib < 8; ++ib) {
            const std::uint32_t aux = detail::iq::load32(b.qs.data() + 64 + 4 * ib);
            const float s = 0.5f + float(aux >> 28);
            const float ds = d * s;
            const float db = ds * 0.5f;
            for (std::size_t l = 0; l < 4; ++l) {
                const std::uint8_t signs = iq2_signs[(aux >> (7 * l)) & 127];
                const std::uint32_t g1 = iq3xxs_grid[b.qs[8 * ib + 2 * l]];
                const std::uint32_t g2 = iq3xxs_grid[b.qs[8 * ib + 2 * l + 1]];
                for (int j = 0; j < 4; ++j) {
                    const float a = db * float(detail::iq::grid_byte(g1, j));
                    y[o + std::size_t(j)] = a * (detail::iq::negative(signs, j) ? -1.0f : 1.0f);
                    const float c = db * float(detail::iq::grid_byte(g2, j));
                    y[o + std::size_t(j) + 4] = c * (detail::iq::negative(signs, j + 4) ? -1.0f : 1.0f);
                }
                o += 8;
            }
        }
    }
}

/// \brief Dequantises IQ3_S blocks, as ggml's `dequantize_row_iq3_s`.
/// \param x The blocks.
/// \param y Receives `256 · x.size()` floats.
constexpr void dequantize_row(std::span<const block_iq3_s> x, std::span<float> y)
{
    assert(y.size() == 256 * x.size());
    std::size_t o = 0;
    for (const auto& b : x) {
        const float d = to_float(b.d);
        for (std::size_t ib = 0; ib < 8; ++ib) {
            const float db = d * float(1 + 2 * ((b.scales[ib / 2] >> (4 * (ib % 2))) & 0xf));
            for (std::size_t l = 0; l < 4; ++l) {
                const std::uint32_t g1 = iq3s_grid[b.qs[8 * ib + 2 * l] | ((b.qh[ib] << (8 - 2 * l)) & 256)];
                const std::uint32_t g2 = iq3s_grid[b.qs[8 * ib + 2 * l + 1] | ((b.qh[ib] << (7 - 2 * l)) & 256)];
                const std::uint8_t signs = b.signs[4 * ib + l];
                for (int j = 0; j < 4; ++j) {
                    const float a = db * float(detail::iq::grid_byte(g1, j));
                    y[o + std::size_t(j)] = a * (detail::iq::negative(signs, j) ? -1.0f : 1.0f);
                    const float c = db * float(detail::iq::grid_byte(g2, j));
                    y[o + std::size_t(j) + 4] = c * (detail::iq::negative(signs, j + 4) ? -1.0f : 1.0f);
                }
                o += 8;
            }
        }
    }
}

/// \brief Dequantises IQ1_S blocks, as ggml's `dequantize_row_iq1_s`.
/// \param x The blocks.
/// \param y Receives `256 · x.size()` floats.
constexpr void dequantize_row(std::span<const block_iq1_s> x, std::span<float> y)
{
    assert(y.size() == 256 * x.size());
    std::size_t o = 0;
    for (const auto& b : x) {
        const float d = to_float(b.d);
        for (std::size_t ib = 0; ib < 8; ++ib) {
            const float dl = d * float(2 * ((b.qh[ib] >> 12) & 7) + 1);
            const float delta = b.qh[ib] & 0x8000 ? -detail::iq::iq1_delta : detail::iq::iq1_delta;
            for (std::size_t l = 0; l < 4; ++l) {
                const std::uint64_t grid = iq1s_grid[b.qs[4 * ib + l] | (((b.qh[ib] >> (3 * l)) & 7) << 8)];
                for (int j = 0; j < 8; ++j) {
                    const float v = float(detail::iq::grid_sbyte(grid, j)) + delta;
                    y[o++] = dl * v;
                }
            }
        }
    }
}

namespace detail::iq
{
    constexpr float16 iq1m_scale(const block_iq1_m& b, std::array<std::uint16_t, 4>& sc)
    {
        for (std::size_t k = 0; k < 4; ++k)
            sc[k] = static_cast<std::uint16_t>(b.scales[2 * k] | (b.scales[2 * k + 1] << 8));
        return float16::from_bits(std::uint32_t(sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000));
    }
} // namespace detail::iq

/// \brief Dequantises IQ1_M blocks, as ggml's `dequantize_row_iq1_m`.
/// \param x The blocks.
/// \param y Receives `256 · x.size()` floats.
constexpr void dequantize_row(std::span<const block_iq1_m> x, std::span<float> y)
{
    assert(y.size() == 256 * x.size());
    std::size_t o = 0;
    for (const auto& b : x) {
        std::array<std::uint16_t, 4> sc{};
        const float d = to_float(detail::iq::iq1m_scale(b, sc));
        for (std::size_t ib = 0; ib < 8; ++ib) {
            const float dl1 = d * float(2 * ((sc[ib / 2] >> (6 * (ib % 2))) & 0x7) + 1);
            const float dl2 = d * float(2 * ((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 0x7) + 1);
            const std::uint8_t h0 = b.qh[2 * ib], h1 = b.qh[2 * ib + 1];
            const std::array<int, 4> idx{b.qs[4 * ib] | ((h0 << 8) & 0x700), b.qs[4 * ib + 1] | ((h0 << 4) & 0x700),
                                         b.qs[4 * ib + 2] | ((h1 << 8) & 0x700), b.qs[4 * ib + 3] | ((h1 << 4) & 0x700)};
            const float dp = detail::iq::iq1_delta;
            const std::array<float, 4> delta{h0 & 0x08 ? -dp : dp, h0 & 0x80 ? -dp : dp, h1 & 0x08 ? -dp : dp, h1 & 0x80 ? -dp : dp};
            for (std::size_t l = 0; l < 4; ++l) {
                const std::uint64_t grid = iq1s_grid[std::size_t(idx[l])];
                for (int j = 0; j < 8; ++j) {
                    const float v = float(detail::iq::grid_sbyte(grid, j)) + delta[l];
                    y[o++] = (l < 2 ? dl1 : dl2) * v;
                }
            }
        }
    }
}

/// \brief Dequantises IQ4_NL blocks, as ggml's `dequantize_row_iq4_nl`.
/// \param x The blocks.
/// \param y Receives `32 · x.size()` floats.
constexpr void dequantize_row(std::span<const block_iq4_nl> x, std::span<float> y)
{
    assert(y.size() == 32 * x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        const float d = to_float(x[i].d);
        for (std::size_t j = 0; j < 16; ++j) {
            y[32 * i + j] = d * float(iq4nl_values[x[i].qs[j] & 0xf]);
            y[32 * i + j + 16] = d * float(iq4nl_values[x[i].qs[j] >> 4]);
        }
    }
}

/// \brief Dequantises IQ4_XS blocks, as ggml's `dequantize_row_iq4_xs`.
/// \param x The blocks.
/// \param y Receives `256 · x.size()` floats.
constexpr void dequantize_row(std::span<const block_iq4_xs> x, std::span<float> y)
{
    assert(y.size() == 256 * x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        const float d = to_float(x[i].d);
        for (std::size_t ib = 0; ib < 8; ++ib) {
            const int ls = ((x[i].scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf) | (((x[i].scales_h >> (2 * ib)) & 3) << 4);
            const float dl = d * float(ls - 32);
            for (std::size_t j = 0; j < 16; ++j) {
                y[256 * i + 32 * ib + j] = dl * float(iq4nl_values[x[i].qs[16 * ib + j] & 0xf]);
                y[256 * i + 32 * ib + j + 16] = dl * float(iq4nl_values[x[i].qs[16 * ib + j] >> 4]);
            }
        }
    }
}

/// \brief The dot product of an IQ2_XXS row with a Q8_K row, as ggml's `ggml_vec_dot_iq2_xxs_q8_K`.
/// \param x The weights.
/// \param y The activations.
/// \return `1/8 · sum over blocks of d_x d_y · sum over 32 of (2s + 1) · (sum ±g · q)`.
constexpr float dot(std::span<const block_iq2_xxs> x, std::span<const block_q8_k> y)
{
    assert(x.size() == y.size());
    float sumf = 0.0f;
    for (std::size_t i = 0; i < x.size(); ++i) {
        const float d = to_float(x[i].d) * y[i].d;
        std::int32_t bsum = 0;
        for (std::size_t ib = 0; ib < 8; ++ib) {
            const std::int8_t* q8 = y[i].qs.data() + 32 * ib;
            const std::uint32_t a0 = x[i].qs[4 * ib] | (std::uint32_t(x[i].qs[4 * ib + 1]) << 16);
            const std::uint32_t a1 = x[i].qs[4 * ib + 2] | (std::uint32_t(x[i].qs[4 * ib + 3]) << 16);
            const auto ls = static_cast<std::int32_t>(2 * (a1 >> 28) + 1);
            std::int32_t sumi = 0;
            for (std::size_t l = 0; l < 4; ++l) {
                const std::uint64_t grid = iq2xxs_grid[(a0 >> (8 * l)) & 255];
                const std::uint8_t signs = iq2_signs[(a1 >> (7 * l)) & 127];
                for (int j = 0; j < 8; ++j) {
                    sumi += detail::iq::grid_byte(grid, j) * q8[8 * l + std::size_t(j)] * detail::iq::sign(signs, j);
                }
            }
            bsum += sumi * ls;
        }
        const float term = d * float(bsum);
        sumf += term;
    }
    return 0.125f * sumf;
}

/// \brief The dot product of an IQ2_XS row with a Q8_K row, as ggml's `ggml_vec_dot_iq2_xs_q8_K`.
/// \param x The weights.
/// \param y The activations.
/// \return `1/8 · sum over blocks of d_x d_y · sum over 16 of (2s + 1) · (sum ±g · q)`.
constexpr float dot(std::span<const block_iq2_xs> x, std::span<const block_q8_k> y)
{
    assert(x.size() == y.size());
    float sumf = 0.0f;
    for (std::size_t i = 0; i < x.size(); ++i) {
        const float d = to_float(x[i].d) * y[i].d;
        std::int32_t bsum = 0;
        for (std::size_t ib = 0; ib < 8; ++ib)
            for (std::size_t h = 0; h < 2; ++h) {
                const int ls = 2 * ((x[i].scales[ib] >> (4 * h)) & 0xf) + 1;
                std::int32_t sumi = 0;
                for (std::size_t l = 2 * h; l < 2 * h + 2; ++l) {
                    const std::uint16_t q = x[i].qs[4 * ib + l];
                    const std::uint64_t grid = iq2xs_grid[q & 511];
                    const std::uint8_t signs = iq2_signs[q >> 9];
                    for (int j = 0; j < 8; ++j) {
                        sumi += detail::iq::grid_byte(grid, j) * y[i].qs[32 * ib + 8 * l + std::size_t(j)] * detail::iq::sign(signs, j);
                    }
                }
                bsum += sumi * ls;
            }
        const float term = d * float(bsum);
        sumf += term;
    }
    return 0.125f * sumf;
}

/// \brief The dot product of an IQ2_S row with a Q8_K row, as ggml's `ggml_vec_dot_iq2_s_q8_K`.
/// \param x The weights.
/// \param y The activations.
/// \return `1/8 · sum over blocks of d_x d_y · sum over 16 of (2s + 1) · (sum ±g · q)`.
constexpr float dot(std::span<const block_iq2_s> x, std::span<const block_q8_k> y)
{
    assert(x.size() == y.size());
    float sumf = 0.0f;
    for (std::size_t i = 0; i < x.size(); ++i) {
        const float d = to_float(x[i].d) * y[i].d;
        int bsum = 0;
        for (std::size_t ib = 0; ib < 8; ++ib) {
            int sumi[2] = {0, 0};
            for (std::size_t l = 0; l < 4; ++l) {
                const std::uint64_t grid = iq2s_grid[x[i].qs[4 * ib + l] | ((x[i].qh[ib] << (8 - 2 * l)) & 0x300)];
                const std::uint8_t signs = x[i].qs[32 + 4 * ib + l];
                for (int j = 0; j < 8; ++j) {
                    sumi[l / 2] += y[i].qs[32 * ib + 8 * l + std::size_t(j)] * detail::iq::grid_byte(grid, j) * detail::iq::sign(signs, j);
                }
            }
            const int ls1 = 1 + 2 * (x[i].scales[ib] & 0xf), ls2 = 1 + 2 * (x[i].scales[ib] >> 4);
            bsum += ls1 * sumi[0] + ls2 * sumi[1];
        }
        const float term = d * float(bsum);
        sumf += term;
    }
    return 0.125f * sumf;
}

/// \brief The dot product of an IQ3_XXS row with a Q8_K row, as ggml's `ggml_vec_dot_iq3_xxs_q8_K`.
/// \param x The weights.
/// \param y The activations.
/// \return `1/4 · sum over blocks of d_x d_y · sum over 32 of (2s + 1) · (sum ±g · q)`.
constexpr float dot(std::span<const block_iq3_xxs> x, std::span<const block_q8_k> y)
{
    assert(x.size() == y.size());
    float sumf = 0.0f;
    for (std::size_t i = 0; i < x.size(); ++i) {
        const float d = to_float(x[i].d) * y[i].d;
        std::int32_t bsum = 0;
        for (std::size_t ib = 0; ib < 8; ++ib) {
            const std::uint32_t aux = detail::iq::load32(x[i].qs.data() + 64 + 4 * ib);
            const auto ls = static_cast<std::int32_t>(2 * (aux >> 28) + 1);
            std::int32_t sumi = 0;
            for (std::size_t l = 0; l < 4; ++l) {
                const std::uint32_t g1 = iq3xxs_grid[x[i].qs[8 * ib + 2 * l]];
                const std::uint32_t g2 = iq3xxs_grid[x[i].qs[8 * ib + 2 * l + 1]];
                const std::uint8_t signs = iq2_signs[(aux >> (7 * l)) & 127];
                for (int j = 0; j < 4; ++j) {
                    sumi += detail::iq::grid_byte(g1, j) * y[i].qs[32 * ib + 8 * l + std::size_t(j)] * detail::iq::sign(signs, j);
                    sumi += detail::iq::grid_byte(g2, j) * y[i].qs[32 * ib + 8 * l + std::size_t(j) + 4] * detail::iq::sign(signs, j + 4);
                }
            }
            bsum += sumi * ls;
        }
        const float term = d * float(bsum);
        sumf += term;
    }
    return 0.25f * sumf;
}

/// \brief The dot product of an IQ3_S row with a Q8_K row, as ggml's `ggml_vec_dot_iq3_s_q8_K`.
/// \param x The weights.
/// \param y The activations.
/// \return `sum over blocks of d_x d_y · sum over 32 of (2s + 1) · (sum ±g · q)`.
constexpr float dot(std::span<const block_iq3_s> x, std::span<const block_q8_k> y)
{
    assert(x.size() == y.size());
    float sumf = 0.0f;
    for (std::size_t i = 0; i < x.size(); ++i) {
        const float d = to_float(x[i].d) * y[i].d;
        std::int32_t bsum = 0;
        for (std::size_t ib = 0; ib < 8; ++ib) {
            const auto ls = static_cast<std::int32_t>(2 * ((x[i].scales[ib / 2] >> (4 * (ib % 2))) & 0xf) + 1);
            std::int32_t sumi = 0;
            for (std::size_t l = 0; l < 4; ++l) {
                const std::uint32_t g1 = iq3s_grid[x[i].qs[8 * ib + 2 * l] | ((x[i].qh[ib] << (8 - 2 * l)) & 256)];
                const std::uint32_t g2 = iq3s_grid[x[i].qs[8 * ib + 2 * l + 1] | ((x[i].qh[ib] << (7 - 2 * l)) & 256)];
                const std::uint8_t signs = x[i].signs[4 * ib + l];
                for (int j = 0; j < 4; ++j) {
                    sumi += detail::iq::grid_byte(g1, j) * y[i].qs[32 * ib + 8 * l + std::size_t(j)] * detail::iq::sign(signs, j);
                    sumi += detail::iq::grid_byte(g2, j) * y[i].qs[32 * ib + 8 * l + std::size_t(j) + 4] * detail::iq::sign(signs, j + 4);
                }
            }
            bsum += sumi * ls;
        }
        const float term = d * float(bsum);
        sumf += term;
    }
    return sumf;
}

/// \brief The dot product of an IQ1_S row with a Q8_K row, as ggml's `ggml_vec_dot_iq1_s_q8_K`: the shift from Q8_K's sums of 16.
/// \param x The weights.
/// \param y The activations.
/// \return `sum over blocks of d_x d_y · (sum (2s + 1) g · q + 1/8 · sum ±(2s + 1) · bsums)`.
constexpr float dot(std::span<const block_iq1_s> x, std::span<const block_q8_k> y)
{
    assert(x.size() == y.size());
    float sumf = 0.0f;
    for (std::size_t i = 0; i < x.size(); ++i) {
        int sumi = 0, sumi1 = 0;
        for (std::size_t ib = 0; ib < 8; ++ib) {
            const int ls = 2 * ((x[i].qh[ib] >> 12) & 7) + 1;
            const int delta = x[i].qh[ib] & 0x8000 ? -1 : 1;
            int lsum = 0;
            for (std::size_t l = 0; l < 4; ++l) {
                const std::uint64_t grid = iq1s_grid[x[i].qs[4 * ib + l] | (((x[i].qh[ib] >> (3 * l)) & 7) << 8)];
                for (int j = 0; j < 8; ++j)
                    lsum += y[i].qs[32 * ib + 8 * l + std::size_t(j)] * detail::iq::grid_sbyte(grid, j);
            }
            sumi += ls * lsum;
            sumi1 += ls * delta * (y[i].bsums[2 * ib] + y[i].bsums[2 * ib + 1]);
        }
        const float dd = to_float(x[i].d) * y[i].d;
        const float shift = detail::iq::iq1_delta * float(sumi1);
        const float s = float(sumi) + shift;
        const float term = dd * s;
        sumf += term;
    }
    return sumf;
}

/// \brief The dot product of an IQ1_M row with a Q8_K row, as ggml's `ggml_vec_dot_iq1_m_q8_K`.
/// \param x The weights.
/// \param y The activations.
/// \return `sum over blocks of d_x d_y · (sum (2s + 1) g · q + 1/8 · sum ±(2s + 1) q)`.
constexpr float dot(std::span<const block_iq1_m> x, std::span<const block_q8_k> y)
{
    assert(x.size() == y.size());
    float sumf = 0.0f;
    for (std::size_t i = 0; i < x.size(); ++i) {
        std::array<std::uint16_t, 4> sc{};
        const float16 scale = detail::iq::iq1m_scale(x[i], sc);
        int sumi1 = 0, sumi2 = 0;
        for (std::size_t ib = 0; ib < 8; ++ib) {
            const std::uint8_t* qh = x[i].qh.data() + 2 * ib;
            const std::array<int, 4> delta{qh[0] & 0x08 ? -1 : 1, qh[0] & 0x80 ? -1 : 1, qh[1] & 0x08 ? -1 : 1, qh[1] & 0x80 ? -1 : 1};
            int sum1[2] = {0, 0}, sum2[2] = {0, 0};
            for (std::size_t l = 0; l < 4; ++l) {
                const std::uint64_t grid = iq1s_grid[x[i].qs[4 * ib + l] | ((std::uint16_t(qh[l / 2]) << (8 - 4 * (l % 2))) & 0x700)];
                int lsum1 = 0, lsum2 = 0;
                for (int j = 0; j < 8; ++j) {
                    const int q = y[i].qs[32 * ib + 8 * l + std::size_t(j)];
                    lsum1 += q * detail::iq::grid_sbyte(grid, j);
                    lsum2 += q;
                }
                sum1[l / 2] += lsum1;
                sum2[l / 2] += lsum2 * delta[l];
            }
            const int ls1 = 2 * ((sc[ib / 2] >> (6 * (ib % 2))) & 0x7) + 1;
            const int ls2 = 2 * ((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 0x7) + 1;
            sumi1 += sum1[0] * ls1 + sum1[1] * ls2;
            sumi2 += sum2[0] * ls1 + sum2[1] * ls2;
        }
        const float dd = to_float(scale) * y[i].d;
        const float shift = detail::iq::iq1_delta * float(sumi2);
        const float s = float(sumi1) + shift;
        const float term = dd * s;
        sumf += term;
    }
    return sumf;
}

/// \brief The dot product of an IQ4_NL row with a Q8_0 row, as ggml's `ggml_vec_dot_iq4_nl_q8_0`.
/// \param x The weights.
/// \param y The activations.
/// \return `sum over blocks of d_x d_y · sum v · q`.
constexpr float dot(std::span<const block_iq4_nl> x, std::span<const block_q8_0> y)
{
    assert(x.size() == y.size());
    float sumf = 0.0f;
    for (std::size_t i = 0; i < x.size(); ++i) {
        const float d = to_float(y[i].d) * to_float(x[i].d);
        int s1 = 0, s2 = 0;
        for (std::size_t j = 0; j < 16; ++j) {
            s1 += y[i].qs[j] * iq4nl_values[x[i].qs[j] & 0xf];
            s2 += y[i].qs[j + 16] * iq4nl_values[x[i].qs[j] >> 4];
        }
        const float term = d * float(s1 + s2);
        sumf += term;
    }
    return sumf;
}

/// \brief The dot product of an IQ4_XS row with a Q8_K row, as ggml's `ggml_vec_dot_iq4_xs_q8_K`.
/// \param x The weights.
/// \param y The activations.
/// \return `sum over blocks and 32s of d_x d_y (s - 32) · sum v · q`.
constexpr float dot(std::span<const block_iq4_xs> x, std::span<const block_q8_k> y)
{
    assert(x.size() == y.size());
    float sumf = 0.0f;
    for (std::size_t i = 0; i < x.size(); ++i) {
        const float d4d8 = to_float(x[i].d) * y[i].d;
        std::uint16_t h = x[i].scales_h;
        const std::uint8_t* qs = x[i].qs.data();
        const std::int8_t* q8 = y[i].qs.data();
        for (std::size_t ib = 0; ib < 8; ib += 2) {
            const int ls[2] = {(x[i].scales_l[ib / 2] & 0xf) | ((h << 4) & 0x30), (x[i].scales_l[ib / 2] >> 4) | ((h << 2) & 0x30)};
            h = static_cast<std::uint16_t>(h >> 4);
            for (const int l : ls) {
                const float dl = d4d8 * float(l - 32);
                int s1 = 0, s2 = 0;
                for (std::size_t j = 0; j < 16; ++j) {
                    s1 += q8[j] * iq4nl_values[qs[j] & 0xf];
                    s2 += q8[j + 16] * iq4nl_values[qs[j] >> 4];
                }
                const float term = dl * float(s1 + s2);
                sumf += term;
                qs += 16;
                q8 += 32;
            }
        }
    }
    return sumf;
}

} // namespace snicholls::ml

#endif // snicholls_core_ml_quant_iq_hpp
