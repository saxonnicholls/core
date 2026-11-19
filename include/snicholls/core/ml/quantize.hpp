//
//  quantize.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::ml` quantised weights: ggml's type ids and block traits, and its 32-element blocks (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, MXFP4) with NVFP4, Q1_0, Q2_0, F8_E4M3 and the Q8_K activations, byte for byte as ggml lays them out, with quantisation (and its importance-weighted form), dequantisation and the integer dot products, plus symmetric int8 rows.
/// \ingroup core
///
/// **Overview.** A quantised model stores its weights in blocks, each block a few numbers with one or two scales.
/// These are the formats llama.cpp and gpt-oss ship, so a GGUF tensor of one of them is an array of these structs,
/// read in place:
///
/// | Block | Elements | Scale | Codes | Bytes | Bits per weight |
/// |---|---|---|---|---|---|
/// | `block_q4_0` | 32 | `float16` `d` = (signed value of largest magnitude) / -8 | 4-bit `x / d + 8`, element `j` low nibble of byte `j`, `j + 16` high | 18 | 4.5 |
/// | `block_q4_1` | 32 | `float16` `d` = (max - min) / 15, `m` = min | 4-bit `(x - m) / d` | 20 | 5 |
/// | `block_q5_0` | 32 | as Q4_0 with -16 | 5-bit, the fifth bits in `qh` | 22 | 5.5 |
/// | `block_q5_1` | 32 | as Q4_1 with 31 | 5-bit, the fifth bits in `qh` | 24 | 6 |
/// | `block_q8_0` | 32 | `float16` `d` = amax / 127 | `int8` `round(x / d)` | 34 | 8.5 |
/// | `block_q8_1` | 32 | as Q8_0, and `s` = `d` · sum q | `int8` | 36 | 9 |
/// | `block_mxfp4` | 32 | E8M0 `e` = floor(log2 amax) - 2 + 127 | FP4 E2M1, packed as Q4_0 | 17 | 4.25 |
/// | `block_nvfp4` | 64 | four UE4M3 scales, one per 16 | FP4 E2M1 | 36 | 4.5 |
/// | `block_q1_0` | 128 | `float16` mean abs | the sign bit | 18 | 1.125 |
/// | `block_q2_0` | 64 | `float16` amax | 2-bit `round(x / d) + 1`, {-1, 0, 1, 2} | 18 | 2.25 |
/// | `block_q8_k` | 256 | `float` amax / 127, sums of 16 | `int8` | 292 | 9.125 |
/// | `float8_e4m3` rows | 1 | none | OCP E4M3, saturating | 1 | 8 |
///
/// The super-block formats are in `ml/quant_k.hpp` (Q2_K to Q6_K), `ml/quant_iq.hpp` (the IQ family) and
/// `ml/quant_ternary.hpp` (TQ1_0, TQ2_0); they follow the same contract:
///
/// - every block has `type` (its `ggml_type`), `elements` and `dot_block` (the activation block ggml pairs it with),
///   so `quantized_block<B>` and `block_traits<B>` describe it, and `ggml_type_traits(id)` describes any id a GGUF
///   file holds, at compile time;
/// - `quantize_row(x, importance, blocks)` is ggml's `quantize_<type>` for one row: `importance` empty is ggml's
///   reference (`quantize_row_<type>_ref`), non-empty the importance-matrix path, which minimises the error
///   weighted by it (`n_per_row` is the row's length, which some formats' weighting reads);
/// - `dequantize_row(blocks, x)` is `dequantize_row_<type>`, and `dot(weights, activations)` is ggml's scalar
///   `ggml_vec_dot_<type>_<dot type>_generic`;
/// - `quantize(x, blocks)`, `quantize(x, importance, blocks)` and `dequantize(blocks, x)` take any row.
///
/// \snippet ml/quantize_demo.cpp quantize
///
/// The type table, importance weighting and the newer formats:
///
/// \snippet ml/quantize_demo.cpp formats
///
/// `quantize_int8(x, q)` and `dequantize_int8(q, scale, x)` give one symmetric scale for a whole row (per-channel or
/// per-tensor int8), the same rounding as Q8_0.
///
/// **Relationships.** `ml/minifloat.hpp` (`float16`, `float8_e4m3`, the E2M1 elements); `ml/row.hpp`;
/// `ml/linear.hpp` (`quantized_weights` over any `quant_block`); `ml/quant_k.hpp`, `ml/quant_iq.hpp`,
/// `ml/quant_ternary.hpp`, which build on the helpers here. Group: \ref core.
///
/// **Implementations.** One per block. The blocks are the formats; the dot products are the scalar reference, which
/// a SIMD or GPU kernel reorders.
///
/// **Guarantees.** `constexpr`, no allocation, linear in the row. The blocks are trivially copyable, have ggml's
/// sizes and no padding (`static_assert`ed against `ggml_type_traits`). Quantised bytes equal ggml's and dot
/// products equal ggml's `_generic` ones on every input of the committed fixtures (`tests/fixtures/ml/`, from
/// `scripts/generate_quant_fixture.cpp` and `scripts/generate_quant_formats_fixture.cpp`, built against ggml). Each
/// multiplication and each addition is its own statement, so no compiler fuses them into an FMA and the result is
/// the same on every machine.
///
/// **Errors.** Preconditions, asserted: the row has `elements · blocks.size()` numbers; `importance` is empty or as
/// long as the row; the rows of a dot product have the same number of blocks. NaN and infinity are not
/// representable in these formats; the result is unspecified, as in ggml (deterministic, but not meaningful).
///
/// **Performance.** Against ggml's scalar loops written out by hand (`benchmarks/ml/quantize_benchmark.cpp`), rows of
/// 4,096: the three dot products level, 1.9 to 2.6 billion weights per second; Q8_0 quantisation 2x faster than
/// ggml's loop (590 against 300 million per second), from rounding without `roundf` (Xeon W-3245, Release, Apple clang 17, 2026-11-11,
/// one thread, scalar code; SIMD kernels are the libraries' above). The formats new in 0.17.0 are in BENCHMARKS.md.
///
/// **Tests and demos.** `tests/ml/quantize_test.cpp`, `demos/ml/quantize_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<bit>`, `<cassert>`, `<cstddef>`, `<cstdint>`, `<limits>`, `<span>`, `<string_view>`,
/// `<type_traits>`, `snicholls/core/math/constexpr_math.hpp`, `ml/minifloat.hpp`, `ml/row.hpp`.
///
/// **Known issues.** ggml computes MXFP4's exponent as `floorf(log2f(amax))`, which depends on the C library and can
/// round up for an amax a few units in the last place below a power of two; core reads the exponent from the bits,
/// as the OCP specification defines it. Where ggml converts a float out of an integer's range (C leaves it undefined:
/// the inverse scale of a block of subnormals is infinite; an MXFP4 amax below 2^-125 makes a negative exponent),
/// core gives what ggml gives on x86, where the fixtures were made (`cvttss2si`'s INT_MIN, then the low byte). NVFP4's scale conversion is ggml's (`ggml_fp32_to_ue4m3`: the rounding bit added, not ties to even),
/// so the bytes are ggml's. ggml leaves Q8_K's `bsums` unset for a block of zeros; core writes zeros. A dot product
/// over infinities is a NaN whose sign is the hardware's (negative on x86, positive on ARM), so results are
/// bit-identical across platforms except for the sign of a NaN. Each multiply and add is its own statement, which
/// Clang never fuses, but GCC fuses across statements by default in C++ (`-ffp-contract=fast`), so on a target with
/// FMA (every ARM64) the last bit can differ: compile with `-ffp-contract=off` for ggml's bits.
///
/// **Status.** Experimental, new in 0.7.0; the type ids, traits, importance weighting and the formats besides Q8_0,
/// Q4_0 and MXFP4 new in 0.17.0.

#ifndef snicholls_core_ml_quantize_hpp
#define snicholls_core_ml_quantize_hpp

#include <snicholls/core/math/constexpr_math.hpp>
#include <snicholls/core/ml/minifloat.hpp>
#include <snicholls/core/ml/row.hpp>

#include <array>
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string_view>
#include <type_traits>

namespace snicholls::ml
{

/// \brief ggml's tensor types, with the ids a GGUF file stores; ids ggml removed are absent.
/// \ingroup core
enum class ggml_type : std::uint32_t
{
    f32 = 0,        ///< `float`.
    f16 = 1,        ///< IEEE binary16.
    q4_0 = 2,       ///< `block_q4_0`.
    q4_1 = 3,       ///< `block_q4_1`.
    q5_0 = 6,       ///< `block_q5_0`.
    q5_1 = 7,       ///< `block_q5_1`.
    q8_0 = 8,       ///< `block_q8_0`.
    q8_1 = 9,       ///< `block_q8_1`.
    q2_k = 10,      ///< `block_q2_k` (ggml's Q2_K).
    q3_k = 11,      ///< `block_q3_k`.
    q4_k = 12,      ///< `block_q4_k`.
    q5_k = 13,      ///< `block_q5_k`.
    q6_k = 14,      ///< `block_q6_k`.
    q8_k = 15,      ///< `block_q8_k`.
    iq2_xxs = 16,   ///< `block_iq2_xxs`.
    iq2_xs = 17,    ///< `block_iq2_xs`.
    iq3_xxs = 18,   ///< `block_iq3_xxs`.
    iq1_s = 19,     ///< `block_iq1_s`.
    iq4_nl = 20,    ///< `block_iq4_nl`.
    iq3_s = 21,     ///< `block_iq3_s`.
    iq2_s = 22,     ///< `block_iq2_s`.
    iq4_xs = 23,    ///< `block_iq4_xs`.
    i8 = 24,        ///< `std::int8_t`.
    i16 = 25,       ///< `std::int16_t`.
    i32 = 26,       ///< `std::int32_t`.
    i64 = 27,       ///< `std::int64_t`.
    f64 = 28,       ///< `double`.
    iq1_m = 29,     ///< `block_iq1_m`.
    bf16 = 30,      ///< bfloat16.
    tq1_0 = 34,     ///< `block_tq1_0`.
    tq2_0 = 35,     ///< `block_tq2_0`.
    mxfp4 = 39,     ///< `block_mxfp4`.
    nvfp4 = 40,     ///< `block_nvfp4`.
    q1_0 = 41,      ///< `block_q1_0`.
    q2_0 = 42,      ///< `block_q2_0`.
    f8_e4m3 = 43,   ///< `float8_e4m3`.
};

/// \brief One past the largest `ggml_type` id.
inline constexpr std::uint32_t ggml_type_count = 44;

/// \brief What a GGUF reader needs of a type: its name and the size of its blocks.
/// \ingroup core
struct ggml_type_info
{
    std::string_view name;           ///< ggml's name (`"q4_K"`, `"iq2_xxs"`); empty for an unknown id.
    std::size_t block_elements = 0;  ///< Numbers per block; 1 for a plain type; 0 for an unknown id.
    std::size_t block_bytes = 0;     ///< Bytes per block.
    bool quantized = false;          ///< Whether the type is a block format.
};

namespace detail::quantize
{
    consteval std::array<ggml_type_info, ggml_type_count> make_type_table()
    {
        std::array<ggml_type_info, ggml_type_count> t{};
        const auto set = [&t](ggml_type id, std::string_view name, std::size_t n, std::size_t bytes, bool q) {
            t[static_cast<std::size_t>(id)] = ggml_type_info{name, n, bytes, q};
        };
        set(ggml_type::f32, "f32", 1, 4, false);
        set(ggml_type::f16, "f16", 1, 2, false);
        set(ggml_type::q4_0, "q4_0", 32, 18, true);
        set(ggml_type::q4_1, "q4_1", 32, 20, true);
        set(ggml_type::q5_0, "q5_0", 32, 22, true);
        set(ggml_type::q5_1, "q5_1", 32, 24, true);
        set(ggml_type::q8_0, "q8_0", 32, 34, true);
        set(ggml_type::q8_1, "q8_1", 32, 36, true);
        set(ggml_type::q2_k, "q2_K", 256, 84, true);
        set(ggml_type::q3_k, "q3_K", 256, 110, true);
        set(ggml_type::q4_k, "q4_K", 256, 144, true);
        set(ggml_type::q5_k, "q5_K", 256, 176, true);
        set(ggml_type::q6_k, "q6_K", 256, 210, true);
        set(ggml_type::q8_k, "q8_K", 256, 292, true);
        set(ggml_type::iq2_xxs, "iq2_xxs", 256, 66, true);
        set(ggml_type::iq2_xs, "iq2_xs", 256, 74, true);
        set(ggml_type::iq3_xxs, "iq3_xxs", 256, 98, true);
        set(ggml_type::iq1_s, "iq1_s", 256, 50, true);
        set(ggml_type::iq4_nl, "iq4_nl", 32, 18, true);
        set(ggml_type::iq3_s, "iq3_s", 256, 110, true);
        set(ggml_type::iq2_s, "iq2_s", 256, 82, true);
        set(ggml_type::iq4_xs, "iq4_xs", 256, 136, true);
        set(ggml_type::i8, "i8", 1, 1, false);
        set(ggml_type::i16, "i16", 1, 2, false);
        set(ggml_type::i32, "i32", 1, 4, false);
        set(ggml_type::i64, "i64", 1, 8, false);
        set(ggml_type::f64, "f64", 1, 8, false);
        set(ggml_type::iq1_m, "iq1_m", 256, 56, true);
        set(ggml_type::bf16, "bf16", 1, 2, false);
        set(ggml_type::tq1_0, "tq1_0", 256, 54, true);
        set(ggml_type::tq2_0, "tq2_0", 256, 66, true);
        set(ggml_type::mxfp4, "mxfp4", 32, 17, true);
        set(ggml_type::nvfp4, "nvfp4", 64, 36, true);
        set(ggml_type::q1_0, "q1_0", 128, 18, true);
        set(ggml_type::q2_0, "q2_0", 64, 18, true);
        set(ggml_type::f8_e4m3, "f8_e4m3", 1, 1, false);
        return t;
    }
    inline constexpr std::array<ggml_type_info, ggml_type_count> type_table = make_type_table();
} // namespace detail::quantize

/// \brief The name and block size of a ggml type, as ggml's `ggml_type_traits` table has them.
/// \param t The type, as a GGUF file stores it; it may be any value.
/// \return Its traits; an empty `ggml_type_info` for an id ggml removed or does not have.
[[nodiscard]] constexpr ggml_type_info ggml_type_traits(ggml_type t) noexcept
{
    const auto i = static_cast<std::uint32_t>(t);
    return i < ggml_type_count ? detail::quantize::type_table[i] : ggml_type_info{};
}

/// \brief The bytes of a row of `n` numbers of type `t`, as ggml's `ggml_row_size`.
/// \param t The type.
/// \param n The numbers in the row, a multiple of the block size.
/// \return `n / block_elements · block_bytes`; 0 for an unknown type.
[[nodiscard]] constexpr std::size_t ggml_row_bytes(ggml_type t, std::size_t n) noexcept
{
    const ggml_type_info info = ggml_type_traits(t);
    assert(info.block_elements == 0 || n % info.block_elements == 0);
    return info.block_elements == 0 ? 0 : n / info.block_elements * info.block_bytes;
}

/// \brief Whether `B` is a ggml block: trivially copyable, with `type`, `elements` and the size `ggml_type_traits` gives.
/// \ingroup core
template <typename B>
concept quantized_block = std::is_trivially_copyable_v<B> && requires {
    { B::type } -> std::convertible_to<ggml_type>;
    { B::elements } -> std::convertible_to<std::size_t>;
    typename B::dot_block;
} && sizeof(B) == ggml_type_traits(B::type).block_bytes && B::elements == ggml_type_traits(B::type).block_elements;

/// \brief A block format's numbers at compile time: elements, bytes, name, bits per weight, and its activation block.
/// \ingroup core
/// \tparam B A `quantized_block`.
template <quantized_block B>
struct block_traits
{
    static constexpr ggml_type type = B::type;                                        ///< The ggml id.
    static constexpr std::size_t elements = B::elements;                              ///< Numbers per block.
    static constexpr std::size_t bytes = sizeof(B);                                   ///< Bytes per block.
    static constexpr std::string_view name = ggml_type_traits(B::type).name;          ///< ggml's name.
    static constexpr double bits_per_weight = 8.0 * double(bytes) / double(elements); ///< Storage per number.
    /// \brief The activations `dot` pairs it with.
    using dot_block = typename B::dot_block;
};

/// \brief The numbers in one block of the 32-element formats.
inline constexpr std::size_t quant_block_size = 32;

struct block_q8_0;

/// \brief ggml's Q8_0: a `float16` scale and 32 signed bytes, `x ≈ d · q`.
/// \ingroup core
struct block_q8_0
{
    static constexpr ggml_type type = ggml_type::q8_0;   ///< The ggml id.
    static constexpr std::size_t elements = 32;          ///< Numbers per block.
    /// \brief ggml's `vec_dot_type`.
    using dot_block = block_q8_0;

    float16 d;                                     ///< The scale, amax / 127.
    std::array<std::int8_t, quant_block_size> qs;  ///< The quantised numbers.
};

/// \brief ggml's Q8_1: Q8_0 with `s` = `d` · (the sum of the codes), the activations of Q4_1 and Q5_1.
/// \ingroup core
struct block_q8_1
{
    static constexpr ggml_type type = ggml_type::q8_1;   ///< The ggml id.
    static constexpr std::size_t elements = 32;          ///< Numbers per block.
    /// \brief Itself: it is an activation format.
    using dot_block = block_q8_1;

    float16 d;                                     ///< The scale, amax / 127.
    float16 s;                                     ///< `d` times the sum of `qs`.
    std::array<std::int8_t, quant_block_size> qs;  ///< The quantised numbers.
};

/// \brief ggml's Q4_0: a `float16` scale and 32 four-bit numbers, `x ≈ d · (q - 8)`.
/// \ingroup core
struct block_q4_0
{
    static constexpr ggml_type type = ggml_type::q4_0;   ///< The ggml id.
    static constexpr std::size_t elements = 32;          ///< Numbers per block.
    /// \brief ggml's `vec_dot_type`.
    using dot_block = block_q8_0;

    float16 d;                                          ///< The scale: the signed value of largest magnitude / -8.
    std::array<std::uint8_t, quant_block_size / 2> qs;  ///< Element `j` in the low nibble of byte `j`, `j + 16` in the high.
};

/// \brief ggml's Q4_1: a `float16` scale and minimum and 32 four-bit numbers, `x ≈ d · q + m`.
/// \ingroup core
struct block_q4_1
{
    static constexpr ggml_type type = ggml_type::q4_1;   ///< The ggml id.
    static constexpr std::size_t elements = 32;          ///< Numbers per block.
    /// \brief ggml's `vec_dot_type`.
    using dot_block = block_q8_1;

    float16 d;                                          ///< The scale, (max - min) / 15.
    float16 m;                                          ///< The minimum.
    std::array<std::uint8_t, quant_block_size / 2> qs;  ///< Packed as Q4_0's.
};

/// \brief ggml's Q5_0: Q4_0 with a fifth bit, `x ≈ d · (q - 16)`.
/// \ingroup core
struct block_q5_0
{
    static constexpr ggml_type type = ggml_type::q5_0;   ///< The ggml id.
    static constexpr std::size_t elements = 32;          ///< Numbers per block.
    /// \brief ggml's `vec_dot_type`.
    using dot_block = block_q8_0;

    float16 d;                                          ///< The scale: the signed value of largest magnitude / -16.
    std::array<std::uint8_t, 4> qh;                     ///< The fifth bits: bit `j` of the little-endian word is element `j`'s.
    std::array<std::uint8_t, quant_block_size / 2> qs;  ///< The low four bits, packed as Q4_0's.
};

/// \brief ggml's Q5_1: Q4_1 with a fifth bit, `x ≈ d · q + m`.
/// \ingroup core
struct block_q5_1
{
    static constexpr ggml_type type = ggml_type::q5_1;   ///< The ggml id.
    static constexpr std::size_t elements = 32;          ///< Numbers per block.
    /// \brief ggml's `vec_dot_type`.
    using dot_block = block_q8_1;

    float16 d;                                          ///< The scale, (max - min) / 31.
    float16 m;                                          ///< The minimum.
    std::array<std::uint8_t, 4> qh;                     ///< The fifth bits, as Q5_0's.
    std::array<std::uint8_t, quant_block_size / 2> qs;  ///< The low four bits, packed as Q4_0's.
};

/// \brief The OCP MXFP4 block as ggml stores it: an E8M0 exponent and 32 FP4 (E2M1) numbers, `x ≈ 2^(e - 127) · v`.
/// \ingroup core
struct block_mxfp4
{
    static constexpr ggml_type type = ggml_type::mxfp4;  ///< The ggml id.
    static constexpr std::size_t elements = 32;          ///< Numbers per block.
    /// \brief ggml's `vec_dot_type`.
    using dot_block = block_q8_0;

    std::uint8_t e;                                     ///< The shared exponent, biased by 127.
    std::array<std::uint8_t, quant_block_size / 2> qs;  ///< E2M1 codes, packed as Q4_0's.
};

/// \brief ggml's NVFP4: 64 FP4 (E2M1) numbers in four groups of 16, each with an unsigned E4M3 scale.
/// \ingroup core
///
/// ggml's form of NVIDIA's NVFP4 has no per-tensor FP32 scale: a tensor's scale is applied outside the block.
/// Group `g` covers elements `16g` to `16g + 15`; within it, element `j` is the low nibble of byte `8g + j` and
/// `j + 8` the high, so each group is packed as a Q4_0 block of 16.
struct block_nvfp4
{
    static constexpr ggml_type type = ggml_type::nvfp4;  ///< The ggml id.
    static constexpr std::size_t elements = 64;          ///< Numbers per block.
    /// \brief ggml's `vec_dot_type` (two Q8_0 blocks per NVFP4 block).
    using dot_block = block_q8_0;

    std::array<std::uint8_t, 4> d;    ///< UE4M3 scales, one per 16 numbers.
    std::array<std::uint8_t, 32> qs;  ///< E2M1 codes.
};

/// \brief ggml's Q1_0: 128 signs and one `float16` scale, the mean absolute value: `x ≈ ±d`.
/// \ingroup core
struct block_q1_0
{
    static constexpr ggml_type type = ggml_type::q1_0;   ///< The ggml id.
    static constexpr std::size_t elements = 128;         ///< Numbers per block.
    /// \brief ggml's `vec_dot_type` (four Q8_0 blocks per Q1_0 block).
    using dot_block = block_q8_0;

    float16 d;                         ///< The mean absolute value.
    std::array<std::uint8_t, 16> qs;   ///< Bit `j % 8` of byte `j / 8` set when element `j` is at least 0.
};

/// \brief ggml's Q2_0: 64 two-bit numbers and a `float16` scale, the largest magnitude: `x ≈ d · (q - 1)`, q - 1 in {-1, 0, 1, 2}.
/// \ingroup core
struct block_q2_0
{
    static constexpr ggml_type type = ggml_type::q2_0;   ///< The ggml id.
    static constexpr std::size_t elements = 64;          ///< Numbers per block.
    /// \brief ggml's `vec_dot_type` (two Q8_0 blocks per Q2_0 block).
    using dot_block = block_q8_0;

    float16 d;                         ///< The largest magnitude.
    std::array<std::uint8_t, 16> qs;   ///< Element `j` in bits `2 (j % 4)` of byte `j / 4`.
};

/// \brief ggml's Q8_K: the activations of the super-block formats, 256 bytes with a `float` scale and the sums of each 16.
/// \ingroup core
struct block_q8_k
{
    static constexpr ggml_type type = ggml_type::q8_k;   ///< The ggml id.
    static constexpr std::size_t elements = 256;         ///< Numbers per block.
    /// \brief Itself: it is an activation format.
    using dot_block = block_q8_k;

    float d;                                  ///< The scale: the signed value of largest magnitude / -127.
    std::array<std::int8_t, 256> qs;          ///< The quantised numbers.
    std::array<std::int16_t, 16> bsums;       ///< The sum of each 16 codes.
};

static_assert(quantized_block<block_q4_0> && quantized_block<block_q4_1> && quantized_block<block_q5_0> && quantized_block<block_q5_1>
                  && quantized_block<block_q8_0> && quantized_block<block_q8_1> && quantized_block<block_mxfp4>
                  && quantized_block<block_nvfp4> && quantized_block<block_q1_0> && quantized_block<block_q2_0> && quantized_block<block_q8_k>,
              "snicholls::ml: the quantised blocks must have ggml's sizes");

/// \brief Whether `B` is a 32-element block paired with Q8_0 activations, as `quantized_weights` (`ml/linear.hpp`) takes.
/// \ingroup core
template <typename B>
concept quant_block = quantized_block<B> && B::elements == quant_block_size && std::same_as<typename B::dot_block, block_q8_0>;

namespace detail::quantize
{
    using block = std::span<const float, quant_block_size>;

    // E2M1 values, doubled (ggml's kvalues_mxfp4): code c is kvalues[c] / 2
    inline constexpr std::array<std::int8_t, 16> fp4_doubled{0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

    // ggml's group thresholds below which a group counts as all zero
    inline constexpr float group_max_eps = 1e-15f;

    constexpr float abs(float x) { return x < 0.0f ? -x : x; }
    constexpr int imax(int a, int b) { return a < b ? b : a; }
    constexpr int imin(int a, int b) { return a < b ? a : b; }
    constexpr float fmax(float a, float b) { return a > b ? a : b; }   // ggml's MAX(a, b): b when unordered

    // C's conversion of a float to an integer as x86 performs it (cvttss2si): toward zero, and INT_MIN for NaN or
    // a value out of range. ggml's code meets those on degenerate blocks (a subnormal amax makes its inverse
    // infinite), where C leaves the conversion undefined; this is ggml's result on x86 (ARM saturates instead)
    constexpr int to_int(float x)
    {
        return x >= -2147483648.0f && x < 2147483648.0f ? static_cast<int>(x) : std::numeric_limits<int>::min();
    }

    // roundf, then the conversion to an integer: halves away from zero, the fraction x - trunc(x) is exact
    constexpr int round_to_int(float x)
    {
        if (!(x >= -2147483648.0f && x < 2147483648.0f))
            return std::numeric_limits<int>::min();
        const int t = static_cast<int>(x);
        const float frac = x - static_cast<float>(t);
        return t + (frac >= 0.5f ? 1 : 0) - (frac <= -0.5f ? 1 : 0);
    }

    // ggml's nearest_int: round to nearest even by adding 1.5 * 2^23, for |x| below 2^22
    constexpr int nearest_int(float x)
    {
        const float v = x + 12582912.0f;
        return static_cast<int>(std::bit_cast<std::uint32_t>(v) & 0x007fffffu) - 0x00400000;
    }

    // ggml's GGML_FP32_TO_FP16 and back: round to nearest even, overflow to infinity
    constexpr float16 fp16(float x) { return to_minifloat<float16>(x); }
    constexpr float f32(float16 h) { return to_float(h); }

    // sqrtf, correctly rounded at compile time as at run time
    constexpr float sqrt(float x) { return math::sqrt(x); }

    // 2^(e - 128): half the MX scale, as ggml_e8m0_to_fp32_half, to pair with the doubled values
    constexpr float e8m0_half(std::uint8_t e)
    {
        return std::bit_cast<float>(e < 2 ? std::uint32_t{0x00200000} << e : std::uint32_t(e - 1) << 23);
    }

    // floor(log2 x) for a positive finite float, from its bits
    constexpr int floor_log2(float x)
    {
        const std::uint32_t b = std::bit_cast<std::uint32_t>(x);
        const int biased = int((b >> 23) & 0xff);
        if (biased != 0)
            return biased - 127;
        return std::bit_width(b & 0x7fffffu) - 1 - 149;   // subnormal: m · 2^-149
    }

    // ggml's make_qx_quants: a symmetric scale for n numbers, codes L in [0, 2 nmax), searched over 19 scales when
    // rmse_type is not 0, weighted by qw (or by x^2, 1, |x|, sqrt|x| for rmse_type 1 to 4 when qw is null)
    constexpr float make_qx_quants(int n, int nmax, const float* x, std::int8_t* L, int rmse_type, const float* qw)
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
        float iscale = float(-nmax) / max;
        if (rmse_type == 0) {
            for (int i = 0; i < n; ++i) {
                const float v = iscale * x[i];
                L[i] = static_cast<std::int8_t>(nmax + imax(-nmax, imin(nmax - 1, nearest_int(v))));
            }
            return 1 / iscale;
        }
        bool return_early = false;
        if (rmse_type < 0) {
            rmse_type = -rmse_type;
            return_early = true;
        }
        const auto weight = [&](int i) {
            if (qw)
                return qw[i];
            if (rmse_type == 1) {
                const float w = x[i] * x[i];
                return w;
            }
            return rmse_type == 2 ? 1.0f : rmse_type == 3 ? abs(x[i]) : sqrt(abs(x[i]));
        };
        float sumlx = 0, suml2 = 0;
        for (int i = 0; i < n; ++i) {
            const float v = iscale * x[i];
            const int l = imax(-nmax, imin(nmax - 1, nearest_int(v)));
            L[i] = static_cast<std::int8_t>(l + nmax);
            const float w = weight(i);
            const float wx = w * x[i];
            const float wxl = wx * float(l);
            sumlx += wxl;
            const float wl = w * float(l);
            const float wll = wl * float(l);
            suml2 += wll;
        }
        float scale = suml2 != 0.0f ? sumlx / suml2 : 0.0f;
        if (return_early) {
            const float inv = 1 / iscale;
            const float sum = scale + inv;
            return suml2 > 0 ? 0.5f * sum : inv;
        }
        float best = scale * sumlx;
        for (int is = -9; is <= 9; ++is) {
            if (is == 0)
                continue;
            const float step = 0.1f * float(is);
            const float top = float(nmax) + step;
            iscale = -top / max;
            sumlx = suml2 = 0;
            for (int i = 0; i < n; ++i) {
                const float v = iscale * x[i];
                const int l = imax(-nmax, imin(nmax - 1, nearest_int(v)));
                const float w = weight(i);
                const float wx = w * x[i];
                const float wxl = wx * float(l);
                sumlx += wxl;
                const float wl = w * float(l);
                const float wll = wl * float(l);
                suml2 += wll;
            }
            const float lhs = sumlx * sumlx;
            const float rhs = best * suml2;
            if (suml2 > 0 && lhs > rhs) {
                for (int i = 0; i < n; ++i) {
                    const float v = iscale * x[i];
                    L[i] = static_cast<std::int8_t>(nmax + imax(-nmax, imin(nmax - 1, nearest_int(v))));
                }
                scale = sumlx / suml2;
                best = scale * sumlx;
            }
        }
        return scale;
    }

    // ggml's make_qkx2_quants (weights given) and make_qkx3_quants (weights or x^2): an affine scale and minimum for n
    // numbers, codes L in [0, nmax], searched over nstep + 1 scales; the minimum, negated, goes to the_min
    template <bool Qkx3>
    constexpr float make_qkx_quants(int n, int nmax, const float* x, const float* weights, std::uint8_t* L, float* the_min,
                                    std::uint8_t* Laux, float rmin, float rdelta, int nstep, bool use_mad)
    {
        const auto weight = [&](int i) {
            if (weights)
                return weights[i];
            const float w = x[i] * x[i];
            return w;
        };
        float min = x[0], max = x[0];
        float sum_w = weight(0);
        float sum_x = sum_w * x[0];
        for (int i = 1; i < n; ++i) {
            if (x[i] < min)
                min = x[i];
            if (x[i] > max)
                max = x[i];
            const float w = weight(i);
            sum_w += w;
            const float wx = w * x[i];
            sum_x += wx;
        }
        if (min > 0)
            min = 0;
        if (Qkx3 ? max <= min : max == min) {
            for (int i = 0; i < n; ++i)
                L[i] = 0;
            *the_min = -min;
            return 0.0f;
        }
        float iscale = float(nmax) / (max - min);
        float scale = 1 / iscale;
        float best_error = 0;
        const auto error = [&](float s, float m, const std::uint8_t* codes) {
            float e = 0;
            for (int i = 0; i < n; ++i) {
                const float sl = s * float(codes[i]);
                const float slm = sl + m;
                float diff = slm - x[i];
                if (use_mad) {
                    diff = abs(diff);
                } else {
                    const float d2 = diff * diff;
                    diff = d2;
                }
                const float wd = weight(i) * diff;
                e += wd;
            }
            return e;
        };
        for (int i = 0; i < n; ++i) {
            const float v = iscale * (x[i] - min);
            L[i] = static_cast<std::uint8_t>(imax(0, imin(nmax, nearest_int(v))));
        }
        best_error = error(scale, min, L);
        if (nstep < 1) {
            *the_min = -min;
            return scale;
        }
        for (int is = 0; is <= nstep; ++is) {
            const float step = rdelta * float(is);
            const float top = rmin + step;
            const float top2 = top + float(nmax);
            iscale = top2 / (max - min);   // min is the best fit so far, which moves
            float sum_l = 0, sum_l2 = 0, sum_xl = 0;
            for (int i = 0; i < n; ++i) {
                const float v = iscale * (x[i] - min);
                int l = 0;
                if constexpr (Qkx3)
                    l = v > 0 ? nearest_int(v < float(nmax) ? v : float(nmax)) : 0;
                else
                    l = imax(0, imin(nmax, nearest_int(v)));
                Laux[i] = static_cast<std::uint8_t>(l);
                const float w = weight(i);
                const float wl = w * float(l);
                sum_l += wl;
                const float wll = wl * float(l);
                sum_l2 += wll;
                const float wlx = wl * x[i];
                sum_xl += wlx;
            }
            const float a = sum_w * sum_l2;
            const float b = sum_l * sum_l;
            const float D = a - b;
            if (D > 0) {
                const float s1 = sum_w * sum_xl;
                const float s2 = sum_x * sum_l;
                float this_scale = (s1 - s2) / D;
                const float m1 = sum_l2 * sum_x;
                const float m2 = sum_l * sum_xl;
                float this_min = (m1 - m2) / D;
                if (this_min > 0) {
                    this_min = 0;
                    this_scale = sum_xl / sum_l2;
                }
                const float cur_error = error(this_scale, this_min, Laux);
                if (cur_error < best_error) {
                    for (int i = 0; i < n; ++i)
                        L[i] = Laux[i];
                    best_error = cur_error;
                    scale = this_scale;
                    min = this_min;
                }
            }
        }
        *the_min = -min;
        return scale;
    }

    // ggml's make_qp_quants: a positive scale for n non-negative numbers (scales of sub-blocks), codes L in [0, nmax],
    // weighted by quant_weights
    constexpr float make_qp_quants(int n, int nmax, const float* x, std::uint8_t* L, const float* quant_weights)
    {
        float max = 0;
        for (int i = 0; i < n; ++i)
            max = fmax(max, x[i]);
        if (max < group_max_eps) {
            for (int i = 0; i < n; ++i)
                L[i] = 0;
            return 0.0f;
        }
        float iscale = float(nmax) / max;
        for (int i = 0; i < n; ++i) {
            const float v = iscale * x[i];
            L[i] = static_cast<std::uint8_t>(nearest_int(v));
        }
        const float scale = 1 / iscale;
        const auto mse_of = [&](float s, auto code) {
            float mse = 0;
            for (int i = 0; i < n; ++i) {
                const float sl = s * float(code(i));
                const float diff = x[i] - sl;
                const float wd = quant_weights[i] * diff;
                const float wdd = wd * diff;
                mse += wdd;
            }
            return mse;
        };
        float best_mse = mse_of(scale, [&](int i) { return int(L[i]); });
        for (int is = -4; is <= 4; ++is) {
            if (is == 0)
                continue;
            const float step = 0.1f * float(is);
            const float top = step + float(nmax);
            const float iscale_is = top / max;
            const float scale_is = 1 / iscale_is;
            const float mse = mse_of(scale_is, [&](int i) {
                const float v = iscale_is * x[i];
                return imin(nmax, nearest_int(v));
            });
            if (mse < best_mse) {
                best_mse = mse;
                iscale = iscale_is;
            }
        }
        float sumlx = 0, suml2 = 0;
        for (int i = 0; i < n; ++i) {
            const float v = iscale * x[i];
            const int l = imin(nmax, nearest_int(v));
            L[i] = static_cast<std::uint8_t>(l);
            const float w = quant_weights[i];
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
                const float w = quant_weights[i];
                const float wx = w * x[i];
                const float wxl = wx * float(L[i]);
                float slx = sumlx - wxl;
                const float wl = w * float(L[i]);
                const float wll = wl * float(L[i]);
                float sl2 = suml2 - wll;
                if (slx > 0 && sl2 > 0) {
                    const float xs = x[i] * sl2;
                    const float v = xs / slx;
                    const int new_l = imin(nmax, nearest_int(v));
                    if (new_l != int(L[i])) {
                        const float wxn = wx * float(new_l);
                        slx += wxn;
                        const float wn = w * float(new_l);
                        const float wnn = wn * float(new_l);
                        sl2 += wnn;
                        const float s1 = slx * slx;
                        const float lhs = s1 * suml2;
                        const float s2 = sumlx * sumlx;
                        const float rhs = s2 * sl2;
                        if (lhs > rhs) {
                            L[i] = static_cast<std::uint8_t>(new_l);
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
        return suml2 > 0.0f ? sumlx / suml2 : 0.0f;
    }

    // ggml's sigma2 of a row for the importance weights: the mean square (sum x^2 / n_per_row)
    constexpr float mean_square(std::span<const float> x)
    {
        float sum_x2 = 0;
        for (const float v : x) {
            const float v2 = v * v;
            sum_x2 += v2;
        }
        return sum_x2 / float(x.size());
    }

    // weight = qw · sqrt(sigma2 + x^2), as every importance-weighted quantiser in ggml forms it
    constexpr float importance_weight(float qw, float sigma2, float x)
    {
        const float x2 = x * x;
        const float s = sigma2 + x2;
        return qw * sqrt(s);
    }

    // ggml's ggml_fp32_to_ue4m3 and ggml_ue4m3_to_fp32 (which returns half the value, to pair with doubled E2M1)
    constexpr std::uint8_t fp32_to_ue4m3(float x)
    {
        if (!(x > 0.0f))
            return 0;
        if (x > 448.0f)
            x = 448.0f;
        const std::uint32_t bits = std::bit_cast<std::uint32_t>(x);
        const int fp32_exp = int((bits >> 23) & 0xff) - 127;
        const int fp32_man = int((bits >> 20) & 0x7);
        int exp = fp32_exp + 7;
        if (exp <= 0) {
            const float scaled = x * 512.0f;
            const float shifted = scaled + 0.5f;
            int man = static_cast<int>(shifted);
            man = man > 7 ? 7 : man;
            return man < 1 ? std::uint8_t{0} : static_cast<std::uint8_t>(man);
        }
        if (exp >= 15)
            return 0x7e;
        int man = fp32_man + int((bits >> 19) & 1);
        if (man > 7) {
            man = 0;
            if (++exp >= 15)
                return 0x7e;
        }
        return static_cast<std::uint8_t>((exp << 3) | man);
    }

    constexpr float ue4m3_half(std::uint8_t x)
    {
        if (x == 0 || x == 0x7f)
            return 0.0f;
        const int exp = (x >> 3) & 0xf;
        const int man = x & 0x7;
        // ldexpf((float) man, -9) or ldexpf(1 + man / 8, exp - 7), then halved: every step exact
        const float raw = exp == 0 ? float(man) * 0x1p-9f
                                   : std::bit_cast<float>(static_cast<std::uint32_t>(exp - 7 + 127) << 23) * (1.0f + float(man) / 8.0f);
        return raw * 0.5f;
    }

    constexpr std::uint8_t nearest_fp4(float v, float half_scale)
    {
        std::uint8_t best = 0;
        float best_err = abs(float(fp4_doubled[0]) * half_scale - v);
        for (std::uint8_t c = 1; c < 16; ++c) {
            const float err = abs(float(fp4_doubled[c]) * half_scale - v);
            if (err < best_err) {
                best = c;
                best_err = err;
            }
        }
        return best;
    }

    // Q4_0's code for v at inverse scale id, offset 8.5 (or 16.5 for Q5_0): MIN(top, (int8_t)(v · id + offset))
    constexpr std::uint8_t offset_code(float v, float id, float offset, int top)
    {
        const float scaled = v * id;
        const float shifted = scaled + offset;
        const int q = static_cast<std::int8_t>(to_int(shifted));
        return static_cast<std::uint8_t>(q < top ? q : top);
    }

    // Q4_1's code for v: MIN(15, (int8_t)((v - min) · id + 0.5)); Q5_1's has no MIN: (uint8_t)(...)
    constexpr std::uint8_t affine_code(float v, float min, float id, int top)
    {
        const float scaled = (v - min) * id;
        const float shifted = scaled + 0.5f;
        if (top > 15)
            return static_cast<std::uint8_t>(to_int(shifted));
        const int q = static_cast<std::int8_t>(to_int(shifted));
        return static_cast<std::uint8_t>(q < top ? q : top);
    }

    constexpr void check(std::size_t elements, std::size_t blocks, std::size_t n, std::size_t importance)
    {
        assert(n == elements * blocks && (importance == 0 || importance == n));
        (void)elements, (void)blocks, (void)n, (void)importance;
    }
} // namespace detail::quantize

/// \brief Quantises one row to Q8_0, as ggml's `quantize_row_q8_0_ref` (Q8_0 has no importance-weighted path).
/// \param x The row, `32 · y.size()` floats.
/// \param importance Ignored, as ggml ignores it.
/// \param y Receives the blocks.
constexpr void quantize_row(std::span<const float> x, std::span<const float> importance, std::span<block_q8_0> y)
{
    using namespace detail::quantize;
    check(32, y.size(), x.size(), importance.size());
    for (std::size_t i = 0; i < y.size(); ++i) {
        const auto b = x.subspan(i * 32, 32);
        float amax = 0.0f;
        for (const float v : b)
            amax = fmax(amax, abs(v));
        const float d = amax / 127.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        y[i].d = fp16(d);
        for (std::size_t j = 0; j < 32; ++j) {
            const float scaled = b[j] * id;
            y[i].qs[j] = static_cast<std::int8_t>(round_to_int(scaled));
        }
    }
}

/// \brief Quantises one row to Q8_1, as ggml's `quantize_row_q8_1_ref`.
/// \param x The row, `32 · y.size()` floats.
/// \param importance Ignored, as ggml ignores it.
/// \param y Receives the blocks.
constexpr void quantize_row(std::span<const float> x, std::span<const float> importance, std::span<block_q8_1> y)
{
    using namespace detail::quantize;
    check(32, y.size(), x.size(), importance.size());
    for (std::size_t i = 0; i < y.size(); ++i) {
        const auto b = x.subspan(i * 32, 32);
        float amax = 0.0f;
        for (const float v : b)
            amax = fmax(amax, abs(v));
        const float d = amax / 127.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        y[i].d = fp16(d);
        int sum = 0;
        for (std::size_t j = 0; j < 16; ++j) {
            const float v0 = b[j] * id;
            const float v1 = b[j + 16] * id;
            y[i].qs[j] = static_cast<std::int8_t>(round_to_int(v0));
            y[i].qs[j + 16] = static_cast<std::int8_t>(round_to_int(v1));
            sum += y[i].qs[j];
            sum += y[i].qs[j + 16];
        }
        const float s = float(sum) * d;
        y[i].s = fp16(s);
    }
}

/// \brief Quantises one row to Q4_0, as ggml's `quantize_q4_0`: the reference, or with importance weights a scale searched for the least weighted error.
/// \param x The row, `32 · y.size()` floats.
/// \param importance Empty, or one weight per number (ggml's importance matrix row).
/// \param y Receives the blocks.
constexpr void quantize_row(std::span<const float> x, std::span<const float> importance, std::span<block_q4_0> y)
{
    using namespace detail::quantize;
    check(32, y.size(), x.size(), importance.size());
    const float sigma2 = importance.empty() ? 0.0f : mean_square(x);
    for (std::size_t i = 0; i < y.size(); ++i) {
        const auto b = x.subspan(i * 32, 32);
        if (!importance.empty()) {
            std::array<float, 32> weight{};
            std::array<std::int8_t, 32> L{};
            for (std::size_t j = 0; j < 32; ++j)
                weight[j] = importance_weight(importance[i * 32 + j], sigma2, b[j]);
            const float d = make_qx_quants(32, 8, b.data(), L.data(), 1, weight.data());
            y[i].d = fp16(d);
            for (std::size_t j = 0; j < 16; ++j)
                y[i].qs[j] = static_cast<std::uint8_t>(L[j] | (L[j + 16] << 4));
            continue;
        }
        float amax = 0.0f, max = 0.0f;
        for (const float v : b)
            if (amax < abs(v)) {
                amax = abs(v);
                max = v;
            }
        const float d = max / -8.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        y[i].d = fp16(d);
        for (std::size_t j = 0; j < 16; ++j)
            y[i].qs[j] = static_cast<std::uint8_t>(offset_code(b[j], id, 8.5f, 15) | (offset_code(b[j + 16], id, 8.5f, 15) << 4));
    }
}

/// \brief Quantises one row to Q4_1, as ggml's `quantize_q4_1`: the reference (min and max), or with importance weights an affine fit searched for the least weighted error.
/// \param x The row, `32 · y.size()` floats.
/// \param importance Empty, or one weight per number.
/// \param y Receives the blocks.
constexpr void quantize_row(std::span<const float> x, std::span<const float> importance, std::span<block_q4_1> y)
{
    using namespace detail::quantize;
    check(32, y.size(), x.size(), importance.size());
    const float sigma2 = importance.empty() ? 0.0f : mean_square(x);
    for (std::size_t i = 0; i < y.size(); ++i) {
        const auto b = x.subspan(i * 32, 32);
        if (!importance.empty()) {
            std::array<float, 32> weight{};
            std::array<std::uint8_t, 32> L{}, Laux{};
            for (std::size_t j = 0; j < 32; ++j)
                weight[j] = importance_weight(importance[i * 32 + j], sigma2, b[j]);
            float the_min = 0;
            const float d = make_qkx_quants<true>(32, 15, b.data(), weight.data(), L.data(), &the_min, Laux.data(), -0.9f, 0.05f, 36, false);
            y[i].d = fp16(d);
            y[i].m = fp16(-the_min);
            for (std::size_t j = 0; j < 16; ++j)
                y[i].qs[j] = static_cast<std::uint8_t>(L[j] | (L[j + 16] << 4));
            continue;
        }
        float min = std::numeric_limits<float>::max(), max = -std::numeric_limits<float>::max();
        for (const float v : b) {
            min = v < min ? v : min;
            max = v > max ? v : max;
        }
        const float d = (max - min) / 15.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        y[i].d = fp16(d);
        y[i].m = fp16(min);
        for (std::size_t j = 0; j < 16; ++j)
            y[i].qs[j] = static_cast<std::uint8_t>(affine_code(b[j], min, id, 15) | (affine_code(b[j + 16], min, id, 15) << 4));
    }
}

/// \brief Quantises one row to Q5_0, as ggml's `quantize_q5_0`.
/// \param x The row, `32 · y.size()` floats.
/// \param importance Empty, or one weight per number.
/// \param y Receives the blocks.
constexpr void quantize_row(std::span<const float> x, std::span<const float> importance, std::span<block_q5_0> y)
{
    using namespace detail::quantize;
    check(32, y.size(), x.size(), importance.size());
    const float sigma2 = importance.empty() ? 0.0f : mean_square(x);
    for (std::size_t i = 0; i < y.size(); ++i) {
        const auto b = x.subspan(i * 32, 32);
        std::array<std::uint8_t, 32> q{};
        if (!importance.empty()) {
            std::array<float, 32> weight{};
            std::array<std::int8_t, 32> L{};
            for (std::size_t j = 0; j < 32; ++j)
                weight[j] = importance_weight(importance[i * 32 + j], sigma2, b[j]);
            y[i].d = fp16(make_qx_quants(32, 16, b.data(), L.data(), 1, weight.data()));
            for (std::size_t j = 0; j < 32; ++j)
                q[j] = static_cast<std::uint8_t>(L[j]);
        } else {
            float amax = 0.0f, max = 0.0f;
            for (const float v : b)
                if (amax < abs(v)) {
                    amax = abs(v);
                    max = v;
                }
            const float d = max / -16.0f;
            const float id = d != 0.0f ? 1.0f / d : 0.0f;
            y[i].d = fp16(d);
            for (std::size_t j = 0; j < 32; ++j)
                q[j] = offset_code(b[j], id, 16.5f, 31);
        }
        std::uint32_t qh = 0;
        for (std::size_t j = 0; j < 16; ++j) {
            y[i].qs[j] = static_cast<std::uint8_t>((q[j] & 0x0f) | ((q[j + 16] & 0x0f) << 4));
            qh |= std::uint32_t((q[j] & 0x10) >> 4) << j;
            qh |= std::uint32_t((q[j + 16] & 0x10) >> 4) << (j + 16);
        }
        for (std::size_t k = 0; k < 4; ++k)
            y[i].qh[k] = static_cast<std::uint8_t>(qh >> (8 * k));
    }
}

/// \brief Quantises one row to Q5_1, as ggml's `quantize_q5_1`.
/// \param x The row, `32 · y.size()` floats.
/// \param importance Empty, or one weight per number.
/// \param y Receives the blocks.
constexpr void quantize_row(std::span<const float> x, std::span<const float> importance, std::span<block_q5_1> y)
{
    using namespace detail::quantize;
    check(32, y.size(), x.size(), importance.size());
    const float sigma2 = importance.empty() ? 0.0f : mean_square(x);
    for (std::size_t i = 0; i < y.size(); ++i) {
        const auto b = x.subspan(i * 32, 32);
        std::array<std::uint8_t, 32> q{};
        if (!importance.empty()) {
            std::array<float, 32> weight{};
            std::array<std::uint8_t, 32> Laux{};
            for (std::size_t j = 0; j < 32; ++j)
                weight[j] = importance_weight(importance[i * 32 + j], sigma2, b[j]);
            float the_min = 0;
            const float d = make_qkx_quants<true>(32, 31, b.data(), weight.data(), q.data(), &the_min, Laux.data(), -0.9f, 0.05f, 36, false);
            y[i].d = fp16(d);
            y[i].m = fp16(-the_min);
        } else {
            float min = std::numeric_limits<float>::max(), max = -std::numeric_limits<float>::max();
            for (const float v : b) {
                min = v < min ? v : min;
                max = v > max ? v : max;
            }
            const float d = (max - min) / 31.0f;
            const float id = d != 0.0f ? 1.0f / d : 0.0f;
            y[i].d = fp16(d);
            y[i].m = fp16(min);
            for (std::size_t j = 0; j < 32; ++j)
                q[j] = affine_code(b[j], min, id, 31);
        }
        std::uint32_t qh = 0;
        for (std::size_t j = 0; j < 16; ++j) {
            y[i].qs[j] = static_cast<std::uint8_t>((q[j] & 0x0f) | ((q[j + 16] & 0x0f) << 4));
            qh |= std::uint32_t((q[j] & 0x10) >> 4) << j;
            qh |= std::uint32_t((q[j + 16] & 0x10) >> 4) << (j + 16);
        }
        for (std::size_t k = 0; k < 4; ++k)
            y[i].qh[k] = static_cast<std::uint8_t>(qh >> (8 * k));
    }
}

/// \brief Quantises one row to MXFP4, as ggml's `quantize_row_mxfp4_ref` (MXFP4 has no importance-weighted path).
/// \param x The row, `32 · y.size()` floats.
/// \param importance Ignored, as ggml ignores it.
/// \param y Receives the blocks.
constexpr void quantize_row(std::span<const float> x, std::span<const float> importance, std::span<block_mxfp4> y)
{
    using namespace detail::quantize;
    check(32, y.size(), x.size(), importance.size());
    for (std::size_t i = 0; i < y.size(); ++i) {
        const auto b = x.subspan(i * 32, 32);
        float amax = 0.0f;
        for (const float v : b)
            amax = amax < abs(v) ? abs(v) : amax;
        // (uint8_t)(floorf(log2f(amax)) - 2 + 127): an infinity converts as x86 converts it, to 0, and an amax
        // below 2^-125 wraps, as the conversion of a negative number does on x86
        const int exponent = !(amax > 0.0f) || amax > std::numeric_limits<float>::max() ? 0 : floor_log2(amax) - 2 + 127;
        y[i].e = static_cast<std::uint8_t>(exponent);
        const float half = e8m0_half(y[i].e);
        for (std::size_t j = 0; j < 16; ++j)
            y[i].qs[j] = static_cast<std::uint8_t>(nearest_fp4(b[j], half) | (nearest_fp4(b[j + 16], half) << 4));
    }
}

/// \brief Quantises one row to NVFP4, as ggml's `quantize_row_nvfp4_ref`: per 16 numbers a UE4M3 scale amax / 6 and the nearest E2M1 codes.
/// \param x The row, `64 · y.size()` floats.
/// \param importance Ignored, as ggml ignores it.
/// \param y Receives the blocks.
constexpr void quantize_row(std::span<const float> x, std::span<const float> importance, std::span<block_nvfp4> y)
{
    using namespace detail::quantize;
    check(64, y.size(), x.size(), importance.size());
    for (std::size_t i = 0; i < y.size(); ++i) {
        for (std::size_t s = 0; s < 4; ++s) {
            const auto b = x.subspan(i * 64 + s * 16, 16);
            float amax = 0.0f;
            for (const float v : b)
                amax = amax < abs(v) ? abs(v) : amax;
            const float scale = amax / 6.0f;
            const std::uint8_t ue = fp32_to_ue4m3(scale);
            y[i].d[s] = ue;
            const float d = ue4m3_half(ue);
            for (std::size_t j = 0; j < 8; ++j)
                y[i].qs[s * 8 + j] = static_cast<std::uint8_t>(nearest_fp4(b[j], d) | (nearest_fp4(b[j + 8], d) << 4));
        }
    }
}

/// \brief Quantises one row to Q1_0, as ggml's `quantize_row_q1_0_ref`: the mean absolute value and the signs.
/// \param x The row, `128 · y.size()` floats.
/// \param importance Ignored, as ggml ignores it.
/// \param y Receives the blocks.
constexpr void quantize_row(std::span<const float> x, std::span<const float> importance, std::span<block_q1_0> y)
{
    using namespace detail::quantize;
    check(128, y.size(), x.size(), importance.size());
    for (std::size_t i = 0; i < y.size(); ++i) {
        const auto b = x.subspan(i * 128, 128);
        float sum_abs = 0.0f;
        for (const float v : b)
            sum_abs += abs(v);
        y[i].d = fp16(sum_abs / 128.0f);
        y[i].qs = {};
        for (std::size_t j = 0; j < 128; ++j)
            if (b[j] >= 0.0f)
                y[i].qs[j / 8] = static_cast<std::uint8_t>(y[i].qs[j / 8] | (1u << (j % 8)));
    }
}

/// \brief Quantises one row to Q2_0, as ggml's `quantize_row_q2_0_ref`: `round(x / amax) + 1` clamped to [0, 3].
/// \param x The row, `64 · y.size()` floats.
/// \param importance Ignored, as ggml ignores it.
/// \param y Receives the blocks.
constexpr void quantize_row(std::span<const float> x, std::span<const float> importance, std::span<block_q2_0> y)
{
    using namespace detail::quantize;
    check(64, y.size(), x.size(), importance.size());
    for (std::size_t i = 0; i < y.size(); ++i) {
        const auto b = x.subspan(i * 64, 64);
        float amax = 0.0f;
        for (const float v : b)
            amax = abs(v) > amax ? abs(v) : amax;
        const float id = amax > 0.0f ? 1.0f / amax : 0.0f;
        y[i].d = fp16(amax);
        y[i].qs = {};
        for (std::size_t j = 0; j < 64; ++j) {
            const float scaled = b[j] * id;
            const int q = imax(0, imin(3, round_to_int(scaled) + 1));
            y[i].qs[j / 4] = static_cast<std::uint8_t>(y[i].qs[j / 4] | (q << (2 * (j % 4))));
        }
    }
}

/// \brief Quantises one row to Q8_K, as ggml's `quantize_row_q8_K_ref`: the activations of the super-block formats.
/// \param x The row, `256 · y.size()` floats.
/// \param importance Ignored.
/// \param y Receives the blocks.
constexpr void quantize_row(std::span<const float> x, std::span<const float> importance, std::span<block_q8_k> y)
{
    using namespace detail::quantize;
    check(256, y.size(), x.size(), importance.size());
    for (std::size_t i = 0; i < y.size(); ++i) {
        const auto b = x.subspan(i * 256, 256);
        float max = 0, amax = 0;
        for (const float v : b)
            if (abs(v) > amax) {
                amax = abs(v);
                max = v;
            }
        y[i].qs = {};
        y[i].bsums = {};
        if (amax == 0.0f) {
            y[i].d = 0.0f;
            continue;
        }
        const float iscale = -127.0f / max;
        for (std::size_t j = 0; j < 256; ++j) {
            const float v = iscale * b[j];
            y[i].qs[j] = static_cast<std::int8_t>(imin(127, nearest_int(v)));
        }
        for (std::size_t j = 0; j < 16; ++j) {
            int sum = 0;
            for (std::size_t k = 0; k < 16; ++k)
                sum += y[i].qs[16 * j + k];
            y[i].bsums[j] = static_cast<std::int16_t>(sum);
        }
        y[i].d = 1 / iscale;
    }
}

/// \brief Dequantises Q8_0 blocks: `x = d · q`, as ggml's `dequantize_row_q8_0`.
/// \param x The blocks.
/// \param y Receives `32 · x.size()` floats.
constexpr void dequantize_row(std::span<const block_q8_0> x, std::span<float> y)
{
    assert(y.size() == 32 * x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        const float d = to_float(x[i].d);
        for (std::size_t j = 0; j < 32; ++j)
            y[i * 32 + j] = float(x[i].qs[j]) * d;
    }
}

/// \brief Dequantises Q8_1 blocks: `x = d · q` (ggml has no `dequantize_row_q8_1`; this is its definition).
/// \param x The blocks.
/// \param y Receives `32 · x.size()` floats.
constexpr void dequantize_row(std::span<const block_q8_1> x, std::span<float> y)
{
    assert(y.size() == 32 * x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        const float d = to_float(x[i].d);
        for (std::size_t j = 0; j < 32; ++j)
            y[i * 32 + j] = float(x[i].qs[j]) * d;
    }
}

/// \brief Dequantises Q4_0 blocks: `x = d · (q - 8)`, as ggml's `dequantize_row_q4_0`.
/// \param x The blocks.
/// \param y Receives `32 · x.size()` floats.
constexpr void dequantize_row(std::span<const block_q4_0> x, std::span<float> y)
{
    assert(y.size() == 32 * x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        const float d = to_float(x[i].d);
        for (std::size_t j = 0; j < 16; ++j) {
            y[i * 32 + j] = float((x[i].qs[j] & 0x0f) - 8) * d;
            y[i * 32 + j + 16] = float((x[i].qs[j] >> 4) - 8) * d;
        }
    }
}

/// \brief Dequantises Q4_1 blocks: `x = d · q + m`, as ggml's `dequantize_row_q4_1`.
/// \param x The blocks.
/// \param y Receives `32 · x.size()` floats.
constexpr void dequantize_row(std::span<const block_q4_1> x, std::span<float> y)
{
    assert(y.size() == 32 * x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        const float d = to_float(x[i].d), m = to_float(x[i].m);
        for (std::size_t j = 0; j < 16; ++j) {
            const float a = float(x[i].qs[j] & 0x0f) * d;
            const float b = float(x[i].qs[j] >> 4) * d;
            y[i * 32 + j] = a + m;
            y[i * 32 + j + 16] = b + m;
        }
    }
}

/// \brief Dequantises Q5_0 blocks: `x = d · (q - 16)`, as ggml's `dequantize_row_q5_0`.
/// \param x The blocks.
/// \param y Receives `32 · x.size()` floats.
constexpr void dequantize_row(std::span<const block_q5_0> x, std::span<float> y)
{
    assert(y.size() == 32 * x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        const float d = to_float(x[i].d);
        const std::uint32_t qh = x[i].qh[0] | (std::uint32_t(x[i].qh[1]) << 8) | (std::uint32_t(x[i].qh[2]) << 16) | (std::uint32_t(x[i].qh[3]) << 24);
        for (std::size_t j = 0; j < 16; ++j) {
            const int x0 = int((x[i].qs[j] & 0x0f) | (((qh >> j) << 4) & 0x10)) - 16;
            const int x1 = int((x[i].qs[j] >> 4) | ((qh >> (j + 12)) & 0x10)) - 16;
            y[i * 32 + j] = float(x0) * d;
            y[i * 32 + j + 16] = float(x1) * d;
        }
    }
}

/// \brief Dequantises Q5_1 blocks: `x = d · q + m`, as ggml's `dequantize_row_q5_1`.
/// \param x The blocks.
/// \param y Receives `32 · x.size()` floats.
constexpr void dequantize_row(std::span<const block_q5_1> x, std::span<float> y)
{
    assert(y.size() == 32 * x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        const float d = to_float(x[i].d), m = to_float(x[i].m);
        const std::uint32_t qh = x[i].qh[0] | (std::uint32_t(x[i].qh[1]) << 8) | (std::uint32_t(x[i].qh[2]) << 16) | (std::uint32_t(x[i].qh[3]) << 24);
        for (std::size_t j = 0; j < 16; ++j) {
            const int x0 = int((x[i].qs[j] & 0x0f) | (((qh >> j) << 4) & 0x10));
            const int x1 = int((x[i].qs[j] >> 4) | ((qh >> (j + 12)) & 0x10));
            const float a = float(x0) * d;
            const float b = float(x1) * d;
            y[i * 32 + j] = a + m;
            y[i * 32 + j + 16] = b + m;
        }
    }
}

/// \brief Dequantises MXFP4 blocks: `x = 2^(e - 127) · v`, as ggml's `dequantize_row_mxfp4`.
/// \param x The blocks.
/// \param y Receives `32 · x.size()` floats.
constexpr void dequantize_row(std::span<const block_mxfp4> x, std::span<float> y)
{
    assert(y.size() == 32 * x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        const float half = detail::quantize::e8m0_half(x[i].e);
        for (std::size_t j = 0; j < 16; ++j) {
            y[i * 32 + j] = half * float(detail::quantize::fp4_doubled[x[i].qs[j] & 0x0f]);
            y[i * 32 + j + 16] = half * float(detail::quantize::fp4_doubled[x[i].qs[j] >> 4]);
        }
    }
}

/// \brief Dequantises NVFP4 blocks: each 16 numbers `x = s · v`, as ggml's `dequantize_row_nvfp4`.
/// \param x The blocks.
/// \param y Receives `64 · x.size()` floats.
constexpr void dequantize_row(std::span<const block_nvfp4> x, std::span<float> y)
{
    assert(y.size() == 64 * x.size());
    for (std::size_t i = 0; i < x.size(); ++i)
        for (std::size_t s = 0; s < 4; ++s) {
            const float d = detail::quantize::ue4m3_half(x[i].d[s]);
            for (std::size_t j = 0; j < 8; ++j) {
                y[i * 64 + s * 16 + j] = d * float(detail::quantize::fp4_doubled[x[i].qs[s * 8 + j] & 0x0f]);
                y[i * 64 + s * 16 + j + 8] = d * float(detail::quantize::fp4_doubled[x[i].qs[s * 8 + j] >> 4]);
            }
        }
}

/// \brief Dequantises Q1_0 blocks: `x = ±d`, as ggml's `dequantize_row_q1_0`.
/// \param x The blocks.
/// \param y Receives `128 · x.size()` floats.
constexpr void dequantize_row(std::span<const block_q1_0> x, std::span<float> y)
{
    assert(y.size() == 128 * x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        const float d = to_float(x[i].d);
        const float neg = -d;
        for (std::size_t j = 0; j < 128; ++j)
            y[i * 128 + j] = (x[i].qs[j / 8] >> (j % 8)) & 1 ? d : neg;
    }
}

/// \brief Dequantises Q2_0 blocks: `x = d · (q - 1)`, as ggml's `dequantize_row_q2_0`.
/// \param x The blocks.
/// \param y Receives `64 · x.size()` floats.
constexpr void dequantize_row(std::span<const block_q2_0> x, std::span<float> y)
{
    assert(y.size() == 64 * x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        const float d = to_float(x[i].d);
        for (std::size_t j = 0; j < 64; ++j)
            y[i * 64 + j] = float(int((x[i].qs[j / 4] >> (2 * (j % 4))) & 3) - 1) * d;
    }
}

/// \brief Dequantises Q8_K blocks: `x = d · q`, as ggml's `dequantize_row_q8_K`.
/// \param x The blocks.
/// \param y Receives `256 · x.size()` floats.
constexpr void dequantize_row(std::span<const block_q8_k> x, std::span<float> y)
{
    assert(y.size() == 256 * x.size());
    for (std::size_t i = 0; i < x.size(); ++i)
        for (std::size_t j = 0; j < 256; ++j)
            y[i * 256 + j] = x[i].d * float(x[i].qs[j]);
}

/// \brief Quantises a row of floats into blocks, as ggml's reference `quantize_row_<type>_ref`.
/// \tparam Block A `quantized_block` with a `quantize_row`.
/// \param x The row, `Block::elements · blocks.size()` floats.
/// \param blocks Receives the blocks.
template <quantized_block Block, std::size_t N, row X>
    requires std::same_as<row_value_t<X>, float>
constexpr void quantize(const X& x, std::span<Block, N> blocks)
{
    quantize_row(std::span<const float>(as_row(x)), std::span<const float>{}, std::span<Block>(blocks));
}

/// \brief Quantises one row with an importance matrix row, as ggml's `quantize_<type>(x, y, 1, n, importance)`.
/// \tparam Block A `quantized_block` with a `quantize_row`.
/// \param x The row, `Block::elements · blocks.size()` floats.
/// \param importance One weight per number (empty: the reference quantisation).
/// \param blocks Receives the blocks.
template <quantized_block Block, std::size_t N, row X, row W>
    requires std::same_as<row_value_t<X>, float> && std::same_as<row_value_t<W>, float>
constexpr void quantize(const X& x, const W& importance, std::span<Block, N> blocks)
{
    quantize_row(std::span<const float>(as_row(x)), std::span<const float>(as_row(importance)), std::span<Block>(blocks));
}

/// \brief Dequantises blocks into a row of floats.
/// \tparam Block A `quantized_block` with a `dequantize_row`.
/// \param blocks The blocks.
/// \param x Receives `Block::elements · blocks.size()` floats.
template <quantized_block Block, std::size_t N, mutable_row X>
    requires std::same_as<row_value_t<X>, float>
constexpr void dequantize(std::span<const Block, N> blocks, X&& x)
{
    dequantize_row(std::span<const Block>(blocks), std::span<float>(as_mutable_row(x)));
}

/// \brief The dot product of two Q8_0 rows, as ggml's `ggml_vec_dot_q8_0_q8_0`.
/// \param x The weights.
/// \param y The activations.
/// \return `sum over blocks of (sum q_x · q_y) · d_x · d_y`.
constexpr float dot(std::span<const block_q8_0> x, std::span<const block_q8_0> y)
{
    assert(x.size() == y.size());
    float sum = 0.0f;
    for (std::size_t i = 0; i < x.size(); ++i) {
        int s = 0;
        for (std::size_t j = 0; j < quant_block_size; ++j)
            s += int(x[i].qs[j]) * int(y[i].qs[j]);
        const float scale = to_float(x[i].d) * to_float(y[i].d);
        const float term = float(s) * scale;
        sum += term;
    }
    return sum;
}

/// \brief The dot product of a Q4_0 row with a Q8_0 row, as ggml's `ggml_vec_dot_q4_0_q8_0`.
/// \param x The weights.
/// \param y The activations.
/// \return `sum over blocks of (sum (q_x - 8) · q_y) · d_x · d_y`.
constexpr float dot(std::span<const block_q4_0> x, std::span<const block_q8_0> y)
{
    assert(x.size() == y.size());
    constexpr std::size_t h = quant_block_size / 2;
    float sum = 0.0f;
    for (std::size_t i = 0; i < x.size(); ++i) {
        int s0 = 0, s1 = 0;
        for (std::size_t j = 0; j < h; ++j) {
            s0 += ((x[i].qs[j] & 0x0f) - 8) * int(y[i].qs[j]);
            s1 += ((x[i].qs[j] >> 4) - 8) * int(y[i].qs[j + h]);
        }
        const float term = float(s0 + s1) * to_float(x[i].d) * to_float(y[i].d);
        sum += term;
    }
    return sum;
}

/// \brief The dot product of a Q4_1 row with a Q8_1 row, as ggml's `ggml_vec_dot_q4_1_q8_1`.
/// \param x The weights.
/// \param y The activations.
/// \return `sum over blocks of d_x d_y (sum q_x · q_y) + m_x s_y`.
constexpr float dot(std::span<const block_q4_1> x, std::span<const block_q8_1> y)
{
    assert(x.size() == y.size());
    float sum = 0.0f;
    for (std::size_t i = 0; i < x.size(); ++i) {
        int s0 = 0, s1 = 0;
        for (std::size_t j = 0; j < 16; ++j) {
            s0 += (x[i].qs[j] & 0x0f) * int(y[i].qs[j]);
            s1 += (x[i].qs[j] >> 4) * int(y[i].qs[j + 16]);
        }
        const float dd = to_float(x[i].d) * to_float(y[i].d);
        const float a = dd * float(s0 + s1);
        const float b = to_float(x[i].m) * to_float(y[i].s);
        const float term = a + b;
        sum += term;
    }
    return sum;
}

/// \brief The dot product of a Q5_0 row with a Q8_0 row, as ggml's `ggml_vec_dot_q5_0_q8_0`.
/// \param x The weights.
/// \param y The activations.
/// \return `sum over blocks of (sum (q_x - 16) · q_y) · d_x · d_y`.
constexpr float dot(std::span<const block_q5_0> x, std::span<const block_q8_0> y)
{
    assert(x.size() == y.size());
    float sum = 0.0f;
    for (std::size_t i = 0; i < x.size(); ++i) {
        const std::uint32_t qh = x[i].qh[0] | (std::uint32_t(x[i].qh[1]) << 8) | (std::uint32_t(x[i].qh[2]) << 16) | (std::uint32_t(x[i].qh[3]) << 24);
        int s0 = 0, s1 = 0;
        for (std::size_t j = 0; j < 16; ++j) {
            const int h0 = int(((qh & (1u << j)) >> j) << 4);
            const int h1 = int((qh & (1u << (j + 16))) >> (j + 12));
            s0 += (((x[i].qs[j] & 0x0f) | h0) - 16) * int(y[i].qs[j]);
            s1 += (((x[i].qs[j] >> 4) | h1) - 16) * int(y[i].qs[j + 16]);
        }
        const float dd = to_float(x[i].d) * to_float(y[i].d);
        const float term = dd * float(s0 + s1);
        sum += term;
    }
    return sum;
}

/// \brief The dot product of a Q5_1 row with a Q8_1 row, as ggml's `ggml_vec_dot_q5_1_q8_1`.
/// \param x The weights.
/// \param y The activations.
/// \return `sum over blocks of d_x d_y (sum q_x · q_y) + m_x s_y`.
constexpr float dot(std::span<const block_q5_1> x, std::span<const block_q8_1> y)
{
    assert(x.size() == y.size());
    float sum = 0.0f;
    for (std::size_t i = 0; i < x.size(); ++i) {
        const std::uint32_t qh = x[i].qh[0] | (std::uint32_t(x[i].qh[1]) << 8) | (std::uint32_t(x[i].qh[2]) << 16) | (std::uint32_t(x[i].qh[3]) << 24);
        int s0 = 0, s1 = 0;
        for (std::size_t j = 0; j < 16; ++j) {
            const int h0 = int(((qh >> j) << 4) & 0x10);
            const int h1 = int((qh >> (j + 12)) & 0x10);
            s0 += ((x[i].qs[j] & 0x0f) | h0) * int(y[i].qs[j]);
            s1 += ((x[i].qs[j] >> 4) | h1) * int(y[i].qs[j + 16]);
        }
        const float dd = to_float(x[i].d) * to_float(y[i].d);
        const float a = dd * float(s0 + s1);
        const float b = to_float(x[i].m) * to_float(y[i].s);
        const float term = a + b;
        sum += term;
    }
    return sum;
}

/// \brief The dot product of an MXFP4 row with a Q8_0 row, as ggml's `ggml_vec_dot_mxfp4_q8_0`.
/// \param x The weights.
/// \param y The activations.
/// \return `sum over blocks of d_y · 2^(e - 128) · (sum 2v_x · q_y)`.
constexpr float dot(std::span<const block_mxfp4> x, std::span<const block_q8_0> y)
{
    assert(x.size() == y.size());
    constexpr std::size_t h = quant_block_size / 2;
    float sum = 0.0f;
    for (std::size_t i = 0; i < x.size(); ++i) {
        const float d = to_float(y[i].d) * detail::quantize::e8m0_half(x[i].e);
        int s1 = 0, s2 = 0;
        for (std::size_t j = 0; j < h; ++j) {
            s1 += int(y[i].qs[j]) * detail::quantize::fp4_doubled[x[i].qs[j] & 0x0f];
            s2 += int(y[i].qs[j + h]) * detail::quantize::fp4_doubled[x[i].qs[j] >> 4];
        }
        const float term = d * float(s1 + s2);
        sum += term;
    }
    return sum;
}

/// \brief The dot product of an NVFP4 row with a Q8_0 row (two Q8_0 blocks per NVFP4 block), as ggml's `ggml_vec_dot_nvfp4_q8_0`.
/// \param x The weights.
/// \param y The activations, `2 · x.size()` blocks.
/// \return `sum over groups of 16 of d_y · s · (sum 2v_x · q_y)`.
constexpr float dot(std::span<const block_nvfp4> x, std::span<const block_q8_0> y)
{
    assert(2 * x.size() == y.size());
    float sum = 0.0f;
    for (std::size_t i = 0; i < x.size(); ++i)
        for (std::size_t s = 0; s < 4; ++s) {
            const float d = detail::quantize::ue4m3_half(x[i].d[s]);
            const block_q8_0& b = y[2 * i + s / 2];
            const std::size_t off = (s % 2) * 16;
            const float dy = to_float(b.d);
            int lo = 0, hi = 0;
            for (std::size_t j = 0; j < 8; ++j) {
                const std::uint8_t q = x[i].qs[s * 8 + j];
                lo += int(b.qs[off + j]) * detail::quantize::fp4_doubled[q & 0x0f];
                hi += int(b.qs[off + j + 8]) * detail::quantize::fp4_doubled[q >> 4];
            }
            const float dd = dy * d;
            const float term = dd * float(lo + hi);
            sum += term;
        }
    return sum;
}

/// \brief The dot product of a Q1_0 row with a Q8_0 row (four Q8_0 blocks per Q1_0 block), as ggml's `ggml_vec_dot_q1_0_q8_0`.
/// \param x The weights.
/// \param y The activations, `4 · x.size()` blocks.
/// \return `sum over blocks of d_x · sum over its Q8_0 blocks of d_y · (sum ±q_y)`.
constexpr float dot(std::span<const block_q1_0> x, std::span<const block_q8_0> y)
{
    assert(4 * x.size() == y.size());
    float sum = 0.0f;
    for (std::size_t i = 0; i < x.size(); ++i) {
        float sumi = 0.0f;
        for (std::size_t k = 0; k < 4; ++k) {
            const block_q8_0& b = y[4 * i + k];
            int s = 0;
            for (std::size_t j = 0; j < 32; ++j)
                s += (x[i].qs[k * 4 + j / 8] >> (j % 8)) & 1 ? int(b.qs[j]) : -int(b.qs[j]);
            const float term = to_float(b.d) * float(s);
            sumi += term;
        }
        const float term = to_float(x[i].d) * sumi;
        sum += term;
    }
    return sum;
}

/// \brief The dot product of a Q2_0 row with a Q8_0 row (two Q8_0 blocks per Q2_0 block), as ggml's `ggml_vec_dot_q2_0_q8_0`.
/// \param x The weights.
/// \param y The activations, `2 · x.size()` blocks.
/// \return `sum over blocks of d_x · sum over its Q8_0 blocks of d_y · (sum (q_x - 1) · q_y)`.
constexpr float dot(std::span<const block_q2_0> x, std::span<const block_q8_0> y)
{
    assert(2 * x.size() == y.size());
    float sum = 0.0f;
    for (std::size_t i = 0; i < x.size(); ++i) {
        float sumi = 0.0f;
        for (std::size_t k = 0; k < 2; ++k) {
            const block_q8_0& b = y[2 * i + k];
            int s = 0;
            for (std::size_t j = 0; j < 32; ++j)
                s += (int((x[i].qs[k * 8 + j / 4] >> (2 * (j % 4))) & 3) - 1) * int(b.qs[j]);
            const float term = to_float(b.d) * float(s);
            sumi += term;
        }
        const float term = to_float(x[i].d) * sumi;
        sum += term;
    }
    return sum;
}

/// \brief Rows of `float` to OCP FP8 E4M3, as ggml's `quantize_row_f8_e4m3_ref`: nearest even, saturating at ±448, NaN to `0x7f`.
/// \param x The row.
/// \param y Receives `x.size()` codes.
constexpr void quantize_row(std::span<const float> x, std::span<float8_e4m3> y)
{
    assert(x.size() == y.size());
    for (std::size_t i = 0; i < x.size(); ++i)
        y[i] = x[i] != x[i] ? float8_e4m3::from_bits(0x7f) : to_minifloat<float8_e4m3>(x[i], overflow_policy::saturate);
}

/// \brief OCP FP8 E4M3 codes to `float`, as ggml's `dequantize_row_f8_e4m3`: exact, every NaN code a positive NaN.
/// \param x The codes.
/// \param y Receives `x.size()` floats.
constexpr void dequantize_row(std::span<const float8_e4m3> x, std::span<float> y)
{
    assert(x.size() == y.size());
    for (std::size_t i = 0; i < x.size(); ++i)
        y[i] = (x[i].bits & 0x7f) == 0x7f ? std::numeric_limits<float>::quiet_NaN() : to_float(x[i]);
}

/// \brief The dot product of an FP8 E4M3 row with a `float` row, in the order of ggml's `ggml_vec_dot_f8_e4m3_f32` on x86 (SSE3).
///
/// ggml's loop is SIMD: per 32 numbers, eight accumulators of four lanes each add their products; the 32 lanes then
/// reduce pairwise (accumulators `i` and `i + 4`, then `i + 2`, then `i + 1`, then the four lanes as `(0 + 1) +
/// (2 + 3)`), and the numbers past the last 32 add one by one. That order is the reference here; ggml's order on
/// another SIMD target (NEON, AVX) differs in the last bits.
/// \param x The weights.
/// \param y The activations.
/// \return `sum x_i · y_i`.
constexpr float dot(std::span<const float8_e4m3> x, std::span<const float> y)
{
    assert(x.size() == y.size());
    const auto value = [&x](std::size_t i) { return (x[i].bits & 0x7f) == 0x7f ? std::numeric_limits<float>::quiet_NaN() : to_float(x[i]); };
    const std::size_t np = x.size() - x.size() % 32;
    std::array<float, 32> acc{};   // accumulator j, lane l at 4 j + l
    for (std::size_t pos = 0; pos < np; pos += 32)
        for (std::size_t k = 0; k < 32; ++k) {
            const float term = value(pos + k) * y[pos + k];
            acc[k] += term;
        }
    for (std::size_t step = 16; step >= 4; step /= 2)
        for (std::size_t k = 0; k < step; ++k)
            acc[k] += acc[k + step];
    const float low = acc[0] + acc[1];
    const float high = acc[2] + acc[3];
    float sum = np == 0 ? 0.0f : low + high;
    for (std::size_t i = np; i < x.size(); ++i) {
        const float term = value(i) * y[i];
        sum += term;
    }
    return sum;
}

/// \brief Symmetric int8 quantisation of a whole row with one scale: `q = round(x / scale)`, `scale = amax / 127`.
/// \param x The row.
/// \param q Receives `x.size()` signed bytes.
/// \return The scale; 0 for a row of zeros.
template <row X>
    requires std::same_as<row_value_t<X>, float>
constexpr float quantize_int8(const X& x, std::span<std::int8_t> q)
{
    const auto in = as_row(x);
    assert(q.size() == in.size());
    float amax = 0.0f;
    for (const float v : in)
        amax = amax < detail::quantize::abs(v) ? detail::quantize::abs(v) : amax;
    const float scale = amax / 127.0f;
    const float inverse = scale != 0.0f ? 1.0f / scale : 0.0f;
    for (std::size_t i = 0; i < in.size(); ++i) {
        const float scaled = in[i] * inverse;
        q[i] = static_cast<std::int8_t>(detail::quantize::round_to_int(scaled));
    }
    return scale;
}

/// \brief Dequantises a symmetric int8 row: `x = q · scale`.
/// \param q The signed bytes.
/// \param scale What `quantize_int8` returned.
/// \param x Receives `q.size()` floats.
template <mutable_row X>
    requires std::same_as<row_value_t<X>, float>
constexpr void dequantize_int8(std::span<const std::int8_t> q, float scale, X&& x)
{
    const auto out = as_mutable_row(x);
    assert(out.size() == q.size());
    for (std::size_t i = 0; i < q.size(); ++i)
        out[i] = float(q[i]) * scale;
}

} // namespace snicholls::ml

#endif // snicholls_core_ml_quantize_hpp
