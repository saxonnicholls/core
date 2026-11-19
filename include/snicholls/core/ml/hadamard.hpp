//
//  hadamard.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::ml` Hadamard rotations: the fast Walsh-Hadamard transform, the orthonormal and block forms ggml computes bit for bit, the randomised transform of QuaRot and SpinQuant, and folding a rotation into weights before quantisation, all `constexpr`.
/// \ingroup core
///
/// **Overview.** A Hadamard matrix `H` of order `n = 2^k` (Sylvester's: `H_ij = (-1)^popcount(i & j)`) is
/// orthogonal up to `sqrt(n)`, and the fast transform applies it in `n log2 n` additions. Rotating a weight
/// matrix and its input by the same orthonormal `Q` changes no product, `W x = (W Q)(Q^T x)`, but spreads each
/// row's outliers over all its columns, so the rotated weights and activations quantise with far less error. That
/// is the pre-step of QuaRot and SpinQuant (`Q = H D`, `D` a diagonal of random signs), and of llama.cpp's folded
/// weights and KV-cache rotation.
///
/// - `fwht(x)`: `x <- H x`, unnormalised, in place. `hadamard(x)`: `x <- H x / sqrt(n)`, orthonormal and its own
///   inverse, as ggml computes it. `block_hadamard(x, block)`: the same on each run of `block` numbers, ggml's form
///   for a width that is not a power of two (a block-diagonal `H`).
/// - `hadamard_signs(signs, seed)`: the ±1 diagonal `D` from a seed (`stats::wyrand`). `random_hadamard(x, signs,
///   block)`: `x <- H (D x)`, the input transform; `random_hadamard_inverse(x, signs, block)`: `x <- D (H x)`.
/// - `rotate_rows(w, cols, signs, block)`: each row `r` of a row-major `W` becomes `r D H`, so the folded
///   `W_f = W D H` times the transformed input `H (D x)` is `W x`; then quantise `W_f` (and the transformed
///   activations) with `ml/quantize.hpp`. The gain is largest where one scale covers many numbers: per-row int8 of
///   a row with two outliers loses ten times less after rotation (tested).
/// - `hadamard_entry(i, j)` and `hadamard_matrix<T>(n, m)`: the matrix itself, for tests and GPU kernels; the
///   matrix is the one llama.cpp builds (`ggml_gen_hadamard`), bit for bit.
///
/// \snippet ml/hadamard_demo.cpp hadamard
///
/// **Relationships.** `ml/quantize.hpp` (what the rotation prepares for), `stats/random.hpp` (`wyrand`),
/// `ml/row.hpp`. Group: \ref core.
///
/// **Implementations.** One butterfly order: the row is scaled first (orthonormal form), then passes of length 1,
/// 2, 4, ... `n / 2`, each pass over pairs `(i + j, i + len + j)` in increasing `i`, then `j`, writing `u + v` and
/// `u - v`. That is ggml's CPU order (`ggml_compute_forward_fwht`, the `GGML_HINT_SRC0_IS_HADAMARD` matrix product);
/// ggml's SIMD passes give the same bits, since `FMA(u, v, -1)` rounds once as `u - v` does. Every addition rounds
/// once in `T`, so the result is the same on every machine.
///
/// **Guarantees.** `constexpr`, `noexcept`, in place, no allocation; `O(n log n)` for one block, `O(n log block)` for
/// a row. `hadamard` and `block_hadamard` of a `float` row equal ggml's bits on every case of the committed fixture
/// (`tests/fixtures/ml/hadamard_fixture.txt`, from `scripts/generate_hadamard_fixture.cpp` built from ggml's own
/// loop). `random_hadamard_inverse` undoes `random_hadamard` to rounding (exactly when `n` is a power of four and
/// no sum rounds).
///
/// **Errors.** Preconditions, asserted: `n` (or `block`) is a power of two; the row length is a multiple of
/// `block`; `signs` has the row's length and holds only ±1. NaN and infinity propagate as IEEE arithmetic says.
///
/// **Performance.** Against ggml's loop written by hand (`benchmarks/ml/hadamard_benchmark.cpp`), a row of 4,096
/// floats: one rotation of 4,096 level, 480 million numbers per second (8.6 us); blocks of 128, 665 against 556 million
/// (Xeon W-3245, Release, Apple clang 17, 2026-11-19, one thread, scalar code).
///
/// **Tests and demos.** `tests/ml/hadamard_test.cpp`, `demos/ml/hadamard_demo.cpp`.
///
/// **Dependencies.** `<bit>`, `<cassert>`, `<concepts>`, `<cstddef>`, `<cstdint>`, `<span>`,
/// `snicholls/core/math/constexpr_math.hpp`, `ml/row.hpp`, `stats/random.hpp` (which includes `<random>`).
///
/// **Known issues.** ggml computes the scale `1.0f / sqrtf(n)` in `float`, but llama.cpp's rotation matrix holds
/// `1.0 / sqrtf(n)` computed in `double` and rounded: the two differ in the last bit for some `n`, and each function
/// here follows the one it matches. A block format with a scale per 32 (Q4_0, Q8_0) already confines an outlier to
/// its own block; rotating spreads it into every block, so on sparse outliers the rotation can raise a block
/// format's error: rotate for per-channel or per-token scales, or measure. Non-power-of-two widths are handled only as ggml handles them, by blocks; a
/// Hadamard matrix of order 12, 20 or 172 (Paley's) is not here.
///
/// **Status.** Experimental, new in 0.17.0.

#ifndef snicholls_core_ml_hadamard_hpp
#define snicholls_core_ml_hadamard_hpp

#include <snicholls/core/math/constexpr_math.hpp>
#include <snicholls/core/ml/row.hpp>
#include <snicholls/core/stats/random.hpp>

#include <bit>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <span>

namespace snicholls::ml
{

namespace detail::hadamard
{
    // The butterflies, ggml's order: passes of length 1, 2, 4, ...; within a pass, pairs (i + j, i + len + j)
    template <typename T>
    constexpr void butterflies(T* x, std::size_t n) noexcept
    {
        for (std::size_t len = 1; len < n; len <<= 1)
            for (std::size_t i = 0; i < n; i += 2 * len)
                for (std::size_t j = 0; j < len; ++j) {
                    const T u = x[i + j];
                    const T v = x[i + len + j];
                    x[i + j] = u + v;
                    x[i + len + j] = u - v;
                }
    }

    // 1 / sqrt(n) in T, as ggml's 1.0f / sqrtf((float) n)
    template <typename T>
    constexpr T inverse_sqrt(std::size_t n) noexcept
    {
        const T root = math::sqrt(static_cast<T>(n));
        return T(1) / root;
    }

    template <typename T>
    constexpr void scale(T* x, std::size_t n, T s) noexcept
    {
        for (std::size_t j = 0; j < n; ++j)
            x[j] = x[j] * s;
    }
} // namespace detail::hadamard

/// \brief Whether `n` is a power of two (1 included): the orders the transforms here take.
/// \param n The order.
/// \return `true` for 1, 2, 4, ...
[[nodiscard]] constexpr bool is_hadamard_order(std::size_t n) noexcept
{
    return std::has_single_bit(n);
}

/// \brief The entry of Sylvester's Hadamard matrix: `H_ij = (-1)^popcount(i & j)`.
/// \param i The row.
/// \param j The column.
/// \return 1 or -1.
[[nodiscard]] constexpr int hadamard_entry(std::size_t i, std::size_t j) noexcept
{
    return std::popcount(i & j) % 2 == 0 ? 1 : -1;
}

/// \brief The unnormalised fast Walsh-Hadamard transform in place: `x <- H x`.
/// \param x A row whose length is a power of two.
template <mutable_row X>
    requires std::floating_point<row_value_t<X>>
constexpr void fwht(X&& x) noexcept
{
    const auto r = as_mutable_row(x);
    assert(is_hadamard_order(r.size()));
    detail::hadamard::butterflies(r.data(), r.size());
}

/// \brief The orthonormal transform in place, ggml's bit for bit: `x <- H x / sqrt(n)`, scaled first; its own inverse.
/// \param x A row whose length is a power of two.
template <mutable_row X>
    requires std::floating_point<row_value_t<X>>
constexpr void hadamard(X&& x) noexcept
{
    using T = row_value_t<X>;
    const auto r = as_mutable_row(x);
    assert(is_hadamard_order(r.size()));
    detail::hadamard::scale(r.data(), r.size(), detail::hadamard::inverse_sqrt<T>(r.size()));
    detail::hadamard::butterflies(r.data(), r.size());
}

/// \brief The orthonormal transform of each run of `block` numbers: a block-diagonal `H`, as ggml applies a rotation of order `block` to a wider row.
/// \param x A row whose length is a multiple of `block`.
/// \param block A power of two.
template <mutable_row X>
    requires std::floating_point<row_value_t<X>>
constexpr void block_hadamard(X&& x, std::size_t block) noexcept
{
    using T = row_value_t<X>;
    const auto r = as_mutable_row(x);
    assert(is_hadamard_order(block) && r.size() % block == 0);
    const T s = detail::hadamard::inverse_sqrt<T>(block);
    for (std::size_t b = 0; b < r.size(); b += block) {
        detail::hadamard::scale(r.data() + b, block, s);
        detail::hadamard::butterflies(r.data() + b, block);
    }
}

/// \brief Random signs for the diagonal `D` of a randomised Hadamard rotation, from a seed.
/// \param signs Receives ±1, one per number; bit `k % 64` of draw `k / 64` of `stats::wyrand(seed)` set gives -1.
/// \param seed The seed: the same seed gives the same signs on every machine.
constexpr void hadamard_signs(std::span<std::int8_t> signs, std::uint64_t seed) noexcept
{
    stats::wyrand g(seed);
    std::uint64_t bits = 0;
    for (std::size_t k = 0; k < signs.size(); ++k) {
        if (k % 64 == 0)
            bits = g();
        signs[k] = (bits >> (k % 64)) & 1 ? std::int8_t{-1} : std::int8_t{1};
    }
}

/// \brief The randomised Hadamard transform of an input, as QuaRot, SpinQuant and llama.cpp's folded weights apply it: `x <- H (D x)`.
/// \param x A row whose length is a multiple of `block`.
/// \param signs The diagonal of `D`, ±1, as long as `x`.
/// \param block The order of `H`, a power of two (the row's length for one rotation of the whole row).
template <mutable_row X>
    requires std::floating_point<row_value_t<X>>
constexpr void random_hadamard(X&& x, std::span<const std::int8_t> signs, std::size_t block) noexcept
{
    const auto r = as_mutable_row(x);
    assert(signs.size() == r.size());
    for (std::size_t i = 0; i < r.size(); ++i) {
        assert(signs[i] == 1 || signs[i] == -1);
        r[i] = signs[i] < 0 ? -r[i] : r[i];
    }
    block_hadamard(r, block);
}

/// \brief The inverse of `random_hadamard`: `x <- D (H x)` (each factor its own inverse), as llama.cpp restores a rotated lookup table.
/// \param x A row whose length is a multiple of `block`.
/// \param signs The diagonal of `D`, ±1, as long as `x`.
/// \param block The order of `H`, a power of two.
template <mutable_row X>
    requires std::floating_point<row_value_t<X>>
constexpr void random_hadamard_inverse(X&& x, std::span<const std::int8_t> signs, std::size_t block) noexcept
{
    const auto r = as_mutable_row(x);
    assert(signs.size() == r.size());
    block_hadamard(r, block);
    for (std::size_t i = 0; i < r.size(); ++i)
        r[i] = signs[i] < 0 ? -r[i] : r[i];
}

/// \brief Folds a randomised Hadamard rotation into a row-major weight matrix: each row `r` becomes `r D H`, so `W_f (H D x) = W x`.
///
/// `H` is symmetric, so `r D H` is `H (D r)`: each row is transformed as an input is. Quantise the folded rows
/// afterwards; at inference, transform each input with `random_hadamard` and the same signs and block.
/// \param w The matrix, `rows · cols` numbers, row-major.
/// \param cols The columns: the input's length, a multiple of `block`.
/// \param signs The diagonal of `D`, `cols` signs.
/// \param block The order of `H`, a power of two.
template <mutable_row W>
    requires std::floating_point<row_value_t<W>>
constexpr void rotate_rows(W&& w, std::size_t cols, std::span<const std::int8_t> signs, std::size_t block) noexcept
{
    const auto m = as_mutable_row(w);
    assert(cols != 0 && m.size() % cols == 0);
    for (std::size_t i = 0; i < m.size(); i += cols)
        random_hadamard(m.subspan(i, cols), signs, block);
}

/// \brief The orthonormal Hadamard matrix of order `n`, row-major, as llama.cpp builds its rotation (`ggml_gen_hadamard`): entries `±(T)(1.0 / sqrtf(n))`.
/// \tparam T `float` or `double`.
/// \param n A power of two.
/// \param m Receives `n · n` entries.
template <std::floating_point T>
constexpr void hadamard_matrix(std::size_t n, std::span<T> m) noexcept
{
    assert(is_hadamard_order(n) && m.size() == n * n);
    const float root = math::sqrt(static_cast<float>(n));
    const T v = static_cast<T>(1.0 / static_cast<double>(root));
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            m[i * n + j] = hadamard_entry(i, j) < 0 ? -v : v;
}

} // namespace snicholls::ml

#endif // snicholls_core_ml_hadamard_hpp
