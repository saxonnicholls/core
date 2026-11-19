//
//  gather.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::ml` rows by index lists, on the backend: `gather_rows`, its backward `scatter_add_rows`, `scatter_rows`, `concat_rows` and `slice_rows`, over a batch of sequences, deterministic.
/// \ingroup core
///
/// **Overview.** A joint embedding predictive architecture (TRAINING.md T11) moves token rows by index: the context
/// encoder reads only the context tokens of each sample, the target tokens are picked out of the target encoder's
/// output, and the predictor reads the context encoder's rows followed by one mask token per target. Each of those
/// moves is one of these calls, made with the pass's backend (TRAINING.md T9), so a device library runs it where the
/// rows live.
///
/// A batch is `sequences` sequences laid end to end. `src` holds `sequences · n` rows of `width`; `indices` holds
/// `sequences · k` positions, sequence `s`'s `k` of them relative to its own first row (in `[0, n)`), as
/// `ml/masking.hpp` writes them. Row `(s, j)` of the result is row `s · n + indices[s · k + j]` of `src`.
///
/// - `gather_rows(b, src, indices, out, width, sequences)`: `out(s, j) = src(s, indices(s, j))`.
/// - `scatter_add_rows(b, dout, indices, dsrc, width, sequences)`: its backward, `dsrc(s, indices(s, j)) += dout(s,
///   j)` for `j` from 0 up, sequence after sequence: a fixed order, so a repeated index sums the same way on every
///   run (no atomics). Untouched rows keep their values.
/// - `scatter_rows(b, rows, indices, dst, width, sequences)`: `dst(s, indices(s, j)) = rows(s, j)`, the last
///   write winning for a repeated index; its backward is `gather_rows` of the gradient (the overwritten rows of `dst`
///   pass none). The way to lay rows out in any order, such as V-JEPA 2-AC's action, state and patch tokens frame by
///   frame.
/// - `concat_rows(b, first, second, out, width, sequences)`: each sequence's rows of `first`, then of `second` (the
///   predictor's input: context rows, then the targets' mask tokens). Its backward is two `slice_rows`.
/// - `slice_rows(b, rows, first, count, out, width, sequences)`: rows `[first, first + count)` of each sequence (the
///   predictor's outputs at the targets). Its backward is `slice_rows_backward`, added into those rows.
///
/// \snippet ml/gather_demo.cpp gather
///
/// **Relationships.** `ml/masking.hpp` (the index lists), `ml/jepa.hpp` (the predictor's input), `ml/patch.hpp`,
/// `ml/linear.hpp` (the other elementwise steps of the batched passes). The mixture of experts' gather and combine
/// (`ml/moe.hpp`, TRAINING.md T10) are a different operation, a permutation of every row by its experts with
/// weights; these are plain index lists per sequence. Group: \ref core.
///
/// **Implementations.** One each, on the host for any backend; a device library overloads them for its own backend
/// type, found by argument-dependent lookup (TRAINING.md T9).
///
/// **Guarantees.** `constexpr`, no allocation, O(rows · width); copies are exact, so every result but
/// `scatter_add_rows`' sums is bit-exact, and those sums run in the stated order. `noexcept` for the standard
/// floating-point types.
///
/// **Errors.** Preconditions, asserted: the sizes divide as stated, every index is below `n`, no output overlaps an
/// input.
///
/// **Performance.** `gather_rows` and `scatter_add_rows` against the loops by hand
/// (`benchmarks/ml/gather_benchmark.cpp`): see `BENCHMARKS.md`. Memory bound: a row copy each.
///
/// **Tests and demos.** `tests/ml/gather_test.cpp`, `demos/ml/gather_demo.cpp`.
///
/// **Dependencies.** `<algorithm>`, `<cassert>`, `<cstddef>`, `<cstdint>`, `<span>`, `<type_traits>`,
/// `snicholls/core/math/constexpr_math.hpp` (the `math_number` concept).
///
/// **Known issues.** Indices are `std::uint32_t`, as the mixture of experts' are: four billion rows per sequence.
///
/// **Status.** Experimental, new in 0.16.0.

#ifndef snicholls_core_ml_gather_hpp
#define snicholls_core_ml_gather_hpp

#include <snicholls/core/math/constexpr_math.hpp>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>

namespace snicholls::ml
{

/// \brief `out(s, j) = src(s, indices(s, j))`: each sequence's rows at its index list.
/// \param src `sequences · n` rows of `width`, end to end.
/// \param indices `sequences · k` positions, each below `n`, sequence by sequence.
/// \param out Receives `sequences · k` rows of `width`.
/// \param width The values of a row.
/// \param sequences The sequences, at least one.
template <typename B, math::math_number T>
constexpr void gather_rows(const B&, std::type_identity_t<std::span<const T>> src, std::span<const std::uint32_t> indices,
                           std::span<T> out, std::size_t width, std::size_t sequences = 1) noexcept(std::is_floating_point_v<T>)
{
    assert(width > 0 && sequences > 0 && src.size() % (width * sequences) == 0 && indices.size() % sequences == 0
           && out.size() == indices.size() * width);
    const std::size_t n = src.size() / width / sequences, k = indices.size() / sequences;
    for (std::size_t s = 0; s < sequences; ++s)
        for (std::size_t j = 0; j < k; ++j) {
            const std::size_t i = indices[s * k + j];
            assert(i < n);
            std::copy_n(src.begin() + std::ptrdiff_t((s * n + i) * width), width, out.begin() + std::ptrdiff_t((s * k + j) * width));
        }
}

/// \brief The backward pass of `gather_rows`: `dsrc(s, indices(s, j)) += dout(s, j)`, sequence by sequence, `j` from 0 up.
/// \param dout The gradients in `gather_rows`' output, `sequences · k` rows.
/// \param indices Its index lists.
/// \param dsrc Added into, `sequences · n` rows; rows no index names are unchanged.
/// \param width The values of a row.
/// \param sequences The sequences.
template <typename B, math::math_number T>
constexpr void scatter_add_rows(const B&, std::type_identity_t<std::span<const T>> dout, std::span<const std::uint32_t> indices,
                                std::span<T> dsrc, std::size_t width, std::size_t sequences = 1) noexcept(std::is_floating_point_v<T>)
{
    assert(width > 0 && sequences > 0 && dsrc.size() % (width * sequences) == 0 && indices.size() % sequences == 0
           && dout.size() == indices.size() * width);
    const std::size_t n = dsrc.size() / width / sequences, k = indices.size() / sequences;
    for (std::size_t s = 0; s < sequences; ++s)
        for (std::size_t j = 0; j < k; ++j) {
            const std::size_t i = indices[s * k + j];
            assert(i < n);
            for (std::size_t c = 0; c < width; ++c)
                dsrc[(s * n + i) * width + c] += dout[(s * k + j) * width + c];
        }
}

/// \brief `dst(s, indices(s, j)) = rows(s, j)`: each sequence's rows written to its positions, in `j` order.
/// \param rows `sequences · k` rows of `width`.
/// \param indices `sequences · k` positions, each below `n`.
/// \param dst `sequences · n` rows: the named rows written, the others unchanged.
/// \param width The values of a row.
/// \param sequences The sequences.
template <typename B, math::math_number T>
constexpr void scatter_rows(const B&, std::type_identity_t<std::span<const T>> rows, std::span<const std::uint32_t> indices,
                            std::span<T> dst, std::size_t width, std::size_t sequences = 1) noexcept(std::is_floating_point_v<T>)
{
    assert(width > 0 && sequences > 0 && dst.size() % (width * sequences) == 0 && indices.size() % sequences == 0
           && rows.size() == indices.size() * width);
    const std::size_t n = dst.size() / width / sequences, k = indices.size() / sequences;
    for (std::size_t s = 0; s < sequences; ++s)
        for (std::size_t j = 0; j < k; ++j) {
            const std::size_t i = indices[s * k + j];
            assert(i < n);
            std::copy_n(rows.begin() + std::ptrdiff_t((s * k + j) * width), width, dst.begin() + std::ptrdiff_t((s * n + i) * width));
        }
}

/// \brief Each sequence's rows of `first`, then its rows of `second`, into `out`.
/// \param first `sequences · a` rows of `width`.
/// \param second `sequences · b` rows of `width`.
/// \param out Receives `sequences · (a + b)` rows.
/// \param width The values of a row.
/// \param sequences The sequences.
template <typename B, math::math_number T>
constexpr void concat_rows(const B&, std::type_identity_t<std::span<const T>> first, std::type_identity_t<std::span<const T>> second,
                           std::span<T> out, std::size_t width, std::size_t sequences = 1) noexcept(std::is_floating_point_v<T>)
{
    assert(width > 0 && sequences > 0 && first.size() % (width * sequences) == 0 && second.size() % (width * sequences) == 0
           && out.size() == first.size() + second.size());
    const std::size_t a = first.size() / sequences, b = second.size() / sequences;
    for (std::size_t s = 0; s < sequences; ++s) {
        std::copy_n(first.begin() + std::ptrdiff_t(s * a), a, out.begin() + std::ptrdiff_t(s * (a + b)));
        std::copy_n(second.begin() + std::ptrdiff_t(s * b), b, out.begin() + std::ptrdiff_t(s * (a + b) + a));
    }
}

/// \brief Rows `[first, first + count)` of each sequence of `n` rows, into `out`.
/// \param rows `sequences · n` rows of `width`.
/// \param first The first row taken from each sequence.
/// \param count The rows taken from each; `first + count <= n`.
/// \param out Receives `sequences · count` rows.
/// \param width The values of a row.
/// \param sequences The sequences.
template <typename B, math::math_number T>
constexpr void slice_rows(const B&, std::type_identity_t<std::span<const T>> rows, std::size_t first, std::size_t count,
                          std::span<T> out, std::size_t width, std::size_t sequences = 1) noexcept(std::is_floating_point_v<T>)
{
    assert(width > 0 && sequences > 0 && rows.size() % (width * sequences) == 0 && out.size() == sequences * count * width);
    const std::size_t n = rows.size() / width / sequences;
    assert(first + count <= n);
    for (std::size_t s = 0; s < sequences; ++s)
        std::copy_n(rows.begin() + std::ptrdiff_t((s * n + first) * width), count * width, out.begin() + std::ptrdiff_t(s * count * width));
}

/// \brief The backward pass of `slice_rows`: the gradients of the slice added into rows `[first, first + count)` of
/// each sequence.
/// \param dout The gradients in `slice_rows`' output, `sequences · count` rows.
/// \param first Its first row.
/// \param count Its rows per sequence.
/// \param drows Added into, `sequences · n` rows.
/// \param width The values of a row.
/// \param sequences The sequences.
template <typename B, math::math_number T>
constexpr void slice_rows_backward(const B&, std::type_identity_t<std::span<const T>> dout, std::size_t first, std::size_t count,
                                   std::span<T> drows, std::size_t width, std::size_t sequences = 1) noexcept(std::is_floating_point_v<T>)
{
    assert(width > 0 && sequences > 0 && drows.size() % (width * sequences) == 0 && dout.size() == sequences * count * width);
    const std::size_t n = drows.size() / width / sequences;
    assert(first + count <= n);
    for (std::size_t s = 0; s < sequences; ++s)
        for (std::size_t i = 0; i < count * width; ++i)
            drows[(s * n + first) * width + i] += dout[s * count * width + i];
}

} // namespace snicholls::ml

#endif // snicholls_core_ml_gather_hpp
