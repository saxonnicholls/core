//
//  sampling.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::ml` sampling the next token: `argmax`, sampling from probabilities or from logits at a temperature, `top_k` and top-k sampling with a fixed-size buffer, and nucleus (top-p) sampling with the caller's scratch.
/// \ingroup core
///
/// **Overview.** The last step of generation, as llm.c's `sample_mult` and nanochat's `sample_next_token` do it.
/// Every sampler takes its randomness as one uniform draw `u` in `[0, 1)` from the caller
/// (`stats::uniform_real(g)` with a `stats::wyrand`), so the same seed gives the same text on every machine, and a
/// batch of sequences can each have their own stream.
///
/// - `argmax(row)`: the index of the largest value, the first on ties (greedy decoding, temperature 0).
/// - `sample(probabilities, u)`: the inverse of the cumulative distribution at `u`.
/// - `sample_logits(logits, temperature, u)`: softmax at a temperature and the inverse CDF, in two passes with no
///   row of probabilities stored. A temperature of 0 is `argmax`.
/// - `top_k<K>(row)`: the `K` largest values and their indices, largest first, in a `std::array`: O(n · K), no
///   allocation, `K` known to the compiler.
/// - `sample_top_k<K>(logits, temperature, u)`: sampling among the `K` largest logits (nanochat's `top_k`).
/// - `sample_top_p(logits, temperature, p, u, scratch)`: sampling among the smallest set of most probable tokens
///   whose probability reaches `p` (Holtzman et al.), sorting the caller's `scratch` of indices.
///
/// \snippet ml/sampling_demo.cpp sampling
///
/// **Relationships.** `ml/softmax.hpp` (the same stable exponentials), `stats/random.hpp` (the draw),
/// `ml/row.hpp`. Group: \ref core.
///
/// **Implementations.** One each.
///
/// **Guarantees.** `constexpr`, no allocation. Deterministic: the same row and draw give the same index.
/// Complexity: linear for `argmax`, `sample`, `sample_logits`; O(n · K) for `top_k`; O(n log n) for
/// `sample_top_p`.
///
/// **Errors.** Preconditions, asserted: the row is not empty; `u` is in `[0, 1)`; the temperature is not negative;
/// `p` is in `(0, 1]`; `scratch` has the row's size. Rounding never makes a sampler return an index past the row:
/// if `u` falls beyond the last cumulative sum, the last index with a nonzero probability is returned.
///
/// **Performance.** Not benchmarked: one or two passes over the row, dominated by the exponentials.
///
/// **Tests and demos.** `tests/ml/sampling_test.cpp`, `demos/ml/sampling_demo.cpp`.
///
/// **Dependencies.** `<algorithm>`, `<array>`, `<cassert>`, `<cstddef>`, `<limits>`, `<span>`,
/// `snicholls/core/math/constexpr_math.hpp`, `ml/row.hpp`.
///
/// **Known issues.** NaN logits are not ordered; a row with one compares as if it were absent from the maximum.
///
/// **Status.** Experimental, new in 0.7.0.

#ifndef snicholls_core_ml_sampling_hpp
#define snicholls_core_ml_sampling_hpp

#include <snicholls/core/math/constexpr_math.hpp>
#include <snicholls/core/ml/row.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <limits>
#include <span>

namespace snicholls::ml
{

/// \brief The index of the largest value, the first of equal ones.
/// \param r A non-empty row.
/// \return The index.
template <row R>
constexpr std::size_t argmax(const R& r)
{
    const auto x = as_row(r);
    assert(!x.empty());
    std::size_t best = 0;
    for (std::size_t i = 1; i < x.size(); ++i)
        if (x[i] > x[best])
            best = i;
    return best;
}

/// \brief Samples an index from probabilities: the first whose cumulative sum exceeds `u`.
/// \param probabilities A non-empty row summing to 1 (up to rounding).
/// \param u A uniform draw in `[0, 1)`.
/// \return The index.
template <row R>
constexpr std::size_t sample(const R& probabilities, std::type_identity_t<row_value_t<R>> u)
{
    using T = row_value_t<R>;
    const auto p = as_row(probabilities);
    assert(!p.empty() && u >= T(0) && u < T(1));
    T cdf = T(0);
    std::size_t last = 0;
    for (std::size_t i = 0; i < p.size(); ++i) {
        cdf += p[i];
        if (p[i] > T(0))
            last = i;
        if (u < cdf)
            return i;
    }
    return last;   // rounding left the sum below u
}

/// \brief Samples an index from logits at a temperature: `softmax(logits / temperature)` and the inverse CDF, with no row of probabilities.
/// \param logits A non-empty row.
/// \param temperature 0 for `argmax`; 1 for the model's own distribution.
/// \param u A uniform draw in `[0, 1)`.
/// \return The index.
template <row R>
constexpr std::size_t sample_logits(const R& logits, std::type_identity_t<row_value_t<R>> temperature,
                                    std::type_identity_t<row_value_t<R>> u)
{
    using T = row_value_t<R>;
    const auto x = as_row(logits);
    assert(!x.empty() && temperature >= T(0) && u >= T(0) && u < T(1));
    const std::size_t best = argmax(x);
    if (temperature == T(0))
        return best;
    const T max = x[best];
    T sum = T(0);
    for (const T v : x)
        sum += math::exp((v - max) / temperature);
    const T target = u * sum;
    T cdf = T(0);
    for (std::size_t i = 0; i < x.size(); ++i) {
        cdf += math::exp((x[i] - max) / temperature);
        if (target < cdf)
            return i;
    }
    return best;
}

/// \brief The `K` largest values of a row and their indices, largest first.
/// \ingroup core
/// \tparam T The number type.
/// \tparam K The most kept.
template <typename T, std::size_t K>
struct top_k_result
{
    std::array<std::size_t, K> index{};   ///< The indices, largest value first; on ties, the lower index first.
    std::array<T, K> value{};             ///< Their values.
    std::size_t count = 0;                ///< `min(K, row size)`.
};

/// \brief The `K` largest values of a row and their indices, largest first: O(n · K), no allocation.
/// \tparam K The most kept, at least 1.
/// \param r The row.
/// \return The values, their indices and how many.
template <std::size_t K, row R>
    requires(K > 0)
constexpr top_k_result<row_value_t<R>, K> top_k(const R& r)
{
    const auto x = as_row(r);
    top_k_result<row_value_t<R>, K> out;
    for (std::size_t i = 0; i < x.size(); ++i) {
        if (out.count == K && !(x[i] > out.value[K - 1]))
            continue;
        std::size_t at = out.count < K ? out.count++ : K - 1;
        for (; at > 0 && x[i] > out.value[at - 1]; --at) {   // strictly greater: an earlier equal stays first
            out.value[at] = out.value[at - 1];
            out.index[at] = out.index[at - 1];
        }
        out.value[at] = x[i];
        out.index[at] = i;
    }
    return out;
}

/// \brief Samples among the `K` largest logits at a temperature.
/// \tparam K The tokens kept, at least 1.
/// \param logits A non-empty row.
/// \param temperature 0 for `argmax`.
/// \param u A uniform draw in `[0, 1)`.
/// \return The index in the row.
template <std::size_t K, row R>
    requires(K > 0)
constexpr std::size_t sample_top_k(const R& logits, std::type_identity_t<row_value_t<R>> temperature,
                                   std::type_identity_t<row_value_t<R>> u)
{
    const auto best = top_k<K>(logits);
    assert(best.count > 0);
    return best.index[sample_logits(std::span(best.value).first(best.count), temperature, u)];
}

/// \brief Nucleus sampling: among the most probable tokens whose probabilities first reach `p`, at a temperature.
/// \param logits A non-empty row.
/// \param temperature 0 for `argmax`.
/// \param p The probability mass kept, in `(0, 1]`.
/// \param u A uniform draw in `[0, 1)`.
/// \param scratch The row's size of indices: overwritten.
/// \return The index in the row.
template <row R>
constexpr std::size_t sample_top_p(const R& logits, std::type_identity_t<row_value_t<R>> temperature,
                                   std::type_identity_t<row_value_t<R>> p, std::type_identity_t<row_value_t<R>> u,
                                   std::span<std::size_t> scratch)
{
    using T = row_value_t<R>;
    const auto x = as_row(logits);
    assert(!x.empty() && scratch.size() == x.size() && p > T(0) && p <= T(1) && u >= T(0) && u < T(1));
    if (temperature == T(0))
        return argmax(x);
    for (std::size_t i = 0; i < scratch.size(); ++i)
        scratch[i] = i;
    std::sort(scratch.begin(), scratch.end(), [&](std::size_t a, std::size_t b) { return x[a] > x[b] || (x[a] == x[b] && a < b); });
    const T max = x[scratch[0]];
    T sum = T(0);
    for (const T v : x)
        sum += math::exp((v - max) / temperature);
    // Keep the most probable until their mass reaches p, then sample among them
    std::size_t kept = 0;
    T mass = T(0);
    while (kept < scratch.size() && (kept == 0 || mass < p * sum))
        mass += math::exp((x[scratch[kept++]] - max) / temperature);
    const T target = u * mass;
    T cdf = T(0);
    for (std::size_t i = 0; i < kept; ++i) {
        cdf += math::exp((x[scratch[i]] - max) / temperature);
        if (target < cdf)
            return scratch[i];
    }
    return scratch[kept - 1];
}

} // namespace snicholls::ml

#endif // snicholls_core_ml_sampling_hpp
