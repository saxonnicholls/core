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
/// - `sample_min_p(logits, temperature, min_p, u)`: sampling among the tokens at least `min_p` times as probable as
///   the most probable (Nguyen et al.), in logit space, with no scratch; `min_p_threshold` is its cut.
/// - `apply_penalties(logits, history, penalties)`: the logit processors of serving stacks, applied in place before
///   sampling: the repetition penalty of Keskar et al. (a seen token's positive logit divided by it, a negative one
///   multiplied) and OpenAI's presence and frequency penalties (subtract `presence` once and `frequency` per
///   occurrence), over the recent tokens the caller passes.
///
/// \snippet ml/sampling_demo.cpp sampling
///
/// **Relationships.** `ml/softmax.hpp` (the same stable exponentials), `stats/random.hpp` (the draw),
/// `ml/row.hpp`. Group: \ref core.
///
/// **Implementations.** One each.
///
/// **Guarantees.** `constexpr`, no allocation. Deterministic: the same row and draw give the same index.
/// Complexity: linear for `argmax`, `sample`, `sample_logits`, `sample_min_p`; O(n · K) for `top_k`; O(n log n) for
/// `sample_top_p`; O(h²) for `apply_penalties` over `h` recent tokens (each token's first occurrence is found by
/// scanning back, so no table of counts is needed; pass the window the penalties look at, usually 64 tokens or so).
///
/// **Errors.** Preconditions, asserted: the row is not empty; `u` is in `[0, 1)`; the temperature is not negative;
/// `p` and `min_p` are in `(0, 1]`; `scratch` has the row's size; a penalty's token is in the row; the repetition
/// penalty is positive. Rounding never makes a sampler return an index past the row:
/// if `u` falls beyond the last cumulative sum, the last index with a nonzero probability is returned.
///
/// **Performance.** Not benchmarked: one or two passes over the row, dominated by the exponentials.
///
/// **Tests and demos.** `tests/ml/sampling_test.cpp`, `demos/ml/sampling_demo.cpp`.
///
/// **Dependencies.** `<algorithm>`, `<array>`, `<cassert>`, `<concepts>`, `<cstddef>`, `<span>`,
/// `snicholls/core/math/constexpr_math.hpp`, `ml/row.hpp`.
///
/// **Known issues.** NaN logits are not ordered; a row with one compares as if it were absent from the maximum.
///
/// **Status.** Experimental, new in 0.7.0; `sample_min_p`, `min_p_threshold` and `apply_penalties` new in 0.8.0.

#ifndef snicholls_core_ml_sampling_hpp
#define snicholls_core_ml_sampling_hpp

#include <snicholls/core/math/constexpr_math.hpp>
#include <snicholls/core/ml/row.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
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

/// \brief The logit below which `sample_min_p` drops a token: `max + temperature · log(min_p)`.
/// \param logits A non-empty row.
/// \param temperature Greater than 0.
/// \param min_p The fraction of the most probable token's probability a token must reach, in `(0, 1]`.
/// \return The threshold: a token is kept when its logit is at least this.
template <row R>
constexpr row_value_t<R> min_p_threshold(const R& logits, std::type_identity_t<row_value_t<R>> temperature,
                                         std::type_identity_t<row_value_t<R>> min_p)
{
    using T [[maybe_unused]] = row_value_t<R>;   // read only by the assert
    const auto x = as_row(logits);
    assert(!x.empty() && temperature > T(0) && min_p > T(0) && min_p <= T(1));
    return x[argmax(x)] + temperature * math::log(min_p);
}

/// \brief Min-p sampling: among the tokens at least `min_p` times as probable as the most probable, at a temperature.
/// \param logits A non-empty row.
/// \param temperature 0 for `argmax`.
/// \param min_p The fraction of the top probability a token must reach, in `(0, 1]`; 1 keeps only the top.
/// \param u A uniform draw in `[0, 1)`.
/// \return The index in the row.
template <row R>
constexpr std::size_t sample_min_p(const R& logits, std::type_identity_t<row_value_t<R>> temperature,
                                   std::type_identity_t<row_value_t<R>> min_p, std::type_identity_t<row_value_t<R>> u)
{
    using T = row_value_t<R>;
    const auto x = as_row(logits);
    assert(!x.empty() && u >= T(0) && u < T(1));
    const std::size_t best = argmax(x);
    if (temperature == T(0))
        return best;
    const T cut = min_p_threshold(x, temperature, min_p), max = x[best];
    T mass = T(0);
    for (const T v : x)
        if (v >= cut)
            mass += math::exp((v - max) / temperature);
    const T target = u * mass;
    T cdf = T(0);
    std::size_t last = best;
    for (std::size_t i = 0; i < x.size(); ++i)
        if (x[i] >= cut) {
            cdf += math::exp((x[i] - max) / temperature);
            last = i;
            if (target < cdf)
                return i;
        }
    return last;
}

/// \brief The penalties of `apply_penalties`; the defaults change nothing.
/// \ingroup core
/// \tparam T The number type.
template <math::math_number T>
struct logit_penalties
{
    T repetition = T(1);   ///< Keskar et al.: a seen token's logit divided by it if positive, multiplied if not; 1 for none.
    T presence = T(0);     ///< Subtracted once from a seen token's logit.
    T frequency = T(0);    ///< Subtracted from a token's logit for each time it was seen.
};

/// \brief Applies the repetition, presence and frequency penalties to the logits of the tokens in `history`, in place.
/// \param logits The row, updated.
/// \param history The recent tokens: indices into the row.
/// \param penalties The penalties.
template <mutable_row R, typename Token>
    requires std::integral<Token>
constexpr void apply_penalties(R&& logits, std::span<const Token> history, const logit_penalties<row_value_t<R>>& penalties)
{
    using T = row_value_t<R>;
    const auto x = as_mutable_row(logits);
    assert(penalties.repetition > T(0));
    for (std::size_t i = 0; i < history.size(); ++i) {
        const auto t = static_cast<std::size_t>(history[i]);
        assert(t < x.size());
        bool first = true;
        std::size_t count = 0;
        for (std::size_t j = 0; j < history.size(); ++j)
            if (history[j] == history[i]) {
                first = first && j >= i;
                ++count;
            }
        if (!first)
            continue;   // each distinct token once, at its first occurrence
        T& v = x[t];
        v = v > T(0) ? v / penalties.repetition : v * penalties.repetition;
        v -= penalties.presence + penalties.frequency * T(count);
    }
}

} // namespace snicholls::ml

#endif // snicholls_core_ml_sampling_hpp
