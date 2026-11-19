//
//  samplers.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::ml` samplers over a candidate array, as llama.cpp chains them: greedy, dist, temperature, top-k, top-p, min-p, penalties, locally typical, dynamic temperature, XTC, top-n-sigma, mirostat v1 and v2, DRY and logit bias, composed at compile time (`sampler_chain`) or at run time (`runtime_sampler_chain`).
/// \ingroup core
///
/// **Overview.** `ml/sampling.hpp` samples straight from a row of logits. Serving stacks instead pass a *candidate
/// array* through a chain of samplers, each of which rescales, masks, reorders or truncates it, and the last of which
/// selects a token. This header is that model, as llama.cpp's `llama_token_data_array` (`src/llama-sampler.cpp`):
///
/// - `token_data<T>` is `{id, logit, p}`; `token_candidates<T>` is a view over the caller's span of them with
///   `sorted` (descending logit) and `selected` (the chosen index, -1 before a selecting sampler ran).
///   `make_candidates(logits, storage)` fills one from a row of logits, token `i` at index `i`.
/// - Each sampler is a small struct with `apply(candidates, g)`, where `g` is a `stats::wyrand`; the stateful ones
///   (`penalties`, `dry`, `mirostat`, `mirostat_v2`) also have `accept(token)` and `reset()`.
///   - selecting: `greedy_sampler` (the first largest logit), `dist_sampler` (softmax, then the inverse CDF at one
///     `stats::uniform_real` draw), `mirostat_sampler` (Basu et al., 2021, algorithm 1: Zipf's exponent estimated from
///     the top `m`, a top-k from it, `mu` moved by the observed surprise) and `mirostat_v2_sampler` (truncate where the
///     surprise exceeds `mu`).
///   - rescaling: `temperature_sampler` (0 keeps only the largest), `temp_ext_sampler` (dynamic temperature: the
///     temperature in `[temp - delta, temp + delta]` set by the normalised entropy raised to `exponent`).
///   - truncating: `top_k_sampler`, `top_p_sampler`, `min_p_sampler`, `typical_sampler` (Meister et al., 2023: the
///     tokens whose surprise is closest to the entropy, up to mass `p`), `xtc_sampler` (exclude top choices: with
///     probability `probability`, drop every token at or above `threshold` but the least probable of them),
///     `top_n_sigma_sampler` (Tang et al., 2024: mask logits below `max - n · σ`).
///   - adjusting logits: `penalties_sampler` (repetition, frequency and presence over the last `last_n` accepted
///     tokens: the same rule as `apply_penalties` in `ml/sampling.hpp`), `dry_sampler` (DRY, "don't repeat
///     yourself": a token that would extend a repeat of `allowed_length` or more of the recent context is penalised
///     `multiplier · base^(length - allowed_length)`, with sequence breakers that reset the match), `logit_bias_sampler`.
/// - `sampler_chain<S...>`: the chain as a tuple, fixed at compile time; `runtime_sampler_chain<T, History>`: a
///   `std::vector` of `any_sampler<T, History>`, a `std::variant` over every sampler here, built at run time. Both
///   `sample(...)` the selected token and forward `accept` and `reset` to the samplers that have them.
///
/// \snippet ml/samplers_demo.cpp samplers
///
/// **Relationships.** `ml/sampling.hpp` (the row samplers, whose `apply_penalties` rule `penalties_sampler` follows),
/// `stats/random.hpp` (`wyrand`, `uniform_real`), `overloaded.hpp` (`first_match`, `overloaded`), `constexpr_for.hpp`,
/// `ml/speculative.hpp` (verification with the same probabilities). Group: \ref core.
///
/// **Implementations.** One each. The semantics are llama.cpp's at commit 39039aa (`src/llama-sampler.cpp`), read
/// function by function: `llama_sampler_temp_impl`, `llama_sampler_softmax_impl`, `llama_sampler_top_k_impl`,
/// `llama_sampler_dist_apply`, `llama_sampler_top_p_apply`, `llama_sampler_min_p_apply` (its unsorted path first),
/// `llama_sampler_typical_apply`, `llama_sampler_temp_ext_apply` (the renormalisation in double), `llama_sample_xtc_apply`,
/// `llama_sampler_top_n_sigma_apply`, `llama_sampler_mirostat_apply`, `llama_sampler_mirostat_v2_apply`,
/// `llama_sampler_penalties_apply`, `llama_sampler_dry_apply` (the restart sequences, the reverse Z-algorithm, the
/// exponent clamp at `88.7228391 / ln base`) and `llama_sampler_logit_bias_apply`. Differences, all deliberate:
///
/// - Randomness is `stats::wyrand`, never `std::mt19937` with a standard distribution (whose output differs between
///   standard libraries), so a seed gives the same tokens everywhere. Every random choice is one `uniform_real` draw.
/// - Sorting orders equal logits by the lower id, so that the order is the same on every standard library
///   (llama.cpp's `std::partial_sort` leaves ties unspecified). `typical_sampler` breaks ties of its score the same way.
/// - `dist_sampler` selects the first token whose cumulative probability *exceeds* `u · sum` (llama.cpp: reaches it),
///   so a token of probability 0 is never chosen, even at `u = 0`.
/// - `typical_sampler` skips tokens of probability 0 in the entropy (llama.cpp's `-p log p` is then NaN).
/// - `mirostat_sampler` clamps the computed `k` to `[1, size]` (llama.cpp converts a non-finite or huge float to int,
///   which is undefined).
/// - Surprise in bits is `-log(p) / ln 2` (llama.cpp: `log2f`), which can differ in the last bit.
/// - DRY's sequence breakers are token sequences the caller has tokenised, head first in reading order.
///   llama.cpp derives them from strings with the vocabulary (`get_overlapping_token_sequences`, every token that
///   contains a breaker or begins one, the tail capped at 20 tokens); that needs the tokenizer and is the caller's.
///
/// Infill (`llama_sampler_infill_apply`) is not here: it sums the probabilities of end-of-generation tokens against
/// text tokens (choosing only EOG tokens when `3 · p_eog · n > p_text`), and merges candidates whose pieces share a
/// prefix, so it needs the vocabulary's EOG classification and the token pieces, which belong to the tokenizer layer
/// above core. A caller can apply it between this chain's truncation and its selection with the same candidates.
///
/// **Guarantees.** `constexpr` (a whole chain runs in a `static_assert`), no allocation except
/// `runtime_sampler_chain`'s vector, no I/O. Deterministic: the same candidates, state and generator give the same
/// token. Each sampler is trivially copyable apart from the spans it views (`dry_sampler`'s breakers,
/// `logit_bias_sampler`'s biases), which must outlive it. Complexity: O(n) for the masks, rescalings and `dist`;
/// O(n log n) where the candidates are sorted; `penalties` O(h² + h · n) for `h` history tokens (O(h²) when the
/// candidates are in id order); `dry` O(h · b) for `b` breakers plus O(h · n) at worst.
///
/// **Errors.** Preconditions, asserted: a selecting sampler has at least one candidate; `top_p` and `min_p` thresholds
/// in their documented ranges; `last_n`, `penalty_last_n` not beyond the compile-time history; a repetition penalty
/// greater than 0. A chain whose samplers selected nothing returns the greedy choice.
///
/// **Performance.** Not benchmarked: one pass over the candidates, dominated by the exponentials, or a sort.
///
/// **Tests and demos.** `tests/ml/samplers_test.cpp`, `demos/ml/samplers_demo.cpp`.
///
/// **Dependencies.** `<algorithm>`, `<array>`, `<cassert>`, `<concepts>`, `<cstddef>`, `<cstdint>`, `<limits>`,
/// `<span>`, `<tuple>`, `<type_traits>`, `<utility>`, `<variant>`, `<vector>`, `snicholls/core/constexpr_for.hpp`,
/// `snicholls/core/math/constexpr_math.hpp`, `snicholls/core/overloaded.hpp`, `snicholls/core/stats/random.hpp`,
/// `ml/row.hpp`.
///
/// **Known issues.** NaN logits are not ordered. Mirostat's estimate needs at least two candidates and `m >= 2`;
/// otherwise its `k` is 1 (greedy). The run-time chain's `std::variant` makes it as large as its largest sampler
/// (the history buffers of `penalties` and `dry`).
///
/// **Status.** Experimental, new in 0.17.0.

#ifndef snicholls_core_ml_samplers_hpp
#define snicholls_core_ml_samplers_hpp

#include <snicholls/core/constexpr_for.hpp>
#include <snicholls/core/math/constexpr_math.hpp>
#include <snicholls/core/ml/row.hpp>
#include <snicholls/core/overloaded.hpp>
#include <snicholls/core/stats/random.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace snicholls::ml
{

/// \brief A token id, as llama.cpp's `llama_token`.
using token_id = std::int32_t;

/// \brief One candidate token: its id, logit and probability (llama.cpp's `llama_token_data`).
/// \ingroup core
/// \tparam T The number type.
template <math::math_number T>
struct token_data
{
    token_id id = 0;   ///< The token.
    T logit = T(0);    ///< Its logit, as the samplers so far left it.
    T p = T(0);        ///< Its probability, once a sampler has computed one.
};

/// \brief The candidate array a chain of samplers works on: a view over the caller's storage.
/// \ingroup core
/// \tparam T The number type.
template <math::math_number T>
struct token_candidates
{
    std::span<token_data<T>> data{};   ///< The current candidates; truncating samplers shorten or advance it.
    bool sorted = false;               ///< Whether `data` is in descending logit order.
    std::ptrdiff_t selected = -1;      ///< The index the selecting sampler chose, or -1.

    /// \brief The number of candidates.
    /// \return `data.size()`.
    [[nodiscard]] constexpr std::size_t size() const noexcept { return data.size(); }
    /// \brief The selected token.
    /// \pre `selected` is an index into `data`.
    /// \return Its id.
    [[nodiscard]] constexpr token_id selected_token() const
    {
        assert(selected >= 0 && static_cast<std::size_t>(selected) < data.size());
        return data[static_cast<std::size_t>(selected)].id;
    }
};

/// \brief Fills `storage` from a row of logits, token `i` at index `i`, and views it.
/// \param logits The model's logits, one per token of the vocabulary.
/// \param storage The caller's candidates, `logits`' size; overwritten.
/// \return The candidates, unsorted, nothing selected.
template <row R>
constexpr token_candidates<row_value_t<R>> make_candidates(const R& logits, std::span<token_data<row_value_t<R>>> storage)
{
    using T = row_value_t<R>;
    const auto x = as_row(logits);
    assert(storage.size() == x.size());
    for (std::size_t i = 0; i < x.size(); ++i)
        storage[i] = {static_cast<token_id>(i), x[i], T(0)};
    return {storage, false, -1};
}

namespace detail::samplers
{
    template <typename T>
    inline constexpr T neg_inf = -std::numeric_limits<T>::infinity();

    template <typename T>
    constexpr T exp_or_zero(T x) { return x == neg_inf<T> ? T(0) : math::exp(x); }

    template <typename T>
    constexpr T log2(T x) { return math::log(x) / math::log(T(2)); }

    // Descending logit, then ascending id: one order on every standard library
    template <typename T>
    constexpr void sort(token_candidates<T>& c)
    {
        if (c.sorted)
            return;
        std::sort(c.data.begin(), c.data.end(),
                  [](const token_data<T>& a, const token_data<T>& b) { return a.logit > b.logit || (a.logit == b.logit && a.id < b.id); });
        c.sorted = true;
    }

    // llama_sampler_softmax_impl
    template <typename T>
    constexpr void softmax(token_candidates<T>& c, bool do_sort)
    {
        assert(c.size() > 0);
        if (do_sort)
            sort(c);
        T max = c.data[0].logit;
        if (!c.sorted)
            for (const auto& d : c.data)
                max = std::max(max, d.logit);
        T sum = T(0);
        for (auto& d : c.data) {
            d.p = exp_or_zero(d.logit - max);
            sum += d.p;
        }
        for (auto& d : c.data)
            d.p /= sum;
    }

    // llama_sampler_temp_impl
    template <typename T>
    constexpr void temperature(token_candidates<T>& c, T temp)
    {
        if (c.size() == 0)
            return;
        if (temp <= T(0)) {
            std::size_t max_i = 0;
            T max_l = c.data[0].logit;
            for (std::size_t i = 1; i < c.size(); ++i) {
                if (c.data[i].logit > max_l) {
                    c.data[max_i].logit = neg_inf<T>;
                    max_i = i;
                    max_l = c.data[i].logit;
                } else {
                    c.data[i].logit = neg_inf<T>;
                }
            }
            return;
        }
        for (auto& d : c.data)
            d.logit /= temp;
    }

    // llama_sampler_top_k_impl
    template <typename T>
    constexpr void top_k(token_candidates<T>& c, std::int32_t k)
    {
        if (k <= 0)
            return;
        sort(c);
        c.data = c.data.first(std::min(static_cast<std::size_t>(k), c.size()));
    }

    // The inverse CDF over the candidates' p: the first whose running sum exceeds u times the total; rounding falls
    // back to the last with a nonzero probability
    template <typename T>
    constexpr std::size_t inverse_cdf(const token_candidates<T>& c, double u)
    {
        assert(c.size() > 0 && u >= 0.0 && u < 1.0);
        double total = 0.0;
        for (const auto& d : c.data)
            total += static_cast<double>(d.p);
        const double target = u * total;
        double run = 0.0;
        std::size_t last = 0;
        for (std::size_t i = 0; i < c.size(); ++i) {
            run += static_cast<double>(c.data[i].p);
            if (c.data[i].p > T(0))
                last = i;
            if (run > target)
                return i;
        }
        return last;
    }

    // Finds token t among the candidates: at index t when the array is still in id order, else by a scan
    template <typename T>
    constexpr token_data<T>* find(token_candidates<T>& c, token_id t)
    {
        if (t >= 0 && static_cast<std::size_t>(t) < c.size() && c.data[static_cast<std::size_t>(t)].id == t)
            return &c.data[static_cast<std::size_t>(t)];
        for (auto& d : c.data)
            if (d.id == t)
                return &d;
        return nullptr;
    }

    // The last `limit` accepted tokens, oldest overwritten: llama.cpp's ring_buffer
    template <std::size_t N>
    struct token_ring
    {
        std::array<token_id, N> buf{};
        std::size_t first = 0, count = 0;

        constexpr void push(token_id t, std::size_t limit)
        {
            if (limit == 0)
                return;
            if (count < limit) {
                buf[(first + count) % N] = t;
                ++count;
            } else {
                buf[(first + count) % N] = t;   // over the oldest when count is N; any limit up to N stays contiguous mod N
                first = (first + 1) % N;
            }
        }
        // i-th from the end: rat(0) is the most recent
        [[nodiscard]] constexpr token_id rat(std::size_t i) const { return buf[(first + count - 1 - i) % N]; }
        constexpr void clear() { first = count = 0; }
    };
} // namespace detail::samplers

/// \brief Selects the first candidate of largest logit (llama.cpp's greedy sampler).
/// \ingroup core
struct greedy_sampler
{
    /// \brief Selects.
    /// \param c The candidates, not empty.
    template <math::math_number T>
    constexpr void apply(token_candidates<T>& c, stats::wyrand&) const
    {
        assert(c.size() > 0);
        std::size_t best = 0;
        for (std::size_t i = 1; i < c.size(); ++i)
            if (c.data[i].logit > c.data[best].logit)
                best = i;
        c.selected = static_cast<std::ptrdiff_t>(best);
    }
};

/// \brief Samples from the softmax of the logits with one `stats::uniform_real` draw (llama.cpp's dist sampler).
/// \ingroup core
struct dist_sampler
{
    /// \brief Computes the probabilities and selects by the inverse CDF.
    /// \param c The candidates, not empty.
    /// \param g The generator: one draw.
    template <math::math_number T>
    constexpr void apply(token_candidates<T>& c, stats::wyrand& g) const
    {
        assert(c.size() > 0);
        const double u = stats::uniform_real(g);   // drawn even for one candidate, as llama.cpp keeps its stream aligned
        T max = c.data[0].logit;
        if (!c.sorted)
            for (const auto& d : c.data)
                max = std::max(max, d.logit);
        double sum = 0.0;   // llama.cpp: float exponentials, a double sum, one pass to select and normalise
        for (auto& d : c.data) {
            d.p = detail::samplers::exp_or_zero(d.logit - max);
            sum += static_cast<double>(d.p);
        }
        const double target = sum * u;
        double run = 0.0;
        std::size_t last = 0;
        c.selected = -1;
        for (std::size_t i = 0; i < c.size(); ++i) {
            if (c.selected < 0) {
                run += static_cast<double>(c.data[i].p);
                if (c.data[i].p > T(0))
                    last = i;
                if (run > target)
                    c.selected = static_cast<std::ptrdiff_t>(i);
            }
            c.data[i].p = static_cast<T>(static_cast<double>(c.data[i].p) / sum);
        }
        if (c.selected < 0)
            c.selected = static_cast<std::ptrdiff_t>(last);
    }
};

/// \brief Divides the logits by a temperature; at 0 or below, keeps only the first largest (others to -infinity).
/// \ingroup core
/// \tparam T The number type.
template <math::math_number T>
struct temperature_sampler
{
    T temp = T(1);   ///< The temperature.

    /// \brief Rescales.
    /// \param c The candidates.
    constexpr void apply(token_candidates<T>& c, stats::wyrand&) const { detail::samplers::temperature(c, temp); }
};

/// \brief Keeps the `k` largest logits, sorted; `k <= 0` keeps all.
/// \ingroup core
struct top_k_sampler
{
    std::int32_t k = 40;   ///< The candidates kept.

    /// \brief Truncates.
    /// \param c The candidates.
    template <math::math_number T>
    constexpr void apply(token_candidates<T>& c, stats::wyrand&) const { detail::samplers::top_k(c, k); }
};

/// \brief Nucleus sampling's truncation: the most probable candidates until their mass reaches `p` (at least `min_keep`).
/// \ingroup core
/// \tparam T The number type.
template <math::math_number T>
struct top_p_sampler
{
    T p = T(0.95);             ///< The mass kept; 1 or more keeps all.
    std::size_t min_keep = 1;  ///< The fewest kept.

    /// \brief Computes probabilities, sorts and truncates (probabilities stay those of the whole array).
    /// \param c The candidates.
    constexpr void apply(token_candidates<T>& c, stats::wyrand&) const
    {
        if (p >= T(1) || c.size() == 0)
            return;
        detail::samplers::softmax(c, false);   // the probabilities in the array's order, then sorted, as llama.cpp
        detail::samplers::sort(c);
        T cum = T(0);
        std::size_t last = c.size();
        for (std::size_t i = 0; i < c.size(); ++i) {
            cum += c.data[i].p;
            if (cum >= p && i + 1 >= min_keep) {
                last = i + 1;
                break;
            }
        }
        c.data = c.data.first(last);
    }
};

/// \brief Min-p: keeps the candidates at least `p` times as probable as the most probable (at least `min_keep`).
/// \ingroup core
/// \tparam T The number type.
template <math::math_number T>
struct min_p_sampler
{
    T p = T(0.05);             ///< The fraction of the top probability; 0 or less keeps all.
    std::size_t min_keep = 1;  ///< The fewest kept.

    /// \brief Truncates, keeping the order when unsorted (llama.cpp's unsorted path), else by the sorted path.
    /// \param c The candidates.
    constexpr void apply(token_candidates<T>& c, stats::wyrand&) const
    {
        if (p <= T(0) || c.size() == 0)
            return;
        if (!c.sorted) {
            T max = c.data[0].logit;
            for (const auto& d : c.data)
                max = std::max(max, d.logit);
            const T cut = max + math::log(p);
            std::size_t kept = 0;
            for (const auto& d : c.data)
                kept += d.logit >= cut ? 1 : 0;
            if (kept > 0 && kept >= min_keep) {
                std::size_t at = 0;
                for (std::size_t i = 0; i < c.size(); ++i)
                    if (c.data[i].logit >= cut)
                        c.data[at++] = c.data[i];
                c.data = c.data.first(kept);
                return;
            }
        }
        detail::samplers::sort(c);
        const T cut = c.data[0].logit + math::log(p);
        std::size_t i = 1;
        for (; i < c.size(); ++i)
            if (c.data[i].logit < cut && i >= min_keep)
                break;
        c.data = c.data.first(i);
    }
};

/// \brief The repetition, frequency and presence penalties over the last `last_n` accepted tokens (llama.cpp's penalties sampler).
/// \ingroup core
/// \tparam T The number type.
/// \tparam N The longest history kept.
template <math::math_number T, std::size_t N = 64>
struct penalties_sampler
{
    static_assert(N > 0, "snicholls::ml::penalties_sampler: the history must hold at least one token");
    std::size_t last_n = N;   ///< The tokens looked at, at most `N`; 0 disables.
    T repeat = T(1);          ///< A seen token's logit divided by it if positive, multiplied if not; 1 for none.
    T frequency = T(0);       ///< Subtracted once per occurrence.
    T presence = T(0);        ///< Subtracted once from a seen token.
    detail::samplers::token_ring<N> history{};   ///< The accepted tokens (state).

    /// \brief Records an accepted token.
    /// \param t The token.
    constexpr void accept(token_id t)
    {
        assert(last_n <= N);
        history.push(t, last_n);
    }
    /// \brief Forgets the history.
    constexpr void reset() { history.clear(); }

    /// \brief Penalises each distinct token of the history once, by its count.
    /// \param c The candidates.
    constexpr void apply(token_candidates<T>& c, stats::wyrand&) const
    {
        if (last_n == 0 || (repeat == T(1) && frequency == T(0) && presence == T(0)))
            return;
        assert(repeat > T(0));
        for (std::size_t i = 0; i < history.count; ++i) {
            const token_id t = history.rat(i);
            bool first = true;
            std::size_t count = 0;
            for (std::size_t j = 0; j < history.count; ++j)
                if (history.rat(j) == t) {
                    first = first && j >= i;
                    ++count;
                }
            if (!first)
                continue;
            auto* const d = detail::samplers::find(c, t);
            if (d == nullptr)
                continue;
            d->logit = d->logit <= T(0) ? d->logit * repeat : d->logit / repeat;
            d->logit -= T(count) * frequency + presence;
        }
        c.sorted = false;
    }
};

/// \brief Locally typical sampling's truncation (Meister et al., 2023): the candidates whose surprise is closest to the entropy, up to mass `p`.
/// \ingroup core
/// \tparam T The number type.
template <math::math_number T>
struct typical_sampler
{
    T p = T(1);                ///< The mass kept; 1 or more keeps all.
    std::size_t min_keep = 1;  ///< The fewest kept.

    /// \brief Truncates, ordering by `|-log p - H|` (ties by the lower id); leaves the array unsorted.
    /// \param c The candidates.
    constexpr void apply(token_candidates<T>& c, stats::wyrand&) const
    {
        if (p >= T(1) || c.size() == 0)
            return;
        detail::samplers::softmax(c, true);
        T entropy = T(0);
        for (const auto& d : c.data)
            if (d.p > T(0))
                entropy += -d.p * math::log(d.p);
        const auto score = [entropy](const token_data<T>& d) {
            return d.p > T(0) ? math::abs(-math::log(d.p) - entropy) : std::numeric_limits<T>::infinity();
        };
        std::sort(c.data.begin(), c.data.end(), [&](const token_data<T>& a, const token_data<T>& b) {
            const T sa = score(a), sb = score(b);
            return sa < sb || (sa == sb && a.id < b.id);
        });
        T cum = T(0);
        std::size_t last = c.size();
        for (std::size_t i = 0; i < c.size(); ++i) {
            cum += c.data[i].p;
            if (cum > p && (min_keep == 0 || i + 1 >= min_keep)) {
                last = i + 1;
                break;
            }
        }
        c.data = c.data.first(last);
        c.sorted = false;
    }
};

/// \brief Dynamic temperature: `temp ± delta`, set by the normalised entropy raised to `exponent` (llama.cpp's temp_ext).
/// \ingroup core
/// \tparam T The number type.
template <math::math_number T>
struct temp_ext_sampler
{
    T temp = T(1);       ///< The centre.
    T delta = T(0);      ///< The half-range; 0 or less is a plain temperature.
    T exponent = T(1);   ///< The power on the normalised entropy.

    /// \brief The temperature it would apply to candidates whose softmax has entropy `entropy` over `n` candidates.
    /// \param entropy The entropy in nats.
    /// \param n The candidates, at least 2.
    /// \return `min + (max - min) · (H / ln n)^exponent`, `min = max(0, temp - delta)`, `max = temp + delta`.
    [[nodiscard]] constexpr T dynamic_temperature(T entropy, std::size_t n) const
    {
        const T lo = std::max(T(0), temp - delta), hi = temp + delta;
        const T max_entropy = -math::log(T(1) / T(n));
        return lo + (hi - lo) * math::pow(entropy / max_entropy, exponent);
    }

    /// \brief Rescales by the dynamic temperature and recomputes the probabilities (in double, as llama.cpp).
    /// \param c The candidates.
    constexpr void apply(token_candidates<T>& c, stats::wyrand&) const
    {
        if (delta <= T(0)) {
            detail::samplers::temperature(c, temp);
            return;
        }
        if (c.size() <= 1)
            return;
        detail::samplers::softmax(c, true);
        T entropy = T(0);
        for (const auto& d : c.data)
            if (d.p > T(0))
                entropy -= d.p * math::log(d.p);
        detail::samplers::temperature(c, dynamic_temperature(entropy, c.size()));
        const double max = static_cast<double>(c.data[0].logit);
        double sum = 0.0;
        for (auto& d : c.data) {
            const double e = d.logit == detail::samplers::neg_inf<T> ? 0.0 : math::exp(static_cast<double>(d.logit) - max);
            d.p = static_cast<T>(e);
            sum += e;
        }
        for (auto& d : c.data)
            d.p = static_cast<T>(static_cast<double>(d.p) / sum);
    }
};

/// \brief XTC, exclude top choices: with probability `probability`, removes every candidate at or above `threshold` except the least probable of them.
/// \ingroup core
/// \tparam T The number type.
template <math::math_number T>
struct xtc_sampler
{
    T probability = T(0);       ///< The chance of acting per call; 0 disables.
    T threshold = T(0.1);       ///< The probability a top choice reaches; above 0.5 disables (at most one could).
    std::size_t min_keep = 1;   ///< The fewest left.

    /// \brief Draws the coin (one `uniform_real`), then truncates from the front.
    /// \param c The candidates.
    /// \param g The generator.
    constexpr void apply(token_candidates<T>& c, stats::wyrand& g) const
    {
        if (probability <= T(0) || threshold > T(0.5) || c.size() < 2)
            return;
        if (static_cast<T>(stats::uniform_real(g)) > probability)
            return;
        detail::samplers::softmax(c, true);
        std::size_t pos_last = 0;
        for (std::size_t i = 0; i < c.size(); ++i) {
            if (c.data[i].p >= threshold)
                pos_last = i;
            else
                break;
        }
        if (c.size() - pos_last >= min_keep && pos_last > 0)
            c.data = c.data.subspan(pos_last);
    }
};

/// \brief Top-n-sigma (Tang et al., 2024): masks every logit below `max - n · σ`, σ the standard deviation of the finite logits.
/// \ingroup core
/// \tparam T The number type.
template <math::math_number T>
struct top_n_sigma_sampler
{
    T n = T(1);   ///< The standard deviations kept; 0 or less disables.

    /// \brief Masks to -infinity.
    /// \param c The candidates.
    constexpr void apply(token_candidates<T>& c, stats::wyrand&) const
    {
        if (n <= T(0) || c.size() <= 1)
            return;
        using W = std::common_type_t<T, double>;
        constexpr T ninf = detail::samplers::neg_inf<T>;
        T max = c.data[0].logit, sum = T(0);
        std::size_t valid = 0;
        for (const auto& d : c.data)
            if (d.logit != ninf) {
                max = std::max(max, d.logit);
                sum += d.logit;
                ++valid;
            }
        const T mean = valid > 0 ? sum / T(valid) : T(0);
        T acc = T(0);
        for (const auto& d : c.data)
            if (d.logit != ninf) {
                const W dv = static_cast<W>(d.logit - mean);
                acc = static_cast<T>(static_cast<W>(acc) + dv * dv);   // pow(float, 2) is a double in llama.cpp
            }
        const T sd = valid > 0 ? math::sqrt(acc / T(valid)) : T(0);
        for (auto& d : c.data)
            if (d.logit < max - n * sd)
                d.logit = ninf;
    }
};

/// \brief Mirostat (Basu et al., 2021, algorithm 1): a top-k from Zipf's exponent and the target surprise, then sampling; `mu` follows the observed surprise.
/// \ingroup core
/// \tparam T The number type.
template <math::math_number T>
struct mirostat_sampler
{
    std::int32_t n_vocab = 0;   ///< The vocabulary's size.
    T tau = T(5);               ///< The target surprise, in bits.
    T eta = T(0.1);             ///< The learning rate.
    std::int32_t m = 100;       ///< The candidates the exponent is estimated from.
    T mu = T(10);               ///< The running maximum surprise (state); `reset` sets `2 tau`.

    /// \brief Restarts at `mu = 2 tau`.
    constexpr void reset() { mu = T(2) * tau; }

    /// \brief Truncates, samples (one draw), and updates `mu`.
    /// \param c The candidates, not empty.
    /// \param g The generator.
    constexpr void apply(token_candidates<T>& c, stats::wyrand& g)
    {
        assert(c.size() > 0 && n_vocab > 0);
        detail::samplers::softmax(c, true);
        T sum_ti_bi = T(0), sum_ti_sq = T(0);
        for (std::size_t i = 0; m > 0 && i < static_cast<std::size_t>(m - 1) && i + 1 < c.size(); ++i) {
            const T t_i = math::log(T(i + 2) / T(i + 1));
            const T b_i = math::log(c.data[i].p / c.data[i + 1].p);
            sum_ti_bi += t_i * b_i;
            sum_ti_sq += t_i * t_i;
        }
        std::size_t kk = 1;
        if (sum_ti_sq > T(0)) {
            const T s_hat = sum_ti_bi / sum_ti_sq, epsilon_hat = s_hat - T(1);
            const T k = math::pow((epsilon_hat * math::pow(T(2), mu)) / (T(1) - math::pow(T(n_vocab), -epsilon_hat)), T(1) / s_hat);
            kk = !(k >= T(1)) ? 1 : k >= T(c.size()) ? c.size() : static_cast<std::size_t>(k);   // NaN to 1
        }
        detail::samplers::top_k(c, static_cast<std::int32_t>(kk));
        detail::samplers::softmax(c, true);
        const std::size_t idx = detail::samplers::inverse_cdf(c, stats::uniform_real(g));
        c.selected = static_cast<std::ptrdiff_t>(idx);
        mu = mu - eta * (-detail::samplers::log2(c.data[idx].p) - tau);
    }
};

/// \brief Mirostat 2.0: keeps the candidates whose surprise is at most `mu`, samples, and updates `mu`.
/// \ingroup core
/// \tparam T The number type.
template <math::math_number T>
struct mirostat_v2_sampler
{
    T tau = T(5);     ///< The target surprise, in bits.
    T eta = T(0.1);   ///< The learning rate.
    T mu = T(10);     ///< The surprise cut (state); `reset` sets `2 tau`.

    /// \brief Restarts at `mu = 2 tau`.
    constexpr void reset() { mu = T(2) * tau; }

    /// \brief Truncates, samples (one draw), and updates `mu`.
    /// \param c The candidates, not empty.
    /// \param g The generator.
    constexpr void apply(token_candidates<T>& c, stats::wyrand& g)
    {
        assert(c.size() > 0);
        detail::samplers::softmax(c, true);
        std::size_t n = 0;
        while (n < c.size() && !(-detail::samplers::log2(c.data[n].p) > mu))
            ++n;
        c.data = c.data.first(n == 0 ? 1 : n);
        detail::samplers::softmax(c, true);
        const std::size_t idx = detail::samplers::inverse_cdf(c, stats::uniform_real(g));
        c.selected = static_cast<std::ptrdiff_t>(idx);
        mu = mu - eta * (-detail::samplers::log2(c.data[idx].p) - tau);
    }
};

/// \brief DRY, "don't repeat yourself": penalises a token that would extend a repeat of the recent context by `multiplier · base^(length - allowed_length)`.
/// \ingroup core
/// \tparam T The number type.
/// \tparam N The longest history kept.
template <math::math_number T, std::size_t N = 64>
struct dry_sampler
{
    static_assert(N > 1, "snicholls::ml::dry_sampler: the history must hold at least two tokens");
    T multiplier = T(0);              ///< The penalty's scale; 0 disables.
    T base = T(1.75);                 ///< The penalty's base; below 1 disables.
    std::int32_t allowed_length = 2;  ///< The longest repeat left unpenalised.
    std::size_t penalty_last_n = N;   ///< The tokens looked at, at most `N`; 0 disables.
    std::span<const std::span<const token_id>> breakers{};   ///< Sequence breakers, head first in reading order: a match stops at one.
    detail::samplers::token_ring<N> history{};               ///< The accepted tokens (state).
    std::array<std::int32_t, N> repeat_count{};              ///< Scratch: the repeat length ending at each position.

    /// \brief Whether the sampler acts at all.
    /// \return Its enabling condition.
    [[nodiscard]] constexpr bool enabled() const noexcept { return multiplier != T(0) && base >= T(1) && penalty_last_n != 0; }

    /// \brief Records an accepted token.
    /// \param t The token.
    constexpr void accept(token_id t)
    {
        assert(penalty_last_n <= N);
        if (enabled())
            history.push(t, penalty_last_n);
    }
    /// \brief Forgets the history.
    constexpr void reset() { history.clear(); }

    /// \brief Penalises the candidates that would extend a repeat.
    /// \param c The candidates.
    constexpr void apply(token_candidates<T>& c, stats::wyrand&)
    {
        if (!enabled())
            return;
        const int last_n_repeat = static_cast<int>(std::min(history.count, penalty_last_n));
        if (last_n_repeat <= allowed_length)
            return;
        const auto rat = [this](int i) { return history.rat(static_cast<std::size_t>(i)); };
        for (int i = 0; i < last_n_repeat; ++i)
            repeat_count[static_cast<std::size_t>(i)] = 0;
        // Step 1: the most recent restart sequence limits the repetition's length
        int rep_limit = last_n_repeat;
        for (int i = 0; i < last_n_repeat; ++i) {
            const token_id token = rat(i);
            int longest = -1;
            for (const auto& b : breakers) {
                if (b.empty() || b[0] != token)
                    continue;
                const int seq_len = static_cast<int>(b.size()) - 1;   // the tail, after the head
                if (seq_len > longest && seq_len <= i) {
                    bool match = true;
                    for (int off = 0; off < seq_len && match; ++off)
                        match = b[static_cast<std::size_t>(off) + 1] == rat(i - off - 1);
                    if (match)
                        longest = seq_len;
                }
            }
            if (longest >= 0) {
                rep_limit = i - longest;
                break;
            }
        }
        if (rep_limit < allowed_length)
            return;
        // Step 2: the reverse Z-algorithm: repeat_count[last - k] is the length of the context's suffix that also ends k back
        {
            const int last = last_n_repeat - 1;
            int rt = 0, lt = 0;
            const auto at = [this](int i) -> std::int32_t& { return repeat_count[static_cast<std::size_t>(i)]; };
            for (int k = 1; k < last_n_repeat; ++k) {
                if (k > rt) {
                    int n = 0;
                    while (n + k < last_n_repeat && rat(n) == rat(n + k))
                        ++n;
                    at(last - k) = std::min(n, rep_limit);
                    if (n > 0) {
                        lt = k;
                        rt = k + n - 1;
                    }
                } else {
                    const int p = k - lt, right_part_len = rt - k + 1;
                    if (at(last - p) < right_part_len) {
                        at(last - k) = std::min(at(last - p), rep_limit);
                    } else {
                        int i = rt + 1;
                        while (i < last_n_repeat && rat(i) == rat(i - k))
                            i += 1;
                        at(last - k) = std::min(i - k, rep_limit);
                        lt = k;
                        rt = i - 1;
                    }
                }
            }
        }
        // Steps 3 and 4: each token that would continue a repeat, penalised once by its longest
        const double max_log = 88.7228391;   // ln of float's largest value, as llama.cpp
        const int max_exponent = base > T(1.000001) ? static_cast<int>(static_cast<float>(max_log) / static_cast<float>(math::log(base))) : 0;
        for (int i = 0; i < last_n_repeat - 1; ++i) {
            const int len = repeat_count[static_cast<std::size_t>(i)];
            if (len < allowed_length)
                continue;
            const token_id token = rat(last_n_repeat - 2 - i);
            bool longest = true;   // only at the longest repeat this token continues (the first of equal ones)
            for (int j = 0; j < last_n_repeat - 1 && longest; ++j)
                if (j != i && rat(last_n_repeat - 2 - j) == token) {
                    const int other = repeat_count[static_cast<std::size_t>(j)];
                    longest = !(other > len || (other == len && j < i));
                }
            if (!longest)
                continue;
            bool single_breaker = false;
            for (const auto& b : breakers)
                single_breaker = single_breaker || (b.size() == 1 && b[0] == token);
            auto* const d = detail::samplers::find(c, token);
            if (d == nullptr || single_breaker)
                continue;
            int e = len - allowed_length;
            if (max_exponent > 0 && e > max_exponent)
                e = max_exponent;
            d->logit -= static_cast<T>(static_cast<double>(multiplier) * math::pow(static_cast<double>(base), static_cast<double>(e)));
        }
        c.sorted = false;
    }
};

/// \brief One entry of `logit_bias_sampler`: a token and the bias added to its logit.
/// \ingroup core
/// \tparam T The number type.
template <math::math_number T>
struct logit_bias_entry
{
    token_id token = 0;   ///< The token.
    T bias = T(0);        ///< Added to its logit; -infinity bans it.
};

/// \brief Adds fixed biases to some tokens' logits.
/// \ingroup core
/// \tparam T The number type.
template <math::math_number T>
struct logit_bias_sampler
{
    std::span<const logit_bias_entry<T>> biases{};   ///< The biases, which must outlive the sampler.

    /// \brief Adds each bias to its token, if present.
    /// \param c The candidates.
    constexpr void apply(token_candidates<T>& c, stats::wyrand&) const
    {
        for (const auto& b : biases)
            if (auto* const d = detail::samplers::find(c, b.token))
                d->logit += b.bias;
    }
};

/// \brief A sampler with state that follows the accepted tokens.
template <typename S>
concept accepting_sampler = requires(S& s, token_id t) { s.accept(t); };

/// \brief A sampler with state to reset.
template <typename S>
concept resettable_sampler = requires(S& s) { s.reset(); };

namespace detail::samplers
{
    template <typename S>
    constexpr void accept_one(S& s, token_id t)
    {
        first_match{[t](accepting_sampler auto& x) { x.accept(t); }, [](auto&) {}}(s);
    }

    template <typename S>
    constexpr void reset_one(S& s)
    {
        first_match{[](resettable_sampler auto& x) { x.reset(); }, [](auto&) {}}(s);
    }

    template <typename T>
    constexpr token_id finish(token_candidates<T>& c, stats::wyrand& g)
    {
        if (c.selected < 0)
            greedy_sampler{}.apply(c, g);
        return c.selected_token();
    }
} // namespace detail::samplers

/// \brief A chain of samplers fixed at compile time: each applied in order, the last selecting.
/// \ingroup core
/// \tparam Samplers The samplers' types.
template <typename... Samplers>
struct sampler_chain
{
    std::tuple<Samplers...> samplers;   ///< The samplers, in order.

    /// \brief Holds the samplers, in the order they apply.
    /// \param s The samplers.
    constexpr sampler_chain(Samplers... s) : samplers(std::move(s)...) {}

    /// \brief Runs the chain over candidates.
    /// \param c The candidates, not empty; left as the last sampler left them.
    /// \param g The generator.
    /// \return The selected token (the greedy choice if no sampler selected).
    template <math::math_number T>
    constexpr token_id sample(token_candidates<T>& c, stats::wyrand& g)
    {
        c.selected = -1;
        constexpr_for<std::size_t{0}, sizeof...(Samplers), std::size_t{1}>([&](auto i) { std::get<decltype(i)::value>(samplers).apply(c, g); });
        return detail::samplers::finish(c, g);
    }

    /// \brief Runs the chain over a row of logits.
    /// \param logits One logit per token.
    /// \param storage The caller's candidates, the row's size.
    /// \param g The generator.
    /// \return The selected token.
    template <row R>
    constexpr token_id sample(const R& logits, std::span<token_data<row_value_t<R>>> storage, stats::wyrand& g)
    {
        auto c = make_candidates(logits, storage);
        return sample(c, g);
    }

    /// \brief Tells every stateful sampler the token that was accepted.
    /// \param t The token.
    constexpr void accept(token_id t)
    {
        constexpr_for<std::size_t{0}, sizeof...(Samplers), std::size_t{1}>([&](auto i) { detail::samplers::accept_one(std::get<decltype(i)::value>(samplers), t); });
    }

    /// \brief Resets every stateful sampler.
    constexpr void reset()
    {
        constexpr_for<std::size_t{0}, sizeof...(Samplers), std::size_t{1}>([&](auto i) { detail::samplers::reset_one(std::get<decltype(i)::value>(samplers)); });
    }
};

/// \cond
template <typename... Samplers>
sampler_chain(Samplers...) -> sampler_chain<Samplers...>;
/// \endcond

/// \brief Any sampler of this header, for a chain built at run time.
/// \tparam T The number type.
/// \tparam History The history of `penalties_sampler` and `dry_sampler`.
template <math::math_number T, std::size_t History = 64>
using any_sampler = std::variant<greedy_sampler, dist_sampler, temperature_sampler<T>, top_k_sampler, top_p_sampler<T>,
                                 min_p_sampler<T>, typical_sampler<T>, temp_ext_sampler<T>, xtc_sampler<T>,
                                 top_n_sigma_sampler<T>, mirostat_sampler<T>, mirostat_v2_sampler<T>,
                                 penalties_sampler<T, History>, dry_sampler<T, History>, logit_bias_sampler<T>>;

/// \brief A chain of samplers built at run time (from a configuration file or a request), as llama.cpp's sampler chain.
/// \ingroup core
/// \tparam T The number type.
/// \tparam History The history of `penalties_sampler` and `dry_sampler`.
template <math::math_number T, std::size_t History = 64>
class runtime_sampler_chain
{
public:
    /// \brief Appends a sampler.
    /// \param s The sampler.
    /// \return This chain.
    constexpr runtime_sampler_chain& add(any_sampler<T, History> s)
    {
        samplers_.push_back(std::move(s));
        return *this;
    }

    /// \brief The samplers, in order.
    /// \return A view of them.
    [[nodiscard]] constexpr std::span<const any_sampler<T, History>> samplers() const noexcept { return samplers_; }

    /// \brief Runs the chain over candidates.
    /// \param c The candidates, not empty.
    /// \param g The generator.
    /// \return The selected token (the greedy choice if no sampler selected).
    constexpr token_id sample(token_candidates<T>& c, stats::wyrand& g)
    {
        c.selected = -1;
        for (auto& s : samplers_)
            std::visit(overloaded{[&](auto& x) { x.apply(c, g); }}, s);
        return detail::samplers::finish(c, g);
    }

    /// \brief Runs the chain over a row of logits.
    /// \param logits One logit per token.
    /// \param storage The caller's candidates, the row's size.
    /// \param g The generator.
    /// \return The selected token.
    template <row R>
        requires std::same_as<row_value_t<R>, T>
    constexpr token_id sample(const R& logits, std::span<token_data<T>> storage, stats::wyrand& g)
    {
        auto c = make_candidates(logits, storage);
        return sample(c, g);
    }

    /// \brief Tells every stateful sampler the token that was accepted.
    /// \param t The token.
    constexpr void accept(token_id t)
    {
        for (auto& s : samplers_)
            std::visit([t](auto& x) { detail::samplers::accept_one(x, t); }, s);
    }

    /// \brief Resets every stateful sampler.
    constexpr void reset()
    {
        for (auto& s : samplers_)
            std::visit([](auto& x) { detail::samplers::reset_one(x); }, s);
    }

private:
    std::vector<any_sampler<T, History>> samplers_;
};

} // namespace snicholls::ml

#endif // snicholls_core_ml_samplers_hpp
