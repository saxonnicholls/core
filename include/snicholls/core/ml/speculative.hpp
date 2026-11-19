//
//  speculative.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::ml` speculative decoding: n-gram (prompt lookup) drafts, draft-and-verify acceptance for greedy decoding and for sampling (the rejection rule that preserves the target distribution), token trees with their attention mask and path selection, and multi-token-prediction heads as drafters.
/// \ingroup core
///
/// **Overview.** A cheap drafter proposes the next few tokens; the target model scores all of them in one forward
/// pass (a batch of positions, `ml/transformer_block.hpp` over T tokens), and a verifier keeps the longest prefix the
/// target agrees with, plus one token of its own. The output is then distributed as if the target had decoded alone.
///
/// - Drafts: `ngram_draft(history, n_min, n_max, draft)` (prompt lookup decoding): the most recent earlier
///   occurrence of the history's last `n` tokens, `n` from `n_max` down to `n_min`, and the tokens that followed it.
///   `mtp_draft_greedy` / `mtp_draft_sample`: the `D` logit rows of DeepSeek-V3's multi-token-prediction modules at the
///   last position (`ml/multi_token_prediction.hpp`) become `D` draft tokens, by argmax or by sampling, the sampled
///   ones keeping their distributions `q` for the rejection rule.
/// - Verification of a chain: `verify_greedy(draft, target_logits, vocab)`: the longest prefix where the target's
///   argmax is the draft token, and the target's argmax after it (the correction, or the bonus token when every draft
///   token was accepted). `verify_sampled(draft, p, q, vocab, g)`: speculative sampling (Leviathan et al., 2023;
///   Chen et al., 2023): draft token `x` is accepted with probability `min(1, p(x) / q(x))`; at the first rejection
///   the token is drawn from the residual `normalize(max(0, p - q))`; after all are accepted the bonus token is drawn
///   from the target's last row. Each emitted token is distributed exactly as `p`.
/// - Trees: a tree of drafts is `parent[i]` for each node (-1 for a child of the root, the last accepted token), in
///   an order where a parent precedes its children. `tree_attention_mask(parent, mask)` gives the mask under which one
///   forward pass scores every branch (node `i` sees node `j` exactly when `j` is `i` or an ancestor of `i`;
///   SpecInfer, Miao et al., 2024; Medusa, Cai et al., 2024); `tree_depth`. `verify_tree_greedy` walks down from the
///   root along the child that is the target's argmax; `verify_tree_sampled` uses multi-candidate speculative sampling
///   (SpecInfer's rule for siblings drawn independently from the same `q`): try each child in turn, accepting with
///   probability `min(1, p'(x) / q(x))` where `p'` starts at `p` and becomes `normalize(max(0, p' - q))` after each
///   rejection; when every child is rejected, the token is drawn from the last `p'`. It too preserves `p`.
///
/// \snippet ml/speculative_demo.cpp speculative
///
/// **Relationships.** `ml/sampling.hpp` (`argmax`), `ml/softmax.hpp` (the target's probabilities), `ml/samplers.hpp`
/// (the same chain can produce `p` as its candidates' probabilities), `ml/multi_token_prediction.hpp`,
/// `stats/random.hpp`. Group: \ref core.
///
/// **Implementations.** One each. Compared with llama.cpp at 39039aa: `common_sampler_sample_and_accept_n`
/// (`common/sampling.cpp`, greedy: accept while the target's token equals the draft, then one more) and
/// `common_sampler_sample_and_accept_n_rejection` (the same rule as `verify_sampled`: accept when `p_x >= q_x` or
/// `u < p_x / q_x`, else the residual over tokens with `p - q > 0`, falling back to the target's own token if the
/// residual is empty), tested by `tests/test-speculative-rejection-sampling.cpp` as here (the first token's
/// histogram against `p`). Its n-gram drafter `common_ngram_simple_draft` (`common/ngram-map.cpp`) looks up one fixed
/// pattern length and ignores a match at position 0; `ngram_draft` tries every length from `n_max` down, as prompt
/// lookup decoding (Saxena, 2023) does, and accepts any earlier match. llama.cpp's n-gram caches (`common/ngram-cache.*`,
/// statistics over many contexts saved to files) are not here. Recalled, not read: the tree rules come from the
/// SpecInfer and Medusa papers, not from llama.cpp (which has no tree verification at this commit).
///
/// **Guarantees.** `constexpr`, no allocation: the caller passes the rows and the scratch. Deterministic from a
/// `stats::wyrand`. Each function consumes a stated number of draws (one per decision: an acceptance test or a
/// residual or bonus sample). Complexity: `ngram_draft` O(h · n_max · (n_max - n_min + 1)); the verifiers O(k · V) for
/// `k` draft tokens over a vocabulary of `V`; `tree_attention_mask` O(n · depth).
///
/// **Errors.** Preconditions, asserted: rows of `vocab` entries, one per draft position (plus one for the bonus);
/// tokens below `vocab`; the parents of a tree precede their children; `q(x) > 0` for a drafted `x` (a draft cannot
/// propose what its own distribution excludes); scratch of `vocab` entries where required.
///
/// **Performance.** Not benchmarked: the verifiers are a few passes over each row, against a forward pass of the
/// target that costs millions of times more.
///
/// **Tests and demos.** `tests/ml/speculative_test.cpp`, `demos/ml/speculative_demo.cpp`.
///
/// **Dependencies.** `<algorithm>`, `<cassert>`, `<concepts>`, `<cstddef>`, `<cstdint>`, `<limits>`, `<span>`, `<type_traits>`,
/// `snicholls/core/math/constexpr_math.hpp`, `snicholls/core/stats/random.hpp`, `ml/row.hpp`, `ml/sampling.hpp`.
///
/// **Known issues.** The target's probabilities must come from the same sampling configuration (temperature,
/// truncation) the drafts are judged by, or the output follows the configured `p`, not the model's. A tree's
/// sampled verification assumes siblings were drawn independently from their parent's `q` (with replacement); a tree
/// built by taking the top-k without replacement needs a different rule. The end-of-generation stop inside a draft
/// (llama.cpp stops accepting after an EOG token) needs the vocabulary and is the caller's.
///
/// **Status.** Experimental, new in 0.17.0.

#ifndef snicholls_core_ml_speculative_hpp
#define snicholls_core_ml_speculative_hpp

#include <snicholls/core/math/constexpr_math.hpp>
#include <snicholls/core/ml/row.hpp>
#include <snicholls/core/ml/sampling.hpp>
#include <snicholls/core/stats/random.hpp>

#include <algorithm>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <type_traits>

namespace snicholls::ml
{

/// \brief Prompt lookup decoding: the tokens that followed the most recent earlier occurrence of the history's last `n` tokens, the longest `n` first.
/// \param history The context so far, the last sampled token included.
/// \param n_min The shortest suffix looked up, at least 1.
/// \param n_max The longest, at least `n_min`.
/// \param draft Receives up to its size of draft tokens.
/// \return The number written: 0 when no suffix of `n_min` or more tokens occurs earlier.
template <std::integral Token>
constexpr std::size_t ngram_draft(std::span<const Token> history, std::size_t n_min, std::size_t n_max,
                                  std::type_identity_t<std::span<Token>> draft)
{
    assert(n_min >= 1 && n_max >= n_min);
    const std::size_t h = history.size();
    for (std::size_t n = std::min(n_max, h == 0 ? 0 : h - 1); n >= n_min; --n) {
        const auto pattern = history.subspan(h - n);
        // the most recent start j < h - n whose n tokens are the pattern
        for (std::size_t j = h - n; j-- > 0;) {
            if (!std::equal(pattern.begin(), pattern.end(), history.begin() + static_cast<std::ptrdiff_t>(j)))
                continue;
            const std::size_t from = j + n, count = std::min(draft.size(), h - from);
            std::copy_n(history.begin() + static_cast<std::ptrdiff_t>(from), count, draft.begin());
            return count;
        }
        if (n == n_min)
            break;
    }
    return 0;
}

/// \brief The outcome of verifying a draft.
/// \ingroup core
/// \tparam Token The token type.
template <std::integral Token>
struct verify_result
{
    std::size_t accepted = 0;   ///< Draft tokens accepted, a prefix of the draft.
    Token next = 0;             ///< The target's token after them: the correction, or the bonus when all were accepted.
};

/// \brief Greedy verification: the longest prefix of the draft where the target's argmax is the draft token, and the argmax after it.
/// \param draft The draft tokens, `k`.
/// \param target_logits The target's logits at the `k + 1` positions, row `i` predicting draft token `i` (row `k` the bonus), `(k + 1) · vocab`.
/// \param vocab The vocabulary's size.
/// \return The tokens accepted and the next token.
template <std::integral Token, row R>
constexpr verify_result<Token> verify_greedy(std::span<const Token> draft, const R& target_logits, std::size_t vocab)
{
    const auto x = as_row(target_logits);
    assert(vocab > 0 && x.size() == (draft.size() + 1) * vocab);
    std::size_t i = 0;
    for (; i <= draft.size(); ++i) {
        const auto best = static_cast<Token>(argmax(x.subspan(i * vocab, vocab)));
        if (i == draft.size() || best != draft[i])
            return {i, best};
    }
    return {draft.size(), Token(0)};   // not reached
}

namespace detail::speculative
{
    // Inverse CDF over weights w (any nonnegative scale), u in [0, 1): the first whose running sum exceeds u · total
    template <typename W>
    constexpr std::size_t pick(W&& weight, std::size_t n, double u)
    {
        double total = 0.0;
        for (std::size_t i = 0; i < n; ++i)
            total += static_cast<double>(weight(i));
        const double target = u * total;
        double run = 0.0;
        std::size_t last = 0;
        for (std::size_t i = 0; i < n; ++i) {
            const double v = static_cast<double>(weight(i));
            run += v;
            if (v > 0.0)
                last = i;
            if (run > target)
                return i;
        }
        return last;
    }

    // One speculative-sampling acceptance test for draft token x: one draw unless p(x) >= q(x), as llama.cpp
    template <typename T>
    constexpr bool accept(std::span<const T> p, std::span<const T> q, std::size_t x, stats::wyrand& g)
    {
        assert(q[x] > T(0));
        const T px = p[x], qx = q[x];
        return px >= qx || static_cast<T>(stats::uniform_real(g)) < px / qx;   // the draw only when needed
    }
} // namespace detail::speculative

/// \brief Speculative sampling (Leviathan et al., 2023): accepts draft token `x` with probability `min(1, p(x)/q(x))`; at the first rejection draws from `normalize(max(0, p - q))`; after all, the bonus from the last `p`.
/// \param draft The draft tokens, `k`.
/// \param target The target's probabilities at the `k + 1` positions, `(k + 1) · vocab` (rows summing to 1).
/// \param proposal The drafter's probabilities the draft tokens were drawn from, `k · vocab`.
/// \param vocab The vocabulary's size.
/// \param g The generator: one draw per acceptance test where `p(x) < q(x)`, and one for the residual or the bonus.
/// \return The tokens accepted and the next token, each distributed as the target's `p`.
template <std::integral Token, row P, row Q>
    requires std::same_as<row_value_t<P>, row_value_t<Q>>
constexpr verify_result<Token> verify_sampled(std::span<const Token> draft, const P& target, const Q& proposal, std::size_t vocab,
                                              stats::wyrand& g)
{
    using T = row_value_t<P>;
    const auto p = as_row(target);
    const auto q = as_row(proposal);
    const std::size_t k = draft.size();
    assert(vocab > 0 && p.size() == (k + 1) * vocab && q.size() == k * vocab);
    for (std::size_t i = 0; i < k; ++i) {
        const auto pi = p.subspan(i * vocab, vocab), qi = q.subspan(i * vocab, vocab);
        const auto x = static_cast<std::size_t>(draft[i]);
        assert(x < vocab);
        if (detail::speculative::accept(pi, qi, x, g))
            continue;
        const auto residual = [&](std::size_t t) { return pi[t] > qi[t] ? pi[t] - qi[t] : T(0); };
        const double u = stats::uniform_real(g);
        bool any = false;
        for (std::size_t t = 0; t < vocab && !any; ++t)
            any = residual(t) > T(0);
        const std::size_t next = any ? detail::speculative::pick(residual, vocab, u) : detail::speculative::pick([&](std::size_t t) { return pi[t]; }, vocab, u);
        return {i, static_cast<Token>(next)};
    }
    const auto last = p.subspan(k * vocab, vocab);
    return {k, static_cast<Token>(detail::speculative::pick([&](std::size_t t) { return last[t]; }, vocab, stats::uniform_real(g)))};
}

/// \brief The depth of tree node `i`: 1 for a child of the root.
/// \param parent Each node's parent, -1 for the root's children; parents precede children.
/// \param i The node.
/// \return The number of nodes on the path from the root to `i`, `i` included.
constexpr std::size_t tree_depth(std::span<const std::int32_t> parent, std::size_t i)
{
    std::size_t d = 0;
    for (std::int32_t at = static_cast<std::int32_t>(i); at >= 0; at = parent[static_cast<std::size_t>(at)]) {
        assert(static_cast<std::size_t>(at) < parent.size() && (parent[static_cast<std::size_t>(at)] < at));
        ++d;
    }
    return d;
}

/// \brief The attention mask of a token tree, for scoring every branch in one forward pass: `mask[i · n + j]` is 1 exactly when node `j` is node `i` or one of its ancestors.
/// \param parent Each node's parent, -1 for the root's children; parents precede children.
/// \param mask Receives `n · n` entries, row by query node (the context before the tree is visible to every node, and not in the mask).
template <std::integral M, std::size_t E>
constexpr void tree_attention_mask(std::span<const std::int32_t> parent, std::span<M, E> mask)
{
    const std::size_t n = parent.size();
    assert(mask.size() == n * n);
    std::fill(mask.begin(), mask.end(), M(0));
    for (std::size_t i = 0; i < n; ++i)
        for (std::int32_t at = static_cast<std::int32_t>(i); at >= 0; at = parent[static_cast<std::size_t>(at)]) {
            assert(parent[static_cast<std::size_t>(at)] < at);
            mask[i * n + static_cast<std::size_t>(at)] = M(1);
        }
}

/// \brief The outcome of verifying a tree: the accepted path and the target's next token.
/// \ingroup core
/// \tparam Token The token type.
template <std::integral Token>
struct tree_verify_result
{
    std::size_t length = 0;   ///< Nodes on the accepted path (written to the caller's `path`).
    Token next = 0;           ///< The target's token after the path.
};

/// \brief Greedy verification of a tree: from the root, follow the child whose token is the target's argmax, as far as one exists.
/// \param tokens Each node's token, `n`.
/// \param parent Each node's parent, -1 for the root's children; parents precede children.
/// \param target_logits The target's logits, `(n + 1) · vocab`: row 0 at the root (predicting its children), row `i + 1` at node `i`.
/// \param vocab The vocabulary's size.
/// \param path Receives the accepted nodes, root side first; at least the tree's depth.
/// \return The path's length and the next token.
template <std::integral Token, row R>
constexpr tree_verify_result<Token> verify_tree_greedy(std::span<const Token> tokens, std::span<const std::int32_t> parent, const R& target_logits,
                                                       std::size_t vocab, std::span<std::int32_t> path)
{
    const auto x = as_row(target_logits);
    const std::size_t n = tokens.size();
    assert(parent.size() == n && x.size() == (n + 1) * vocab);
    std::int32_t at = -1;
    std::size_t len = 0;
    for (;;) {
        const auto best = static_cast<Token>(argmax(x.subspan(static_cast<std::size_t>(at + 1) * vocab, vocab)));
        std::int32_t child = -1;
        for (std::size_t c = 0; c < n && child < 0; ++c)
            if (parent[c] == at && tokens[c] == best)
                child = static_cast<std::int32_t>(c);
        if (child < 0)
            return {len, best};
        assert(len < path.size());
        path[len++] = child;
        at = child;
    }
}

/// \brief Sampled verification of a tree (multi-candidate speculative sampling): at each node, try its children in order, each accepted with probability `min(1, p'(x)/q(x))`, `p'` replaced by `normalize(max(0, p' - q))` after each rejection; with every child rejected, draw from `p'`.
/// \param tokens Each node's token, `n`; siblings drawn independently from their parent's `q`.
/// \param parent Each node's parent, -1 for the root's children; parents precede children.
/// \param target The target's probabilities, `(n + 1) · vocab`: row 0 at the root, row `i + 1` at node `i`.
/// \param proposal The drafter's probabilities the children were drawn from, laid out as `target`.
/// \param vocab The vocabulary's size.
/// \param path Receives the accepted nodes, root side first.
/// \param scratch `vocab` values: the residual distribution.
/// \param g The generator.
/// \return The path's length and the next token, distributed as the target's `p`.
template <std::integral Token, row P, row Q>
    requires std::same_as<row_value_t<P>, row_value_t<Q>>
constexpr tree_verify_result<Token> verify_tree_sampled(std::span<const Token> tokens, std::span<const std::int32_t> parent, const P& target,
                                                        const Q& proposal, std::size_t vocab, std::span<std::int32_t> path,
                                                        std::span<row_value_t<P>> scratch, stats::wyrand& g)
{
    using T = row_value_t<P>;
    const auto p = as_row(target);
    const auto q = as_row(proposal);
    const std::size_t n = tokens.size();
    assert(parent.size() == n && p.size() == (n + 1) * vocab && q.size() == p.size() && scratch.size() == vocab);
    std::int32_t at = -1;
    std::size_t len = 0;
    for (;;) {
        const std::size_t row = static_cast<std::size_t>(at + 1) * vocab;
        const auto qi = q.subspan(row, vocab);
        std::copy_n(p.begin() + static_cast<std::ptrdiff_t>(row), vocab, scratch.begin());
        std::int32_t accepted = -1;
        for (std::size_t c = 0; c < n && accepted < 0; ++c) {
            if (parent[c] != at)
                continue;
            const auto x = static_cast<std::size_t>(tokens[c]);
            assert(x < vocab);
            if (detail::speculative::accept(std::span<const T>(scratch), qi, x, g)) {
                accepted = static_cast<std::int32_t>(c);
                break;
            }
            T sum = T(0);   // p' <- normalize(max(0, p' - q)); left as it was if that is empty
            for (std::size_t t = 0; t < vocab; ++t)
                sum += scratch[t] > qi[t] ? scratch[t] - qi[t] : T(0);
            if (sum > T(0))
                for (std::size_t t = 0; t < vocab; ++t)
                    scratch[t] = (scratch[t] > qi[t] ? scratch[t] - qi[t] : T(0)) / sum;
        }
        if (accepted < 0)
            return {len, static_cast<Token>(detail::speculative::pick([&](std::size_t t) { return scratch[t]; }, vocab, stats::uniform_real(g)))};
        assert(len < path.size());
        path[len++] = accepted;
        at = accepted;
    }
}

/// \brief Multi-token-prediction heads as a drafter, greedily: module `k`'s logit row at the last position gives draft token `k` as its argmax.
/// \param logits The `D` modules' logits, `D · vocab`, module 1 first.
/// \param vocab The vocabulary's size.
/// \param draft Receives `D` tokens.
template <row R, std::integral Token, std::size_t E>
constexpr void mtp_draft_greedy(const R& logits, std::size_t vocab, std::span<Token, E> draft)
{
    const auto x = as_row(logits);
    assert(vocab > 0 && x.size() == draft.size() * vocab);
    for (std::size_t k = 0; k < draft.size(); ++k)
        draft[k] = static_cast<Token>(argmax(x.subspan(k * vocab, vocab)));
}

/// \brief Multi-token-prediction heads as a sampling drafter: each module's softmax at a temperature, kept as `q` for `verify_sampled`, and one token drawn from it.
/// \param logits The `D` modules' logits, `D · vocab`.
/// \param vocab The vocabulary's size.
/// \param temperature Greater than 0, the target's.
/// \param draft Receives `D` tokens.
/// \param q Receives the `D` distributions, `D · vocab`.
/// \param g The generator: one draw per module.
template <row R, std::integral Token, std::size_t E>
constexpr void mtp_draft_sample(const R& logits, std::size_t vocab, std::type_identity_t<row_value_t<R>> temperature, std::span<Token, E> draft,
                                std::span<row_value_t<R>> q, stats::wyrand& g)
{
    using T = row_value_t<R>;
    const auto x = as_row(logits);
    assert(vocab > 0 && temperature > T(0) && x.size() == draft.size() * vocab && q.size() == x.size());
    for (std::size_t k = 0; k < draft.size(); ++k) {
        const auto row = x.subspan(k * vocab, vocab);
        const auto out = q.subspan(k * vocab, vocab);
        const T max = row[argmax(row)];
        T sum = T(0);
        for (std::size_t t = 0; t < vocab; ++t) {
            out[t] = math::exp((row[t] - max) / temperature);
            sum += out[t];
        }
        for (T& v : out)
            v /= sum;
        draft[k] = static_cast<Token>(detail::speculative::pick([&](std::size_t t) { return out[t]; }, vocab, stats::uniform_real(g)));
    }
}

} // namespace snicholls::ml

#endif // snicholls_core_ml_speculative_hpp
