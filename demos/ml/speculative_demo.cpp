//
//  speculative_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/ml/speculative.hpp>: a prompt-lookup draft verified greedily, speculative
//  sampling with its acceptance rate against sum min(p, q), a token tree with its attention mask verified
//  greedily and by sampling, and multi-token-prediction rows as drafters. Exits non-zero if a result is
//  not what it documents.
//

#include <snicholls/core/ml/speculative.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <vector>

namespace {

namespace ml = snicholls::ml;
using snicholls::stats::wyrand;

// A "target model" over a vocabulary of 6 whose greedy choice after token t is (t + 1) mod 6
std::vector<float> target_logits(std::span<const int> context_then_draft, std::size_t rows)
{
    std::vector<float> out(rows * 6, 0.0f);
    for (std::size_t i = 0; i < rows; ++i) {
        const int last = context_then_draft[context_then_draft.size() - rows + i];
        out[i * 6 + static_cast<std::size_t>((last + 1) % 6)] = 1.0f;
    }
    return out;
}

} // namespace

int main()
{
    std::printf("snicholls-core speculative demo\n");
    bool ok = true;

    //! [speculative]
    // Prompt lookup: the history ends in "2 3", which occurred before, followed by "4 5 0".
    const std::vector<int> history{0, 1, 2, 3, 4, 5, 0, 1, 2, 3};
    std::array<int, 3> draft{};
    const std::size_t n = ml::ngram_draft(std::span<const int>(history), 1, 3, std::span<int>(draft));

    // The target scores the last token and the draft in one pass: rows predict draft[0], draft[1], draft[2], then the bonus.
    std::vector<int> seq = history;
    seq.insert(seq.end(), draft.begin(), draft.begin() + static_cast<std::ptrdiff_t>(n));
    const auto logits = target_logits(seq, n + 1);
    const auto greedy = ml::verify_greedy(std::span<const int>(draft.data(), n), logits, 6);

    // Speculative sampling: a draft token from q, judged against the target's p; the result follows p.
    const std::vector<double> p{0.5, 0.2, 0.3, /* bonus row */ 0.2, 0.3, 0.5};
    const std::vector<double> q{0.2, 0.6, 0.2};
    wyrand g(42);
    const std::array<int, 1> one{1};
    const auto sampled = ml::verify_sampled(std::span<const int>(one), p, q, 3, g);
    //! [speculative]

    std::printf("  ngram draft: %zu tokens: %d %d %d\n", n, draft[0], draft[1], draft[2]);
    ok &= n == 3 && draft == std::array<int, 3>{4, 5, 0};
    std::printf("  greedy: accepted %zu, next %d\n", greedy.accepted, greedy.next);
    ok &= greedy.accepted == 3 && greedy.next == 1;
    std::printf("  sampled: accepted %zu, next %d\n", sampled.accepted, sampled.next);
    ok &= sampled.accepted <= 1;

    // The acceptance rate of draft tokens drawn from q is sum min(p, q) = 0.2 + 0.2 + 0.2
    {
        int accepted = 0;
        const int trials = 20000;
        wyrand gq(1);
        for (int t = 0; t < trials; ++t) {
            const double u = snicholls::stats::uniform_real(gq);
            const std::array<int, 1> d{u < 0.2 ? 0 : u < 0.8 ? 1 : 2};
            accepted += static_cast<int>(ml::verify_sampled(std::span<const int>(d), p, q, 3, g).accepted);
        }
        const double rate = static_cast<double>(accepted) / trials;
        std::printf("  acceptance rate %.3f (expected 0.600)\n", rate);
        ok &= std::abs(rate - 0.6) < 0.02;
    }

    // A tree: root -> {4, 2}, 4 -> {5}, 2 -> {3}; its mask, and greedy verification
    {
        const std::array<int, 4> tokens{4, 2, 5, 3};
        const std::array<std::int32_t, 4> parent{-1, -1, 0, 1};
        std::array<std::uint8_t, 16> mask{};
        ml::tree_attention_mask(std::span<const std::int32_t>(parent), std::span(mask));
        ok &= mask[2 * 4 + 0] == 1 && mask[2 * 4 + 1] == 0 && mask[3 * 4 + 1] == 1 && ml::tree_depth(parent, 3) == 2;
        // the target after the root token 1 predicts 2, then 3, then 4 (row 0 root, row i+1 node i)
        std::vector<float> rows(5 * 6, 0.0f);
        const std::array<int, 5> best{2, 5, 3, 0, 4};
        for (std::size_t i = 0; i < 5; ++i)
            rows[i * 6 + static_cast<std::size_t>(best[i])] = 1.0f;
        std::array<std::int32_t, 2> path{};
        const auto r = ml::verify_tree_greedy(std::span<const int>(tokens), std::span<const std::int32_t>(parent), rows, 6, std::span(path));
        std::printf("  tree greedy: path of %zu (nodes %d %d), next %d\n", r.length, path[0], path[1], r.next);
        ok &= r.length == 2 && path[0] == 1 && path[1] == 3 && r.next == 4;

        // Sampled: two siblings drawn from q at the root, verified against p
        const std::array<int, 2> sib{1, 0};
        const std::array<std::int32_t, 2> roots{-1, -1};
        const std::vector<double> tp{0.5, 0.2, 0.3, 0.5, 0.2, 0.3, 0.5, 0.2, 0.3}, tq{0.2, 0.6, 0.2, 0.2, 0.6, 0.2, 0.2, 0.6, 0.2};
        std::array<double, 3> scratch{};
        std::array<std::int32_t, 1> spath{};
        const auto s = ml::verify_tree_sampled(std::span<const int>(sib), std::span<const std::int32_t>(roots), tp, tq, 3, std::span(spath),
                                               std::span(scratch), g);
        std::printf("  tree sampled: path of %zu, next %d\n", s.length, s.next);
        ok &= s.length <= 1;
    }

    // Multi-token prediction: two modules' rows become two draft tokens, greedily and by sampling
    {
        const std::array<float, 6> mtp{0.0f, 4.0f, 1.0f, /**/ 3.0f, 0.0f, 0.5f};
        std::array<int, 2> d{};
        ml::mtp_draft_greedy(mtp, 3, std::span(d));
        std::printf("  mtp greedy: %d %d\n", d[0], d[1]);
        ok &= d[0] == 1 && d[1] == 0;
        std::array<float, 6> dq{};
        ml::mtp_draft_sample(mtp, 3, 1.0f, std::span(d), std::span(dq), g);
        ok &= std::abs(dq[0] + dq[1] + dq[2] - 1.0f) < 1e-6f && dq[1] > dq[2] && d[0] >= 0 && d[0] < 3;
    }

    std::printf(ok ? "speculative demo: ok\n" : "speculative demo: FAILED\n");
    return ok ? 0 : 1;
}
