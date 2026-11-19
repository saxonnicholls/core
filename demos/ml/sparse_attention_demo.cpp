//
//  sparse_attention_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/ml/sparse_attention.hpp>: DeepSeek Sparse Attention for one query: the lightning indexer
//  scores every key, the top k are chosen, attention runs over those only; the backward passes; the indexer's KL loss
//  against the main attention's weights. Exits non-zero if a result is not what it documents.
//

#include <snicholls/core/ml/attention.hpp>
#include <snicholls/core/ml/sparse_attention.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <numeric>
#include <span>

int main()
{
    std::printf("snicholls-core sparse attention demo\n");
    namespace ml = snicholls::ml;
    bool ok = true;
    constexpr float none = -std::numeric_limits<float>::infinity();

    //! [sparse_attention]
    // Twelve keys of 4 channels; the indexer has 2 heads of 3 channels and one key of 3 per position
    constexpr std::size_t n = 12, d = 4, ih = 2, id = 3, k = 4;
    std::array<float, n * d> keys{}, values{};
    std::array<float, n * id> index_keys{};
    for (std::size_t j = 0; j < n; ++j) {
        for (std::size_t i = 0; i < d; ++i) {
            keys[j * d + i] = std::sin(0.7f * static_cast<float>(j * d + i));
            values[j * d + i] = static_cast<float>(j);
        }
        for (std::size_t i = 0; i < id; ++i)
            index_keys[j * id + i] = std::cos(0.9f * static_cast<float>(j + 2 * i));
    }
    const std::array<float, d> q{0.5f, -0.5f, 1.0f, 0.25f};
    const std::array<float, ih * id> index_q{1.0f, 0.5f, -0.5f, 0.2f, 0.8f, 0.4f};
    const std::array<float, ih> index_w{0.7f, 0.3f};

    // I_s = Σ_j w_j ReLU(q_j · k_s); keep the best k, in position order
    std::array<float, n> scores{};
    ml::indexer_scores(index_q, index_w, index_keys, scores);
    std::array<std::uint32_t, k> chosen{};
    const std::size_t count = ml::select_top_k(scores, std::span<std::uint32_t>(chosen));

    // Attention over the chosen keys only
    std::array<float, d> out{};
    const float scale = 0.5f;
    const float lse = ml::attend_selected(q, keys, values, std::span<const std::uint32_t>(chosen).first(count), out, scale);

    // Its backward pass reaches the chosen rows only
    const std::array<float, d> dout{1.0f, 0.0f, 0.0f, 0.0f};
    std::array<float, d> dq{};
    std::array<float, n * d> dkeys{}, dvalues{};
    ml::attend_selected_backward(q, keys, values, std::span<const std::uint32_t>(chosen).first(count), out, dout, lse, scale, none, dq,
                                 dkeys, dvalues);

    // The indexer learns from the main attention's weights over every key (dense warm-up), by KL divergence
    std::array<float, n> target{}, dscores{};
    std::array<float, d> full{};
    const float full_lse = ml::attend(q, keys, values, full, scale);
    for (std::size_t j = 0; j < n; ++j) {
        float s = 0.0f;
        for (std::size_t i = 0; i < d; ++i)
            s += q[i] * keys[j * d + i];
        target[j] = std::exp(scale * s - full_lse);
    }
    const float kl = ml::indexer_kl_loss(target, scores, dscores);
    std::array<float, ih * id> dindex_q{};
    std::array<float, ih> dindex_w{};
    std::array<float, n * id> dindex_keys{};
    ml::indexer_scores_backward(index_q, index_w, index_keys, dscores, dindex_q, dindex_w, dindex_keys);
    //! [sparse_attention]

    std::printf("  chosen keys:");
    for (std::size_t j = 0; j < count; ++j)
        std::printf(" %u", static_cast<unsigned>(chosen[j]));
    std::printf("\n  sparse output: mean position %.4f (dense %.4f)\n", static_cast<double>(out[0]), static_cast<double>(full[0]));
    std::printf("  indexer KL to the dense weights: %.4f nats\n", static_cast<double>(kl));
    float weights = 0.0f, untouched = 0.0f;
    for (std::size_t j = 0; j < n; ++j) {
        weights += dvalues[j * d];
        const bool picked = std::find(chosen.begin(), chosen.begin() + std::ptrdiff_t(count), j) != chosen.begin() + std::ptrdiff_t(count);
        if (!picked)
            untouched += std::fabs(dvalues[j * d]) + std::fabs(dkeys[j * d]);
    }
    ok &= count == k && std::is_sorted(chosen.begin(), chosen.end()) && std::fabs(weights - 1.0f) < 1e-5f && untouched == 0.0f;
    ok &= kl >= 0.0f && std::fabs(std::accumulate(dscores.begin(), dscores.end(), 0.0f)) < 1e-5f;
    // Every key chosen in order is dense attention
    std::array<std::uint32_t, n> all{};
    std::iota(all.begin(), all.end(), 0u);
    std::array<float, d> again{};
    ok &= ml::attend_selected(q, keys, values, all, again, scale) == full_lse && again == full;
    std::printf(ok ? "sparse attention demo: ok\n" : "sparse attention demo: FAILED\n");
    return ok ? 0 : 1;
}
