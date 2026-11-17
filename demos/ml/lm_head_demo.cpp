//
//  lm_head_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/ml/lm_head.hpp>: the loss of a sequence through a tied head, its gradients, and gradient
//  descent on the shared table lowering it. Exits non-zero if a result is not what it documents.
//

#include <snicholls/core/ml/lm_head.hpp>

#include <snicholls/core/linalg/reference.hpp>

#include <snicholls/core/ml/embedding.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <vector>

int main()
{
    std::printf("snicholls-core lm_head demo\n");
    namespace ml = snicholls::ml;
    bool ok = true;

    //! [lm_head]
    // A vocabulary of 5 tokens, 3 channels, one table for the embedding and the head (tied, as GPT-2)
    constexpr std::size_t vocab = 5, d = 3;
    std::vector<float> table(vocab * d);
    for (std::size_t i = 0; i < table.size(); ++i)
        table[i] = 0.3f * std::sin(static_cast<float>(i));
    const std::vector<std::size_t> tokens{0, 1, 2, 3}, next{1, 2, 3, 4};   // learn "the next token is one more"

    std::vector<float> hs(tokens.size() * d), logits(vocab), dhs(hs.size()), dtable(table.size());
    float first = 0.0f, last = 0.0f;
    for (int step = 0; step < 50; ++step) {
        for (std::size_t t = 0; t < tokens.size(); ++t)
            ml::embed(table, tokens[t], std::span<float>(hs).subspan(t * d, d));   // here the model is the embedding alone
        std::fill(dhs.begin(), dhs.end(), 0.0f);
        std::fill(dtable.begin(), dtable.end(), 0.0f);
        const ml::dense_weights<float> head{table, vocab, d};
        const float loss = ml::lm_head_loss(head, hs, next, logits, dhs, dtable);   // the head's gradient into dtable
        for (std::size_t t = 0; t < tokens.size(); ++t)
            ml::embedding_backward(std::span<const float>(dhs).subspan(t * d, d), tokens[t], dtable);   // and the embedding's
        for (std::size_t i = 0; i < table.size(); ++i)
            table[i] -= 0.5f * dtable[i];
        (step == 0 ? first : last) = loss;
    }
    // Evaluation, with the trained table, needs no gradient buffers
    for (std::size_t t = 0; t < tokens.size(); ++t)
        ml::embed(table, tokens[t], std::span<float>(hs).subspan(t * d, d));
    const float eval = ml::lm_head_loss(ml::dense_weights<float>{table, vocab, d}, hs, next, logits);
    //! [lm_head]

    std::printf("  loss: %.4f at step 0, %.4f at step 49, %.4f after the last step\n", static_cast<double>(first),
                static_cast<double>(last), static_cast<double>(eval));
    ok &= last < 0.7f * first && eval < last;   // descent lowers the loss, and the last step lowered it again
    //! [lm_head_sequence]
    // For training on a device: the whole sequence's logits as one product through a linalg backend (here core's
    // reference), then the fused classifier on every row, then the gradients; the same loss as lm_head_loss
    std::vector<float> all_logits(tokens.size() * vocab), losses(tokens.size()), dhs2(hs.size()), dtable2(table.size());
    const std::vector<std::uint32_t> next32(next.begin(), next.end());
    const float seq_loss = ml::lm_head_loss_sequence(snicholls::linalg::reference, ml::dense_weights<float>{table, vocab, d}, std::span<const float>(hs),
                                                     next32, std::span<float>(all_logits), std::span<float>(losses), std::span<float>(dhs2),
                                                     std::span<float>(dtable2));
    // A head padded past its classes, as GPT-2's 50,257 tokens are laid out in 50,304 rows: the logits rows as wide as
    // the padding, the loss over the classes only
    std::vector<float> padded(table.size() + 3 * d, 0.0f);
    std::copy(table.begin(), table.end(), padded.begin());
    std::vector<float> wide_logits(tokens.size() * (vocab + 3)), dhs3(hs.size()), dpadded(padded.size());
    const float padded_loss = ml::lm_head_loss_sequence(snicholls::linalg::reference, ml::dense_weights<float>{padded, vocab + 3, d},
                                                        std::span<const float>(hs), next32, std::span<float>(wide_logits), std::span<float>(losses),
                                                        std::span<float>(dhs3), std::span<float>(dpadded), vocab);
    //! [lm_head_sequence]
    std::printf("  the batched loss: %.4f (the per-row %.4f); padded %.4f\n", static_cast<double>(seq_loss), static_cast<double>(eval),
                static_cast<double>(padded_loss));
    ok &= std::fabs(seq_loss - eval) < 1e-6f && padded_loss == seq_loss;

    std::printf(ok ? "lm_head demo: ok\n" : "lm_head demo: FAILED\n");
    return ok ? 0 : 1;
}
