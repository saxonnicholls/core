//
//  lm_head.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::ml::lm_head_loss`: a language model's loss over a sequence, the head's logits and the cross-entropy fused, and its backward pass into the hidden states and the head (tied or not).
/// \ingroup core
///
/// **Overview.** The end of a language model in training: each position's final hidden state `h_t` becomes
/// logits `W h_t` over the vocabulary, and the loss is the mean cross-entropy against the next tokens. The whole
/// `n x V` matrix of logits is never stored: one row of `V` is computed, turned in place into its gradient
/// (`cross_entropy_with_gradient`, llm.c's fused classifier), and taken back through the head (`linear_backward`)
/// before the next position's. Memory is one row of logits whatever the sequence's length.
///
/// - `lm_head_loss(head, hs, targets, logits)`: the mean loss, for evaluation.
/// - `lm_head_loss(head, hs, targets, logits, dhs, dhead)`: the same loss, adding the gradients of the mean into
///   the hidden states' and the head's.
///
/// A tied head (GPT-2, nanochat's option) is the embedding table itself, `V` rows of `d_model`, viewed as
/// `dense_weights{table, V, d_model}`; its gradient then adds into the embedding's, beside `embedding_backward`'s.
///
/// \snippet ml/lm_head_demo.cpp lm_head
///
/// **Relationships.** `ml/linear.hpp` (`linear`, `linear_backward`), `ml/softmax.hpp`
/// (`cross_entropy_with_gradient`), `ml/embedding.hpp` (a tied table), `ml/transformer_block.hpp` (what produces the
/// hidden states). Group: \ref core.
///
/// **Implementations.** One.
///
/// **Guarantees.** `constexpr`, no allocation (the caller's row of logits), O(`n · V · d_model`) for the loss and
/// twice that with the gradients. The gradients are added into, so micro-batches accumulate.
///
/// **Errors.** Preconditions, asserted: `hs` is `n · d_model` for `n` targets; every target is below `V`;
/// `logits` has `V`; `dhs` has `hs`' size and `dhead` the head's. With no targets the loss is 0 and nothing is added.
///
/// **Performance.** Not benchmarked on its own: it is `linear`, `cross_entropy_with_gradient` and `linear_backward`
/// in sequence, each benchmarked or level with llm.c's loops.
///
/// **Tests and demos.** `tests/ml/lm_head_test.cpp`, `demos/ml/lm_head_demo.cpp`.
///
/// **Dependencies.** `<cassert>`, `<cstddef>`, `<span>`, `ml/linear.hpp`, `ml/row.hpp`, `ml/softmax.hpp`.
///
/// **Known issues.** A soft cap on the logits (Gemma 2, nanochat's 15 · tanh(x / 15)) is not built in.
///
/// **Status.** Experimental, new in 0.9.0.

#ifndef snicholls_core_ml_lm_head_hpp
#define snicholls_core_ml_lm_head_hpp

#include <snicholls/core/ml/linear.hpp>
#include <snicholls/core/ml/row.hpp>
#include <snicholls/core/ml/softmax.hpp>

#include <cassert>
#include <cstddef>
#include <span>

namespace snicholls::ml
{

/// \brief The mean cross-entropy of a sequence's next-token predictions through a head.
/// \param head `V x d_model`, or a tied embedding table viewed as one.
/// \param hs `n` final hidden states of `d_model`, end to end.
/// \param targets `n` next tokens, each below `V`.
/// \param logits Working space, `V`.
/// \return The mean loss, in nats; 0 with no targets.
template <typename S, row H, mutable_row L>
    requires std::same_as<row_value_t<H>, row_value_t<L>>
constexpr row_value_t<H> lm_head_loss(const dense_weights<S>& head, const H& hs, std::span<const std::size_t> targets,
                                      L&& logits)
{
    using T = row_value_t<H>;
    const auto h = as_row(hs);
    const auto z = as_mutable_row(logits);
    const std::size_t n = targets.size(), d = head.cols;
    assert(h.size() == n * d && z.size() == head.rows);
    T sum = T(0);
    for (std::size_t t = 0; t < n; ++t) {
        assert(targets[t] < head.rows);
        linear(head, h.subspan(t * d, d), z);
        sum += cross_entropy(z, targets[t]);
    }
    return n == 0 ? T(0) : sum / T(n);
}

/// \brief The mean cross-entropy, and its gradients added into the hidden states' and the head's.
/// \param head `V x d_model`, or a tied embedding table viewed as one.
/// \param hs `n` final hidden states of `d_model`, end to end.
/// \param targets `n` next tokens, each below `V`.
/// \param logits Working space, `V`: left holding the last position's gradient.
/// \param dhs Added into, `n · d_model`.
/// \param dhead Added into, `V · d_model`.
/// \return The mean loss, in nats; 0 with no targets.
template <typename S, row H, mutable_row L, mutable_row DH, mutable_row DW>
    requires std::same_as<row_value_t<H>, row_value_t<L>> && std::same_as<row_value_t<H>, row_value_t<DH>>
             && std::same_as<row_value_t<H>, row_value_t<DW>>
constexpr row_value_t<H> lm_head_loss(const dense_weights<S>& head, const H& hs, std::span<const std::size_t> targets,
                                      L&& logits, DH&& dhs, DW&& dhead)
{
    using T = row_value_t<H>;
    const auto h = as_row(hs);
    const auto z = as_mutable_row(logits);
    const auto gh = as_mutable_row(dhs);
    const std::size_t n = targets.size(), d = head.cols;
    assert(h.size() == n * d && z.size() == head.rows && gh.size() == h.size());
    T sum = T(0);
    for (std::size_t t = 0; t < n; ++t) {
        assert(targets[t] < head.rows);
        const auto ht = h.subspan(t * d, d);
        linear(head, ht, z);
        sum += cross_entropy_with_gradient(z, targets[t], z, T(1) / T(n));
        linear_backward(head, ht, std::span<const T>(z), gh.subspan(t * d, d), dhead);
    }
    return n == 0 ? T(0) : sum / T(n);
}

} // namespace snicholls::ml

#endif // snicholls_core_ml_lm_head_hpp
