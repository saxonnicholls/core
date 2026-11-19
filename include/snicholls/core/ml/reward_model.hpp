//
//  reward_model.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::ml` reward models: a scalar value head on hidden states, the Bradley-Terry pairwise loss on its rewards, and each sequence's last response position.
/// \ingroup core
///
/// **Overview.** A reward model is a language model whose LM head is replaced by a *value head*, one linear output
/// `r = w · h + b` on the hidden state of the last token of a response, trained on preference pairs to give the chosen
/// response the higher reward (TRAINING.md T13). Its loss is Bradley-Terry's: the probability that `y_w` is preferred
/// is `σ(r_w - r_l)`, and the loss its negative logarithm, `-log σ(r_w - r_l - m)` (InstructGPT, Ouyang et al. 2022,
/// eq. 1; Llama 2, Touvron et al. 2023, adds the margin `m` for pairs rated far apart). The same head on every
/// position is PPO's value function (`ml/rl.hpp`).
///
/// - `bradley_terry{margin}` with `preference_loss(params, r_chosen, r_rejected)`: the loss of one pair, in the shape
///   of `ml/preference.hpp`, so `preference_loss_rows` and `preference_metrics` serve reward models unchanged.
/// - `value_head(w, bias, h)` and `value_head_backward(w, h, dr, dh, dw, db)`: one hidden state.
/// - `value_head_rows(backend, w, bias, hs, values)` and `value_head_rows_backward(...)`: every row, on the backend.
/// - `last_response_positions(mask, positions)`: each sequence's last response position, the index list
///   `gather_rows(b, hs, positions, out, d, rows)` (`ml/gather.hpp`) reads the hidden states at, and its backward
///   `scatter_add_rows` takes the gradient back into the sequence's.
///
/// \snippet ml/reward_model_demo.cpp reward_model
///
/// **Relationships.** `ml/preference.hpp` (`pair_loss`, `preference_loss_rows`, `preference_metrics`), `ml/gather.hpp`
/// (`gather_rows`, `scatter_add_rows`), `ml/transformer_block.hpp` (the hidden states), `ml/rl.hpp` (the value loss).
/// Group: \ref core.
///
/// **Implementations.** One each, on the host; the backend-first functions are the seam for a device (TRAINING.md T9).
///
/// **Guarantees.** `constexpr`, no allocation, `noexcept` for the standard floating-point types; linear in the hidden
/// states. Gradients are added into (micro-batches accumulate); `value_head_rows` writes.
///
/// **Errors.** Preconditions, asserted: sizes agree; every row of a mask has a response token for
/// `last_response_positions`.
///
/// **Definitions.** The Bradley-Terry loss `-log σ(r_w - r_l)` and the margin form: recalled from InstructGPT and
/// Llama 2, not re-read; it is DPO's loss with the implicit reward replaced by the head's.
///
/// **Performance.** Not benchmarked: one dot product a row, negligible next to the model.
///
/// **Tests and demos.** `tests/ml/reward_model_test.cpp`, `demos/ml/reward_model_demo.cpp`.
///
/// **Dependencies.** `<cassert>`, `<concepts>`, `<cstddef>`, `<cstdint>`, `<span>`, `<type_traits>`, `ml/activations.hpp`, `ml/preference.hpp`, `ml/row.hpp`.
///
/// **Known issues.** A head with more than one output (a multi-objective reward) is `linear`'s job.
///
/// **Status.** Experimental, new in Unreleased.

#ifndef snicholls_core_ml_reward_model_hpp
#define snicholls_core_ml_reward_model_hpp

#include <snicholls/core/ml/activations.hpp>
#include <snicholls/core/ml/preference.hpp>
#include <snicholls/core/ml/row.hpp>

#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>

namespace snicholls::ml
{

/// \brief The Bradley-Terry pairwise loss of a reward model: `-log σ(r_w - r_l - margin)`.
/// \ingroup core
struct bradley_terry
{
    double margin = 0.0;   ///< Llama 2's margin `m`: 0 for the plain loss.
};

/// \brief The Bradley-Terry loss of one pair of rewards and its gradient: the reference arguments are ignored.
/// \param c The parameters.
/// \param r_chosen The chosen response's reward.
/// \param r_rejected The rejected response's reward.
/// \return The loss, `∂/∂r_chosen = -σ(-z) = -∂/∂r_rejected` with `z = r_w - r_l - m`, and the rewards themselves.
template <math::math_number T>
constexpr pair_loss<T> preference_loss(const bradley_terry& c, T r_chosen, std::type_identity_t<T> r_rejected,
                                       std::type_identity_t<T> = T(0), std::type_identity_t<T> = T(0)) noexcept(std::is_floating_point_v<T>)
{
    const T z = (r_chosen - r_rejected) - T(c.margin);
    const T s = sigmoid(-z);
    return {softplus(-z), -s, s, r_chosen, r_rejected};
}

/// \brief The value head on one hidden state: `r = Σ_j w_j h_j + bias`, summed in order from the bias.
/// \param w The head's weights, `d`.
/// \param bias The head's bias.
/// \param h The hidden state, `d`.
/// \return The scalar.
template <row W, row H>
    requires std::same_as<row_value_t<W>, row_value_t<H>>
constexpr row_value_t<W> value_head(const W& w, std::type_identity_t<row_value_t<W>> bias, const H& h) noexcept(std::is_floating_point_v<row_value_t<W>>)
{
    const auto a = as_row(w);
    const auto x = as_row(h);
    assert(a.size() == x.size());
    auto r = bias;
    for (std::size_t j = 0; j < a.size(); ++j)
        r += a[j] * x[j];
    return r;
}

/// \brief The backward pass of `value_head`: `dh += dr · w`, `dw += dr · h`, `db += dr`.
/// \param w The weights. \param h The hidden state. \param dr The gradient in the output.
/// \param dh Added into, `d`, or empty. \param dw Added into, `d`. \param db Added into.
template <math::math_number T>
constexpr void value_head_backward(std::span<const T> w, std::type_identity_t<std::span<const T>> h, std::type_identity_t<T> dr,
                                   std::type_identity_t<std::span<T>> dh, std::type_identity_t<std::span<T>> dw, T& db) noexcept(std::is_floating_point_v<T>)
{
    assert(h.size() == w.size() && dw.size() == w.size() && (dh.empty() || dh.size() == w.size()));
    for (std::size_t j = 0; j < w.size(); ++j) {
        if (!dh.empty())
            dh[j] += dr * w[j];
        dw[j] += dr * h[j];
    }
    db += dr;
}

/// \brief The value head on every row, on the backend. \param w `d` weights. \param bias The bias.
/// \param hs `n` hidden states of `d`. \param values Receives `n` scalars.
template <math::math_number T, typename B>
constexpr void value_head_rows(const B&, std::span<const T> w, std::type_identity_t<T> bias, std::type_identity_t<std::span<const T>> hs,
                               std::type_identity_t<std::span<T>> values) noexcept(std::is_floating_point_v<T>)
{
    const std::size_t d = w.size();
    assert(d > 0 && hs.size() == values.size() * d);
    for (std::size_t t = 0; t < values.size(); ++t)
        values[t] = value_head(w, bias, hs.subspan(t * d, d));
}

/// \brief The backward pass of `value_head_rows`, row by row in order. \param w The weights. \param hs The hidden
/// states. \param dvalues The gradients in the outputs, `n`. \param dhs Added into, `n · d`, or empty.
/// \param dw Added into, `d`. \param db Added into.
template <math::math_number T, typename B>
constexpr void value_head_rows_backward(const B&, std::span<const T> w, std::type_identity_t<std::span<const T>> hs,
                                        std::type_identity_t<std::span<const T>> dvalues, std::type_identity_t<std::span<T>> dhs,
                                        std::type_identity_t<std::span<T>> dw, T& db) noexcept(std::is_floating_point_v<T>)
{
    const std::size_t d = w.size();
    assert(d > 0 && hs.size() == dvalues.size() * d && (dhs.empty() || dhs.size() == hs.size()));
    for (std::size_t t = 0; t < dvalues.size(); ++t)
        value_head_backward(w, hs.subspan(t * d, d), dvalues[t], dhs.empty() ? dhs : dhs.subspan(t * d, d), dw, db);
}

/// \brief The position of each sequence's last response token, within the sequence (`t` in `[0, length)`): one index a
/// sequence, as `gather_rows(b, hs, positions, out, d, rows)` reads it.
/// \param mask `rows · length`, 1 for a response token.
/// \param positions Receives `rows` positions.
/// \pre Every row has at least one response token.
constexpr void last_response_positions(std::span<const std::uint8_t> mask, std::span<std::uint32_t> positions) noexcept
{
    const std::size_t rows = positions.size();
    assert(rows == 0 || mask.size() % rows == 0);
    if (rows == 0)
        return;
    const std::size_t len = mask.size() / rows;
    for (std::size_t r = 0; r < rows; ++r) {
        std::size_t last = len;
        for (std::size_t t = 0; t < len; ++t)
            if (mask[r * len + t] != 0)
                last = t;
        assert(last < len);
        positions[r] = static_cast<std::uint32_t>(last);
    }
}

} // namespace snicholls::ml

#endif // snicholls_core_ml_reward_model_hpp
