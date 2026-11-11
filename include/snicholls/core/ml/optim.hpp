//
//  optim.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::ml` training arithmetic: the AdamW step, learning-rate schedules, clipping by the global norm, Muon's Newton-Schulz polynomial, and the FLOPs and utilisation of a transformer.
/// \ingroup core
///
/// **Overview.** The per-element and per-step arithmetic of training, as llm.c and nanochat do it. Each is a pure
/// function or an elementwise loop over the caller's rows, so a tensor is updated in shards by as many workers as
/// it has, and a step is the same wherever it runs.
///
/// - `adamw_step(params, grads, m, v, step, config)`: one AdamW update (Loshchilov and Hutter), with bias
///   correction and decoupled weight decay, exactly llm.c's `adamw_update`. `step` counts from 1.
/// - `warmup_cosine(step, warmup, total, max_lr, min_lr)`: linear warm-up then cosine decay (llm.c, GPT-2).
/// - `warmup_warmdown(step, total, warmup, warmdown, final_fraction)`: the multiplier of nanochat: linear warm-up, a
///   constant, then linear decay to `final_fraction` (the "trapezoid" schedule).
/// - `squared_norm(row)` and `clip_coefficient(norm, max_norm)`: clipping by the global norm, as PyTorch's
///   `clip_grad_norm_`. Each worker sums the squares of its shard, the sums add (the merge), and every gradient is
///   multiplied by the coefficient.
/// - `muon_newton_schulz<T>` and `newton_schulz_polynomial(x)`: the quintic Muon iterates on each singular value
///   (Jordan et al., nanochat's optimiser), `a x + b x³ + c x⁵` with `(a, b, c) = (3.4445, -4.7750, 2.0315)`. The
///   matrix iteration, `X ← a X + (b A + c A²) X` with `A = X Xᵀ`, belongs with the matrices, above core.
/// - `transformer_flops_per_token(...)` and `model_flops_utilisation(...)`: `6N + 12·L·H·Q·T` (PaLM's appendix,
///   llm.c, nanochat) and the fraction of the hardware's peak a run achieves.
///
/// \snippet ml/optim_demo.cpp optim
///
/// **Relationships.** `ml/minifloat.hpp` (`to_minifloat_stochastic`, for bfloat16 master weights),
/// `ml/row.hpp`. Group: \ref core.
///
/// **Implementations.** One each.
///
/// **Guarantees.** `constexpr`, no allocation, linear in the rows. `adamw_config` is trivially copyable.
///
/// **Errors.** Preconditions, asserted: the rows of `adamw_step` have one size; `step` is at least 1; `warmup` and
/// `warmdown` fit in `total`. `clip_coefficient` is 1 at or below `max_norm`, and 1 for a NaN norm: a NaN
/// gradient is the caller's to catch.
///
/// **Performance.** `adamw_step` against llm.c's loop written by hand (`benchmarks/ml/optim_benchmark.cpp`), 1M float
/// parameters: level, 860 million parameters per second (Xeon W-3245, Release, Apple clang 17, 2026-11-11, one thread). The schedules and
/// the polynomial are a few operations each and not benchmarked.
///
/// **Tests and demos.** `tests/ml/optim_test.cpp`, `demos/ml/optim_demo.cpp`.
///
/// **Dependencies.** `<cassert>`, `<concepts>`, `<cstddef>`, `<cstdint>`, `<numbers>`,
/// `snicholls/core/math/constexpr_math.hpp`, `ml/row.hpp`.
///
/// **Known issues.** The moments accumulate in the row's type, as llm.c's do in `float`.
///
/// **Status.** Experimental, new in 0.7.0.

#ifndef snicholls_core_ml_optim_hpp
#define snicholls_core_ml_optim_hpp

#include <snicholls/core/math/constexpr_math.hpp>
#include <snicholls/core/ml/row.hpp>

#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <numbers>

namespace snicholls::ml
{

/// \brief The hyperparameters of AdamW.
/// \ingroup core
/// \tparam T The number type.
template <math::math_number T>
struct adamw_config
{
    T learning_rate = T(1e-3);   ///< The step size, usually from a schedule.
    T beta1 = T(0.9);            ///< The decay of the first moment.
    T beta2 = T(0.999);          ///< The decay of the second moment: 0.95 in GPT-2 training.
    T eps = T(1e-8);             ///< Added to the root of the second moment.
    T weight_decay = T(0);       ///< Decoupled weight decay, multiplied by the learning rate.
};

/// \brief One AdamW step over a row of parameters: updates `params`, `m` and `v` in place.
///
/// `m = β1 m + (1 - β1) g`, `v = β2 v + (1 - β2) g²`, then
/// `p -= lr · (m / (1 - β1^t) / (sqrt(v / (1 - β2^t)) + eps) + wd · p)`.
/// \param params The parameters.
/// \param grads Their gradients.
/// \param m The first moments, zero before the first step.
/// \param v The second moments, zero before the first step.
/// \param step The step, from 1.
/// \param config The hyperparameters.
template <mutable_row P, row G, mutable_row M, mutable_row V>
    requires std::same_as<row_value_t<P>, row_value_t<G>> && std::same_as<row_value_t<P>, row_value_t<M>>
             && std::same_as<row_value_t<P>, row_value_t<V>>
constexpr void adamw_step(P&& params, const G& grads, M&& m, V&& v, std::uint64_t step, const adamw_config<row_value_t<P>>& config)
{
    using T = row_value_t<P>;
    const auto p = as_mutable_row(params);
    const auto g = as_row(grads);
    const auto mm = as_mutable_row(m);
    const auto vv = as_mutable_row(v);
    assert(step >= 1 && g.size() == p.size() && mm.size() == p.size() && vv.size() == p.size());
    const T c1 = T(1) - math::pow(config.beta1, T(step));
    const T c2 = T(1) - math::pow(config.beta2, T(step));
    for (std::size_t i = 0; i < p.size(); ++i) {
        mm[i] = config.beta1 * mm[i] + (T(1) - config.beta1) * g[i];
        vv[i] = config.beta2 * vv[i] + (T(1) - config.beta2) * g[i] * g[i];
        const T m_hat = mm[i] / c1;
        const T v_hat = vv[i] / c2;
        p[i] -= config.learning_rate * (m_hat / (math::sqrt(v_hat) + config.eps) + config.weight_decay * p[i]);
    }
}

/// \brief The learning rate at `step` of linear warm-up then cosine decay: llm.c's and GPT-2's schedule.
/// \param step The step, from 0.
/// \param warmup The warm-up steps.
/// \param total The steps of the run; past it, `min_lr`.
/// \param max_lr The rate at the end of warm-up.
/// \param min_lr The rate at the end of decay.
/// \return The learning rate.
template <math::math_number T>
constexpr T warmup_cosine(std::uint64_t step, std::uint64_t warmup, std::uint64_t total, T max_lr, std::type_identity_t<T> min_lr)
{
    assert(warmup <= total);
    if (step < warmup)
        return max_lr * T(step + 1) / T(warmup);
    if (step >= total)
        return min_lr;
    const T ratio = T(step - warmup) / T(total - warmup);
    const T coefficient = T(0.5) * (T(1) + math::cos(std::numbers::pi_v<T> * ratio));
    return min_lr + coefficient * (max_lr - min_lr);
}

/// \brief The learning-rate multiplier at `step` of nanochat's schedule: linear warm-up, constant, linear warm-down to `final_fraction`.
/// \param step The step, from 0.
/// \param total The steps of the run.
/// \param warmup The warm-up steps.
/// \param warmdown The warm-down steps, at the end.
/// \param final_fraction The multiplier at the last step.
/// \return The multiplier, in `[final_fraction, 1]` after warm-up.
template <math::math_number T>
constexpr T warmup_warmdown(std::uint64_t step, std::uint64_t total, std::uint64_t warmup, std::uint64_t warmdown, T final_fraction)
{
    assert(warmup + warmdown <= total);
    if (step < warmup)
        return T(step + 1) / T(warmup);
    if (step + warmdown <= total)
        return T(1);
    const T progress = step >= total ? T(0) : T(total - step) / T(warmdown);
    return progress + (T(1) - progress) * final_fraction;
}

/// \brief The sum of the squares of a row: one shard's part of the global norm; shards' parts add.
/// \param r The row.
/// \return `sum r_i²`.
template <row R>
constexpr row_value_t<R> squared_norm(const R& r)
{
    row_value_t<R> s(0);
    for (const auto x : as_row(r))
        s += x * x;
    return s;
}

/// \brief The factor that clips gradients to a global norm: `min(1, max_norm / (norm + 1e-6))`, as PyTorch's `clip_grad_norm_`.
/// \param norm The global norm, `sqrt` of the summed `squared_norm`s.
/// \param max_norm The largest norm allowed: 1.0 in GPT-2 training.
/// \return The factor to multiply every gradient by.
template <math::math_number T>
constexpr T clip_coefficient(T norm, std::type_identity_t<T> max_norm)
{
    const T c = max_norm / (norm + T(1e-6));
    return c < T(1) ? c : T(1);
}

/// \brief The coefficients of Muon's quintic Newton-Schulz iteration and its step count.
/// \ingroup core
/// \tparam T The number type.
template <math::math_number T>
struct muon_newton_schulz
{
    static constexpr T a = T(3.4445);    ///< The linear coefficient.
    static constexpr T b = T(-4.7750);   ///< The cubic coefficient.
    static constexpr T c = T(2.0315);    ///< The quintic coefficient.
    static constexpr int steps = 5;      ///< The iterations Muon runs.
};

/// \brief One Newton-Schulz step on a singular value: `a x + b x³ + c x⁵`.
///
/// Iterated `steps` times from a singular value in `(0, 1]` (the matrix is first divided by its norm), it lands
/// near 1, roughly in `[0.7, 1.2]`: Muon's approximate orthogonalisation, traded for speed against exactness.
/// \param x A singular value.
/// \return The next.
template <math::math_number T>
constexpr T newton_schulz_polynomial(T x)
{
    using k = muon_newton_schulz<T>;
    const T x2 = x * x;
    return x * (k::a + x2 * (k::b + x2 * k::c));
}

/// \brief The training FLOPs per token of a transformer: `6N + 12·L·H·Q·T`.
///
/// `6N` for the matrix multiplications forward and backward over the `N` parameters that are not embeddings, and
/// `12·L·H·Q·T` for attention over a context of `T`.
/// \param parameters The parameters, without the embedding table.
/// \param layers `L`.
/// \param heads `H`, the query heads.
/// \param head_dim `Q`.
/// \param context `T`, the sequence length.
/// \return FLOPs per token.
constexpr double transformer_flops_per_token(std::uint64_t parameters, std::uint64_t layers, std::uint64_t heads,
                                             std::uint64_t head_dim, std::uint64_t context) noexcept
{
    return 6.0 * double(parameters) + 12.0 * double(layers) * double(heads) * double(head_dim) * double(context);
}

/// \brief The model FLOPs utilisation of a run: the fraction of the hardware's peak it achieves.
/// \param flops_per_token From `transformer_flops_per_token`.
/// \param tokens_per_second The run's throughput.
/// \param peak_flops The hardware's peak in FLOP/s at the run's precision (989e12 for an H100's bfloat16, dense).
/// \return The utilisation, 0 to 1.
constexpr double model_flops_utilisation(double flops_per_token, double tokens_per_second, double peak_flops) noexcept
{
    return flops_per_token * tokens_per_second / peak_flops;
}

} // namespace snicholls::ml

#endif // snicholls_core_ml_optim_hpp
