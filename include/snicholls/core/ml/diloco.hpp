//
//  diloco.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::ml` arithmetic of data-parallel training across machines: the fixed-order sum of the workers' parts, DiLoCo's pseudo-gradient and its outer step.
/// \ingroup core
///
/// **Overview.** Data-parallel training runs one model on many workers, each on its own data, and merges what they
/// learn. Two ways of merging share this header's arithmetic:
///
/// - **Every step** (TRAINING.md T5): each worker's gradients are summed across workers (an all-reduce) before the
///   optimiser steps. `ordered_sum(parts, out)` is that sum's reference.
/// - **Rarely: DiLoCo** (Douillard et al., 2023, "DiLoCo: Distributed Low-Communication Training of Language
///   Models"). Every worker starts a round from the same global weights, takes `H` local steps with its own inner
///   optimiser (AdamW in the paper) on its own data, and only then sends the change of its weights, the
///   *pseudo-gradient* `global - local` (`pseudo_gradient`). The pseudo-gradients are averaged (`ordered_mean`, or
///   `ordered_sum` and a division by the worker count) and an outer optimiser steps the global weights with the mean
///   as its gradient (`diloco_outer_step`). The paper's `H` is hundreds of steps (500 in its main runs), so the
///   network carries one exchange of the weights per `H` steps instead of one per step: for clusters, or machines
///   across sites, whose network is too slow for an all-reduce every step. Its outer optimiser is SGD with Nesterov
///   momentum 0.9 and learning rate 0.7, the defaults of `diloco_config`; with momentum 0, learning rate 1 and no
///   Nesterov the outer step replaces the global weights by the workers' mean (federated averaging).
///
/// One round, as the demo runs it:
///
/// \snippet ml/diloco_demo.cpp round
///
/// **Why the order of the sum is fixed.** Floating-point addition is not associative: `(a + b) + c` and
/// `a + (b + c)` can differ in the last bit, and over thousands of steps the differences grow into different models.
/// An all-reduce that adds parts as they arrive, or in the pairing its ring or tree topology happens to give, makes a
/// run depend on timing and on the cluster's shape. `ordered_sum` defines the one order, `parts[0] + parts[1]`, then
/// `+ parts[2]`, and so on, left to right in rank order, in `T`'s own arithmetic; any transport that gathers the parts
/// and sums them in that order gives the same bits, on any number of machines, whatever arrives first. The sum is
/// sequential, not pairwise: pairwise summation is more accurate for many parts, but a data-parallel sum has few parts
/// (the workers) and the order is the contract, reproduced independently by the all-reduce of a library that pins
/// core.
///
/// **Relationships.** `ml/optim.hpp` (`adamw_step`, the inner optimiser; `squared_norm` and `clip_coefficient`, which
/// merge across shards as these sums do), `ml/muon.hpp` (the same momentum convention), `stats/random.hpp`
/// (`wyrand::stream`, one data stream per worker). The transport, the all-reduce over TCP between machines, belongs to
/// ts-moveables; the device kernels that run the same arithmetic over device memory belong to snicholls-gpu, checked
/// against these functions. Group: \ref core.
///
/// **Implementations.** One each: elementwise loops over the caller's spans. Each multiplication and addition of the
/// outer step is its own statement, so no compiler fuses them into an FMA and the bits are the same on every machine
/// (CLAUDE.md, "Machine learning").
///
/// **Guarantees.** `constexpr`, no allocation, linear in the elements (`ordered_sum`: parts times elements
/// additions). `noexcept` for the standard floating-point types; a number type of the caller's own throws what its
/// arithmetic throws. `diloco_config` is trivially copyable. `out` may be `parts[0]` itself (the sum accumulates
/// into the first part) but no other part; `pseudo_gradient`'s `out` may be `global` or `local`.
///
/// **Errors.** Preconditions, asserted: at least one part; every span the same size.
///
/// **Exactness.** With one worker and `{.outer_lr = 1, .momentum = 0, .nesterov = false}` the outer step computes
/// `global - (global - local)`. That is `local` to the bit whenever `global - local` is exact, which Sterbenz's
/// lemma guarantees when `local` and `global` are within a factor of two of each other (the usual case: a round moves
/// a weight a little); otherwise it is within one rounding of `global - local` of `local` (`global = 1`,
/// `local = 1e-8` in float gives 0, since `1 - 1e-8` rounds to 1). Workers with identical pseudo-gradients `d` give a
/// mean of `d` to the bit with one, two or four workers (for four, `((d + d) + d) + d` is exactly `4d`, checked for
/// every finite float), but not with three (`(d + d) + d` rounds).
///
/// **Example.** \snippet ml/diloco_demo.cpp ordered_sum
///
/// **Performance.** `ordered_sum` (four parts) and `diloco_outer_step` against the same loops by hand
/// (`benchmarks/ml/diloco_benchmark.cpp`), 1M float elements: level, within the run-to-run noise of 10% either way;
/// about 500 million elements per second for the sum of four parts and 1.1 billion for the outer step, memory bound
/// (Xeon W-3245, Release, Apple clang 17, 2026-11-19, one thread, a loaded machine). The outer step runs once a round,
/// negligible next to the local steps.
///
/// **Tests and demos.** `tests/ml/diloco_test.cpp`, `demos/ml/diloco_demo.cpp`.
///
/// **Dependencies.** `<cassert>`, `<cstddef>`, `<span>`, `<type_traits>`, `snicholls/core/math/constexpr_math.hpp`
/// (for the `math_number` concept).
///
/// **Known issues.** The outer momentum is kept in `T`, as the weights are; the paper keeps it in the weights'
/// precision too. `diloco_config` holds `double`s, converted to `T` once per step.
///
/// **Status.** Experimental, new in 0.15.0.

#ifndef snicholls_core_ml_diloco_hpp
#define snicholls_core_ml_diloco_hpp

#include <snicholls/core/math/constexpr_math.hpp>

#include <cassert>
#include <cstddef>
#include <span>
#include <type_traits>

namespace snicholls::ml
{

/// \brief The elementwise sum of the parts, left to right in the order given: the reference for any all-reduce.
///
/// `out[i] = (((parts[0][i] + parts[1][i]) + parts[2][i]) + ...)`, in `T`'s arithmetic. A transport that sums the
/// workers' parts in rank order gives these bits whatever their timing or topology (see the file's overview).
/// \tparam T The number type; give it explicitly when the arguments are containers (`ordered_sum<float>(...)`).
/// \param parts One span per worker, in rank order, each the size of `out`.
/// \param out The sum; may be `parts[0]` itself, no other part.
/// \pre `!parts.empty()` and every part has `out.size()` elements.
template <math::math_number T>
constexpr void ordered_sum(std::span<const std::span<const T>> parts, std::span<T> out) noexcept(std::is_floating_point_v<T>)
{
    assert(!parts.empty());
    for ([[maybe_unused]] const auto& part : parts)
        assert(part.size() == out.size());
    const auto first = parts[0];
    for (std::size_t i = 0; i < out.size(); ++i)
        out[i] = first[i];
    for (std::size_t k = 1; k < parts.size(); ++k) {
        const auto part = parts[k];
        for (std::size_t i = 0; i < out.size(); ++i)
            out[i] = out[i] + part[i];
    }
}

/// \brief The elementwise mean of the parts: `ordered_sum`, then each element divided by the number of parts.
///
/// Divided by `T(parts.size())`, not multiplied by its reciprocal, so a transport that reproduces it divides too.
/// It is the `mean_delta` of `diloco_outer_step`.
/// \tparam T The number type.
/// \param parts One span per worker, in rank order, each the size of `out`.
/// \param out The mean; may be `parts[0]` itself, no other part.
/// \pre `!parts.empty()` and every part has `out.size()` elements.
template <math::math_number T>
constexpr void ordered_mean(std::span<const std::span<const T>> parts, std::span<T> out) noexcept(std::is_floating_point_v<T>)
{
    ordered_sum(parts, out);
    const T count = T(parts.size());
    for (std::size_t i = 0; i < out.size(); ++i)
        out[i] = out[i] / count;
}

/// \brief A worker's pseudo-gradient after its local steps: `out = global - local` (DiLoCo, Douillard et al. 2023).
///
/// The weights moved by `-out` in the round, so `out` points, as a gradient does, uphill.
/// \tparam T The number type.
/// \param global The global weights the round started from.
/// \param local The worker's weights after its local steps.
/// \param out The pseudo-gradient; may be `global` or `local`.
/// \pre The three spans have one size.
template <math::math_number T>
constexpr void pseudo_gradient(std::span<const T> global, std::span<const T> local, std::span<T> out) noexcept(std::is_floating_point_v<T>)
{
    assert(global.size() == out.size() && local.size() == out.size());
    for (std::size_t i = 0; i < out.size(); ++i)
        out[i] = global[i] - local[i];
}

/// \brief The settings of DiLoCo's outer optimiser, SGD with momentum: the paper's by default.
/// \ingroup core
struct diloco_config
{
    double outer_lr = 0.7;   ///< The outer learning rate: 0.7 in the paper; 1 with no momentum is plain averaging.
    double momentum = 0.9;   ///< The outer momentum `μ`: 0.9 in the paper.
    bool nesterov = true;    ///< Nesterov momentum, as the paper uses.
};

/// \brief DiLoCo's outer step: SGD with (Nesterov) momentum on the global weights, with the mean pseudo-gradient as the gradient.
///
/// For each element, with `g` the mean pseudo-gradient (already divided by the worker count) and `μ` the momentum:
/// `momentum = μ · momentum + g`; `step = nesterov ? g + μ · momentum : momentum`; `global -= outer_lr · step`, as
/// PyTorch's `SGD(nesterov=True)` with no dampening. Each product and sum is its own statement (no FMA), in `T`.
/// \tparam T The number type.
/// \param global The global weights, updated in place.
/// \param mean_delta The mean of the workers' pseudo-gradients (`ordered_mean`).
/// \param momentum The outer momentum, zero before the first round, updated in place.
/// \param c The outer settings.
/// \pre The three spans have one size.
template <math::math_number T>
constexpr void diloco_outer_step(std::span<T> global, std::span<const T> mean_delta, std::span<T> momentum,
                                 const diloco_config& c) noexcept(std::is_floating_point_v<T>)
{
    assert(mean_delta.size() == global.size() && momentum.size() == global.size());
    const T mu = T(c.momentum);
    const T lr = T(c.outer_lr);
    for (std::size_t i = 0; i < global.size(); ++i) {
        const T g = mean_delta[i];
        const T decayed = mu * momentum[i];
        momentum[i] = decayed + g;
        T step = momentum[i];
        if (c.nesterov) {
            const T ahead = mu * momentum[i];
            step = g + ahead;
        }
        const T change = lr * step;
        global[i] = global[i] - change;
    }
}

} // namespace snicholls::ml

#endif // snicholls_core_ml_diloco_hpp
