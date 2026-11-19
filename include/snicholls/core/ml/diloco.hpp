//
//  diloco.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::ml` arithmetic of data-parallel training across machines: the fixed-order sum of the workers' parts, DiLoCo's pseudo-gradient, its bfloat16 compression with error feedback, and its outer step.
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
/// **bfloat16 pseudo-gradients.** Sending the pseudo-gradient in bfloat16 halves the bytes on the wire (GPT-2
/// small's 124M parameters: 248 MB instead of 496 MB a round). Three functions fix the arithmetic, so that the
/// transport and the training library agree to the bit:
///
/// - `compress_bf16(delta, residual, out)`: each worker rounds `delta + residual` to bfloat16 (to nearest, ties to
///   even) and keeps what the rounding dropped as its residual, added to the next round's delta (error feedback:
///   Seide et al. 2014, Karimireddy et al. 2019, 1-bit Adam, Tang et al. 2021). The residual is exact, so over rounds
///   the sum of what a worker sent plus its residual is the sum of its deltas, up to the float additions
///   `delta + residual`.
/// - `ordered_sum_bf16(parts, out)`: the reducing rank decodes each worker's bfloat16 to float (exact) and sums in
///   rank order in float, as `ordered_sum`.
/// - `ordered_mean_bf16(parts, out)`: what every worker holds after the all-reduce and the mean. ts-moveables'
///   `allreduce_sum_bf16` leaves bf16(ordered_sum_bf16) on every rank, and the training library divides by the worker
///   count in float.
///
/// \snippet ml/diloco_demo.cpp bf16
///
/// **Elastic membership.** The worker count may change between rounds, a worker leaving or rejoining. `ordered_mean`
/// and `ordered_mean_bf16` divide by `parts.size()`, the count of that round's parts, so the outer step needs nothing
/// new. The outer momentum is global state, held identically by every worker, so a rejoining worker receives it with
/// the global weights by broadcast. Its residual restarts at zero: its last round's pseudo-gradient was discarded with
/// the failed round, and a residual from an older checkpoint would re-send a remainder that later rounds already
/// carried; zero loses at most one bfloat16 rounding of one delta. Its inner optimiser state is the training
/// library's choice. Restarting it at zero is the better default: a stale first moment pushes along an old direction,
/// while zero moments, bias-corrected at a late step, begin with short steps.
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
/// additions). `compress_bf16`, `ordered_sum_bf16` and `ordered_mean_bf16` are `noexcept`, and are their own
/// specification in float on every machine (core's integer bfloat16 rounding, no FMA possible). `noexcept` for the standard floating-point types; a number type of the caller's own throws what its
/// arithmetic throws. `diloco_config` is trivially copyable. `out` may be `parts[0]` itself (the sum accumulates
/// into the first part) but no other part; `pseudo_gradient`'s `out` may be `global` or `local`.
///
/// **Errors.** Preconditions, asserted: at least one part; every span the same size. No errors otherwise: a NaN or an
/// infinity in a pseudo-gradient travels on as a NaN or infinity (its residual 0), so the run sees it.
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
/// negligible next to the local steps. The bfloat16 pair against the same loops by hand, same machine and date, 1M
/// elements, the mean of three runs: `compress_bf16` 1.11 billion elements per second (by hand 1.04), `ordered_sum_bf16`
/// of four parts 0.90 billion (by hand 0.96): level within the noise, and the sum of bfloat16 parts about twice the
/// float sum's rate, having half the bytes to read.
///
/// **Tests and demos.** `tests/ml/diloco_test.cpp`, `demos/ml/diloco_demo.cpp`.
///
/// **Dependencies.** `<bit>`, `<cassert>`, `<cstddef>`, `<cstdint>`, `<span>`, `<type_traits>`,
/// `snicholls/core/math/constexpr_math.hpp` (for the `math_number` concept), `snicholls/core/ml/minifloat.hpp` (bfloat16).
///
/// **Known issues.** The outer momentum is kept in `T`, as the weights are; the paper keeps it in the weights'
/// precision too. `diloco_config` holds `double`s, converted to `T` once per step. With bfloat16 pseudo-gradients the
/// sum's own rounding to bfloat16 (once a round, relative 2^-9 of the mean) is not covered by the workers' error
/// feedback: that would need a second residual at the reducing rank, as 1-bit Adam keeps, and is not done. The
/// residual is the worker's own state; it is not averaged.
///
/// **Status.** Experimental, new in 0.15.0; the bfloat16 functions new in 0.16.0.

#ifndef snicholls_core_ml_diloco_hpp
#define snicholls_core_ml_diloco_hpp

#include <snicholls/core/math/constexpr_math.hpp>
#include <snicholls/core/ml/minifloat.hpp>

#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
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
    const auto first = parts.front();
    for (std::size_t i = 0; i < out.size(); ++i)
        out[i] = first[i];
    for (const auto part : parts.subspan(1))   // no index into parts: GCC 15's -Warray-bounds misreads one
        for (std::size_t i = 0; i < out.size(); ++i)
            out[i] = out[i] + part[i];
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

namespace detail::diloco_
{
// Finite by the exponent bits: constexpr, where std::isfinite is not in C++20
constexpr bool finite(float x) noexcept
{
    return (std::bit_cast<std::uint32_t>(x) & 0x7f800000u) != 0x7f800000u;
}
} // namespace detail::diloco_

/// \brief A pseudo-gradient rounded to bfloat16 for the wire, with error feedback: what the rounding drops is kept and
/// added to the next round's pseudo-gradient, so nothing is lost over rounds.
///
/// Per element: `v = delta[i] + residual[i]` (one float addition); `out[i]` is the bits of `v` rounded to the
/// nearest bfloat16, ties to even (`to_bfloat16`: a NaN stays a quiet NaN with its sign, a value beyond bfloat16's
/// largest finite becomes infinity); `residual[i] = v - to_float(out[i])`. That subtraction is exact: the bfloat16
/// is `v` rounded to 8 significant bits, within a factor of two of `v` (Sterbenz's lemma; subnormals too, both being
/// multiples of the smallest float subnormal), so `v == to_float(out[i]) + residual[i]` exactly. A non-finite `v`,
/// or one that overflows to infinity, carries no residual (0): the NaN or infinity goes on the wire, so the run sees
/// it. Error feedback after Seide et al. (2014, 1-bit SGD), Karimireddy et al. (2019, "Error Feedback Fixes SignSGD"),
/// Tang et al. (2021, 1-bit Adam); low-precision pseudo-gradients in DiLoCo and streaming DiLoCo (Douillard et al.,
/// 2023, 2025).
/// \param delta This round's pseudo-gradient (`pseudo_gradient`).
/// \param residual The worker's residual, zero before the first round; updated in place. Must not alias `delta`.
/// \param out The bfloat16 bits to send.
/// \pre The three spans have one size.
constexpr void compress_bf16(std::span<const float> delta, std::span<float> residual, std::span<std::uint16_t> out) noexcept
{
    assert(residual.size() == delta.size() && out.size() == delta.size());
    for (std::size_t i = 0; i < delta.size(); ++i) {
        const float v = delta[i] + residual[i];
        const bfloat16 b = to_bfloat16(v);
        out[i] = b.bits;
        const float back = to_float(b);
        residual[i] = detail::diloco_::finite(v) && detail::diloco_::finite(back) ? v - back : 0.0f;
    }
}

/// \brief The workers' bfloat16 parts decoded to float and summed left to right in rank order, in float: the
/// reducing rank's arithmetic in a bfloat16 all-reduce.
///
/// `out[i] = ((float(parts[0][i]) + float(parts[1][i])) + ...)`, each decoding exact; `ordered_sum` over the
/// decoded parts to the bit.
/// \param parts One span of bfloat16 bits per worker, in rank order, each the size of `out`.
/// \param out The float sum.
/// \pre `!parts.empty()` and every part has `out.size()` elements.
constexpr void ordered_sum_bf16(std::span<const std::span<const std::uint16_t>> parts, std::span<float> out) noexcept
{
    assert(!parts.empty());
    for ([[maybe_unused]] const auto& part : parts)
        assert(part.size() == out.size());
    const auto first = parts.front();
    for (std::size_t i = 0; i < out.size(); ++i)
        out[i] = to_float(bfloat16::from_bits(first[i]));
    for (const auto part : parts.subspan(1))   // no index into parts: GCC 15's -Warray-bounds misreads one
        for (std::size_t i = 0; i < out.size(); ++i)
            out[i] = out[i] + to_float(bfloat16::from_bits(part[i]));
}

/// \brief What a bfloat16 all-reduce followed by the mean gives every worker: the float sum rounded once to
/// bfloat16, decoded, and divided by the worker count.
///
/// `out[i] = to_float(to_bfloat16(s)) / float(parts.size())`, with `s` the element of `ordered_sum_bf16`. The
/// contract: ts-moveables' `allreduce_sum_bf16` leaves bf16(ordered_sum_bf16) on every rank, and the training
/// library divides by the worker count in float. The sum is rounded (to nearest, ties to even) because the
/// all-gather sends bfloat16; the division is a division, not a multiplication by the reciprocal.
/// \param parts One span of bfloat16 bits per worker, in rank order, each the size of `out`.
/// \param out The mean.
/// \pre `!parts.empty()` and every part has `out.size()` elements.
constexpr void ordered_mean_bf16(std::span<const std::span<const std::uint16_t>> parts, std::span<float> out) noexcept
{
    ordered_sum_bf16(parts, out);
    const float count = float(parts.size());
    for (std::size_t i = 0; i < out.size(); ++i)
        out[i] = to_float(to_bfloat16(out[i])) / count;
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
