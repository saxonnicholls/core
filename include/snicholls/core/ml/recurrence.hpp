//
//  recurrence.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::ml` recurrences over a sequence: a step and its backward pass give the sequence forward with checkpoints, the backward pass through time from them, and decoding one token at a time with a fixed-size state.
/// \ingroup core
///
/// **Overview.** Every linear-time sequence mixer (`ml/ssd.hpp`, `ml/gla.hpp`, `ml/rwkv.hpp`, `ml/delta_rule.hpp`) is a
/// recurrence: a state of a fixed size, updated by each token, from which the token's output is read. Written once
/// here, a mixer supplies only its step and the step's backward pass, and gets
///
/// - `recurrence_forward(r, n, sequences, chunk, states, checkpoints)`: every token of `sequences` sequences of `n`
///   tokens laid end to end, each from its own initial state (written back as its final state), saving the state at
///   the start of every chunk of `chunk` tokens;
/// - `recurrence_backward(r, n, sequences, chunk, checkpoints, dstates, work)`: backpropagation through time, one chunk
///   at a time from the last: the chunk's states are recomputed from its checkpoint (so memory is `n / chunk + chunk`
///   states, not `n`), then the step's backward pass runs from the chunk's last token to its first;
/// - decoding: `r.step` on one token with the sequence's state, which is the forward pass's own step, so decoding gives
///   the forward pass's bits at every position.
///
/// A recurrence (`concept recurrence`) is a small object over the caller's views with `state_size` and `work_size`
/// (`static constexpr`), `step(row, state)` (the output of global row `row`, the state updated in place), `advance(row,
/// state)` (the update alone, for recomputation) and `step_backward(row, previous, dstate, work)` (with `previous` the
/// state before the row and `dstate` the gradient in the state after it, replaced by the gradient in the state before
/// it; the gradients in the row's inputs written, those in the mixer's own parameters added into).
///
/// \snippet ml/recurrence_demo.cpp recurrence
///
/// **Relationships.** The mixers above; their chunked forms write the same checkpoints, so either forward pass feeds
/// `recurrence_backward`. Group: \ref core.
///
/// **Implementations.** One driver; the mixers are the implementations of the concept.
///
/// **Guarantees.** `constexpr`, no allocation (the caller owns the states, checkpoints and `work`), the order of every
/// operation fixed, so the bits do not depend on scheduling. Sequences are independent and could run in parallel; tokens
/// are sequential (the chunked forms are the parallel ones). The checkpoints of a forward pass with any `chunk` are its
/// states at those tokens, so the backward pass's result does not depend on `chunk` (tested to the bit).
///
/// **Errors.** Preconditions, asserted: the states `sequences · state_size`, the checkpoints
/// `sequences · recurrence_checkpoints(n, chunk) · state_size` (or empty in the forward pass: none saved), `work`
/// `recurrence_backward_work<R>(chunk)`.
///
/// **Performance.** The driver adds a copy of the state per chunk forward and per token backward to the steps' own work;
/// the mixers' benchmarks measure it (`benchmarks/ml/ssd_benchmark.cpp`).
///
/// **Tests and demos.** `tests/ml/recurrence_test.cpp`, `demos/ml/recurrence_demo.cpp`.
///
/// **Dependencies.** `<algorithm>`, `<cassert>`, `<concepts>`, `<cstddef>`, `<span>`, `<type_traits>`.
///
/// **Known issues.** Backpropagation through time is sequential in the tokens: a device runs the chunked backward pass
/// of each mixer instead and checks it against this one.
///
/// **Status.** Experimental, new in 0.17.0.

#ifndef snicholls_core_ml_recurrence_hpp
#define snicholls_core_ml_recurrence_hpp

#include <algorithm>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <span>
#include <type_traits>

namespace snicholls::ml
{

/// \brief A recurrence over the rows of a sequence: a fixed-size state, a step, and the step's backward pass.
/// \ingroup core
/// \tparam R The recurrence: with `value_type`, `static constexpr std::size_t state_size` and `work_size`, and the
/// members `step`, `advance` and `step_backward` described in the file's overview.
template <typename R>
concept recurrence = requires(const R& r, std::size_t row, std::span<typename R::value_type> s, std::span<const typename R::value_type> c) {
    { R::state_size } -> std::convertible_to<std::size_t>;
    { R::work_size } -> std::convertible_to<std::size_t>;
    r.step(row, s);
    r.advance(row, s);
    r.step_backward(row, c, s, s);
};

/// \brief The checkpoints a sequence of `n` tokens keeps with chunks of `chunk`: one at the start of each chunk.
/// \param n The tokens. \param chunk The chunk, at least 1.
/// \return `ceil(n / chunk)`.
[[nodiscard]] constexpr std::size_t recurrence_checkpoints(std::size_t n, std::size_t chunk) noexcept
{
    return chunk == 0 ? 0 : (n + chunk - 1) / chunk;
}

/// \brief The working space `recurrence_backward` needs: a chunk's states and the step's own.
/// \tparam R The recurrence. \param chunk The chunk.
/// \return `chunk · state_size + work_size`.
template <recurrence R>
[[nodiscard]] constexpr std::size_t recurrence_backward_work(std::size_t chunk) noexcept
{
    return chunk * R::state_size + R::work_size;
}

/// \brief Runs a recurrence over `sequences` sequences of `n` rows laid end to end, writing every row's output.
/// \param r The recurrence.
/// \param n The rows of each sequence.
/// \param sequences The sequences.
/// \param chunk The tokens between checkpoints, at least 1.
/// \param states `sequences · state_size`: each sequence's initial state, replaced by its final state.
/// \param checkpoints `sequences · recurrence_checkpoints(n, chunk) · state_size`: receives the state at the start of
/// each chunk, sequence by sequence; or empty to keep none (decoding, inference).
template <recurrence R>
constexpr void recurrence_forward(const R& r, std::size_t n, std::size_t sequences, std::size_t chunk,
                                  std::span<typename R::value_type> states, std::span<typename R::value_type> checkpoints = {})
{
    constexpr std::size_t S = R::state_size;
    const std::size_t m = recurrence_checkpoints(n, chunk);
    assert(chunk > 0 && states.size() == sequences * S && (checkpoints.empty() || checkpoints.size() == sequences * m * S));
    for (std::size_t s = 0; s < sequences; ++s) {
        const auto state = states.subspan(s * S, S);
        for (std::size_t i = 0; i < n; ++i) {
            if (!checkpoints.empty() && i % chunk == 0)
                std::copy(state.begin(), state.end(), checkpoints.begin() + static_cast<std::ptrdiff_t>((s * m + i / chunk) * S));
            r.step(s * n + i, state);
        }
    }
}

/// \brief Backpropagation through time from the checkpoints of a forward pass: each chunk's states recomputed from its
/// checkpoint, then the step's backward pass from its last row to its first.
/// \param r The recurrence, over the forward pass's inputs and the gradients to fill.
/// \param n The rows of each sequence. \param sequences The sequences. \param chunk The forward pass's chunk.
/// \param checkpoints What the forward pass saved.
/// \param dstates `sequences · state_size`: the gradient in each final state (zeros if the loss does not read it),
/// replaced by the gradient in each initial state.
/// \param work `recurrence_backward_work<R>(chunk)` values.
template <recurrence R>
constexpr void recurrence_backward(const R& r, std::size_t n, std::size_t sequences, std::size_t chunk,
                                   std::span<const typename R::value_type> checkpoints, std::span<typename R::value_type> dstates,
                                   std::span<typename R::value_type> work)
{
    constexpr std::size_t S = R::state_size;
    const std::size_t m = recurrence_checkpoints(n, chunk);
    assert(chunk > 0 && checkpoints.size() == sequences * m * S && dstates.size() == sequences * S
           && work.size() == recurrence_backward_work<R>(chunk));
    const auto saved = work.first(chunk * S);
    const auto scratch = work.subspan(chunk * S);
    for (std::size_t s = 0; s < sequences; ++s) {
        const auto dstate = dstates.subspan(s * S, S);
        for (std::size_t c = m; c-- > 0;) {
            const std::size_t first = c * chunk, len = std::min(chunk, n - first);
            const auto from = checkpoints.subspan((s * m + c) * S, S);
            std::copy(from.begin(), from.end(), saved.begin());
            for (std::size_t j = 1; j < len; ++j) {
                const auto prev = saved.subspan((j - 1) * S, S), next = saved.subspan(j * S, S);
                std::copy(prev.begin(), prev.end(), next.begin());
                r.advance(s * n + first + j - 1, next);
            }
            for (std::size_t j = len; j-- > 0;)
                r.step_backward(s * n + first + j, std::span<const typename R::value_type>(saved.subspan(j * S, S)), dstate, scratch);
        }
    }
}

} // namespace snicholls::ml

#endif // snicholls_core_ml_recurrence_hpp
