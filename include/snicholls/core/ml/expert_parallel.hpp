//
//  expert_parallel.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::ml` expert parallelism, the arithmetic: which rank holds which experts, the dispatch and combine layouts, and a reference all-to-all over in-process ranks that fixes the order of every received row.
/// \ingroup core
///
/// **Overview.** With expert parallelism (GShard, Lepikhin et al., 2020; DeepSeek-V3) each of `P` ranks holds `E / P`
/// of a mixture of experts' routed experts, and each rank's tokens travel to the ranks of the experts they chose and
/// back. This header is that arithmetic, and nothing of the transport (sockets, NCCL, a library above core):
///
/// - Placement: rank `r` holds the contiguous experts `[r · E/P, (r + 1) · E/P)` (`first_expert`, `expert_rank`,
///   `experts_per_rank`). Since a rank's permuted rows (`moe_permutation`) are in expert order, the rows for one
///   destination are one contiguous run: `moe_send_offsets`.
/// - The exchange: the ranks first exchange their counts (`E` numbers each), then the rows. `moe_all_to_all` is the
///   reference: destination `r` receives, in source order `s = 0, 1, ...`, source `s`'s run for `r`. So a received
///   buffer is ordered by `(source, expert, row, slot)`.
/// - The receiving rank's layout: its grouped products want `(expert, source, row, slot)`, which is the single-process
///   permuted order restricted to its experts, since global rows are numbered source by source. `moe_receive_layout`
///   gives that order and the local experts' offsets; `moe_permute_rows` applies it, `moe_unpermute_rows` undoes it.
/// - The way back: each rank sends every source its rows back, in the order received, with `moe_receive_offsets` as
///   the send offsets; the same `moe_all_to_all` delivers them in rank order, which is the source's own permuted order.
///
/// One forward step on rank `r`, the backward passes the same layouts in reverse:
///
/// \snippet ml/expert_parallel_demo.cpp dispatch
///
/// **Exactness.** An expert multiplies the same rows in the same order as in one process holding every rank's tokens,
/// so the outputs, the input gradients and each expert's weight gradients are the single-process layer's to the bit
/// (tested with 1, 2 and 4 ranks on core's reference backend). The router's and the shared experts' weight gradients
/// are each rank's own part (data parallel), summed across ranks in rank order by `ordered_sum` (`ml/diloco.hpp`);
/// they agree with one process within rounding, since one process sums all the rows in one product. The balance loss
/// and z-loss are exact when each rank passes the whole batch's denominator (`moe_balance_loss`, `moe_z_loss`).
///
/// The payload may travel in `bfloat16` (`to_bfloat16_rows`, `to_float_rows` in `ml/minifloat.hpp`) and be combined in
/// float; that halves the bytes and is no longer bit-for-bit the float layer.
///
/// **Relationships.** `ml/moe.hpp` (the layer's steps), `ml/diloco.hpp` (`ordered_sum`), `ml/minifloat.hpp`; the plan
/// is TRAINING.md T10. A transport of a library above core moves the bytes; a device library overloads these ops for
/// its backend. Group: \ref core.
///
/// **Implementations.** One, for any backend, on the host: copies in a fixed order.
///
/// **Guarantees.** `constexpr`, no allocation; the placement functions `noexcept`. Every op is linear in what it copies.
///
/// **Errors.** Preconditions, asserted: `P` divides `E`, every span has its documented size, the received buffers are
/// large enough.
///
/// **Performance.** Not benchmarked: copies, each row once, dominated in practice by the transport.
///
/// **Tests and demos.** `tests/ml/expert_parallel_test.cpp`, `demos/ml/expert_parallel_demo.cpp`.
///
/// **Dependencies.** `<cassert>`, `<cstddef>`, `<cstdint>`, `<span>`, `<type_traits>`, `ml/transformer_config.hpp`.
///
/// **Known issues.** Only the contiguous placement; an uneven split of experts over ranks is not supported.
///
/// **Status.** Experimental, new in 0.16.0.

#ifndef snicholls_core_ml_expert_parallel_hpp
#define snicholls_core_ml_expert_parallel_hpp

#include <snicholls/core/ml/transformer_config.hpp>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>

namespace snicholls::ml
{

/// \brief The experts each rank holds.
/// \param experts The routed experts, `E`.
/// \param ranks The ranks, `P`, dividing `E`.
/// \return `E / P`.
[[nodiscard]] constexpr std::size_t experts_per_rank(std::size_t experts, std::size_t ranks) noexcept
{
    return experts / ranks;
}

/// \brief The first expert a rank holds.
/// \param rank The rank, below `P`.
/// \param experts The routed experts, `E`.
/// \param ranks The ranks, `P`, dividing `E`.
/// \return `rank · E / P`.
[[nodiscard]] constexpr std::size_t first_expert(std::size_t rank, std::size_t experts, std::size_t ranks) noexcept
{
    return rank * experts_per_rank(experts, ranks);
}

/// \brief The rank holding an expert.
/// \param expert The expert, below `E`.
/// \param experts The routed experts, `E`.
/// \param ranks The ranks, `P`, dividing `E`.
/// \return `expert / (E / P)`.
[[nodiscard]] constexpr std::size_t expert_rank(std::size_t expert, std::size_t experts, std::size_t ranks) noexcept
{
    return expert / experts_per_rank(experts, ranks);
}

/// \brief Where each destination's rows start in a rank's permuted rows: `send_offsets[r] = offsets[r · E / P]`.
/// \tparam C A mixture-of-experts configuration.
/// \param offsets The rank's expert offsets (`moe_offsets`), `E + 1`.
/// \param ranks The ranks, `P`, dividing `E`.
/// \param send_offsets Receives `P + 1` offsets, in rows.
template <transformer_config C, typename B>
    requires(C.experts > 0)
constexpr void moe_send_offsets(const B&, std::span<const std::uint32_t> offsets, std::size_t ranks, std::span<std::uint32_t> send_offsets)
{
    assert(ranks > 0 && C.experts % ranks == 0 && offsets.size() == C.experts + 1 && send_offsets.size() == ranks + 1);
    for (std::size_t r = 0; r <= ranks; ++r)
        send_offsets[r] = offsets[first_expert(r, C.experts, ranks)];
}

/// \brief Where each source's rows start in what a rank receives: `receive_offsets[s + 1] = receive_offsets[s] +` the
/// rows source `s` routed to this rank's experts. They are also the send offsets of the way back.
/// \tparam C A mixture-of-experts configuration.
/// \param counts Every source's counts (`moe_count_rows`), `P` spans of `E`, in rank order.
/// \param rank This rank.
/// \param receive_offsets Receives `P + 1` offsets, in rows.
template <transformer_config C, typename B>
    requires(C.experts > 0)
constexpr void moe_receive_offsets(const B&, std::span<const std::span<const std::uint32_t>> counts, std::size_t rank,
                                   std::span<std::uint32_t> receive_offsets)
{
    const std::size_t ranks = counts.size(), first = first_expert(rank, C.experts, ranks), local = experts_per_rank(C.experts, ranks);
    assert(ranks > 0 && C.experts % ranks == 0 && rank < ranks && receive_offsets.size() == ranks + 1);
    receive_offsets[0] = 0;
    for (std::size_t s = 0; s < ranks; ++s) {
        assert(counts[s].size() == C.experts);
        std::uint32_t n = 0;
        for (std::size_t j = 0; j < local; ++j)
            n += counts[s][first + j];
        receive_offsets[s + 1] = receive_offsets[s] + n;
    }
}

/// \brief The order a rank's grouped products read its received rows in, `(expert, source, row, slot)`, and the local
/// experts' offsets: `order[p]` is the received row at expert-ordered position `p`.
/// \tparam C A mixture-of-experts configuration.
/// \param counts Every source's counts, `P` spans of `E`, in rank order.
/// \param rank This rank.
/// \param expert_offsets Receives the local experts' offsets, `E / P + 1`.
/// \param order Receives one index per received row.
template <transformer_config C, typename B>
    requires(C.experts > 0)
constexpr void moe_receive_layout(const B&, std::span<const std::span<const std::uint32_t>> counts, std::size_t rank,
                                  std::span<std::uint32_t> expert_offsets, std::span<std::uint32_t> order)
{
    const std::size_t ranks = counts.size(), first = first_expert(rank, C.experts, ranks), local = experts_per_rank(C.experts, ranks);
    assert(ranks > 0 && C.experts % ranks == 0 && rank < ranks && expert_offsets.size() == local + 1);
    std::uint32_t p = 0;
    for (std::size_t j = 0; j < local; ++j) {
        expert_offsets[j] = p;
        std::uint32_t source_start = 0;   // where source s's run starts in the received rows
        for (std::size_t s = 0; s < ranks; ++s) {
            std::uint32_t within = 0;     // where expert first + j starts within source s's run
            for (std::size_t i = 0; i < j; ++i)
                within += counts[s][first + i];
            for (std::uint32_t i = 0; i < counts[s][first + j]; ++i)
                order[p++] = source_start + within + i;
            for (std::size_t i = 0; i < local; ++i)
                source_start += counts[s][first + i];
        }
    }
    expert_offsets[local] = p;
    assert(order.size() == p);
}

/// \brief The reference all-to-all over in-process ranks: destination `r` receives, in source order, each source's
/// rows `[send_offsets[s][r], send_offsets[s][r + 1])`.
///
/// It defines the order of every received row, which a transport must reproduce for a distributed run to match one
/// process. Any trivially copyable payload: float rows, or `bfloat16` rows.
/// \tparam T The payload's element type.
/// \param send Each source's rows, `P` spans of `width`-element rows.
/// \param send_offsets Each source's offsets by destination, `P` spans of `P + 1`, in rows.
/// \param width The elements of a row.
/// \param receive Each destination's buffer, `P` spans, each exactly the rows it receives.
template <typename B, typename T>
    requires std::is_trivially_copyable_v<T>
constexpr void moe_all_to_all(const B&, std::span<const std::span<const T>> send, std::span<const std::span<const std::uint32_t>> send_offsets,
                              std::size_t width, std::span<const std::span<T>> receive)
{
    const std::size_t ranks = send.size();
    assert(send_offsets.size() == ranks && receive.size() == ranks);
    for (std::size_t r = 0; r < ranks; ++r) {
        std::size_t at = 0;
        for (std::size_t s = 0; s < ranks; ++s) {
            assert(send_offsets[s].size() == ranks + 1);
            for (std::size_t i = std::size_t(send_offsets[s][r]) * width; i < std::size_t(send_offsets[s][r + 1]) * width; ++i)
                receive[r][at++] = send[s][i];
        }
        assert(at == receive[r].size());
    }
}

/// \brief Rows in a new order: `to[p] = from[order[p]]`.
/// \tparam T The element type.
/// \param from The rows, `width` elements each.
/// \param order The source row of each destination row.
/// \param to Receives `order.size()` rows.
/// \param width The elements of a row.
template <typename B, typename T>
    requires std::is_trivially_copyable_v<T>
constexpr void moe_permute_rows(const B&, std::type_identity_t<std::span<const T>> from, std::span<const std::uint32_t> order, std::span<T> to,
                                std::size_t width)
{
    assert(to.size() == order.size() * width);
    for (std::size_t p = 0; p < order.size(); ++p)
        for (std::size_t i = 0; i < width; ++i)
            to[p * width + i] = from[std::size_t(order[p]) * width + i];
}

/// \brief The inverse of `moe_permute_rows`: `to[order[p]] = from[p]`.
/// \tparam T The element type.
/// \param from The rows in the permuted order.
/// \param order The order `moe_permute_rows` used.
/// \param to Receives `order.size()` rows in the original order.
/// \param width The elements of a row.
template <typename B, typename T>
    requires std::is_trivially_copyable_v<T>
constexpr void moe_unpermute_rows(const B&, std::type_identity_t<std::span<const T>> from, std::span<const std::uint32_t> order, std::span<T> to,
                                  std::size_t width)
{
    assert(from.size() == order.size() * width && to.size() == from.size());
    for (std::size_t p = 0; p < order.size(); ++p)
        for (std::size_t i = 0; i < width; ++i)
            to[std::size_t(order[p]) * width + i] = from[p * width + i];
}

} // namespace snicholls::ml

#endif // snicholls_core_ml_expert_parallel_hpp
