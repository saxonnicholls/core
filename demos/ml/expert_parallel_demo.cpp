//
//  expert_parallel_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/ml/expert_parallel.hpp>: a mixture of experts' forward pass split across two in-process
//  ranks, each holding half the experts and half the tokens, the rows exchanged through the reference all-to-all, and
//  the result checked against one process holding everything, to the bit; then the same with a bfloat16 payload.
//  Exits non-zero if a result is not what it documents.
//

#include <snicholls/core/ml/expert_parallel.hpp>

#include <snicholls/core/linalg/reference.hpp>
#include <snicholls/core/ml/minifloat.hpp>
#include <snicholls/core/ml/moe.hpp>
#include <snicholls/core/stats/random.hpp>

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
namespace stats = snicholls::stats;
using u32 = std::uint32_t;

std::vector<float> noise(std::size_t n, std::uint64_t seed, double scale)
{
    auto g = stats::wyrand::stream(seed, 0);
    std::vector<float> v(n);
    for (auto& e : v)
        e = static_cast<float>(scale * (2.0 * stats::uniform_real(g) - 1.0));
    return v;
}

constexpr auto C = [] { auto c = ml::tiny_moe; c.shared_experts = 0; return c; }();   // 4 experts, top 2
constexpr std::size_t P = 2, d = C.d_model, K = C.top_k, f = C.expert_d_ff, local = ml::experts_per_rank(C.experts, P);

struct rank
{
    std::vector<float> xs, logits, weights, xp, recv, xe, ye, back, yp, ys, up, gate, hidden;
    std::vector<u32> experts, counts = std::vector<u32>(C.experts), offsets = std::vector<u32>(C.experts + 1), order, inverse,
                              send = std::vector<u32>(P + 1), receive = std::vector<u32>(P + 1), expert_offsets = std::vector<u32>(local + 1),
                              layout;
};

// The rows between ranks, as floats or rounded to bfloat16 on the wire
void exchange(std::array<rank, P>& ranks, std::vector<float> rank::*from, std::vector<float> rank::*to,
              std::vector<u32> rank::*offsets, bool bf16)
{
    const auto& backend = snicholls::linalg::reference;
    std::array<std::span<const u32>, P> o{};
    for (std::size_t r = 0; r < P; ++r)
        o[r] = ranks[r].*offsets;
    if (!bf16) {
        std::array<std::span<const float>, P> out{};
        std::array<std::span<float>, P> in{};
        for (std::size_t r = 0; r < P; ++r) {
            out[r] = ranks[r].*from;
            in[r] = ranks[r].*to;
        }
        ml::moe_all_to_all(backend, std::span<const std::span<const float>>(out), o, d, std::span<const std::span<float>>(in));
        return;
    }
    std::array<std::vector<ml::bfloat16>, P> wout, win;
    std::array<std::span<const ml::bfloat16>, P> out{};
    std::array<std::span<ml::bfloat16>, P> in{};
    for (std::size_t r = 0; r < P; ++r) {
        wout[r].resize((ranks[r].*from).size());
        win[r].resize((ranks[r].*to).size());
        ml::to_bfloat16_rows(backend, ranks[r].*from, wout[r]);
        out[r] = wout[r];
        in[r] = win[r];
    }
    ml::moe_all_to_all(backend, std::span<const std::span<const ml::bfloat16>>(out), o, d, std::span<const std::span<ml::bfloat16>>(in));
    for (std::size_t r = 0; r < P; ++r)
        ml::to_float_rows(backend, std::span<const ml::bfloat16>(win[r]), ranks[r].*to);
}

// The forward pass over two ranks; returns every rank's output, rank by rank
std::vector<float> two_ranks(const ml::moe_weights<float>& w, const std::vector<float>& xs, bool bf16)
{
    const auto& backend = snicholls::linalg::reference;
    const std::size_t rows = xs.size() / d / P, n = rows * K;
    std::array<rank, P> ranks;
    //! [dispatch]
    // Each rank routes and permutes its own tokens, and cuts its permuted rows by destination
    for (std::size_t r = 0; r < P; ++r) {
        auto& s = ranks[r];
        s.xs.assign(xs.begin() + std::ptrdiff_t(r * rows * d), xs.begin() + std::ptrdiff_t((r + 1) * rows * d));
        s.logits.resize(rows * C.experts), s.weights.resize(n), s.experts.resize(n), s.order.resize(n), s.inverse.resize(n);
        s.xp.resize(n * d), s.yp.resize(n * d), s.ys.resize(rows * d);
        ml::linear_sequence(backend, w.router, s.xs, s.logits);
        ml::moe_route_rows<C>(backend, s.logits, {}, s.experts, std::span<float>(s.weights));
        ml::moe_count_rows<C>(backend, s.experts, s.counts);
        ml::moe_offsets<C>(backend, s.counts, s.offsets);
        ml::moe_permutation<C>(backend, s.experts, s.offsets, s.order, s.inverse);
        ml::moe_gather_rows<C>(backend, s.xs, s.order, std::span<float>(s.xp));
        ml::moe_send_offsets<C>(backend, s.offsets, P, s.send);
    }
    // The ranks exchange their counts; each learns what it receives and the order its experts read it in
    const std::array<std::span<const u32>, P> counts{ranks[0].counts, ranks[1].counts};
    for (std::size_t r = 0; r < P; ++r) {
        auto& s = ranks[r];
        ml::moe_receive_offsets<C>(backend, counts, r, s.receive);
        const std::size_t m = s.receive[P];
        s.recv.resize(m * d), s.xe.resize(m * d), s.ye.resize(m * d), s.back.resize(m * d), s.layout.resize(m);
        s.up.resize(m * f), s.gate.resize(m * f), s.hidden.resize(m * f);
        ml::moe_receive_layout<C>(backend, counts, r, s.expert_offsets, s.layout);
    }
    // Rows out, each rank's experts, rows back, and each rank combines its own tokens
    exchange(ranks, &rank::xp, &rank::recv, &rank::send, bf16);
    for (std::size_t r = 0; r < P; ++r) {
        auto& s = ranks[r];
        ml::moe_permute_rows(backend, std::span<const float>(s.recv), s.layout, std::span<float>(s.xe), d);
        ml::moe_experts_forward<C>(backend, ml::moe_expert_range(w.experts, ml::first_expert(r, C.experts, P), local), s.expert_offsets, s.xe,
                                   s.ye, {s.up, s.gate, s.hidden});
        ml::moe_unpermute_rows(backend, std::span<const float>(s.ye), s.layout, std::span<float>(s.back), d);
    }
    exchange(ranks, &rank::back, &rank::yp, &rank::receive, bf16);
    for (auto& s : ranks)
        ml::moe_combine_rows<C>(backend, s.yp, s.weights, s.inverse, std::span<float>(s.ys));
    //! [dispatch]
    std::vector<float> out;
    for (const auto& s : ranks)
        out.insert(out.end(), s.ys.begin(), s.ys.end());
    std::printf("  rank 0 sent %u rows to itself and %u to rank 1; rank 1 holds experts %zu-%zu (expert %u is on rank %zu)\n",
                ranks[0].send[1], ranks[0].send[2] - ranks[0].send[1], ml::first_expert(1, C.experts, P), ml::first_expert(1, C.experts, P) + local - 1,
                3u, ml::expert_rank(3, C.experts, P));
    return out;
}

} // namespace

int main()
{
    std::printf("snicholls-core expert_parallel demo\n");
    bool ok = true;
    const auto& backend = snicholls::linalg::reference;
    constexpr std::size_t rows = 10;
    const auto params = noise(ml::moe_parameter_count(C), 1, 0.4), xs = noise(rows * d, 2, 1.0);
    const auto w = ml::moe_weights_in<C>(std::span<const float>(params));

    // One process holding every token and every expert
    std::vector<float> values(ml::moe_activation_values(C, rows)), ys(rows * d);
    std::vector<u32> indices(ml::moe_activation_indices(C, rows));
    ml::moe_forward<C>(backend, w, xs, ys, ml::moe_activations_in<C>(std::span<float>(values), std::span<u32>(indices), rows));

    const auto split = two_ranks(w, xs, false);
    std::printf("  two ranks %s one process to the bit\n", split == ys ? "give" : "do NOT give");
    ok &= split == ys;

    const auto wire = two_ranks(w, xs, true);
    double worst = 0;
    for (std::size_t i = 0; i < ys.size(); ++i)
        worst = std::max(worst, double(std::abs(wire[i] - ys[i])));
    std::printf("  with a bfloat16 payload: within %.4f of it\n", worst);
    ok &= worst > 0 && worst < 0.05;

    std::printf(ok ? "expert_parallel demo: ok\n" : "expert_parallel demo: FAILED\n");
    return ok ? 0 : 1;
}
