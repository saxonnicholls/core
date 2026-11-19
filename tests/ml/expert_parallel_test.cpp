//
//  expert_parallel_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

#include <snicholls/core/ml/expert_parallel.hpp>

#include <snicholls/core/linalg/reference.hpp>
#include <snicholls/core/ml/diloco.hpp>
#include <snicholls/core/ml/minifloat.hpp>
#include <snicholls/core/ml/moe.hpp>
#include <snicholls/core/stats/random.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace {

namespace ml = snicholls::ml;
namespace stats = snicholls::stats;
using u32 = std::uint32_t;
constexpr auto& ref = snicholls::linalg::reference;

template <typename T>
std::vector<T> noise(std::size_t n, std::uint64_t seed, double scale = 1.0)
{
    auto g = stats::wyrand::stream(seed, 0);
    std::vector<T> v(n);
    for (auto& e : v)
        e = static_cast<T>(scale * (2.0 * stats::uniform_real(g) - 1.0));
    return v;
}

// Eight fine-grained experts (top 3), a shared expert, DeepSeek-V3's sigmoid scoring with a selection bias, both
// auxiliary losses, biases on the projections
constexpr ml::transformer_config C{.d_model = 6, .n_heads = 1, .n_kv_heads = 1, .head_dim = 6, .d_ff = 4, .mlp = ml::mlp_kind::swiglu,
                                   .bias = true, .experts = 8, .top_k = 3, .shared_experts = 1, .expert_d_ff = 4,
                                   .scoring = ml::moe_scoring::sigmoid, .route_scale = 2.5, .balance_loss_weight = 0.01,
                                   .z_loss_weight = 0.001};

TEST(MlExpertParallel, PlacesContiguousBlocksOfExperts)
{
    static_assert(ml::experts_per_rank(64, 8) == 8 && ml::first_expert(3, 64, 8) == 24 && ml::expert_rank(31, 64, 8) == 3
                  && ml::expert_rank(32, 64, 8) == 4 && ml::expert_rank(63, 64, 1) == 0);
    static_assert(noexcept(ml::expert_rank(0, 1, 1)) && noexcept(ml::first_expert(0, 1, 1)));
    for (std::size_t e = 0; e < 64; ++e) {
        const std::size_t r = ml::expert_rank(e, 64, 4);
        EXPECT_GE(e, ml::first_expert(r, 64, 4));
        EXPECT_LT(e, ml::first_expert(r + 1, 64, 4));
    }
}

TEST(MlExpertParallel, TheLayoutsOfTwoRanks)
{
    constexpr auto D = [] { auto c = ml::tiny_moe; return c; }();   // E 4: rank 0 holds 0-1, rank 1 holds 2-3
    // Source 0 routed 2, 0, 1, 3 rows to experts 0..3; source 1 routed 1, 2, 0, 1
    const std::vector<u32> c0{2, 0, 1, 3}, c1{1, 2, 0, 1}, o0{0, 2, 2, 3, 6};
    const std::vector<std::span<const u32>> counts{c0, c1};
    std::vector<u32> send(3), recv0(3), recv1(3), eo0(3), eo1(3), order0(5), order1(5);
    ml::moe_send_offsets<D>(ref, o0, 2, send);
    EXPECT_EQ(send, (std::vector<u32>{0, 2, 6}));
    ml::moe_receive_offsets<D>(ref, counts, 0, recv0);
    ml::moe_receive_offsets<D>(ref, counts, 1, recv1);
    EXPECT_EQ(recv0, (std::vector<u32>{0, 2, 5}));   // source 0's 2 rows for experts 0-1, then source 1's 3
    EXPECT_EQ(recv1, (std::vector<u32>{0, 4, 5}));
    ml::moe_receive_layout<D>(ref, counts, 0, eo0, order0);
    ml::moe_receive_layout<D>(ref, counts, 1, eo1, order1);
    // Rank 0 received [s0 e0, s0 e0, s1 e0, s1 e1, s1 e1]; expert order: e0 from s0, s1, then e1 from s1
    EXPECT_EQ(eo0, (std::vector<u32>{0, 3, 5}));
    EXPECT_EQ(order0, (std::vector<u32>{0, 1, 2, 3, 4}));
    // Rank 1 received [s0 e2, s0 e3 x3, s1 e3]; expert 2 from s0, then expert 3 from s0 and s1
    EXPECT_EQ(eo1, (std::vector<u32>{0, 1, 5}));
    EXPECT_EQ(order1, (std::vector<u32>{0, 1, 2, 3, 4}));
    // Interleaved: source 1 also has an expert-2 row, which comes before source 0's expert-3 rows on rank 1
    const std::vector<u32> c1b{0, 0, 1, 1};
    const std::vector<std::span<const u32>> counts_b{c0, c1b};
    std::vector<u32> eo(3), order(6);
    ml::moe_receive_layout<D>(ref, counts_b, 1, eo, order);
    EXPECT_EQ(eo, (std::vector<u32>{0, 2, 6}));
    EXPECT_EQ(order, (std::vector<u32>{0, 4, 1, 2, 3, 5}));
}

TEST(MlExpertParallel, TheAllToAllDeliversInSourceOrderAndPermutingIsUndone)
{
    // Two ranks, rows of width 2: rank 0 sends 1 row to itself and 2 to rank 1; rank 1 sends 2 to rank 0 and none to itself
    const std::vector<int> s0{1, 1, 2, 2, 3, 3}, s1{4, 4, 5, 5};
    const std::vector<u32> o0{0, 1, 3}, o1{0, 2, 2};
    std::vector<int> r0(6), r1(4);
    const std::vector<std::span<const int>> send{s0, s1};
    const std::vector<std::span<const u32>> offsets{o0, o1};
    const std::vector<std::span<int>> receive{r0, r1};
    ml::moe_all_to_all(ref, std::span<const std::span<const int>>(send), offsets, 2, std::span<const std::span<int>>(receive));
    EXPECT_EQ(r0, (std::vector<int>{1, 1, 4, 4, 5, 5}));
    EXPECT_EQ(r1, (std::vector<int>{2, 2, 3, 3}));
    const std::vector<u32> order{2, 0, 1};
    std::vector<int> p(6), back(6);
    ml::moe_permute_rows(ref, std::span<const int>(r0), order, std::span<int>(p), 2);
    EXPECT_EQ(p, (std::vector<int>{5, 5, 1, 1, 4, 4}));
    ml::moe_unpermute_rows(ref, std::span<const int>(p), order, std::span<int>(back), 2);
    EXPECT_EQ(back, r0);
}

template <typename T>
std::span<const T> dy_of(const std::vector<T>& dys, std::size_t r, std::size_t rows)
{
    return std::span<const T>(dys).subspan(r * rows * C.d_model, rows * C.d_model);
}

// One rank's share of an expert-parallel layer: its own rows routed and permuted, the exchange, its experts' products
template <typename T>
struct rank_state
{
    std::size_t rows = 0;
    std::vector<T> xs, ys, logits, weights, xp, yp, dxs, dyp, dweights, dlogits, dxp, recv, xe, ye, back, dye, dxe;
    std::vector<T> eup, egate, ehidden, shared_up, shared_gate, shared_hidden, shared_y, grads, scratch, shared_scratch;
    std::vector<u32> experts, counts, offsets, order, inverse, send, receive, expert_offsets, layout;
};

// The layer over `total` rows split across `ranks` in-process ranks, each step as a distributed run makes it, the
// exchanges through the reference all-to-all: returns the ranks' states, with each rank's gradients in a full buffer
template <typename T>
std::vector<rank_state<T>> expert_parallel(std::size_t ranks, const std::vector<T>& params, const std::vector<T>& bias, const std::vector<T>& xs,
                                           const std::vector<T>& dys, std::size_t sequences, bool bf16 = false)
{
    constexpr std::size_t d = C.d_model, E = C.experts, K = C.top_k, f = C.expert_d_ff;
    constexpr auto S = ml::moe_shared_config(C);
    const std::size_t total = xs.size() / d, rows = total / ranks, local = ml::experts_per_rank(E, ranks);
    const auto w = ml::moe_weights_in<C>(std::span<const T>(params), std::span<const T>(bias));
    std::vector<rank_state<T>> st(ranks);
    // Each rank: route, count, permute and gather its own rows
    for (std::size_t r = 0; r < ranks; ++r) {
        auto& s = st[r];
        const std::size_t n = rows * K;
        s.rows = rows;
        s.xs.assign(xs.begin() + std::ptrdiff_t(r * rows * d), xs.begin() + std::ptrdiff_t((r + 1) * rows * d));
        s.ys.resize(rows * d);
        s.logits.resize(rows * E);
        s.weights.resize(n);
        s.experts.resize(n);
        s.counts.resize(E);
        s.offsets.resize(E + 1);
        s.order.resize(n);
        s.inverse.resize(n);
        s.xp.resize(n * d);
        s.yp.resize(n * d);
        s.send.resize(ranks + 1);
        s.receive.resize(ranks + 1);
        s.expert_offsets.resize(local + 1);
        ml::linear_sequence(ref, w.router, s.xs, s.logits);
        ml::moe_route_rows<C>(ref, s.logits, w.selection_bias, s.experts, std::span<T>(s.weights));
        ml::moe_count_rows<C>(ref, s.experts, s.counts);
        ml::moe_offsets<C>(ref, s.counts, s.offsets);
        ml::moe_permutation<C>(ref, s.experts, s.offsets, s.order, s.inverse);
        ml::moe_gather_rows<C>(ref, s.xs, s.order, std::span<T>(s.xp));
        ml::moe_send_offsets<C>(ref, s.offsets, ranks, s.send);
    }
    // The counts are exchanged; each rank sizes what it receives and the order its products read
    std::vector<std::span<const u32>> counts, sends, receives;
    for (auto& s : st) {
        counts.emplace_back(s.counts);
        sends.emplace_back(s.send);
    }
    for (std::size_t r = 0; r < ranks; ++r) {
        auto& s = st[r];
        ml::moe_receive_offsets<C>(ref, counts, r, s.receive);
        const std::size_t m = s.receive[ranks];
        s.recv.resize(m * d);
        s.xe.resize(m * d);
        s.ye.resize(m * d);
        s.back.resize(m * d);
        s.dye.resize(m * d);
        s.dxe.resize(m * d);
        s.layout.resize(m);
        s.eup.resize(m * f);
        s.egate.resize(m * f);
        s.ehidden.resize(m * f);
        s.scratch.resize(2 * m * f);
        ml::moe_receive_layout<C>(ref, counts, r, s.expert_offsets, s.layout);
        receives.emplace_back(s.receive);
    }
    auto exchange = [&](auto from, auto to, const std::vector<std::span<const u32>>& offsets) {
        std::vector<std::span<const T>> out;
        std::vector<std::span<T>> in;
        for (auto& s : st) {
            out.emplace_back(s.*from);
            in.emplace_back(s.*to);
        }
        if (!bf16) {
            ml::moe_all_to_all(ref, std::span<const std::span<const T>>(out), offsets, d, std::span<const std::span<T>>(in));
            return;
        }
        // The payload rounded to bfloat16 for the wire, widened on arrival
        std::vector<std::vector<ml::bfloat16>> wire_out(st.size()), wire_in(st.size());
        std::vector<std::span<const ml::bfloat16>> o;
        std::vector<std::span<ml::bfloat16>> i;
        for (std::size_t r = 0; r < st.size(); ++r) {
            wire_out[r].resize(out[r].size());
            wire_in[r].resize(in[r].size());
            ml::to_bfloat16_rows(ref, out[r], wire_out[r]);
            o.emplace_back(wire_out[r]);
            i.emplace_back(wire_in[r]);
        }
        ml::moe_all_to_all(ref, std::span<const std::span<const ml::bfloat16>>(o), offsets, d, std::span<const std::span<ml::bfloat16>>(i));
        for (std::size_t r = 0; r < st.size(); ++r)
            ml::to_float_rows(ref, std::span<const ml::bfloat16>(wire_in[r]), in[r]);
    };
    // Dispatch, the experts, and back
    exchange(&rank_state<T>::xp, &rank_state<T>::recv, sends);
    for (std::size_t r = 0; r < ranks; ++r) {
        auto& s = st[r];
        ml::moe_permute_rows(ref, std::span<const T>(s.recv), s.layout, std::span<T>(s.xe), d);
        const auto we = ml::moe_expert_range(w.experts, ml::first_expert(r, E, ranks), local);
        ml::moe_experts_forward<C>(ref, we, s.expert_offsets, s.xe, s.ye, ml::mlp_activations<T>{s.eup, s.egate, s.ehidden});
        ml::moe_unpermute_rows(ref, std::span<const T>(s.ye), s.layout, std::span<T>(s.back), d);
    }
    exchange(&rank_state<T>::back, &rank_state<T>::yp, receives);
    // Each rank: combine, the shared expert, and the backward pass of the same steps
    for (std::size_t r = 0; r < ranks; ++r) {
        auto& s = st[r];
        ml::moe_combine_rows<C>(ref, s.yp, s.weights, s.inverse, std::span<T>(s.ys));
        s.shared_up.resize(rows * S.d_ff);
        s.shared_gate.resize(rows * S.d_ff);
        s.shared_hidden.resize(rows * S.d_ff);
        s.shared_y.resize(rows * d);
        s.shared_scratch.resize(2 * rows * S.d_ff);
        ml::mlp_forward_sequence<S>(ref, w.shared, s.xs, s.shared_y, ml::mlp_activations<T>{s.shared_up, s.shared_gate, s.shared_hidden});
        ml::add_rows(ref, std::span<const T>(s.shared_y), std::span<T>(s.ys));
        s.dyp.resize(rows * K * d);
        s.dweights.resize(rows * K);
        s.dxs.assign(rows * d, T(0));
        s.dlogits.resize(rows * E);
        s.dxp.resize(rows * K * d);
        s.grads.assign(params.size(), T(0));
        const std::span<const T> dy = std::span<const T>(dys).subspan(r * rows * d, rows * d);
        ml::moe_combine_rows_backward<C>(ref, dy, s.yp, s.weights, s.inverse, s.dyp, std::span<T>(s.dweights));
    }
    exchange(&rank_state<T>::dyp, &rank_state<T>::recv, sends);
    for (std::size_t r = 0; r < ranks; ++r) {
        auto& s = st[r];
        std::vector<T> work(ml::moe_gradient_work(C, rows));
        const auto g = ml::moe_gradients_in<C>(std::span<T>(s.grads), std::span<T>(work), rows);
        auto ge = ml::moe_expert_range<C>(g.experts, ml::first_expert(r, E, ranks), local);
        ge.scratch = s.scratch;
        const auto we = ml::moe_expert_range(w.experts, ml::first_expert(r, E, ranks), local);
        ml::moe_permute_rows(ref, std::span<const T>(s.recv), s.layout, std::span<T>(s.dye), d);
        std::fill(s.dxe.begin(), s.dxe.end(), T(0));
        ml::moe_experts_backward<C>(ref, we, s.expert_offsets, s.xe, ml::mlp_activations<T>{s.eup, s.egate, s.ehidden}, s.dye, s.dxe, ge);
        ml::moe_unpermute_rows(ref, std::span<const T>(s.dxe), s.layout, std::span<T>(s.back), d);
    }
    exchange(&rank_state<T>::back, &rank_state<T>::dxp, receives);
    for (std::size_t r = 0; r < ranks; ++r) {
        auto& s = st[r];
        std::vector<T> work(ml::moe_gradient_work(C, rows));
        const auto g = ml::moe_gradients_in<C>(std::span<T>(s.grads), std::span<T>(work), rows);
        ml::moe_gather_rows_backward<C>(ref, std::span<const T>(s.dxp), s.inverse, std::span<T>(s.dxs));
        ml::moe_route_rows_backward<C>(ref, s.logits, s.experts, s.dweights, std::span<T>(s.dlogits));
        // The losses with the whole batch's denominators: the batch's sequences and rows
        ml::moe_balance_loss<C>(ref, s.logits, s.experts, std::span<T>(s.dlogits), sequences / ranks, sequences);
        ml::moe_z_loss<C>(ref, s.logits, std::span<T>(s.dlogits), total);
        ml::linear_sequence_backward(ref, w.router, s.xs, s.dlogits, s.dxs, g.router);
        auto gs = g.shared;
        gs.scratch = s.shared_scratch;
        ml::mlp_backward_sequence<S>(ref, w.shared, s.xs, ml::mlp_activations<T>{s.shared_up, s.shared_gate, s.shared_hidden}, dy_of(dys, r, rows), s.dxs, gs);
    }
    return st;
}


// The single-process layer over all the rows
struct single
{
    std::vector<float> ys, dxs, grads;
    ml::moe_losses<float> losses;
};

single one_process(const std::vector<float>& params, const std::vector<float>& bias, const std::vector<float>& xs, const std::vector<float>& dys,
                   std::size_t sequences)
{
    const std::size_t rows = xs.size() / C.d_model;
    single s{std::vector<float>(xs.size()), std::vector<float>(xs.size()), std::vector<float>(params.size()), {}};
    std::vector<float> values(ml::moe_activation_values(C, rows)), work(ml::moe_gradient_work(C, rows));
    std::vector<u32> indices(ml::moe_activation_indices(C, rows));
    const auto w = ml::moe_weights_in<C>(std::span<const float>(params), std::span<const float>(bias));
    const auto act = ml::moe_activations_in<C>(std::span<float>(values), std::span<u32>(indices), rows);
    s.losses = ml::moe_forward<C>(ref, w, xs, s.ys, act, sequences);
    ml::moe_backward<C>(ref, w, xs, act, dys, s.dxs, ml::moe_gradients_in<C>(std::span<float>(s.grads), std::span<float>(work), rows), sequences);
    return s;
}

TEST(MlExpertParallel, OneTwoAndFourRanksAreTheSingleProcessLayerToTheBit)
{
    constexpr std::size_t d = C.d_model, E = C.experts, total = 24, sequences = 4;
    const auto params = noise<float>(ml::moe_parameter_count(C), 1, 0.5), bias = noise<float>(E, 2, 0.2), xs = noise<float>(total * d, 3),
               dys = noise<float>(total * d, 4);
    const auto one = one_process(params, bias, xs, dys, sequences);
    // The experts' tensors in the parameter layout: after the router, before the shared MLP
    const std::size_t first = E * d, last = first + E * (3 * C.expert_d_ff * d + 2 * C.expert_d_ff + d);
    for (const std::size_t ranks : {1u, 2u, 4u}) {
        const auto st = expert_parallel<float>(ranks, params, bias, xs, dys, sequences);
        std::vector<std::span<const float>> parts;
        float balance = 0, z = 0;
        for (std::size_t r = 0; r < ranks; ++r) {
            const auto& s = st[r];
            for (std::size_t i = 0; i < s.ys.size(); ++i) {
                ASSERT_EQ(s.ys[i], one.ys[r * s.ys.size() + i]) << ranks << " ranks, rank " << r << " output " << i;
                ASSERT_EQ(s.dxs[i], one.dxs[r * s.dxs.size() + i]) << ranks << " ranks, rank " << r << " input gradient " << i;
            }
            parts.emplace_back(s.grads);
            balance += ml::moe_balance_loss<C>(ref, s.logits, s.experts, std::span<float>{}, sequences / ranks, sequences);
            z += ml::moe_z_loss<C>(ref, s.logits, std::span<float>{}, total);
        }
        EXPECT_NEAR(balance, one.losses.balance, 1e-7f);
        EXPECT_NEAR(z, one.losses.z, 1e-7f);
        std::vector<float> sum(params.size());
        ml::ordered_sum<float>(parts, sum);   // the data-parallel all-reduce
        for (std::size_t i = 0; i < sum.size(); ++i) {
            if (i >= first && i < last) {
                ASSERT_EQ(sum[i], one.grads[i]) << ranks << " ranks, expert gradient " << i;   // one rank's alone
            } else {
                ASSERT_NEAR(sum[i], one.grads[i], 2e-6f * std::max(1.0f, std::abs(one.grads[i]))) << ranks << " ranks, gradient " << i;
            }
        }
        if (ranks == 1) {
            EXPECT_EQ(sum, one.grads);   // one rank is the single process, to the bit
        }
    }
}

TEST(MlExpertParallel, ABfloat16PayloadIsTheFloatLayerWithinItsRounding)
{
    constexpr std::size_t d = C.d_model, total = 16, sequences = 2;
    const auto params = noise<float>(ml::moe_parameter_count(C), 5, 0.5), bias = noise<float>(C.experts, 6, 0.2), xs = noise<float>(total * d, 7),
               dys = noise<float>(total * d, 8);
    const auto exact = expert_parallel<float>(2, params, bias, xs, dys, sequences);
    const auto wire = expert_parallel<float>(2, params, bias, xs, dys, sequences, true);
    double worst = 0, scale = 0;
    for (std::size_t r = 0; r < 2; ++r)
        for (std::size_t i = 0; i < exact[r].ys.size(); ++i) {
            worst = std::max(worst, double(std::abs(exact[r].ys[i] - wire[r].ys[i])));
            scale = std::max(scale, double(std::abs(exact[r].ys[i])));
        }
    EXPECT_GT(worst, 0.0);                 // it does round
    EXPECT_LT(worst, 0.02 * scale);        // by about bfloat16's 2^-8, twice
    EXPECT_EQ(exact[0].weights, wire[0].weights);   // the routing never sees the wire
}

} // namespace
