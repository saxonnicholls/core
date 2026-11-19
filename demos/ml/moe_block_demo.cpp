//
//  moe_block_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/ml/moe_block.hpp>: a transformer block whose MLP is a mixture of experts (ml::tiny_moe:
//  4 experts, top 2, one shared, RoPE), over a batch of two sequences, trained for a few SGD steps to map its input to
//  a fixed target; the loss, with the router's z-loss, falls. Exits non-zero if a result is not what it documents.
//

#include <snicholls/core/ml/moe_block.hpp>

#include <snicholls/core/linalg/reference.hpp>
#include <snicholls/core/ml/rope.hpp>
#include <snicholls/core/stats/random.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <utility>
#include <vector>

namespace {

namespace ml = snicholls::ml;
namespace stats = snicholls::stats;
using u32 = std::uint32_t;

std::vector<float> noise(std::size_t n, std::uint64_t seed, double scale, double offset = 0)
{
    auto g = stats::wyrand::stream(seed, 0);
    std::vector<float> v(n);
    for (auto& e : v)
        e = static_cast<float>(offset + scale * (2.0 * stats::uniform_real(g) - 1.0));
    return v;
}

constexpr auto C = [] { auto c = ml::tiny_moe; c.z_loss_weight = 0.001; return c; }();

} // namespace

int main()
{
    std::printf("snicholls-core moe_block demo\n");
    bool ok = true;
    const auto& backend = snicholls::linalg::reference;

    //! [moe_block]
    constexpr std::size_t seqs = 2, n = 6, rows = seqs * n, d = C.d_model, q = C.q_size(), kv = C.kv_size();
    static const ml::rope_table<float, n, C.head_dim> rope(ml::rope_base_v<C, float>);
    // The parameters: two RMS-norm weights, the attention's four matrices, the mixture of experts
    std::vector<float> norm1 = noise(d, 1, 0.1, 1.0), norm2 = noise(d, 2, 0.1, 1.0), wq = noise(q * d, 3, 0.4), wk = noise(kv * d, 4, 0.4),
                       wv = noise(kv * d, 5, 0.4), wo = noise(d * q, 6, 0.4), moe = noise(ml::moe_parameter_count(C), 7, 0.3);
    const ml::moe_block_weights<float> w{norm1, {}, {{wq, q, d}, {wk, kv, d}, {wv, kv, d}, {wo, d, q}}, norm2, {},
                                         ml::moe_weights_in<C>(std::span<const float>(moe))};
    // The activations and gradients, the MoE's laid out by its own size functions
    std::vector<float> x0(rows * d), h1 = x0, x1 = x0, h2 = x0, aq(rows * q), ak(rows * kv), av = ak, ay = aq, lse(rows * C.n_heads),
                       mv(ml::moe_activation_values(C, rows)), mw(ml::moe_gradient_work(C, rows)), gm(moe.size()), gq(wq.size()), gk(wk.size()),
                       gv(wv.size()), go(wo.size()), gn1(d), gn2(d), dq(rows * q), dk(rows * kv), dv = dk, dy = dq, scratch(rows * d);
    std::vector<u32> mi(ml::moe_activation_indices(C, rows));
    const ml::moe_block_activations<float> act{x0, h1, x1, h2, {aq, ak, av, {}, {}, ay, lse},
                                               ml::moe_activations_in<C>(std::span<float>(mv), std::span<u32>(mi), rows)};
    const ml::moe_block_gradients<float> grads{gn1, {}, gn2, {}, {dq, dk, dv, gq, gk, gv, go, {}, {}, {}, {}, {}, {}, dy},
                                               ml::moe_gradients_in<C>(std::span<float>(gm), std::span<float>(mw), rows), scratch};
    const auto input = noise(rows * d, 8, 1.0), target = noise(rows * d, 9, 1.0);
    auto step = [&] {
        std::vector<float> xs = input;
        const auto aux = ml::block_forward_sequence<C>(backend, w, xs, rope, act, seqs);
        float loss = aux.balance + aux.z;
        std::vector<float> dxs(xs.size());
        for (std::size_t i = 0; i < xs.size(); ++i) {
            loss += 0.5f * (xs[i] - target[i]) * (xs[i] - target[i]) / rows;
            dxs[i] = (xs[i] - target[i]) / rows;
        }
        for (auto* g : {&gm, &gq, &gk, &gv, &go, &gn1, &gn2})
            std::fill(g->begin(), g->end(), 0.0f);
        ml::block_backward<C>(backend, w, rope, act, dxs, grads, seqs);
        for (auto [p, g] : {std::pair{&moe, &gm}, std::pair{&wq, &gq}, std::pair{&wk, &gk}, std::pair{&wv, &gv}, std::pair{&wo, &go},
                            std::pair{&norm1, &gn1}, std::pair{&norm2, &gn2}})
            for (std::size_t i = 0; i < p->size(); ++i)
                (*p)[i] -= 0.2f * (*g)[i];
        return loss;
    };
    //! [moe_block]
    const float first = step();
    float last = first;
    for (int i = 0; i < 40; ++i)
        last = step();
    std::printf("  block loss %.4f -> %.4f after 40 SGD steps; %u of %zu choices went to expert 0 last\n", static_cast<double>(first),
                static_cast<double>(last), act.moe.counts[0], rows * C.top_k);
    ok &= last < 0.8f * first;

    std::printf(ok ? "moe_block demo: ok\n" : "moe_block demo: FAILED\n");
    return ok ? 0 : 1;
}
