//
//  delta_rule_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/ml/delta_rule.hpp>: the delta rule overwriting what a key stored where linear attention
//  would add to it; the gate forgetting; the chunked WY form (a triangular solve on the backend) giving the recurrent
//  form's outputs; decoding with a fixed state; the normalised keys and a backward pass. Exits non-zero if a result is
//  not what it documents.
//

#include <snicholls/core/linalg/reference.hpp>
#include <snicholls/core/ml/delta_rule.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <span>

int main()
{
    std::printf("snicholls-core gated delta rule demo\n");
    namespace ml = snicholls::ml;
    constexpr auto& backend = snicholls::linalg::reference;
    bool ok = true;

    //! [delta_rule]
    // One head, 2 key channels, 2 value channels, chunks of 4
    constexpr ml::transformer_config C = [] {
        auto c = ml::tiny_gated_deltanet;
        c.mixer_heads = c.mixer_groups = 1;
        c.mixer_state = c.mixer_head_dim = 2;
        c.hybrid_period = c.hybrid_attention_offset = 0;
        return c;
    }();
    constexpr std::size_t n = 4, S = ml::delta_rule_state_size(C);
    // Keys: (3, 4) normalised to (0.6, 0.8) at tokens 0 and 1; token 1 stores a new value under the same key with β = 1
    std::array<double, n * 2> k{3, 4, 3, 4, 0, 0, 0, 0};
    ml::l2_normalise_rows(backend, std::span<const double>(k), std::span<double>(k), 2, 1e-12);
    const std::array<double, n * 2> q{0.6, 0.8, 0.6, 0.8, 0.6, 0.8, 0.6, 0.8}, v{1, 0, 0, 1, 0, 0, 0, 0};
    const std::array<double, n> g{0, 0, 0, std::log(0.5)}, beta{1, 1, 0, 0};   // token 3 halves the memory
    const ml::delta_rule_inputs<double> in{q, k, v, g, beta};
    std::array<double, n * 2> o{}, o_chunked{};
    std::array<double, S> s{}, s_chunked{};
    std::array<double, ml::delta_rule_chunk_work(C)> work{};
    ml::delta_rule_forward<C>(backend, in, std::span<double>(o), 1, std::span<double>(s));
    ml::delta_rule_forward_chunked<C>(backend, in, std::span<double>(o_chunked), 1, std::span<double>(s_chunked), {}, std::span<double>(work));

    // Decoding the next token (β = 0: read only)
    const std::array<double, 2> q5{0.6, 0.8}, zero{0, 0};
    const std::array<double, 1> g5{0}, b5{0};
    std::array<double, 2> o5{};
    ml::delta_rule_step<C>(ml::delta_rule_inputs<double>{q5, zero, zero, g5, b5}, std::span<double>(o5), std::span<double>(s));

    // Training: the gradient of the last output's second channel in β at token 1
    std::array<double, n * 2> dout{0, 0, 0, 0, 0, 0, 0, 1}, dq{}, dk{}, dv{};
    std::array<double, n> dg{}, dbeta{};
    std::array<double, ml::recurrence_checkpoints(n, C.mixer_chunk) * S> checkpoints{};
    std::array<double, S> s0{}, dstate{};
    ml::delta_rule_forward_chunked<C>(backend, in, std::span<double>(o_chunked), 1, std::span<double>(s0), std::span<double>(checkpoints),
                                      std::span<double>(work));
    std::array<double, ml::delta_rule_backward_work<C, double>()> bwork{};
    ml::delta_rule_backward<C>(backend, in, std::span<const double>(dout), ml::delta_rule_gradients<double>{dq, dk, dv, dg, dbeta}, 1,
                               std::span<const double>(checkpoints), std::span<double>(dstate), std::span<double>(bwork));
    //! [delta_rule]

    const double scale = 1 / std::sqrt(2.0);
    std::printf("  reads (%.3f, %.3f), then (%.3f, %.3f) after the overwrite, (%.3f, %.3f) after the gate\n", o[0] / scale, o[1] / scale,
                o[2] / scale, o[3] / scale, o[6] / scale, o[7] / scale);
    ok &= std::fabs(o[0] / scale - 1) < 1e-12 && std::fabs(o[2] / scale) < 1e-12 && std::fabs(o[3] / scale - 1) < 1e-12;
    ok &= std::fabs(o[7] / scale - 0.5) < 1e-12 && std::fabs(o5[1] / scale - 0.5) < 1e-12;
    for (std::size_t i = 0; i < o.size(); ++i)
        ok &= std::fabs(o[i] - o_chunked[i]) < 1e-14;
    // S after token 1 is k ((1, 0) + β_1 ((0, 1) - (1, 0)))ᵀ and q · k = 1, so o_3[1] = scale · 0.5 · β_1
    ok &= std::fabs(dbeta[1] - 0.5 * scale) < 1e-12;
    std::printf(ok ? "gated delta rule demo: ok\n" : "gated delta rule demo: FAILED\n");
    return ok ? 0 : 1;
}
