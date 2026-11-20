//
//  gla_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/ml/gla.hpp>: gated linear attention as an associative memory that forgets: a key written
//  early is recalled less the longer its channel's gate has been closing; the chunked form gives the recurrent form's
//  outputs; decoding with a fixed state; the gradient in a gate. Exits non-zero if a result is not what it documents.
//

#include <snicholls/core/linalg/reference.hpp>
#include <snicholls/core/ml/gla.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <span>

int main()
{
    std::printf("snicholls-core gated linear attention demo\n");
    namespace ml = snicholls::ml;
    constexpr auto& backend = snicholls::linalg::reference;
    bool ok = true;

    //! [gla]
    // One head, 2 key channels, 2 value channels, chunks of 2
    constexpr ml::transformer_config C = [] {
        auto c = ml::tiny_mamba2;
        c.mixer = ml::mixer_kind::gla;
        c.mixer_heads = c.mixer_groups = 1;
        c.mixer_state = c.mixer_head_dim = 2;
        c.conv_width = 0;
        c.mixer_chunk = 2;
        return c;
    }();
    // Token 0 writes the value (1, 0) under the key e0; every token reads with the query e0. Key channel 0's gate is
    // log 0.5 from token 1: the memory halves each step
    constexpr std::size_t n = 4;
    const std::array<double, n * 2> q{1, 0, 1, 0, 1, 0, 1, 0}, k{1, 0, 0, 0, 0, 0, 0, 0}, v{1, 0, 0, 0, 0, 0, 0, 0};
    const double half = std::log(0.5);
    const std::array<double, n * 2> gk{0, 0, half, 0, half, 0, half, 0};
    const ml::gla_inputs<double> in{q, k, v, gk};
    std::array<double, n * 2> o{}, o_chunked{};
    std::array<double, ml::gla_state_size(C)> s{}, s_chunked{};
    std::array<double, ml::gla_chunk_work(C)> work{};
    ml::gla_forward<C>(backend, in, std::span<double>(o), 1, std::span<double>(s));
    ml::gla_forward_chunked<C>(backend, in, std::span<double>(o_chunked), 1, std::span<double>(s_chunked), {}, std::span<double>(work));

    // Decoding a fifth token from the state: it recalls 1/16 of the value, scaled by 1/sqrt(2)
    const std::array<double, 2> q5{1, 0}, k5{0, 0}, v5{0, 0}, g5{half, 0};
    std::array<double, 2> o5{};
    ml::gla_step<C>(ml::gla_inputs<double>{q5, k5, v5, g5}, std::span<double>(o5), std::span<double>(s));

    // Training: how the last recall depends on channel 0's gate at token 2
    std::array<double, n * 2> dout{0, 0, 0, 0, 0, 0, 1, 0}, dq{}, dk{}, dv{}, dg{};
    std::array<double, ml::recurrence_checkpoints(n, C.mixer_chunk) * ml::gla_state_size(C)> checkpoints{};
    std::array<double, ml::gla_state_size(C)> s0{}, dstate{};
    ml::gla_forward<C>(backend, in, std::span<double>(o), 1, std::span<double>(s0), std::span<double>(checkpoints));
    std::array<double, ml::gla_backward_work<C, double>()> bwork{};
    ml::gla_backward<C>(backend, in, std::span<const double>(dout), ml::gla_gradients<double>{dq, dk, dv, dg}, 1,
                        std::span<const double>(checkpoints), std::span<double>(dstate), std::span<double>(bwork));
    //! [gla]

    const double scale = 1 / std::sqrt(2.0);
    std::printf("  recall of the first value: %.4f %.4f %.4f %.4f, then %.4f decoded\n", o[0] / scale, o[2] / scale, o[4] / scale,
                o[6] / scale, o5[0] / scale);
    ok &= std::fabs(o[6] - 0.125 * scale) < 1e-15 && std::fabs(o5[0] - 0.0625 * scale) < 1e-15;
    for (std::size_t i = 0; i < o.size(); ++i)
        ok &= std::fabs(o[i] - o_chunked[i]) < 1e-15;
    // o_3 = scale · 0.5³ = scale · e^{3 log 0.5}: d o_3 / d gk_2[0] = o_3
    ok &= std::fabs(dg[4] - o[6]) < 1e-15 && dg[1] == 0;
    std::printf(ok ? "gated linear attention demo: ok\n" : "gated linear attention demo: FAILED\n");
    return ok ? 0 : 1;
}
