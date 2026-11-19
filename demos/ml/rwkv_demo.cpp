//
//  rwkv_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/ml/rwkv.hpp>: token shift; RWKV-6's WKV reading the state before the update with the
//  bonus for the current token; RWKV-7's generalised delta rule replacing what a key stored; decoding with the fixed
//  state; a backward pass. Exits non-zero if a result is not what it documents.
//

#include <snicholls/core/linalg/reference.hpp>
#include <snicholls/core/ml/rwkv.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <span>

int main()
{
    std::printf("snicholls-core RWKV demo\n");
    namespace ml = snicholls::ml;
    constexpr auto& backend = snicholls::linalg::reference;
    bool ok = true;

    //! [rwkv]
    // One head of 2 channels, chunks of 2
    constexpr ml::transformer_config C = [] {
        auto c = ml::tiny_mamba2;
        c.mixer = ml::mixer_kind::rwkv7;
        c.mixer_heads = c.mixer_groups = 1;
        c.mixer_state = c.mixer_head_dim = 2;
        c.conv_width = 0;
        c.mixer_chunk = 2;
        return c;
    }();
    constexpr std::size_t n = 3, S = ml::wkv_state_size(C);

    // Token shift: each channel halfway between the token and the one before
    const std::array<double, n * 2> x{2, 4, 6, 8, 10, 12};
    const std::array<double, 2> mu{0.5, 0.5};
    std::array<double, n * 2> shifted{};
    ml::token_shift(backend, std::span<const double>(x), std::span<const double>(mu), std::span<double>(shifted));

    // RWKV-6: write value (1, 0) under key e0 at token 0, read with r = e0 at tokens 0-2; no decay; bonus u = 3
    const std::array<double, n * 2> r{1, 0, 1, 0, 1, 0}, k{1, 0, 0, 0, 0, 0}, v{1, 0, 0, 0, 0, 0}, lw{0, 0, 0, 0, 0, 0};
    const std::array<double, 2> u{3, 3};
    std::array<double, n * 2> o6{};
    std::array<double, S> s6{};
    ml::wkv6_forward<C>(backend, ml::wkv6_inputs<double>{r, k, v, lw, u}, std::span<double>(o6), 1, std::span<double>(s6));

    // RWKV-7: write (1, 0) under e0, then replace it with (0, 1): a = -e0, b = e0 removes what e0 held first
    const std::array<double, n * 2> k7{1, 0, 1, 0, 0, 0}, v7{1, 0, 0, 1, 0, 0}, a7{0, 0, -1, 0, 0, 0}, b7{0, 0, 1, 0, 0, 0};
    std::array<double, n * 2> o7{};
    std::array<double, S> s7{};
    ml::wkv7_forward<C>(backend, ml::wkv7_inputs<double>{r, lw, k7, v7, a7, b7}, std::span<double>(o7), 1, std::span<double>(s7));

    // Decoding the next token from RWKV-7's state: it reads (0, 1)
    const std::array<double, 2> r4{1, 0}, zero{0, 0};
    std::array<double, 2> o4{};
    ml::wkv7_step<C>(ml::wkv7_inputs<double>{r4, zero, zero, zero, zero, zero}, std::span<double>(o4), std::span<double>(s7));

    // Training: RWKV-6's gradient in the bonus from the outputs' sum
    const std::array<double, n * 2> dout{1, 1, 1, 1, 1, 1};
    std::array<double, n * 2> dr{}, dk{}, dv{}, dlw{};
    std::array<double, 2> du{};
    std::array<double, ml::recurrence_checkpoints(n, C.mixer_chunk) * S> checkpoints{};
    std::array<double, S> s0{}, dstate{};
    ml::wkv6_forward<C>(backend, ml::wkv6_inputs<double>{r, k, v, lw, u}, std::span<double>(o6), 1, std::span<double>(s0),
                        std::span<double>(checkpoints));
    std::array<double, ml::wkv_backward_work<C, double>()> work{};
    ml::wkv6_backward<C>(backend, ml::wkv6_inputs<double>{r, k, v, lw, u}, std::span<const double>(dout),
                         ml::wkv6_gradients<double>{dr, dk, dv, dlw, du}, 1, std::span<const double>(checkpoints), std::span<double>(dstate),
                         std::span<double>(work));
    //! [rwkv]

    std::printf("  token shift: %.1f %.1f %.1f\n", shifted[0], shifted[2], shifted[4]);
    std::printf("  RWKV-6 reads %.1f (the bonus), then %.1f %.1f (the state)\n", o6[0], o6[2], o6[4]);
    std::printf("  RWKV-7 reads (%.1f, %.1f), then (%.1f, %.1f) after the replacement\n", o7[0], o7[1], o7[2], o7[3]);
    ok &= shifted[0] == 1 && shifted[2] == 4 && shifted[4] == 8;
    ok &= o6[0] == 3 && o6[2] == 1 && o6[4] == 1;
    ok &= o7[0] == 1 && o7[1] == 0 && o7[2] == 0 && o7[3] == 1 && o4[0] == 0 && o4[1] == 1;
    ok &= du[0] == 1 && du[1] == 0;   // only token 0's own key and value meet the bonus
    std::printf(ok ? "RWKV demo: ok\n" : "RWKV demo: FAILED\n");
    return ok ? 0 : 1;
}
