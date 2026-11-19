//
//  ssd_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/ml/ssd.hpp>: Mamba-2's selective scan over a sequence, the chunked SSD form giving the same
//  outputs, decoding with a state whose size does not grow, and the backward pass from the chunked form's checkpoints.
//  Exits non-zero if a result is not what it documents.
//

#include <snicholls/core/linalg/reference.hpp>
#include <snicholls/core/ml/ssd.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <span>

int main()
{
    std::printf("snicholls-core SSD (Mamba-2) demo\n");
    namespace ml = snicholls::ml;
    constexpr auto& backend = snicholls::linalg::reference;
    bool ok = true;

    //! [ssd]
    // tiny_mamba2: 2 heads of 4 channels sharing one group of 4 state channels, chunks of 4
    constexpr ml::transformer_config C = ml::tiny_mamba2;
    constexpr std::size_t n = 10, H = C.mixer_heads, P = C.mixer_head_dim, N = C.mixer_state;
    std::array<double, n * H * P> x{};
    std::array<double, n * H> dt{};
    std::array<double, n * N> b{}, c{};
    const std::array<double, H> a{-0.5, -2.0};   // A: the second head forgets faster
    for (std::size_t t = 0; t < n; ++t) {
        for (std::size_t i = 0; i < H * P; ++i)
            x[t * H * P + i] = std::sin(0.3 * static_cast<double>(t + i));
        for (std::size_t h = 0; h < H; ++h)
            dt[t * H + h] = 0.1 + 0.05 * static_cast<double>(h + t % 3);
        for (std::size_t i = 0; i < N; ++i) {
            b[t * N + i] = std::cos(0.2 * static_cast<double>(t * N + i));
            c[t * N + i] = std::sin(0.7 * static_cast<double>(t + i));
        }
    }
    const ml::ssd_inputs<double> in{x, dt, a, b, c};

    // The scan and the chunked form (training's), each from a zero state, each saving the state at every chunk's start
    constexpr std::size_t S = ml::ssd_state_size(C), chunks = ml::recurrence_checkpoints(n, C.mixer_chunk);
    std::array<double, n * H * P> y_scan{}, y_chunked{};
    std::array<double, S> state_scan{}, state_chunked{};
    std::array<double, chunks * S> checkpoints{};
    ml::ssd_forward<C>(backend, in, std::span<double>(y_scan), 1, std::span<double>(state_scan));
    ml::ssd_forward_chunked<C>(backend, in, std::span<double>(y_chunked), 1, std::span<double>(state_chunked), std::span<double>(checkpoints));

    // Decoding: one token at a time, the same S numbers whatever the length
    std::array<double, S> state{};
    std::array<double, H * P> y_t{};
    for (std::size_t t = 0; t < n; ++t)
        ml::ssd_step<C>(ml::ssd_inputs<double>{std::span<const double>(x).subspan(t * H * P, H * P), std::span<const double>(dt).subspan(t * H, H), a,
                                               std::span<const double>(b).subspan(t * N, N), std::span<const double>(c).subspan(t * N, N)},
                        std::span<double>(y_t), std::span<double>(state));

    // Training: the gradient of Σ y in every input, from the chunked form's checkpoints
    std::array<double, n * H * P> dy{}, dx{};
    dy.fill(1.0);
    std::array<double, n * H> ddt{};
    std::array<double, H> da{};
    std::array<double, n * N> db{}, dc{};
    std::array<double, S> dstate{};
    std::array<double, ml::ssd_backward_work<C, double>()> work{};
    ml::ssd_backward<C>(backend, in, std::span<const double>(dy), ml::ssd_gradients<double>{dx, ddt, da, db, dc}, 1,
                        std::span<const double>(checkpoints), std::span<double>(dstate), std::span<double>(work));
    //! [ssd]

    double worst = 0;
    for (std::size_t i = 0; i < y_scan.size(); ++i)
        worst = std::fmax(worst, std::fabs(y_scan[i] - y_chunked[i]));
    std::printf("  %zu tokens, a state of %zu numbers; chunked against the scan: %.2e\n", n, S, worst);
    std::printf("  d(sum y)/dA = %.5f, %.5f\n", da[0], da[1]);
    ok &= worst < 1e-13;
    for (std::size_t i = 0; i < H * P; ++i)
        ok &= y_t[i] == y_scan[(n - 1) * H * P + i];
    ok &= state == state_scan;
    // The gradient in A against a central difference
    const auto total = [&](double a0) {
        const std::array<double, H> a2{a0, a[1]};
        std::array<double, n * H * P> y2{};
        std::array<double, S> s2{};
        ml::ssd_forward<C>(backend, ml::ssd_inputs<double>{x, dt, a2, b, c}, std::span<double>(y2), 1, std::span<double>(s2));
        double sum = 0;
        for (const double v : y2)
            sum += v;
        return sum;
    };
    ok &= std::fabs(da[0] - (total(a[0] + 1e-6) - total(a[0] - 1e-6)) / 2e-6) < 1e-7;
    std::printf(ok ? "SSD demo: ok\n" : "SSD demo: FAILED\n");
    return ok ? 0 : 1;
}
