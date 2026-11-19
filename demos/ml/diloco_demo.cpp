//
//  diloco_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/ml/diloco.hpp>: the fixed-order sum of three workers' gradients and why its order is
//  fixed; then DiLoCo on a small least-squares fit, four workers taking 25 local SGD steps a round on their own data for 40 rounds,
//  their pseudo-gradients averaged in rank order and applied by the outer step (Nesterov momentum 0.9, outer learning
//  rate 0.7), against one worker trained for the same number of steps; then the pseudo-gradients in bfloat16 with error
//  feedback, summed and averaged as the bfloat16 all-reduce does. Exits non-zero if a result is not what it
//  documents.
//

#include <snicholls/core/ml/diloco.hpp>

#include <snicholls/core/stats/random.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>

namespace {

namespace ml = snicholls::ml;
namespace stats = snicholls::stats;

constexpr std::size_t dim = 4;
constexpr std::array<double, dim> w_star{1.5, -2.0, 0.5, 3.0};

// One SGD step on y = w* . x (no noise) from a minibatch of four drawn from the worker's own stream
void sgd_step(std::span<double> w, stats::wyrand& g)
{
    std::array<double, dim> grad{};
    for (int b = 0; b < 4; ++b) {
        std::array<double, dim> x{};
        double e = 0;
        for (std::size_t j = 0; j < dim; ++j) {
            x[j] = 2.0 * stats::uniform_real(g) - 1.0;
            e += (w[j] - w_star[j]) * x[j];
        }
        for (std::size_t j = 0; j < dim; ++j)
            grad[j] += e * x[j] / 4.0;
    }
    for (std::size_t j = 0; j < dim; ++j)
        w[j] -= 0.1 * grad[j];
}

double distance(std::span<const double> w)
{
    double s = 0;
    for (std::size_t j = 0; j < dim; ++j)
        s += (w[j] - w_star[j]) * (w[j] - w_star[j]);
    return std::sqrt(s);
}

} // namespace

int main()
{
    std::printf("snicholls-core diloco demo\n");
    bool ok = true;

    //! [ordered_sum]
    // Three workers' gradients summed in rank order: the bits every all-reduce must reproduce
    const std::array<float, 2> rank0{1e8f, 0.5f}, rank1{1.0f, 0.25f}, rank2{-1e8f, 0.25f};
    const std::array<std::span<const float>, 3> parts{rank0, rank1, rank2};
    std::array<float, 2> sum{};
    ml::ordered_sum<float>(parts, sum);   // ((1e8 + 1) - 1e8) = 0: the 1 is lost to rounding in this order
    //! [ordered_sum]
    std::printf("  rank-order sum: %g %g (another order would give 1 for the first)\n", static_cast<double>(sum[0]),
                static_cast<double>(sum[1]));
    ok &= sum[0] == 0.0f && sum[1] == 1.0f;

    constexpr std::size_t workers = 4, rounds = 40, h = 25;
    std::array<double, dim> global{}, momentum{}, mean{};
    std::array<std::array<double, dim>, workers> deltas{};
    std::array<stats::wyrand, workers> streams{stats::wyrand::stream(7, 0), stats::wyrand::stream(7, 1),
                                               stats::wyrand::stream(7, 2), stats::wyrand::stream(7, 3)};
    std::array<std::span<const double>, workers> delta_parts{};
    for (std::size_t k = 0; k < workers; ++k)
        delta_parts[k] = deltas[k];

    //! [round]
    for (std::size_t r = 0; r < rounds; ++r) {
        for (std::size_t k = 0; k < workers; ++k) {   // on worker k, its own machine
            std::array<double, dim> local = global;
            for (std::size_t t = 0; t < h; ++t)
                sgd_step(local, streams[k]);
            ml::pseudo_gradient<double>(global, local, deltas[k]);   // what worker k sends
        }
        ml::ordered_mean<double>(delta_parts, mean);   // the all-reduce, in rank order, divided by the workers
        ml::diloco_outer_step<double>(global, mean, momentum, ml::diloco_config{});   // outer lr 0.7, Nesterov 0.9
    }
    //! [round]
    std::printf("  DiLoCo, %zu workers x %zu rounds x %zu local steps: distance to w* %.3g, %zu exchanges\n", workers, rounds,
                h, distance(global), rounds);

    std::array<double, dim> single{};
    auto g = stats::wyrand::stream(7, 0);
    for (std::size_t t = 0; t < rounds * h; ++t)
        sgd_step(single, g);
    // Without noise one worker converges faster per step; what DiLoCo saves is the network: an exchange per round,
    // where an all-reduce every step would make one per step
    std::printf("  one worker, %zu steps:                          distance to w* %.3g\n", rounds * h, distance(single));
    ok &= distance(global) < 1e-3 && distance(single) < 1e-3;

    // With momentum 0, outer learning rate 1 and no Nesterov the outer step is plain averaging
    std::array<double, 2> avg_global{2.0, 2.0}, avg_momentum{};
    const std::array<double, 2> avg_mean{0.5, -0.5};
    ml::diloco_outer_step<double>(avg_global, avg_mean, avg_momentum, {.outer_lr = 1.0, .momentum = 0.0, .nesterov = false});
    std::printf("  plain averaging: 2 - mean (0.5, -0.5) = (%g, %g)\n", avg_global[0], avg_global[1]);
    ok &= avg_global[0] == 1.5 && avg_global[1] == 2.5;

    //! [bf16]
    // Two workers send their pseudo-gradients in bfloat16 (half the bytes); each keeps what the rounding dropped
    const std::array<float, 2> delta0{1.0f + 0x1.0p-10f, 0.25f}, delta1{1.0f, -0.75f};
    std::array<float, 2> residual0{}, residual1{};   // zero before the first round; each worker's own
    std::array<std::uint16_t, 2> wire0{}, wire1{};
    ml::compress_bf16(delta0, residual0, wire0);   // 1 + 2^-10 goes as 1; residual0[0] = 2^-10, sent next round
    ml::compress_bf16(delta1, residual1, wire1);
    const std::array<std::span<const std::uint16_t>, 2> wires{wire0, wire1};
    std::array<float, 2> bf16_sum{}, bf16_mean{};
    ml::ordered_sum_bf16(wires, bf16_sum);     // the reducing rank: decoded, summed in rank order in float
    ml::ordered_mean_bf16(wires, bf16_mean);   // every worker: bf16(sum) / 2, as the bfloat16 all-reduce leaves it
    //! [bf16]
    std::printf("  bfloat16: sent %#06x %#06x, residual %g; sum (%g, %g), mean (%g, %g)\n", unsigned(wire0[0]), unsigned(wire1[0]),
                static_cast<double>(residual0[0]), static_cast<double>(bf16_sum[0]), static_cast<double>(bf16_sum[1]),
                static_cast<double>(bf16_mean[0]), static_cast<double>(bf16_mean[1]));
    ok &= wire0[0] == 0x3f80 && residual0[0] == 0x1.0p-10f && residual1[0] == 0.0f && bf16_sum[0] == 2.0f && bf16_sum[1] == -0.5f
          && bf16_mean[0] == 1.0f && bf16_mean[1] == -0.25f;

    std::printf("%s\n", ok ? "OK" : "FAILED");
    return ok ? 0 : 1;
}
