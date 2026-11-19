//
//  tensor_ops_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/ml/tensor_ops.hpp>: a diffusion model's timestep embedding, arange, cumsum and its
//  backward, argsort, the top-k mask, a causal mask from tri, and a lower-triangular solve with its backward. Exits
//  non-zero if a result is not what it documents.
//

#include <snicholls/core/ml/tensor_ops.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <limits>

int main()
{
    std::printf("snicholls-core tensor_ops demo\n");
    namespace ml = snicholls::ml;
    bool ok = true;

    //! [tensor_ops]
    // Two timesteps embedded in 6 channels: three cosines, then three sines.
    std::array<float, 12> embedding{};
    ml::timestep_embedding(std::array<float, 2>{0.0f, 10.0f}, 6, embedding);

    // arange(0, 1, 0.25) and the running sum of it; cumsum's backward is the running sum from the end.
    std::array<float, ml::arange_count(0.0f, 1.0f, 0.25f)> steps{};
    ml::arange(0.0f, 0.25f, steps);
    std::array<float, steps.size()> sums{}, back{};
    ml::cumsum(steps, sums);
    ml::cumsum_backward(std::array<float, 4>{1, 1, 1, 1}, back);

    // The indices that sort a row (ties in index order), and the row with all but its two largest masked.
    const std::array<float, 5> logits{0.5f, 2.0f, -1.0f, 2.0f, 1.0f};
    std::array<std::size_t, 5> order{}, scratch{};
    ml::argsort(logits, order, ml::sort_order::descending);
    std::array<float, 5> kept{};
    ml::top_k_mask(logits, 2, kept, scratch);

    // A causal mask: 0 on and below the diagonal, -infinity above.
    std::array<float, 9> mask{};
    ml::tri(ml::tri_kind::lower_diag, 3, 3, mask, mask, -std::numeric_limits<float>::infinity());

    // Solve L x = b for a lower-triangular L, then the gradients of sum(x).
    const std::array<double, 9> L{2, 0, 0, 1, 1, 0, 1, 1, 1};
    const std::array<double, 3> b{2, 3, 6};
    std::array<double, 3> x{}, db{};
    std::array<double, 9> dL{};
    ml::solve_tri(L, b, x, 3, 1);
    ml::solve_tri_backward(L, x, std::array<double, 3>{1, 1, 1}, dL, db, 3, 1);
    //! [tensor_ops]

    std::printf("  embedding t=10: cos %.4f %.4f %.4f, sin %.4f %.4f %.4f\n", static_cast<double>(embedding[6]),
                static_cast<double>(embedding[7]), static_cast<double>(embedding[8]), static_cast<double>(embedding[9]),
                static_cast<double>(embedding[10]), static_cast<double>(embedding[11]));
    ok &= embedding[0] == 1 && embedding[3] == 0 && embedding[6] == std::cos(10.0f) && embedding[9] == std::sin(10.0f);
    std::printf("  arange %zu values, cumsum %.2f %.2f %.2f %.2f, backward %.0f %.0f %.0f %.0f\n", steps.size(),
                static_cast<double>(sums[0]), static_cast<double>(sums[1]), static_cast<double>(sums[2]), static_cast<double>(sums[3]),
                static_cast<double>(back[0]), static_cast<double>(back[1]), static_cast<double>(back[2]), static_cast<double>(back[3]));
    ok &= steps.size() == 4 && sums[3] == 1.5f && back[0] == 4 && back[3] == 1;
    std::printf("  argsort descending %zu %zu %zu %zu %zu; top-2 keeps tokens 1 and 3: %s\n", order[0], order[1], order[2], order[3],
                order[4], std::isinf(kept[0]) && kept[1] == 2 && kept[3] == 2 && std::isinf(kept[4]) ? "yes" : "no");
    ok &= order == std::array<std::size_t, 5>{1, 3, 4, 0, 2} && kept[1] == 2 && std::isinf(kept[4]);
    std::printf("  causal mask row 0: %.0f %.0f %.0f\n", static_cast<double>(mask[0]), static_cast<double>(mask[1]), static_cast<double>(mask[2]));
    ok &= mask[0] == 0 && std::isinf(mask[1]) && mask[8] == 0;
    std::printf("  solve: x = %.1f %.1f %.1f, dB = %.1f %.1f %.1f\n", x[0], x[1], x[2], db[0], db[1], db[2]);
    ok &= x == std::array<double, 3>{1, 2, 3} && db == std::array<double, 3>{0, 0, 1} && dL[8] == -3;

    std::printf(ok ? "tensor_ops demo: ok\n" : "tensor_ops demo: FAILED\n");
    return ok ? 0 : 1;
}
