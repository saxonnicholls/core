//
//  distill_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/ml/distill.hpp>: a student's logits pulled towards a teacher's at temperature 2 by the
//  forward KL (Hinton's T^2 weighting in the scale) over the response positions of a batch, the reverse KL beside it,
//  and the same with only the teacher's top-2 logits stored. Exits non-zero if a result is not what it documents.
//

#include <snicholls/core/ml/distill.hpp>

#include <snicholls/core/linalg/reference.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>

namespace ml = snicholls::ml;

int main()
{
    std::printf("snicholls-core distill demo\n");
    bool ok = true;
    const auto& backend = snicholls::linalg::reference;

    constexpr std::size_t rows = 3, vocab = 4;
    const std::array<double, rows * vocab> teacher{4, 1, 0, -2, /**/ 0, 3, 3, 0, /**/ -1, -1, 5, 0};
    std::array<double, rows * vocab> student{};   // starts uniform
    const std::array<std::uint8_t, rows> mask{0, 1, 1};   // position 0 is the prompt
    double first = 0, last = 0;
    for (int step = 0; step < 200; ++step) {
        //! [distill]
        // Forward KL(teacher || student) at T = 2 over the response positions, the gradient of the mean times T^2
        const ml::forward_kl kd{.temperature = 2.0};
        std::array<double, rows> losses{};
        std::array<double, rows * vocab> grad{};
        ml::distill_rows<double>(backend, kd, student, teacher, mask, losses, grad, 4.0 / 2.0);
        //! [distill]
        (step == 0 ? first : last) = (losses[1] + losses[2]) / 2;
        for (std::size_t i = 0; i < student.size(); ++i)
            student[i] -= 0.5 * grad[i];
    }
    std::printf("  forward KL at T = 2: %.5f -> %.2e; the prompt row untouched: %g\n", first, last, student[0]);
    ok &= last < 1e-4 && first > 0.1 && student[0] == 0.0;

    // One row: forward and reverse, and the gradient of the reverse
    std::array<double, vocab> g{};
    const auto t1 = std::span<const double>(teacher).subspan(vocab, vocab);
    const std::array<double, vocab> s0{};
    const double fwd = ml::distill_loss(ml::forward_kl{}, s0, t1, g);
    const double rev = ml::distill_loss(ml::reverse_kl{}, s0, t1, g);
    std::printf("  uniform student vs teacher row 1: forward %.4f, reverse %.4f\n", fwd, rev);
    ok &= fwd > 0 && rev > 0 && std::fabs(g[0] + g[1] + g[2] + g[3]) < 1e-15;

    // Storing the teacher's top-2 logits a position instead of the vocabulary
    std::array<std::uint32_t, 2> ids{};
    std::array<double, 2> values{};
    ml::teacher_topk(t1, ids, values);   // tokens 1 and 2, both 3: ties to the lower index
    const double topk = ml::distill_loss_topk(ml::forward_kl{}, s0, ids, values, g);
    std::printf("  top-2 of row 1: tokens %u %u; KL against them %.4f (log 2 = %.4f)\n", ids[0], ids[1], topk, std::log(2.0));
    ok &= ids[0] == 1 && ids[1] == 2 && std::fabs(topk - std::log(2.0)) < 1e-15;
    std::array<double, rows> losses{};
    std::array<double, rows * vocab> grads{};
    std::array<std::uint32_t, rows * 2> all_ids{};
    std::array<double, rows * 2> all_values{};
    for (std::size_t r = 0; r < rows; ++r)
        ml::teacher_topk(std::span<const double>(teacher).subspan(r * vocab, vocab), std::span<std::uint32_t>(all_ids).subspan(r * 2, 2),
                         std::span<double>(all_values).subspan(r * 2, 2));
    ml::distill_topk_rows<double>(backend, ml::forward_kl{}, std::array<double, rows * vocab>{}, all_ids, all_values, mask, losses, grads, 1.0);
    ok &= losses[0] == 0.0 && std::fabs(losses[1] - std::log(2.0)) < 1e-15;

    std::printf("%s\n", ok ? "ok" : "FAILED");
    return ok ? 0 : 1;
}
