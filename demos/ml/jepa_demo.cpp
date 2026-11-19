//
//  jepa_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/ml/jepa.hpp>: the size of I-JEPA's predictor; the predictor's input built from two
//  samples' context rows and mask tokens, and its backward; the normalised targets and the four losses with their
//  gradients, LLM-JEPA's cosine term on two views; the target encoder following the context encoder by EMA, and the
//  momentum schedule. Exits non-zero if a result is not what it documents.
//

#include <snicholls/core/ml/jepa.hpp>

#include <snicholls/core/linalg/reference.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>

int main()
{
    std::printf("snicholls-core jepa demo\n");
    namespace ml = snicholls::ml;
    const auto& backend = snicholls::linalg::reference;
    bool ok = true;

    // I-JEPA's ViT-H/14 predictor: 12 blocks of 384
    constexpr std::size_t predictor = ml::jepa_predictor_parameters(ml::ijepa_vit_h14_predictor);
    std::printf("  I-JEPA ViT-H/14 predictor: %.1f M parameters (a block %.2f M)\n", double(predictor) / 1e6,
                double(ml::block_parameter_count(ml::ijepa_vit_h14_predictor.block)) / 1e6);
    ok &= ml::valid(ml::ijepa_vit_h14_predictor) && predictor > 20'000'000 && predictor < 23'000'000;

    //! [predictor_input]
    // Two samples, each with 2 context rows and 1 target, in a predictor 2 wide; a grid of 4 tokens' positions
    const std::array<float, 2 * 2 * 2> context{1, 1, 2, 2, 3, 3, 4, 4};   // the context encoder's rows, projected
    const std::array<float, 2> mask_token{0.5f, -0.5f};
    const std::array<float, 4 * 2> positions{0, 0, 10, 10, 20, 20, 30, 30};
    const std::array<std::uint32_t, 4> context_index{0, 3, 1, 2};
    const std::array<std::uint32_t, 2> target_index{1, 0};
    std::array<float, 2 * 3 * 2> rows{};   // each sample: its context rows, then its mask token at its target's position
    ml::jepa_predictor_input_rows(backend, std::span<const float>(context), std::span<const float>(mask_token), std::span<const float>(positions),
                                  context_index, target_index, std::span<float>(rows), 2, 2);
    // The backward: the context rows' gradients, and the mask token's summed over every target
    std::array<float, 2 * 3 * 2> drows{};
    drows.fill(1.0f);
    std::array<float, 2 * 2 * 2> dcontext{};
    std::array<float, 2> dmask{};
    ml::jepa_predictor_input_rows_backward(backend, std::span<const float>(drows), 2, 1, std::span<float>(dcontext), std::span<float>(dmask), 2, 2);
    //! [predictor_input]
    std::printf("  predictor rows: (%g %g) (%g %g) (%g %g) | (%g %g) (%g %g) (%g %g)\n", double(rows[0]), double(rows[1]), double(rows[2]),
                double(rows[3]), double(rows[4]), double(rows[5]), double(rows[6]), double(rows[7]), double(rows[8]), double(rows[9]),
                double(rows[10]), double(rows[11]));
    ok &= rows[2] == 32 && rows[4] == 10.5f && rows[6] == 13 && rows[10] == 0.5f && dmask[0] == 2 && dcontext[7] == 1;

    //! [loss]
    // Targets: the target encoder's rows normalised without parameters (no gradient flows into them)
    const std::array<float, 2 * 4> raw{1, 2, 3, 4, -2, 0, 2, 4};
    std::array<float, 2 * 4> targets{};
    ml::jepa_target_norm_rows(backend, std::span<const float>(raw), std::span<float>(targets), 4);
    const std::array<float, 2 * 4> predictions{-1.0f, -0.5f, 0.5f, 1.5f, -1.5f, -0.5f, 0.5f, 1.5f};
    std::array<float, 2 * 4> dpred{};
    const float l2 = ml::jepa_loss_rows(backend, ml::jepa_loss::l2, std::span<const float>(predictions), std::span<const float>(targets), 4,
                                        std::span<float>(dpred));
    const float l1 = ml::jepa_loss_rows(backend, ml::jepa_loss::l1, std::span<const float>(predictions), std::span<const float>(targets), 4, std::span<float>{});
    const float smooth = ml::jepa_loss_rows(backend, ml::jepa_loss::smooth_l1, std::span<const float>(predictions), std::span<const float>(targets), 4, std::span<float>{});
    // LLM-JEPA: the cosine distance between two views' embeddings, weighted by lambda, both views taking gradients
    std::array<float, 2 * 4> dview1{}, dview2{};
    const float cosine = ml::jepa_loss_rows(backend, ml::jepa_loss::cosine, std::span<const float>(predictions), std::span<const float>(raw), 4,
                                            std::span<float>(dview1), std::span<float>(dview2), {}, 2.0f);
    //! [loss]
    std::printf("  losses: L2 %.4f, L1 %.4f, smooth L1 %.4f, 2 x cosine distance %.4f\n", double(l2), double(l1), double(smooth), double(cosine));
    ok &= std::abs(targets[0] + targets[3]) < 1e-6f && l2 < l1 && smooth < l1 && smooth > 0 && cosine > 0 && cosine < 1 && dview2[0] != 0;

    //! [ema]
    // The target encoder follows the context encoder: target = m target + (1 - m) context, m from 0.996 to 1
    std::array<float, 3> target_weights{1, 1, 1};
    std::array<float, 2> other{0, 0};
    const std::array<float, 3> context_weights{0, 2, 4};
    const std::array<float, 2> other_context{8, 8};
    const std::array<ml::ema_segment<float>, 2> model{{{target_weights, context_weights}, {other, other_context}}};
    const double m0 = ml::ema_momentum(0, 1000), m_half = ml::ema_momentum(500, 1000),
                 m_cos = ml::ema_momentum(500, 1000, 0.996, 1.0, ml::momentum_schedule::cosine);
    ml::ema_segments(backend, std::span<const ml::ema_segment<float>>(model), 0.5);
    ml::ema_rows(backend, std::span<float>(target_weights), std::span<const float>(context_weights), 0.0);   // m = 0 copies
    //! [ema]
    std::printf("  momentum %.4f at step 0, %.4f at 500 of 1000 (cosine %.4f); EMA at 0.5 of 0 and 8: %g\n", m0, m_half, m_cos, double(other[0]));
    ok &= m0 == 0.996 && std::abs(m_half - 0.998) < 1e-15 && std::abs(m_cos - 0.998) < 1e-12 && other[0] == 4 && target_weights == context_weights;

    std::printf(ok ? "jepa demo: ok\n" : "jepa demo: FAILED\n");
    return ok ? 0 : 1;
}
