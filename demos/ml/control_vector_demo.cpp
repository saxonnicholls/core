//
//  control_vector_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/ml/control_vector.hpp>: two control vectors combined at strengths, applied to the residual
//  rows after layers 2 and 3 only, and the gradient of a steering direction. Exits non-zero if a result is not what it
//  documents.
//

#include <snicholls/core/ml/control_vector.hpp>

#include <array>
#include <cstdint>
#include <cstdio>
#include <span>

int main()
{
    std::printf("snicholls-core control_vector demo\n");
    namespace ml = snicholls::ml;
    bool ok = true;

    //! [control_vector]
    // llama.cpp's layout: width 2, layers 1 to 3 (layer 0 has no entry)
    const std::array<float, 6> happy{0, 0, 1, 1, 2, 2}, formal{0, 0, 1, -1, 0, 0};
    std::array<float, 6> combined{};
    ml::add_control_vector(std::span<float>(combined), happy, 0.5f);
    ml::add_control_vector(std::span<float>(combined), formal, 1.0f);
    const ml::control_vector_view<float> cv{combined, 2, 2, 3};   // applied after layers 2 and 3

    std::array<float, 4> residual{1, 1, 1, 1};   // two tokens
    for (std::int32_t il = 0; il < 4; ++il)
        ml::apply_control_vector(cv, il, std::span<float>(residual));   // between block_forward calls

    // Training a steering direction: its gradient is the sum of the rows' gradients
    std::array<float, 2> ddirection{};
    ml::control_vector_backward(cv, 2, std::array<float, 4>{1, 0, 1, 0}, std::span<float>(ddirection));
    //! [control_vector]

    std::printf("  combined layer 2 [%.1f %.1f], residual after layer 3 [%.1f %.1f]\n", static_cast<double>(combined[2]),
                static_cast<double>(combined[3]), static_cast<double>(residual[0]), static_cast<double>(residual[1]));
    ok &= combined[2] == 1.5f && combined[3] == -0.5f;
    ok &= residual[0] == 1.0f + 1.5f + 1.0f && residual[1] == 1.0f - 0.5f + 1.0f && residual[2] == residual[0];
    ok &= ddirection[0] == 2.0f && ddirection[1] == 0.0f && cv.direction(1).empty();

    std::printf(ok ? "control_vector demo: ok\n" : "control_vector demo: FAILED\n");
    return ok ? 0 : 1;
}
