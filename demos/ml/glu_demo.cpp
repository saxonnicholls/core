//
//  glu_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/ml/glu.hpp>: every gated linear unit over one gate and up row, gpt-oss's clamped SwiGLU
//  at its limits, the single-row split form ggml uses, and the backward pass. Exits non-zero if a result is not what
//  it documents.
//

#include <snicholls/core/ml/glu.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>

int main()
{
    std::printf("snicholls-core glu demo\n");
    namespace ml = snicholls::ml;
    bool ok = true;

    //! [glu]
    // A gate row and an up row of four, through each gated linear unit
    const std::array<float, 4> gate{-1.0f, 0.5f, 2.0f, 9.0f}, up{1.0f, -2.0f, 0.5f, 3.0f};
    constexpr std::array kinds{ml::glu_kind::reglu,  ml::glu_kind::geglu,      ml::glu_kind::geglu_erf,   ml::glu_kind::geglu_quick,
                               ml::glu_kind::swiglu, ml::glu_kind::swiglu_oai, ml::glu_kind::swiglu_clamp};
    std::array<std::array<float, 4>, kinds.size()> out{};
    for (std::size_t k = 0; k < kinds.size(); ++k)
        ml::glu_rows(kinds[k], gate, up, out[k]);   // gpt-oss's alpha 1.702 and limit 7 by default

    // ggml's single-row form: one row of 2n, the gate first (or the up projection first when swapped)
    const std::array<float, 8> both{-1.0f, 0.5f, 2.0f, 9.0f, 1.0f, -2.0f, 0.5f, 3.0f};
    std::array<float, 4> split{}, swapped{};
    ml::glu_split(ml::glu_kind::swiglu, both, split);
    ml::glu_split(ml::glu_kind::swiglu, both, swapped, true);

    // The backward pass: the gradients in the gate and the up rows, for a gradient of 1 in each output
    const std::array<float, 4> dout{1.0f, 1.0f, 1.0f, 1.0f};
    std::array<float, 4> dgate{}, dup{};
    ml::glu_rows_backward(ml::glu_kind::swiglu_oai, gate, up, dout, dgate, dup);
    std::array<float, 8> dboth{};
    ml::glu_split_backward(ml::glu_kind::swiglu, both, dout, dboth);
    //! [glu]

    const char* names[] = {"reglu", "geglu", "geglu_erf", "geglu_quick", "swiglu", "swiglu_oai", "swiglu_clamp"};
    for (std::size_t k = 0; k < kinds.size(); ++k)
        std::printf("  %-12s %9.5f %9.5f %9.5f %9.5f\n", names[k], static_cast<double>(out[k][0]), static_cast<double>(out[k][1]),
                    static_cast<double>(out[k][2]), static_cast<double>(out[k][3]));
    ok &= out[0][0] == 0.0f && out[0][2] == 1.0f;                                   // relu(-1) 1, relu(2) 0.5
    ok &= out[4][3] == ml::silu(9.0f) * 3.0f && out[6][3] == ml::silu(7.0f) * 3.0f;   // the clamp caps the gate at 7
    ok &= out[5][3] == ml::glu(ml::glu_kind::swiglu_oai, 7.0f, 3.0f);                 // gpt-oss: min(9, 7)
    ok &= std::abs(out[5][1] - 0.5f / (1.0f + std::exp(-0.851f)) * (-2.0f + 1.0f)) < 1e-6f;
    std::printf("  split %9.5f %9.5f, swapped %9.5f %9.5f\n", static_cast<double>(split[0]), static_cast<double>(split[1]),
                static_cast<double>(swapped[0]), static_cast<double>(swapped[1]));
    ok &= split == out[4] && swapped[0] == ml::silu(1.0f) * -1.0f;
    std::printf("  swiglu_oai backward: dgate %9.5f %9.5f %9.5f %9.5f, dup %9.5f\n", static_cast<double>(dgate[0]),
                static_cast<double>(dgate[1]), static_cast<double>(dgate[2]), static_cast<double>(dgate[3]), static_cast<double>(dup[0]));
    ok &= dgate[3] == 0.0f;   // a gate past the limit gets no gradient
    ok &= dboth[4] == ml::silu(-1.0f) && dboth[0] == 1.0f * ml::silu_derivative(-1.0f);

    std::printf(ok ? "glu demo: ok\n" : "glu demo: FAILED\n");
    return ok ? 0 : 1;
}
