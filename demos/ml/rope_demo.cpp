//
//  rope_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/ml/rope.hpp>: a table of cosines and sines built by the compiler, a query and a key
//  rotated in both layouts, the score depending only on their distance, and the inverse rotation as the backward
//  pass. Exits non-zero if a result is not what it documents.
//

#include <snicholls/core/ml/rope.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <span>

namespace {

namespace ml = snicholls::ml;

//! [rope]
// A context of 16 positions and heads of 8 channels: the compiler builds the whole table, in float as a kernel reads it
constexpr ml::rope_table<float, 16, 8> table{};
static_assert(table.cos[0][0] == 1.0f && table.sin[0][0] == 0.0f);   // position 0 does not rotate

float score(std::array<float, 8> q, std::size_t pq, std::array<float, 8> k, std::size_t pk, ml::rope_layout layout)
{
    ml::rope(q, table.cos_at(pq), table.sin_at(pq), layout);
    ml::rope(k, table.cos_at(pk), table.sin_at(pk), layout);
    float s = 0.0f;
    for (std::size_t i = 0; i < 8; ++i)
        s += q[i] * k[i];
    return s;
}
//! [rope]

} // namespace

int main()
{
    std::printf("snicholls-core rope demo\n");
    bool ok = true;
    const std::array<float, 8> q{0.5f, -1.0f, 0.25f, 2.0f, 1.0f, 0.0f, -0.5f, 0.75f};
    const std::array<float, 8> k{1.0f, 0.5f, -0.25f, 0.5f, -1.0f, 1.5f, 0.0f, 0.25f};

    for (const auto layout : {ml::rope_layout::half, ml::rope_layout::interleaved}) {
        const float near = score(q, 5, k, 3, layout), far = score(q, 14, k, 12, layout);
        std::printf("  %-11s q@5.k@3 = %.6f, q@14.k@12 = %.6f\n", layout == ml::rope_layout::half ? "half" : "interleaved",
                    static_cast<double>(near), static_cast<double>(far));
        ok &= std::fabs(near - far) < 1e-5f;
    }

    // Rotating at run time without a table, then back: the inverse is also the backward pass
    std::array<double, 4> x{1.0, 2.0, 3.0, 4.0};
    std::array<double, 2> c{}, s{};
    ml::rope_angles(1000, c, s, 500000.0);
    ml::rope(x, c, s);
    std::printf("  [1 2 3 4] at position 1000: [%.4f %.4f %.4f %.4f]\n", x[0], x[1], x[2], x[3]);
    ml::rope_inverse(x, c, s);
    ok &= std::fabs(x[0] - 1.0) < 1e-12 && std::fabs(x[3] - 4.0) < 1e-12;
    std::printf("  frequency of pair 1 of 4 at base 10000: %.4f\n", ml::rope_frequency<double>(1, 4));
    ok &= std::fabs(ml::rope_frequency<double>(1, 4) - 0.01) < 1e-15;

    std::printf(ok ? "rope demo: ok\n" : "rope demo: FAILED\n");
    return ok ? 0 : 1;
}
