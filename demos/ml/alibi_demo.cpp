//
//  alibi_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/ml/alibi.hpp>: the slopes of an eight-head and a twelve-head layer, built by the compiler,
//  and one query attending with ALiBi to four keys, forward and backward. Exits non-zero if a result is not what it
//  documents.
//

#include <snicholls/core/ml/alibi.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>

int main()
{
    std::printf("snicholls-core alibi demo\n");
    namespace ml = snicholls::ml;
    bool ok = true;

    //! [alibi]
    // Each head's slope, at compile time: a geometric series, 1/2 to 1/256 for eight heads
    constexpr auto slopes = ml::alibi_slopes<float, 8>();
    constexpr auto twelve = ml::alibi_slopes<float, 12>();   // not a power of two: the odd powers of the next series

    // One query at position 3 against the keys at positions 0 to 3, every key the same, so only the bias tells them apart
    const std::array<float, 2> q{1.0f, 0.0f};
    const std::array<float, 8> keys{1, 0, 1, 0, 1, 0, 1, 0}, values{1, 0, 2, 0, 3, 0, 4, 0};
    std::array<float, 2> out{};
    const float lse = ml::attend_alibi(q, keys, values, out, 1.0f, slopes[0], 3, 0);   // the steepest head: nearer keys weigh more

    // The backward pass, for a gradient of 1 in the output's first channel
    std::array<float, 2> dq{};
    std::array<float, 8> dkeys{}, dvalues{};
    ml::attend_alibi_backward(q, keys, values, out, std::array<float, 2>{1.0f, 0.0f}, lse, 1.0f, slopes[0], 3, 0, dq, dkeys, dvalues);
    //! [alibi]

    std::printf("  slopes (8 heads): %.4f %.4f ... %.6f; 12 heads, head 8: %.4f\n", static_cast<double>(slopes[0]),
                static_cast<double>(slopes[1]), static_cast<double>(slopes[7]), static_cast<double>(twelve[8]));
    ok &= slopes[0] == 0.5f && std::abs(slopes[7] - 1.0f / 256.0f) < 1e-9f && std::abs(twelve[8] - std::sqrt(0.5f)) < 1e-7f;
    // weights e^(-0.5 · distance): 1, e^-0.5, e^-1, e^-1.5 for the keys at 3, 2, 1, 0
    const float w[4] = {std::exp(-1.5f), std::exp(-1.0f), std::exp(-0.5f), 1.0f};
    const float want = (1 * w[0] + 2 * w[1] + 3 * w[2] + 4 * w[3]) / (w[0] + w[1] + w[2] + w[3]);
    std::printf("  output %.5f (by hand %.5f); d value of the nearest key %.5f\n", static_cast<double>(out[0]), static_cast<double>(want),
                static_cast<double>(dvalues[6]));
    ok &= std::abs(out[0] - want) < 1e-6f && std::abs(dvalues[6] - w[3] / (w[0] + w[1] + w[2] + w[3])) < 1e-6f;
    ok &= ml::alibi_bias(0.5f, 3, 0) == -1.5f && ml::alibi_slope<float>(0, 8) == slopes[0];

    std::printf(ok ? "alibi demo: ok\n" : "alibi demo: FAILED\n");
    return ok ? 0 : 1;
}
