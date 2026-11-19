//
//  attention_sink_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/ml/attention_sink.hpp>: a head with a learnable sink logit, the share of attention the
//  sink takes, keys and values of different widths, no sink as plain attention, and the backward pass with the sink's
//  gradient. Exits non-zero if a result is not what it documents.
//

#include <snicholls/core/ml/attention.hpp>
#include <snicholls/core/ml/attention_sink.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <limits>

int main()
{
    std::printf("snicholls-core attention sink demo\n");
    namespace ml = snicholls::ml;
    bool ok = true;

    //! [attention_sink]
    // Keys of 4 channels, values of 2: four positions whose value is their position
    constexpr std::size_t dk = 4, dv = 2, n = 4;
    const std::array<float, dk> q{0.2f, -0.1f, 0.4f, 0.3f};
    std::array<float, n * dk> keys{};
    std::array<float, n * dv> values{};
    for (std::size_t j = 0; j < n; ++j) {
        for (std::size_t i = 0; i < dk; ++i)
            keys[j * dk + i] = std::cos(static_cast<float>(j + i));
        values[j * dv] = values[j * dv + 1] = static_cast<float>(j);
    }
    const float scale = 0.5f, sink = 1.0f;   // the head's learned sink logit (not scaled)
    std::array<float, dv> out{};
    const float lse = ml::attend_with_sink(q, keys, values, out, scale, sink);
    const float to_sink = ml::attention_sink_weight(lse, sink);   // the attention that went nowhere

    // Without a sink (-infinity) it is plain attention
    std::array<float, dv> plain{};
    ml::attend_with_sink(q, keys, values, plain, scale, -std::numeric_limits<float>::infinity());

    // Training: the sink has a gradient like any weight
    const std::array<float, dv> dout{1.0f, 0.0f};
    std::array<float, dk> dq{};
    std::array<float, n * dk> dkeys{};
    std::array<float, n * dv> dvalues{};
    const float dsink = ml::attend_with_sink_backward(q, keys, values, out, dout, lse, scale, sink, dq, dkeys, dvalues);
    //! [attention_sink]

    std::printf("  with the sink: mean position %.4f, the sink took %.1f%%\n", static_cast<double>(out[0]), 100.0 * static_cast<double>(to_sink));
    std::printf("  without:       mean position %.4f\n", static_cast<double>(plain[0]));
    std::printf("  d loss / d sink = %.6f\n", static_cast<double>(dsink));
    // The sink shrinks the output by its share: out = (1 - share) · plain
    ok &= std::fabs(out[0] - (1.0f - to_sink) * plain[0]) < 1e-5f && to_sink > 0.0f && to_sink < 1.0f;
    ok &= std::fabs(dsink + to_sink * out[0]) < 1e-6f;   // -share · (dout · out)
    std::printf(ok ? "attention sink demo: ok\n" : "attention sink demo: FAILED\n");
    return ok ? 0 : 1;
}
