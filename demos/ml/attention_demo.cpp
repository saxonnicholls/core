//
//  attention_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/ml/attention.hpp>: one query over a cache of keys and values in one pass, the same
//  split across two workers and merged, a sliding window, and the backward pass. Exits non-zero if a result is
//  not what it documents.
//

#include <snicholls/core/ml/attention.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <span>

int main()
{
    std::printf("snicholls-core attention demo\n");
    namespace ml = snicholls::ml;
    bool ok = true;

    //! [attention]
    // A head of 4 channels and a cache of 6 positions, keys and values end to end
    constexpr std::size_t d = 4, n = 6;
    const std::array<float, d> q{1.0f, 0.5f, -0.5f, 0.0f};
    std::array<float, n * d> keys{}, values{};
    for (std::size_t j = 0; j < n; ++j)
        for (std::size_t i = 0; i < d; ++i) {
            keys[j * d + i] = std::sin(static_cast<float>(j * d + i));
            values[j * d + i] = static_cast<float>(j);   // value j is all j: the output is the mean position attended
        }
    const float scale = 1.0f / std::sqrt(static_cast<float>(d));

    // One pass: no row of scores is stored
    std::array<float, d> out{};
    const float lse = ml::attend(q, keys, values, out, scale);

    // Two workers take half the cache each and merge: the same output
    ml::attention_accumulator<float, d> a, b;
    for (std::size_t j = 0; j < n; ++j) {
        float s = 0.0f;
        for (std::size_t i = 0; i < d; ++i)
            s += q[i] * keys[j * d + i];
        (j < n / 2 ? a : b).add(scale * s, std::span<const float>(values).subspan(j * d, d));
    }
    a.merge(b);
    std::array<float, d> merged{};
    a.result(merged);

    // A sliding window of 2 at position 5: only keys 4 and 5
    const ml::key_range w = ml::attention_window(5, 2);
    std::array<float, d> windowed{};
    ml::attend(q, std::span<const float>(keys).subspan(w.first * d, w.size() * d),
               std::span<const float>(values).subspan(w.first * d, w.size() * d), windowed, scale);
    //! [attention]

    std::printf("  one pass: mean position %.4f, logsumexp %.4f\n", static_cast<double>(out[0]), static_cast<double>(lse));
    std::printf("  merged:   mean position %.4f\n", static_cast<double>(merged[0]));
    std::printf("  window [%zu, %zu): mean position %.4f\n", w.first, w.last, static_cast<double>(windowed[0]));
    ok &= std::fabs(out[0] - merged[0]) < 1e-5f && std::fabs(a.log_sum_exp() - lse) < 1e-5f;
    ok &= out[0] > 0.0f && out[0] < 5.0f && windowed[0] >= 4.0f && windowed[0] <= 5.0f;

    // The backward pass: dq written, dkeys and dvalues accumulated. The gradient of the mean position in the values is the weights.
    std::array<float, d> dout{1.0f, 0.0f, 0.0f, 0.0f}, dq{};
    std::array<float, n * d> dkeys{}, dvalues{};
    ml::attend_backward(q, keys, values, out, dout, lse, scale, dq, dkeys, dvalues);
    float weights = 0.0f;
    for (std::size_t j = 0; j < n; ++j)
        weights += dvalues[j * d];
    std::printf("  backward: the weights sum to %.6f\n", static_cast<double>(weights));
    ok &= std::fabs(weights - 1.0f) < 1e-5f;

    std::printf(ok ? "attention demo: ok\n" : "attention demo: FAILED\n");
    return ok ? 0 : 1;
}
