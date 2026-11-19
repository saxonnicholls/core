//
//  probe_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/ml/probe.hpp>: frozen features of two sequences, one padded, pooled by their mean and by an
//  attentive probe (one learned query, two heads), and one gradient step of the probe that lowers a squared error.
//  Exits non-zero if a result is not what it documents.
//

#include <snicholls/core/ml/probe.hpp>

#include <snicholls/core/linalg/reference.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <span>
#include <vector>

int main()
{
    std::printf("snicholls-core probe demo\n");
    namespace ml = snicholls::ml;
    const auto& backend = snicholls::linalg::reference;
    bool ok = true;
    constexpr std::size_t d = 4, n = 3, seqs = 2, heads = 2;

    //! [probe]
    // Two sequences of three feature rows; the second has two real rows and one of padding
    std::vector<float> features(seqs * n * d);
    for (std::size_t i = 0; i < features.size(); ++i)
        features[i] = std::sin(0.7f * float(i));
    const std::array<std::size_t, 2> lengths{3, 2};
    // The mean of each sequence's real rows
    std::array<float, seqs * d> pooled{};
    ml::mean_pool_rows(backend, std::span<const float>(features), d, std::span<float>(pooled), seqs, lengths);

    // The attentive probe: a learned query attends over the rows (V-JEPA's pooler), out = query + attention
    std::vector<float> query(d, 0.1f), wq(d * d), wk(d * d), wv(d * d);
    for (std::size_t i = 0; i < d * d; ++i) {
        wq[i] = 0.3f * std::cos(float(i));
        wk[i] = 0.3f * std::sin(float(i) + 1);
        wv[i] = i % (d + 1) == 0 ? 1.0f : 0.0f;   // the identity
    }
    ml::attentive_probe_weights<float> w{query, {wq, d, d}, {wk, d, d}, {wv, d, d}};
    std::vector<float> h(features.size()), k(features.size()), v(features.size()), q(d), y(seqs * d), lse(seqs * heads);
    const ml::attentive_probe_activations<float> act{h, k, v, q, y, lse};
    std::array<float, seqs * d> out{};
    ml::attentive_probe_forward(backend, w, std::span<const float>(features), act, std::span<float>(out), heads, seqs, lengths);
    //! [probe]
    std::printf("  mean pooled: %.3f %.3f | attentive: %.3f %.3f\n", double(pooled[0]), double(pooled[d]), double(out[0]), double(out[d]));
    ok &= std::abs(pooled[d] - (features[n * d] + features[(n + 1) * d]) / 2) < 1e-6f;

    // A squared error to a target of zero, its gradient through the probe, and one step on the query
    auto loss = [&](const std::array<float, seqs * d>& o) {
        float s = 0;
        for (const float x : o)
            s += x * x;
        return s;
    };
    const float before = loss(out);
    std::array<float, seqs * d> dout{};
    for (std::size_t i = 0; i < dout.size(); ++i)
        dout[i] = 2 * out[i];
    std::vector<float> dquery(d), dwq(d * d), dwk(d * d), dwv(d * d), dq(d), dk(features.size()), dv(features.size()), dh(features.size()),
        dfeatures(features.size());
    ml::attentive_probe_backward(backend, w, std::span<const float>(features), act, std::span<const float>(dout), std::span<float>(dfeatures),
                                 ml::attentive_probe_gradients<float>{dquery, dwq, dwk, dwv, {}, {}, {}, {}, {}, dq, dk, dv, dh}, heads, seqs, lengths);
    for (std::size_t i = 0; i < d; ++i)
        query[i] -= 0.05f * dquery[i];
    std::vector<float> dpool(features.size());
    ml::mean_pool_rows_backward(backend, std::span<const float>(dout), d, std::span<float>(dpool), seqs, lengths);
    ml::attentive_probe_forward(backend, w, std::span<const float>(features), act, std::span<float>(out), heads, seqs, lengths);
    std::printf("  squared error %.4f, after one step on the query %.4f; padding's gradient %g\n", double(before), double(loss(out)),
                double(dfeatures[(n + 2) * d] + dpool[(n + 2) * d]));
    ok &= loss(out) < before && dfeatures[(n + 2) * d] == 0 && dpool[(n + 2) * d] == 0 && dpool[0] == dout[0] / 3;

    std::printf(ok ? "probe demo: ok\n" : "probe demo: FAILED\n");
    return ok ? 0 : 1;
}
