//
//  probe_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/ml/probe.hpp>: mean_pool_rows against a loop by hand to the bit, with lengths, and its
//  backward; the attentive probe against the cross-attention written out (layer norm, projections, a softmax per head,
//  the residual), padded rows ignored, and every gradient by central differences in double.
//

#include <snicholls/core/ml/probe.hpp>

#include <snicholls/core/linalg/reference.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

namespace {

namespace ml = snicholls::ml;
const auto& backend = snicholls::linalg::reference;

std::vector<double> wave(std::size_t n, double a, double b)
{
    std::vector<double> v(n);
    for (std::size_t i = 0; i < n; ++i)
        v[i] = a * std::sin(0.41 * double(i) + b);
    return v;
}

TEST(MlMeanPool, IsTheMeanOfTheValidRowsAndSpreadsItsGradient)
{
    constexpr std::size_t w = 3, n = 4, s = 2;
    const auto x = wave(s * n * w, 1.0, 0.2);
    const std::array<std::size_t, 2> lengths{4, 2};
    std::vector<double> out(s * w);
    ml::mean_pool_rows(backend, std::span<const double>(x), w, std::span<double>(out), s, lengths);
    for (std::size_t c = 0; c < w; ++c) {
        EXPECT_EQ(out[c], (((x[c] + x[w + c]) + x[2 * w + c]) + x[3 * w + c]) / 4.0);
        EXPECT_EQ(out[w + c], (x[n * w + c] + x[(n + 1) * w + c]) / 2.0);
    }
    std::vector<double> dx(x.size()), dout{1, 2, 3, 4, 6, 8};
    ml::mean_pool_rows_backward(backend, std::span<const double>(dout), w, std::span<double>(dx), s, lengths);
    EXPECT_EQ(dx[0], 0.25);
    EXPECT_EQ(dx[(n + 1) * w + 2], 4.0);
    EXPECT_EQ(dx[(n + 2) * w], 0.0);   // padding
    std::vector<double> all(w);
    ml::mean_pool_rows(backend, std::span<const double>(x).first(n * w), w, std::span<double>(all));
    EXPECT_EQ(all[1], out[1]);
}

constexpr std::size_t d = 6, heads = 2, n = 5, seqs = 2;

struct probe {
    std::vector<double> query = wave(d, 0.8, 0.1), wq = wave(d * d, 0.5, 0.3), wk = wave(d * d, 0.6, 1.3), wv = wave(d * d, 0.7, 2.3),
                        bq = wave(d, 0.1, 0.5), bk = wave(d, 0.2, 1.5), bv = wave(d, 0.1, 2.5), nw = wave(d, 0.3, 0.9), nb = wave(d, 0.2, 0.4);
    probe()
    {
        for (double& v : nw)
            v += 1.0;
    }
    [[nodiscard]] ml::attentive_probe_weights<double> weights() const
    {
        return {query, {wq, d, d}, {wk, d, d}, {wv, d, d}, bq, bk, bv, nw, nb, 1e-6};
    }
};

struct buffers {
    std::vector<double> h = std::vector<double>(seqs * n * d), k = h, v = h, q = std::vector<double>(d), y = std::vector<double>(seqs * d),
                        lse = std::vector<double>(seqs * heads), out = y;
    ml::attentive_probe_activations<double> act() { return {h, k, v, q, y, lse}; }
};

// The block written out in double
std::vector<double> reference(const probe& p, const std::vector<double>& x, std::span<const std::size_t> lengths)
{
    std::vector<double> out(seqs * d);
    std::vector<double> q(d);
    for (std::size_t i = 0; i < d; ++i) {
        q[i] = p.bq[i];
        for (std::size_t j = 0; j < d; ++j)
            q[i] += p.wq[i * d + j] * p.query[j];
    }
    for (std::size_t s = 0; s < seqs; ++s) {
        const std::size_t len = lengths.empty() ? n : lengths[s];
        std::vector<std::vector<double>> k(len, std::vector<double>(d)), v = k;
        for (std::size_t r = 0; r < len; ++r) {
            const double* xr = &x[(s * n + r) * d];
            double mu = 0, var = 0;
            for (std::size_t j = 0; j < d; ++j)
                mu += xr[j] / d;
            for (std::size_t j = 0; j < d; ++j)
                var += (xr[j] - mu) * (xr[j] - mu) / d;
            std::vector<double> h(d);
            for (std::size_t j = 0; j < d; ++j)
                h[j] = (xr[j] - mu) / std::sqrt(var + 1e-6) * p.nw[j] + p.nb[j];
            for (std::size_t i = 0; i < d; ++i) {
                k[r][i] = p.bk[i];
                v[r][i] = p.bv[i];
                for (std::size_t j = 0; j < d; ++j) {
                    k[r][i] += p.wk[i * d + j] * h[j];
                    v[r][i] += p.wv[i * d + j] * h[j];
                }
            }
        }
        const std::size_t dh = d / heads;
        for (std::size_t hd = 0; hd < heads; ++hd) {
            std::vector<double> sc(len);
            double mx = -1e300, sum = 0;
            for (std::size_t r = 0; r < len; ++r) {
                for (std::size_t i = 0; i < dh; ++i)
                    sc[r] += q[hd * dh + i] * k[r][hd * dh + i] / std::sqrt(double(dh));
                mx = std::max(mx, sc[r]);
            }
            for (std::size_t r = 0; r < len; ++r)
                sum += std::exp(sc[r] - mx);
            for (std::size_t i = 0; i < dh; ++i) {
                double y = 0;
                for (std::size_t r = 0; r < len; ++r)
                    y += std::exp(sc[r] - mx) / sum * v[r][hd * dh + i];
                out[s * d + hd * dh + i] = p.query[hd * dh + i] + y;
            }
        }
    }
    return out;
}

TEST(MlAttentiveProbe, IsTheCrossAttentionWrittenOutIgnoringPadding)
{
    const probe p;
    auto x = wave(seqs * n * d, 1.1, 0.7);
    const std::array<std::size_t, 2> lengths{5, 3};
    buffers b;
    ml::attentive_probe_forward(backend, p.weights(), std::span<const double>(x), b.act(), std::span<double>(b.out), heads, seqs, lengths);
    const auto want = reference(p, x, lengths);
    for (std::size_t i = 0; i < want.size(); ++i)
        EXPECT_NEAR(b.out[i], want[i], 1e-12) << i;
    // Padded rows change nothing
    for (std::size_t i = (n + 3) * d; i < 2 * n * d; ++i)
        x[i] = 100.0;
    buffers b2;
    ml::attentive_probe_forward(backend, p.weights(), std::span<const double>(x), b2.act(), std::span<double>(b2.out), heads, seqs, lengths);
    EXPECT_EQ(b2.out, b.out);
}

TEST(MlAttentiveProbe, BackwardMatchesCentralDifferences)
{
    probe p;
    auto x = wave(seqs * n * d, 1.1, 0.7);
    const auto gout = wave(seqs * d, 0.9, 1.9);
    const std::array<std::size_t, 2> lengths{5, 4};
    auto loss = [&] {
        buffers b;
        ml::attentive_probe_forward(backend, p.weights(), std::span<const double>(x), b.act(), std::span<double>(b.out), heads, seqs, lengths);
        double l = 0;
        for (std::size_t i = 0; i < b.out.size(); ++i)
            l += gout[i] * b.out[i];
        return l;
    };
    buffers b;
    ml::attentive_probe_forward(backend, p.weights(), std::span<const double>(x), b.act(), std::span<double>(b.out), heads, seqs, lengths);
    std::vector<double> dquery(d), dwq(d * d), dwk(d * d), dwv(d * d), dbq(d), dbk(d), dbv(d), dnw(d), dnb(d), dq(d), dk(x.size()), dv(x.size()),
        dh(x.size()), dx(x.size());
    ml::attentive_probe_backward(backend, p.weights(), std::span<const double>(x), b.act(), std::span<const double>(gout), std::span<double>(dx),
                                 ml::attentive_probe_gradients<double>{dquery, dwq, dwk, dwv, dbq, dbk, dbv, dnw, dnb, dq, dk, dv, dh}, heads, seqs,
                                 lengths);
    auto check = [&](std::vector<double>& param, const std::vector<double>& grad, const char* what) {
        for (std::size_t i = 0; i < param.size(); ++i) {
            const double keep = param[i], e = 1e-6;
            param[i] = keep + e;
            const double hi = loss();
            param[i] = keep - e;
            const double lo = loss();
            param[i] = keep;
            EXPECT_NEAR(grad[i], (hi - lo) / (2 * e), 2e-8) << what << " " << i;
        }
    };
    check(x, dx, "x");
    check(p.query, dquery, "query");
    check(p.wq, dwq, "wq");
    check(p.wk, dwk, "wk");
    check(p.wv, dwv, "wv");
    check(p.bq, dbq, "bq");
    check(p.bk, dbk, "bk");
    check(p.bv, dbv, "bv");
    check(p.nw, dnw, "norm_weight");
    check(p.nb, dnb, "norm_bias");
}

} // namespace
