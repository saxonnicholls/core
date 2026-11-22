//
//  quant_k_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/ml/quant_k.hpp>: a row of weights quantised to each K-quant as llama.cpp stores it,
//  with and without an importance row, its error and its dot product with activations quantised to Q8_K against
//  the exact one. Exits non-zero if a result is not what it documents.
//

#include <snicholls/core/ml/quant_k.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <span>
#include <vector>

namespace {

namespace ml = snicholls::ml;

bool ok = true;

template <typename B>
void show(const std::vector<float>& w, const std::vector<float>& a, const std::vector<float>& importance, double exact, float worst)
{
    //! [quant_k]
    // The weights as K-quant blocks: the reference quantisation, and the importance-weighted one llama-quantize
    // runs with an imatrix (one weight per column: how much each input matters)
    std::vector<B> q(w.size() / B::elements), qi(w.size() / B::elements);
    ml::quantize(w, std::span<B>(q));
    ml::quantize(w, importance, std::span<B>(qi));

    // The activations as Q8_K, then the integer dot product ggml runs
    std::vector<ml::block_q8_k> a8(a.size() / 256);
    ml::quantize(a, std::span<ml::block_q8_k>(a8));
    const float d = ml::dot(std::span<const B>(q), std::span<const ml::block_q8_k>(a8));
    //! [quant_k]

    std::vector<float> back(w.size()), backi(w.size());
    ml::dequantize(std::span<const B>(q), back);
    ml::dequantize(std::span<const B>(qi), backi);
    double err = 0, erri = 0;
    for (std::size_t i = 0; i < w.size(); ++i) {
        err += double(importance[i]) * double(back[i] - w[i]) * double(back[i] - w[i]);
        erri += double(importance[i]) * double(backi[i] - w[i]) * double(backi[i] - w[i]);
    }
    std::printf("  %-5s %6.4f bits/weight  w.a %9.5f (exact %9.5f)  weighted error %.3e, with the importance row %.3e\n",
                ml::block_traits<B>::name.data(), ml::block_traits<B>::bits_per_weight, double(d), exact, err, erri);
    ok &= std::fabs(double(d) - exact) < worst && erri <= err;
}

} // namespace

int main()
{
    std::printf("snicholls-core quant_k demo\n");
    std::vector<float> w(1024), a(1024), importance(1024);
    for (std::size_t i = 0; i < w.size(); ++i) {
        w[i] = 0.05f * std::sin(0.37f * static_cast<float>(i)) + 0.01f * std::cos(2.1f * static_cast<float>(i));
        a[i] = std::cos(0.3f * static_cast<float>(i));
        importance[i] = i % 5 == 0 ? 20.0f : 1.0f;
    }
    double exact = 0;
    for (std::size_t i = 0; i < w.size(); ++i)
        exact += double(w[i]) * double(a[i]);
    show<ml::block_q2_k>(w, a, importance, exact, 0.5f);
    show<ml::block_q3_k>(w, a, importance, exact, 0.2f);
    show<ml::block_q4_k>(w, a, importance, exact, 0.1f);
    show<ml::block_q5_k>(w, a, importance, exact, 0.05f);
    show<ml::block_q6_k>(w, a, importance, exact, 0.05f);
    std::printf(ok ? "quant_k demo: ok\n" : "quant_k demo: FAILED\n");
    return ok ? 0 : 1;
}
