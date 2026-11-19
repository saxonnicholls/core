//
//  quant_iq_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/ml/quant_iq.hpp>: one row of 1,024 weights quantised to every IQ format with an
//  importance row, as llama.cpp's quantiser does, its size, the error of each format, and the integer dot product
//  with activations quantised to Q8_K (Q8_0 for IQ4_NL) against the exact one; IQ4_NL's reference quantiser.
//  Exits non-zero if a result is not what it documents.
//

#include <snicholls/core/ml/quant_iq.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <vector>

namespace ml = snicholls::ml;

namespace {

constexpr std::size_t n = 1024;

std::vector<float> make(std::uint64_t s, float scale)
{
    std::vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i) {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        const float u = float(static_cast<std::int32_t>(s >> 40) - (1 << 23)) * 0x1p-23f;
        v[i] = scale * u * u * u;   // heavier tails than uniform, as trained weights
    }
    return v;
}

} // namespace

int main()
{
    std::printf("snicholls-core quant_iq demo\n");
    bool ok = true;

    const std::vector<float> w = make(1, 0.05f), a = make(2, 1.0f);
    std::vector<float> importance(n);   // an importance matrix row: the mean square of each column's activations
    for (std::size_t i = 0; i < n; ++i)
        importance[i] = 0.5f + float(i % 7) * 0.25f;

    double exact = 0, magnitude = 0;
    for (std::size_t i = 0; i < n; ++i) {
        exact += double(w[i]) * double(a[i]);
        magnitude += std::fabs(double(w[i]) * double(a[i]));
    }

    const auto run = [&](const char* name, auto block, double worst_rms) {
        using B = decltype(block);
        using A = typename B::dot_block;
        //! [iq]
        // Weights to an IQ format, minimising the importance-weighted error as llama-quantize does
        std::vector<B> wq(n / B::elements);
        ml::quantize(w, importance, std::span<B>(wq));

        // Activations to the format's partner (Q8_K, or Q8_0 for IQ4_NL), then the integer dot product
        std::vector<A> aq(n / A::elements);
        ml::quantize(a, std::span<A>(aq));
        const float d = ml::dot(std::span<const B>(wq), std::span<const A>(aq));

        // And back to floats
        std::vector<float> back(n);
        ml::dequantize(std::span<const B>(wq), back);
        //! [iq]
        double e2 = 0, s2 = 0;
        for (std::size_t i = 0; i < n; ++i) {
            e2 += (double(back[i]) - w[i]) * (double(back[i]) - w[i]);
            s2 += double(w[i]) * w[i];
        }
        const double rms = std::sqrt(e2 / s2);
        std::printf("  %-8s %5.3f bits/weight  relative error %.3f  w . a %.5f (exact %.5f)\n", name, ml::block_traits<B>::bits_per_weight, rms,
                    double(d), exact);
        ok &= rms < worst_rms && std::fabs(d - exact) < worst_rms * magnitude;
    };
    run("IQ1_S", ml::block_iq1_s{}, 0.65);
    run("IQ1_M", ml::block_iq1_m{}, 0.65);
    run("IQ2_XXS", ml::block_iq2_xxs{}, 0.45);
    run("IQ2_XS", ml::block_iq2_xs{}, 0.42);
    run("IQ2_S", ml::block_iq2_s{}, 0.35);
    run("IQ3_XXS", ml::block_iq3_xxs{}, 0.3);
    run("IQ3_S", ml::block_iq3_s{}, 0.2);
    run("IQ4_NL", ml::block_iq4_nl{}, 0.12);
    run("IQ4_XS", ml::block_iq4_xs{}, 0.12);

    // IQ4_NL as ggml's CPU backend quantises activations and caches: no scale search
    std::vector<ml::block_iq4_nl> fast(n / 32), searched(n / 32);
    ml::quantize_row_reference(w, std::span<ml::block_iq4_nl>(fast));
    ml::quantize_row(w, {}, std::span<ml::block_iq4_nl>(searched));
    const auto error = [&](const std::vector<ml::block_iq4_nl>& b) {
        std::vector<float> back(n);
        ml::dequantize(std::span<const ml::block_iq4_nl>(b), back);
        double e = 0;
        for (std::size_t i = 0; i < n; ++i)
            e += (double(back[i]) - w[i]) * (double(back[i]) - w[i]);
        return e;
    };
    std::printf("  IQ4_NL squared error: reference %.3g, with the scale search %.3g\n", error(fast), error(searched));
    ok &= error(searched) <= error(fast);

    std::printf(ok ? "quant_iq demo: ok\n" : "quant_iq demo: FAILED\n");
    return ok ? 0 : 1;
}
