//
//  samplers_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/ml/samplers.hpp>: a llama.cpp-style chain of samplers over a candidate
//  array, fixed at compile time and the same chain built at run time, every sampler applied once and
//  checked against its documented effect, and a short generation with the stateful samplers (penalties,
//  DRY, mirostat). Exits non-zero if a result is not what it documents.
//

#include <snicholls/core/ml/samplers.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <span>
#include <vector>

namespace {

namespace ml = snicholls::ml;
using snicholls::stats::wyrand;

// One sampler applied to fresh candidates from `logits`: the ids left, in order
template <typename S>
std::vector<ml::token_id> after(S s, const std::vector<float>& logits, wyrand& g, std::ptrdiff_t* selected = nullptr)
{
    std::vector<ml::token_data<float>> storage(logits.size());
    auto c = ml::make_candidates(logits, std::span(storage));
    s.apply(c, g);
    std::vector<ml::token_id> out;
    for (const auto& d : c.data)
        out.push_back(d.id);
    if (selected != nullptr)
        *selected = c.selected;
    return out;
}

void show(const char* name, const std::vector<ml::token_id>& ids)
{
    std::printf("  %-12s", name);
    for (const auto t : ids)
        std::printf(" %d", t);
    std::printf("\n");
}

} // namespace

int main()
{
    std::printf("snicholls-core samplers demo\n");
    bool ok = true;

    //! [samplers]
    // A vocabulary of eight tokens and one step's logits.
    const std::vector<float> logits{1.0f, 3.0f, 0.5f, 2.5f, -1.0f, 2.0f, 0.0f, 1.5f};
    std::vector<ml::token_data<float>> storage(logits.size());

    // The chain fixed at compile time: penalties over the history, top-k, top-p, temperature, then a draw.
    ml::sampler_chain chain{ml::penalties_sampler<float, 32>{.last_n = 32, .repeat = 1.3f}, ml::top_k_sampler{5},
                            ml::top_p_sampler<float>{0.9f}, ml::temperature_sampler<float>{0.8f}, ml::dist_sampler{}};

    // The same chain built at run time, from a request, say.
    ml::runtime_sampler_chain<float, 32> dynamic;
    dynamic.add(ml::penalties_sampler<float, 32>{.last_n = 32, .repeat = 1.3f})
        .add(ml::top_k_sampler{5})
        .add(ml::top_p_sampler<float>{0.9f})
        .add(ml::temperature_sampler<float>{0.8f})
        .add(ml::dist_sampler{});

    // A generation: each step samples, then tells the chain which token was accepted.
    wyrand g1(2026), g2(2026);
    std::vector<ml::token_id> text;
    for (int step = 0; step < 12; ++step) {
        const ml::token_id a = chain.sample(logits, std::span(storage), g1);
        const ml::token_id b = dynamic.sample(logits, std::span(storage), g2);
        ok &= a == b;   // the same seed, the same chain: the same token
        chain.accept(a);
        dynamic.accept(b);
        text.push_back(a);
    }
    //! [samplers]

    show("generated", text);
    for (const auto t : text)
        ok &= t == 1 || t == 3 || t == 5 || t == 7 || t == 0;   // top-k 5 never leaves {1, 3, 5, 7, 0}

    // Every sampler once, on fresh candidates
    wyrand g(7);
    const auto top_k = after(ml::top_k_sampler{3}, logits, g);
    show("top_k 3", top_k);
    ok &= top_k == std::vector<ml::token_id>{1, 3, 5};
    const auto top_p = after(ml::top_p_sampler<float>{0.5f}, logits, g);
    show("top_p 0.5", top_p);
    ok &= top_p == std::vector<ml::token_id>{1, 3};
    const auto min_p = after(ml::min_p_sampler<float>{0.3f}, logits, g);
    show("min_p 0.3", min_p);   // logits within ln(1/0.3) = 1.2 of the top, in their order
    ok &= min_p == std::vector<ml::token_id>{1, 3, 5};
    const auto typical = after(ml::typical_sampler<float>{0.5f}, logits, g);
    show("typical 0.5", typical);
    ok &= !typical.empty() && typical.size() < logits.size();
    const auto xtc = after(ml::xtc_sampler<float>{.probability = 1.0f, .threshold = 0.2f}, logits, g);
    show("xtc 0.2", xtc);   // tokens 1 and 3 reach 0.2: 1 is removed, 3 is the least probable of them and stays
    ok &= !xtc.empty() && xtc[0] == 3;
    const ml::temp_ext_sampler<float> dyn{.temp = 1.0f, .delta = 0.5f, .exponent = 1.0f};
    std::printf("  temp_ext     %.3f at zero entropy, %.3f at uniform\n", static_cast<double>(dyn.dynamic_temperature(0.0f, 8)),
                static_cast<double>(dyn.dynamic_temperature(std::log(8.0f), 8)));
    ok &= dyn.dynamic_temperature(0.0f, 8) == 0.5f && std::abs(dyn.dynamic_temperature(std::log(8.0f), 8) - 1.5f) < 1e-6f;
    ok &= after(dyn, logits, g).size() == logits.size();
    {
        std::vector<ml::token_data<float>> s(logits.size());
        auto c = ml::make_candidates(logits, std::span(s));
        ml::top_n_sigma_sampler<float>{1.0f}.apply(c, g);
        int masked = 0;
        for (const auto& d : c.data)
            masked += d.logit == -std::numeric_limits<float>::infinity() ? 1 : 0;
        std::printf("  top_n_sigma  %d of 8 masked\n", masked);
        ok &= masked == 5;   // mean 1.19, sigma 1.21: only logits from 1.79 up stay (3.0, 2.5, 2.0)
    }
    std::ptrdiff_t sel = -1;
    after(ml::greedy_sampler{}, logits, g, &sel);
    ok &= sel == 1;
    after(ml::temperature_sampler<float>{0.0f}, logits, g);
    const std::array<ml::logit_bias_entry<float>, 1> ban{{{1, -std::numeric_limits<float>::infinity()}}};
    {
        ml::sampler_chain biased{ml::logit_bias_sampler<float>{ban}, ml::greedy_sampler{}};
        const auto t = biased.sample(logits, std::span(storage), g);
        std::printf("  logit_bias   token 1 banned: greedy picks %d\n", t);
        ok &= t == 3;
    }

    // DRY: "a b c x a b" makes c the continuation of a repeat of 2
    {
        ml::dry_sampler<float, 16> dry{.multiplier = 1.0f, .base = 2.0f, .allowed_length = 2, .penalty_last_n = 16};
        for (const ml::token_id t : {0, 1, 2, 4, 0, 1})
            dry.accept(t);
        std::vector<float> zeros(8, 0.0f);
        std::vector<ml::token_data<float>> s(8);
        auto c = ml::make_candidates(zeros, std::span(s));
        dry.apply(c, g);
        std::printf("  dry          c penalised by %.1f\n", static_cast<double>(-c.data[2].logit));
        ok &= c.data[2].logit == -1.0f && c.data[0].logit == 0.0f;
    }

    // Mirostat 2.0 over ten steps: mu follows the observed surprise
    {
        ml::mirostat_v2_sampler<float> m{.tau = 1.5f, .eta = 0.1f};
        m.reset();
        ml::sampler_chain mc{m};
        for (int i = 0; i < 10; ++i)
            mc.accept(mc.sample(logits, std::span(storage), g));
        const float mu = std::get<0>(mc.samplers).mu;
        std::printf("  mirostat_v2  mu %.3f after 10 steps (from 3.0)\n", static_cast<double>(mu));
        ok &= mu != 3.0f && mu > 0.0f;
        ml::mirostat_sampler<float> m1{.n_vocab = 8, .tau = 1.5f, .eta = 0.1f, .m = 4};
        m1.reset();
        std::ptrdiff_t s1 = -1;
        after(m1, logits, g, &s1);
        ok &= s1 >= 0;
    }

    std::printf(ok ? "samplers demo: ok\n" : "samplers demo: FAILED\n");
    return ok ? 0 : 1;
}
