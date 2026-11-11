//
//  sampling_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/ml/sampling.hpp>: the next token from one row of logits greedily, at a temperature,
//  from the top k and from the nucleus, each draw from a seeded generator so the run repeats exactly. Exits
//  non-zero if a result is not what it documents.
//

#include <snicholls/core/ml/sampling.hpp>
#include <snicholls/core/stats/random.hpp>

#include <array>
#include <cstddef>
#include <cstdio>

int main()
{
    std::printf("snicholls-core sampling demo\n");
    namespace ml = snicholls::ml;
    namespace st = snicholls::stats;
    bool ok = true;

    //! [sampling]
    // One row of logits over a vocabulary of eight tokens
    const std::array<float, 8> logits{1.0f, 4.0f, 2.5f, -1.0f, 3.5f, 0.0f, 2.0f, -2.0f};
    st::wyrand g(2026);   // one stream per sequence: the same seed gives the same text everywhere
    const auto u = [&] { return static_cast<float>(st::uniform_real(g)); };

    const std::size_t greedy = ml::argmax(logits);                         // temperature 0
    const std::size_t warm = ml::sample_logits(logits, 0.8f, u());          // the model's distribution, sharpened
    const std::size_t top3 = ml::sample_top_k<3>(logits, 1.0f, u());        // among tokens 1, 4 and 2 only
    std::array<std::size_t, 8> scratch{};
    const std::size_t nucleus = ml::sample_top_p(logits, 1.0f, 0.9f, u(), scratch);
    const auto best = ml::top_k<3>(logits);                                 // the candidates, largest first
    //! [sampling]

    std::printf("  greedy %zu, temperature 0.8 -> %zu, top-3 -> %zu, top-p 0.9 -> %zu\n", greedy, warm, top3, nucleus);
    std::printf("  top 3: %zu %zu %zu\n", best.index[0], best.index[1], best.index[2]);
    ok &= greedy == 1 && best.index[0] == 1 && best.index[1] == 4 && best.index[2] == 2;
    ok &= top3 == 1 || top3 == 4 || top3 == 2;
    const std::array<float, 4> p{0.1f, 0.2f, 0.3f, 0.4f};
    std::printf("  sample([0.1 0.2 0.3 0.4], 0.35) = %zu\n", ml::sample(p, 0.35f));
    ok &= ml::sample(p, 0.35f) == 2;

    // The same seed again gives the same choices
    st::wyrand again(2026);
    const auto v = [&] { return static_cast<float>(st::uniform_real(again)); };
    ok &= ml::sample_logits(logits, 0.8f, v()) == warm && ml::sample_top_k<3>(logits, 1.0f, v()) == top3
          && ml::sample_top_p(logits, 1.0f, 0.9f, v(), scratch) == nucleus;

    std::printf(ok ? "sampling demo: ok\n" : "sampling demo: FAILED\n");
    return ok ? 0 : 1;
}
