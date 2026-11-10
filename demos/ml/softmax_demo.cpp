//
//  softmax_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/ml/softmax.hpp>: the next-token distribution of a tiny vocabulary, its loss and
//  gradient in one pass, a row split across four workers and merged, and bits per byte. Exits non-zero if
//  a result is not what it documents.
//

#include <snicholls/core/ml/softmax.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <vector>

int main()
{
    std::printf("snicholls-core softmax demo\n");
    namespace ml = snicholls::ml;
    bool ok = true;

    //! [softmax]
    // Logits for a five-token vocabulary; the target is token 2.
    const std::vector<float> logits{2.0f, -1.0f, 3.5f, 0.0f, 1.0f};
    std::vector<float> probs(logits.size()), log_probs(logits.size()), grad(logits.size());
    ml::softmax(logits, probs);
    ml::log_softmax(logits, log_probs);
    const float loss = ml::cross_entropy_with_gradient(logits, 2, grad, 1.0f / 4);   // a mean over 4 rows: scale 1/4

    // The backward pass of softmax on its own, for a loss that is not cross-entropy: d loss / d probs given.
    const std::vector<float> dprobs{0.0f, 0.0f, -1.0f, 0.0f, 0.0f};
    std::vector<float> dlogits(logits.size());
    ml::softmax_backward(probs, dprobs, dlogits);

    // A row of 100,000 logits split across four workers: each keeps an online_softmax, and the partials merge.
    std::vector<double> big(100000);
    for (std::size_t i = 0; i < big.size(); ++i)
        big[i] = 20.0 * std::sin(0.001 * static_cast<double>(i));
    std::array<ml::online_softmax<double>, 4> worker{};
    for (std::size_t i = 0; i < big.size(); ++i)
        worker[i * 4 / big.size()].add(big[i]);
    ml::online_softmax<double> all;
    for (const auto& w : worker)
        all.merge(w);

    // Bits per byte: 1,000 tokens with a mean loss of 2.1 nats, encoding 4,200 bytes of text.
    constexpr double bpb = ml::bits_per_byte(2.1 * 1000, 4200);
    //! [softmax]

    std::printf("  probabilities");
    for (const float p : probs)
        std::printf(" %.4f", static_cast<double>(p));
    std::printf("\n  loss %.4f nats = -log_softmax[2] %.4f; gradient[2] %.4f\n", static_cast<double>(loss),
                static_cast<double>(-log_probs[2]), static_cast<double>(grad[2]));
    ok &= std::abs(loss + log_probs[2]) < 1e-6f && std::abs(grad[2] - (probs[2] - 1.0f) / 4) < 1e-7f;
    std::printf("  softmax_backward: dlogits[2] %.4f\n", static_cast<double>(dlogits[2]));
    ok &= std::abs(dlogits[2] - (-probs[2] * (1.0f - probs[2]))) < 1e-6f;

    const double one_pass = ml::logsumexp(big);
    std::printf("  logsumexp of 100,000 logits: merged from 4 workers %.12f, one pass %.12f\n", all.log_sum_exp(), one_pass);
    ok &= std::abs(all.log_sum_exp() - one_pass) < 1e-9;

    std::printf("  bits per byte %.4f\n", bpb);
    ok &= std::abs(bpb - 2.1 * 1000 / (std::log(2.0) * 4200)) < 1e-15 && ml::cross_entropy(logits, 2) == loss;

    std::printf(ok ? "softmax demo: ok\n" : "softmax demo: FAILED\n");
    return ok ? 0 : 1;
}
