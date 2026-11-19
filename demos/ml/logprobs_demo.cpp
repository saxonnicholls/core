//
//  logprobs_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/ml/logprobs.hpp>: the log-probability of a response under a toy model, one token at a
//  time and as a batch of rows with the prompt masked; the entropies; the sum and the length-normalised mean; and the
//  backward pass from the sequence's log-probability to the logits, which leaves the prompt's logits untouched. Exits
//  non-zero if a result is not what it documents.
//

#include <snicholls/core/ml/logprobs.hpp>

#include <snicholls/core/linalg/reference.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>

namespace ml = snicholls::ml;

int main()
{
    std::printf("snicholls-core logprobs demo\n");
    bool ok = true;
    const auto& backend = snicholls::linalg::reference;

    //! [logprobs]
    // One sequence of 4 positions over a vocabulary of 3; position 0 predicts a prompt token, 1-3 the response
    constexpr std::size_t length = 4, vocab = 3;
    std::array<double, length * vocab> logits{2.0, 0.0, 0.0, /**/ 0.0, 3.0, 0.0, /**/ 1.0, 1.0, 1.0, /**/ 0.0, 0.0, 5.0};
    const std::array<std::uint32_t, length> targets{0, 1, 2, 2};
    const std::array<std::uint8_t, length> mask{0, 1, 1, 1};
    std::array<double, length> token_logps{}, entropies{};
    ml::token_logprobs_rows<double>(backend, logits, targets, token_logps, entropies);
    std::array<double, 1> sum{}, mean{};
    ml::sequence_logprobs<double>(backend, token_logps, mask, sum, ml::logprob_reduction::sum);    // log pi(y | x)
    ml::sequence_logprobs<double>(backend, token_logps, mask, mean, ml::logprob_reduction::mean);  // per response token
    //! [logprobs]
    std::printf("  token log-probabilities: %.4f %.4f %.4f %.4f\n", token_logps[0], token_logps[1], token_logps[2], token_logps[3]);
    std::printf("  entropies: %.4f %.4f %.4f %.4f (the uniform row: log 3 = %.4f)\n", entropies[0], entropies[1], entropies[2],
                entropies[3], std::log(3.0));
    std::printf("  response: sum %.4f, mean %.4f over %zu tokens\n", sum[0], mean[0], ml::response_length(mask));
    ok &= std::fabs(token_logps[2] + std::log(3.0)) < 1e-15 && std::fabs(entropies[2] - std::log(3.0)) < 1e-15;
    ok &= std::fabs(sum[0] - (token_logps[1] + token_logps[2] + token_logps[3])) < 1e-15 && std::fabs(mean[0] - sum[0] / 3) < 1e-15;
    ok &= ml::token_logprob(std::span<const double>(logits).first(vocab), 0) == token_logps[0];
    ok &= ml::token_entropy(std::span<const double>(logits).subspan(3, vocab)) == entropies[1];

    // One row's gradient, fused: dlogp (onehot - softmax)
    std::array<double, vocab> g{};
    const double lp = ml::token_logprob_with_gradient(std::span<const double>(logits).subspan(3, vocab), 1, g, 1.0);
    std::printf("  d log p / d logits at position 1: %.4f %.4f %.4f\n", g[0], g[1], g[2]);
    ok &= lp == token_logps[1] && std::fabs(g[0] + g[1] + g[2]) < 1e-15 && g[1] > 0;

    // The backward from the sequence's log-probability (and an entropy bonus) to the logits
    std::array<double, 1> dseq{1.0};
    std::array<double, length> dtok{}, dent{0.0, 0.1, 0.1, 0.1};
    ml::sequence_logprobs_backward<double>(backend, mask, dseq, dtok, ml::logprob_reduction::sum);
    ml::token_logprobs_rows_backward<double>(backend, logits, targets, dtok, dent);   // logits replaced by the gradient
    std::printf("  gradient at the prompt position: %g %g %g\n", logits[0], logits[1], logits[2]);
    ok &= logits[0] == 0.0 && logits[1] == 0.0 && logits[2] == 0.0 && logits[3 + 1] > 0.0;

    std::printf("%s\n", ok ? "ok" : "FAILED");
    return ok ? 0 : 1;
}
