//
//  rl_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/ml/rl.hpp>: GRPO on a toy policy that answers each of two prompts with one of four
//  one-token answers, the correct one rewarded: groups of eight samples drawn with stats::wyrand, group-normalised
//  advantages, the clipped objective with the k3 KL to the reference, and the gradient back to the policy's logits
//  through ml/logprobs.hpp, until the expected reward is near 1. Then the variants (Dr. GRPO, DAPO's clip and token
//  mean), PPO's GAE, value loss and whitening on one rollout. Exits non-zero if a result is not what it documents.
//

#include <snicholls/core/ml/rl.hpp>

#include <snicholls/core/linalg/reference.hpp>
#include <snicholls/core/ml/logprobs.hpp>
#include <snicholls/core/ml/softmax.hpp>
#include <snicholls/core/stats/random.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <vector>

namespace ml = snicholls::ml;
namespace stats = snicholls::stats;

int main()
{
    std::printf("snicholls-core rl demo\n");
    bool ok = true;
    const auto& backend = snicholls::linalg::reference;

    constexpr std::size_t prompts = 2, group = 8, vocab = 4, rows = prompts * group;
    const std::array<std::uint32_t, prompts> correct{2, 0};
    std::array<double, prompts * vocab> theta{};   // the policy: one row of logits per prompt
    const auto reference = theta;                  // the frozen reference: the policy at the start
    stats::wyrand g(42);
    const ml::policy_loss_config config{.kl_beta = 0.04};   // GRPO's beta; PPO's symmetric clip 0.2
    double first_reward = 0, last_reward = 0;
    for (int step = 0; step < 40; ++step) {
        // Sample G answers a prompt (a sequence of one response token), score them
        std::vector<std::uint32_t> targets(rows);
        std::vector<double> rewards(rows), logits(rows * vocab), ref_logits(rows * vocab);
        const std::vector<std::uint8_t> mask(rows, 1);
        for (std::size_t r = 0; r < rows; ++r) {
            const std::size_t p = r / group;
            std::array<double, vocab> probs{};
            ml::softmax(std::span<const double>(theta).subspan(p * vocab, vocab), probs);
            double u = stats::uniform_real(g), c = 0;
            targets[r] = vocab - 1;
            for (std::uint32_t k = 0; k < vocab; ++k)
                if (u < (c += probs[k])) {
                    targets[r] = k;
                    break;
                }
            rewards[r] = targets[r] == correct[p] ? 1.0 : 0.0;
            for (std::size_t k = 0; k < vocab; ++k) {
                logits[r * vocab + k] = theta[p * vocab + k];
                ref_logits[r * vocab + k] = reference[p * vocab + k];
            }
        }
        //! [grpo]
        // A rollout as plain data, then one GRPO step: advantages within each group, the clipped objective with the
        // k3 KL to the reference, and the gradient in each token's log-probability, then in the logits
        const ml::rollout_view<double> rollout{prompts, group, 1, targets, mask, rewards};
        std::vector<double> adv(rows), adv_tok(rows), logps(rows), ref_logps(rows), dlogps(rows);
        ml::group_advantages<double>(backend, rollout.rewards, rollout.group_size, adv);
        ml::expand_rows<double>(backend, adv, adv_tok);
        ml::token_logprobs_rows<double>(backend, logits, rollout.targets, logps);
        ml::token_logprobs_rows<double>(backend, ref_logits, rollout.targets, ref_logps);
        const auto& old_logps = logps;   // one step per rollout: the sampling policy is the policy, ratios 1
        ml::policy_loss_metrics<double> metrics;
        ml::policy_loss_rows<double>(backend, rollout.response_mask, logps, old_logps, ref_logps, adv_tok, {}, dlogps, {},
                                     rollout.rows(), config, metrics);
        ml::token_logprobs_rows_backward<double>(backend, logits, rollout.targets, dlogps);   // logits: their gradient
        //! [grpo]
        for (std::size_t r = 0; r < rows; ++r)
            for (std::size_t k = 0; k < vocab; ++k)
                theta[rollout.group_of(r) * vocab + k] -= 2.0 * logits[r * vocab + k];
        double expected = 0;
        for (std::size_t p = 0; p < prompts; ++p)
            expected += std::exp(ml::token_logprob(std::span<const double>(theta).subspan(p * vocab, vocab), correct[p])) / prompts;
        (step == 0 ? first_reward : last_reward) = expected;
        ok &= metrics.clip_fraction() == 0.0 && metrics.mean_ratio() == 1.0 && metrics.tokens == rows;
    }
    std::printf("  GRPO: expected reward %.3f -> %.3f\n", first_reward, last_reward);
    ok &= last_reward > 0.9 && last_reward > first_reward;

    // The variants on one batch of two responses, one token and three tokens long
    const std::vector<std::uint8_t> mask{1, 0, 0, 1, 1, 1};
    const std::vector<double> lp{-0.5, 0, 0, -1.0, -0.2, -0.3}, old{-0.6, 0, 0, -0.7, -0.2, -0.3}, adv{1.0, 0, 0, -1.0, -1.0, -1.0};
    const std::vector<double> rewards_group{1.0, 0.0};
    std::vector<double> centred(2), d(6);
    ml::group_advantages<double>(backend, rewards_group, 2, centred, ml::advantage_scale::none);   // Dr. GRPO
    ml::policy_loss_metrics<double> m;
    const double grpo = ml::policy_loss_rows<double>(backend, mask, lp, old, {}, adv, {}, d, {}, 2, {}, m);
    const double dapo = ml::policy_loss_rows<double>(backend, mask, lp, old, {}, adv, {}, d, {}, 2,
                                                     {.clip_high = 0.28, .aggregation = ml::token_aggregation::token_mean}, m);
    const double dr = ml::policy_loss_rows<double>(backend, mask, lp, old, {}, adv, {}, d, {}, 2, {.aggregation = ml::token_aggregation::constant}, m);
    std::printf("  one batch: GRPO %.4f, DAPO %.4f, Dr. GRPO %.4f; centred advantages %.1f %.1f; clipped %.2f of tokens\n", grpo, dapo, dr,
                centred[0], centred[1], m.clip_fraction());
    ok &= centred[0] == 0.5 && m.clip_fraction() > 0.0 && std::fabs(ml::kl_k3(-1.0, -1.5) - (std::exp(-0.5) + 0.5 - 1.0)) < 1e-15;
    ok &= ml::kl_k3_derivative(-1.0, -1.0) == 0.0 && ml::token_weight(mask, 2, {}, 0, 4) == 0.5;
    const auto t = ml::policy_token_loss(-0.5, -0.6, 0.0, 1.0, {});
    ok &= !t.clipped && std::fabs(t.loss + std::exp(0.1)) < 1e-15;

    // PPO: GAE over a reward on the last token, the clipped value loss, and whitened advantages
    const std::vector<std::uint8_t> ppo_mask{0, 1, 1, 1};
    const std::vector<double> token_rewards{0, 0, 0, 1}, values{0, 0.3, 0.5, 0.8}, old_values{0, 0.3, 0.5, 0.8};
    std::vector<double> gae(4), returns(4), dvalues(4), white(4);
    ml::gae_rows<double>(backend, ppo_mask, token_rewards, values, 1.0, 0.95, gae, returns, 1);
    const double vloss = ml::value_loss_rows<double>(backend, ppo_mask, values, old_values, returns, 0.2, dvalues);
    ml::masked_whiten<double>(backend, ppo_mask, gae, white);
    std::printf("  PPO: advantages %.4f %.4f %.4f, value loss %.5f\n", gae[1], gae[2], gae[3], vloss);
    ok &= std::fabs(gae[3] - 0.2) < 1e-15 && std::fabs(returns[3] - 1.0) < 1e-15 && vloss > 0 && std::fabs(white[1] + white[2] + white[3]) < 1e-12;

    std::printf("%s\n", ok ? "ok" : "FAILED");
    return ok ? 0 : 1;
}
