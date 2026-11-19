//
//  preference_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/ml/preference.hpp>: one preference pair under DPO, IPO, SimPO and ORPO, with each loss's
//  gradient and implicit rewards; a batch of pairs on the backend with its metrics merged across two workers; and a
//  short DPO run on a toy policy (a free table of log-probabilities) in which the reward margin rises. Exits non-zero
//  if a result is not what it documents.
//

#include <snicholls/core/ml/preference.hpp>

#include <snicholls/core/linalg/reference.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <span>

namespace ml = snicholls::ml;

int main()
{
    std::printf("snicholls-core preference demo\n");
    bool ok = true;
    const auto& backend = snicholls::linalg::reference;

    //! [dpo]
    // Sequence log-probabilities (sums over the response) of the policy and of the frozen reference
    const double chosen = -12.0, rejected = -15.0, ref_chosen = -13.0, ref_rejected = -14.0;
    const ml::pair_loss<double> d = ml::preference_loss(ml::dpo{.beta = 0.1}, chosen, rejected, ref_chosen, ref_rejected);
    // d.loss = -log sigma(0.1 ((-12 + 13) - (-15 + 14))) = -log sigma(0.2); d.d_chosen = -0.1 sigma(-0.2)
    //! [dpo]
    std::printf("  DPO:   loss %.6f, d/dchosen %.6f, d/drejected %.6f, rewards %.2f / %.2f\n", d.loss, d.d_chosen, d.d_rejected,
                d.chosen_reward, d.rejected_reward);
    ok &= std::fabs(d.loss - std::log1p(std::exp(-0.2))) < 1e-15 && d.d_rejected == -d.d_chosen;

    const auto i = ml::preference_loss(ml::ipo{.tau = 0.5}, chosen, rejected, ref_chosen, ref_rejected);
    std::printf("  IPO:   loss %.6f (h = 2 against the target 1 / (2 tau) = 1)\n", i.loss);
    ok &= i.loss == 1.0;
    // SimPO and ORPO take mean log-probabilities per token and no reference
    const auto s = ml::preference_loss(ml::simpo{.beta = 2.0, .gamma = 0.5}, -0.4, -0.9);
    std::printf("  SimPO: loss %.6f (margin 2 (0.5) - 0.5 = 0.5)\n", s.loss);
    ok &= std::fabs(s.loss - std::log1p(std::exp(-0.5))) < 1e-15;
    const auto o = ml::preference_loss(ml::orpo{.lambda = 0.1}, -0.4, -0.9);
    std::printf("  ORPO:  loss %.6f = SFT 0.4 + 0.1 x odds-ratio term; log odds %.4f vs %.4f\n", o.loss, ml::log_odds_from_logprob(-0.4),
                ml::log_odds_from_logprob(-0.9));
    ok &= o.loss > 0.4 && o.d_chosen < -1.0 && o.d_rejected > 0.0;

    //! [rows]
    // A batch of three pairs: the losses, the gradients of the mean, and the metrics, on the backend
    const std::array<double, 3> pc{-10, -20, -8}, pr{-12, -18, -9}, rc{-11, -19, -8.5}, rr{-11, -19, -8.5};
    std::array<double, 3> dc{}, dr{};
    ml::preference_metrics<double> worker0, worker1;
    const double mean_loss = ml::preference_loss_rows<double>(backend, ml::dpo{.beta = 0.5}, pc, pr, rc, rr, dc, dr, worker0);
    worker1.merge(worker0);   // metrics from many workers merge
    //! [rows]
    std::printf("  batch: mean loss %.6f, accuracy %.3f, margin %.4f, chosen %.4f, rejected %.4f\n", mean_loss, worker1.accuracy(),
                worker1.mean_margin(), worker1.mean_chosen_reward(), worker1.mean_rejected_reward());
    ok &= std::fabs(worker1.mean_loss() - mean_loss) < 1e-15 && std::fabs(worker1.accuracy() - 2.0 / 3) < 1e-15;

    // A toy policy: the log-probabilities themselves are its parameters; DPO pushes chosen up and rejected down
    std::array<double, 3> policy_c = rc, policy_r = rr;
    double first = 0, last = 0;
    for (int step = 0; step < 50; ++step) {
        ml::preference_metrics<double> m;
        ml::preference_loss_rows<double>(backend, ml::dpo{.beta = 0.5}, policy_c, policy_r, rc, rr, dc, dr, m);
        (step == 0 ? first : last) = m.mean_margin();
        for (std::size_t k = 0; k < 3; ++k) {
            policy_c[k] -= 2.0 * dc[k];
            policy_r[k] -= 2.0 * dr[k];
        }
    }
    std::printf("  toy DPO run: reward margin %.4f -> %.4f\n", first, last);
    ok &= first == 0.0 && last > 1.0;

    std::printf("%s\n", ok ? "ok" : "FAILED");
    return ok ? 0 : 1;
}
