//
//  reward_model_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/ml/reward_model.hpp>: a value head reads the hidden state of each response's last token
//  (found from the mask and gathered), the Bradley-Terry loss compares the chosen and rejected rewards, and the
//  gradient goes back through the head and the gather into the sequence's hidden states; a few steps of training the
//  head lift its pairwise accuracy. Exits non-zero if a result is not what it documents.
//

#include <snicholls/core/ml/reward_model.hpp>

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
    std::printf("snicholls-core reward_model demo\n");
    bool ok = true;
    const auto& backend = snicholls::linalg::reference;

    //! [reward_model]
    // Two sequences (chosen, rejected) of 3 positions with hidden states of 2; the response ends at different places
    constexpr std::size_t d = 2, length = 3;
    const std::array<double, 2 * length * d> hs{0.1, 0.0, 0.9, 0.2, 0.0, 0.0, /**/ 0.3, 0.1, 0.0, 0.8, 0.5, 0.5};
    const std::array<std::uint8_t, 2 * length> mask{0, 1, 0, /**/ 0, 1, 1};
    std::array<std::uint32_t, 2> last{};
    ml::last_response_positions(mask, last);                  // rows 1 and 5
    std::array<double, 2 * d> h_last{};
    ml::gather_rows<double>(backend, hs, d, last, h_last);
    std::array<double, d> w{1.0, -1.0};
    double bias = 0.0;
    std::array<double, 2> rewards{};
    ml::value_head_rows<double>(backend, w, bias, h_last, rewards);
    const ml::pair_loss<double> p = ml::preference_loss(ml::bradley_terry{}, rewards[0], rewards[1]);
    //! [reward_model]
    std::printf("  last response positions %u %u, rewards %.2f / %.2f, loss %.6f\n", last[0], last[1], rewards[0], rewards[1], p.loss);
    ok &= last[0] == 1 && last[1] == 5 && std::fabs(rewards[0] - 0.7) < 1e-15 && rewards[1] == 0.0;
    ok &= std::fabs(p.loss - std::log1p(std::exp(-0.7))) < 1e-15 && ml::value_head(w, bias, std::array<double, 2>{0.9, 0.2}) == rewards[0];

    // The gradient back into the hidden states
    std::array<double, 2> dr{p.d_chosen, p.d_rejected};
    std::array<double, 2 * d> dh_last{};
    std::array<double, d> dw{};
    double db = 0;
    ml::value_head_rows_backward<double>(backend, w, h_last, dr, dh_last, dw, db);
    std::array<double, 2 * length * d> dhs{};
    ml::scatter_add_rows<double>(backend, dh_last, d, last, dhs);
    std::printf("  d loss / d h at the chosen response's end: %.4f %.4f; db %.1f (a shared bias cancels)\n", dhs[2], dhs[3], db);
    ok &= dhs[2] < 0 && dhs[3] > 0 && dhs[0] == 0.0 && db == 0.0;
    std::array<double, d> dh_one{}, dw_one{};
    double db_one = 0;
    ml::value_head_backward<double>(w, std::span<const double>(h_last).first(d), p.d_chosen, dh_one, dw_one, db_one);   // one row
    ok &= dh_one[0] == dh_last[0] && dw_one[0] == p.d_chosen * h_last[0];

    // A margin (Llama 2) asks for a larger gap; training the head on one pair widens it
    const auto m = ml::preference_loss(ml::bradley_terry{.margin = 1.0}, rewards[0], rewards[1]);
    ok &= m.loss > p.loss;
    for (int step = 0; step < 20; ++step) {
        ml::value_head_rows<double>(backend, w, bias, h_last, rewards);
        const auto q = ml::preference_loss(ml::bradley_terry{.margin = 1.0}, rewards[0], rewards[1]);
        dr = {q.d_chosen, q.d_rejected};
        dw = {};
        ml::value_head_rows_backward<double>(backend, w, h_last, dr, {}, dw, db);
        for (std::size_t j = 0; j < d; ++j)
            w[j] -= 1.0 * dw[j];
    }
    ml::value_head_rows<double>(backend, w, bias, h_last, rewards);
    std::printf("  after training with margin 1: rewards %.3f / %.3f\n", rewards[0], rewards[1]);
    ok &= rewards[0] - rewards[1] > 1.0;

    std::printf("%s\n", ok ? "ok" : "FAILED");
    return ok ? 0 : 1;
}
