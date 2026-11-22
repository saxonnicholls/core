//
//  discrete_demo.cpp
//  snicholls-core
//
//  Demo for <snicholls/core/tracking/discrete.hpp>: a swimmer's hidden state (surfaced, at the surface, submerged)
//  filtered by a hidden Markov model through a sequence of a camera's observations, one of them a classifier's soft
//  output, with the predicted observation and each observation's likelihood; then the impossible observation.
//  Exits non-zero if a result is not what it documents.
//

#include <snicholls/core/tracking/discrete.hpp>

#include <snicholls/core/linalg/fixed.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>

namespace {

namespace la = snicholls::linalg;
namespace tr = snicholls::tracking;

using v2 = la::fixed_vector<double, 2>;
using v3 = la::fixed_vector<double, 3>;
using m33 = la::fixed_matrix<double, 3, 3>;
using m23 = la::fixed_matrix<double, 2, 3>;

bool check(bool ok, const char* what)
{
    std::printf("  %-62s %s\n", what, ok ? "ok" : "FAILED");
    return ok;
}

} // namespace

int main()
{
    std::printf("snicholls-core discrete demo\n");
    bool ok = true;

    //! [discrete]
    // Hidden: surfaced, at the surface, submerged. Columns are "from", rows "to"; counts, normalised per column
    const m33 transition = tr::normalise_columns(m33{{8, 1, 0,
                                                      2, 7, 2,
                                                      0, 2, 8}});
    // Observed: seen, not seen. A submerged swimmer is never seen
    const m23 emission{{0.9, 0.5, 0.0,
                        0.1, 0.5, 1.0}};
    const std::array<v2, 4> observations{v2{{1, 0}}, v2{{0, 1}}, v2{{0, 1}}, v2{{0.7, 0.3}}};   // the last a classifier's soft output

    v3 belief{{1.0 / 3, 1.0 / 3, 1.0 / 3}};
    double log_evidence = 0;
    std::array<v3, 4> history{};
    std::size_t step = 0;
    for (const v2& z : observations) {
        const auto predicted = tr::hmm_predict(belief, transition).unwrap();
        const auto expect = tr::hmm_predict_measurement(predicted, emission).unwrap();   // P(seen) before looking
        log_evidence += std::log(tr::hmm_likelihood(predicted, z, emission));
        belief = tr::hmm_update(predicted, z, emission).unwrap();
        history[step++] = belief;
        std::printf("  P(seen) predicted %.3f; observed (%.1f, %.1f): belief %.4f %.4f %.4f\n", expect[0], z[0], z[1], belief[0], belief[1], belief[2]);
    }
    //! [discrete]

    std::printf("  log evidence of the sequence %.4f\n", log_evidence);
    ok &= check(std::abs(transition(0, 0) - 0.8) < 1e-15 && std::abs(transition(1, 1) - 0.7) < 1e-15, "normalise_columns makes the counts probabilities");
    ok &= check(std::abs(belief[0] + belief[1] + belief[2] - 1) < 1e-14, "the belief sums to 1");
    ok &= check(history[0][2] == 0, "a sighting rules out submerged");
    ok &= check(history[2][2] > history[1][2] && history[3][2] < history[2][2], "not seen raises P(submerged); the soft sighting lowers it");
    ok &= check(std::isfinite(log_evidence) && log_evidence < 0, "every observation had positive likelihood");

    // By hand, one step from certainty "surfaced": predicted (0.8, 0.2, 0), P(seen) = 0.72 + 0.1 = 0.82
    const auto one = tr::hmm_predict(v3{{1, 0, 0}}, transition).unwrap();
    const auto seen = tr::hmm_predict_measurement(one, emission).unwrap();
    ok &= check(std::abs(one[0] - 0.8) < 1e-15 && std::abs(seen[0] - 0.82) < 1e-15, "hmm_predict and hmm_predict_measurement by hand");
    const auto post = tr::hmm_update(one, v2{{1, 0}}, emission).unwrap();
    ok &= check(std::abs(post[0] - 0.72 / 0.82) < 1e-15 && post[2] == 0, "hmm_update by hand: Bayes' rule");
    ok &= check(std::abs(tr::hmm_likelihood(one, v2{{1, 0}}, emission) - 0.82) < 1e-15, "hmm_likelihood is the marginal probability");

    // An observation impossible under the prior: certainly submerged, yet seen
    const auto impossible = tr::hmm_update(v3{{0, 0, 1}}, v2{{1, 0}}, emission);
    ok &= check(impossible.is_err() && impossible.unwrap_err_unchecked() == tr::hmm_error::zero_probability, "an impossible observation is zero_probability");
    ok &= check(tr::normalise(v3{}).is_err(), "normalise refuses a zero vector");
    ok &= check(tr::hmm_predict(belief, m33{}).is_err() && tr::hmm_predict_measurement(v3{}, emission).is_err(), "a zero prediction is zero_probability");

    std::printf(ok ? "discrete demo: ok\n" : "discrete demo: FAILED\n");
    return ok ? 0 : 1;
}
