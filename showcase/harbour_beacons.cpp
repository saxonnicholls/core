//
//  harbour_beacons.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Showcase: an underwater vehicle crosses a harbour seeded with acoustic beacons. A k-d tree over the beacons
//  decides which one the vehicle can hear; the hydrophone's samples (tone plus seeded noise) go through a Goertzel
//  bank whose coefficients the compiler computed; the strongest tone and the k-d tree's nearest candidates identify
//  the beacon; a state machine keeps the navigation lock (searching, locked, coasting) through a stretch where a
//  wreck shadows the signal; and two workers' signal statistics merge into exactly what one pass would have seen.
//  Exits non-zero if any result is not the one documented.
//

#include <snicholls/core/acout.hpp>
#include <snicholls/core/containers/kd_tree.hpp>
#include <snicholls/core/display.hpp>
#include <snicholls/core/fsm/state_machine.hpp>
#include <snicholls/core/math/constexpr_math.hpp>
#include <snicholls/core/signal/goertzel.hpp>
#include <snicholls/core/stats/accumulators.hpp>
#include <snicholls/core/stats/normal.hpp>
#include <snicholls/core/stats/random.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace harbour {

using namespace snicholls;
namespace sig = snicholls::signal;
namespace st = snicholls::stats;
namespace math = snicholls::math;
using xy = std::array<double, 2>;

// ---- The acoustics, fixed before the program runs ----------------------------------------------------------------
constexpr double sample_rate = 8000.0;
constexpr std::size_t block = 200;   // 25 ms; bins 40 Hz apart
constexpr std::array<double, 8> tones_hz{1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400};
constexpr sig::goertzel_bank<float, 8> receiver(tones_hz, sample_rate, block);
static_assert(receiver.coefficients(0).bin == 25.0f && receiver.coefficients(7).bin == 60.0f, "every tone on a bin: no leakage");
using pilot = sig::static_goertzel<float, 1000.0, sample_rate, block>;   // one tone, its coefficient a constant
static_assert(pilot::coeff == receiver.coefficients(0).coeff);

constexpr double spacing = 200.0, hearing_range = 140.0;

// ---- The navigation lock: a state machine over an enumeration that knows how it looks ----------------------------
enum class lock : std::uint8_t { searching, locked, coasting };
constexpr auto attributes(lock l)
{
    return per_enumerator(l, [] { return std::array{attrs(colours::grey, label{"searching"}), attrs(colours::cyan, label{"locked"}),
                                                    attrs(colours::yellow, label{"coasting"})}; });
}
struct heard {
    std::uint32_t beacon;
};
struct silent {};
struct navigation {
    using state_enum = lock;
    constexpr auto operator()() const noexcept
    {
        using namespace snicholls::fsm;
        constexpr auto searching = enum_state<lock::searching>;
        constexpr auto locked = enum_state<lock::locked>;
        constexpr auto coasting = enum_state<lock::coasting>;
        return make_transition_table(*searching + event<heard> = locked,
                                      locked + event<silent> = coasting,
                                      coasting + event<heard> = locked,
                                      coasting + event<silent> = searching);
    }
};

// ---- What the receiver reports per block ---------------------------------------------------------------------------
struct snr_db {   // grey when marginal, then cyan
    double value;
};
constexpr auto attributes(const snr_db&)
{
    return attrs(threshold_colours{colours::grey, {{6.0, colours::cyan}}}, number_format{"{:.1f}"}, units{"dB"});
}

// The tone with clearly the most power, or none: a decision, not a guess.
std::optional<std::size_t> decide(const sig::goertzel_bank<float, 8>& bank, double& ratio_db)
{
    const std::size_t best = bank.strongest();
    float runner_up = 0;
    for (std::size_t i = 0; i < bank.size(); ++i)
        runner_up = i == best ? runner_up : std::max(runner_up, bank.power(i));
    ratio_db = 10 * std::log10(static_cast<double>(bank.power(best)) / static_cast<double>(runner_up));
    return ratio_db > 6.0 ? std::optional{best} : std::nullopt;
}

} // namespace harbour

int main()
{
    using namespace harbour;
    namespace ct = snicholls::containers;
    bool ok = true;
    const auto expect = [&](bool good, std::string_view what) {
        acout{} << "  " << styled(good ? colours::green : colours::red, good ? "ok  " : "FAIL") << ' ' << what << '\n';
        ok &= good;
    };
    println(colours::cyan, "Harbour crossing: beacons, a hydrophone and a navigation lock");

    // A 6 x 4 grid of beacons, each dropped a little off its mark; neighbours never share a tone.
    st::wyrand g(2026);
    std::vector<xy> beacons;
    std::vector<std::size_t> tone_of;
    for (int row = 0; row < 4; ++row)
        for (int col = 0; col < 6; ++col) {
            beacons.push_back({col * spacing + (st::uniform_real(g) - 0.5) * 40, row * spacing + (st::uniform_real(g) - 0.5) * 40});
            tone_of.push_back(static_cast<std::size_t>(col + 3 * row) % tones_hz.size());
        }
    std::vector<std::uint32_t> order(beacons.size());
    const ct::kd_tree<xy, 2> field(beacons, order);

    // The vehicle crosses diagonally; its own position estimate drifts by a few metres. A wreck shadows steps 24..29.
    constexpr int steps = 48;
    st::normal_inverse_cdf normal;
    snicholls::fsm::sm<navigation> nav;
    st::variance<double> worker[2];   // the first half of the crossing on one worker, the second on another
    st::variance<double> one_pass;
    int identified = 0, wrong = 0, heard_blocks = 0, losses = 0;
    std::string track;
    for (int step = 0; step < steps; ++step) {
        const double f = static_cast<double>(step) / (steps - 1);
        const xy truth{20 + f * 960, 40 + f * 520};
        const xy estimate{truth[0] + 8 * normal(g), truth[1] + 8 * normal(g)};
        const bool shadowed = step >= 24 && step < 30;

        // Who sings: the nearest beacon within hearing range, louder when closer
        const auto nearest = field.nearest(truth);
        const bool audible = !shadowed && nearest->distance < hearing_range;
        std::vector<float> samples(block);
        for (std::size_t j = 0; j < block; ++j) {
            const double tone = audible ? 0.6 * (1 - nearest->distance / 200) *
                                              math::sin(2 * math::math_limits<double>::pi * tones_hz[tone_of[nearest->index]] * static_cast<double>(j) / sample_rate)
                                        : 0.0;
            samples[j] = static_cast<float>(tone + 0.25 * normal(g));
        }

        auto bank = receiver;   // the constants, a zeroed state
        bank.update(samples);
        double ratio = 0;
        const auto tone = decide(bank, ratio);
        const lock before = nav.current();
        if (tone) {
            // The tone names a class of beacons; the k-d tree's nearest candidates name the one
            std::array<ct::kd_neighbour, 4> candidates{};
            const std::size_t n = field.k_nearest(estimate, candidates);
            std::optional<std::uint32_t> who;
            for (std::size_t c = 0; c < n && !who; ++c)
                who = tone_of[candidates[c].index] == *tone ? std::optional{candidates[c].index} : std::nullopt;
            ++heard_blocks;
            identified += who && *who == nearest->index;
            wrong += who && *who != nearest->index;
            worker[step < steps / 2].add(ratio);
            one_pass.add(ratio);
            nav.process_event(heard{who.value_or(0)});
        } else {
            nav.process_event(silent{});
        }
        losses += before != lock::searching && nav.current() == lock::searching;
        track += "s|-"[static_cast<std::size_t>(nav.current())];
        if (step % 6 == 0 || before != nav.current())
            acout{} << "  step " << step << "  (" << static_cast<int>(truth[0]) << ", " << static_cast<int>(truth[1]) << ")  "
                    << (tone ? "tone " + std::to_string(static_cast<int>(tones_hz[*tone])) + " Hz" : std::string("no tone    ")) << "  "
                    << snr_db{ratio} << "  " << before << " -> " << nav.current() << (shadowed ? "  (wreck)" : "") << '\n';
    }

    acout{} << "  lock along the crossing: " << track << "   (| locked, - coasting, s searching)\n";
    expect(identified == heard_blocks && wrong == 0, "every tone heard was matched to the right beacon");
    expect(heard_blocks > steps / 2, "the vehicle heard a beacon in most blocks");
    expect(track.substr(24, 6).find('|') == std::string::npos && losses >= 1, "the wreck's shadow cost the lock, which came back after");
    expect(nav.current() == lock::locked, "locked again at the far side");

    // Two workers' statistics merge into the single pass, exactly as far as rounding allows
    worker[1].merge(worker[0]);
    acout{} << "  tone-to-runner-up ratio over " << one_pass.count() << " blocks: mean " << snr_db{one_pass.mean()} << ", spread "
            << snr_db{one_pass.standard_deviation()} << '\n';
    expect(worker[1].count() == one_pass.count() && std::abs(worker[1].mean() - one_pass.mean()) < 1e-12 &&
               std::abs(worker[1].value() - one_pass.value()) < 1e-9,
           "two workers' accumulators merge into one pass's");

    acout{} << '\n';
    println(ok ? colours::green : colours::red, "harbour beacons: {}", ok ? "ok" : "FAILED");
    return ok ? 0 : 1;
}
