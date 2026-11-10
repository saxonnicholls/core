//
//  attributes_benchmark.cpp
//  snicholls-core
//
//  The colours of 1,048,576 greeks known only at run time: through their compile-time attributes
//  (snicholls::attributes_of, per_enumerator), against the same table written by hand (a colour and a label per
//  greek, guarded against a value that is not an enumerator), and, for reference, a bare table of colours with no
//  guard. The fairness check requires the same sum of colours from all three.
//

#include <snicholls/core/display.hpp>
#include <snicholls/core/meta/attributes.hpp>

#include <benchmark/benchmark.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <string_view>
#include <vector>

namespace bench {

enum class greek : std::uint8_t { delta, gamma, vega, vanna, volga, theta, rho };

constexpr auto attributes(greek g)
{
    using namespace snicholls;
    return per_enumerator(g, [] { return std::array{attrs(colours::blue, label{"delta"}), attrs(colours::cyan, label{"gamma"}),
                                        attrs(colours::magenta, label{"vega"}), attrs(colours::orange, label{"vanna"}),
                                        attrs(colours::yellow, label{"volga"}), attrs(colours::grey, label{"theta"}),
                                        attrs(rgb(0x953800), label{"rho"})}; });
}

struct greek_display {
    snicholls::colour colour;
    std::string_view label;
};
constexpr std::array<greek_display, 7> by_hand_table{{{snicholls::colours::blue, "delta"}, {snicholls::colours::cyan, "gamma"},
                                                      {snicholls::colours::magenta, "vega"}, {snicholls::colours::orange, "vanna"},
                                                      {snicholls::colours::yellow, "volga"}, {snicholls::colours::grey, "theta"},
                                                      {snicholls::rgb(0x953800), "rho"}}};
constexpr std::array<std::uint32_t, 7> bare_colours{0x0969da, 0x1b7c83, 0x8250df, 0xbc4c00, 0xbf8700, 0x6e7781, 0x953800};

} // namespace bench

namespace {

const std::vector<bench::greek> greeks = [] {
    std::vector<bench::greek> v(1u << 20);
    std::uint64_t x = 0x243f6a8885a308d3ULL;
    for (auto& g : v) {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        g = static_cast<bench::greek>(x % 7);
    }
    return v;
}();

std::uint64_t via_attributes()
{
    std::uint64_t s = 0;
    for (const bench::greek g : greeks)
        s += snicholls::attributes_of(g).get<snicholls::colour>().rgb;
    return s;
}

std::uint64_t by_hand()
{
    std::uint64_t s = 0;
    for (const bench::greek g : greeks) {
        const auto at = static_cast<std::size_t>(g);
        s += bench::by_hand_table[at < 7 ? at : 0].colour.rgb;
    }
    return s;
}

std::uint64_t bare()
{
    std::uint64_t s = 0;
    for (const bench::greek g : greeks)
        s += bench::bare_colours[static_cast<std::size_t>(g)];
    return s;
}

void BM_Attributes_Colour(benchmark::State& state)
{
    for (auto _ : state)
        benchmark::DoNotOptimize(via_attributes());
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(greeks.size()));
}
BENCHMARK(BM_Attributes_Colour)->Unit(benchmark::kMicrosecond);

void BM_Attributes_TableByHand(benchmark::State& state)
{
    for (auto _ : state)
        benchmark::DoNotOptimize(by_hand());
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(greeks.size()));
}
BENCHMARK(BM_Attributes_TableByHand)->Unit(benchmark::kMicrosecond);

void BM_Attributes_BareColoursNoGuard(benchmark::State& state)
{
    for (auto _ : state)
        benchmark::DoNotOptimize(bare());
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(greeks.size()));
}
BENCHMARK(BM_Attributes_BareColoursNoGuard)->Unit(benchmark::kMicrosecond);

} // namespace

int main(int argc, char** argv)
{
    if (via_attributes() != by_hand() || by_hand() != bare()) {
        std::fprintf(stderr, "attributes benchmark: fairness check FAILED - the colours differ\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
