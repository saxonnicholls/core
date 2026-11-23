//
//  money_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  snicholls::add on money against the same sign-magnitude addition written by hand on uint256 (the same checked add,
//  the same compare and subtract, in the same order), on 4,096 pairs of amounts of one asset. main() runs a fairness
//  check first and refuses to benchmark if the two sides disagree on any pair (CLAUDE.md "Zero overhead").
//

#include <snicholls/core/money/money.hpp>

#include <benchmark/benchmark.h>

#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

using namespace snicholls;

const asset usd = *make_asset("USD", 2);

struct pair_of {
    money a, b;
};

std::vector<pair_of> make_pairs()
{
    std::vector<pair_of> out(4096);
    std::uint64_t x = 0x243f6a8885a308d3ULL;
    auto next = [&] { x ^= x << 13; x ^= x >> 7; x ^= x << 17; return x; };
    for (auto& p : out) {
        p.a = money::from_units(usd, uint256::from_limbs(next(), next() >> 3), (next() & 1) != 0);
        p.b = money::from_units(usd, uint256::from_limbs(next(), next() >> 3), (next() & 1) != 0);
    }
    return out;
}

struct by_hand {
    bool ok;
    bool negative;
    uint256 magnitude;
};

by_hand add_by_hand(bool an, const uint256& a, bool bn, const uint256& b) noexcept
{
    if (an == bn) {
        const auto s = uint256::checked_add(a, b);
        return s ? by_hand{true, an, *s} : by_hand{false, false, {}};
    }
    if (a >= b)
        return {true, an && a != b, a - b};
    return {true, bn, b - a};
}

const std::vector<pair_of>& pairs()
{
    static const std::vector<pair_of> p = make_pairs();
    return p;
}

void money_add(benchmark::State& state)
{
    for (auto _ : state)
        for (const auto& p : pairs()) {
            auto r = add(p.a, p.b);
            benchmark::DoNotOptimize(r);
        }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(pairs().size()));
}

void hand_add(benchmark::State& state)
{
    for (auto _ : state)
        for (const auto& p : pairs()) {
            bool same = p.a.asset_of() == p.b.asset_of();
            auto r = add_by_hand(p.a.is_negative(), p.a.magnitude(), p.b.is_negative(), p.b.magnitude());
            benchmark::DoNotOptimize(same);
            benchmark::DoNotOptimize(r);
        }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(pairs().size()));
}

BENCHMARK(money_add);
BENCHMARK(hand_add);

} // namespace

int main(int argc, char** argv)
{
    for (const auto& p : pairs()) {
        const auto lib = add(p.a, p.b);
        const auto hand = add_by_hand(p.a.is_negative(), p.a.magnitude(), p.b.is_negative(), p.b.magnitude());
        if (lib.is_ok() != hand.ok || (lib && (lib->magnitude() != hand.magnitude || lib->is_negative() != hand.negative))) {
            std::fprintf(stderr, "money_benchmark: the fairness check failed: the two sides disagree\n");
            return 1;
        }
    }
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
