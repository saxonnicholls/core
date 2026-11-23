//
//  rate_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  snicholls::convert against the same scaled multiply-divide and half-even rounding written by hand (the same
//  checked products, the same detail::mul_div, the same rounding test), on 4,096 amounts. main() runs a fairness check
//  first and refuses to benchmark if the two sides disagree on any amount (CLAUDE.md "Zero overhead").
//

#include <snicholls/core/detail/mul_div.hpp>
#include <snicholls/core/money/rate.hpp>

#include <benchmark/benchmark.h>

#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

using namespace snicholls;

const asset eth = *make_asset("ETH", 18);
const asset usd = *make_asset("USD", 2);
const rate eth_usd = *rate::parse(eth, usd, "3000.123456789012345678");

std::vector<money> make_amounts()
{
    std::vector<money> out(4096);
    std::uint64_t x = 0x13198a2e03707344ULL;
    auto next = [&] { x ^= x << 13; x ^= x >> 7; x ^= x << 17; return x; };
    for (auto& m : out)
        m = money::from_units(eth, uint256::from_limbs(next(), next() >> 40), (next() & 1) != 0);
    return out;
}

const std::vector<money>& amounts()
{
    static const std::vector<money> a = make_amounts();
    return a;
}

// the same steps convert takes, half_even, written out
bool by_hand(const money& m, uint256& units, bool& negative) noexcept
{
    const auto n = uint256::checked_mul(eth_usd.numerator(), uint256::pow10(2));
    const auto d = uint256::checked_mul(eth_usd.denominator(), uint256::pow10(18));
    if (!n || !d)
        return false;
    const auto qr = detail::mul_div(m.magnitude(), *n, *d);
    if (!qr)
        return false;
    auto [q, r] = *qr;
    const uint256 rest = *d - r;
    if (!r.is_zero() && (r > rest || (r == rest && q.bit(0))))
        q = q + 1;
    units = q;
    negative = m.is_negative() && !q.is_zero();
    return true;
}

void convert_library(benchmark::State& state)
{
    for (auto _ : state)
        for (const auto& m : amounts()) {
            auto c = convert(m, eth_usd, rounding::half_even);
            benchmark::DoNotOptimize(c);
        }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(amounts().size()));
}

void convert_hand(benchmark::State& state)
{
    for (auto _ : state)
        for (const auto& m : amounts()) {
            uint256 units;
            bool negative = false;
            const bool same = m.asset_of() == eth_usd.from();
            const bool ok = by_hand(m, units, negative);
            benchmark::DoNotOptimize(same);
            benchmark::DoNotOptimize(ok);
            benchmark::DoNotOptimize(units);
        }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(amounts().size()));
}

BENCHMARK(convert_library);
BENCHMARK(convert_hand);

} // namespace

int main(int argc, char** argv)
{
    for (const auto& m : amounts()) {
        const auto c = convert(m, eth_usd, rounding::half_even);
        uint256 units;
        bool negative = false;
        if (!c || !by_hand(m, units, negative) || c->converted.magnitude() != units ||
            c->converted.is_negative() != negative) {
            std::fprintf(stderr, "rate_benchmark: the fairness check failed: the two sides disagree\n");
            return 1;
        }
    }
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
