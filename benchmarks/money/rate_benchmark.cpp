//
//  rate_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  snicholls::convert against the same scaled multiply-divide and half-even rounding written by hand (the same
//  checked products, the same detail::mul_div, the same rounding test), on 4,096 amounts. main() runs a fairness check
//  first and refuses to benchmark if the two sides disagree on any amount (CLAUDE.md "Zero overhead"). A cash dividend
//  to 100,000 holders (convert per line, or uint256's checked_mul and divmod with rounds_away, and sum_residuals over the
//  run) runs against the same arithmetic in 64 bits.
//

#include <snicholls/core/detail/mul_div.hpp>
#include <snicholls/core/money/rate.hpp>

#include <benchmark/benchmark.h>

#include <cstddef>
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
            bool same = m.asset_of() == eth_usd.from();
            bool ok = by_hand(m, units, negative);
            benchmark::DoNotOptimize(same);
            benchmark::DoNotOptimize(ok);
            benchmark::DoNotOptimize(units);
        }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(amounts().size()));
}

BENCHMARK(convert_library);
BENCHMARK(convert_hand);

// A cash dividend of 0.2534 AUD a share to 100,000 holders, each line rounded half-even, with the exact total of
// what rounding left: every operand fits in 64 bits, the case uint256's fast path is for.
const asset share = *make_asset("SHARE", 0);
const asset aud = *make_asset("AUD", 2);
const rate per_share = *rate::parse(share, aud, "0.2534");
// run-time values, so no side divides by a constant the compiler turns into a multiplication
std::uint64_t cents_n = 253400;   // 2534 * 10^2
std::uint64_t cents_d = 10000;    // 10^4 * 10^0

std::vector<std::uint64_t> make_register()
{
    std::vector<std::uint64_t> out(100000);
    std::uint64_t x = 0xa4093822299f31d0ULL;
    for (auto& h : out) {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        h = 1 + (x >> 44);   // 1 to about a million shares
    }
    return out;
}

const std::vector<std::uint64_t>& holders()
{
    static const std::vector<std::uint64_t> h = make_register();
    return h;
}

struct dividend_run {
    uint256 paid;
    residual left;
};

// through the library: convert per holder, sum_residuals over the run
dividend_run dividend_library(std::vector<residual>& lines)
{
    const auto& hs = holders();
    uint256 paid = 0;
    for (std::size_t i = 0; i < hs.size(); ++i) {
        const auto c = convert(money::from_units(share, hs[i]), per_share, rounding::half_even);
        paid += c->converted.magnitude();
        lines[i] = c->residual;
    }
    return {paid, *sum_residuals(lines, cents_d)};
}

// through uint256 directly, as an entitlement engine writes it: checked_mul, divmod, rounds_away, sum_residuals
dividend_run dividend_uint256(std::vector<residual>& lines)
{
    const auto& hs = holders();
    uint256 paid = 0;
    const uint256 n = cents_n, d = cents_d;
    for (std::size_t i = 0; i < hs.size(); ++i) {
        const auto [q, r] = uint256::checked_mul(hs[i], n)->divmod(d);
        const residual fraction{false, r, d};
        const bool up = rounds_away(q, fraction, rounding::half_even);
        paid += up ? q + 1 : q;
        lines[i] = up ? residual{true, d - r, d} : fraction;
    }
    return {paid, *sum_residuals(lines, d)};
}

// by hand, in 64 bits: the same product, division, half-even test, a residual kept per line, then the signed sum
dividend_run dividend_hand(std::vector<std::int64_t>& lines)
{
    const auto& hs = holders();
    const std::uint64_t n = cents_n, d = cents_d;
    std::uint64_t paid = 0;
    for (std::size_t i = 0; i < hs.size(); ++i) {
        const std::uint64_t p = hs[i] * n, q = p / d, r = p % d, rest = d - r;
        const bool up = r != 0 && (r > rest || (r == rest && (q & 1) != 0));
        paid += up ? q + 1 : q;
        lines[i] = up ? -static_cast<std::int64_t>(rest) : static_cast<std::int64_t>(r);
    }
    std::int64_t left = 0;
    for (const std::int64_t l : lines)
        left += l;
    const auto magnitude = static_cast<std::uint64_t>(left < 0 ? -left : left);
    return {paid, residual{left < 0, magnitude, d}};
}

void dividend_100k_library(benchmark::State& state)
{
    std::vector<residual> lines(holders().size());
    for (auto _ : state) {
        auto run = dividend_library(lines);
        benchmark::DoNotOptimize(run);
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(holders().size()));
}

void dividend_100k_uint256(benchmark::State& state)
{
    std::vector<residual> lines(holders().size());
    for (auto _ : state) {
        auto run = dividend_uint256(lines);
        benchmark::DoNotOptimize(run);
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(holders().size()));
}

void dividend_100k_hand(benchmark::State& state)
{
    std::vector<std::int64_t> lines(holders().size());
    for (auto _ : state) {
        auto run = dividend_hand(lines);
        benchmark::DoNotOptimize(run);
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(holders().size()));
}

BENCHMARK(dividend_100k_library)->Unit(benchmark::kMillisecond);
BENCHMARK(dividend_100k_uint256)->Unit(benchmark::kMillisecond);
BENCHMARK(dividend_100k_hand)->Unit(benchmark::kMillisecond);

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
    std::vector<residual> lines(holders().size());
    const auto library = dividend_library(lines);
    std::vector<std::int64_t> hand_lines(holders().size());
    const auto hand = dividend_hand(hand_lines);
    const auto direct = dividend_uint256(lines);
    if (library.paid != hand.paid || library.left != hand.left || direct.paid != hand.paid || direct.left != hand.left) {
        std::fprintf(stderr, "rate_benchmark: the dividend fairness check failed: the two sides disagree\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
