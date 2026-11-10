//
//  uint256_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  snicholls::uint256 addition and multiplication against the same limb loops
//  written out by hand (the same carries, the same 128-bit products from
//  detail/wide_mul.hpp, in the same order), on 4,096 pairs of values of every
//  width. Division runs alone: its hand-written equivalent is the same Knuth
//  algorithm. main() runs a fairness check first and refuses to benchmark if the
//  two sides disagree on any pair (CLAUDE.md "Zero overhead").
//

#include <snicholls/core/detail/wide_mul.hpp>
#include <snicholls/core/uint256.hpp>

#include <benchmark/benchmark.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

using snicholls::uint256;
using limbs = std::array<std::uint64_t, 4>;

struct pair_of {
    uint256 a;
    uint256 b;
};

std::vector<pair_of> make_pairs()
{
    std::vector<pair_of> out(4096);
    std::uint64_t x = 0x243f6a8885a308d3ULL;   // a fixed xorshift stream: identical data for both
    auto next = [&] { x ^= x << 13; x ^= x >> 7; x ^= x << 17; return x; };
    std::size_t i = 0;
    for (auto& p : out) {
        p.a = uint256::from_limbs(next(), next(), i % 3 ? next() : 0, i % 4 ? next() : 0);
        p.b = uint256::from_limbs(next() | 1, i % 2 ? next() : 0, i % 5 ? next() : 0, i % 7 ? next() : 0);
        ++i;
    }
    return out;
}

const std::vector<pair_of> pairs = make_pairs();

// By hand: the carry loop and the schoolbook product over plain arrays
limbs add_by_hand(const limbs& a, const limbs& b) noexcept
{
    limbs r{};
    std::uint64_t carry = 0;
    for (std::size_t i = 0; i < 4; ++i) {
        const std::uint64_t sum = a[i] + b[i];
        std::uint64_t c = sum < a[i] ? 1u : 0u;
        const std::uint64_t with_carry = sum + carry;
        c += with_carry < sum ? 1u : 0u;
        r[i] = with_carry;
        carry = c;
    }
    return r;
}

limbs mul_by_hand(const limbs& a, const limbs& b) noexcept
{
    std::array<std::uint64_t, 8> r{};
    for (std::size_t i = 0; i < 4; ++i) {
        std::uint64_t carry = 0;
        for (std::size_t j = 0; j < 4; ++j) {
            const snicholls::detail::wide_product p = snicholls::detail::umul128(a[i], b[j]);
            std::uint64_t low = r[i + j] + p.low;
            std::uint64_t c = low < p.low ? 1u : 0u;
            const std::uint64_t with_carry = low + carry;
            c += with_carry < low ? 1u : 0u;
            r[i + j] = with_carry;
            carry = p.high + c;
        }
        r[i + 4] = carry;
    }
    return {r[0], r[1], r[2], r[3]};
}

void BM_Uint256_Add_Core(benchmark::State& state)
{
    for (auto _ : state)
        for (const auto& p : pairs)
            benchmark::DoNotOptimize(p.a + p.b);
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(pairs.size()));
}
BENCHMARK(BM_Uint256_Add_Core);

void BM_Uint256_Add_HandWritten(benchmark::State& state)
{
    for (auto _ : state)
        for (const auto& p : pairs)
            benchmark::DoNotOptimize(add_by_hand(p.a.limbs(), p.b.limbs()));
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(pairs.size()));
}
BENCHMARK(BM_Uint256_Add_HandWritten);

void BM_Uint256_Mul_Core(benchmark::State& state)
{
    for (auto _ : state)
        for (const auto& p : pairs)
            benchmark::DoNotOptimize(p.a * p.b);
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(pairs.size()));
}
BENCHMARK(BM_Uint256_Mul_Core);

void BM_Uint256_Mul_HandWritten(benchmark::State& state)
{
    for (auto _ : state)
        for (const auto& p : pairs)
            benchmark::DoNotOptimize(mul_by_hand(p.a.limbs(), p.b.limbs()));
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(pairs.size()));
}
BENCHMARK(BM_Uint256_Mul_HandWritten);

void BM_Uint256_Divmod_Core(benchmark::State& state)
{
    for (auto _ : state)
        for (const auto& p : pairs)
            benchmark::DoNotOptimize(p.a.divmod(p.b));
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(pairs.size()));
}
BENCHMARK(BM_Uint256_Divmod_Core);

void BM_Uint256_ToString_Core(benchmark::State& state)
{
    for (auto _ : state)
        for (const auto& p : pairs)
            benchmark::DoNotOptimize(p.a.to_string());
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(pairs.size()));
}
BENCHMARK(BM_Uint256_ToString_Core);

bool fair()
{
    for (const auto& p : pairs) {
        if ((p.a + p.b).limbs() != add_by_hand(p.a.limbs(), p.b.limbs()))
            return false;
        if ((p.a * p.b).limbs() != mul_by_hand(p.a.limbs(), p.b.limbs()))
            return false;
        const auto [q, r] = p.a.divmod(p.b);
        if (q * p.b + r != p.a || !(r < p.b))
            return false;
    }
    return true;
}

} // namespace

int main(int argc, char** argv)
{
    if (!fair()) {
        std::fprintf(stderr, "uint256 benchmark: fairness check FAILED - the two sides disagree; not benchmarking\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
