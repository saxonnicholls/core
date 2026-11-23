//
//  inplace_string_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  inplace_string<64>::make and == against the same copy and compare of a hand-written record (64 characters, zero
//  tail, a one-byte length; the same length check, the same copy, the same compare of the text), over 4,096
//  contract ids of 8 to 40 characters. main() runs a fairness check first and refuses to benchmark if the two
//  sides disagree on any id (CLAUDE.md "Zero overhead").
//

#include <snicholls/core/containers/inplace_string.hpp>

#include <benchmark/benchmark.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

namespace {

using snicholls::inplace_string;

const std::vector<std::string>& ids()
{
    static const std::vector<std::string> v = [] {
        std::vector<std::string> out;
        std::uint64_t x = 0xa4093822299f31d0ULL;
        auto next = [&] { x ^= x << 13; x ^= x >> 7; x ^= x << 17; return x; };
        for (int i = 0; i < 4096; ++i) {
            std::string s = "contract-";
            const auto n = 8 + next() % 32;
            while (s.size() < n) s += static_cast<char>('a' + next() % 26);
            out.push_back(s);
        }
        return out;
    }();
    return v;
}

struct by_hand {   // the record written out: characters, zero tail, length
    char chars[64];
    std::uint8_t size;
};

bool make_by_hand(std::string_view s, by_hand& out) noexcept
{
    if (s.size() > 64)
        return false;
    out = by_hand{};
    std::memcpy(out.chars, s.data(), s.size());
    out.size = static_cast<std::uint8_t>(s.size());
    return true;
}

bool equal_by_hand(const by_hand& a, const by_hand& b) noexcept
{
    return std::string_view(a.chars, a.size) == std::string_view(b.chars, b.size);
}

void make_library(benchmark::State& state)
{
    for (auto _ : state)
        for (const auto& s : ids()) {
            auto r = inplace_string<64>::make(s);
            benchmark::DoNotOptimize(r);
        }
    state.SetItemsProcessed(state.iterations() * 4096);
}

void make_hand(benchmark::State& state)
{
    for (auto _ : state)
        for (const auto& s : ids()) {
            by_hand r;
            bool ok = make_by_hand(s, r);
            benchmark::DoNotOptimize(ok);
            benchmark::DoNotOptimize(r);
        }
    state.SetItemsProcessed(state.iterations() * 4096);
}

void equal_library(benchmark::State& state)
{
    std::vector<inplace_string<64>> v;
    for (const auto& s : ids()) v.push_back(*inplace_string<64>::make(s));
    for (auto _ : state)
        for (std::size_t i = 1; i < v.size(); ++i) {
            bool same = v[i] == v[i - 1];
            benchmark::DoNotOptimize(same);
        }
    state.SetItemsProcessed(state.iterations() * 4095);
}

void equal_hand(benchmark::State& state)
{
    std::vector<by_hand> v(ids().size());
    for (std::size_t i = 0; i < v.size(); ++i) make_by_hand(ids()[i], v[i]);
    for (auto _ : state)
        for (std::size_t i = 1; i < v.size(); ++i) {
            bool same = equal_by_hand(v[i], v[i - 1]);
            benchmark::DoNotOptimize(same);
        }
    state.SetItemsProcessed(state.iterations() * 4095);
}

BENCHMARK(make_library);
BENCHMARK(make_hand);
BENCHMARK(equal_library);
BENCHMARK(equal_hand);

} // namespace

int main(int argc, char** argv)
{
    for (const auto& s : ids()) {
        const auto r = inplace_string<64>::make(s);
        by_hand h;
        if (!r || !make_by_hand(s, h) || std::memcmp(r->chars_, h.chars, 64) != 0 || r->size() != h.size) {
            std::fprintf(stderr, "inplace_string_benchmark: the fairness check failed: the two sides disagree\n");
            return 1;
        }
    }
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
