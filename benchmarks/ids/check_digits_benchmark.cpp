//
//  check_digits_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  validate_isin and validate_iban against the same checks written by hand (the same character tests, the same Luhn
//  sum right to left, the same MOD 97 digit by digit), over 4,096 identifiers of which a quarter have a wrong check
//  digit. main() runs a fairness check first and refuses to benchmark if the two sides disagree on any identifier
//  (CLAUDE.md "Zero overhead").
//

#include <snicholls/core/ids/check_digits.hpp>

#include <benchmark/benchmark.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace snicholls;

constexpr std::string_view alnum = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";

struct identifiers {
    std::vector<std::string> isin, iban;
};

identifiers make()
{
    identifiers out;
    std::uint64_t x = 0x452821e638d01377ULL;
    auto next = [&] { x ^= x << 13; x ^= x >> 7; x ^= x << 17; return x; };
    for (int i = 0; i < 4096; ++i) {
        std::string body = "US";
        for (int k = 0; k < 9; ++k) body += alnum[next() % alnum.size()];
        std::string isin = body + *isin_check_digit(body);
        std::string bban;
        for (int k = 0; k < 18; ++k) bban += alnum[next() % 10];
        const auto ck = *iban_check_digits("GB", bban);
        std::string iban = std::string("GB") + ck[0] + ck[1] + bban;
        if (i % 4 == 0) {
            isin.back() = isin.back() == '9' ? '0' : static_cast<char>(isin.back() + 1);
            iban.back() = iban.back() == '9' ? '0' : static_cast<char>(iban.back() + 1);
        }
        out.isin.push_back(isin);
        out.iban.push_back(iban);
    }
    return out;
}

const identifiers& data()
{
    static const identifiers d = make();
    return d;
}

bool digit(char c) { return c >= '0' && c <= '9'; }
bool upper(char c) { return c >= 'A' && c <= 'Z'; }
int value(char c) { return digit(c) ? c - '0' : c - 'A' + 10; }

bool isin_by_hand(std::string_view s) noexcept
{
    if (s.size() != 12)
        return false;
    for (const char c : s)
        if (!digit(c) && !upper(c))
            return false;
    if (!upper(s[0]) || !upper(s[1]) || !digit(s[11]))
        return false;
    int sum = 0;
    bool twice = false;
    const auto add = [&](int d) {
        if (twice)
            d = d * 2 > 9 ? d * 2 - 9 : d * 2;
        sum += d;
        twice = !twice;
    };
    for (std::size_t i = s.size(); i-- > 0;) {
        const int v = value(s[i]);
        add(v % 10);
        if (v > 9)
            add(v / 10);
    }
    return sum % 10 == 0;
}

bool iban_by_hand(std::string_view s) noexcept
{
    if (s.size() < 5 || s.size() > 34)
        return false;
    for (const char c : s)
        if (!digit(c) && !upper(c))
            return false;
    if (!upper(s[0]) || !upper(s[1]) || !digit(s[2]) || !digit(s[3]))
        return false;
    const int check = value(s[2]) * 10 + value(s[3]);
    if (check < 2 || check > 98)
        return false;
    int r = 0;
    const auto feed = [&](char c) {
        const int v = value(c);
        r = v > 9 ? (r * 100 + v) % 97 : (r * 10 + v) % 97;
    };
    for (const char c : s.substr(4)) feed(c);
    for (const char c : s.substr(0, 4)) feed(c);
    return r == 1;
}

void isin_library(benchmark::State& state)
{
    for (auto _ : state)
        for (const auto& s : data().isin) {
            bool ok = validate_isin(s).is_ok();
            benchmark::DoNotOptimize(ok);
        }
    state.SetItemsProcessed(state.iterations() * 4096);
}

void isin_hand(benchmark::State& state)
{
    for (auto _ : state)
        for (const auto& s : data().isin) {
            bool ok = isin_by_hand(s);
            benchmark::DoNotOptimize(ok);
        }
    state.SetItemsProcessed(state.iterations() * 4096);
}

void iban_library(benchmark::State& state)
{
    for (auto _ : state)
        for (const auto& s : data().iban) {
            bool ok = validate_iban(s).is_ok();
            benchmark::DoNotOptimize(ok);
        }
    state.SetItemsProcessed(state.iterations() * 4096);
}

void iban_hand(benchmark::State& state)
{
    for (auto _ : state)
        for (const auto& s : data().iban) {
            bool ok = iban_by_hand(s);
            benchmark::DoNotOptimize(ok);
        }
    state.SetItemsProcessed(state.iterations() * 4096);
}

BENCHMARK(isin_library);
BENCHMARK(isin_hand);
BENCHMARK(iban_library);
BENCHMARK(iban_hand);

} // namespace

int main(int argc, char** argv)
{
    int valid = 0;
    for (std::size_t i = 0; i < 4096; ++i) {
        const bool a = validate_isin(data().isin[i]).is_ok(), b = validate_iban(data().iban[i]).is_ok();
        valid += a;
        if (a != isin_by_hand(data().isin[i]) || b != iban_by_hand(data().iban[i]) || a != (i % 4 != 0) || b != (i % 4 != 0)) {
            std::fprintf(stderr, "check_digits_benchmark: the fairness check failed: the two sides disagree\n");
            return 1;
        }
    }
    std::printf("fairness check: %d of 4096 identifiers valid on both sides\n", valid);
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
