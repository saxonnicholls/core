//
//  token_verifier_reference_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Verifying a token through the reference verifier against the same work written by hand: split the token at its
//  last dot, HMAC-SHA256 the body under the same key, compare the hexadecimal tag in constant time, parse the serial
//  and copy the claims out. Same key, same token, same claims. The fairness check refuses to run unless both accept
//  the token with equal claims and both refuse an altered one.
//

#include <snicholls/core/hash/hmac.hpp>
#include <snicholls/core/hash/secure.hpp>
#include <snicholls/core/hash/sha2.hpp>
#include <snicholls/core/interfaces/token_verifier_reference.hpp>

#include <benchmark/benchmark.h>

#include <charconv>
#include <cstddef>
#include <cstdio>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace snicholls;

const std::string key = "a benchmark-only HS256 key of 32 bytes or more";

token_claims the_claims()
{
    token_claims c;
    c.subject = uuid::from_u64(0xa11ce, 1);
    c.roles = {"admin"};
    c.scopes = {"payments.refund"};
    return c;
}

// The hand-written equivalent: one stored claim set at serial 0.
std::optional<token_claims> by_hand(std::string_view token, const std::vector<token_claims>& issued)
{
    const auto dot = token.rfind('.');
    if (!token.starts_with("mt1.") || dot == std::string_view::npos || dot < 4)
        return std::nullopt;
    const auto body = token.substr(0, dot);
    if (!constant_time_equal(hmac<sha256>::mac(key, body).hex().view(), token.substr(dot + 1)))
        return std::nullopt;
    std::size_t serial = 0;
    const auto digits = body.substr(4);
    const auto [end, ec] = std::from_chars(digits.data(), digits.data() + digits.size(), serial);
    if (ec != std::errc{} || end != digits.data() + digits.size() || serial >= issued.size())
        return std::nullopt;
    return issued[serial];
}

struct fixture {
    memory_token_verifier verifier{secret<std::string>{key}, manual_utc_clock{utc_s::from_unix_seconds(1'800'000'000)}};
    std::vector<token_claims> issued{the_claims()};
    std::string token = verifier.issue(the_claims());
};

bool fair()
{
    fixture f;
    const auto ours = f.verifier.verify(f.token);
    const auto hand = by_hand(f.token, f.issued);
    std::string altered = f.token;
    altered.back() = altered.back() == '0' ? '1' : '0';
    return ours.is_ok() && hand && *ours == *hand && f.verifier.verify(altered).is_err() && !by_hand(altered, f.issued);
}

void bm_reference_verify(benchmark::State& state)
{
    fixture f;
    for (auto _ : state) {
        auto r = f.verifier.verify(f.token);
        benchmark::DoNotOptimize(r);
    }
}
BENCHMARK(bm_reference_verify);

void bm_hand_written_verify(benchmark::State& state)
{
    fixture f;
    for (auto _ : state) {
        auto r = by_hand(f.token, f.issued);
        benchmark::DoNotOptimize(r);
    }
}
BENCHMARK(bm_hand_written_verify);

} // namespace

int main(int argc, char** argv)
{
    if (!fair()) {
        std::fprintf(stderr, "token_verifier_reference benchmark: fairness check FAILED - the two disagree; not benchmarking\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
