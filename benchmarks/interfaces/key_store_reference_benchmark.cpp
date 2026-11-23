//
//  key_store_reference_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  HS256 signing through the reference signer against core's hmac<sha256>::mac written by hand, on the same key
//  and the same JWS signing input, both returning the tag's bytes. The fairness check refuses to run unless the
//  two tags are identical.
//

#include <snicholls/core/hash/hmac.hpp>
#include <snicholls/core/hash/sha2.hpp>
#include <snicholls/core/interfaces/key_store_reference.hpp>

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <span>
#include <string>
#include <vector>

namespace {

using namespace snicholls;

const std::string input = "eyJhbGciOiJIUzI1NiIsImtpZCI6ImtleS0xLjEifQ.eyJzdWIiOiJ1c2VyLTEiLCJleHAiOjE4MDAwMDA5MDB9";

std::string the_key()   // memory_key_store::deterministic()'s first key: bytes 0, 1, ..., 31
{
    std::string k(memory_key_store::key_bytes, '\0');
    for (std::size_t i = 0; i < k.size(); ++i)
        k[i] = static_cast<char>(i);
    return k;
}

std::span<const std::byte> bytes() { return std::as_bytes(std::span<const char>(input.data(), input.size())); }

bool fair()
{
    auto store = memory_key_store::deterministic();
    const auto id = store.create({signature_algorithm::hs256, "bench"})->key_id;
    const auto ours = store.sign({id.view(), bytes(), payload_kind::message});
    const auto hand = hmac<sha256>::mac(the_key(), input);
    return ours.is_ok() && std::equal(ours->bytes.begin(), ours->bytes.end(), hand.bytes.begin(), hand.bytes.end());
}

void bm_reference_sign(benchmark::State& state)
{
    auto store = memory_key_store::deterministic();
    const auto id = store.create({signature_algorithm::hs256, "bench"})->key_id;
    for (auto _ : state) {
        auto s = store.sign({id.view(), bytes(), payload_kind::message});
        benchmark::DoNotOptimize(s);
    }
}
BENCHMARK(bm_reference_sign);

void bm_hand_written_hmac(benchmark::State& state)
{
    const std::string key = the_key();
    for (auto _ : state) {
        const auto tag = hmac<sha256>::mac(key, input);
        std::vector<std::byte> out(tag.bytes.begin(), tag.bytes.end());   // the reference returns its bytes too
        benchmark::DoNotOptimize(out);
    }
}
BENCHMARK(bm_hand_written_hmac);

} // namespace

int main(int argc, char** argv)
{
    if (!fair()) {
        std::fprintf(stderr, "key_store_reference benchmark: fairness check FAILED - the tags differ; not benchmarking\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
