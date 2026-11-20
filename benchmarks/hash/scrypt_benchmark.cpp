//
//  scrypt_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  scrypt's memory-hard middle, ROMix (more than 99 % of its time), in core against the same ROMix written by hand
//  (memcpy block copies, Salsa20/8 written out), over the same working memory: N = 16384, r = 8 (16 MiB), one
//  lane. The whole snicholls::scrypt runs beside them, and OpenSSL's EVP_PBE_scrypt when CMake found it
//  (informational). main() runs a fairness check first and refuses to benchmark if any results disagree.
//

#include <snicholls/core/hash/scrypt.hpp>

#include "by_hand.hpp"

#include <benchmark/benchmark.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

constexpr std::uint64_t n = 16384;
constexpr std::size_t r = 8;

std::vector<std::uint32_t> lane()
{
    std::vector<std::uint32_t> x(32 * r);
    std::uint32_t s = 0x9e3779b9u;
    for (auto& w : x)
        w = s = s * 1664525u + 1013904223u;
    return x;
}

std::vector<std::uint32_t> ro_mix_core()
{
    static std::vector<std::uint32_t> v(n * 32 * r), y(32 * r);
    auto x = lane();
    snicholls::detail::scrypt_ro_mix(x.data(), v.data(), y.data(), r, n);
    return x;
}

std::vector<std::uint32_t> ro_mix_hand()
{
    static std::vector<std::uint32_t> v(n * 32 * r), y(32 * r);
    auto x = lane();
    by_hand::ro_mix(x.data(), v.data(), y.data(), r, n);
    return x;
}

std::array<std::byte, 64> scrypt_core()
{
    static std::vector<std::uint32_t> work(snicholls::scrypt_work_words({n, r, 1}).unwrap_unchecked());
    std::array<std::byte, 64> out{};
    (void)snicholls::scrypt("pleaseletmein", "SodiumChloride", {n, r, 1}, work, out);
    return out;
}

void BM_ScryptRoMix_Core(benchmark::State& s)
{
    for (auto _ : s)
        benchmark::DoNotOptimize(ro_mix_core());
}
void BM_ScryptRoMix_HandWritten(benchmark::State& s)
{
    for (auto _ : s)
        benchmark::DoNotOptimize(ro_mix_hand());
}
void BM_Scrypt_Core(benchmark::State& s)
{
    for (auto _ : s)
        benchmark::DoNotOptimize(scrypt_core());
}
BENCHMARK(BM_ScryptRoMix_Core)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_ScryptRoMix_HandWritten)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Scrypt_Core)->Unit(benchmark::kMillisecond);

#if SNICHOLLS_CORE_BENCH_OPENSSL
std::array<std::byte, 64> scrypt_openssl()
{
    std::array<std::byte, 64> out{};
    EVP_PBE_scrypt("pleaseletmein", 13, reinterpret_cast<const unsigned char*>("SodiumChloride"), 14, n, r, 1, 0,
                   reinterpret_cast<unsigned char*>(out.data()), out.size());
    return out;
}
void BM_Scrypt_OpenSSL(benchmark::State& s)
{
    for (auto _ : s)
        benchmark::DoNotOptimize(scrypt_openssl());
}
BENCHMARK(BM_Scrypt_OpenSSL)->Unit(benchmark::kMillisecond);
#endif

bool fair()
{
    const auto core = scrypt_core();
    // RFC 7914's third vector
    const auto expected = snicholls::digest<64>::from_hex(
        "7023bdcb3afd7348461c06cd81fd38ebfda8fbba904f8e3ea9b543f6545da1f2d5432955613f0fcf62d49705242a9af9e61e85dc0d651e40dfcf017b45575887");
    bool ok = ro_mix_core() == ro_mix_hand() && expected && core == expected->bytes;
#if SNICHOLLS_CORE_BENCH_OPENSSL
    ok = ok && scrypt_openssl() == core;
#endif
    return ok;
}

} // namespace

int main(int argc, char** argv)
{
    if (!fair()) {
        std::fprintf(stderr, "scrypt benchmark: fairness check FAILED - the results disagree; not benchmarking\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
