//
//  kdf_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  snicholls::pbkdf2_hmac<sha256> and <sha512> (and so hmac<H>) against PBKDF2 written by hand over the hand SHA-2
//  of by_hand.hpp, with the key's inner and outer states primed once as core's copied hmac does: 1,000 iterations,
//  a 64-byte key. OpenSSL's PKCS5_PBKDF2_HMAC beside them when CMake found it (informational). main() runs a
//  fairness check first and refuses to benchmark if any derived keys disagree.
//

#include <snicholls/core/hash/kdf.hpp>
#include <snicholls/core/hash/sha2.hpp>

#include "by_hand.hpp"

#include <benchmark/benchmark.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string_view>
#include <vector>

namespace {

constexpr std::string_view password = "correct horse battery staple";
constexpr std::string_view salt = "16 bytes of salt";
constexpr std::uint32_t iterations = 1000;

const auto* pw = reinterpret_cast<const unsigned char*>(password.data());
const auto* sa = reinterpret_cast<const unsigned char*>(salt.data());

std::vector<std::byte> core256()
{
    std::vector<std::byte> out(64);
    (void)snicholls::pbkdf2_hmac<snicholls::sha256>(password, salt, iterations, out);
    return out;
}
std::vector<std::byte> core512()
{
    std::vector<std::byte> out(64);
    (void)snicholls::pbkdf2_hmac<snicholls::sha512>(password, salt, iterations, out);
    return out;
}
by_hand::bytes hand256()
{
    const by_hand::hmac256 m(by_hand::sha256_iv, pw, password.size(), by_hand::sha256_block);
    return by_hand::pbkdf2<by_hand::hmac256, 32>(m, sa, salt.size(), iterations, 64);
}
by_hand::bytes hand512()
{
    const by_hand::hmac512 m(by_hand::sha512_iv, pw, password.size(), by_hand::sha512_block);
    return by_hand::pbkdf2<by_hand::hmac512, 64>(m, sa, salt.size(), iterations, 64);
}

void BM_Pbkdf2Sha256_Core(benchmark::State& s)
{
    for (auto _ : s)
        benchmark::DoNotOptimize(core256());
}
void BM_Pbkdf2Sha256_HandWritten(benchmark::State& s)
{
    for (auto _ : s)
        benchmark::DoNotOptimize(hand256());
}
void BM_Pbkdf2Sha512_Core(benchmark::State& s)
{
    for (auto _ : s)
        benchmark::DoNotOptimize(core512());
}
void BM_Pbkdf2Sha512_HandWritten(benchmark::State& s)
{
    for (auto _ : s)
        benchmark::DoNotOptimize(hand512());
}
BENCHMARK(BM_Pbkdf2Sha256_Core);
BENCHMARK(BM_Pbkdf2Sha256_HandWritten);
BENCHMARK(BM_Pbkdf2Sha512_Core);
BENCHMARK(BM_Pbkdf2Sha512_HandWritten);

#if SNICHOLLS_CORE_BENCH_OPENSSL
by_hand::bytes openssl(const EVP_MD* md)
{
    by_hand::bytes out(64);
    PKCS5_PBKDF2_HMAC(password.data(), static_cast<int>(password.size()), sa, static_cast<int>(salt.size()), static_cast<int>(iterations), md,
                      static_cast<int>(out.size()), out.data());
    return out;
}
void BM_Pbkdf2Sha256_OpenSSL(benchmark::State& s)
{
    for (auto _ : s)
        benchmark::DoNotOptimize(openssl(EVP_sha256()));
}
void BM_Pbkdf2Sha512_OpenSSL(benchmark::State& s)
{
    for (auto _ : s)
        benchmark::DoNotOptimize(openssl(EVP_sha512()));
}
BENCHMARK(BM_Pbkdf2Sha256_OpenSSL);
BENCHMARK(BM_Pbkdf2Sha512_OpenSSL);
#endif

bool same(const std::vector<std::byte>& a, const by_hand::bytes& b)
{
    if (a.size() != b.size())
        return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (static_cast<unsigned char>(a[i]) != b[i])
            return false;
    return true;
}

bool fair()
{
    const auto a = core256(), b = core512();
    bool ok = same(a, hand256()) && same(b, hand512());
#if SNICHOLLS_CORE_BENCH_OPENSSL
    ok = ok && same(a, openssl(EVP_sha256())) && same(b, openssl(EVP_sha512()));
#endif
    return ok;
}

} // namespace

int main(int argc, char** argv)
{
    if (!fair()) {
        std::fprintf(stderr, "kdf benchmark: fairness check FAILED - the derived keys disagree; not benchmarking\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
