//
//  wyhash_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/hash/wyhash.hpp>: a compile-time fingerprint of a
//  definition, a switch on a string, and a hash of bytes. Exits non-zero if any
//  value is not the one documented.
//

#include <snicholls/core/hash/wyhash.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <string>
#include <string_view>

namespace {

//! [hash]
using namespace snicholls::literals;

// A closed alphabet's fingerprint, computed by the compiler from its definition: two machines
// that derive the same 64 bits from the same text agree on what they are talking about.
constexpr std::string_view alphabet_definition = "ACGT v1";
constexpr std::uint64_t alphabet_fingerprint = snicholls::wyhash(alphabet_definition);

// A switch on a string: the labels are constants, the subject is hashed at run time.
int route(std::string_view method)
{
    switch (snicholls::wyhash(method)) {
    case "GET"_wyhash: return 1;
    case "POST"_wyhash: return 2;
    case "DELETE"_wyhash: return 3;
    default: return 0;
    }
}
//! [hash]

} // namespace

int main()
{
    std::printf("snicholls-core wyhash demo\n");
    bool ok = true;

    std::printf("  fingerprint(\"%.*s\") = %016llx\n", static_cast<int>(alphabet_definition.size()), alphabet_definition.data(),
                static_cast<unsigned long long>(alphabet_fingerprint));
    ok &= alphabet_fingerprint == snicholls::wyhash(std::string{"ACGT v1"});   // the same at run time

    std::printf("  route(GET) = %d, route(POST) = %d, route(PATCH) = %d\n", route("GET"), route("POST"), route("PATCH"));
    ok &= route("GET") == 1 && route("POST") == 2 && route("DELETE") == 3 && route("PATCH") == 0;

    // Bytes hash the same whichever way they are carried
    const std::string text{"message digest"};
    const auto as_text = snicholls::wyhash(text, 3);
    const auto as_bytes = snicholls::wyhash(std::span<const std::byte>{reinterpret_cast<const std::byte*>(text.data()), text.size()}, 3);
    std::printf("  wyhash(\"message digest\", 3) = %016llx\n", static_cast<unsigned long long>(as_text));
    ok &= as_text == as_bytes && as_text == 0x786d1f1df3801df4ull;   // upstream's vector

    // The mixers
    std::printf("  splitmix64(0) = %016llx, wymix(3, 5) = %llu\n", static_cast<unsigned long long>(snicholls::splitmix64(0)),
                static_cast<unsigned long long>(snicholls::wymix(3, 5)));
    ok &= snicholls::splitmix64(0) == 0xe220a8397b1dcdafull && snicholls::wymix(3, 5) == 15;

    std::printf(ok ? "wyhash demo: ok\n" : "wyhash demo: FAILED\n");
    return ok ? 0 : 1;
}
