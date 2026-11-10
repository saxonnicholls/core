//
//  fnv_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/hash/fnv.hpp>: a switch on a string with labels hashed
//  at compile time, and a composite hash that two machines agree on. Exits
//  non-zero if a result is not what it documents.
//

#include <snicholls/core/hash/fnv.hpp>

#include <cstdint>
#include <cstdio>
#include <string_view>

//! [switch]
using namespace snicholls::literals;

// Labels are hashed by the compiler; the subject at run time. Check the string
// too when a collision would matter: here the cases are a fixed known set.
const char* verb(std::string_view method)
{
    switch (snicholls::fnv1a_64(method)) {
    case "GET"_fnv64: return "read";
    case "POST"_fnv64: return "create";
    case "DELETE"_fnv64: return "remove";
    default: return "unknown";
    }
}
//! [switch]

int main()
{
    std::printf("snicholls-core fnv demo\n");
    bool ok = true;

    for (const char* m : {"GET", "POST", "DELETE", "BREW"})
        std::printf("  %-7s -> %s\n", m, verb(m));
    ok &= std::string_view(verb("POST")) == "create" && std::string_view(verb("BREW")) == "unknown";

    // A composite hash of a name and two integers: the same on a Mac and a Pi,
    // because integers are folded in through their little-endian bytes
    std::uint64_t h = snicholls::fnv1a_64("price_chunk");
    h = snicholls::fnv1a_64_append(h, std::uint32_t{3});
    h = snicholls::fnv1a_64_append(h, std::uint64_t{8});
    std::printf("  signature %016llx\n", static_cast<unsigned long long>(h));

    constexpr std::uint64_t at_compile_time = snicholls::fnv1a_64_append(
        snicholls::fnv1a_64_append(snicholls::fnv1a_64("price_chunk"), std::uint32_t{3}), std::uint64_t{8});
    ok &= h == at_compile_time;
    ok &= snicholls::fnv1a_64("foobar") == 0x85944171f73967e8ull;

    std::printf(ok ? "fnv demo: ok\n" : "fnv demo: FAILED\n");
    return ok ? 0 : 1;
}
