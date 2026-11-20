//
//  blake2_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/hash/blake2.hpp>: BLAKE2b with a chosen output length, a keyed BLAKE2b as a MAC
//  without HMAC, and BLAKE2s. Exits non-zero if any value is not the one documented.
//

#include <snicholls/core/hash/blake2.hpp>

#include <cstdio>
#include <stdexcept>
#include <string>

namespace {

//! [blake2]
// The output length is part of the type: BLAKE2b-256 is not BLAKE2b-512 truncated
constexpr auto short_id = snicholls::blake2b<16>::hash("order 42");
static_assert(short_id.size() == 16);

// Keyed: a MAC in one pass, no HMAC around it
snicholls::digest<32> tag(const std::string& key, const std::string& message)
{
    return snicholls::blake2b_256::keyed_hash(key, message);
}
//! [blake2]

} // namespace

int main()
{
    std::printf("snicholls-core blake2 demo\n");
    bool ok = true;

    std::printf("  blake2b<16>(order 42) = %s\n", short_id.hex().c_str());
    ok &= short_id != snicholls::blake2b_512::hash("order 42").first<16>();   // a different function, not a prefix

    const auto t = tag("a 32-byte key for this demo only", "pay 100 to alice");
    std::printf("  keyed tag             = %s\n", t.hex().c_str());
    ok &= t != tag("a 32-byte key for this demo only", "pay 900 to alice");

    // RFC 7693's examples, through the incremental interface
    snicholls::blake2b_512 b;
    b.update("a").update("bc");
    ok &= b.finish().hex().view().starts_with("ba80a53f981c4d0d");
    snicholls::blake2s_256 s{std::string()};   // an empty key is the unkeyed hash
    s.update("abc");
    ok &= s.finish() == snicholls::blake2s_256::hash("abc");
    ok &= snicholls::blake2s<32>::hash("abc").hex().view() == "508c5e8c327c14e2e1a72ba34eeb452f37458b209ed63a294d999b4c86675982";

    // A key longer than the variant allows is an error
    bool threw = false;
    try {
        snicholls::blake2s_256 too_long{std::string(33, 'k')};
    }
    catch (const std::length_error&) {
        threw = true;
    }
    std::printf("  33-byte key for BLAKE2s throws length_error: %s\n", threw ? "yes" : "no");
    ok &= threw && snicholls::blake2b_512::max_key_size == 64 && snicholls::blake2s_256::max_key_size == 32;

    std::printf(ok ? "blake2 demo: ok\n" : "blake2 demo: FAILED\n");
    return ok ? 0 : 1;
}
