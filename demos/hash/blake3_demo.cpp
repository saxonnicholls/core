//
//  blake3_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/hash/blake3.hpp>: the hash, a keyed MAC, two independent keys derived from one secret,
//  and extendable output read from an offset. Exits non-zero if any value is not the one documented.
//

#include <snicholls/core/hash/blake3.hpp>

#include <array>
#include <cstddef>
#include <cstdio>
#include <span>
#include <stdexcept>
#include <string>

namespace {

//! [blake3]
// The hash, at compile time
static_assert(snicholls::blake3::hash("").hex().view() == "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262");

// Two keys from one secret: each context string is hard-coded and unique to its purpose
struct session_keys
{
    snicholls::digest<32> encrypt, authenticate;
};
session_keys derive(const snicholls::digest<32>& shared_secret)
{
    return {snicholls::blake3::derive_key("example.org 2026-11-20 session encryption v1", shared_secret),
            snicholls::blake3::derive_key("example.org 2026-11-20 session authentication v1", shared_secret)};
}
//! [blake3]

} // namespace

int main()
{
    std::printf("snicholls-core blake3 demo\n");
    bool ok = true;

    const auto secret = snicholls::blake3::hash("a shared secret from a key exchange");
    const session_keys keys = derive(secret);
    std::printf("  encryption key     = %s\n", keys.encrypt.hex().c_str());
    std::printf("  authentication key = %s\n", keys.authenticate.hex().c_str());
    ok &= keys.encrypt != keys.authenticate;

    // A keyed hash is a MAC; the key is exactly 32 bytes
    const auto mac = snicholls::blake3::keyed_hash(keys.authenticate, "message");
    snicholls::blake3 incremental{keys.authenticate};
    incremental.update("mess").update("age");
    ok &= incremental.finish() == mac;

    // Extendable output: any length, from any offset; finish is const, so the hasher can go on
    snicholls::blake3 h;
    h.update("stream");
    std::array<std::byte, 100> all{};
    h.finish_into(all);
    std::array<std::byte, 20> window{};
    h.finish_into(window, 80);
    bool same = true;
    for (std::size_t i = 0; i < 20; ++i)
        same &= window[i] == all[80 + i];
    ok &= same && h.finish<100>().bytes == all && h.finish() == snicholls::blake3::hash("stream");
    std::printf("  output bytes 80..99 read directly match: %s\n", same ? "yes" : "no");

    // Key derivation as a hasher, for long or streamed key material
    auto kd = snicholls::blake3::key_derivation("example.org 2026-11-20 session encryption v1");
    kd.update(secret);
    ok &= kd.finish() == keys.encrypt;

    bool threw = false;
    try {
        (void)snicholls::blake3::keyed_hash("too short", "m");
    }
    catch (const std::length_error&) {
        threw = true;
    }
    ok &= threw && snicholls::blake3::key_size == 32;

    std::printf(ok ? "blake3 demo: ok\n" : "blake3 demo: FAILED\n");
    return ok ? 0 : 1;
}
