//
//  secure_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/hash/secure.hpp>: a key wiped when it is no longer needed, and a token compared in
//  constant time. Exits non-zero if any value is not the one documented.
//

#include <snicholls/core/hash/secure.hpp>
#include <snicholls/core/hash/sha2.hpp>

#include <array>
#include <cstddef>
#include <cstdio>
#include <span>
#include <string>

namespace {

//! [secure]
// Compare a presented token with the expected one in time that does not depend on where they differ
bool token_matches(const std::string& presented, const snicholls::digest<32>& expected_hash)
{
    return snicholls::constant_time_equal(snicholls::sha256::hash(presented), expected_hash);
}

// Wipe a key before its memory is reused: a plain loop here is a dead store the optimiser may delete
void use_and_forget(std::array<std::byte, 32>& key)
{
    // ... use the key ...
    snicholls::secure_wipe(std::span<std::byte, 32>(key));
}
//! [secure]

} // namespace

int main()
{
    std::printf("snicholls-core secure demo\n");
    bool ok = true;

    const auto expected = snicholls::sha256::hash("tok_live_123");
    std::printf("  token_matches(right) = %s, (wrong) = %s\n", token_matches("tok_live_123", expected) ? "yes" : "no",
                token_matches("tok_live_124", expected) ? "yes" : "no");
    ok &= token_matches("tok_live_123", expected) && !token_matches("tok_live_124", expected);
    ok &= !snicholls::constant_time_equal("abc", "abcd");   // lengths differ: not equal, and not secret

    std::array<std::byte, 32> key{};
    key.fill(std::byte{0xa5});
    use_and_forget(key);
    bool zero = true;
    for (const auto b : key)
        zero &= b == std::byte{0};
    std::printf("  key wiped: %s\n", zero ? "yes" : "no");
    ok &= zero;

    auto derived = snicholls::sha256::hash("a derived key");
    snicholls::secure_wipe(derived);
    ok &= derived == snicholls::digest<32>{};

    std::printf(ok ? "secure demo: ok\n" : "secure demo: FAILED\n");
    return ok ? 0 : 1;
}
