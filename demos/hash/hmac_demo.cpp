//
//  hmac_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/hash/hmac.hpp>: a webhook signature verified in constant time, a MAC computed in
//  pieces, and BIP-32's master key derivation. Exits non-zero if any value is not the one documented.
//

#include <snicholls/core/hash/hmac.hpp>
#include <snicholls/core/hash/sha2.hpp>
#include <snicholls/core/hash/sha3.hpp>

#include <array>
#include <cstdio>
#include <string>

namespace {

//! [hmac]
// A webhook carries HMAC-SHA256(secret, body); check it in constant time, never with ==
bool authentic(const std::string& secret, const std::string& body, const snicholls::digest<32>& signature)
{
    return snicholls::hmac<snicholls::sha256>::verify(secret, body, signature);
}

// RFC 4231 test case 2, at compile time
static_assert(snicholls::hmac<snicholls::sha256>::mac("Jefe", "what do ya want for nothing?").hex().view() ==
              "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843");
//! [hmac]

} // namespace

int main()
{
    std::printf("snicholls-core hmac demo\n");
    bool ok = true;

    const std::string secret = "whsec_demo", body = R"({"event":"paid","amount":100})";
    const auto signature = snicholls::hmac<snicholls::sha256>::mac(secret, body);
    std::printf("  signature            = %s\n", signature.hex().c_str());
    std::printf("  authentic(body)      = %s\n", authentic(secret, body, signature) ? "yes" : "no");
    std::printf("  authentic(tampered)  = %s\n", authentic(secret, R"({"event":"paid","amount":900})", signature) ? "yes" : "no");
    ok &= authentic(secret, body, signature) && !authentic(secret, R"({"event":"paid","amount":900})", signature);

    // In pieces, and from a keyed object copied so the key's work is done once
    const snicholls::hmac<snicholls::sha256> keyed{secret};
    auto m = keyed;
    m.update(R"({"event":)").update(R"("paid","amount":100})");
    ok &= m.finish() == signature;

    // BIP-32: the master key and chain code are HMAC-SHA512("Bitcoin seed", seed); test vector 1's seed
    const std::array<unsigned char, 16> seed{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    const auto master = snicholls::hmac<snicholls::sha512>::mac("Bitcoin seed", seed);
    std::printf("  BIP-32 master key    = %s\n", master.first<32>().hex().c_str());
    ok &= master.first<32>().hex().view() == "e8f32e723decf4051aefac8e2c93c9c5b214313817cdb01a1494b917c8436b35";

    // Any hash: SHA3-256's block is its rate
    ok &= snicholls::hmac<snicholls::sha3_256>::block_size == 136 && snicholls::hmac<snicholls::sha3_256>::digest_size == 32;

    std::printf(ok ? "hmac demo: ok\n" : "hmac demo: FAILED\n");
    return ok ? 0 : 1;
}
