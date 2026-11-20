//
//  ripemd160_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/hash/ripemd160.hpp>: Bitcoin's hash160 of a public key, the twenty bytes inside a
//  P2PKH or P2WPKH address. Exits non-zero if any value is not the one documented.
//

#include <snicholls/core/hash/ripemd160.hpp>
#include <snicholls/core/hash/sha2.hpp>

#include <array>
#include <cstdint>
#include <cstdio>

namespace {

//! [ripemd160]
// hash160 = RIPEMD-160(SHA-256(x)): Bitcoin's address payload, and BIP-32's key fingerprint (its first 4 bytes)
template <snicholls::hash_input Key>
constexpr snicholls::digest<20> hash160(const Key& public_key) noexcept
{
    return snicholls::ripemd160::hash(snicholls::sha256::hash(public_key));
}

// The compressed public key of private key 1 (the secp256k1 generator)
constexpr std::array<std::uint8_t, 33> generator = {
    0x02, 0x79, 0xbe, 0x66, 0x7e, 0xf9, 0xdc, 0xbb, 0xac, 0x55, 0xa0, 0x62, 0x95, 0xce, 0x87, 0x0b, 0x07,
    0x02, 0x9b, 0xfc, 0xdb, 0x2d, 0xce, 0x28, 0xd9, 0x59, 0xf2, 0x81, 0x5b, 0x16, 0xf8, 0x17, 0x98};
static_assert(hash160(generator).hex().view() == "751e76e8199196d454941c45d1b3a323f1433bd6");
//! [ripemd160]

} // namespace

int main()
{
    std::printf("snicholls-core ripemd160 demo\n");
    bool ok = true;

    const auto h = hash160(generator);
    std::printf("  hash160(G)        = %s (address 1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMH)\n", h.hex().c_str());
    std::printf("  fingerprint       = %s\n", h.first<4>().hex().c_str());
    ok &= h.first<4>().hex().view() == "751e76e8";

    snicholls::ripemd160 r;
    r.update("message ").update("digest");
    ok &= r.finish().hex().view() == "5d0689ef49d2fae572b881b123a85ffa21595f36";
    ok &= snicholls::ripemd160::digest_size == 20 && snicholls::ripemd160::block_size == 64;

    std::printf(ok ? "ripemd160 demo: ok\n" : "ripemd160 demo: FAILED\n");
    return ok ? 0 : 1;
}
