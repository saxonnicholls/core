//
//  kdf_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/hash/kdf.hpp>: HKDF splitting one shared secret into two keys, PBKDF2 for a BIP-39
//  seed, and the errors. Exits non-zero if any value is not the one documented.
//

#include <snicholls/core/hash/kdf.hpp>
#include <snicholls/core/hash/sha2.hpp>

#include <array>
#include <cstddef>
#include <cstdio>
#include <string>
#include <vector>

namespace {

//! [kdf]
// HKDF: one shared secret, independent keys bound to their purpose by `info`
struct keys
{
    std::array<std::byte, 32> client_to_server, server_to_client;
};
keys split(const std::string& shared_secret, const std::string& handshake_salt)
{
    keys k{};
    const auto prk = snicholls::hkdf_extract<snicholls::sha256>(handshake_salt, shared_secret);
    (void)snicholls::hkdf_expand<snicholls::sha256>(prk, "c2s key", k.client_to_server);   // 32 <= 255 * 32: cannot fail
    (void)snicholls::hkdf_expand<snicholls::sha256>(prk, "s2c key", k.server_to_client);
    return k;
}

// PBKDF2: BIP-39's seed is PBKDF2-HMAC-SHA512(mnemonic, "mnemonic" + passphrase, 2048 iterations, 64 bytes)
snicholls::result<snicholls::digest<64>, snicholls::kdf_error> bip39_seed(const std::string& mnemonic, const std::string& passphrase)
{
    return snicholls::pbkdf2_hmac<snicholls::sha512, 64>(mnemonic, "mnemonic" + passphrase, 2048);
}
//! [kdf]

} // namespace

int main()
{
    std::printf("snicholls-core kdf demo\n");
    bool ok = true;

    const keys k = split("a shared secret", "salt from the handshake");
    std::printf("  c2s != s2c: %s\n", k.client_to_server != k.server_to_client ? "yes" : "no");
    ok &= k.client_to_server != k.server_to_client;

    // hkdf in one call, and the fixed-length form
    std::array<std::byte, 32> again{};
    ok &= snicholls::hkdf<snicholls::sha256>("salt from the handshake", "a shared secret", "c2s key", again).is_ok();
    ok &= again == k.client_to_server;
    ok &= snicholls::hkdf<snicholls::sha256, 32>("salt from the handshake", "a shared secret", "c2s key").bytes == again;
    const auto prk = snicholls::hkdf_extract<snicholls::sha256>("salt from the handshake", "a shared secret");
    ok &= snicholls::hkdf_expand<snicholls::sha256, 32>(prk, "s2c key").bytes == k.server_to_client;

    const auto seed = bip39_seed("abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about", "TREZOR");
    std::printf("  BIP-39 seed = %s\n", seed ? seed.unwrap_unchecked().hex().c_str() : "error");
    ok &= seed && seed.unwrap_unchecked().first<8>().hex().view() == "c55257c360c07c72";

    // Into a span of any length
    std::vector<std::byte> dk(20);
    ok &= snicholls::pbkdf2_hmac<snicholls::sha256>("password", "salt", 1, dk).is_ok();

    // Errors are values: nothing is written
    std::vector<std::byte> too_long(255 * 32 + 1);
    ok &= snicholls::hkdf_expand<snicholls::sha256>(prk, "", too_long).unwrap_err_unchecked() == snicholls::kdf_error::output_too_long;
    ok &= snicholls::pbkdf2_hmac<snicholls::sha256>("p", "s", 0, dk).unwrap_err_unchecked() == snicholls::kdf_error::zero_iterations;
    std::printf("  zero iterations and an over-long HKDF output are errors: %s\n", ok ? "yes" : "no");

    std::printf(ok ? "kdf demo: ok\n" : "kdf demo: FAILED\n");
    return ok ? 0 : 1;
}
