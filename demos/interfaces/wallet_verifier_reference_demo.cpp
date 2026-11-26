//
//  wallet_verifier_reference_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/interfaces/wallet_verifier_reference.hpp>: test wallets, their test signatures, and a
//  failure armed for a test. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/interfaces/wallet_verifier_reference.hpp>

#include <cstdio>
#include <string_view>

using namespace snicholls;

namespace {
int failures = 0;
void check(bool ok, const char* what)
{
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (!ok)
        ++failures;
}
} // namespace

int main()
{
    std::printf("snicholls-core interfaces/wallet_verifier_reference.hpp demo\n");
    //! [reference]
    constexpr std::string_view alice = "0x52908400098527886e0f7030069857d2e4169ee7";
    constexpr std::string_view bob = "0x8617e340b3d01fa5f11f306f4090fd50e238070d";
    memory_wallet_verifier wallets;   // not ECDSA: a keyed test scheme, for tests only
    static_cast<void>(wallets.add_wallet(alice, "alice's test key"));
    static_cast<void>(wallets.add_wallet(bob, "bob's test key"));
    const auto sig = wallets.sign(alice, "sign in");   // 65 bytes, r||s||v
    const auto as_alice = wallets.verify({1, alice, "sign in", sig});
    const auto as_bob = wallets.verify({1, bob, "sign in", sig});
    //! [reference]
    check(sig.size() == memory_wallet_verifier::signature_bytes, "65 bytes, as an EOA signature");
    check(as_alice.is_ok(), "alice's signature verifies as alice");
    check(as_bob.unwrap_err().code == capability_errc::signature_invalid, "and not as bob");
    check(wallets.add_wallet("not an address", "k").unwrap_err().code == capability_errc::invalid_argument,
          "a wallet's address is in the contract's form");
    wallets.fail_next(capability_errc::unavailable, "rpc_down");
    check(wallets.verify({1, alice, "sign in", sig}).unwrap_err().code == capability_errc::unavailable,
          "an armed failure is returned once");
    return failures == 0 ? 0 : 1;
}
