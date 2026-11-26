//
//  wallet_verifier_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/interfaces/wallet_verifier.hpp>: a wallet verifier chosen at run time, behind the virtual
//  interface. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/interfaces/wallet_verifier.hpp>
#include <snicholls/core/interfaces/wallet_verifier_reference.hpp>

#include <cstdio>
#include <string_view>
#include <utility>

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
    std::printf("snicholls-core interfaces/wallet_verifier.hpp demo\n");
    constexpr std::string_view wallet = "0x52908400098527886e0f7030069857d2e4169ee7";
    memory_wallet_verifier reference;
    static_cast<void>(reference.add_wallet(wallet, "test-only key"));
    const auto signature = reference.sign(wallet, "sign in");
    //! [virtual]
    wallet_verifier_adapter<memory_wallet_verifier&> adapter{reference};   // borrows
    const wallet_verifier_interface& chosen = adapter;
    const wallet_verifier_ref handle{chosen};
    const auto checked = handle.verify({8453, wallet, "sign in", signature});
    //! [virtual]
    check(checked.is_ok(), "verified through the interface and the handle");
    check(&adapter.get() == &reference, "the adapter borrows");
    const wallet_verifier_adapter owned{std::move(reference)};
    check(owned.verify({8453, wallet, "sign out", signature}).unwrap_err().code == capability_errc::signature_invalid,
          "an owning adapter forwards: another message does not verify");
    return failures == 0 ? 0 : 1;
}
