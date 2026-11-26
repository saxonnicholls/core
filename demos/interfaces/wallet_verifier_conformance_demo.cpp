//
//  wallet_verifier_conformance_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/interfaces/wallet_verifier_conformance.hpp>: the kit run over the reference verifier.
//  Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/interfaces/wallet_verifier_conformance.hpp>
#include <snicholls/core/interfaces/wallet_verifier_reference.hpp>

#include <cstddef>
#include <cstdio>
#include <string>
#include <string_view>
#include <vector>

using namespace snicholls;

namespace {
int failures = 0;
void check(bool ok, const char* what)
{
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (!ok)
        ++failures;
}

constexpr std::string_view alice = "0x52908400098527886e0f7030069857d2e4169ee7";

struct probe {
    memory_wallet_verifier impl;
    probe()
    {
        static_cast<void>(impl.add_wallet(alice, "k1"));
        static_cast<void>(impl.add_wallet(other_address(), "k2"));
    }
    auto& verifier() { return impl; }
    static std::string signer_address() { return std::string(alice); }
    static std::string other_address() { return "0x8617e340b3d01fa5f11f306f4090fd50e238070d"; }
    std::vector<std::byte> sign(std::string_view message) { return impl.sign(alice, message); }
    void provider_fails(capability_errc code, std::string_view response) { impl.fail_next(code, response); }
};
} // namespace

int main()
{
    std::printf("snicholls-core interfaces/wallet_verifier_conformance.hpp demo\n");
    //! [kit]
    const capability_conformance_report report = run_wallet_verifier_conformance<probe>();
    //! [kit]
    check(wallet_conformance_message.starts_with("example.test wants you to sign in"), "the kit signs a SIWE message");
    check(report.ok() && report.checks == 6, "the reference passes all six checks");
    return failures == 0 ? 0 : 1;
}
