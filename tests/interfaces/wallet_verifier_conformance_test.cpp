//
//  wallet_verifier_conformance_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/interfaces/wallet_verifier_conformance.hpp>: the kit passes the reference and catches
//  broken implementations: one that leaks the node's response, one that ignores the address, one that accepts any
//  address text.
//

#include <snicholls/core/interfaces/wallet_verifier_conformance.hpp>

#include "capability_probes.hpp"

#include <gtest/gtest.h>

#include <string>
#include <string_view>

using namespace snicholls;

namespace {

enum class defect { leaks, ignores_the_address, accepts_any_address };

template <defect D>
struct broken_verifier {
    memory_wallet_verifier inner = test::make_wallet_verifier();
    mutable bool armed = false;
    capability_result<void> verify(const wallet_signature_request& r) const
    {
        if (D == defect::leaks && armed) {
            armed = false;
            return err(capability_error{capability_errc::unavailable, *provider_code::make(capability_redaction_marker), {}});
        }
        wallet_signature_request q = r;
        if (D == defect::ignores_the_address && is_lowercase_evm_address(r.address))
            q.address = test::probe_wallet;   // checks the signature, not who made it
        if (D == defect::accepts_any_address && !is_lowercase_evm_address(r.address))
            return fail(capability_errc::signature_invalid);
        return inner.verify(q);
    }
};

template <defect D>
struct broken_probe {
    broken_verifier<D> v;
    auto& verifier() { return v; }
    static std::string signer_address() { return std::string(test::probe_wallet); }
    static std::string other_address() { return std::string(test::probe_other_wallet); }
    std::vector<std::byte> sign(std::string_view m) { return v.inner.sign(test::probe_wallet, m); }
    void provider_fails(capability_errc c, std::string_view r)
    {
        v.armed = true;
        v.inner.fail_next(c, r);
    }
};

TEST(WalletVerifierConformance, TheReferencePasses)
{
    const auto r = run_wallet_verifier_conformance<test::wallet_verifier_probe<test::form::concept_model>>();
    EXPECT_TRUE(r.ok()) << test::describe(r);
    EXPECT_EQ(r.checks, 6u);
}

TEST(WalletVerifierConformance, ALeakIsCaught)
{
    const auto r = run_wallet_verifier_conformance<broken_probe<defect::leaks>>();
    EXPECT_TRUE(r.failed("a provider failure never leaks its response")) << test::describe(r);
    EXPECT_EQ(r.failures.size(), 1u) << test::describe(r);
}

TEST(WalletVerifierConformance, IgnoringTheAddressIsCaught)
{
    const auto r = run_wallet_verifier_conformance<broken_probe<defect::ignores_the_address>>();
    EXPECT_TRUE(r.failed("another address is signature_invalid")) << test::describe(r);
}

TEST(WalletVerifierConformance, AcceptingAnyAddressTextIsCaught)
{
    const auto r = run_wallet_verifier_conformance<broken_probe<defect::accepts_any_address>>();
    EXPECT_TRUE(r.failed("an address not in the contract's form is invalid_argument")) << test::describe(r);
}

} // namespace
