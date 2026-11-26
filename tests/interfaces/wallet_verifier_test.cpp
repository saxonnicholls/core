//
//  wallet_verifier_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/interfaces/wallet_verifier.hpp>: the conformance kit passes on the reference in every
//  form (the concept, the virtual interface over a borrowing adapter, the handle), and the adapters forward.
//

#include <snicholls/core/interfaces/wallet_verifier.hpp>
#include <snicholls/core/interfaces/wallet_verifier_conformance.hpp>

#include "capability_probes.hpp"

#include <gtest/gtest.h>

#include <type_traits>

using namespace snicholls;

namespace {

static_assert(wallet_verifier<wallet_verifier_interface>);
static_assert(wallet_verifier<wallet_verifier_ref>);
static_assert(std::is_abstract_v<wallet_verifier_interface>);

TEST(WalletVerifierInterface, TheKitPassesInEveryForm)
{
    const auto direct = run_wallet_verifier_conformance<test::wallet_verifier_probe<test::form::concept_model>>();
    const auto virtual_form = run_wallet_verifier_conformance<test::wallet_verifier_probe<test::form::interface>>();
    const auto handle = run_wallet_verifier_conformance<test::wallet_verifier_probe<test::form::ref>>();
    EXPECT_TRUE(direct.ok()) << test::describe(direct);
    EXPECT_TRUE(virtual_form.ok()) << test::describe(virtual_form);
    EXPECT_TRUE(handle.ok()) << test::describe(handle);
    EXPECT_EQ(direct.checks, 6u);
    EXPECT_EQ(virtual_form.checks, 6u);
    EXPECT_EQ(handle.checks, 6u);
}

struct counting_verifier {
    mutable int calls = 0;
    capability_result<void> verify(const wallet_signature_request&) const
    {
        ++calls;
        return fail(capability_errc::signature_invalid);
    }
};
static_assert(wallet_verifier<wallet_verifier_adapter<counting_verifier>>);

TEST(WalletVerifierInterface, TheAdaptersForward)
{
    counting_verifier impl;
    wallet_verifier_adapter<counting_verifier&> borrowed{impl};
    const wallet_verifier_ref v{borrowed};
    EXPECT_EQ(v.verify({}).unwrap_err().code, capability_errc::signature_invalid);
    EXPECT_EQ(impl.calls, 1);
    EXPECT_EQ(&borrowed.get(), &impl);
    wallet_verifier_adapter owned{test::make_wallet_verifier()};
    const auto sig = owned.get().sign(test::probe_wallet, "hi");
    EXPECT_TRUE(owned.verify({1, test::probe_wallet, "hi", sig}).is_ok());
    EXPECT_EQ(owned.verify({}).unwrap_err().code, capability_errc::invalid_argument);
}

} // namespace
