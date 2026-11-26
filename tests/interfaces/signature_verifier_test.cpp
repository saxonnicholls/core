//
//  signature_verifier_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/interfaces/signature_verifier.hpp>: the conformance kit passes on the reference in every
//  form (the concept, the virtual interface over a borrowing adapter, the handle), and the adapters forward.
//

#include <snicholls/core/interfaces/signature_verifier.hpp>
#include <snicholls/core/interfaces/signature_verifier_conformance.hpp>

#include "capability_probes.hpp"

#include <gtest/gtest.h>

#include <type_traits>

using namespace snicholls;

namespace {

static_assert(signature_verifier<signature_verifier_interface>);
static_assert(signature_verifier<signature_verifier_ref>);
static_assert(std::is_abstract_v<signature_verifier_interface>);

TEST(SignatureVerifierInterface, TheKitPassesInEveryForm)
{
    const auto direct = run_signature_verifier_conformance<test::signature_verifier_probe<test::form::concept_model>>();
    const auto virtual_form = run_signature_verifier_conformance<test::signature_verifier_probe<test::form::interface>>();
    const auto handle = run_signature_verifier_conformance<test::signature_verifier_probe<test::form::ref>>();
    EXPECT_TRUE(direct.ok()) << test::describe(direct);
    EXPECT_TRUE(virtual_form.ok()) << test::describe(virtual_form);
    EXPECT_TRUE(handle.ok()) << test::describe(handle);
    EXPECT_EQ(direct.checks, 6u);
    EXPECT_EQ(virtual_form.checks, 6u);
    EXPECT_EQ(handle.checks, 6u);
}

struct counting_verifier {
    mutable int calls = 0;
    capability_result<void> verify(const verify_request&) const
    {
        ++calls;
        return fail(capability_errc::signature_invalid);
    }
};
static_assert(signature_verifier<signature_verifier_adapter<counting_verifier>>);

TEST(SignatureVerifierInterface, TheAdaptersForward)
{
    counting_verifier impl;
    signature_verifier_adapter<counting_verifier&> borrowed{impl};
    const signature_verifier_ref v{borrowed};
    EXPECT_EQ(v.verify({}).unwrap_err().code, capability_errc::signature_invalid);
    EXPECT_EQ(impl.calls, 1);
    EXPECT_EQ(&borrowed.get(), &impl);
    const auto keys = memory_key_store::deterministic();
    signature_verifier_adapter owned{memory_signature_verifier{keys}};
    EXPECT_EQ(owned.verify({}).unwrap_err().code, capability_errc::unsupported);
}

} // namespace
