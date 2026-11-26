//
//  token_verifier_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/interfaces/token_verifier.hpp>: the conformance kit passes on the reference in every form
//  (the concept, the virtual interface over a borrowing adapter, the handle), and the adapters forward.
//

#include <snicholls/core/interfaces/token_verifier.hpp>
#include <snicholls/core/interfaces/token_verifier_conformance.hpp>

#include "capability_probes.hpp"

#include <gtest/gtest.h>

#include <string_view>
#include <type_traits>

using namespace snicholls;

namespace {

static_assert(token_verifier<token_verifier_interface>);
static_assert(token_verifier<token_verifier_ref>);
static_assert(std::is_abstract_v<token_verifier_interface>);
static_assert(std::is_trivially_copyable_v<token_verifier_ref>);

TEST(TokenVerifierInterface, TheKitPassesInEveryForm)
{
    const auto direct = run_token_verifier_conformance<test::token_verifier_probe<test::form::concept_model>>();
    const auto virtual_form = run_token_verifier_conformance<test::token_verifier_probe<test::form::interface>>();
    const auto handle = run_token_verifier_conformance<test::token_verifier_probe<test::form::ref>>();
    EXPECT_TRUE(direct.ok()) << test::describe(direct);
    EXPECT_TRUE(virtual_form.ok()) << test::describe(virtual_form);
    EXPECT_TRUE(handle.ok()) << test::describe(handle);
    EXPECT_EQ(direct.checks, 6u);
    EXPECT_EQ(virtual_form.checks, 6u);
    EXPECT_EQ(handle.checks, 6u);
}

struct counting_verifier {
    mutable int calls = 0;
    capability_result<token_claims> verify(std::string_view) const
    {
        ++calls;
        return fail(capability_errc::unauthorized);
    }
};
static_assert(token_verifier<token_verifier_adapter<counting_verifier>>);

TEST(TokenVerifierInterface, TheAdaptersForward)
{
    counting_verifier impl;
    token_verifier_adapter<counting_verifier&> borrowed{impl};
    const token_verifier_ref v{borrowed};
    EXPECT_EQ(v.verify("t").unwrap_err().code, capability_errc::unauthorized);
    EXPECT_EQ(impl.calls, 1);
    EXPECT_EQ(&borrowed.get(), &impl);
    token_verifier_adapter owned{memory_token_verifier::deterministic()};
    EXPECT_EQ(owned.verify("").unwrap_err().code, capability_errc::unauthorized);
}

} // namespace
