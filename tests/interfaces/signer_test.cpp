//
//  signer_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/interfaces/signer.hpp>: the conformance kit passes on the reference in every form (the
//  concept, the virtual interface over a borrowing adapter, the handle), and the adapters forward every operation.
//

#include <snicholls/core/interfaces/signer.hpp>
#include <snicholls/core/interfaces/signer_conformance.hpp>

#include "capability_probes.hpp"

#include <gtest/gtest.h>

#include <string_view>
#include <type_traits>
#include <vector>

using namespace snicholls;

namespace {

static_assert(signer<signer_interface>);
static_assert(signer<signer_ref>);
static_assert(std::is_abstract_v<signer_interface>);

TEST(SignerInterface, TheKitPassesInEveryForm)
{
    const auto direct = run_signer_conformance<test::signer_probe<test::form::concept_model>>();
    const auto virtual_form = run_signer_conformance<test::signer_probe<test::form::interface>>();
    const auto handle = run_signer_conformance<test::signer_probe<test::form::ref>>();
    EXPECT_TRUE(direct.ok()) << test::describe(direct);
    EXPECT_TRUE(virtual_form.ok()) << test::describe(virtual_form);
    EXPECT_TRUE(handle.ok()) << test::describe(handle);
    EXPECT_EQ(direct.checks, 7u);
    EXPECT_EQ(virtual_form.checks, 7u);
    EXPECT_EQ(handle.checks, 7u);
}

struct counting_signer {
    mutable int calls = 0;
    capability_result<signature> sign(const sign_request&) { ++calls; return fail(capability_errc::not_found); }
    [[nodiscard]] capability_result<public_key> public_key_of(const key_ref&) const { ++calls; return fail(capability_errc::unsupported); }
};
static_assert(signer<signer_adapter<counting_signer>>);

TEST(SignerInterface, TheAdaptersForwardEveryOperation)
{
    counting_signer impl;
    signer_adapter<counting_signer&> borrowed{impl};
    signer_ref s{borrowed};
    EXPECT_EQ(s.sign({}).unwrap_err().code, capability_errc::not_found);
    EXPECT_EQ(s.public_key_of({}).unwrap_err().code, capability_errc::unsupported);
    EXPECT_EQ(impl.calls, 2);
    signer_adapter owned{memory_signer::deterministic()};
    EXPECT_EQ(owned.public_key_of({"key-1", 0}).unwrap_err().code, capability_errc::not_found);
}

} // namespace
