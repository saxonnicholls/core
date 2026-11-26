//
//  security_provider_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/interfaces/security_provider.hpp>: the conformance kit passes on the reference in every
//  form (the concept, the virtual interface over a borrowing adapter, the handle), and the adapters forward every
//  operation.
//

#include <snicholls/core/interfaces/security_provider.hpp>
#include <snicholls/core/interfaces/security_provider_conformance.hpp>

#include "security_provider_probe.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <type_traits>
#include <vector>

using namespace snicholls;

namespace {

static_assert(security_provider<security_provider_interface>);
static_assert(security_provider<security_provider_ref>);
static_assert(std::is_abstract_v<security_provider_interface>);
static_assert(std::is_copy_constructible_v<security_provider_ref>);

TEST(SecurityProviderInterface, TheKitPassesInEveryForm)
{
    const auto direct = run_security_provider_conformance<test::security_provider_probe<test::form::concept_model>>();
    const auto virtual_form = run_security_provider_conformance<test::security_provider_probe<test::form::interface>>();
    const auto handle = run_security_provider_conformance<test::security_provider_probe<test::form::ref>>();
    EXPECT_TRUE(direct.ok()) << test::describe(direct);
    EXPECT_TRUE(virtual_form.ok()) << test::describe(virtual_form);
    EXPECT_TRUE(handle.ok()) << test::describe(handle);
    EXPECT_EQ(direct.checks, 8u);
    EXPECT_EQ(virtual_form.checks, 8u);
    EXPECT_EQ(handle.checks, 8u);
}

struct counting_provider {
    mutable int calls = 0;
    [[nodiscard]] security_capabilities capabilities() const { ++calls; return {}; }
    [[nodiscard]] capability_result<webhook_event> verify_webhook(const webhook_delivery&) const { ++calls; return fail(capability_errc::signature_invalid); }
    capability_result<std::vector<security_fact>> resolve(const webhook_event&) { ++calls; return ok(std::vector<security_fact>{}); }
    capability_result<security_statement_page> statement(const security_statement_query&) { ++calls; return fail(capability_errc::unsupported); }
    capability_result<std::vector<security_holding>> holdings(std::chrono::sys_days) { ++calls; return fail(capability_errc::unavailable); }
};
static_assert(security_provider<security_provider_adapter<counting_provider>>);

TEST(SecurityProviderInterface, TheAdaptersForwardEveryOperation)
{
    counting_provider impl;
    security_provider_adapter<counting_provider&> borrowed{impl};
    security_provider_ref p{borrowed};
    static_cast<void>(p.capabilities());
    EXPECT_EQ(p.verify_webhook({}).unwrap_err().code, capability_errc::signature_invalid);
    EXPECT_TRUE(p.resolve({})->empty());
    EXPECT_EQ(p.statement({}).unwrap_err().code, capability_errc::unsupported);
    EXPECT_EQ(p.holdings({}).unwrap_err().code, capability_errc::unavailable);
    EXPECT_EQ(impl.calls, 5);
    EXPECT_EQ(&borrowed.get(), &impl);
    security_provider_adapter owned{test::make_security_provider()};
    EXPECT_TRUE(owned.capabilities().holdings);
}

} // namespace
