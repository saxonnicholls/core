//
//  payment_provider_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/interfaces/payment_provider.hpp>: the conformance kit passes on the reference in every form (the
//  concept, the virtual interface over a borrowing adapter, the handle), and the adapters forward every operation.
//

#include <snicholls/core/interfaces/payment_provider.hpp>
#include <snicholls/core/interfaces/payment_provider_conformance.hpp>

#include "capability_probes.hpp"

#include <gtest/gtest.h>

#include <string_view>
#include <type_traits>
#include <vector>

using namespace snicholls;

namespace {

static_assert(payment_provider<payment_provider_interface>);
static_assert(payment_provider<payment_provider_ref>);
static_assert(std::is_abstract_v<payment_provider_interface>);

TEST(PaymentProviderInterface, TheKitPassesInEveryForm)
{
    const auto direct = run_payment_provider_conformance<test::payment_provider_probe<test::form::concept_model>>();
    const auto virtual_form = run_payment_provider_conformance<test::payment_provider_probe<test::form::interface>>();
    const auto handle = run_payment_provider_conformance<test::payment_provider_probe<test::form::ref>>();
    EXPECT_TRUE(direct.ok()) << test::describe(direct);
    EXPECT_TRUE(virtual_form.ok()) << test::describe(virtual_form);
    EXPECT_TRUE(handle.ok()) << test::describe(handle);
    EXPECT_EQ(direct.checks, 13u);
    EXPECT_EQ(virtual_form.checks, 13u);
    EXPECT_EQ(handle.checks, 13u);
}

struct counting_provider {
    mutable int calls = 0;
    [[nodiscard]] provider_capabilities capabilities() const { ++calls; return {}; }
    capability_result<collection> collect(const collection_request&) { ++calls; return fail(capability_errc::unsupported); }
    capability_result<payment> payout(const payout_request&) { ++calls; return fail(capability_errc::unsupported); }
    capability_result<payment> refund(const refund_request&) { ++calls; return fail(capability_errc::unsupported); }
    capability_result<fx_quote> quote(const quote_request&) { ++calls; return fail(capability_errc::unsupported); }
    capability_result<payment> status(std::string_view) { ++calls; return fail(capability_errc::not_found); }
    capability_result<std::vector<money>> balances() { ++calls; return ok(std::vector<money>{}); }
    capability_result<std::vector<credit>> credits(const credit_query&) { ++calls; return fail(capability_errc::unsupported); }
    [[nodiscard]] capability_result<webhook_event> verify_webhook(const webhook_delivery&) const { ++calls; return fail(capability_errc::signature_invalid); }
    capability_result<std::vector<payment_fact>> resolve(const webhook_event&) { ++calls; return ok(std::vector<payment_fact>{}); }
    capability_result<statement_page> statement(const statement_query&) { ++calls; return fail(capability_errc::unsupported); }
};
static_assert(payment_provider<payment_provider_adapter<counting_provider>>);

TEST(PaymentProviderInterface, TheAdaptersForwardEveryOperation)
{
    counting_provider impl;
    payment_provider_adapter<counting_provider&> borrowed{impl};
    payment_provider_ref p{borrowed};
    static_cast<void>(p.capabilities());
    EXPECT_EQ(p.collect({}).unwrap_err().code, capability_errc::unsupported);
    EXPECT_EQ(p.payout({}).unwrap_err().code, capability_errc::unsupported);
    EXPECT_EQ(p.refund({}).unwrap_err().code, capability_errc::unsupported);
    EXPECT_EQ(p.quote({}).unwrap_err().code, capability_errc::unsupported);
    EXPECT_EQ(p.status("x").unwrap_err().code, capability_errc::not_found);
    EXPECT_TRUE(p.balances().is_ok());
    EXPECT_EQ(p.credits({}).unwrap_err().code, capability_errc::unsupported);
    EXPECT_EQ(p.verify_webhook({}).unwrap_err().code, capability_errc::signature_invalid);
    EXPECT_TRUE(p.resolve({})->empty());
    EXPECT_EQ(p.statement({}).unwrap_err().code, capability_errc::unsupported);
    EXPECT_EQ(impl.calls, 11);
    payment_provider_adapter owned{test::make_payment_provider()};
    EXPECT_EQ(owned.capabilities().currencies.size(), 1u);
}

} // namespace
