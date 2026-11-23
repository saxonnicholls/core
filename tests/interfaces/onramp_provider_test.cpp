//
//  onramp_provider_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/interfaces/onramp_provider.hpp>: the conformance kit passes on the reference in every form (the
//  concept, the virtual interface over a borrowing adapter, the handle), and the adapters forward every operation.
//

#include <snicholls/core/interfaces/onramp_provider.hpp>
#include <snicholls/core/interfaces/onramp_provider_conformance.hpp>

#include "capability_probes.hpp"

#include <gtest/gtest.h>

#include <string_view>
#include <type_traits>
#include <vector>

using namespace snicholls;

namespace {

static_assert(onramp_provider<onramp_provider_interface>);
static_assert(onramp_provider<onramp_provider_ref>);
static_assert(std::is_abstract_v<onramp_provider_interface>);

TEST(OnrampProviderInterface, TheKitPassesInEveryForm)
{
    const auto direct = run_onramp_provider_conformance<test::onramp_provider_probe<test::form::concept_model>>();
    const auto virtual_form = run_onramp_provider_conformance<test::onramp_provider_probe<test::form::interface>>();
    const auto handle = run_onramp_provider_conformance<test::onramp_provider_probe<test::form::ref>>();
    EXPECT_TRUE(direct.ok()) << test::describe(direct);
    EXPECT_TRUE(virtual_form.ok()) << test::describe(virtual_form);
    EXPECT_TRUE(handle.ok()) << test::describe(handle);
    EXPECT_EQ(direct.checks, 9u);
    EXPECT_EQ(virtual_form.checks, 9u);
    EXPECT_EQ(handle.checks, 9u);
}

struct counting_venue {
    mutable int calls = 0;
    [[nodiscard]] std::vector<onramp_asset> assets() const { ++calls; return {}; }
    capability_result<onramp_quote> quote(const onramp_quote_request&) { ++calls; return fail(capability_errc::unsupported); }
    capability_result<order> buy(const buy_request&) { ++calls; return fail(capability_errc::unsupported); }
    capability_result<order> order_status(std::string_view) { ++calls; return fail(capability_errc::not_found); }
    capability_result<withdrawal> withdraw(const withdraw_request&) { ++calls; return fail(capability_errc::unsupported); }
    capability_result<withdrawal> withdrawal_status(std::string_view) { ++calls; return fail(capability_errc::not_found); }
    capability_result<std::vector<venue_balance>> balances() { ++calls; return ok(std::vector<venue_balance>{}); }
    capability_result<onramp_statement_page> statement(const onramp_statement_query&) { ++calls; return fail(capability_errc::unsupported); }
};
static_assert(onramp_provider<onramp_provider_adapter<counting_venue>>);

TEST(OnrampProviderInterface, TheAdaptersForwardEveryOperation)
{
    counting_venue impl;
    onramp_provider_adapter<counting_venue&> borrowed{impl};
    onramp_provider_ref o{borrowed};
    EXPECT_TRUE(o.assets().empty());
    EXPECT_EQ(o.quote({}).unwrap_err().code, capability_errc::unsupported);
    EXPECT_EQ(o.buy({}).unwrap_err().code, capability_errc::unsupported);
    EXPECT_EQ(o.order_status("x").unwrap_err().code, capability_errc::not_found);
    EXPECT_EQ(o.withdraw({}).unwrap_err().code, capability_errc::unsupported);
    EXPECT_EQ(o.withdrawal_status("x").unwrap_err().code, capability_errc::not_found);
    EXPECT_TRUE(o.balances()->empty());
    EXPECT_EQ(o.statement({}).unwrap_err().code, capability_errc::unsupported);
    EXPECT_EQ(impl.calls, 8);
    onramp_provider_adapter owned{test::make_onramp_provider()};
    EXPECT_EQ(owned.assets().size(), 1u);
}

} // namespace
