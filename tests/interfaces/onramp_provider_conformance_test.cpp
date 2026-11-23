//
//  onramp_provider_conformance_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/interfaces/onramp_provider_conformance.hpp>: the kit passes the reference and catches broken
//  implementations: one that leaks the venue's response and one that ignores max_price.
//

#include <snicholls/core/interfaces/onramp_provider_conformance.hpp>

#include "capability_probes.hpp"

#include <gtest/gtest.h>

#include <optional>
#include <string>
#include <string_view>

using namespace snicholls;

namespace {

// The redaction rule's opposite: an implementation that copies the provider's response into its error.
capability_error leak(capability_errc code)
{
    return capability_error{code, *provider_code::make(capability_redaction_marker), {}};
}

struct broken_venue {
    memory_onramp_provider inner = test::make_onramp_provider();
    bool leaks = false;
    bool ignores_max_price = false;
    bool armed = false;
    [[nodiscard]] std::vector<onramp_asset> assets() const { return inner.assets(); }
    capability_result<onramp_quote> quote(const onramp_quote_request& q)
    {
        if (armed && leaks) {
            armed = false;
            return err(leak(capability_errc::unavailable));
        }
        return inner.quote(q);
    }
    capability_result<order> buy(buy_request b)
    {
        if (ignores_max_price)
            b.max_price.reset();
        return inner.buy(b);
    }
    capability_result<order> order_status(std::string_view id) { return inner.order_status(id); }
    capability_result<withdrawal> withdraw(const withdraw_request& w) { return inner.withdraw(w); }
    capability_result<withdrawal> withdrawal_status(std::string_view id) { return inner.withdrawal_status(id); }
    capability_result<std::vector<venue_balance>> balances() { return inner.balances(); }
    capability_result<onramp_statement_page> statement(const onramp_statement_query& q) { return inner.statement(q); }
};

template <bool Leaks, bool Ignores>
struct broken_probe {
    broken_venue v{test::make_onramp_provider(), Leaks, Ignores};
    auto& venue() { return v; }
    static asset unit() { return test::usdc; }
    static asset quote_currency() { return test::aud; }
    static money ticket() { return *parse_money(test::aud, "153"); }
    static money too_much() { return *parse_money(test::aud, "1000000"); }
    static std::uint64_t chain() { return 8453; }
    static std::string address() { return "0x52908400098527886E0F7030069857D2E4169EE7"; }
    void provider_fails(capability_errc c, std::string_view r)
    {
        v.armed = true;
        v.inner.fail_next(c, r);
    }
};

TEST(OnrampProviderConformance, TheReferencePasses)
{
    const auto r = run_onramp_provider_conformance<test::onramp_provider_probe<test::form::concept_model>>();
    EXPECT_TRUE(r.ok()) << test::describe(r);
}

TEST(OnrampProviderConformance, ALeakIsCaught)
{
    const auto r = run_onramp_provider_conformance<broken_probe<true, false>>();
    EXPECT_TRUE(r.failed("a provider failure never leaks its response")) << test::describe(r);
    EXPECT_EQ(r.failures.size(), 1u) << test::describe(r);
}

TEST(OnrampProviderConformance, AFillAboveMaxPriceIsCaught)
{
    const auto r = run_onramp_provider_conformance<broken_probe<false, true>>();
    EXPECT_TRUE(r.failed("a buy never fills above max_price or beyond the balance")) << test::describe(r);
}

} // namespace
