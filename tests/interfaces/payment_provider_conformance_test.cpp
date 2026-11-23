//
//  payment_provider_conformance_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/interfaces/payment_provider_conformance.hpp>: the kit passes the reference and catches broken
//  implementations: one that leaks the provider's response and one that accepts unsigned webhooks.
//

#include <snicholls/core/interfaces/payment_provider_conformance.hpp>

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

struct broken_provider {
    memory_payment_provider inner = test::make_payment_provider();
    bool leaks = false;
    bool trusts_unsigned = false;
    bool armed = false;
    [[nodiscard]] provider_capabilities capabilities() const { return inner.capabilities(); }
    capability_result<collection> collect(const collection_request& c)
    {
        if (armed && leaks) {
            armed = false;
            return err(leak(capability_errc::rejected));
        }
        return inner.collect(c);
    }
    capability_result<payment> payout(const payout_request& o) { return inner.payout(o); }
    capability_result<payment> refund(const refund_request& r) { return inner.refund(r); }
    capability_result<fx_quote> quote(const quote_request& q) { return inner.quote(q); }
    capability_result<payment> status(std::string_view id) { return inner.status(id); }
    capability_result<std::vector<money>> balances() { return inner.balances(); }
    capability_result<std::vector<credit>> credits(const credit_query& q) { return inner.credits(q); }
    [[nodiscard]] capability_result<webhook_event> verify_webhook(const webhook_delivery& w) const
    {
        auto v = inner.verify_webhook(w);
        if (trusts_unsigned && v.is_err() && v.unwrap_err().code == capability_errc::signature_invalid)
            return ok(webhook_event{});   // the fail-open defect
        return v;
    }
    capability_result<std::vector<payment_fact>> resolve(const webhook_event& e) { return inner.resolve(e); }
    capability_result<statement_page> statement(const statement_query& q) { return inner.statement(q); }
};

template <bool Leaks, bool Trusts>
struct broken_probe {
    broken_provider p{test::make_payment_provider(), Leaks, Trusts};
    auto& provider() { return p; }
    static money amount() { return test::payment_provider_probe<test::form::concept_model>::amount(); }
    static asset unsupported_currency() { return checked_asset("XYZ", 2); }
    void payer_pays(const collection& c) { static_cast<void>(p.inner.complete(c.pay.payment_id.view())); }
    owned_delivery deliver(const webhook_event& e) { return p.inner.signed_webhook(e); }
    owned_delivery deliver_late(const webhook_event& e) { return p.inner.signed_webhook(e, test::probe_time - std::chrono::minutes{6}); }
    void provider_fails(capability_errc c, std::string_view r)
    {
        p.armed = true;
        p.inner.fail_next(c, r);
    }
};

TEST(PaymentProviderConformance, TheReferencePasses)
{
    const auto r = run_payment_provider_conformance<test::payment_provider_probe<test::form::concept_model>>();
    EXPECT_TRUE(r.ok()) << test::describe(r);
}

TEST(PaymentProviderConformance, ALeakIsCaught)
{
    const auto r = run_payment_provider_conformance<broken_probe<true, false>>();
    EXPECT_TRUE(r.failed("a provider failure never leaks its response")) << test::describe(r);
    EXPECT_EQ(r.failures.size(), 1u) << test::describe(r);
}

TEST(PaymentProviderConformance, AWebhookThatFailsOpenIsCaught)
{
    const auto r = run_payment_provider_conformance<broken_probe<false, true>>();
    EXPECT_TRUE(r.failed("a webhook fails closed")) << test::describe(r);
}

} // namespace
