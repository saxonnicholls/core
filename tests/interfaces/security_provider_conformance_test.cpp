//
//  security_provider_conformance_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/interfaces/security_provider_conformance.hpp>: the kit passes the reference and catches
//  broken implementations: one that leaks the provider's response, one that accepts unsigned webhooks, one that
//  counts a trade as settled on its trade date, and one that books test deliveries.
//

#include <snicholls/core/interfaces/security_provider_conformance.hpp>

#include "security_provider_probe.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <string_view>
#include <vector>

using namespace snicholls;

namespace {

enum class defect { leaks, trusts_unsigned, settles_at_once, books_tests };

template <defect D>
struct broken_provider {
    memory_security_provider inner = test::make_security_provider();
    bool armed = false;
    [[nodiscard]] security_capabilities capabilities() const { return inner.capabilities(); }
    [[nodiscard]] capability_result<webhook_event> verify_webhook(const webhook_delivery& w) const
    {
        auto v = inner.verify_webhook(w);
        if (D == defect::trusts_unsigned && v.is_err() && v.unwrap_err().code == capability_errc::signature_invalid)
            return ok(webhook_event{});   // the fail-open defect
        return v;
    }
    capability_result<std::vector<security_fact>> resolve(const webhook_event& e)
    {
        if (D == defect::books_tests && e.is_test) {
            webhook_event real = e;
            real.is_test = false;
            return inner.resolve(real);
        }
        return inner.resolve(e);
    }
    capability_result<security_statement_page> statement(const security_statement_query& q)
    {
        if (D == defect::leaks && armed) {
            armed = false;
            return err(capability_error{capability_errc::unavailable, *provider_code::make(capability_redaction_marker), {}});
        }
        return inner.statement(q);
    }
    capability_result<std::vector<security_holding>> holdings(std::chrono::sys_days as_of)
    {
        auto h = inner.holdings(as_of);
        if (D == defect::settles_at_once && h.is_ok())
            for (auto& x : *h)
                x.settled = x.traded;
        return h;
    }
};

template <defect D>
struct broken_probe {
    broken_provider<D> p{};
    auto& provider() { return p; }
    static money quantity() { return test::security_provider_probe<test::form::concept_model>::quantity(); }
    static money consideration() { return test::security_provider_probe<test::form::concept_model>::consideration(); }
    void trade(std::string_view order, const money& q, const money& c) { static_cast<void>(p.inner.execute(order, q, c)); }
    void settle(std::string_view order) { static_cast<void>(p.inner.settle(order)); }
    owned_delivery deliver(const webhook_event& e) { return p.inner.signed_webhook(e); }
    owned_delivery deliver_late(const webhook_event& e) { return p.inner.signed_webhook(e, test::probe_time - std::chrono::minutes{6}); }
    void provider_fails(capability_errc c, std::string_view r)
    {
        p.armed = true;
        p.inner.fail_next(c, r);
    }
};

TEST(SecurityProviderConformance, TheReferencePasses)
{
    const auto r = run_security_provider_conformance<test::security_provider_probe<test::form::concept_model>>();
    EXPECT_TRUE(r.ok()) << test::describe(r);
    EXPECT_EQ(r.checks, 8u);
}

TEST(SecurityProviderConformance, ALeakIsCaught)
{
    const auto r = run_security_provider_conformance<broken_probe<defect::leaks>>();
    EXPECT_TRUE(r.failed("a provider failure never leaks its response")) << test::describe(r);
    EXPECT_EQ(r.failures.size(), 1u) << test::describe(r);
}

TEST(SecurityProviderConformance, AWebhookThatFailsOpenIsCaught)
{
    const auto r = run_security_provider_conformance<broken_probe<defect::trusts_unsigned>>();
    EXPECT_TRUE(r.failed("a webhook fails closed")) << test::describe(r);
}

TEST(SecurityProviderConformance, SettlingOnTheTradeDateIsCaught)
{
    const auto r = run_security_provider_conformance<broken_probe<defect::settles_at_once>>();
    EXPECT_TRUE(r.failed("settlement moves the settled holding")) << test::describe(r);
    EXPECT_EQ(r.failures.size(), 1u) << test::describe(r);
}

TEST(SecurityProviderConformance, BookingATestDeliveryIsCaught)
{
    const auto r = run_security_provider_conformance<broken_probe<defect::books_tests>>();
    EXPECT_TRUE(r.failed("an event that books nothing resolves to no facts")) << test::describe(r);
}

} // namespace
