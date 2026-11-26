//
//  https_client_conformance_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/interfaces/https_client_conformance.hpp>: the kit passes the reference and catches broken
//  implementations: one that leaks the TLS layer's response, one that sends http://, one that treats a 5xx as an
//  error.
//

#include <snicholls/core/interfaces/https_client_conformance.hpp>

#include "capability_probes.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <string_view>

using namespace snicholls;

namespace {

enum class defect { leaks, sends_plain_http, fails_on_5xx };

template <defect D>
struct broken_client {
    memory_https_client inner;
    bool armed = false;
    capability_result<https_response> send(const https_request& r)
    {
        if (D == defect::leaks && armed) {
            armed = false;
            return err(capability_error{capability_errc::unavailable, *provider_code::make(capability_redaction_marker), {}});
        }
        if (D == defect::sends_plain_http && r.url.starts_with("http://"))
            return ok(https_response{200, {}, ""});
        auto got = inner.send(r);
        if (D == defect::fails_on_5xx && got.is_ok() && got->status >= 500)
            return fail(capability_errc::unavailable);
        return got;
    }
};

template <defect D>
struct broken_probe {
    std::shared_ptr<test::scripted_server> server = std::make_shared<test::scripted_server>();
    broken_client<D> c{test::routed_to(server)};
    auto& client() { return c; }
    static std::string url(std::string_view p) { return "https://api.example.test" + std::string(p); }
    static std::string unreachable_url() { return "https://unreachable.example.test/"; }
    void answer(https_response r) { server->next = std::move(r); }
    https_observed_request seen() { return server->last; }
    void provider_fails(capability_errc code, std::string_view r)
    {
        c.armed = true;
        c.inner.fail_next(code, r);
    }
};

TEST(HttpsClientConformance, TheReferencePasses)
{
    const auto r = run_https_client_conformance<test::https_client_probe<test::form::concept_model>>();
    EXPECT_TRUE(r.ok()) << test::describe(r);
    EXPECT_EQ(r.checks, 7u);
}

TEST(HttpsClientConformance, ALeakIsCaught)
{
    const auto r = run_https_client_conformance<broken_probe<defect::leaks>>();
    EXPECT_TRUE(r.failed("a provider failure never leaks its response")) << test::describe(r);
    EXPECT_EQ(r.failures.size(), 1u) << test::describe(r);
}

TEST(HttpsClientConformance, SendingPlainHttpIsCaught)
{
    const auto r = run_https_client_conformance<broken_probe<defect::sends_plain_http>>();
    EXPECT_TRUE(r.failed("only https URLs and clean headers are sent")) << test::describe(r);
}

TEST(HttpsClientConformance, TreatingA5xxAsAnErrorIsCaught)
{
    const auto r = run_https_client_conformance<broken_probe<defect::fails_on_5xx>>();
    EXPECT_TRUE(r.failed("an error status is a response, not an error")) << test::describe(r);
}

} // namespace
