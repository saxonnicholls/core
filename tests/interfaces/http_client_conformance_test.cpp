//
//  http_client_conformance_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/interfaces/http_client_conformance.hpp>: the kit passes the reference and catches broken
//  clients: one that turns error statuses into errors, one that sends header injections, one that leaks the transport's
//  response.
//

#include <snicholls/core/interfaces/http_client_conformance.hpp>

#include "contract_probes.hpp"

#include <gtest/gtest.h>

#include <string_view>

using namespace snicholls;

namespace {

enum class defect { status_is_error, no_validation, leaks };

template <defect D>
struct broken_client {
    route_table server = test::echo_server();
    bool armed = false;
    capability_result<http_response> send(const http_request& r)
    {
        if (armed) {
            armed = false;
            return err(capability_error{capability_errc::unavailable,
                                        D == defect::leaks ? *provider_code::make(capability_redaction_marker) : provider_code{}, {}});
        }
        if (D != defect::no_validation)
            if (auto v = validate_request(r); !v)
                return err(v.unwrap_err());
        http_response out = server.handle(r);
        if (D == defect::status_is_error && out.status >= 400)
            return fail(capability_errc::not_found);
        if (r.method == http_method::head)
            out.body.clear();
        return ok(std::move(out));
    }
};

template <defect D>
struct broken_probe {
    broken_client<D> c;
    auto& client() { return c; }
    void transport_fails(capability_errc, std::string_view) { c.armed = true; }
};

TEST(HttpClientConformance, TheReferencePasses)
{
    const auto r = run_http_client_conformance<test::http_client_probe<test::form::concept_model>>();
    EXPECT_TRUE(r.ok()) << test::describe(r);
}

TEST(HttpClientConformance, BrokenClientsAreCaught)
{
    const auto a = run_http_client_conformance<broken_probe<defect::status_is_error>>();
    EXPECT_TRUE(a.failed("an error status is a response")) << test::describe(a);
    const auto b = run_http_client_conformance<broken_probe<defect::no_validation>>();
    EXPECT_TRUE(b.failed("a malformed request is refused")) << test::describe(b);
    const auto c = run_http_client_conformance<broken_probe<defect::leaks>>();
    EXPECT_TRUE(c.failed("a transport failure leaks nothing")) << test::describe(c);
    EXPECT_EQ(c.failures.size(), 1u) << test::describe(c);
}

} // namespace
