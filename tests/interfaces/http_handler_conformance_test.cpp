//
//  http_handler_conformance_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/interfaces/http_handler_conformance.hpp>: the kit passes the reference and catches broken
//  implementations: one that answers an exception with its message and one that answers HEAD with a body.
//

#include <snicholls/core/interfaces/http_handler_conformance.hpp>

#include "capability_probes.hpp"

#include <gtest/gtest.h>

#include <optional>
#include <string>
#include <string_view>

using namespace snicholls;

namespace {

// Answers like the reference, but a throwing route's message goes into the 500, or HEAD keeps its body.
struct broken_handler {
    route_table inner = test::make_routes();
    bool echoes_exceptions = false;
    bool head_has_body = false;
    http_response handle(const http_request& r)
    {
        if (echoes_exceptions && r.path == "/boom")
            return respond(500, "error: db password=" + std::string(capability_redaction_marker));
        if (head_has_body && r.method == http_method::head) {
            http_request get = r;
            get.method = http_method::get;
            return inner.handle(get);
        }
        return inner.handle(r);
    }
};

template <bool Echoes, bool HeadBody>
struct broken_probe {
    broken_handler h{test::make_routes(), Echoes, HeadBody};
    auto& handler() { return h; }
    static http_request known() { return test::request(http_method::get, "/health"); }
    static http_request unknown() { return test::request(http_method::get, "/nowhere"); }
    static http_request throwing() { return test::request(http_method::get, "/boom"); }
};

TEST(HttpHandlerConformance, TheReferencePasses)
{
    const auto r = run_http_handler_conformance<test::http_handler_probe<test::form::concept_model>>();
    EXPECT_TRUE(r.ok()) << test::describe(r);
}

TEST(HttpHandlerConformance, AnEchoedExceptionIsCaught)
{
    const auto r = run_http_handler_conformance<broken_probe<true, false>>();
    EXPECT_TRUE(r.failed("a throwing handler answers 500 without detail")) << test::describe(r);
    EXPECT_EQ(r.failures.size(), 1u) << test::describe(r);
}

TEST(HttpHandlerConformance, AHeadWithABodyIsCaught)
{
    const auto r = run_http_handler_conformance<broken_probe<false, true>>();
    EXPECT_TRUE(r.failed("HEAD is GET without a body")) << test::describe(r);
}

} // namespace
