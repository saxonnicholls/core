//
//  http_handler_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/interfaces/http_handler.hpp>: the conformance kit passes on the reference in every form (the
//  concept, the virtual interface over a borrowing adapter, the handle), and the adapters forward every operation.
//

#include <snicholls/core/interfaces/http_handler.hpp>
#include <snicholls/core/interfaces/http_handler_conformance.hpp>

#include "capability_probes.hpp"

#include <gtest/gtest.h>

#include <string_view>
#include <type_traits>
#include <vector>

using namespace snicholls;

namespace {

static_assert(http_handler<http_handler_interface>);
static_assert(http_handler<http_handler_ref>);
static_assert(std::is_abstract_v<http_handler_interface>);

TEST(HttpHandlerInterface, TheKitPassesInEveryForm)
{
    const auto direct = run_http_handler_conformance<test::http_handler_probe<test::form::concept_model>>();
    const auto virtual_form = run_http_handler_conformance<test::http_handler_probe<test::form::interface>>();
    const auto handle = run_http_handler_conformance<test::http_handler_probe<test::form::ref>>();
    EXPECT_TRUE(direct.ok()) << test::describe(direct);
    EXPECT_TRUE(virtual_form.ok()) << test::describe(virtual_form);
    EXPECT_TRUE(handle.ok()) << test::describe(handle);
    EXPECT_EQ(direct.checks, 5u);
    EXPECT_EQ(virtual_form.checks, 5u);
    EXPECT_EQ(handle.checks, 5u);
}

struct counting_handler {
    int calls = 0;
    http_response handle(const http_request&) { ++calls; return respond(204); }
};
static_assert(http_handler<http_handler_adapter<counting_handler>>);

TEST(HttpHandlerInterface, TheAdaptersForwardEveryOperation)
{
    counting_handler impl;
    http_handler_adapter<counting_handler&> borrowed{impl};
    http_handler_ref h{borrowed};
    EXPECT_EQ(h.handle({}).status, 204);
    EXPECT_EQ(impl.calls, 1);
    http_handler_adapter owned{test::make_routes()};
    EXPECT_EQ(owned.handle(test::request(http_method::get, "/health")).body, "ok");
}

} // namespace
