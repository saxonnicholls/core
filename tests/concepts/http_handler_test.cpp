//
//  http_handler_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/concepts/http_handler.hpp>: the models and the non-models, method tokens, respond.
//

#include <snicholls/core/concepts/http_handler.hpp>
#include <snicholls/core/interfaces/http_handler_reference.hpp>

#include <gtest/gtest.h>

#include <string>

using namespace snicholls;

namespace {

static_assert(http_handler<route_table>);
static_assert(http_handler<http_handler_interface>);
static_assert(http_handler<http_handler_adapter<route_table>>);
static_assert(http_handler<http_handler_adapter<route_table&>>);
static_assert(http_handler<http_handler_ref>);

struct returns_int {
    int handle(const http_request&);
};
struct takes_a_string {
    http_response handle(const std::string&);
};
static_assert(!http_handler<returns_int>);
static_assert(!http_handler<takes_a_string>);
static_assert(!http_handler<int>);

static_assert(to_string(http_method::del) == "DELETE" && to_string(http_method::options) == "OPTIONS");
static_assert(to_string(static_cast<http_method>(0)) == "UNKNOWN");
static_assert(parse_http_method("PATCH") == http_method::patch);
static_assert(!parse_http_method("get"), "tokens are case-sensitive");

TEST(HttpHandlerConcept, RespondSetsTheContentTypeOnlyWithABody)
{
    const auto with = respond(200, "{}", "application/json");
    EXPECT_EQ(with.status, 200);
    EXPECT_EQ(find_header(with.headers, "content-type"), "application/json");
    EXPECT_EQ(with.body, "{}");
    const auto without = respond(204);
    EXPECT_TRUE(without.headers.empty());
    for (const auto m : {http_method::get, http_method::head, http_method::post, http_method::put, http_method::patch,
                         http_method::del, http_method::options})
        EXPECT_EQ(parse_http_method(to_string(m)), m);
}

} // namespace
