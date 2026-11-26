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

TEST(HttpHandlerConcept, MatchRouteCapturesNamedSegments)
{
    std::vector<http_header> c;
    EXPECT_TRUE(match_route("/a/:x/b", "/a/1/b", c));
    ASSERT_EQ(c.size(), 1u);
    EXPECT_EQ(c[0].name, "x");
    EXPECT_EQ(c[0].value, "1");
    EXPECT_TRUE(match_route("/", "/", c));
    EXPECT_FALSE(match_route("/a/:x", "/a", c));
    EXPECT_FALSE(match_route("/a/:x", "/a/1/2", c));
    EXPECT_FALSE(match_route("/a/b", "/a/c", c));
    EXPECT_FALSE(match_route("/a", "a", c));
    EXPECT_EQ(c.size(), 1u);   // a failed match appends nothing
}

TEST(HttpHandlerConcept, RequestHelpers)
{
    http_request r;
    r.path = "/search";
    EXPECT_EQ(request_target(r), "/search");
    r.query = "q=x&n=2";
    EXPECT_EQ(request_target(r), "/search?q=x&n=2");
    r.path_params.push_back({"id", "42"});
    EXPECT_EQ(path_param(r, "id"), std::optional<std::string_view>{"42"});
    EXPECT_EQ(path_param(r, "ID"), std::nullopt);
    EXPECT_TRUE(is_success(respond(204)));
    EXPECT_FALSE(is_success(respond(302)));
    EXPECT_FALSE(is_success(respond(404)));
}

} // namespace
