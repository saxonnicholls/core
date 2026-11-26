//
//  http_handler_reference_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/interfaces/http_handler_reference.hpp>: routing by method and exact path, 405 with Allow, 500 without detail, add's refusals.
//

#include <snicholls/core/interfaces/http_handler_reference.hpp>

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>
#include <utility>

using namespace snicholls;

namespace {

http_request req(http_method m, std::string path)
{
    http_request r;
    r.method = m;
    r.path = std::move(path);
    return r;
}

TEST(RouteTable, RoutesByMethodAndExactPath)
{
    route_table t;
    ASSERT_TRUE(t.add(http_method::get, "/a", [](const http_request&) { return respond(200, "get a"); }).is_ok());
    ASSERT_TRUE(t.add(http_method::post, "/a", [](const http_request& r) { return respond(201, r.body); }).is_ok());
    EXPECT_EQ(t.size(), 2u);
    EXPECT_EQ(t.handle(req(http_method::get, "/a")).body, "get a");
    auto post = req(http_method::post, "/a");
    post.body = "payload";
    EXPECT_EQ(t.handle(post).status, 201);
    EXPECT_EQ(t.handle(post).body, "payload");
    EXPECT_EQ(t.handle(req(http_method::get, "/a/")).status, 404);
}

TEST(RouteTable, AnotherMethodIs405WithAllow)
{
    route_table t;
    static_cast<void>(t.add(http_method::get, "/a", [](const http_request&) { return respond(200, "x"); }));
    static_cast<void>(t.add(http_method::put, "/a", [](const http_request&) { return respond(200, "x"); }));
    const auto r = t.handle(req(http_method::del, "/a"));
    EXPECT_EQ(r.status, 405);
    EXPECT_EQ(find_header(r.headers, "allow"), "GET, PUT");
}

TEST(RouteTable, AThrowingHandlerIs500WithNoDetail)
{
    route_table t;
    static_cast<void>(t.add(http_method::get, "/boom", [](const http_request&) -> http_response { throw std::runtime_error("db password=hunter2"); }));
    const auto r = t.handle(req(http_method::get, "/boom"));
    EXPECT_EQ(r.status, 500);
    EXPECT_EQ(r.body, "internal error");
}

TEST(RouteTable, AddRefusesDuplicatesBadPathsAndEmptyHandlers)
{
    route_table t;
    const auto h = [](const http_request&) { return respond(200, "x"); };
    ASSERT_TRUE(t.add(http_method::get, "/a", h).is_ok());
    EXPECT_EQ(t.add(http_method::get, "/a", h).unwrap_err().code, capability_errc::conflict);
    EXPECT_EQ(t.add(http_method::get, "a", h).unwrap_err().code, capability_errc::invalid_argument);
    EXPECT_EQ(t.add(http_method::get, "/b", nullptr).unwrap_err().code, capability_errc::invalid_argument);
    EXPECT_EQ(t.size(), 1u);
}

TEST(RouteTable, AnExplicitHeadRouteWins)
{
    route_table t;
    static_cast<void>(t.add(http_method::get, "/a", [](const http_request&) { return respond(200, "body"); }));
    static_cast<void>(t.add(http_method::head, "/a", [](const http_request&) { return respond(204); }));
    EXPECT_EQ(t.handle(req(http_method::head, "/a")).status, 204);
}

TEST(RouteTable, PatternsCaptureTheirSegments)
{
    route_table t;
    static_cast<void>(t.add(http_method::get, "/orders/open", [](const http_request&) { return respond(200, "exact"); }));
    ASSERT_TRUE(t.add_pattern(http_method::get, "/orders/:id", [](const http_request& r) {
        return respond(200, std::string(path_param(r, "id").value_or("?")));
    }).is_ok());
    ASSERT_TRUE(t.add_pattern(http_method::get, "/users/:user/orders/:id", [](const http_request& r) {
        return respond(200, std::string(*path_param(r, "user")) + "/" + std::string(*path_param(r, "id")));
    }).is_ok());
    EXPECT_EQ(t.handle(req(http_method::get, "/orders/open")).body, "exact");   // the exact route wins
    EXPECT_EQ(t.handle(req(http_method::get, "/orders/42")).body, "42");
    EXPECT_EQ(t.handle(req(http_method::get, "/users/ann/orders/7")).body, "ann/7");
    EXPECT_EQ(t.handle(req(http_method::get, "/orders/")).status, 404);        // a capture is never empty
    EXPECT_EQ(t.handle(req(http_method::get, "/orders/1/2")).status, 404);
    EXPECT_EQ(t.handle(req(http_method::post, "/orders/42")).status, 405);
    EXPECT_EQ(t.handle(req(http_method::head, "/orders/42")).body, "");
    EXPECT_EQ(t.size(), 3u);
    const auto h = [](const http_request&) { return respond(200); };
    EXPECT_EQ(t.add_pattern(http_method::get, "/orders/:id", h).unwrap_err().code, capability_errc::conflict);
    EXPECT_TRUE(t.add_pattern(http_method::post, "/orders/:id", h).is_ok());
    EXPECT_EQ(t.add_pattern(http_method::get, "x/:id", h).unwrap_err().code, capability_errc::invalid_argument);
    EXPECT_EQ(t.add_pattern(http_method::get, "/y/:id", nullptr).unwrap_err().code, capability_errc::invalid_argument);
}

} // namespace
