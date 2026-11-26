//
//  http_client_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/concepts/http_client.hpp>: the models and the non-models, validate_request and http_get.
//

#include <snicholls/core/concepts/http_client.hpp>
#include <snicholls/core/interfaces/http_client_reference.hpp>
#include <snicholls/core/interfaces/http_handler_reference.hpp>

#include <gtest/gtest.h>

#include <string_view>

using namespace snicholls;

namespace {

static_assert(http_client<loopback_http_client<route_table>>);
static_assert(http_client<loopback_http_client<route_table&>>);
static_assert(http_client<http_client_interface>);
static_assert(http_client<http_client_adapter<loopback_http_client<route_table>>>);
static_assert(http_client<http_client_ref>);

struct throws_instead {
    http_response send(const http_request&);
};
static_assert(!http_client<throws_instead>);
static_assert(!http_client<route_table>);   // a handler is not a client
static_assert(!http_client<int>);

TEST(HttpClientConcept, ValidateRequestNamesTheFieldNeverItsValue)
{
    http_request r;
    EXPECT_EQ(validate_request(r).unwrap_err().message, std::string_view{"a path starts with /"});
    r.path = "/a";
    EXPECT_TRUE(validate_request(r).is_ok());
    r.query = "x=1\r\nHost: evil";
    EXPECT_EQ(validate_request(r).unwrap_err().message, std::string_view{"a line break in the target"});
    r.query.clear();
    r.headers.push_back({"X: Y", "v"});
    EXPECT_EQ(validate_request(r).unwrap_err().code, capability_errc::invalid_argument);
    r.headers = {{"", "v"}};
    EXPECT_TRUE(validate_request(r).is_err());
}

TEST(HttpClientConcept, HttpGetSplitsTheTarget)
{
    route_table t;
    static_cast<void>(t.add(http_method::get, "/q", [](const http_request& r) { return respond(200, r.query + "|" + std::string(*find_header(r.headers, "A"))); }));
    loopback_http_client c{std::move(t)};
    EXPECT_EQ(http_get(c, "/q?x=1&y=2", {{"A", "b"}})->body, "x=1&y=2|b");
    EXPECT_EQ(http_get(c, "/q", {{"A", "c"}})->body, "|c");
}

} // namespace
