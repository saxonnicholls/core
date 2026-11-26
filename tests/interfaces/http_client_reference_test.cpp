//
//  http_client_reference_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/interfaces/http_client_reference.hpp>: the loopback validates, forwards, strips HEAD's body,
//  fails once on command without keeping the response, and borrows a handler.
//

#include <snicholls/core/interfaces/http_client_reference.hpp>

#include "contract_probes.hpp"

#include <gtest/gtest.h>

#include <string_view>

using namespace snicholls;

namespace {

TEST(LoopbackHttpClient, ForwardsValidRequestsOnly)
{
    test::test_http_client c{test::echo_server()};
    http_request r;
    r.path = "/echo";
    r.body = "hi";
    EXPECT_EQ(c.send(r)->body, "hi");
    r.path = "echo";
    EXPECT_EQ(c.send(r).unwrap_err().code, capability_errc::invalid_argument);
    EXPECT_EQ(c.sent(), 1u);
}

TEST(LoopbackHttpClient, FailsOnceOnCommandAndKeepsOnlyTheCode)
{
    test::test_http_client c{test::echo_server()};
    c.fail_next(capability_errc::timeout, "gateway_timeout {\"secret\":\"x\"}");
    http_request echo;
    echo.path = "/echo";
    const auto r = c.send(echo);
    EXPECT_EQ(r.unwrap_err().code, capability_errc::timeout);
    EXPECT_EQ(r.unwrap_err().provider_code, std::string_view{"gateway_timeout"});
    http_request ok_request;
    ok_request.path = "/echo";
    EXPECT_TRUE(c.send(ok_request).is_ok());
    EXPECT_EQ(c.sent(), 1u);
}

TEST(LoopbackHttpClient, BorrowsAHandler)
{
    route_table routes = test::echo_server();
    loopback_http_client<route_table&> c{routes};
    EXPECT_EQ(&c.handler(), &routes);
    http_request head;
    head.method = http_method::head;
    head.path = "/echo";
    head.body = "x";
    EXPECT_TRUE(c.send(head)->body.empty());
}

} // namespace
