//
//  https_client_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/concepts/https_client.hpp>: the models and the non-models, idempotency, sendable
//  headers, URL parsing, the credential printing masked.
//

#include <snicholls/core/concepts/https_client.hpp>
#include <snicholls/core/interfaces/https_client_reference.hpp>

#include <gtest/gtest.h>

#include <sstream>
#include <string>
#include <string_view>

using namespace snicholls;

namespace {

static_assert(https_client<memory_https_client>);
static_assert(https_client<replay_https_client>);
static_assert(https_client<recording_https_client<memory_https_client>>);
static_assert(https_client<recording_https_client<memory_https_client&>>);
static_assert(https_client<https_client_interface>);
static_assert(https_client<https_client_adapter<memory_https_client>>);
static_assert(https_client<https_client_adapter<memory_https_client&>>);
static_assert(https_client<https_client_ref>);

struct no_send {};
struct send_returns_a_string {
    std::string send(const https_request&);
};
struct send_takes_a_url {
    capability_result<https_response> send(std::string_view url);
};
static_assert(!https_client<no_send>);
static_assert(!https_client<send_returns_a_string>);
static_assert(!https_client<send_takes_a_url>);

static_assert(is_idempotent(http_method::get) && is_idempotent(http_method::del) && is_idempotent(http_method::put));
static_assert(is_idempotent(http_method::head) && is_idempotent(http_method::options));
static_assert(!is_idempotent(http_method::post) && !is_idempotent(http_method::patch));
static_assert(https_response{204, {}, {}}.ok() && !https_response{301, {}, {}}.ok() && !https_response{199, {}, {}}.ok());

constexpr auto parsed = parse_https_url("https://api.stripe.com:8443/v1/refunds?limit=3");
static_assert(parsed.is_ok() && parsed->host == "api.stripe.com" && parsed->port == 8443 && parsed->target == "/v1/refunds?limit=3");
static_assert(parse_https_url("https://a.test")->port == 443 && parse_https_url("https://a.test")->target.empty());
static_assert(parse_https_url("https://a.test?x=1")->target == "?x=1");
static_assert(parse_https_url("http://a.test/").is_err());
static_assert(parse_https_url("https://").is_err());
static_assert(parse_https_url("https:///path").is_err());
static_assert(parse_https_url("https://user@a.test/").is_err());
static_assert(parse_https_url("https://a.test:0/").is_err() && parse_https_url("https://a.test:65536/").is_err());
static_assert(parse_https_url("https://a.test:/").is_err() && parse_https_url("https://a.test:44a/").is_err());
static_assert(parse_https_url("https://a.test:123456/").is_err());
static_assert(parse_https_url("https://[::1]/").is_err());
static_assert(parse_https_url("https://a.test/#frag").is_err());
static_assert(parse_https_url("https://a.test:65535/")->port == 65535);

TEST(HttpsClientConcept, OnlyCleanHeadersAreSendable)
{
    EXPECT_TRUE(is_sendable({"Content-Type", "application/json"}));
    EXPECT_TRUE(is_sendable({"X-Custom_Header.1", ""}));
    EXPECT_FALSE(is_sendable({"", "v"}));
    EXPECT_FALSE(is_sendable({"Bad:Name", "v"}));
    EXPECT_FALSE(is_sendable({"Bad Name", "v"}));
    EXPECT_FALSE(is_sendable({"X", "a\r\nInjected: yes"}));
    EXPECT_FALSE(is_sendable({"X", "a\nb"}));
    EXPECT_FALSE(is_sendable({"X", std::string("a\0b", 3)}));
}

TEST(HttpsClientConcept, TheCredentialPrintsMasked)
{
    https_request r;
    r.authorization = secret<std::string>{"Bearer sk_test_never_printed"};
    std::ostringstream out;
    out << r.authorization;
    EXPECT_EQ(out.str().find("sk_test"), std::string::npos);
}

TEST(HttpsClientConcept, RequestsCompareByEveryMember)
{
    https_request a;
    a.url = "https://a.test/";
    https_request b = a;
    EXPECT_EQ(a, b);
    b.client_certificate = "bank-mtls-1";
    EXPECT_NE(a, b);
    b = a;
    b.timeout = std::chrono::seconds{5};
    EXPECT_NE(a, b);
}

} // namespace
