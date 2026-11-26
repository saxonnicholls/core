//
//  https_client_reference_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/interfaces/https_client_reference.hpp>: routing by host, to handlers and to an
//  http_handler; what the memory client records (never the credential or a sensitive body); cassettes saved and
//  loaded byte for byte; replay in order, by method, URL, idempotency key and body; malformed cassettes.
//

#include <snicholls/core/interfaces/http_handler_reference.hpp>
#include <snicholls/core/interfaces/https_client_reference.hpp>

#include <gtest/gtest.h>

#include <string>

using namespace snicholls;

namespace {

https_request post(std::string url, std::string body, std::string_view key = {})
{
    https_request r;
    r.method = http_method::post;
    r.url = std::move(url);
    r.body = std::move(body);
    r.idempotency_key = *client_id::make(key);
    r.authorization = secret<std::string>{"Bearer sk_test_secret_value"};
    r.headers = {{"Content-Type", "application/x-www-form-urlencoded"}};
    return r;
}

memory_https_client echo()
{
    return memory_https_client{[](const https_request& r) -> capability_result<https_response> {
        return ok(https_response{200, {{"Set-Cookie", "s=1"}, {"Request-Id", "req_1"}}, "echo:" + r.body});
    }};
}

TEST(MemoryHttpsClient, RoutesByHostIgnoringCase)
{
    memory_https_client c;
    c.route("api.stripe.test", [](const https_request&) -> capability_result<https_response> { return ok(https_response{200, {}, "stripe"}); });
    c.route("api.wise.test", [](const https_request&) -> capability_result<https_response> { return ok(https_response{200, {}, "wise"}); });
    https_request r;
    r.url = "https://API.Stripe.test:8443/v1/x";
    EXPECT_EQ(c.send(r)->body, "stripe");
    r.url = "https://api.wise.test";
    EXPECT_EQ(c.send(r)->body, "wise");
    c.route("api.wise.test", [](const https_request&) -> capability_result<https_response> { return ok(https_response{201, {}, "wise2"}); });
    EXPECT_EQ(c.send(r)->body, "wise2") << "a route is replaced";
    r.url = "https://api.okx.test/";
    EXPECT_EQ(c.send(r).unwrap_err().code, capability_errc::unavailable);
    EXPECT_EQ(c.transport_failures(), 1u);
    EXPECT_EQ(c.seen().exchanges.size(), 3u);
}

TEST(MemoryHttpsClient, ServesAnHttpHandler)
{
    route_table routes;
    static_cast<void>(routes.add(http_method::post, "/v1/refunds", [](const http_request& in) {
        const auto auth = find_header(in.headers, "Authorization").value_or("");
        const auto key = find_header(in.headers, "Idempotency-Key").value_or("");
        return respond(201, std::string(auth) + "|" + std::string(key) + "|" + in.query + "|" + in.body);
    }));
    memory_https_client c;
    c.serve("api.example.test", routes);
    auto r = post("https://api.example.test/v1/refunds?expand=x", "amount=1", "k-1");
    const auto got = c.send(r);
    ASSERT_TRUE(got.is_ok());
    EXPECT_EQ(got->status, 201);
    EXPECT_EQ(got->body, "Bearer sk_test_secret_value|k-1|expand=x|amount=1");
    r.url = "https://api.example.test";
    r.method = http_method::get;
    EXPECT_EQ(c.send(r)->status, 404) << "an empty target is /";
}

TEST(MemoryHttpsClient, RecordsWithoutTheCredential)
{
    auto c = echo();
    auto r = post("https://api.test/v1/x", "a=1", "k-1");
    ASSERT_TRUE(c.send(r).is_ok());
    const auto& x = c.seen().exchanges.front();
    EXPECT_TRUE(x.had_authorization);
    EXPECT_EQ(x.idempotency_key(), "k-1");
    EXPECT_FALSE(find_header(x.response.headers, "Set-Cookie").has_value()) << "cookies are dropped";
    EXPECT_EQ(c.seen().save().find("sk_test_secret_value"), std::string::npos);
    r.headers.push_back({"authorization", "Bearer smuggled"});
    ASSERT_TRUE(c.send(r).is_ok());
    EXPECT_EQ(c.seen().save().find("smuggled"), std::string::npos) << "an Authorization header is never recorded either";
    memory_https_client quiet{[](const https_request&) -> capability_result<https_response> { return ok(https_response{200, {}, "ok"}); }};
    r.sensitive_body = true;
    r.body = "otp=123456";
    ASSERT_TRUE(quiet.send(r).is_ok());
    EXPECT_EQ(quiet.seen().exchanges.back().request_body, cassette::redacted);
    EXPECT_EQ(quiet.seen().save().find("123456"), std::string::npos) << "a sensitive body is never recorded";
}

TEST(MemoryHttpsClient, RefusesWhatARealClientWould)
{
    auto c = echo();
    https_request r;
    r.url = "http://api.test/";
    EXPECT_EQ(c.send(r).unwrap_err().code, capability_errc::invalid_argument);
    r.url = "https://api.test/";
    r.headers = {{"Bad:Name", "v"}};
    EXPECT_EQ(c.send(r).unwrap_err().code, capability_errc::invalid_argument);
    r.headers.clear();
    r.idempotency_key = "k\r\nX: 1";
    EXPECT_EQ(c.send(r).unwrap_err().code, capability_errc::invalid_argument);
    memory_https_client down{[](const https_request&) -> capability_result<https_response> { return fail(capability_errc::timeout); }};
    r.idempotency_key = {};
    EXPECT_EQ(down.send(r).unwrap_err().code, capability_errc::timeout);
    EXPECT_EQ(down.transport_failures(), 1u);
    EXPECT_TRUE(down.seen().exchanges.empty());
}

TEST(MemoryHttpsClient, AnArmedFailureIsReturnedOnce)
{
    auto c = echo();
    https_request r;
    r.url = "https://api.test/";
    c.fail_next(capability_errc::unavailable, "tls_handshake_failed {\"cert\":\"never kept\"}");
    const auto e = c.send(r).unwrap_err();
    EXPECT_EQ(e.code, capability_errc::unavailable);
    EXPECT_EQ(e.provider_code.view(), "tls_handshake_failed");
    EXPECT_TRUE(c.send(r).is_ok());
    EXPECT_EQ(c.transport_failures(), 1u);
}

TEST(HttpsCassette, SavesAndLoadsByteForByte)
{
    recording_https_client<memory_https_client> rec{echo()};
    static_cast<void>(rec.send(post("https://api.test/v1/a", "line1\nline2\n< 200\n> b 9", "k")));
    https_request get;
    get.url = "https://api.test/v1/b?x=1";
    static_cast<void>(rec.send(get));
    const std::string text = rec.tape().save();
    const auto back = cassette::load(text);
    ASSERT_TRUE(back.is_ok());
    EXPECT_EQ(*back, rec.tape());
    EXPECT_EQ(back->save(), text);
    EXPECT_TRUE(cassette::load("").is_ok());
    for (const char* bad : {"> GET https://x\n> b 3\nab\n", "garbage\n", "> GET https://x\n> b 0\n\n< 2x0\n",
                            "> BREW https://x\n> b 0\n\n< 200\n< b 0\n\n", "> GET https://x\n> h nocolon\n",
                            "> GET https://x\n> b 0\n\n< 200\n< h nocolon\n", "> GET https://x\n> b 0\n\n< 200\n",
                            "> GET https://x\n> b 0\n\n", "> GET https://x\n> b 99\nab\n", "> GET https://x\n> q\n",
                            "> GET https://x\n> b 0\n\n< 200\n< q\n", "> GET\n"})
        EXPECT_EQ(cassette::load(bad).unwrap_err().code, capability_errc::invalid_argument) << bad;
}

TEST(HttpsReplay, AnswersInOrderAndOnlyWhatWasRecorded)
{
    recording_https_client<memory_https_client> rec{echo()};
    static_cast<void>(rec.send(post("https://api.test/v1/a", "n=1", "k1")));
    static_cast<void>(rec.send(post("https://api.test/v1/a", "n=1", "k1")));
    static_cast<void>(rec.send(post("https://api.test/v1/a", "n=2", "k2")));
    replay_https_client replay{rec.tape()};
    EXPECT_EQ(replay.send(post("https://api.test/v1/a", "n=2", "k2"))->body, "echo:n=2");
    EXPECT_EQ(replay.send(post("https://api.test/v1/a", "n=1", "k1"))->body, "echo:n=1");
    EXPECT_EQ(replay.unused(), 1u);
    EXPECT_EQ(replay.send(post("https://api.test/v1/a", "n=1", "other-key")).unwrap_err().code, capability_errc::not_found);
    EXPECT_EQ(replay.send(post("https://api.test/v1/a", "n=3", "k1")).unwrap_err().code, capability_errc::not_found);
    auto get = post("https://api.test/v1/a", "n=1", "k1");
    get.method = http_method::get;
    EXPECT_EQ(replay.send(get).unwrap_err().code, capability_errc::not_found);
    EXPECT_TRUE(replay.send(post("https://api.test/v1/a", "n=1", "k1")).is_ok());
    EXPECT_EQ(replay.send(post("https://api.test/v1/a", "n=1", "k1")).unwrap_err().code, capability_errc::not_found) << "each exchange once";
}

TEST(HttpsReplay, ARedactedBodyReplaysForAnyBody)
{
    recording_https_client<memory_https_client> rec{echo()};
    auto r = post("https://api.test/v1/sca", "signature=abc");
    r.sensitive_body = true;
    static_cast<void>(rec.send(r));
    replay_https_client replay{rec.tape()};
    r.body = "signature=different";
    EXPECT_TRUE(replay.send(r).is_ok());
}

TEST(HttpsRecording, RecordsOnlyAnsweredExchanges)
{
    memory_https_client inner;   // no routes
    recording_https_client<memory_https_client&> rec{inner};
    https_request r;
    r.url = "https://nowhere.test/";
    EXPECT_TRUE(rec.send(r).is_err());
    EXPECT_TRUE(rec.tape().exchanges.empty());
}

} // namespace
