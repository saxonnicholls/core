//
//  https_client_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/concepts/https_client.hpp>: a provider call through the concept, URL parsing, idempotency
//  and the header check. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/concepts/https_client.hpp>
#include <snicholls/core/interfaces/https_client_reference.hpp>

#include <cstdio>
#include <string>

using namespace snicholls;

namespace {
int failures = 0;
void check(bool ok, const char* what)
{
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (!ok)
        ++failures;
}

//! [concept]
// A provider adapter's call: any client, a POST that changes state, so it carries an idempotency key.
template <https_client C>
capability_result<https_response> refund(C& client, std::string_view payment, std::string_view key)
{
    https_request r;
    r.method = http_method::post;
    r.url = "https://api.stripe.test/v1/refunds";
    r.headers = {{"Content-Type", "application/x-www-form-urlencoded"}};
    r.authorization = secret<std::string>{"Bearer sk_test_demo"};
    r.body = "payment_intent=" + std::string(payment);
    r.idempotency_key = *client_id::make(key);
    r.timeout = std::chrono::seconds{30};
    return client.send(r);
}
//! [concept]
} // namespace

int main()
{
    std::printf("snicholls-core concepts/https_client.hpp demo\n");
    memory_https_client stripe;   // in memory: no network
    stripe.route("api.stripe.test", [](const https_request& r) -> capability_result<https_response> {
        return ok(https_response{r.idempotency_key.empty() ? 400 : 200, {{"Request-Id", "req_1"}}, R"({"status":"succeeded"})"});
    });
    const auto answer = refund(stripe, "pi_1", "refund-pi_1-1");
    check(answer.is_ok() && answer->ok(), "the refund is answered 200");
    check(answer.is_ok() && find_header(answer->headers, "request-id") == "req_1", "headers are found ignoring case");

    constexpr auto url = parse_https_url("https://api.stripe.test:443/v1/refunds?limit=1");
    check(url.is_ok() && url->host == "api.stripe.test" && url->target == "/v1/refunds?limit=1", "parse_https_url");
    check(parse_https_url("http://api.stripe.test/").unwrap_err().code == capability_errc::invalid_argument, "only https");
    check(!is_idempotent(http_method::post) && is_idempotent(http_method::del), "POST needs an idempotency key to retry");
    check(!is_sendable({"X-Evil", "a\r\nInjected: yes"}), "a header with CR LF is never sent");
    return failures == 0 ? 0 : 1;
}
