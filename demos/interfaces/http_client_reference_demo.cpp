//
//  http_client_reference_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/interfaces/http_client_reference.hpp>: a loopback client over a fake API, with a transport failure on command. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/interfaces/http_client_reference.hpp>
#include <snicholls/core/interfaces/http_handler_reference.hpp>

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
} // namespace

namespace {
// A fake of a price API, as an http_handler.
route_table price_api()
{
    route_table t;
    static_cast<void>(t.add_pattern(http_method::get, "/prices/:symbol", [](const http_request& r) {
        return respond(200, std::string(*path_param(r, "symbol")) + "=41.20", "text/plain");
    }));
    return t;
}
} // namespace

int main()
{
    std::printf("snicholls-core interfaces/http_client_reference.hpp demo\n");
    //! [reference]
    loopback_http_client api{price_api()};
    api.fail_next(capability_errc::timeout, "upstream_timeout <html>...</html>");
    const auto first = http_get(api, "/prices/BHP");    // the transport fails once
    const auto second = http_get(api, "/prices/BHP");   // then the fake answers
    //! [reference]
    check(first.is_err() && first.unwrap_err().code == capability_errc::timeout && first.unwrap_err().retryable(), "a retryable timeout");
    check(first.unwrap_err().provider_code == std::string_view{"upstream_timeout"}, "only the code of what was received is kept");
    check(second && second->status == 200 && api.sent() == 1, "the retry succeeds");
    check(api.handler().size() == 1, "the handler is reachable");
    return failures == 0 ? 0 : 1;
}
