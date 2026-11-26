//
//  http_client_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/concepts/http_client.hpp>: an API call written once against the concept, tested against a fake server with no network. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/concepts/http_client.hpp>
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

namespace {
// Generic code: any client.
template <http_client C>
capability_result<std::string> price(C& api, std::string_view symbol)
{
    SNICHOLLS_TRY_ASSIGN(const http_response r, http_get(api, "/prices/" + std::string(symbol)));
    if (!is_success(r))
        return fail(capability_errc::not_found);
    return ok(r.body);
}
} // namespace

int main()
{
    std::printf("snicholls-core concepts/http_client.hpp demo\n");
    //! [concept]
    loopback_http_client api{price_api()};
    const auto bhp = price(api, "BHP");
    const auto missing = http_get(api, "/nowhere");   // a 404 is a response
    //! [concept]
    check(bhp && *bhp == "BHP=41.20", "a price through the client");
    check(missing && missing->status == 404, "an error status is a response");
    http_request bad;
    check(validate_request(bad).is_err(), "validate_request refuses a request without a path");
    return failures == 0 ? 0 : 1;
}
