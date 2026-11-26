//
//  http_client_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/interfaces/http_client.hpp>: a client chosen at run time through the virtual interface, the adapter and the handle. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/interfaces/http_client.hpp>
#include <snicholls/core/interfaces/http_client_reference.hpp>
#include <snicholls/core/interfaces/http_handler_reference.hpp>

#include <cstdio>
#include <memory>
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
    std::printf("snicholls-core interfaces/http_client.hpp demo\n");
    //! [virtual]
    std::unique_ptr<http_client_interface> chosen = std::make_unique<http_client_adapter<loopback_http_client<route_table>>>(
        loopback_http_client{price_api()});
    http_client_ref api{*chosen};   // copyable, models the concept
    const auto r = http_get(api, "/prices/CBA");
    //! [virtual]
    check(r && r->body == "CBA=41.20", "sent through the handle");
    auto& adapter = static_cast<http_client_adapter<loopback_http_client<route_table>>&>(*chosen);
    check(adapter.get().sent() == 1, "the adapter reaches the model");
    return failures == 0 ? 0 : 1;
}
