//
//  http_handler_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/interfaces/http_handler.hpp>: routes chosen at run time behind the virtual interface. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/interfaces/http_handler.hpp>
#include <snicholls/core/interfaces/http_handler_reference.hpp>

#include <cstdio>
#include <memory>

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

int main()
{
    std::printf("snicholls-core interfaces/http_handler.hpp demo\n");
    route_table routes;
    static_cast<void>(routes.add(http_method::get, "/health", [](const http_request&) { return respond(200, "ok"); }));
    //! [virtual]
    std::unique_ptr<http_handler_interface> app = std::make_unique<http_handler_adapter<route_table>>(std::move(routes));
    http_handler_ref handle{*app};
    http_request r;
    r.path = "/health";
    const http_response answer = handle.handle(r);
    //! [virtual]
    check(answer.status == 200, "answered through the interface");
    r.path = "/nowhere";
    check(app->handle(r).status == 404, "an unknown path is 404");
    return failures == 0 ? 0 : 1;
}
