//
//  http_client_conformance_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/interfaces/http_client_conformance.hpp>: the kit run on the loopback client over an echo server. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/interfaces/http_client_conformance.hpp>
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
route_table echo()
{
    route_table t;
    static_cast<void>(t.add(http_method::get, "/echo", [](const http_request& r) {
        http_response out = respond(200, r.body);
        if (const auto h = find_header(r.headers, "X-Echo"))
            out.headers.push_back({"X-Echo", std::string(*h)});
        return out;
    }));
    return t;
}
struct probe {
    loopback_http_client<route_table> c{echo()};
    auto& client() { return c; }
    void transport_fails(capability_errc code, std::string_view response) { c.fail_next(code, response); }
};
} // namespace

int main()
{
    std::printf("snicholls-core interfaces/http_client_conformance.hpp demo\n");
    //! [kit]
    const capability_conformance_report report = run_http_client_conformance<probe>();
    //! [kit]
    check(report.ok(), "the loopback client passes");
    check(report.checks == 5, "five checks ran");
    return failures == 0 ? 0 : 1;
}
