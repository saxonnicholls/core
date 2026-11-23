//
//  http_handler_conformance_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/interfaces/http_handler_conformance.hpp>: the kit run over the reference router. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/interfaces/http_handler_conformance.hpp>
#include <snicholls/core/interfaces/http_handler_reference.hpp>

#include <cstdio>
#include <stdexcept>
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

struct probe {
    route_table t;
    probe()
    {
        static_cast<void>(t.add(http_method::get, "/health", [](const http_request&) { return respond(200, "ok"); }));
        static_cast<void>(t.add(http_method::get, "/boom", [](const http_request&) -> http_response {
            throw std::runtime_error(std::string(capability_redaction_marker));
        }));
    }
    auto& handler() { return t; }
    static http_request at(std::string path)
    {
        http_request r;
        r.path = std::move(path);
        return r;
    }
    http_request known() { return at("/health"); }
    http_request unknown() { return at("/nowhere"); }
    http_request throwing() { return at("/boom"); }
};
} // namespace

int main()
{
    std::printf("snicholls-core interfaces/http_handler_conformance.hpp demo\n");
    //! [kit]
    const capability_conformance_report report = run_http_handler_conformance<probe>();
    //! [kit]
    check(report.ok() && report.checks == 5, "the reference passes all five checks");
    return failures == 0 ? 0 : 1;
}
