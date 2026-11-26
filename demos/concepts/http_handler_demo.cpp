//
//  http_handler_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/concepts/http_handler.hpp>: a handler as a plain type, method tokens and respond. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/concepts/http_handler.hpp>

#include <cstdio>
#include <vector>

using namespace snicholls;

namespace {
int failures = 0;
void check(bool ok, const char* what)
{
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (!ok)
        ++failures;
}

// The smallest handler: a type with handle(), no base class.
struct health {
    http_response handle(const http_request& r) { return r.path == "/health" ? respond(200, "ok") : respond(404, "not found"); }
};
static_assert(http_handler<health>);
} // namespace

int main()
{
    std::printf("snicholls-core concepts/http_handler.hpp demo\n");
    //! [concept]
    health h;
    http_request r;
    r.path = "/health";
    const http_response answer = h.handle(r);
    //! [concept]
    check(answer.status == 200 && answer.body == "ok", "a 200 with a body");
    check(find_header(answer.headers, "content-type") == "text/plain; charset=utf-8", "respond sets the content type");
    check(to_string(http_method::del) == "DELETE" && parse_http_method("HEAD") == http_method::head, "method tokens");
    std::vector<http_header> captured;
    check(match_route("/orders/:id", "/orders/42", captured) && captured.at(0).value == "42", "match_route captures :id");
    http_request search;
    search.path = "/search";
    search.query = "q=bhp";
    search.path_params = captured;
    check(request_target(search) == "/search?q=bhp" && path_param(search, "id") == std::optional<std::string_view>{"42"},
          "request_target and path_param");
    check(is_success(respond(201)) && !is_success(respond(500)), "is_success");
    return failures == 0 ? 0 : 1;
}
