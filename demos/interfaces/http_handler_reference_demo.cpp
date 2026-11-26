//
//  http_handler_reference_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/interfaces/http_handler_reference.hpp>: the reference router: 404, 405 with Allow, HEAD as GET and 500 without detail. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/interfaces/http_handler_reference.hpp>

#include <cstdio>
#include <stdexcept>

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
    std::printf("snicholls-core interfaces/http_handler_reference.hpp demo\n");
    //! [reference]
    route_table t;
    static_cast<void>(t.add(http_method::get, "/orders", [](const http_request&) { return respond(200, "[]", "application/json"); }));
    static_cast<void>(t.add(http_method::post, "/boom", [](const http_request&) -> http_response { throw std::runtime_error("secret detail"); }));
    http_request r;
    r.method = http_method::put;
    r.path = "/orders";
    const http_response not_allowed = t.handle(r);   // 405, Allow: GET
    //! [reference]
    check(not_allowed.status == 405 && find_header(not_allowed.headers, "Allow") == "GET", "405 with Allow");
    r.method = http_method::head;
    const auto head = t.handle(r);
    check(head.status == 200 && head.body.empty(), "HEAD is GET without a body");
    r.method = http_method::post;
    r.path = "/boom";
    const auto boom = t.handle(r);
    check(boom.status == 500 && boom.body == "internal error", "500 without the exception's detail");
    check(t.size() == 2 && t.add(http_method::get, "bad", [](const http_request&) { return respond(200); }).is_err(), "add refuses a bad path");
    route_table orders;
    static_cast<void>(orders.add_pattern(http_method::get, "/orders/:id", [](const http_request& r) {
        return respond(200, "order " + std::string(*path_param(r, "id")));
    }));
    http_request get_order;
    get_order.path = "/orders/42";
    check(orders.handle(get_order).body == "order 42", "add_pattern captures into path_params");
    return failures == 0 ? 0 : 1;
}
