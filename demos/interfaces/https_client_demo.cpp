//
//  https_client_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/interfaces/https_client.hpp>: a transport chosen at run time, behind the virtual
//  interface. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/interfaces/https_client.hpp>
#include <snicholls/core/interfaces/https_client_reference.hpp>

#include <cstdio>

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
    std::printf("snicholls-core interfaces/https_client.hpp demo\n");
    memory_https_client reference;
    reference.route("api.example.test", [](const https_request&) -> capability_result<https_response> {
        return ok(https_response{200, {}, "ok"});
    });
    //! [virtual]
    https_client_adapter<memory_https_client&> adapter{reference};   // borrows
    https_client_interface& chosen = adapter;
    https_client_ref client{chosen};
    https_request r;
    r.url = "https://api.example.test/health";
    const auto answer = client.send(r);
    //! [virtual]
    check(answer.is_ok() && answer->body == "ok", "sent through the interface and the handle");
    check(&adapter.get() == &reference, "the adapter borrows");
    https_client_adapter owned{memory_https_client{}};
    check(owned.send(r).unwrap_err().code == capability_errc::unavailable, "an owning adapter forwards: no route is unavailable");
    return failures == 0 ? 0 : 1;
}
