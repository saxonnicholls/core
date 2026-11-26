//
//  https_client_reference_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/interfaces/https_client_reference.hpp>: the in-memory router, a product's http_handler
//  served behind it, a day recorded to a cassette, saved, loaded and replayed with no server at all. Exits non-zero if
//  a result is not the one documented.
//

#include <snicholls/core/interfaces/http_handler_reference.hpp>
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
} // namespace

int main()
{
    std::printf("snicholls-core interfaces/https_client_reference.hpp demo\n");
    //! [reference]
    route_table routes;   // a product's own routes, as an http_handler
    static_cast<void>(routes.add(http_method::get, "/health", [](const http_request&) { return respond(200, "ok"); }));
    memory_https_client network;
    network.serve("api.example.test", routes);
    network.route("down.example.test", [](const https_request&) -> capability_result<https_response> {
        return fail(capability_errc::timeout);   // a scripted outage
    });
    recording_https_client<memory_https_client&> recorder{network};
    https_request health;
    health.url = "https://api.example.test/health";
    health.authorization = secret<std::string>{"Bearer never-recorded"};
    const auto live = recorder.send(health);
    const auto text = recorder.tape().save();   // the recorded day: no credential in it
    replay_https_client replay{*cassette::load(text)};
    const auto replayed = replay.send(health);
    //! [reference]
    std::printf("  cassette:\n%s", text.c_str());
    check(live.is_ok() && live->body == "ok", "the route answered through the router");
    check(replayed.is_ok() && *replayed == *live, "the replay answers the same with no server");
    check(text.find("never-recorded") == std::string::npos && text.find("> a\n") != std::string::npos,
          "the cassette notes a credential was sent, never its value");
    https_request down;
    down.url = "https://down.example.test/";
    check(network.send(down).unwrap_err().code == capability_errc::timeout, "a handler can script a transport failure");
    down.url = "https://nowhere.example.test/";
    check(network.send(down).unwrap_err().code == capability_errc::unavailable, "a host with no route is unavailable");
    check(network.transport_failures() == 2 && network.seen().exchanges.size() == 1, "what was sent is kept, redacted");
    network.fail_next(capability_errc::unavailable, "tls_refused");
    check(network.send(health).unwrap_err().provider_code.view() == "tls_refused", "an armed failure keeps only its code");
    check(replay.unused() == 0 && replay.send(health).is_err(), "each recorded exchange answers once");
    const recorded_exchange& x = recorder.tape().exchanges.front();
    check(x.idempotency_key().empty() && cassette::record(health, *live) == x, "cassette::record is what was kept");
    return failures == 0 ? 0 : 1;
}
