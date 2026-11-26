//
//  https_client_conformance_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/interfaces/https_client_conformance.hpp>: the kit run over the reference client and a
//  scripted server. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/interfaces/https_client_conformance.hpp>
#include <snicholls/core/interfaces/https_client_reference.hpp>

#include <cstdio>
#include <memory>
#include <string>
#include <string_view>

using namespace snicholls;

namespace {
int failures = 0;
void check(bool ok, const char* what)
{
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (!ok)
        ++failures;
}

struct server {
    https_response next{200, {}, ""};
    https_observed_request last;
};

struct probe {
    std::shared_ptr<server> s = std::make_shared<server>();
    memory_https_client impl;
    probe()
    {
        impl.route("api.example.test", [s = s](const https_request& r) -> capability_result<https_response> {
            s->last = {std::string(to_string(r.method)), std::string(parse_https_url(r.url)->target), r.headers, r.body};
            if (!r.authorization.reveal().empty())
                s->last.headers.push_back({"Authorization", r.authorization.reveal()});
            if (!r.idempotency_key.empty())
                s->last.headers.push_back({"Idempotency-Key", std::string(r.idempotency_key.view())});
            return ok(s->next);
        });
    }
    auto& client() { return impl; }
    static std::string url(std::string_view path) { return "https://api.example.test" + std::string(path); }
    static std::string unreachable_url() { return "https://unreachable.example.test/"; }
    void answer(https_response r) { s->next = std::move(r); }
    https_observed_request seen() { return s->last; }
    void provider_fails(capability_errc code, std::string_view response) { impl.fail_next(code, response); }
};
} // namespace

int main()
{
    std::printf("snicholls-core interfaces/https_client_conformance.hpp demo\n");
    //! [kit]
    const capability_conformance_report report = run_https_client_conformance<probe>();
    //! [kit]
    for (const auto& f : report.failures)
        std::printf("  %.*s: %.*s\n", static_cast<int>(f.check.size()), f.check.data(), static_cast<int>(f.expectation.size()),
                    f.expectation.data());
    check(report.ok() && report.checks == 7, "the reference passes all seven checks");
    return failures == 0 ? 0 : 1;
}
