//
//  role_store_conformance_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/interfaces/role_store_conformance.hpp>: the kit run over the reference store. Exits
//  non-zero if a result is not the one documented.
//

#include <snicholls/core/interfaces/role_store_conformance.hpp>
#include <snicholls/core/interfaces/role_store_reference.hpp>

#include <cstdio>
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

struct probe {
    memory_role_store impl;
    auto& store() { return impl; }
    void provider_fails(capability_errc code, std::string_view response) { impl.fail_next(code, response); }
};
} // namespace

int main()
{
    std::printf("snicholls-core interfaces/role_store_conformance.hpp demo\n");
    //! [kit]
    const capability_conformance_report report = run_role_store_conformance<probe>();
    //! [kit]
    for (const auto& f : report.failures)
        std::printf("  %.*s: %.*s\n", static_cast<int>(f.check.size()), f.check.data(), static_cast<int>(f.expectation.size()),
                    f.expectation.data());
    check(report.ok() && report.checks == 6, "the reference passes all six checks");
    return failures == 0 ? 0 : 1;
}
