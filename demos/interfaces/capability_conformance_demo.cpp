//
//  capability_conformance_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/interfaces/capability_conformance.hpp>: the kits' report and the redaction check. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/interfaces/capability_conformance.hpp>

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
    std::printf("snicholls-core interfaces/capability_conformance.hpp demo\n");
    //! [report]
    capability_conformance_report report;
    report.checks = 2;
    report.failures.push_back({"a provider failure never leaks its response", "neither provider_code nor message carries it"});
    const capability_error leaky{capability_errc::internal, "REDACTION-MARKER-c0re", {}};
    const bool leaked = capability_error_leaks(leaky, capability_redaction_marker);
    //! [report]
    check(!report.ok() && report.failed("a provider failure never leaks its response"), "the report names the failed check");
    check(leaked, "the marker is found in a provider code");
    check(!capability_error_leaks(provider_error(capability_errc::internal, capability_provider_response), capability_redaction_marker),
          "provider_error keeps nothing of the kits' response");
    for (const auto& f : report.failures)
        std::printf("  %.*s: %.*s\n", static_cast<int>(f.check.size()), f.check.data(), static_cast<int>(f.expectation.size()), f.expectation.data());
    return failures == 0 ? 0 : 1;
}
