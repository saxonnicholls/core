//
//  remote_files_conformance_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/interfaces/remote_files_conformance.hpp>: the kit run over the reference through a probe.
//  Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/interfaces/remote_files_conformance.hpp>
#include <snicholls/core/interfaces/remote_files_reference.hpp>

#include <cstddef>
#include <cstdio>
#include <span>
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

// A probe: the implementation, a way to put a file in the drop, and a way to make the server fail.
struct probe {
    memory_remote_files impl;
    auto& files() { return impl; }
    void seed(std::string_view name, std::span<const std::byte> bytes, utc_s modified) { static_cast<void>(impl.put(name, bytes, modified)); }
    void provider_fails(capability_errc code, std::string_view response) { impl.fail_next(code, response); }
};
} // namespace

int main()
{
    std::printf("snicholls-core interfaces/remote_files_conformance.hpp demo\n");
    //! [kit]
    const capability_conformance_report report = run_remote_files_conformance<probe>();
    //! [kit]
    check(report.ok() && report.checks == 7, "the reference passes all seven checks");
    for (const auto& f : report.failures)
        std::printf("    %.*s: %.*s\n", static_cast<int>(f.check.size()), f.check.data(), static_cast<int>(f.expectation.size()),
                    f.expectation.data());
    return failures == 0 ? 0 : 1;
}
