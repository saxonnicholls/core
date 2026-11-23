//
//  message_sender_conformance_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/interfaces/message_sender_conformance.hpp>: the kit run over the reference through a probe. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/interfaces/message_sender_conformance.hpp>
#include <snicholls/core/interfaces/message_sender_reference.hpp>

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

// A probe: the implementation, and a way to make the provider fail.
struct probe {
    memory_message_sender impl;
    auto& sender() { return impl; }
    void provider_fails(capability_errc code, std::string_view response) { impl.fail_next(code, response); }
};
} // namespace

int main()
{
    std::printf("snicholls-core interfaces/message_sender_conformance.hpp demo\n");
    //! [kit]
    const capability_conformance_report report = run_message_sender_conformance<probe>();
    //! [kit]
    check(report.ok() && report.checks == 5, "the reference passes all five checks");
    return failures == 0 ? 0 : 1;
}
