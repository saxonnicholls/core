//
//  named_conformance_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/interfaces/named_conformance.hpp>.
//

// the named conformance kit
#include <cstdio>
#include <snicholls/core/interfaces/named.hpp>
#include <snicholls/core/interfaces/named_conformance.hpp>
#include <snicholls/core/money/iso4217.hpp>

namespace {
int failures = 0;
void check(bool ok, const char* what)
{
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++failures;
}
} // namespace

int main()
{
    using namespace snicholls;

    //! [conformance]
    const auto type_form = check_named(currency::AUD{});
    const named_model<currency::AUD> model;
    const auto virtual_form = check_named(static_cast<const named_interface&>(model));
    //! [conformance]
    std::puts("named conformance");
    check(!type_form && !virtual_form, "both forms conform");
    std::printf("%s\n", failures == 0 ? "all checks passed" : "FAILED");
    return failures == 0 ? 0 : 1;
}
