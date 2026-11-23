//
//  named_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/interfaces/named.hpp>.
//

// named behind a virtual interface
#include <cstdio>
#include <snicholls/core/interfaces/named.hpp>
#include <snicholls/core/money/iso4217.hpp>
#include <memory>
#include <vector>

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

    //! [interface]
    std::vector<std::unique_ptr<named_interface>> list;
    list.push_back(std::make_unique<named_model<currency::AUD>>());
    list.push_back(std::make_unique<named_model<currency::CHF>>());
    //! [interface]
    std::puts("named_interface");
    check(list[0]->name() == "Australian Dollar", "AUD's name, at run time");
    check(list[1]->name() == "Swiss Franc", "CHF's name");
    std::printf("%s\n", failures == 0 ? "all checks passed" : "FAILED");
    return failures == 0 ? 0 : 1;
}
