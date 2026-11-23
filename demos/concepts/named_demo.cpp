//
//  named_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/concepts/named.hpp>.
//

// the named concept
#include <cstdio>
#include <snicholls/core/concepts/named.hpp>
#include <snicholls/core/money/iso4217.hpp>
#include <string_view>

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

    //! [named]
    struct ledger_book {
        static constexpr std::string_view name() noexcept { return "General"; }
    };
    static_assert(has_name<ledger_book> && has_name<currency::AUD>);
    //! [named]
    std::puts("named");
    check(ledger_book{}.name() == "General", "a type with a static name");
    check(!has_name<int>, "an int is not named");
    std::printf("%s\n", failures == 0 ? "all checks passed" : "FAILED");
    return failures == 0 ? 0 : 1;
}
