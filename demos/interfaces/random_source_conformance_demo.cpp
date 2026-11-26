//
//  random_source_conformance_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  The random_source conformance kit, as an implementation (ts-moveables' operating-system source) would run it.
//

#include <snicholls/core/interfaces/random_source_conformance.hpp>
#include <snicholls/core/interfaces/random_source_reference.hpp>

#include <string>
#include <vector>
#include <cstdio>

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
    std::puts("random_source conformance");
    //! [kit]
    test_random_source source{99};
    const std::vector<std::string> problems = check_random_source(source);   // empty: it conforms
    for (const std::string& p : problems)
        std::printf("    %s\n", p.c_str());
    unavailable_random_source down;
    const auto closed = check_random_source_fails_closed(down);              // empty: it fails closed
    //! [kit]
    check(problems.empty(), "the test source conforms");
    check(closed.empty(), "the unavailable source fails closed");
    std::printf("%s\n", failures == 0 ? "all checks passed" : "FAILED");
    return failures == 0 ? 0 : 1;
}
