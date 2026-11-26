//
//  sql_connection_conformance_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/interfaces/sql_connection_conformance.hpp>: the kit run on the in-memory reference. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/interfaces/sql_connection_conformance.hpp>
#include <snicholls/core/interfaces/sql_connection_reference.hpp>

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

namespace {
struct probe {
    memory_sql_connection c;
    auto& connection() { return c; }
    void connection_breaks() { c.break_connection(); }
};
} // namespace

int main()
{
    std::printf("snicholls-core interfaces/sql_connection_conformance.hpp demo\n");
    //! [kit]
    const capability_conformance_report report = run_sql_connection_conformance<probe>();
    //! [kit]
    check(report.ok(), "the reference passes");
    check(report.checks == 9, "nine checks ran");
    return failures == 0 ? 0 : 1;
}
