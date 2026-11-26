//
//  sql_dialect_conformance_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/interfaces/sql_dialect_conformance.hpp>: the kit run on both reference dialects. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/interfaces/sql_dialect_conformance.hpp>
#include <snicholls/core/interfaces/sql_dialect_reference.hpp>

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
template <typename D>
struct probe {
    D d;
    const auto& dialect() { return d; }
};
} // namespace

int main()
{
    std::printf("snicholls-core interfaces/sql_dialect_conformance.hpp demo\n");
    //! [kit]
    const capability_conformance_report sqlite = run_sql_dialect_conformance<probe<sqlite_dialect>>();
    const capability_conformance_report postgres = run_sql_dialect_conformance<probe<postgres_dialect>>();
    //! [kit]
    check(sqlite.ok() && postgres.ok(), "both dialects pass");
    check(sqlite.checks == 6, "six checks ran");
    return failures == 0 ? 0 : 1;
}
