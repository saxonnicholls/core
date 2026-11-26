//
//  sql_dialect_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/interfaces/sql_dialect.hpp>: a dialect chosen at run time, and the generators through it. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/interfaces/sql_dialect.hpp>
#include <snicholls/core/interfaces/sql_dialect_reference.hpp>

#include <cstdio>
#include <memory>
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
} // namespace

int main(int argc, char**)
{
    std::printf("snicholls-core interfaces/sql_dialect.hpp demo\n");
    //! [virtual]
    const bool use_postgres = argc < 2;   // configuration, at run time
    std::unique_ptr<sql_dialect_interface> d;
    if (use_postgres)
        d = std::make_unique<sql_dialect_adapter<postgres_dialect>>(postgres_dialect{});
    else
        d = std::make_unique<sql_dialect_adapter<sqlite_dialect>>(sqlite_dialect{});
    const sql_dialect_ref handle{*d};   // copyable, models the concept
    const auto insert = sql_insert(handle, sql_table{"t", {{"k", sql_type::uuid, false}}, {0}, false});
    //! [virtual]
    check(d->name() == "postgres" && *insert == "INSERT INTO \"t\" (\"k\") VALUES ($1)", "PostgreSQL's placeholders through the interface");
    check(*handle.column_type(sql_type::uuid) == "BYTEA", "a uuid is BYTEA");
    return failures == 0 ? 0 : 1;
}
