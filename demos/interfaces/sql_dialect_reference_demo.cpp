//
//  sql_dialect_reference_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/interfaces/sql_dialect_reference.hpp>: SQLite's and PostgreSQL's names, side by side, at compile time. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/interfaces/sql_dialect_reference.hpp>

#include <cstdio>
#include <string>

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
    std::printf("snicholls-core interfaces/sql_dialect_reference.hpp demo\n");
    //! [reference]
    constexpr sqlite_dialect sqlite;
    constexpr postgres_dialect postgres;
    static_assert(*sqlite.column_type(sql_type::timestamp) == "INTEGER" && *postgres.column_type(sql_type::timestamp) == "BIGINT");
    std::string a, b;
    sqlite.append_placeholder(a, 3);
    postgres.append_placeholder(b, 3);
    //! [reference]
    for (const sql_type t : sql_types)
        std::printf("    %-8.*s %.*s\n", static_cast<int>(sqlite.column_type(t)->size()), sqlite.column_type(t)->data(),
                    static_cast<int>(postgres.column_type(t)->size()), postgres.column_type(t)->data());
    check(a == "?3" && b == "$3", "placeholders");
    check(sqlite.begin_transaction() == "BEGIN IMMEDIATE" && postgres.begin_transaction() == "BEGIN", "transactions");
    std::string id;
    check(sqlite.append_identifier(id, "Price") && id == "\"Price\"" && postgres.clustered_suffix().empty(), "quoting and clustering");
    check(sqlite.name() == "sqlite" && postgres.name() == "postgres", "names");
    return failures == 0 ? 0 : 1;
}
