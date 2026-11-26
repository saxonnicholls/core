//
//  sql_dialect_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/concepts/sql_dialect.hpp>: a time-series table described once and generated for both engines. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/concepts/sql_dialect.hpp>
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

namespace {
// Generic code: the statements a store needs, for any dialect.
template <sql_dialect D>
std::string ddl(const D& d, const sql_table& t)
{
    return sql_create_table(d, t).unwrap_or("");
}
} // namespace

int main()
{
    std::printf("snicholls-core concepts/sql_dialect.hpp demo\n");
    //! [concept]
    const sql_table quotes{"quote", {{"series", sql_type::text, false}, {"ts", sql_type::timestamp, false}, {"px", sql_type::decimal, false}}, {0, 1}, true};
    const std::string sqlite = ddl(sqlite_dialect{}, quotes);
    const std::string postgres = ddl(postgres_dialect{}, quotes);
    sql_select latest;
    latest.equal = {0};
    latest.order_by = {1};
    latest.descending = true;
    latest.limit = 1;
    const auto select = sql_select_rows(postgres_dialect{}, quotes, latest);
    //! [concept]
    std::printf("    %s\n    %s\n    %s\n", sqlite.c_str(), postgres.c_str(), select->c_str());
    check(sqlite.ends_with("WITHOUT ROWID") && sqlite.find("\"ts\" INTEGER") != std::string::npos, "SQLite's types and clustering");
    check(postgres.find("\"ts\" BIGINT") != std::string::npos, "PostgreSQL's types");
    check(*select == "SELECT \"series\", \"ts\", \"px\" FROM \"quote\" WHERE \"series\" = $1 ORDER BY \"ts\" DESC LIMIT 1", "the latest quote of a series");
    check(sql_insert(sqlite_dialect{}, quotes, sql_conflict::update)->find("excluded.\"px\"") != std::string::npos, "an upsert");
    check(sql_count(sqlite_dialect{}, quotes).is_ok() && sql_delete(sqlite_dialect{}, quotes).is_ok() && sql_drop_table(sqlite_dialect{}, "quote").is_ok(), "count, delete, drop");
    std::string id;
    check(append_quoted_identifier(id, "x\"y") && id == "\"x\"\"y\"", "append_quoted_identifier");
    return failures == 0 ? 0 : 1;
}
