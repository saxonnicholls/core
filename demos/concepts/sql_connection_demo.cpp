//
//  sql_connection_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/concepts/sql_connection.hpp>: a small portfolio store written once against the concept, its SQL generated through the connection's dialect. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/concepts/sql_connection.hpp>
#include <snicholls/core/interfaces/sql_connection_reference.hpp>

#include <cstdint>
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
const sql_table positions{"position", {{"symbol", sql_type::text, false}, {"qty", sql_type::int64, false}}, {0}, false};

// Generic code: any engine, its dialect from the connection; every failure a result.
template <sql_connection C>
sql_result<std::int64_t> book(C& c, std::string_view symbol, std::int64_t qty)
{
    SNICHOLLS_TRY_VOID(sql_execute(c, *sql_create_table(c.dialect(), positions)));
    SNICHOLLS_TRY_ASSIGN(auto tx, sql_transaction<C>::begin(c));
    SNICHOLLS_TRY_VOID(sql_execute(c, *sql_insert(c.dialect(), positions, sql_conflict::update), {sql_text(symbol), qty}));
    SNICHOLLS_TRY_VOID(tx.commit());
    std::int64_t total = 0;
    SNICHOLLS_TRY_VOID(sql_query(c, *sql_select_rows(c.dialect(), positions), {}, [&total](const sql_row& r) { total += r.get_int64(1).unwrap_or(0); }));
    return ok(total);
}
} // namespace

int main()
{
    std::printf("snicholls-core concepts/sql_connection.hpp demo\n");
    //! [concept]
    memory_sql_connection db;   // SQLite or PostgreSQL in production
    const auto after_bhp = book(db, "BHP", 100);
    const auto after_cba = book(db, "CBA", 50);
    const auto after_bhp_again = book(db, "BHP", 120);   // an upsert
    //! [concept]
    check(*after_bhp == 100 && *after_cba == 150 && *after_bhp_again == 170, "booked through transactions");
    check(db.row_count("position") == 2 && !db.in_transaction(), "two rows, no open transaction");
    return failures == 0 ? 0 : 1;
}
