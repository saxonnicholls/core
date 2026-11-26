//
//  sql_connection_reference_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/interfaces/sql_connection_reference.hpp>: the in-memory store: a batch, a range query, a rollback and a lost connection. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/interfaces/sql_connection_reference.hpp>

#include <cstdint>
#include <cstdio>
#include <span>
#include <vector>

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
    std::printf("snicholls-core interfaces/sql_connection_reference.hpp demo\n");
    //! [reference]
    memory_sql_connection db;
    const sql_table ticks{"tick", {{"ts", sql_type::int64, false}, {"px", sql_type::float64, false}}, {0}, true};
    static_cast<void>(sql_execute(db, *sql_create_table(db.dialect(), ticks)));
    const auto n = db.execute_batch(*sql_insert(db.dialect(), ticks), 2, 100, sql_row_filler([](std::size_t row, std::span<sql_param> p) {
        p[0] = static_cast<std::int64_t>(row);
        p[1] = 100.0 + static_cast<double>(row) / 10;
    }));
    sql_select window;
    window.range_column = 0;
    window.has_from = true;
    window.has_to = true;
    window.order_by = {0};
    std::vector<std::int64_t> seen;
    static_cast<void>(sql_query(db, *sql_select_rows(db.dialect(), ticks, window), {std::int64_t{10}, std::int64_t{13}},
                                [&seen](const sql_row& r) { seen.push_back(*r.get_int64(0)); }));
    //! [reference]
    check(n.is_ok() && *n == 100 && db.row_count("tick") == 100, "a batch of 100");
    check(seen == std::vector<std::int64_t>{10, 11, 12}, "10 <= ts < 13");
    static_cast<void>(db.begin());
    static_cast<void>(sql_execute(db, *sql_delete(db.dialect(), ticks)));
    static_cast<void>(db.rollback());
    check(db.row_count("tick") == 100, "a rollback restores");
    check(db.prepared() > 0, "statements are cached by text");
    db.break_connection();
    check(db.ping().unwrap_err().code == sql_errc::connection_lost && !db.is_open(), "a lost connection");
    return failures == 0 ? 0 : 1;
}
