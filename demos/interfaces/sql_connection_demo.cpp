//
//  sql_connection_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/interfaces/sql_connection.hpp>: a connection chosen at run time through the virtual interface, the adapter and the handle. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/interfaces/sql_connection.hpp>
#include <snicholls/core/interfaces/sql_connection_reference.hpp>

#include <cstdint>
#include <cstdio>
#include <memory>

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
    std::printf("snicholls-core interfaces/sql_connection.hpp demo\n");
    //! [virtual]
    std::unique_ptr<sql_connection_interface> db = std::make_unique<sql_connection_adapter<memory_sql_connection>>(memory_sql_connection{});
    sql_connection_ref c{*db};   // copyable, models the concept
    const sql_table t{"t", {{"k", sql_type::int64, false}}, {0}, false};
    const auto created = sql_execute(c, *sql_create_table(c.dialect(), t));
    const auto inserted = sql_execute(c, *sql_insert(c.dialect(), t), {std::int64_t{1}});
    //! [virtual]
    check(created.is_ok() && inserted.is_ok() && *inserted == 1, "statements through the handle");
    check(db->dialect().name() == "sqlite", "the model's dialect behind the dialect interface");
    check(db->ping().is_ok() && db->is_open(), "ping and is_open through the interface");
    return failures == 0 ? 0 : 1;
}
