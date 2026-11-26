//
//  sql_value_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/sql_value.hpp>: parameters as views, a row read with typed getters, core's types mapped exactly, and the retry policy. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/sql_value.hpp>

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
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

int main()
{
    std::printf("snicholls-core sql_value.hpp demo\n");
    //! [values]
    const utc_ms at = utc_ms::from_unix_millis(1'800'000'000'000);
    const uuid id = *parse_uuid("0190a5c4-7e1b-7c3a-9f12-3b4c5d6e7f80");
    const sql_money_text price = money_text(*parse_money(checked_asset("AUD", 2), "12.5"));
    const std::array<sql_param, 4> params{to_sql(id), to_sql(at), to_sql(price), sql_null};   // views: no copies
    const std::array<std::string_view, 4> names{"id", "at", "price", "note"};
    const sql_row row{names, params};                                                        // as a visitor sees it
    const auto back = utc_from_sql<std::chrono::milliseconds>(*row.cell(1));
    //! [values]
    check(*uuid_from_sql(*row.cell(0)) == id, "a uuid is 16 bytes and back");
    check(back && *back == at, "a utc is its ticks and back");
    check(*row.get_text(2) == "12.50", "money is its exact text at the asset's scale");
    check(row.is_null(3) && row.get_int64(3).is_err(), "NULL is NULL, never 0");
    check(kind_of(to_value(params[0])) == sql_kind::blob, "to_value owns a copy");
    check(may_retry(sql_errc::serialization_failure, sql_retry::safe), "a serialization failure may be retried");
    check(*decimal_from_sql(sql_text("1.25")) == *decimal::parse("1.25"), "a decimal from text");
    return failures == 0 ? 0 : 1;
}
