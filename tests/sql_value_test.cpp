//
//  sql_value_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/sql_value.hpp>: kinds, views and owned values, the row getters' conversions and errors, the
//  retry policy, and the exact mappings of utc, uuid, decimal and money.
//

#include <snicholls/core/sql_value.hpp>

#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <string_view>
#include <type_traits>

using namespace snicholls;

namespace {

static_assert(std::is_trivially_copyable_v<sql_param>);
static_assert(kind_of(sql_param{sql_null}) == sql_kind::null);
static_assert(kind_of(sql_param{true}) == sql_kind::boolean);
static_assert(kind_of(sql_param{std::int64_t{1}}) == sql_kind::int64);
static_assert(kind_of(sql_param{1.5}) == sql_kind::float64);
static_assert(kind_of(sql_text("x")) == sql_kind::text);
static_assert(kind_of(sql_blob({})) == sql_kind::blob);
static_assert(storage_of(sql_type::timestamp) == sql_kind::int64 && storage_of(sql_type::uuid) == sql_kind::blob &&
              storage_of(sql_type::decimal) == sql_kind::text && storage_of(sql_type::boolean) == sql_kind::boolean);
static_assert(may_retry(sql_errc::busy, sql_retry::safe) && !may_retry(sql_errc::busy, sql_retry::never));
static_assert(!may_retry(sql_errc::connection_lost, sql_retry::safe) && may_retry(sql_errc::connection_lost, sql_retry::idempotent));
static_assert(!may_retry(sql_errc::unique_violation, sql_retry::idempotent));
static_assert(sql_fail(sql_errc::busy, "locked", "55P03").error.sqlstate == std::string_view{"55P03"});
static_assert(sql_fail(sql_errc::busy, "locked", "toolong").error.sqlstate.empty());
static_assert(*utc_from_sql<std::chrono::milliseconds>(to_sql(utc_ms::from_unix_millis(-5))) == utc_ms::from_unix_millis(-5));

TEST(SqlValue, OwnedValuesAndViewsRoundTrip)
{
    const std::array<std::byte, 2> b{std::byte{1}, std::byte{2}};
    for (const sql_param p : {sql_param{sql_null}, sql_param{false}, sql_param{std::int64_t{-9}}, sql_param{2.5}, sql_text("t"), sql_blob(b)}) {
        const sql_value v = to_value(p);
        EXPECT_EQ(kind_of(v), kind_of(p));
        const sql_param back = view_of(v);
        EXPECT_EQ(kind_of(back), kind_of(p));
        EXPECT_EQ(to_value(back), v);
    }
}

TEST(SqlValue, RowGettersConvertExactlyOrRefuse)
{
    const std::array<std::string_view, 5> names{"b", "i", "big", "t", "x"};
    const std::array<sql_param, 5> cells{std::int64_t{1}, std::int64_t{42}, std::int64_t{1} << 60, sql_text("hi"), sql_null};
    const sql_row r{names, cells};
    EXPECT_EQ(r.size(), 5u);
    EXPECT_EQ(r.column_name(3), "t");
    EXPECT_EQ(r.column_name(9), "");
    EXPECT_TRUE(*r.get_bool(0));
    EXPECT_EQ(r.get_bool(1).unwrap_err().code, sql_errc::type_mismatch);
    EXPECT_EQ(*r.get_double(1), 42.0);
    EXPECT_EQ(r.get_double(2).unwrap_err().code, sql_errc::out_of_range);
    EXPECT_EQ(*r.get_text(3), "hi");
    EXPECT_EQ(r.get_int64(3).unwrap_err().code, sql_errc::type_mismatch);
    EXPECT_EQ(r.get_blob(3).unwrap_err().code, sql_errc::type_mismatch);
    EXPECT_EQ(r.get_text(1).unwrap_err().code, sql_errc::type_mismatch);
    EXPECT_TRUE(r.is_null(4) && r.is_null(5));
    EXPECT_EQ(r.get_int64(5).unwrap_err().code, sql_errc::misuse);
    const std::array<sql_param, 1> flag{true};
    EXPECT_EQ(*sql_row({}, flag).get_int64(0), 1);
    EXPECT_EQ(sql_row({}, flag).get_double(0).unwrap_err().code, sql_errc::type_mismatch);
}

TEST(SqlValue, CoreTypesMapExactly)
{
    const utc_ns t = utc_ns::from_unix_nanos(1'800'000'000'123'456'789);
    EXPECT_EQ(std::get<std::int64_t>(to_sql(t)), 1'800'000'000'123'456'789);
    EXPECT_EQ(utc_from_sql<std::chrono::nanoseconds>(sql_text("x")).unwrap_err().code, sql_errc::type_mismatch);

    const uuid u = *parse_uuid("0190a5c4-7e1b-7c3a-9f12-3b4c5d6e7f80");
    EXPECT_EQ(*uuid_from_sql(to_sql(u)), u);
    const std::array<std::byte, 3> short_blob{};
    EXPECT_EQ(uuid_from_sql(sql_blob(short_blob)).unwrap_err().code, sql_errc::out_of_range);
    EXPECT_EQ(uuid_from_sql(std::int64_t{1}).unwrap_err().code, sql_errc::type_mismatch);

    const decimal d = *decimal::parse("-0.0001");
    const decimal_text dt = d.text();
    EXPECT_EQ(*decimal_from_sql(to_sql(dt)), d);
    EXPECT_EQ(decimal_from_sql(sql_text("1e5x")).unwrap_err().code, sql_errc::type_mismatch);
    EXPECT_EQ(decimal_from_sql(1.0).unwrap_err().code, sql_errc::type_mismatch);
}

TEST(SqlValue, MoneyIsItsTextAtTheAssetsScale)
{
    constexpr asset aud = checked_asset("AUD", 2);
    const money m = *parse_money(aud, "-1000.5");
    const sql_money_text text = money_text(m);
    EXPECT_EQ(text.view(), "-1000.50");
    EXPECT_EQ(*money_from_sql(aud, to_sql(text)), m);
    EXPECT_EQ(money_from_sql(aud, sql_text("1.005")).unwrap_err().code, sql_errc::out_of_range);
    EXPECT_EQ(money_from_sql(aud, sql_text("abc")).unwrap_err().code, sql_errc::type_mismatch);
    EXPECT_EQ(money_from_sql(aud, std::int64_t{1}).unwrap_err().code, sql_errc::type_mismatch);
}

} // namespace
