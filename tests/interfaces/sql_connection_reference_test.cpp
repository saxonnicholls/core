//
//  sql_connection_reference_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/interfaces/sql_connection_reference.hpp>: the statements it understands (both placeholder
//  forms, quoted names, BEGIN as text), its typing, the statement cache and a broken connection.
//

#include <snicholls/core/interfaces/sql_connection_reference.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <span>
#include <string>
#include <vector>

using namespace snicholls;

namespace {

TEST(MemorySqlConnection, RunsPostgresPlaceholdersAndQuotedNames)
{
    memory_sql_connection c;
    const sql_table t{"odd \"name\"", {{"Key", sql_type::int64, false}, {"v", sql_type::float64, true}}, {0}, true};
    ASSERT_TRUE(sql_execute(c, *sql_create_table(postgres_dialect{}, t)).is_ok());
    ASSERT_TRUE(sql_execute(c, *sql_insert(postgres_dialect{}, t), {std::int64_t{7}, std::int64_t{2}}).is_ok());   // an integer into a double
    double v = 0;
    static_cast<void>(sql_query(c, *sql_select_rows(postgres_dialect{}, t), {}, [&v](const sql_row& r) { v = *r.get_double(1); }));
    EXPECT_EQ(v, 2.0);
    EXPECT_EQ(c.row_count("odd \"name\""), 1u);
}

TEST(MemorySqlConnection, TransactionsAsText)
{
    memory_sql_connection c;
    EXPECT_TRUE(sql_execute(c, "BEGIN IMMEDIATE").is_ok());
    EXPECT_TRUE(c.in_transaction());
    EXPECT_TRUE(sql_execute(c, "ROLLBACK;").is_ok());
    EXPECT_TRUE(sql_execute(c, "BEGIN").is_ok());
    EXPECT_TRUE(sql_execute(c, "COMMIT").is_ok());
    EXPECT_EQ(sql_execute(c, "COMMIT").unwrap_err().code, sql_errc::misuse);
}

TEST(MemorySqlConnection, CachesStatementsByText)
{
    memory_sql_connection c;
    EXPECT_TRUE(c.prepare("BEGIN").is_ok());
    EXPECT_TRUE(c.prepare("BEGIN").is_ok());
    EXPECT_EQ(c.prepared(), 1u);
    const auto bad = c.prepare("UPDATE t SET x = 1");
    EXPECT_EQ(bad.unwrap_err().code, sql_errc::syntax_error);
    EXPECT_EQ(bad.unwrap_err().sqlstate, std::string_view{"42601"});
    EXPECT_EQ(c.prepare("SELECT \"unterminated").unwrap_err().code, sql_errc::syntax_error);
}

TEST(MemorySqlConnection, CreateAndDropAreCheckedUnlessToldNot)
{
    memory_sql_connection c;
    const sql_table t{"t", {{"k", sql_type::int64, false}}, {0}, false};
    ASSERT_TRUE(sql_execute(c, *sql_create_table(c.dialect(), t)).is_ok());
    EXPECT_TRUE(sql_execute(c, *sql_create_table(c.dialect(), t)).is_ok());   // IF NOT EXISTS
    EXPECT_EQ(sql_execute(c, "CREATE TABLE \"t\" (\"k\" INTEGER)").unwrap_err().code, sql_errc::misuse);
    EXPECT_TRUE(sql_execute(c, *sql_drop_table(c.dialect(), "t", false)).is_ok());
    EXPECT_EQ(sql_execute(c, *sql_drop_table(c.dialect(), "t", false)).unwrap_err().code, sql_errc::undefined_object);
    EXPECT_TRUE(sql_execute(c, *sql_drop_table(c.dialect(), "t")).is_ok());
}

TEST(MemorySqlConnection, TypesAreCheckedAsPostgresChecksThem)
{
    memory_sql_connection c;
    const sql_table t{"t", {{"k", sql_type::int64, false}, {"b", sql_type::boolean, true}, {"raw", sql_type::blob, true}}, {0}, false};
    ASSERT_TRUE(sql_execute(c, *sql_create_table(postgres_dialect{}, t)).is_ok());   // BOOLEAN, BIGINT, BYTEA
    const std::string ins = *sql_insert(c.dialect(), t);
    EXPECT_TRUE(sql_execute(c, ins, {std::int64_t{1}, std::int64_t{1}, sql_null}).is_ok());   // 1 into a boolean
    EXPECT_EQ(sql_execute(c, ins, {std::int64_t{2}, std::int64_t{2}, sql_null}).unwrap_err().code, sql_errc::type_mismatch);
    EXPECT_EQ(sql_execute(c, ins, {std::int64_t{3}, true, 1.5}).unwrap_err().code, sql_errc::type_mismatch);
    bool b = false;
    static_cast<void>(sql_query(c, *sql_select_rows(c.dialect(), t), {}, [&b](const sql_row& r) { b = *r.get_bool(1); }));
    EXPECT_TRUE(b);
}

TEST(MemorySqlConnection, ABrokenConnectionIsLostEverywhere)
{
    memory_sql_connection c;
    c.break_connection();
    EXPECT_FALSE(c.is_open());
    for (const auto code : {c.ping().unwrap_err().code, c.begin().unwrap_err().code, c.commit().unwrap_err().code,
                            c.rollback().unwrap_err().code, sql_execute(c, "BEGIN").unwrap_err().code,
                            c.execute_batch("BEGIN", 0, 1, sql_row_filler([](std::size_t, std::span<sql_param>) {})).unwrap_err().code})
        EXPECT_EQ(code, sql_errc::connection_lost);
}

TEST(MemorySqlConnection, AnUnknownColumnIsUndefined)
{
    memory_sql_connection c;
    const sql_table t{"t", {{"k", sql_type::int64, false}}, {0}, false};
    ASSERT_TRUE(sql_execute(c, *sql_create_table(c.dialect(), t)).is_ok());
    EXPECT_EQ(sql_execute(c, "INSERT INTO \"t\" (\"nope\") VALUES (?1)", {std::int64_t{1}}).unwrap_err().code, sql_errc::undefined_object);
    EXPECT_EQ(sql_execute(c, "SELECT \"nope\" FROM \"t\"").unwrap_err().code, sql_errc::undefined_object);
    EXPECT_EQ(sql_execute(c, "DELETE FROM \"t\" WHERE \"nope\" = ?1", {std::int64_t{1}}).unwrap_err().code, sql_errc::undefined_object);
}

} // namespace
