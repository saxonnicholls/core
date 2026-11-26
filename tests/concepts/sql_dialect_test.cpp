//
//  sql_dialect_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/concepts/sql_dialect.hpp>: the models and the non-models, and the generators' golden SQL for
//  both reference dialects, with their misuse errors.
//

#include <snicholls/core/concepts/sql_dialect.hpp>
#include <snicholls/core/interfaces/sql_dialect_reference.hpp>

#include <gtest/gtest.h>

#include <string>

using namespace snicholls;

namespace {

static_assert(sql_dialect<sqlite_dialect>);
static_assert(sql_dialect<postgres_dialect>);
static_assert(sql_dialect<sql_dialect_interface>);
static_assert(sql_dialect<sql_dialect_adapter<sqlite_dialect>>);
static_assert(sql_dialect<sql_dialect_adapter<const postgres_dialect&>>);
static_assert(sql_dialect<sql_dialect_ref>);

struct no_suffix {
    std::string_view name() const;
    std::optional<std::string_view> column_type(sql_type) const;
    bool append_identifier(std::string&, std::string_view) const;
    void append_placeholder(std::string&, std::size_t) const;
    std::string_view begin_transaction() const;
};
struct identifier_not_checked {
    std::string_view name() const;
    std::optional<std::string_view> column_type(sql_type) const;
    void append_identifier(std::string&, std::string_view) const;
    void append_placeholder(std::string&, std::size_t) const;
    std::string_view begin_transaction() const;
    std::string_view clustered_suffix() const;
};
static_assert(!sql_dialect<no_suffix>);
static_assert(!sql_dialect<identifier_not_checked>);
static_assert(!sql_dialect<int>);

const sql_table quotes{"quote",
                       {{"series", sql_type::text, false}, {"ts", sql_type::timestamp, false}, {"px", sql_type::float64, true}},
                       {0, 1},
                       true};

TEST(SqlDialectConcept, CreateTableIsGoldenOnBothEngines)
{
    EXPECT_EQ(*sql_create_table(sqlite_dialect{}, quotes),
              "CREATE TABLE IF NOT EXISTS \"quote\" (\"series\" TEXT NOT NULL, \"ts\" INTEGER NOT NULL, \"px\" REAL, "
              "PRIMARY KEY (\"series\", \"ts\")) WITHOUT ROWID");
    EXPECT_EQ(*sql_create_table(postgres_dialect{}, quotes),
              "CREATE TABLE IF NOT EXISTS \"quote\" (\"series\" TEXT NOT NULL, \"ts\" BIGINT NOT NULL, \"px\" DOUBLE PRECISION, "
              "PRIMARY KEY (\"series\", \"ts\"))");
    EXPECT_EQ(*sql_drop_table(postgres_dialect{}, "quote"), "DROP TABLE IF EXISTS \"quote\"");
    EXPECT_EQ(*sql_drop_table(sqlite_dialect{}, "quote", false), "DROP TABLE \"quote\"");
}

TEST(SqlDialectConcept, InsertIsGoldenForEveryConflict)
{
    EXPECT_EQ(*sql_insert(sqlite_dialect{}, quotes), "INSERT INTO \"quote\" (\"series\", \"ts\", \"px\") VALUES (?1, ?2, ?3)");
    EXPECT_EQ(*sql_insert(postgres_dialect{}, quotes, sql_conflict::ignore),
              "INSERT INTO \"quote\" (\"series\", \"ts\", \"px\") VALUES ($1, $2, $3) ON CONFLICT (\"series\", \"ts\") DO NOTHING");
    EXPECT_EQ(*sql_insert(postgres_dialect{}, quotes, sql_conflict::update),
              "INSERT INTO \"quote\" (\"series\", \"ts\", \"px\") VALUES ($1, $2, $3) ON CONFLICT (\"series\", \"ts\") DO UPDATE SET "
              "\"px\" = excluded.\"px\"");
    sql_table keys_only = quotes;
    keys_only.columns.pop_back();
    EXPECT_NE(sql_insert(sqlite_dialect{}, keys_only, sql_conflict::update)->find("DO NOTHING"), std::string::npos);
}

TEST(SqlDialectConcept, SelectCountAndDeleteNumberTheirParameters)
{
    sql_select w;
    w.equal = {0};
    w.range_column = 1;
    w.has_from = true;
    w.has_to = true;
    w.order_by = {1};
    w.descending = true;
    w.limit = 10;
    EXPECT_EQ(*sql_select_rows(postgres_dialect{}, quotes, w),
              "SELECT \"series\", \"ts\", \"px\" FROM \"quote\" WHERE \"series\" = $1 AND \"ts\" >= $2 AND \"ts\" < $3 "
              "ORDER BY \"ts\" DESC LIMIT 10");
    EXPECT_EQ(*sql_count(sqlite_dialect{}, quotes, w), "SELECT count(*) FROM \"quote\" WHERE \"series\" = ?1 AND \"ts\" >= ?2 AND \"ts\" < ?3");
    EXPECT_EQ(*sql_delete(sqlite_dialect{}, quotes), "DELETE FROM \"quote\"");
    sql_select to_only;
    to_only.range_column = 1;
    to_only.has_to = true;
    EXPECT_EQ(*sql_delete(sqlite_dialect{}, quotes, to_only), "DELETE FROM \"quote\" WHERE \"ts\" < ?1");
    EXPECT_EQ(*sql_select_rows(sqlite_dialect{}, quotes), "SELECT \"series\", \"ts\", \"px\" FROM \"quote\"");
}

TEST(SqlDialectConcept, MisuseIsAnError)
{
    const sqlite_dialect d;
    EXPECT_EQ(sql_create_table(d, sql_table{}).unwrap_err().code, sql_errc::misuse);
    sql_table no_key = quotes;
    no_key.primary_key.clear();
    EXPECT_EQ(sql_insert(d, no_key, sql_conflict::ignore).unwrap_err().code, sql_errc::misuse);
    EXPECT_EQ(sql_create_table(d, no_key)->find("PRIMARY KEY"), std::string::npos);
    sql_table bad_key = quotes;
    bad_key.primary_key = {7};
    EXPECT_EQ(sql_create_table(d, bad_key).unwrap_err().code, sql_errc::misuse);
    sql_select past;
    past.equal = {9};
    EXPECT_EQ(sql_select_rows(d, quotes, past).unwrap_err().code, sql_errc::misuse);
    sql_select no_column;
    no_column.has_from = true;
    EXPECT_EQ(sql_count(d, quotes, no_column).unwrap_err().message, std::string_view{"a range without a column"});
    EXPECT_EQ(sql_drop_table(d, "").unwrap_err().code, sql_errc::misuse);
}

TEST(SqlDialectConcept, StandardQuoting)
{
    std::string out;
    EXPECT_TRUE(append_quoted_identifier(out, "a\"b"));
    EXPECT_EQ(out, "\"a\"\"b\"");
    EXPECT_FALSE(append_quoted_identifier(out, ""));
    EXPECT_FALSE(append_quoted_identifier(out, std::string_view("\0", 1)));
    EXPECT_EQ(out, "\"a\"\"b\"");
}

} // namespace
