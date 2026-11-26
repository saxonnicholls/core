//
//  sql_dialect_reference_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/interfaces/sql_dialect_reference.hpp>: every type name of both dialects, their placeholders
//  and transaction statements, at compile time where the language allows.
//

#include <snicholls/core/interfaces/sql_dialect_reference.hpp>

#include <gtest/gtest.h>

#include <string>
#include <type_traits>

using namespace snicholls;

namespace {

static_assert(sqlite_dialect{}.name() == "sqlite" && postgres_dialect{}.name() == "postgres");
static_assert(*sqlite_dialect{}.column_type(sql_type::boolean) == "INTEGER");
static_assert(*sqlite_dialect{}.column_type(sql_type::uuid) == "BLOB");
static_assert(*sqlite_dialect{}.column_type(sql_type::decimal) == "TEXT");
static_assert(*sqlite_dialect{}.column_type(sql_type::float64) == "REAL");
static_assert(*postgres_dialect{}.column_type(sql_type::boolean) == "BOOLEAN");
static_assert(*postgres_dialect{}.column_type(sql_type::timestamp) == "BIGINT");
static_assert(*postgres_dialect{}.column_type(sql_type::uuid) == "BYTEA");
static_assert(*postgres_dialect{}.column_type(sql_type::float64) == "DOUBLE PRECISION");
static_assert(!sqlite_dialect{}.column_type(static_cast<sql_type>(99)));
static_assert(sqlite_dialect{}.begin_transaction() == "BEGIN IMMEDIATE" && postgres_dialect{}.begin_transaction() == "BEGIN");
static_assert(sqlite_dialect{}.clustered_suffix() == " WITHOUT ROWID" && postgres_dialect{}.clustered_suffix().empty());
static_assert(std::is_empty_v<sqlite_dialect> && std::is_trivially_copyable_v<postgres_dialect>);

TEST(SqlDialectReference, Placeholders)
{
    std::string s, p;
    sqlite_dialect{}.append_placeholder(s, 12);
    postgres_dialect{}.append_placeholder(p, 12);
    EXPECT_EQ(s, "?12");
    EXPECT_EQ(p, "$12");
}

TEST(SqlDialectReference, EveryTypeHasAName)
{
    for (const sql_type t : sql_types) {
        EXPECT_TRUE(sqlite_dialect{}.column_type(t).has_value());
        EXPECT_TRUE(postgres_dialect{}.column_type(t).has_value());
    }
    std::string out;
    EXPECT_TRUE(postgres_dialect{}.append_identifier(out, "Mixed Case"));
    EXPECT_EQ(out, "\"Mixed Case\"");
}

} // namespace
