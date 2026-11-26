//
//  sql_connection_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/concepts/sql_connection.hpp>: the models and the non-models, sql_execute, both sql_query
//  forms, and sql_transaction.
//

#include <snicholls/core/concepts/sql_connection.hpp>
#include <snicholls/core/interfaces/sql_connection_reference.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <span>
#include <vector>

using namespace snicholls;

namespace {

static_assert(sql_connection<memory_sql_connection>);
static_assert(sql_connection<sql_connection_interface>);
static_assert(sql_connection<sql_connection_adapter<memory_sql_connection>>);
static_assert(sql_connection<sql_connection_adapter<memory_sql_connection&>>);
static_assert(sql_connection<sql_connection_ref>);

struct throws_instead {
    const sqlite_dialect& dialect() const noexcept;
    void prepare(std::string_view);
    std::size_t execute(std::string_view, std::span<const sql_param>);
    std::size_t query(std::string_view, std::span<const sql_param>, sql_row_visitor);
    std::size_t execute_batch(std::string_view, std::size_t, std::size_t, sql_row_filler);
    void begin();
    void commit();
    void rollback();
    bool in_transaction() const noexcept;
    void ping();
    bool is_open() const noexcept;
};
static_assert(!sql_connection<throws_instead>);
static_assert(!sql_connection<int>);

const sql_table t{"t", {{"k", sql_type::int64, false}, {"v", sql_type::text, true}}, {0}, false};

memory_sql_connection with_rows()
{
    memory_sql_connection c;
    EXPECT_TRUE(sql_execute(c, *sql_create_table(c.dialect(), t)).is_ok());
    for (std::int64_t k = 1; k <= 3; ++k)
        EXPECT_TRUE(sql_execute(c, *sql_insert(c.dialect(), t), {k, sql_text("v")}).is_ok());
    return c;
}

TEST(SqlConnectionConcept, QueryTakesVoidAndBoolVisitors)
{
    memory_sql_connection c = with_rows();
    const std::string q = *sql_select_rows(c.dialect(), t);
    std::vector<std::int64_t> keys;
    EXPECT_EQ(*sql_query(c, q, {}, [&keys](const sql_row& r) { keys.push_back(*r.get_int64(0)); }), 3u);
    EXPECT_EQ(keys, (std::vector<std::int64_t>{1, 2, 3}));
    EXPECT_EQ(*sql_query(c, q, std::span<const sql_param>{}, [](const sql_row&) { return false; }), 1u);
}

TEST(SqlConnectionConcept, ATransactionRollsBackUnlessCommitted)
{
    memory_sql_connection c = with_rows();
    const std::string del = *sql_delete(c.dialect(), t);
    {
        auto tx = sql_transaction<memory_sql_connection>::begin(c);
        ASSERT_TRUE(tx.is_ok());
        EXPECT_EQ(*sql_execute(c, del), 3u);
        EXPECT_EQ(c.row_count("t"), 0u);
    }
    EXPECT_EQ(c.row_count("t"), 3u);
    EXPECT_FALSE(c.in_transaction());
    auto tx = sql_transaction<memory_sql_connection>::begin(c);
    EXPECT_EQ(sql_transaction<memory_sql_connection>::begin(c).unwrap_err().code, sql_errc::misuse);
    static_cast<void>(sql_execute(c, del));
    auto moved = std::move(*tx);
    EXPECT_TRUE(moved.commit().is_ok());
    EXPECT_EQ(c.row_count("t"), 0u);
    auto again = sql_transaction<memory_sql_connection>::begin(c);
    EXPECT_TRUE(again->rollback().is_ok());
    EXPECT_FALSE(c.in_transaction());
}

} // namespace
