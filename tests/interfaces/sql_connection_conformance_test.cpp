//
//  sql_connection_conformance_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/interfaces/sql_connection_conformance.hpp>: the kit passes the reference and catches broken
//  connections: one whose rollback keeps, one whose batch is not atomic, one that never reports a lost connection.
//

#include <snicholls/core/interfaces/sql_connection_conformance.hpp>

#include "contract_probes.hpp"

#include <gtest/gtest.h>

#include <span>
#include <string_view>

using namespace snicholls;

namespace {

enum class defect { rollback_keeps, batch_not_atomic, never_lost };

template <defect D>
struct broken_connection {
    memory_sql_connection inner;
    bool open = true;
    const sqlite_dialect& dialect() const noexcept { return inner.dialect(); }
    sql_result<void> prepare(std::string_view s) { return inner.prepare(s); }
    sql_result<std::size_t> execute(std::string_view s, std::span<const sql_param> p) { return inner.execute(s, p); }
    sql_result<std::size_t> query(std::string_view s, std::span<const sql_param> p, sql_row_visitor v) { return inner.query(s, p, v); }
    sql_result<std::size_t> execute_batch(std::string_view s, std::size_t w, std::size_t rows, sql_row_filler fill)
    {
        if (D != defect::batch_not_atomic)
            return inner.execute_batch(s, w, rows, fill);
        std::vector<sql_param> params(w);
        for (std::size_t r = 0; r < rows; ++r) {
            fill(r, params);
            SNICHOLLS_TRY_VOID(inner.execute(s, params));
        }
        return ok(rows);
    }
    sql_result<void> begin() { return inner.begin(); }
    sql_result<void> commit() { return inner.commit(); }
    sql_result<void> rollback() { return D == defect::rollback_keeps ? inner.commit() : inner.rollback(); }
    bool in_transaction() const noexcept { return inner.in_transaction(); }
    sql_result<void> ping() { return D == defect::never_lost ? sql_result<void>(ok()) : inner.ping(); }
    bool is_open() const noexcept { return D == defect::never_lost || inner.is_open(); }
};

template <defect D>
struct broken_probe {
    broken_connection<D> c;
    auto& connection() { return c; }
    void connection_breaks() { c.inner.break_connection(); }
};

TEST(SqlConnectionConformance, TheReferencePasses)
{
    const auto r = run_sql_connection_conformance<test::sql_connection_probe<test::form::concept_model>>();
    EXPECT_TRUE(r.ok()) << test::describe(r);
}

TEST(SqlConnectionConformance, BrokenConnectionsAreCaught)
{
    const auto a = run_sql_connection_conformance<broken_probe<defect::rollback_keeps>>();
    EXPECT_TRUE(a.failed("a rollback discards and a commit keeps")) << test::describe(a);
    const auto b = run_sql_connection_conformance<broken_probe<defect::batch_not_atomic>>();
    EXPECT_TRUE(b.failed("a failed batch changes nothing")) << test::describe(b);
    EXPECT_FALSE(b.failed("values round-trip"));
    const auto c = run_sql_connection_conformance<broken_probe<defect::never_lost>>();
    EXPECT_TRUE(c.failed("a lost connection is connection_lost")) << test::describe(c);
    EXPECT_EQ(c.failures.size(), 2u) << test::describe(c);
}

} // namespace
