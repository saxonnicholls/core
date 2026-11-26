//
//  sql_dialect_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/interfaces/sql_dialect.hpp>: the conformance kit passes on both reference dialects in every
//  form (the concept, the virtual interface over a borrowing adapter, the handle), and the adapters forward.
//

#include <snicholls/core/interfaces/sql_dialect.hpp>
#include <snicholls/core/interfaces/sql_dialect_conformance.hpp>
#include <snicholls/core/interfaces/sql_dialect_reference.hpp>

#include "capability_probes.hpp"

#include <gtest/gtest.h>

#include <string>
#include <type_traits>

using namespace snicholls;

namespace {

static_assert(std::is_abstract_v<sql_dialect_interface>);

template <typename D, test::form F>
struct probe {
    D impl{};
    sql_dialect_adapter<const D&> adapter{impl};
    sql_dialect_ref handle{adapter};
    const auto& dialect()
    {
        if constexpr (F == test::form::concept_model)
            return impl;
        else if constexpr (F == test::form::interface)
            return static_cast<const sql_dialect_interface&>(adapter);
        else
            return handle;
    }
};

TEST(SqlDialectInterface, TheKitPassesInEveryForm)
{
    using test::form;
    const auto a = run_sql_dialect_conformance<probe<sqlite_dialect, form::concept_model>>();
    const auto b = run_sql_dialect_conformance<probe<sqlite_dialect, form::interface>>();
    const auto c = run_sql_dialect_conformance<probe<sqlite_dialect, form::ref>>();
    const auto d = run_sql_dialect_conformance<probe<postgres_dialect, form::concept_model>>();
    const auto e = run_sql_dialect_conformance<probe<postgres_dialect, form::interface>>();
    const auto f = run_sql_dialect_conformance<probe<postgres_dialect, form::ref>>();
    for (const auto* r : {&a, &b, &c, &d, &e, &f}) {
        EXPECT_TRUE(r->ok()) << test::describe(*r);
        EXPECT_EQ(r->checks, 6u);
    }
}

TEST(SqlDialectInterface, TheGeneratorsTakeTheInterface)
{
    const sql_dialect_adapter pg{postgres_dialect{}};
    const sql_dialect_interface& d = pg;
    const sql_table t{"t", {{"k", sql_type::int64, false}}, {0}, false};
    EXPECT_EQ(*sql_insert(d, t), "INSERT INTO \"t\" (\"k\") VALUES ($1)");
    EXPECT_EQ(d.name(), "postgres");
    EXPECT_EQ(pg.get().name(), "postgres");
    EXPECT_EQ(d.clustered_suffix(), "");
    EXPECT_EQ(d.begin_transaction(), "BEGIN");
}

} // namespace
