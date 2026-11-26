//
//  sql_connection_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/interfaces/sql_connection.hpp>: the conformance kit passes on the reference in every form
//  (the concept, the virtual interface over a borrowing adapter, the handle), and the adapter puts the model's dialect
//  behind the dialect interface.
//

#include <snicholls/core/interfaces/sql_connection.hpp>
#include <snicholls/core/interfaces/sql_connection_conformance.hpp>

#include "contract_probes.hpp"

#include <gtest/gtest.h>

#include <type_traits>

using namespace snicholls;

namespace {

static_assert(std::is_abstract_v<sql_connection_interface>);
static_assert(!std::is_copy_constructible_v<sql_connection_adapter<memory_sql_connection>>);

TEST(SqlConnectionInterface, TheKitPassesInEveryForm)
{
    const auto direct = run_sql_connection_conformance<test::sql_connection_probe<test::form::concept_model>>();
    const auto virtual_form = run_sql_connection_conformance<test::sql_connection_probe<test::form::interface>>();
    const auto handle = run_sql_connection_conformance<test::sql_connection_probe<test::form::ref>>();
    for (const auto* r : {&direct, &virtual_form, &handle}) {
        EXPECT_TRUE(r->ok()) << test::describe(*r);
        EXPECT_EQ(r->checks, 9u);
    }
}

TEST(SqlConnectionInterface, TheAdapterExposesTheModelsDialect)
{
    sql_connection_adapter<memory_sql_connection> owned{memory_sql_connection{}};
    sql_connection_interface& c = owned;
    EXPECT_EQ(c.dialect().name(), "sqlite");
    const sql_table t{"t", {{"k", sql_type::int64, false}}, {0}, false};
    EXPECT_TRUE(sql_execute(c, *sql_create_table(c.dialect(), t)).is_ok());
    EXPECT_TRUE(sql_execute(c, *sql_insert(c.dialect(), t), {std::int64_t{1}}).is_ok());
    EXPECT_EQ(owned.get().row_count("t"), 1u);
    EXPECT_TRUE(c.ping().is_ok());
    owned.get().break_connection();
    EXPECT_FALSE(c.is_open());
    sql_connection_ref ref{c};
    EXPECT_EQ(ref.prepare("COMMIT").unwrap_err().code, sql_errc::connection_lost);
}

} // namespace
