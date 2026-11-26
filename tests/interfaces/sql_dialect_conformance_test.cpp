//
//  sql_dialect_conformance_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/interfaces/sql_dialect_conformance.hpp>: the kit passes the references and catches broken
//  dialects: one that does not quote, one whose placeholders repeat, one with a missing type.
//

#include <snicholls/core/interfaces/sql_dialect_conformance.hpp>
#include <snicholls/core/interfaces/sql_dialect_reference.hpp>

#include "capability_probes.hpp"

#include <gtest/gtest.h>

#include <string>

using namespace snicholls;

namespace {

enum class defect { unquoted, same_placeholder, missing_type };

template <defect D>
struct broken_dialect : sqlite_dialect {
    std::optional<std::string_view> column_type(sql_type t) const noexcept
    {
        if (D == defect::missing_type && t == sql_type::uuid)
            return std::nullopt;
        return sqlite_dialect::column_type(t);
    }
    bool append_identifier(std::string& out, std::string_view id) const
    {
        if (D != defect::unquoted)
            return sqlite_dialect::append_identifier(out, id);
        out += id;
        return true;
    }
    void append_placeholder(std::string& out, std::size_t n) const
    {
        if (D == defect::same_placeholder)
            out += '?';
        else
            sqlite_dialect::append_placeholder(out, n);
    }
};

template <defect D>
struct broken_probe {
    broken_dialect<D> d;
    const auto& dialect() { return d; }
};
struct reference_probe {
    postgres_dialect d;
    const auto& dialect() { return d; }
};

TEST(SqlDialectConformance, TheReferencePasses)
{
    const auto r = run_sql_dialect_conformance<reference_probe>();
    EXPECT_TRUE(r.ok()) << test::describe(r);
}

TEST(SqlDialectConformance, BrokenDialectsAreCaught)
{
    const auto a = run_sql_dialect_conformance<broken_probe<defect::unquoted>>();
    EXPECT_TRUE(a.failed("identifiers are quoted")) << test::describe(a);
    EXPECT_TRUE(a.failed("unquotable identifiers are refused"));
    const auto b = run_sql_dialect_conformance<broken_probe<defect::same_placeholder>>();
    EXPECT_TRUE(b.failed("placeholders are distinct")) << test::describe(b);
    const auto c = run_sql_dialect_conformance<broken_probe<defect::missing_type>>();
    EXPECT_TRUE(c.failed("every type has a name")) << test::describe(c);
    EXPECT_EQ(c.failures.size(), 1u) << test::describe(c);
}

} // namespace
