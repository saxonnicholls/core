//
//  iso4217_table_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/detail/iso4217_table.hpp>, the generated table: its rows agree with its types. That the
//  table is what SIX's lists say is the CTest generated.iso4217_up_to_date.
//

#include <snicholls/core/detail/iso4217_table.hpp>

#include <gtest/gtest.h>

namespace {

using namespace snicholls;

static_assert(detail::iso4217_rows.size() > 300);
static_assert(currency::EUR::numeric() == 978 && currency::EUR::scale() == 2);

TEST(Iso4217Table, RowsMatchTheirTypes)
{
    bool found = false;
    for (const auto& r : detail::iso4217_rows) {
        if (r.code != "CHF")
            continue;
        found = true;
        EXPECT_EQ(r.numeric, currency::CHF::numeric());
        EXPECT_EQ(r.scale, currency::CHF::scale());
        EXPECT_EQ(r.name, currency::CHF::name());
        EXPECT_TRUE(r.withdrawn.empty());
    }
    EXPECT_TRUE(found);
}

TEST(Iso4217Table, MarksMissingScalesAndWithdrawals)
{
    for (const auto& r : detail::iso4217_rows) {
        if (r.code == "XAU") {
            EXPECT_EQ(r.scale, -1);
        }
        if (!r.withdrawn.empty()) {
            EXPECT_EQ(r.scale, -1) << r.code;
            EXPECT_GE(r.withdrawn.size(), 4u) << r.code;   // free text: "2025-03", "1989 to 1990"
        }
    }
}

} // namespace
