//
//  iso3166_table_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/detail/iso3166_table.hpp>, the generated table: its rows agree with its types, its
//  permutations are sorted, and its calling codes are rows of the E.164 table. That the table is what its sources say
//  is the CTest generated.iso3166_up_to_date.
//

#include <snicholls/core/detail/iso3166_table.hpp>

#include <gtest/gtest.h>

#include <set>

namespace {

using namespace snicholls;

static_assert(detail::iso3166_rows.size() == 249);
static_assert(detail::iso3166_rows[0].alpha2 == "AD");
static_assert(country::CH::numeric() == 756 && country::CH::calling_code() == 41);

TEST(Iso3166Table, RowsMatchTheirTypes)
{
    bool found = false;
    for (const auto& r : detail::iso3166_rows) {
        if (r.alpha2 != "CH")
            continue;
        found = true;
        EXPECT_EQ(r.alpha3, country::CH::alpha3());
        EXPECT_EQ(r.numeric, country::CH::numeric());
        EXPECT_EQ(r.name, country::CH::name());
        EXPECT_EQ(r.calling_code, country::CH::calling_code());
        EXPECT_EQ(r.currencies.data(), country::CH::currencies().data());
    }
    EXPECT_TRUE(found);
}

TEST(Iso3166Table, PermutationsAreSortedAndComplete)
{
    std::set<int> seen3, seen_n;
    for (std::size_t i = 0; i < detail::iso3166_rows.size(); ++i) {
        seen3.insert(detail::iso3166_by_alpha3[i]);
        seen_n.insert(detail::iso3166_by_numeric[i]);
        if (i == 0)
            continue;
        EXPECT_LT(detail::iso3166_rows[detail::iso3166_by_alpha3[i - 1]].alpha3,
                  detail::iso3166_rows[detail::iso3166_by_alpha3[i]].alpha3);
        EXPECT_LT(detail::iso3166_rows[detail::iso3166_by_numeric[i - 1]].numeric,
                  detail::iso3166_rows[detail::iso3166_by_numeric[i]].numeric);
    }
    EXPECT_EQ(seen3.size(), detail::iso3166_rows.size());
    EXPECT_EQ(seen_n.size(), detail::iso3166_rows.size());
}

TEST(Iso3166Table, CallingCodesComeFromTheE164Table)
{
    std::set<std::uint16_t> e164;
    for (const auto& c : detail::e164::country_codes)
        e164.insert(c.code);
    int none = 0;
    for (const auto& r : detail::iso3166_rows) {
        if (r.calling_code == 0)
            ++none;
        else
            EXPECT_TRUE(e164.contains(r.calling_code)) << r.alpha2;
    }
    EXPECT_EQ(none, 7);   // AQ BV GS HM PN TF UM: libphonenumber assigns them none
}

} // namespace
