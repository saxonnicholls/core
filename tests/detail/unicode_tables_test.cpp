//
//  unicode_tables_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for the generated <snicholls/core/detail/unicode_tables.hpp>: every table against the committed extract
//  it was generated from (scripts/data/ucd-16.0.0.txt), and the tables' invariants (sorted, covering every code
//  point from 0, categories and classes in range).
//

#include <snicholls/core/detail/unicode_tables.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <string_view>

namespace {

namespace t = snicholls::detail::unicode_tables;

static_assert(std::string_view(t::version) == "16.0.0");
static_assert(t::categories[0] >> 5 == 0, "the category runs start at U+0000");
static_assert(std::size(t::lowercase_runs) % 4 == 0);

TEST(UnicodeTables, AreSortedAndInRange)
{
    for (std::size_t i = 1; i < std::size(t::categories); ++i) {
        EXPECT_LT(t::categories[i - 1] >> 5, t::categories[i] >> 5);
        EXPECT_NE(t::categories[i - 1] & 31u, t::categories[i] & 31u) << "adjacent runs differ";
    }
    for (const auto c : t::categories) {
        EXPECT_LT(c & 31u, 30u);
        EXPECT_LE(c >> 5, 0x10FFFFu);
    }
    for (std::size_t i = 4; i < std::size(t::lowercase_runs); i += 4)
        EXPECT_LT(t::lowercase_runs[i - 4], t::lowercase_runs[i]);
    for (std::size_t i = 1; i < std::size(t::decompositions); ++i)
        EXPECT_LT(t::decompositions[i - 1] >> 42, t::decompositions[i] >> 42);
    for (std::size_t i = 0; i < std::size(t::combining_classes); ++i) {
        const auto first = t::combining_classes[i] >> 29, last = (t::combining_classes[i] >> 8) & 0x1FFFFF;
        EXPECT_LE(first, last);
        EXPECT_NE(t::combining_classes[i] & 0xFF, 0u);
        if (i > 0)
            EXPECT_LT((t::combining_classes[i - 1] >> 8) & 0x1FFFFF, first);
    }
}

TEST(UnicodeTables, MatchTheCommittedExtract)
{
    std::ifstream in(std::string(SNICHOLLS_CORE_FIXTURE_DIR) + "/../../scripts/data/ucd-16.0.0.txt");
    ASSERT_TRUE(in);
    std::size_t gc = 0, lower = 0, dec = 0, ccc = 0;
    for (std::string line; std::getline(in, line);) {
        if (line.empty() || line[0] == '#')
            continue;
        std::istringstream w(line);
        std::string kind;
        w >> kind;
        if (kind == "gc") {
            std::string cp, cat;
            w >> cp >> cat;
            ASSERT_LT(gc, std::size(t::categories));
            EXPECT_EQ(t::categories[gc] >> 5, std::stoul(cp, nullptr, 16));
            ++gc;
        } else if (kind == "lower") {
            std::string first;
            long count = 0, stride = 0, delta = 0;
            w >> first >> count >> stride >> delta;
            EXPECT_EQ(t::lowercase_runs[4 * lower], static_cast<std::int32_t>(std::stoul(first, nullptr, 16)));
            EXPECT_EQ(t::lowercase_runs[4 * lower + 1], count);
            EXPECT_EQ(t::lowercase_runs[4 * lower + 2], stride);
            EXPECT_EQ(t::lowercase_runs[4 * lower + 3], delta);
            ++lower;
        } else if (kind == "dec") {
            std::string cp, a, b;
            w >> cp >> a >> b;
            const std::uint64_t want = (std::stoull(cp, nullptr, 16) << 42) | (std::stoull(a, nullptr, 16) << 21) |
                                       (b.empty() ? 0 : std::stoull(b, nullptr, 16));
            EXPECT_EQ(t::decompositions[dec], want);
            ++dec;
        } else if (kind == "ccc") {
            std::string a, b;
            unsigned k = 0;
            w >> a >> b >> k;
            EXPECT_EQ(t::combining_classes[ccc], (std::stoull(a, nullptr, 16) << 29) | (std::stoull(b, nullptr, 16) << 8) | k);
            ++ccc;
        }
    }
    EXPECT_EQ(gc, std::size(t::categories));
    EXPECT_EQ(lower, std::size(t::lowercase_runs) / 4);
    EXPECT_EQ(dec, std::size(t::decompositions));
    EXPECT_EQ(ccc, std::size(t::combining_classes));
}

} // namespace
