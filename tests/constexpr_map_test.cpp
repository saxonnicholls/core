//
//  constexpr_map_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  The lookups are constant expressions, so most claims are static_asserts; the throwing path and iteration
//  are run-time tests.
//

#include <snicholls/core/constexpr_map.hpp>
#include <snicholls/core/fixed_string.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace {

enum class colour { red, green, blue };

constexpr auto names = snicholls::make_constexpr_map(
    std::pair{colour::red, std::string_view{"red"}},
    std::pair{colour::green, std::string_view{"green"}},
    std::pair{colour::blue, std::string_view{"blue"}});

using names_t = std::remove_const_t<decltype(names)>;
static_assert(std::is_same_v<names_t, snicholls::constexpr_map<colour, std::string_view, 3>>);
static_assert(std::is_aggregate_v<names_t>);
static_assert(names.size() == 3 && !names.empty());

// Lookups both ways are constant expressions
static_assert(*names.find(colour::green) == "green");
static_assert(names.at(colour::blue) == "blue");
static_assert(names.contains(colour::red));
static_assert(*names.find_key(std::string_view{"red"}) == colour::red);
static_assert(names.keys_unique());

// Absence is null, not an error, and noexcept
static_assert(names.find(static_cast<colour>(9)) == nullptr);
static_assert(names.find_key(std::string_view{"mauve"}) == nullptr);
static_assert(!names.contains(static_cast<colour>(9)));
static_assert(noexcept(names.find(colour::red)) && noexcept(names.find_key(std::string_view{})) &&
              noexcept(names.contains(colour::red)) && noexcept(names.keys_unique()));

// Entries keep the order written, and iterate
static_assert(names.entries[0].first == colour::red && names.entries[2].first == colour::blue);
static_assert(names.begin()->second == "red" && (names.end() - 1)->second == "blue");

// The deduction guide from a std::array, and text keys through fixed_string
constexpr snicholls::constexpr_map scales{std::array{std::pair{std::string_view{"ms"}, 1000}, std::pair{std::string_view{"s"}, 1}}};
static_assert(std::is_same_v<std::remove_const_t<decltype(scales)>, snicholls::constexpr_map<std::string_view, int, 2>>);
static_assert(scales.at("ms") == 1000);   // const char* converts to the string_view key
static_assert(scales.at(snicholls::fixed_string{"s"}.view()) == 1);

// Duplicate keys: the first wins, and keys_unique says so
constexpr auto dup = snicholls::make_constexpr_map(std::pair{1, 'a'}, std::pair{1, 'b'}, std::pair{2, 'c'});
static_assert(*dup.find(1) == 'a');
static_assert(!dup.keys_unique());

// The empty map
constexpr snicholls::constexpr_map<int, int, 0> none{};
static_assert(none.empty() && none.find(1) == nullptr && none.keys_unique());

TEST(ConstexprMapAt, ThrowsOutOfRangeForAnAbsentKey)
{
    EXPECT_EQ(names.at(colour::green), "green");
    EXPECT_THROW((void)names.at(static_cast<colour>(42)), std::out_of_range);
    try {
        (void)names.at(static_cast<colour>(42));
    } catch (const std::out_of_range& e) {
        EXPECT_STREQ(e.what(), "snicholls::constexpr_map::at: key not found");
    }
}

TEST(ConstexprMapIteration, VisitsEveryEntryInOrder)
{
    std::string seen;
    for (const auto& [key, name] : names)
        seen += name;
    EXPECT_EQ(seen, "redgreenblue");
    EXPECT_EQ(static_cast<std::size_t>(names.end() - names.begin()), names.size());
}

TEST(ConstexprMapRuntime, WorksWithNonLiteralKeysAtRunTime)
{
    const auto words = snicholls::make_constexpr_map(std::pair{std::string{"one"}, 1}, std::pair{std::string{"two"}, 2});
    EXPECT_EQ(words.at("two"), 2);
    EXPECT_EQ(words.find("three"), nullptr);
    EXPECT_EQ(*words.find_key(1), "one");
}

} // namespace
