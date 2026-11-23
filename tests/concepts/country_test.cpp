//
//  country_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/concepts/country.hpp>, at compile time: the generated types, the interface and the
//  adapter model country_like; a type missing an operation, or with a wrong signature, does not.
//

#include <snicholls/core/concepts/country.hpp>
#include <snicholls/core/country.hpp>
#include <snicholls/core/interfaces/country.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <span>
#include <string_view>

namespace {

using namespace snicholls;

struct no_calling_code {
    static constexpr std::string_view name() noexcept { return "x"; }
    static constexpr std::string_view alpha2() noexcept { return "XA"; }
    static constexpr std::string_view alpha3() noexcept { return "XAA"; }
    static constexpr std::uint16_t numeric() noexcept { return 900; }
    static constexpr std::span<const std::string_view> currencies() noexcept { return {}; }
};
struct currencies_not_a_span {
    static constexpr std::string_view name() noexcept { return "x"; }
    static constexpr std::string_view alpha2() noexcept { return "XA"; }
    static constexpr std::string_view alpha3() noexcept { return "XAA"; }
    static constexpr std::uint16_t numeric() noexcept { return 900; }
    static constexpr int currencies() noexcept { return 0; }
    static constexpr std::uint16_t calling_code() noexcept { return 0; }
};
struct unnamed {
    static constexpr std::string_view alpha2() noexcept { return "XA"; }
    static constexpr std::string_view alpha3() noexcept { return "XAA"; }
    static constexpr std::uint16_t numeric() noexcept { return 900; }
    static constexpr std::span<const std::string_view> currencies() noexcept { return {}; }
    static constexpr std::uint16_t calling_code() noexcept { return 0; }
};

static_assert(country_like<country::AU> && country_like<country::AQ>);
static_assert(country_like<country_interface>);
static_assert(country_like<country_model<country::NZ>>);
static_assert(!country_like<no_calling_code> && !country_like<currencies_not_a_span> && !country_like<unnamed>);
static_assert(!country_like<int> && !country_like<country_entry>, "the table's row has fields, not members");

TEST(CountryConcepts, TypesAnswerAtCompileTime)
{
    EXPECT_EQ(country::NZ::alpha3(), "NZL");
    EXPECT_EQ(country::NZ::calling_code(), 64);
}

} // namespace
