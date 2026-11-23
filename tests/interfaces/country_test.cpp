//
//  country_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/interfaces/country.hpp>: the adapter in constant expressions, a run-time choice of
//  country through the interface, a caller's own type adapted, and the conformance kit through both forms.
//

#include <snicholls/core/country.hpp>
#include <snicholls/core/interfaces/country.hpp>
#include <snicholls/core/interfaces/country_conformance.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

namespace {

using namespace snicholls;

constexpr country_model<country::AU> au_model{};
static_assert(au_model.alpha3() == "AUS" && au_model.calling_code() == 61 && au_model.currencies()[0] == "AUD",
              "the adapter is usable in constant expressions");

// A caller's own record, held by value.
struct record {
    std::string_view a2, a3, display;
    std::uint16_t number = 0, code = 0;
    std::span<const std::string_view> money;
    constexpr std::string_view name() const noexcept { return display; }
    constexpr std::string_view alpha2() const noexcept { return a2; }
    constexpr std::string_view alpha3() const noexcept { return a3; }
    constexpr std::uint16_t numeric() const noexcept { return number; }
    constexpr std::span<const std::string_view> currencies() const noexcept { return money; }
    constexpr std::uint16_t calling_code() const noexcept { return code; }
};

TEST(CountryInterface, ChoosesACountryAtRunTime)
{
    std::vector<std::unique_ptr<country_interface>> form;
    form.push_back(std::make_unique<country_model<country::AU>>());
    form.push_back(std::make_unique<country_model<country::CH>>());
    form.push_back(std::make_unique<country_model<country::AQ>>());
    EXPECT_EQ(form[0]->name(), "Australia");
    EXPECT_EQ(form[1]->currencies()[0], "CHF");
    EXPECT_EQ(form[1]->numeric(), 756);
    EXPECT_EQ(form[2]->calling_code(), 0);
    for (const auto& c : form)
        EXPECT_FALSE(check_country(*c)) << c->alpha2();
}

TEST(CountryInterface, AdaptsACallersOwnValue)
{
    static constexpr std::array<std::string_view, 1> eur{"EUR"};
    const country_model<record> kosovo{record{"XK", "XKK", "Kosovo", 983, 383, eur}};
    const country_interface& c = kosovo;
    EXPECT_EQ(c.alpha2(), "XK");
    EXPECT_EQ(c.calling_code(), 383);
    EXPECT_EQ(c.currencies().size(), 1u);
    EXPECT_FALSE(check_country(c)) << "a user-assigned code is checked for form only";
}

} // namespace
