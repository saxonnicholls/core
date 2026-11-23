//
//  country_conformance_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/interfaces/country_conformance.hpp>: every generated type conforms, and each check
//  names itself when it fails.
//

#include <snicholls/core/country.hpp>
#include <snicholls/core/interfaces/country.hpp>
#include <snicholls/core/interfaces/country_conformance.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <span>
#include <string_view>

namespace {

using namespace snicholls;

struct fake {
    std::string_view n = "Australia", a2 = "AU", a3 = "AUS";
    std::uint16_t number = 36, code = 61;
    std::span<const std::string_view> money = {};
    constexpr std::string_view name() const noexcept { return n; }
    constexpr std::string_view alpha2() const noexcept { return a2; }
    constexpr std::string_view alpha3() const noexcept { return a3; }
    constexpr std::uint16_t numeric() const noexcept { return number; }
    constexpr std::span<const std::string_view> currencies() const noexcept { return money; }
    constexpr std::uint16_t calling_code() const noexcept { return code; }
};

constexpr std::array<std::string_view, 1> lower{"aud"};

static_assert(!check_country(country::AU{}) && !check_country(fake{}));
static_assert(*check_country(fake{.n = ""}) == "named: name() is empty");
static_assert(*check_country(fake{.a2 = "Au"}) == "country: alpha2() is not two capitals");
static_assert(*check_country(fake{.a3 = "AU"}) == "country: alpha3() is not three capitals");
static_assert(*check_country(fake{.number = 0}) == "country: numeric() is not 1 to 999");
static_assert(*check_country(fake{.number = 1000}) == "country: numeric() is not 1 to 999");
static_assert(*check_country(fake{.a3 = "AUT"}) == "country: alpha3() or numeric() disagrees with ISO 3166-1");
static_assert(*check_country(fake{.number = 554}) == "country: alpha3() or numeric() disagrees with ISO 3166-1");
static_assert(*check_country(fake{.money = lower}) == "country: a currency is not three capitals");
static_assert(*check_country(fake{.code = 1000}) == "country: calling_code() is above 999");

template <typename... Cs>
bool all_conform()
{
    return (!check_country(Cs{}) && ...) && (!check_country(country_model<Cs>{}) && ...);
}

TEST(CountryConformance, EveryGeneratedEntryConforms)
{
    EXPECT_TRUE((all_conform<country::AU, country::AQ, country::GB, country::US, country::CH, country::ZW>()));
    for (const auto& e : iso3166_entries()) {
        const fake f{e.name, e.alpha2, e.alpha3, e.numeric, e.calling_code, e.currencies};
        EXPECT_FALSE(check_country(f)) << e.alpha2;
    }
}

} // namespace
