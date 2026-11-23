//
//  asset_conformance_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/interfaces/asset_conformance.hpp>: the kit catches each broken rule.
//

#include <snicholls/core/interfaces/asset_conformance.hpp>

#include <gtest/gtest.h>

namespace {

using namespace snicholls;

struct bad_code {
    static constexpr std::string_view code() noexcept { return "usd"; }
    static constexpr std::uint8_t scale() noexcept { return 2; }
};
struct bad_scale {
    static constexpr std::string_view code() noexcept { return "USD"; }
    static constexpr std::uint8_t scale() noexcept { return 40; }
};
struct bad_numeric {
    static constexpr std::string_view code() noexcept { return "USD"; }
    static constexpr std::uint8_t scale() noexcept { return 2; }
    static constexpr std::string_view name() noexcept { return "Dollar"; }
    static constexpr std::uint16_t numeric() noexcept { return 0; }
};
struct unnamed {
    static constexpr std::string_view code() noexcept { return "USD"; }
    static constexpr std::uint8_t scale() noexcept { return 2; }
    static constexpr std::string_view name() noexcept { return ""; }
    static constexpr std::uint16_t numeric() noexcept { return 840; }
};

TEST(AssetConformance, NamesTheFailedCheck)
{
    EXPECT_EQ(*check_asset(bad_code{}), "asset: code() is not 1 to 24 of A-Z 0-9 : . - _");
    EXPECT_EQ(*check_asset(bad_scale{}), "asset: scale() is above max_scale");
    EXPECT_EQ(*check_currency(bad_numeric{}), "currency: numeric() is not between 1 and 999");
    EXPECT_EQ(*check_currency(unnamed{}), "named: name() is empty");
}

} // namespace
