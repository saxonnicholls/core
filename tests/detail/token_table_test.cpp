//
//  token_table_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/detail/token_table.hpp>, the generated types: the list, the kinds and the access date. That they are
//  what the committed data says is tests/token/tokens_test.cpp and the CTest generated.tokens_up_to_date.
//

#include <snicholls/core/detail/token_table.hpp>

#include <gtest/gtest.h>

#include <type_traits>

namespace {

using namespace snicholls;

static_assert(meta::size_v<token::all> == 19);
static_assert(std::is_same_v<meta::front_t<token::all>, token::ethereum::DAI>);
static_assert(std::is_same_v<meta::back_t<token::all>, token::tron::USDT>);

static_assert(token::issuance::native != token::issuance::bridged);
static_assert(std::is_same_v<decltype(token::ethereum::USDT::kind()), token::issuance>);

TEST(TokenTable, RecordsTheAccessDate)
{
    EXPECT_EQ(detail::token_data_accessed, "2026-11-24");
    EXPECT_EQ(token::ethereum::USDT::name(), "Tether USD");
}

} // namespace
