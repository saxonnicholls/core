//
//  colour_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

#include <snicholls/core/colour.hpp>

#include <gtest/gtest.h>

#include <string_view>
#include <type_traits>

namespace {

static_assert(std::is_trivially_copyable_v<snicholls::colour>);
static_assert(snicholls::hex(snicholls::colours::green).view() == "#2da44e");
static_assert(snicholls::hex(snicholls::rgb(0x0a0b0c)).view() == "#0a0b0c");
static_assert(snicholls::hex(snicholls::rgb(0xFF123456)).view() == "#123456");   // bits above 24 ignored
static_assert(snicholls::rgb(0x123456).ansi == snicholls::ansi_colour::none);
static_assert(snicholls::ansi_start(snicholls::colours::green).view() == "\x1b[32m");
static_assert(snicholls::ansi_start(snicholls::colours::grey).view() == "\x1b[90m");
static_assert(snicholls::ansi_start(snicholls::rgb(0x123456)).view() == "\x1b[38;2;18;52;86m");
static_assert(snicholls::ansi_start(snicholls::rgb(0xFFFFFF)).view() == "\x1b[38;2;255;255;255m");   // the longest form fits
static_assert(snicholls::ansi_start(snicholls::rgb(0)).view() == "\x1b[38;2;0;0;0m");
static_assert(snicholls::ansi_reset == "\x1b[0m");
static_assert(snicholls::colours::red == snicholls::colour{0xcf222e, snicholls::ansi_colour::red});
static_assert(snicholls::colours::red != snicholls::colours::orange);
static_assert(noexcept(snicholls::hex(snicholls::colours::red)) && noexcept(snicholls::ansi_start(snicholls::colours::red)));

TEST(ColourSmallText, AppendStopsAtCapacityAndConvertsToStringView)
{
    snicholls::colour_text<3> t;
    t.append("abcdef");
    const std::string_view v = t;
    EXPECT_EQ(v, "abc");
    t.append_decimal(7);
    EXPECT_EQ(t.view(), "abc");
}

TEST(ColourPalette, EveryPaletteColourHasATerminalColour)
{
    for (const auto c : {snicholls::colours::red, snicholls::colours::green, snicholls::colours::yellow, snicholls::colours::orange,
                         snicholls::colours::blue, snicholls::colours::magenta, snicholls::colours::cyan, snicholls::colours::grey})
        EXPECT_NE(c.ansi, snicholls::ansi_colour::none);
}

} // namespace
