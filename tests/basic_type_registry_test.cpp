//
//  basic_type_registry_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/basic_type_registry.hpp>.
//  Moved from base-encode-decode (tests/test_basic_type_registry.cpp at
//  c3a2ed0), from plain asserts to GoogleTest; every assertion is kept. The
//  original's "did not pull in the encoder" checks have nothing to check here:
//  core has no encoder, and the header-check target compiles the header alone.
//

#include <snicholls/core/basic_type_registry.hpp>
#include <snicholls/core/overloaded.hpp>

#include <gtest/gtest.h>

#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace {

using snicholls::basic_type_registry;
using snicholls::named;

struct Celsius { int value; };
struct Label   { std::string text; };

// A trivial text format: "42" for Celsius, the raw string for Label.
struct TextFormat {
    Celsius operator()(std::type_identity<Celsius>, std::string_view s) const {
        return Celsius{ std::stoi(std::string(s)) };
    }
    Label operator()(std::type_identity<Label>, std::string_view s) const {
        return Label{ std::string(s) };
    }
};

using Registry = basic_type_registry<
    named<"celsius", Celsius>,
    named<"label", Label>>;

static_assert(Registry::count == 2);
static_assert(Registry::names[0] == "celsius" && Registry::names[1] == "label");
static_assert(Registry::contains("label") && !Registry::contains("kelvin"));
static_assert(Registry::name_of<Celsius>() == "celsius");
static_assert(Registry::name_of<Label>() == "label");
static_assert(std::is_same_v<named<"x", int>::type, int> && named<"x", int>::name == "x");

TEST(BasicTypeRegistryDispatch, UsesTheCallersFormatAndAPerTypeHandler)
{
    int degrees = 0;
    std::string text;
    const bool handled = Registry::dispatch("celsius", "21",
        snicholls::overloaded{
            [&](Celsius&& c) { degrees = c.value; },
            [&](Label&& l)   { text = std::move(l.text); },
        },
        TextFormat{});
    EXPECT_TRUE(handled);
    EXPECT_EQ(degrees, 21);
    EXPECT_TRUE(text.empty());
}

TEST(BasicTypeRegistryDispatch, GenericHandlerSeesTheBoundType)
{
    std::string text;
    EXPECT_TRUE(Registry::dispatch("label", "hello",
        [&](auto&& obj) {
            if constexpr (std::is_same_v<std::decay_t<decltype(obj)>, Label>) text = obj.text;
        },
        TextFormat{}));
    EXPECT_EQ(text, "hello");
}

TEST(BasicTypeRegistryDispatch, UnknownNameCallsNeitherHandlerNorDeserialiser)
{
    bool called = false;
    EXPECT_FALSE(Registry::dispatch("kelvin", "0",
        [&](auto&&) { called = true; },
        [&]<typename T>(std::type_identity<T>, std::string_view) -> T { called = true; return T{}; }));
    EXPECT_FALSE(called);
}

TEST(BasicTypeRegistryContains, RuntimeAgreesWithTheCompileTimeNames)
{
    for (const auto name : Registry::names) {
        EXPECT_TRUE(Registry::contains(name)) << name;
    }
    const std::string runtime_name = "celsius";
    EXPECT_TRUE(Registry::contains(runtime_name));
}

} // namespace
