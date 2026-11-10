//
//  type_name_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/meta/type_name.hpp>: the portable spellings as
//  static_asserts, the normalisation of class-keys and inline namespaces, and
//  the shape of what is not portable.
//

#include <snicholls/core/meta/type_name.hpp>

#include <gtest/gtest.h>

#include <string>
#include <string_view>
#include <vector>

namespace demo_names {
struct widget {};
class gadget {};
struct my_struct {};
enum class colour { red };
template <typename T>
struct box {};
template <typename A, typename B>
struct two {};
} // namespace demo_names

namespace {

using snicholls::type_name;

// Fundamental types spell the same everywhere
static_assert(type_name<int>() == "int");
static_assert(type_name<double>() == "double");
static_assert(type_name<unsigned int>() == "unsigned int");
static_assert(type_name<bool>() == "bool");
static_assert(type_name<char>() == "char");

// User types: the namespace is kept, the class-key MSVC writes is not
static_assert(type_name<demo_names::widget>() == "demo_names::widget");
static_assert(type_name<demo_names::gadget>() == "demo_names::gadget");
static_assert(type_name<demo_names::colour>() == "demo_names::colour");
static_assert(type_name<demo_names::box<int>>() == "demo_names::box<int>");

// A class-key is stripped as a whole word only, and std:: survives the inline namespace
static_assert(type_name<demo_names::my_struct>() == "demo_names::my_struct");
static_assert(type_name<std::string>().starts_with("std::basic_string<char") || type_name<std::string>() == "std::string");

// Qualifiers are part of the name
static_assert(type_name<const int&>() != type_name<int>());
static_assert(type_name<int*>() == "int *" || type_name<int*>() == "int*");

// Constant, noexcept, and the same from a value
static_assert(noexcept(type_name<int>()));
constexpr int some_value = 0;
static_assert(type_name(some_value) == "int");

// At run time: GCC's -fsanitize=undefined refuses string_view::find on static storage in a constant expression
TEST(TypeName, InlineNamespacesAreRemoved)
{
    EXPECT_EQ(type_name<std::string>().find("__1"), std::string_view::npos);
    EXPECT_EQ(type_name<std::string>().find("__cxx11"), std::string_view::npos);
}

TEST(TypeName, LongNamesAreNotTruncated)
{
    using tall = demo_names::box<demo_names::box<demo_names::box<demo_names::box<demo_names::box<demo_names::box<int>>>>>>;
    using wide = demo_names::two<demo_names::two<demo_names::two<tall, tall>, demo_names::two<tall, tall>>,
                                 demo_names::two<demo_names::two<tall, tall>, demo_names::two<tall, tall>>>;
    EXPECT_GT(type_name<wide>().size(), 512u);
    EXPECT_NE(type_name<wide>().find("box<int"), std::string_view::npos);
}

TEST(TypeName, ViewsAreStableAndDistinct)
{
    const std::string_view a = type_name<std::vector<int>>();
    const std::string_view b = type_name<std::vector<double>>();
    EXPECT_NE(a, b);
    EXPECT_TRUE(a.starts_with("std::vector<int"));
    EXPECT_TRUE(b.starts_with("std::vector<double"));
    EXPECT_EQ(a.data(), type_name<std::vector<int>>().data());   // static storage
}

} // namespace
