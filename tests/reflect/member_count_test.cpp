//
//  member_count_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/reflect/member_count.hpp>.
//

#include <any>
#include <array>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include <snicholls/core/reflect/member_count.hpp>

namespace {

struct empty_s {};
struct one { int a; };
struct basic { int a; double b; std::string c; };
struct std_members { std::optional<int> a; std::vector<int> b; std::variant<int, std::string> c; std::map<int, int> d; };
struct with_any { std::any a; int b; };
struct inner_s { int x; int y; };
struct nested { inner_s in; int z; };
struct with_array { std::array<int, 3> arr; int x; };
struct move_only { std::unique_ptr<int> p; const char* s; };
struct with_const { const int a; int b; };
struct opt_nested { std::optional<inner_s> o; int z; };
struct with_defaults { int a = 1; std::string b = "x"; };
struct with_ctor { with_ctor(int v) : a(v) {} int a; };   // a type with a user constructor: not an aggregate

TEST(MemberCount, CountsScalarAndClassMembers)
{
    static_assert(snicholls::member_count_v<one> == 1);
    static_assert(snicholls::member_count_v<basic> == 3);
}

TEST(MemberCount, EmptyStructHasZero)
{
    static_assert(snicholls::member_count_v<empty_s> == 0);
}

TEST(MemberCount, StandardLibraryMembersWithGreedyConstructors)
{
    static_assert(snicholls::member_count_v<std_members> == 4);
    static_assert(snicholls::member_count_v<with_any> == 2);
    static_assert(snicholls::member_count_v<opt_nested> == 2);
}

TEST(MemberCount, NestedAggregatesAndStdArrayCountAsOne)
{
    static_assert(snicholls::member_count_v<nested> == 2);
    static_assert(snicholls::member_count_v<with_array> == 2);
}

TEST(MemberCount, MoveOnlyAndConstMembers)
{
    static_assert(snicholls::member_count_v<move_only> == 2);
    static_assert(snicholls::member_count_v<with_const> == 2);
}

TEST(MemberCount, DefaultMemberInitialisers)
{
    static_assert(snicholls::member_count_v<with_defaults> == 2);
}

TEST(MemberCount, IgnoresCvAndReferences)
{
    static_assert(snicholls::member_count_v<const basic&> == 3);
    static_assert(snicholls::member_count<basic>::value == snicholls::member_count_v<basic>);
}

TEST(PlainAggregateConcept, AcceptsAggregatesRejectsOthers)
{
    static_assert(snicholls::plain_aggregate<basic>);
    static_assert(snicholls::plain_aggregate<empty_s>);
    static_assert(!snicholls::plain_aggregate<int>);
    static_assert(!snicholls::plain_aggregate<std::string>);
    static_assert(!snicholls::plain_aggregate<int[3]>);
    union u { int a; float b; };
    static_assert(!snicholls::plain_aggregate<u>);
    static_assert(!snicholls::plain_aggregate<with_ctor>);   // user-declared constructors
}

TEST(MemberCount, LargeStruct)
{
    struct big { int m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15,
                 m16, m17, m18, m19, m20, m21, m22, m23, m24, m25, m26, m27, m28, m29, m30, m31, m32; };
    static_assert(snicholls::member_count_v<big> == 33);
}

} // namespace
