//
//  forward_like_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

#include <snicholls/core/detail/forward_like.hpp>

#include <gtest/gtest.h>

#include <string>
#include <type_traits>
#include <utility>

namespace {

using snicholls::detail::like_t;

struct owner {
    std::string member;
};

// T as deduced for a forwarding reference: X& for lvalues, X for rvalues
static_assert(std::is_same_v<like_t<owner&, std::string>, std::string&>);
static_assert(std::is_same_v<like_t<const owner&, std::string>, const std::string&>);
static_assert(std::is_same_v<like_t<owner, std::string>, std::string&&>);
static_assert(std::is_same_v<like_t<const owner, std::string>, const std::string&&>);

// An explicit rvalue reference is an rvalue too
static_assert(std::is_same_v<like_t<owner&&, std::string>, std::string&&>);
static_assert(std::is_same_v<like_t<const owner&&, std::string>, const std::string&&>);

// The constness of U is kept: const is only ever added, as by std::forward_like
static_assert(std::is_same_v<like_t<owner&, const int>, const int&>);
static_assert(std::is_same_v<like_t<owner, const int>, const int&&>);

// The value category of U itself is ignored; only T's matters
static_assert(std::is_same_v<like_t<owner&, int&&>, int&>);
static_assert(std::is_same_v<like_t<owner, int&>, int&>);   // reference collapsing: int& && is int&

// The use it exists for: forward a member with the owner's value category
template <typename O>
like_t<O, std::string> member_of(O&& o)
{
    return static_cast<like_t<O, std::string>>(o.member);
}

TEST(DetailForwardLike, LvalueOwnerGivesAnLvalueMember)
{
    owner o{"kept"};
    std::string& m = member_of(o);
    EXPECT_EQ(&m, &o.member);
}

TEST(DetailForwardLike, ConstOwnerGivesAConstMember)
{
    const owner o{"kept"};
    static_assert(std::is_same_v<decltype(member_of(o)), const std::string&>);
    EXPECT_EQ(&member_of(o), &o.member);
}

TEST(DetailForwardLike, RvalueOwnerLetsTheMemberBeMovedFrom)
{
    owner o{std::string(100, 'x')};
    std::string taken = member_of(std::move(o));
    EXPECT_EQ(taken.size(), 100u);
    EXPECT_TRUE(o.member.empty()); // NOLINT(bugprone-use-after-move): moved-from is valid but empty
}

} // namespace
