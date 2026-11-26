//
//  random_source_conformance_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/interfaces/random_source_conformance.hpp>: the kit passes conforming sources, in every
//  form, and catches broken ones.
//

#include <snicholls/core/interfaces/random_source.hpp>
#include <snicholls/core/interfaces/random_source_conformance.hpp>
#include <snicholls/core/interfaces/random_source_reference.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <cstring>
#include <system_error>

namespace {

using namespace snicholls;

struct overruns {   // writes one byte past the end
    test_random_source inner{3};
    std::errc fill(std::byte* d, std::size_t n) noexcept { return n ? inner.fill(d, n + 1) : std::errc{}; }
};
struct constant {   // the same bytes every time
    std::errc fill(std::byte* d, std::size_t n) noexcept
    {
        for (std::size_t i = 0; i < n; ++i)
            d[i] = std::byte{0x42};
        return std::errc{};
    }
};
struct zeros {
    std::errc fill(std::byte* d, std::size_t n) noexcept
    {
        if (n)
            std::memset(d, 0, n);
        return std::errc{};
    }
};

TEST(RandomSourceConformance, ConformingSourcesPassInEveryForm)
{
    test_random_source value{9};
    EXPECT_TRUE(check_random_source(value).empty());
    random_source_adapter<test_random_source> adapted{test_random_source{9}};
    random_source_interface& virtual_form = adapted;
    EXPECT_TRUE(check_random_source(virtual_form).empty());
    any_random_source any = any_random_source::of(test_random_source{9});
    EXPECT_TRUE(check_random_source(any).empty());
}

TEST(RandomSourceConformance, TheKitCatchesBrokenSources)
{
    overruns o;
    EXPECT_FALSE(check_random_source(o).empty());
    constant c;
    EXPECT_FALSE(check_random_source(c).empty());
    zeros z;
    EXPECT_FALSE(check_random_source(z).empty());
    unavailable_random_source u;
    EXPECT_FALSE(check_random_source(u).empty());
}

TEST(RandomSourceConformance, FailsClosedIsCheckedToo)
{
    unavailable_random_source u;
    EXPECT_TRUE(check_random_source_fails_closed(u).empty());
    test_random_source t{1};
    EXPECT_FALSE(check_random_source_fails_closed(t).empty()) << "a working source does not fail closed";
}

} // namespace
