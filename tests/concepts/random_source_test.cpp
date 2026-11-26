//
//  random_source_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/concepts/random_source.hpp>.
//

#include <snicholls/core/concepts/random_source.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <system_error>

namespace {

struct good {
    std::errc fill(std::byte*, std::size_t) noexcept { return std::errc{}; }
};
struct throws {   // not noexcept: a source must not throw
    std::errc fill(std::byte*, std::size_t) { return std::errc{}; }
};
struct wrong_return {
    bool fill(std::byte*, std::size_t) noexcept { return true; }
};
struct const_only {   // fill on a const object only is still callable on a mutable one
    std::errc fill(std::byte*, std::size_t) const noexcept { return std::errc{}; }
};
struct no_member {};

static_assert(snicholls::is_random_source_like_v<good>);
static_assert(snicholls::is_random_source_like_v<const_only>);
static_assert(!snicholls::is_random_source_like_v<throws>, "a source that may throw does not model the contract");
static_assert(!snicholls::is_random_source_like_v<wrong_return>, "fill returns std::errc");
static_assert(!snicholls::is_random_source_like_v<no_member>);
static_assert(!snicholls::is_random_source_like_v<int>);
#if defined(__cpp_concepts)
static_assert(snicholls::random_source_like<good> && !snicholls::random_source_like<throws>);
#endif

TEST(RandomSourceConcept, TheTraitAndTheConceptAgree)
{
    EXPECT_TRUE(snicholls::is_random_source_like_v<good>);
    EXPECT_FALSE(snicholls::is_random_source_like_v<wrong_return>);
}

} // namespace
