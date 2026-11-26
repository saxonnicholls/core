//
//  uuid_generator_reference_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/interfaces/uuid_generator_reference.hpp>: determinism, the bits, monotonic version 7 through
//  counter overflow and a clock that steps back, failing sources and the clock's range, at compile time too.
//

#include <snicholls/core/interfaces/uuid_generator_reference.hpp>

#include "contract_probes.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <optional>
#include <system_error>

using namespace snicholls;

namespace {

constexpr utc_ms start = utc_ms::from_unix_millis(1'800'000'000'000);

constexpr bool v7_is_composed_at_compile_time()
{
    auto g = make_reference_uuid_generator(7, start);
    const auto a = g.v7();
    const auto b = g.v7();
    return a && b && *a < *b && a->version() == 7 && b->timestamp() == a->timestamp();
}
static_assert(v7_is_composed_at_compile_time());

TEST(ReferenceUuidGenerator, TheSameSeedAndStartMakeTheSameIdentifiers)
{
    auto a = make_reference_uuid_generator(99, start);
    auto b = make_reference_uuid_generator(99, start);
    auto c = make_reference_uuid_generator(100, start);
    for (int i = 0; i < 10; ++i) {
        const auto x = a.v4();
        EXPECT_EQ(x, b.v4());
        EXPECT_NE(*x, *c.v4());
        EXPECT_EQ(a.v7(), b.v7());
    }
}

TEST(ReferenceUuidGenerator, V4HasItsVersionAndVariant)
{
    auto g = make_reference_uuid_generator(3, start);
    const uuid u = *g.v4();
    EXPECT_EQ(u.version(), 4u);
    EXPECT_EQ(u.variant(), uuid_variant::rfc9562);
}

TEST(ReferenceUuidGenerator, V7FollowsTheClockAndStaysMonotonic)
{
    auto g = make_reference_uuid_generator(3, start);
    const uuid first = *g.v7();
    EXPECT_EQ(first.timestamp()->time_since_epoch().count(), start.unix_millis());
    uuid last = first;
    for (int i = 0; i < 4096; ++i) {   // overflows the 12-bit counter at least once
        const uuid next = *g.v7();
        ASSERT_LT(last, next);
        last = next;
    }
    EXPECT_GT(last.timestamp(), first.timestamp());   // the overflow borrowed a millisecond
    g.clock().advance(std::chrono::seconds{1});
    const uuid later = *g.v7();
    EXPECT_EQ(later.timestamp()->time_since_epoch().count(), start.unix_millis() + 1000);
}

TEST(ReferenceUuidGenerator, AClockThatStepsBackDoesNotReorder)
{
    test::test_uuid_generator g{test::failing_random{}, test::settable_clock{}};
    const uuid a = *g.v7();
    *g.clock().t -= std::chrono::hours{1};
    const uuid b = *g.v7();
    EXPECT_LT(a, b);
    EXPECT_EQ(b.timestamp(), a.timestamp());
}

TEST(ReferenceUuidGenerator, AFailedSourceIsUnavailableAndChangesNothing)
{
    test::test_uuid_generator g{test::failing_random{}, test::settable_clock{}};
    const uuid a = *g.v7();
    *g.random().fails = true;
    EXPECT_EQ(g.v4().unwrap_err().code, capability_errc::unavailable);
    EXPECT_EQ(g.v7().unwrap_err().code, capability_errc::unavailable);
    *g.random().fails = false;
    EXPECT_LT(a, *g.v7());
}

TEST(ReferenceUuidGenerator, AClockOutsideVersion7sRangeIsInvalid)
{
    test::test_uuid_generator g{test::failing_random{}, test::settable_clock{}};
    *g.clock().t = utc_ms::from_unix_millis(-1);
    EXPECT_EQ(g.v7().unwrap_err().code, capability_errc::invalid_argument);
    *g.clock().t = utc_ms::from_unix_millis(std::int64_t{1} << 48);
    EXPECT_EQ(g.v7().unwrap_err().code, capability_errc::invalid_argument);
    *g.clock().t = utc_ms::from_unix_millis((std::int64_t{1} << 48) - 1);
    EXPECT_TRUE(g.v7().is_ok());
    EXPECT_TRUE(g.v4().is_ok());   // v4 needs no clock
}

// The shape of a random_source_like source (concepts/random_source.hpp): fill, failing closed.
struct fill_source {
    stats::wyrand r{5};
    bool fails = false;
    std::errc fill(std::byte* data, std::size_t size) noexcept
    {
        if (fails)
            return std::errc::resource_unavailable_try_again;
        for (std::size_t i = 0; i < size; ++i)
            data[i] = static_cast<std::byte>(r());
        return {};
    }
};
static_assert(uuid_random_source<fill_source>);

TEST(ReferenceUuidGenerator, DrawsFromARandomSource)
{
    basic_uuid_generator g{fill_source{}, manual_utc_clock{start}};
    EXPECT_EQ(g.v4()->version(), 4u);
    EXPECT_EQ(g.v7()->version(), 7u);
    g.random().fails = true;
    EXPECT_EQ(g.v4().unwrap_err().code, capability_errc::unavailable);
    EXPECT_EQ(g.v7().unwrap_err().code, capability_errc::unavailable);
}

} // namespace
