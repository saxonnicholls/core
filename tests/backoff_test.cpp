//
//  backoff_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  The schedule is a pure function, so the ceilings and the fraction form are static_asserts; the generator
//  form is checked against its bounds over many draws.
//

#include <snicholls/core/backoff.hpp>

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <random>

namespace {

using namespace std::chrono_literals;
using snicholls::backoff_ceiling;
using snicholls::backoff_delay;
using snicholls::backoff_policy;
using snicholls::next_attempt;

constexpr backoff_policy policy{.min = 100ms, .max = 3000ms, .session_grace = 5s};

// Doubling from min, capped at max
static_assert(backoff_ceiling(policy, 0) == 100ms);
static_assert(backoff_ceiling(policy, 1) == 200ms);
static_assert(backoff_ceiling(policy, 4) == 1600ms);
static_assert(backoff_ceiling(policy, 5) == 3000ms);   // 3200 would pass max
static_assert(backoff_ceiling(policy, 1000) == 3000ms);
static_assert(backoff_ceiling(policy, UINT32_MAX) == 3000ms);   // no overflow, no long loop

// A policy with max below min behaves as max == min
static_assert(backoff_ceiling(backoff_policy{.min = 500ms, .max = 10ms, .session_grace = 1s}, 3) == 10ms);

// The fraction form is exact and clamped
static_assert(backoff_delay(policy, 3, 0.0) == 0ms);
static_assert(backoff_delay(policy, 3, 0.5) == 400ms);   // ceiling 800
static_assert(backoff_delay(policy, 3, 1.0) == 800ms);
static_assert(backoff_delay(policy, 3, -4.0) == 0ms);
static_assert(backoff_delay(policy, 3, 7.0) == 800ms);

// The attempt count resets after a working session, else grows, and saturates
static_assert(next_attempt(policy, 7, 5s) == 0);
static_assert(next_attempt(policy, 7, 6h) == 0);
static_assert(next_attempt(policy, 7, 4999ms) == 8);
static_assert(next_attempt(policy, 0, 0ns) == 1);
static_assert(next_attempt(policy, UINT32_MAX, 1ms) == UINT32_MAX);

constexpr std::chrono::seconds one_second{1};   // the chrono literal operators are not noexcept; the functions are
static_assert(noexcept(backoff_ceiling(policy, 1)) && noexcept(backoff_delay(policy, 1, 0.5)) &&
              noexcept(next_attempt(policy, 1, one_second)));

TEST(BackoffDelay, DrawsWithinTheCeilingAndUsesTheWholeWindow)
{
    std::mt19937_64 rng{42};
    std::chrono::milliseconds lowest = 1h, highest = 0ms;
    for (int i = 0; i < 2000; ++i) {
        const auto d = backoff_delay(policy, 3, rng);   // ceiling 800ms
        EXPECT_GE(d, 0ms);
        EXPECT_LE(d, 800ms);
        lowest = std::min(lowest, d);
        highest = std::max(highest, d);
    }
    EXPECT_LT(lowest, 50ms);     // full jitter reaches the bottom of the window
    EXPECT_GT(highest, 750ms);   // and the top
}

TEST(BackoffDelay, AttemptZeroIsBoundedByMin)
{
    std::mt19937 rng{7};
    for (int i = 0; i < 200; ++i)
        EXPECT_LE(backoff_delay(policy, 0, rng), policy.min);
}

TEST(BackoffSchedule, AReconnectLoopGrowsThenResets)
{
    std::uint32_t attempt = 0;
    attempt = next_attempt(policy, attempt, 10ms);   // dropped at once
    attempt = next_attempt(policy, attempt, 10ms);
    EXPECT_EQ(attempt, 2u);
    EXPECT_EQ(backoff_ceiling(policy, attempt), 400ms);
    attempt = next_attempt(policy, attempt, 1min);   // a session that worked
    EXPECT_EQ(attempt, 0u);
    EXPECT_EQ(backoff_ceiling(policy, attempt), 100ms);
}

} // namespace
