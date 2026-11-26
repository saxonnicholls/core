//
//  loop_clock_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/concepts/loop_clock.hpp>: the trait and the concept agree on the models and the non-models,
//  at compile time.
//

#include <snicholls/core/concepts/loop_clock.hpp>
#include <snicholls/core/interfaces/loop_clock.hpp>
#include <snicholls/core/interfaces/loop_clock_reference.hpp>

#include <gtest/gtest.h>

#include <chrono>
#include <functional>

using namespace snicholls;

namespace {

template <typename C>
constexpr bool both = is_loop_clock_like_v<C> && loop_clock_like<C>;
template <typename C>
constexpr bool neither = !is_loop_clock_like_v<C> && !loop_clock_like<C>;

static_assert(both<manual_loop_clock>);
static_assert(both<loop_clock_interface>);
static_assert(both<loop_clock_adapter<manual_loop_clock>>);
static_assert(both<loop_clock_adapter<manual_loop_clock&>>);
static_assert(both<any_loop_clock>);

struct now_may_throw {
    loop_time_point now() const;
    long long block_for(loop_time_point) const noexcept;
    void set_waker(std::function<void()>);
};
struct block_for_nanoseconds {
    loop_time_point now() const noexcept;
    std::chrono::nanoseconds block_for(loop_time_point) const noexcept;
    void set_waker(std::function<void()>);
};
struct system_time_point {
    std::chrono::system_clock::time_point now() const noexcept;
    long long block_for(loop_time_point) const noexcept;
    void set_waker(std::function<void()>);
};
struct no_waker {
    loop_time_point now() const noexcept;
    long long block_for(loop_time_point) const noexcept;
};
static_assert(neither<now_may_throw>);
static_assert(neither<block_for_nanoseconds>);
static_assert(neither<system_time_point>);
static_assert(neither<no_waker>);
static_assert(neither<int>);

template <SNICHOLLS_LOOP_CLOCK Clock>
bool due(const Clock& c, loop_time_point deadline) noexcept
{
    return !(c.now() < deadline);
}

TEST(LoopClockConcept, TheMacroConstrainsATemplate)
{
    manual_loop_clock c;
    EXPECT_TRUE(due(c, loop_time_point{}));
    EXPECT_FALSE(due(c, loop_time_point{} + std::chrono::seconds{1}));
}

} // namespace
