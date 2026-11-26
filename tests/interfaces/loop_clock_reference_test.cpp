//
//  loop_clock_reference_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/interfaces/loop_clock_reference.hpp>: moves only on command, never backwards, wakes on
//  every move, never asks to block.
//

#include <snicholls/core/interfaces/loop_clock_reference.hpp>

#include <gtest/gtest.h>

#include <chrono>

using namespace snicholls;
using namespace std::chrono_literals;

namespace {

TEST(ManualLoopClock, StartsWhereToldAndStaysThere)
{
    const manual_loop_clock c{loop_time_point{1h}};
    EXPECT_EQ(c.now(), loop_time_point{1h});
    EXPECT_EQ(c.now(), c.now());
    EXPECT_EQ(manual_loop_clock{}.now(), loop_time_point{});
}

TEST(ManualLoopClock, MovesOnlyForwards)
{
    manual_loop_clock c;
    c.advance(10ms);
    c.advance(-5ms);
    c.advance_to(loop_time_point{1ms});
    EXPECT_EQ(c.now(), loop_time_point{10ms});
    c.advance_to(loop_time_point{1s});
    EXPECT_EQ(c.now(), loop_time_point{1s});
}

TEST(ManualLoopClock, WakesTheLoopWhenItMoves)
{
    manual_loop_clock c;
    int woken = 0;
    c.set_waker([&woken] { ++woken; });
    c.advance(1ms);
    c.advance_to(loop_time_point{1s});
    c.advance(-1ms);   // ignored, and no wake
    EXPECT_EQ(woken, 2);
    c.set_waker({});
    c.advance(1ms);
    EXPECT_EQ(woken, 2);
}

TEST(ManualLoopClock, NeverAsksThePollerToBlock)
{
    const manual_loop_clock c;
    EXPECT_EQ(c.block_for(loop_time_point{1h}), 0);
    EXPECT_EQ(c.block_for(loop_time_point{}), 0);
}

} // namespace
