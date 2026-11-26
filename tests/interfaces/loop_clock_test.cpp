//
//  loop_clock_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/interfaces/loop_clock.hpp>: the kit passes on the reference in every form, the adapters
//  forward every member (set_waker included, so the model wakes the loop), and the base keeps a waker for wake().
//

#include <snicholls/core/interfaces/loop_clock.hpp>
#include <snicholls/core/interfaces/loop_clock_conformance.hpp>
#include <snicholls/core/interfaces/loop_clock_reference.hpp>

#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <type_traits>

using namespace snicholls;
using namespace std::chrono_literals;

namespace {

static_assert(std::is_abstract_v<loop_clock_interface>);

TEST(LoopClockInterface, TheKitPassesInEveryForm)
{
    manual_loop_clock model;
    EXPECT_TRUE(check_loop_clock_moves(model, [&] { model.advance(1ms); }).empty());
    loop_clock_adapter<manual_loop_clock> owned;
    loop_clock_interface& virtual_form = owned;
    EXPECT_TRUE(check_loop_clock_moves(virtual_form, [&] { owned.implementation().advance(1ms); }).empty());
    manual_loop_clock viewed;
    loop_clock_adapter<manual_loop_clock&> view{viewed};
    loop_clock_interface& view_form = view;
    EXPECT_TRUE(check_loop_clock_moves(view_form, [&] { viewed.advance_to(viewed.now() + 1s); }).empty());
    any_loop_clock value = any_loop_clock::of(manual_loop_clock{});
    EXPECT_TRUE(check_loop_clock(value).empty());
}

// A clock written against the interface, as ts-moveables' are: it wakes through the base.
class stepping_clock final : public loop_clock_interface {
public:
    time_point now() const noexcept override { return t_; }
    long long block_for(time_point) const noexcept override { return 0; }
    void step()
    {
        t_ += 1ms;
        wake();
    }

private:
    time_point t_{};
};

TEST(LoopClockInterface, TheBaseKeepsTheWaker)
{
    stepping_clock c;
    int woken = 0;
    c.set_waker([&woken] { ++woken; });
    c.step();
    EXPECT_EQ(woken, 1);
    c.set_waker({});
    c.step();   // no waker: nothing to call
    EXPECT_EQ(woken, 1);
    EXPECT_TRUE(check_loop_clock_moves(c, [&] { c.step(); }).empty());
}

TEST(LoopClockInterface, TheAdapterForwardsSetWaker)
{
    manual_loop_clock model{loop_time_point{5s}};
    loop_clock_adapter<manual_loop_clock&> view{model};
    int woken = 0;
    view.set_waker([&woken] { ++woken; });
    model.advance(1s);
    EXPECT_EQ(woken, 1);
    EXPECT_EQ(view.now(), loop_time_point{6s});
    EXPECT_EQ(view.block_for(loop_time_point{1h}), 0);
    EXPECT_EQ(&view.implementation(), &model);
}

} // namespace
