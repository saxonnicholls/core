//
//  loop_clock_conformance_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/interfaces/loop_clock_conformance.hpp>: the kit passes the reference and catches broken
//  clocks: one that goes backwards, one that blocks for a due deadline, one that moves without waking the loop.
//

#include <snicholls/core/interfaces/loop_clock_conformance.hpp>
#include <snicholls/core/interfaces/loop_clock_reference.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <functional>
#include <string>
#include <vector>

using namespace snicholls;
using namespace std::chrono_literals;

namespace {

enum class defect { backwards, blocks_when_due, silent };

template <defect D>
struct broken_clock {
    manual_loop_clock inner{loop_time_point{1h}};
    mutable int reads = 0;
    loop_time_point now() const noexcept
    {
        ++reads;
        return D == defect::backwards && reads % 2 == 0 ? inner.now() - 1ns : inner.now();
    }
    long long block_for(loop_time_point t) const noexcept { return D == defect::blocks_when_due ? 5 : inner.block_for(t); }
    void set_waker(std::function<void()> w)
    {
        if (D != defect::silent)
            inner.set_waker(std::move(w));
    }
};

bool has(const std::vector<std::string>& f, const std::string& what) { return std::find(f.begin(), f.end(), what) != f.end(); }

TEST(LoopClockConformance, TheReferencePasses)
{
    manual_loop_clock c;
    EXPECT_TRUE(check_loop_clock(c).empty());
    EXPECT_TRUE(check_loop_clock_moves(c, [&] { c.advance(1ms); }).empty());
}

TEST(LoopClockConformance, BrokenClocksAreCaught)
{
    broken_clock<defect::backwards> a;
    EXPECT_TRUE(has(check_loop_clock(a), "now never goes backwards"));
    broken_clock<defect::blocks_when_due> b;
    const auto fb = check_loop_clock(b);
    EXPECT_TRUE(has(fb, "a past deadline does not block"));
    EXPECT_TRUE(has(fb, "a deadline at now does not block"));
    broken_clock<defect::silent> c;
    const auto fc = check_loop_clock_moves(c, [&] { c.inner.advance(1ms); });
    EXPECT_TRUE(has(fc, "a step wakes the loop once"));
    EXPECT_EQ(fc.size(), 1u);
}

} // namespace
