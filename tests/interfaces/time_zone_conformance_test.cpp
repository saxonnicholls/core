//
//  time_zone_conformance_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/interfaces/time_zone_conformance.hpp>: the kit passes the references and catches broken
//  zones: one whose intervals lie, one that calls a gap unique, one that swaps a fold's offsets.
//

#include <snicholls/core/interfaces/time_zone_conformance.hpp>

#include "contract_probes.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <string_view>
#include <utility>

using namespace snicholls;
using namespace std::chrono_literals;   // not all of std::chrono: libstdc++'s std::chrono::time_zone and local_info
                                        // would collide with core's
using std::chrono::sys_seconds;
using std::chrono::local_seconds;
using std::chrono::hours;

namespace {

enum class defect { short_interval, gap_is_unique, fold_swapped, no_name };

template <defect D>
struct broken_zone {
    two_rule_zone inner = sydney_like_zone();
    std::string_view name() const { return D == defect::no_name ? std::string_view{} : inner.name(); }
    zone_info info(sys_seconds t) const
    {
        zone_info i = inner.info(t);
        if (D == defect::short_interval)
            i.end = i.begin + hours{1};
        return i;
    }
    local_info info(local_seconds l) const
    {
        local_info li = inner.info(l);
        if (D == defect::gap_is_unique && li.kind == local_time_kind::nonexistent)
            li.kind = local_time_kind::unique;
        if (D == defect::fold_swapped && li.kind == local_time_kind::ambiguous)
            std::swap(li.first, li.second);
        return li;
    }
};

template <defect D>
struct broken_probe {
    broken_zone<D> z;
    auto& zone() { return z; }
    static sys_seconds from() { return test::zone_window_from; }
    static sys_seconds to() { return test::zone_window_to; }
};

TEST(TimeZoneConformance, TheReferencesPass)
{
    const auto a = run_time_zone_conformance<test::time_zone_probe<two_rule_zone, test::form::concept_model>>();
    const auto b = run_time_zone_conformance<test::time_zone_probe<fixed_offset_zone, test::form::concept_model>>();
    EXPECT_TRUE(a.ok()) << test::describe(a);
    EXPECT_TRUE(b.ok()) << test::describe(b);
}

TEST(TimeZoneConformance, AShortIntervalIsCaught)
{
    const auto r = run_time_zone_conformance<broken_probe<defect::short_interval>>();
    EXPECT_TRUE(r.failed("the rules hold at the instant")) << test::describe(r);
}

TEST(TimeZoneConformance, AGapCalledUniqueIsCaught)
{
    const auto r = run_time_zone_conformance<broken_probe<defect::gap_is_unique>>();
    EXPECT_TRUE(r.failed("a gap is nonexistent and resolves by policy")) << test::describe(r);
    EXPECT_FALSE(r.failed("a fold is ambiguous and resolves by policy"));
}

TEST(TimeZoneConformance, ASwappedFoldIsCaught)
{
    const auto r = run_time_zone_conformance<broken_probe<defect::fold_swapped>>();
    EXPECT_TRUE(r.failed("a fold is ambiguous and resolves by policy")) << test::describe(r);
}

TEST(TimeZoneConformance, ANamelessZoneIsCaught)
{
    const auto r = run_time_zone_conformance<broken_probe<defect::no_name>>();
    EXPECT_TRUE(r.failed("the name is stable")) << test::describe(r);
    EXPECT_EQ(r.failures.size(), 1u) << test::describe(r);
}

} // namespace
