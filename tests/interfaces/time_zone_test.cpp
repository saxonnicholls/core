//
//  time_zone_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/interfaces/time_zone.hpp>: the conformance kit passes on both references in every form (the
//  concept, the virtual interface over a borrowing adapter, the handle), and the adapters forward every operation.
//

#include <snicholls/core/interfaces/time_zone.hpp>
#include <snicholls/core/interfaces/time_zone_conformance.hpp>

#include "contract_probes.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <type_traits>

using namespace snicholls;
using namespace std::chrono;

namespace {

static_assert(std::is_abstract_v<time_zone_interface>);

TEST(TimeZoneInterface, TheKitPassesInEveryForm)
{
    using test::form;
    const auto a = run_time_zone_conformance<test::time_zone_probe<two_rule_zone, form::concept_model>>();
    const auto b = run_time_zone_conformance<test::time_zone_probe<two_rule_zone, form::interface>>();
    const auto c = run_time_zone_conformance<test::time_zone_probe<two_rule_zone, form::ref>>();
    const auto d = run_time_zone_conformance<test::time_zone_probe<fixed_offset_zone, form::concept_model>>();
    const auto e = run_time_zone_conformance<test::time_zone_probe<fixed_offset_zone, form::interface>>();
    const auto f = run_time_zone_conformance<test::time_zone_probe<fixed_offset_zone, form::ref>>();
    for (const auto* r : {&a, &b, &c, &d, &e, &f}) {
        EXPECT_TRUE(r->ok()) << test::describe(*r);
        EXPECT_EQ(r->checks, 6u);
    }
}

// Counts every call, to see the adapters forward.
struct counting_zone {
    mutable int calls = 0;
    std::string_view name() const { ++calls; return "counting"; }
    zone_info info(sys_seconds) const { ++calls; return {}; }
    local_info info(local_seconds) const { ++calls; return {}; }
};

TEST(TimeZoneInterface, TheAdaptersForwardEveryOperation)
{
    counting_zone impl;
    time_zone_adapter<counting_zone&> borrowed{impl};
    time_zone_ref z{borrowed};
    EXPECT_EQ(z.name(), "counting");
    EXPECT_EQ(z.info(sys_seconds{}), zone_info{});
    EXPECT_EQ(z.info(local_seconds{}).kind, local_time_kind::unique);
    EXPECT_EQ(impl.calls, 3);
    time_zone_adapter owned{counting_zone{}};   // deduced: owns its copy
    static_assert(std::is_same_v<decltype(owned), time_zone_adapter<counting_zone>>);
    EXPECT_EQ(owned.name(), "counting");
    EXPECT_EQ(owned.get().calls, 1);
}

TEST(TimeZoneInterface, TheConversionsTakeTheInterface)
{
    const time_zone_adapter sydney{sydney_like_zone()};
    const time_zone_interface& z = sydney;
    const utc_s t{sys_days{year{2026} / January / 15}};
    EXPECT_EQ(offset_at(z, t), hours{11});
    EXPECT_EQ(*from_local(z, to_local(z, t)), t);
}

} // namespace
