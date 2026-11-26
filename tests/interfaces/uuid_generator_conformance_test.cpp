//
//  uuid_generator_conformance_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/interfaces/uuid_generator_conformance.hpp>: the kit passes the reference and catches broken
//  generators: one with a constant counter, one that follows the clock back, one that hides a failed source.
//

#include <snicholls/core/interfaces/uuid_generator_conformance.hpp>

#include "contract_probes.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>

using namespace snicholls;

namespace {

enum class defect { repeats, follows_clock_back, hides_failure, wrong_version };

template <defect D>
struct broken_generator {
    test::failing_random random;
    test::settable_clock clock;
    capability_result<uuid> v4()
    {
        if (D == defect::hides_failure && *random.fails)
            return ok(uuid::nil());
        const auto hi = random();
        if (!hi)
            return fail(capability_errc::unavailable);
        std::array<std::uint8_t, 16> b{};
        for (std::size_t i = 0; i < 8; ++i)
            b[i] = b[8 + i] = static_cast<std::uint8_t>(*hi >> (8 * i));
        uuid u = make_uuid_v4(b);
        if (D == defect::wrong_version)
            u.bytes[6] = static_cast<std::uint8_t>((u.bytes[6] & 0x0F) | 0x10);
        return ok(u);
    }
    capability_result<uuid> v7()
    {
        if (D == defect::hides_failure && *random.fails)
            return ok(uuid::nil());
        const auto r = random();
        if (!r)
            return fail(capability_errc::unavailable);
        const auto ms = static_cast<std::uint64_t>(clock.now().unix_millis());
        // repeats: a fixed rand_a and rand_b; follows_clock_back: random bits, no memory of the last one
        return ok(D == defect::repeats ? make_uuid_v7(ms, 0, 0) : make_uuid_v7(ms, static_cast<std::uint16_t>(*r), *r));
    }
};

template <defect D>
struct broken_probe {
    broken_generator<D> g;
    auto& generator() { return g; }
    std::int64_t clock_ms() { return g.clock.now().unix_millis(); }
    void clock_steps_back() { *g.clock.t -= std::chrono::seconds{1}; }
    void randomness_fails() { *g.random.fails = true; }
};

TEST(UuidGeneratorConformance, TheReferencePasses)
{
    const auto r = run_uuid_generator_conformance<test::uuid_generator_probe<test::form::concept_model>>();
    EXPECT_TRUE(r.ok()) << test::describe(r);
}

TEST(UuidGeneratorConformance, ARepeatingV7IsCaught)
{
    const auto r = run_uuid_generator_conformance<broken_probe<defect::repeats>>();
    EXPECT_TRUE(r.failed("v7 strictly increases within a millisecond")) << test::describe(r);
}

TEST(UuidGeneratorConformance, FollowingTheClockBackIsCaught)
{
    const auto r = run_uuid_generator_conformance<broken_probe<defect::follows_clock_back>>();
    EXPECT_TRUE(r.failed("v7 never goes back with the clock")) << test::describe(r);
}

TEST(UuidGeneratorConformance, AHiddenFailureIsCaught)
{
    const auto r = run_uuid_generator_conformance<broken_probe<defect::hides_failure>>();
    EXPECT_TRUE(r.failed("a failed random source is unavailable")) << test::describe(r);
}

TEST(UuidGeneratorConformance, AWrongVersionIsCaught)
{
    const auto r = run_uuid_generator_conformance<broken_probe<defect::wrong_version>>();
    EXPECT_TRUE(r.failed("v4 is version 4 and distinct")) << test::describe(r);
}

} // namespace
