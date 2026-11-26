//
//  uuid_generator_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/interfaces/uuid_generator.hpp>: the conformance kit passes on the reference in every form
//  (the concept, the virtual interface over a borrowing adapter, the handle), and the adapters forward every operation.
//

#include <snicholls/core/interfaces/uuid_generator.hpp>
#include <snicholls/core/interfaces/uuid_generator_conformance.hpp>

#include "contract_probes.hpp"

#include <gtest/gtest.h>

#include <type_traits>

using namespace snicholls;

namespace {

static_assert(std::is_abstract_v<uuid_generator_interface>);

TEST(UuidGeneratorInterface, TheKitPassesInEveryForm)
{
    const auto direct = run_uuid_generator_conformance<test::uuid_generator_probe<test::form::concept_model>>();
    const auto virtual_form = run_uuid_generator_conformance<test::uuid_generator_probe<test::form::interface>>();
    const auto handle = run_uuid_generator_conformance<test::uuid_generator_probe<test::form::ref>>();
    for (const auto* r : {&direct, &virtual_form, &handle}) {
        EXPECT_TRUE(r->ok()) << test::describe(*r);
        EXPECT_EQ(r->checks, 5u);
    }
}

// Counts every call, to see the adapters forward.
struct counting_generator {
    int calls = 0;
    capability_result<uuid> v4() { ++calls; return ok(uuid::nil()); }
    capability_result<uuid> v7() { ++calls; return fail(capability_errc::unavailable); }
};

TEST(UuidGeneratorInterface, TheAdaptersForwardEveryOperation)
{
    counting_generator impl;
    uuid_generator_adapter<counting_generator&> borrowed{impl};
    uuid_generator_ref g{borrowed};
    EXPECT_TRUE(g.v4()->is_nil());
    EXPECT_EQ(g.v7().unwrap_err().code, capability_errc::unavailable);
    EXPECT_EQ(impl.calls, 2);
    uuid_generator_adapter owned{counting_generator{}};   // deduced: owns its copy
    static_assert(std::is_same_v<decltype(owned), uuid_generator_adapter<counting_generator>>);
    static_cast<void>(owned.v4());
    EXPECT_EQ(owned.get().calls, 1);
}

} // namespace
