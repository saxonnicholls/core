//
//  uuid_generator_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/concepts/uuid_generator.hpp>: the models and the non-models, at compile time.
//

#include <snicholls/core/concepts/uuid_generator.hpp>
#include <snicholls/core/interfaces/uuid_generator_reference.hpp>

#include <gtest/gtest.h>

#include <optional>

using namespace snicholls;

namespace {

static_assert(uuid_generator<reference_uuid_generator>);
static_assert(uuid_generator<uuid_generator_interface>);
static_assert(uuid_generator<uuid_generator_adapter<reference_uuid_generator>>);
static_assert(uuid_generator<uuid_generator_adapter<reference_uuid_generator&>>);
static_assert(uuid_generator<uuid_generator_ref>);

struct only_v4 {
    capability_result<uuid> v4();
};
struct returns_optional {
    std::optional<uuid> v4();
    std::optional<uuid> v7();
};
struct const_only {
    capability_result<uuid> v4() const;
    capability_result<uuid> v7() const;
};
static_assert(!uuid_generator<only_v4>);
static_assert(!uuid_generator<returns_optional>);
static_assert(uuid_generator<const_only>);   // const members are callable on a non-const generator
static_assert(!uuid_generator<int>);

TEST(UuidGeneratorConcept, AGeneratorIsUsedThroughTheConcept)
{
    auto g = make_reference_uuid_generator(1, utc_ms::from_unix_millis(1'700'000'000'000));
    const auto id = g.v7();
    ASSERT_TRUE(id.is_ok());
    EXPECT_EQ(id->version(), 7u);
    EXPECT_EQ(id->timestamp()->time_since_epoch().count(), 1'700'000'000'000);
}

} // namespace
