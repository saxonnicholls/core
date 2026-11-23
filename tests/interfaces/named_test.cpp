//
//  named_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/interfaces/named.hpp>.
//

#include <snicholls/core/interfaces/named.hpp>
#include <snicholls/core/interfaces/named_conformance.hpp>
#include <snicholls/core/money/iso4217.hpp>

#include <gtest/gtest.h>

#include <array>
#include <memory>

namespace {

using namespace snicholls;

constexpr named_model<currency::AUD> aud_model{};
static_assert(aud_model.name() == "Australian Dollar", "the adapter is usable in constant expressions");

TEST(NamedInterface, AdaptsAnyNamedTypeForRunTimeChoice)
{
    const std::array<std::unique_ptr<named_interface>, 2> all{std::make_unique<named_model<currency::AUD>>(),
                                                                std::make_unique<named_model<currency::JPY>>()};
    EXPECT_EQ(all[0]->name(), "Australian Dollar");
    EXPECT_EQ(all[1]->name(), "Yen");
}

TEST(NamedInterface, TheKitPassesThroughBothForms)
{
    EXPECT_FALSE(check_named(currency::AUD{}));
    const named_interface& virtual_form = aud_model;
    EXPECT_FALSE(check_named(virtual_form));
}

} // namespace
