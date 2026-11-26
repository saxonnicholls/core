//
//  encryptor_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/interfaces/encryptor.hpp>: the conformance kit passes on the test double in every form (the
//  concept, the virtual interface over a borrowing adapter, the handle), and the adapters forward every operation.
//

#include <snicholls/core/interfaces/encryptor.hpp>
#include <snicholls/core/interfaces/encryptor_conformance.hpp>

#include "contract_probes.hpp"

#include <gtest/gtest.h>

#include <type_traits>

using namespace snicholls;

namespace {

static_assert(std::is_abstract_v<encryptor_interface>);

TEST(EncryptorInterface, TheKitPassesInEveryForm)
{
    const auto direct = run_encryptor_conformance<test::encryptor_probe<test::form::concept_model>>();
    const auto virtual_form = run_encryptor_conformance<test::encryptor_probe<test::form::interface>>();
    const auto handle = run_encryptor_conformance<test::encryptor_probe<test::form::ref>>();
    for (const auto* r : {&direct, &virtual_form, &handle}) {
        EXPECT_TRUE(r->ok()) << test::describe(*r);
        EXPECT_EQ(r->checks, 8u);
    }
}

// Counts every call, to see the adapters forward.
struct counting_encryptor {
    mutable int calls = 0;
    capability_result<std::vector<std::byte>> encrypt(std::string_view, std::span<const std::byte>, std::span<const std::byte>)
    {
        ++calls;
        return ok(std::vector<std::byte>{std::byte{1}});
    }
    capability_result<std::vector<std::byte>> decrypt(std::span<const std::byte>, std::span<const std::byte>) const
    {
        ++calls;
        return fail(capability_errc::signature_invalid);
    }
    std::string_view algorithm() const { ++calls; return "counting"; }
};

TEST(EncryptorInterface, TheAdaptersForwardEveryOperation)
{
    counting_encryptor impl;
    encryptor_adapter<counting_encryptor&> borrowed{impl};
    encryptor_ref e{borrowed};
    EXPECT_EQ(e.encrypt("k", {}, {})->size(), 1u);
    EXPECT_EQ(e.decrypt({}, {}).unwrap_err().code, capability_errc::signature_invalid);
    EXPECT_EQ(e.algorithm(), "counting");
    EXPECT_EQ(impl.calls, 3);
    encryptor_adapter owned{counting_encryptor{}};   // deduced: owns its copy
    static_assert(std::is_same_v<decltype(owned), encryptor_adapter<counting_encryptor>>);
    EXPECT_EQ(owned.algorithm(), "counting");
    EXPECT_EQ(owned.get().calls, 1);
}

} // namespace
