//
//  message_sender_conformance_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/interfaces/message_sender_conformance.hpp>: the kit passes the reference and catches broken
//  implementations: one that leaks the provider's response and one that ignores idempotency keys.
//

#include <snicholls/core/interfaces/message_sender_conformance.hpp>

#include "capability_probes.hpp"

#include <gtest/gtest.h>

#include <optional>
#include <string>
#include <string_view>

using namespace snicholls;

namespace {

// The redaction rule's opposite: an implementation that copies the provider's response into its error.
capability_error leak(capability_errc code)
{
    return capability_error{code, *provider_code::make(capability_redaction_marker), {}};
}

// Sends through the reference, but leaks a provider's response, or forgets idempotency keys.
struct broken_sender {
    memory_message_sender inner;
    bool leaks = false;
    bool forgets_keys = false;
    bool armed = false;
    capability_result<send_receipt> send(message m)
    {
        if (armed && leaks) {
            armed = false;
            return err(leak(capability_errc::unavailable));
        }
        if (forgets_keys)
            m.idempotency_key = {};
        return inner.send(m);
    }
    [[nodiscard]] bool supports(channel c) const { return inner.supports(c); }
};

template <bool Leaks, bool Forgets>
struct broken_probe {
    broken_sender s{memory_message_sender{}, Leaks, Forgets};
    auto& sender() { return s; }
    void provider_fails(capability_errc c, std::string_view r)
    {
        s.armed = true;
        s.inner.fail_next(c, r);
    }
};

TEST(MessageSenderConformance, TheReferencePasses)
{
    const auto r = run_message_sender_conformance<test::message_sender_probe<test::form::concept_model>>();
    EXPECT_TRUE(r.ok()) << test::describe(r);
}

TEST(MessageSenderConformance, ALeakIsCaught)
{
    const auto r = run_message_sender_conformance<broken_probe<true, false>>();
    EXPECT_TRUE(r.failed("a provider failure never leaks its response")) << test::describe(r);
    EXPECT_EQ(r.failures.size(), 1u) << test::describe(r);
}

TEST(MessageSenderConformance, ForgottenIdempotencyKeysAreCaught)
{
    const auto r = run_message_sender_conformance<broken_probe<false, true>>();
    EXPECT_TRUE(r.failed("the same idempotency key sends once")) << test::describe(r);
}

} // namespace
