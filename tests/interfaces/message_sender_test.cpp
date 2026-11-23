//
//  message_sender_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/interfaces/message_sender.hpp>: the conformance kit passes on the reference in every form (the
//  concept, the virtual interface over a borrowing adapter, the handle), and the adapters forward every operation.
//

#include <snicholls/core/interfaces/message_sender.hpp>
#include <snicholls/core/interfaces/message_sender_conformance.hpp>

#include "capability_probes.hpp"

#include <gtest/gtest.h>

#include <string_view>
#include <type_traits>
#include <vector>

using namespace snicholls;

namespace {

static_assert(message_sender<message_sender_interface>);
static_assert(message_sender<message_sender_ref>);
static_assert(std::is_abstract_v<message_sender_interface>);

TEST(MessageSenderInterface, TheKitPassesInEveryForm)
{
    const auto direct = run_message_sender_conformance<test::message_sender_probe<test::form::concept_model>>();
    const auto virtual_form = run_message_sender_conformance<test::message_sender_probe<test::form::interface>>();
    const auto handle = run_message_sender_conformance<test::message_sender_probe<test::form::ref>>();
    EXPECT_TRUE(direct.ok()) << test::describe(direct);
    EXPECT_TRUE(virtual_form.ok()) << test::describe(virtual_form);
    EXPECT_TRUE(handle.ok()) << test::describe(handle);
    EXPECT_EQ(direct.checks, 5u);
    EXPECT_EQ(virtual_form.checks, 5u);
    EXPECT_EQ(handle.checks, 5u);
}

// Counts every call, to see the adapters forward.
struct counting_sender {
    mutable int calls = 0;
    capability_result<send_receipt> send(const message&) { ++calls; return fail(capability_errc::unavailable); }
    [[nodiscard]] bool supports(channel c) const { ++calls; return c == channel::sms; }
};
static_assert(message_sender<counting_sender>);
static_assert(message_sender<message_sender_adapter<counting_sender>>);
static_assert(message_sender<message_sender_adapter<counting_sender&>>);

TEST(MessageSenderInterface, TheAdaptersForwardEveryOperation)
{
    counting_sender impl;
    message_sender_adapter<counting_sender&> borrowed{impl};
    message_sender_ref s{borrowed};
    EXPECT_EQ(s.send({}).unwrap_err().code, capability_errc::unavailable);
    EXPECT_TRUE(s.supports(channel::sms));
    EXPECT_FALSE(s.supports(channel::email));
    EXPECT_EQ(impl.calls, 3);
    message_sender_adapter owned{counting_sender{}};   // deduced: owns its copy
    static_assert(std::is_same_v<decltype(owned), message_sender_adapter<counting_sender>>);
    EXPECT_TRUE(owned.supports(channel::sms));
    EXPECT_EQ(owned.get().calls, 1);
}

} // namespace
