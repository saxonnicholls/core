//
//  https_client_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/interfaces/https_client.hpp>: the conformance kit passes on the reference in every form
//  (the concept, the virtual interface over a borrowing adapter, the handle), and the adapters forward.
//

#include <snicholls/core/interfaces/https_client.hpp>
#include <snicholls/core/interfaces/https_client_conformance.hpp>

#include "capability_probes.hpp"

#include <gtest/gtest.h>

#include <type_traits>

using namespace snicholls;

namespace {

static_assert(https_client<https_client_interface>);
static_assert(https_client<https_client_ref>);
static_assert(std::is_abstract_v<https_client_interface>);

TEST(HttpsClientInterface, TheKitPassesInEveryForm)
{
    const auto direct = run_https_client_conformance<test::https_client_probe<test::form::concept_model>>();
    const auto virtual_form = run_https_client_conformance<test::https_client_probe<test::form::interface>>();
    const auto handle = run_https_client_conformance<test::https_client_probe<test::form::ref>>();
    EXPECT_TRUE(direct.ok()) << test::describe(direct);
    EXPECT_TRUE(virtual_form.ok()) << test::describe(virtual_form);
    EXPECT_TRUE(handle.ok()) << test::describe(handle);
    EXPECT_EQ(direct.checks, 7u);
    EXPECT_EQ(virtual_form.checks, 7u);
    EXPECT_EQ(handle.checks, 7u);
}

TEST(HttpsClientInterface, TheAdaptersForward)
{
    int calls = 0;
    memory_https_client inner{[&](const https_request&) -> capability_result<https_response> {
        ++calls;
        return ok(https_response{200, {}, "x"});
    }};
    https_client_adapter<memory_https_client&> adapter{inner};
    https_client_ref ref{adapter};
    https_request r;
    r.url = "https://a.test/";
    EXPECT_EQ(ref.send(r)->body, "x");
    EXPECT_EQ(calls, 1);
    EXPECT_EQ(&adapter.get(), &inner);
    https_client_adapter owned{memory_https_client{}};
    EXPECT_EQ(owned.send(r).unwrap_err().code, capability_errc::unavailable) << "no routes: nothing is reachable";
}

} // namespace
