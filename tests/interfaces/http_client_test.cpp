//
//  http_client_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/interfaces/http_client.hpp>: the conformance kit passes on the reference in every form (the
//  concept, the virtual interface over a borrowing adapter, the handle), and the adapters forward.
//

#include <snicholls/core/interfaces/http_client.hpp>
#include <snicholls/core/interfaces/http_client_conformance.hpp>

#include "contract_probes.hpp"

#include <gtest/gtest.h>

#include <type_traits>

using namespace snicholls;

namespace {

static_assert(std::is_abstract_v<http_client_interface>);

TEST(HttpClientInterface, TheKitPassesInEveryForm)
{
    const auto direct = run_http_client_conformance<test::http_client_probe<test::form::concept_model>>();
    const auto virtual_form = run_http_client_conformance<test::http_client_probe<test::form::interface>>();
    const auto handle = run_http_client_conformance<test::http_client_probe<test::form::ref>>();
    for (const auto* r : {&direct, &virtual_form, &handle}) {
        EXPECT_TRUE(r->ok()) << test::describe(*r);
        EXPECT_EQ(r->checks, 5u);
    }
}

struct counting_client {
    int calls = 0;
    capability_result<http_response> send(const http_request&) { ++calls; return ok(respond(204)); }
};

TEST(HttpClientInterface, TheAdaptersForward)
{
    counting_client impl;
    http_client_adapter<counting_client&> borrowed{impl};
    http_client_ref c{borrowed};
    EXPECT_EQ(c.send({})->status, 204);
    EXPECT_EQ(impl.calls, 1);
    http_client_adapter owned{counting_client{}};
    static_assert(std::is_same_v<decltype(owned), http_client_adapter<counting_client>>);
    static_cast<void>(owned.send({}));
    EXPECT_EQ(owned.get().calls, 1);
}

} // namespace
