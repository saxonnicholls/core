//
//  remote_files_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/interfaces/remote_files.hpp>: the conformance kit passes on the reference in every form (the
//  concept, the virtual interface over a borrowing adapter, the handle), and the adapters forward every operation.
//

#include <snicholls/core/interfaces/remote_files.hpp>
#include <snicholls/core/interfaces/remote_files_conformance.hpp>

#include "capability_probes.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <span>
#include <string_view>
#include <type_traits>
#include <vector>

using namespace snicholls;

namespace {

static_assert(remote_files<remote_files_interface>);
static_assert(remote_files<remote_files_ref>);
static_assert(std::is_abstract_v<remote_files_interface>);

TEST(RemoteFilesInterface, TheKitPassesInEveryForm)
{
    const auto direct = run_remote_files_conformance<test::remote_files_probe<test::form::concept_model>>();
    const auto virtual_form = run_remote_files_conformance<test::remote_files_probe<test::form::interface>>();
    const auto handle = run_remote_files_conformance<test::remote_files_probe<test::form::ref>>();
    EXPECT_TRUE(direct.ok()) << test::describe(direct);
    EXPECT_TRUE(virtual_form.ok()) << test::describe(virtual_form);
    EXPECT_TRUE(handle.ok()) << test::describe(handle);
    EXPECT_EQ(direct.checks, 7u);
    EXPECT_EQ(virtual_form.checks, 7u);
    EXPECT_EQ(handle.checks, 7u);
}

// Counts every call, to see the adapters forward.
struct counting_files {
    mutable int calls = 0;
    capability_result<std::vector<remote_file>> list(std::string_view) { ++calls; return fail(capability_errc::unavailable); }
    capability_result<file_digest> fetch(std::string_view, remote_file_sink sink)
    {
        ++calls;
        const std::byte b{42};
        if (auto r = sink(std::span{&b, 1}); !r)
            return err(r.unwrap_err());
        return ok(file_digester::of(std::span{&b, 1}));
    }
    capability_result<void> move(std::string_view, std::string_view) { ++calls; return fail(capability_errc::unsupported); }
    [[nodiscard]] bool supports(remote_files_op op) const { ++calls; return op != remote_files_op::move; }
};
static_assert(remote_files<counting_files>);
static_assert(remote_files<remote_files_adapter<counting_files>>);
static_assert(remote_files<remote_files_adapter<counting_files&>>);

TEST(RemoteFilesInterface, TheAdaptersForwardEveryOperation)
{
    counting_files impl;
    remote_files_adapter<counting_files&> borrowed{impl};
    remote_files_ref f{borrowed};
    EXPECT_EQ(f.list("").unwrap_err().code, capability_errc::unavailable);
    std::vector<std::byte> got;
    const auto d = f.fetch("a", [&got](std::span<const std::byte> c) -> capability_result<void> {
        got.insert(got.end(), c.begin(), c.end());
        return ok();
    });
    ASSERT_TRUE(d.is_ok());
    EXPECT_EQ(d->size, 1u);
    EXPECT_EQ(got, std::vector<std::byte>{std::byte{42}});
    EXPECT_EQ(f.move("a", "b").unwrap_err().code, capability_errc::unsupported);
    EXPECT_TRUE(f.supports(remote_files_op::fetch));
    EXPECT_FALSE(f.supports(remote_files_op::move));
    EXPECT_EQ(impl.calls, 5);
    remote_files_adapter owned{counting_files{}};   // deduced: owns its copy
    static_assert(std::is_same_v<decltype(owned), remote_files_adapter<counting_files>>);
    EXPECT_TRUE(owned.supports(remote_files_op::list));
    EXPECT_EQ(owned.get().calls, 1);
}

} // namespace
