//
//  remote_files_conformance_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/interfaces/remote_files_conformance.hpp>: the kit passes the reference, read-only or not,
//  and catches broken drops: one that leaks the server's response, one that lists out of order, one that reports a
//  digest of the whole file whatever the sink took, one that echoes a name, one that ignores the sink's error.
//

#include <snicholls/core/interfaces/remote_files_conformance.hpp>
#include <snicholls/core/interfaces/remote_files_reference.hpp>

#include "capability_probes.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <span>
#include <string_view>
#include <vector>

using namespace snicholls;

namespace {

enum class defect { none, leaks, unsorted, wrong_digest, echoes_name, ignores_sink };

// The reference, with one defect.
template <defect D>
struct broken_files {
    memory_remote_files inner{true, 1000};
    bool armed = false;

    capability_result<std::vector<remote_file>> list(std::string_view prefix)
    {
        if (D == defect::leaks && armed) {
            armed = false;
            return err(capability_error{capability_errc::unavailable, *provider_code::make(capability_redaction_marker), {}});
        }
        auto r = inner.list(prefix);
        if (D == defect::unsorted && r.is_ok())
            std::reverse(r->begin(), r->end());
        return r;
    }
    capability_result<file_digest> fetch(std::string_view name, remote_file_sink sink)
    {
        if (D == defect::echoes_name && inner.contents(name) == nullptr && is_remote_name(name)) {
            auto code = provider_code::make(name.substr(name.find('/') + 1, 21));   // "REDACTION-MARKER-c0re"
            return err(capability_error{capability_errc::not_found, code ? *code : provider_code{}, {}});
        }
        if (D == defect::ignores_sink)
            return inner.fetch(name, [&](std::span<const std::byte> c) -> capability_result<void> {
                static_cast<void>(sink(c));
                return ok();
            });
        auto r = inner.fetch(name, sink);
        if (D == defect::wrong_digest && r.is_ok())
            r->size += 1;
        return r;
    }
    capability_result<void> move(std::string_view from, std::string_view to) { return inner.move(from, to); }
    [[nodiscard]] bool supports(remote_files_op op) const { return inner.supports(op); }
};

template <defect D>
struct broken_probe {
    broken_files<D> f;
    auto& files() { return f; }
    void seed(std::string_view n, std::span<const std::byte> b, utc_s t) { static_cast<void>(f.inner.put(n, b, t)); }
    void provider_fails(capability_errc c, std::string_view r)
    {
        f.armed = true;
        f.inner.fail_next(c, r);
    }
};

struct read_only_probe {
    memory_remote_files impl{false, 3};
    auto& files() { return impl; }
    void seed(std::string_view n, std::span<const std::byte> b, utc_s t) { static_cast<void>(impl.put(n, b, t)); }
    void provider_fails(capability_errc c, std::string_view r) { impl.fail_next(c, r); }
};

TEST(RemoteFilesConformance, TheReferencePassesReadOnlyOrNot)
{
    const auto r = run_remote_files_conformance<test::remote_files_probe<test::form::concept_model>>();
    EXPECT_TRUE(r.ok()) << test::describe(r);
    const auto ro = run_remote_files_conformance<read_only_probe>();
    EXPECT_TRUE(ro.ok()) << test::describe(ro);
    const auto none = run_remote_files_conformance<broken_probe<defect::none>>();
    EXPECT_TRUE(none.ok()) << test::describe(none);
}

TEST(RemoteFilesConformance, ALeakIsCaught)
{
    const auto r = run_remote_files_conformance<broken_probe<defect::leaks>>();
    EXPECT_TRUE(r.failed("a provider failure never leaks its response")) << test::describe(r);
    EXPECT_EQ(r.failures.size(), 1u) << test::describe(r);
}

TEST(RemoteFilesConformance, AnUnsortedListingIsCaught)
{
    const auto r = run_remote_files_conformance<broken_probe<defect::unsorted>>();
    EXPECT_TRUE(r.failed("list returns the matching files sorted by name")) << test::describe(r);
}

TEST(RemoteFilesConformance, AWrongDigestIsCaught)
{
    const auto r = run_remote_files_conformance<broken_probe<defect::wrong_digest>>();
    EXPECT_TRUE(r.failed("fetch streams every byte with its digest")) << test::describe(r);
}

TEST(RemoteFilesConformance, AnEchoedNameIsCaught)
{
    const auto r = run_remote_files_conformance<broken_probe<defect::echoes_name>>();
    EXPECT_TRUE(r.failed("a missing file is not_found without echoing its name")) << test::describe(r);
}

TEST(RemoteFilesConformance, AnIgnoredSinkErrorIsCaught)
{
    const auto r = run_remote_files_conformance<broken_probe<defect::ignores_sink>>();
    EXPECT_TRUE(r.failed("a sink's error stops the fetch and is returned")) << test::describe(r);
}

} // namespace
