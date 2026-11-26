//
//  remote_files_reference_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/interfaces/remote_files_reference.hpp>: put, the listing, chunked fetches with the digest,
//  move on and off, the injected failure and the redaction rule.
//

#include <snicholls/core/interfaces/remote_files_reference.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <span>
#include <string_view>
#include <vector>

using namespace snicholls;

namespace {

const utc_s t0 = utc_s::from_unix_seconds(1'800'000'000);

std::vector<std::byte> bytes_of(std::string_view s)
{
    std::vector<std::byte> out;
    for (const char c : s)
        out.push_back(static_cast<std::byte>(c));
    return out;
}

TEST(MemoryRemoteFiles, PutRefusesABadOrTakenName)
{
    memory_remote_files f;
    EXPECT_TRUE(f.put("in/a.csv", bytes_of("1,2\n"), t0).is_ok());
    EXPECT_EQ(f.put("in/a.csv", bytes_of("other"), t0).unwrap_err().code, capability_errc::conflict);
    EXPECT_EQ(f.put("../a.csv", bytes_of("x"), t0).unwrap_err().code, capability_errc::invalid_argument);
    EXPECT_EQ(f.size(), 1u);
    ASSERT_NE(f.contents("in/a.csv"), nullptr);
    EXPECT_EQ(*f.contents("in/a.csv"), bytes_of("1,2\n"));
    EXPECT_EQ(f.contents("in/b.csv"), nullptr);
}

TEST(MemoryRemoteFiles, ListsByPrefixInNameOrder)
{
    memory_remote_files f;
    ASSERT_TRUE(f.put("in/2026-11-26.csv", bytes_of("abc"), t0 + std::chrono::seconds{1}).is_ok());
    ASSERT_TRUE(f.put("in/2026-11-25.csv", bytes_of("ab"), t0).is_ok());
    ASSERT_TRUE(f.put("archive/2026-11-24.csv", bytes_of("a"), t0).is_ok());
    const auto in = f.list("in/");
    ASSERT_TRUE(in.is_ok());
    ASSERT_EQ(in->size(), 2u);
    EXPECT_EQ((*in)[0], (remote_file{*short_text::make("in/2026-11-25.csv"), 2, t0}));
    EXPECT_EQ((*in)[1], (remote_file{*short_text::make("in/2026-11-26.csv"), 3, t0 + std::chrono::seconds{1}}));
    EXPECT_EQ(f.list("")->size(), 3u);
    EXPECT_TRUE(f.list("out/")->empty());
}

TEST(MemoryRemoteFiles, FetchesInChunksWithTheDigestOfWhatWasDelivered)
{
    memory_remote_files f{true, 4};
    const auto text = bytes_of("0123456789");
    ASSERT_TRUE(f.put("in/a.csv", text, t0).is_ok());
    std::vector<std::size_t> sizes;
    std::vector<std::byte> got;
    const auto d = f.fetch("in/a.csv", [&](std::span<const std::byte> c) -> capability_result<void> {
        sizes.push_back(c.size());
        got.insert(got.end(), c.begin(), c.end());
        return ok();
    });
    ASSERT_TRUE(d.is_ok());
    EXPECT_EQ(sizes, (std::vector<std::size_t>{4, 4, 2}));
    EXPECT_EQ(got, text);
    EXPECT_EQ(*d, file_digester::of(text));
    memory_remote_files zero_chunk{true, 0};   // a chunk size of 0 is 1
    ASSERT_TRUE(zero_chunk.put("a", text, t0).is_ok());
    std::size_t calls = 0;
    EXPECT_TRUE(zero_chunk.fetch("a", [&](std::span<const std::byte>) -> capability_result<void> { ++calls; return ok(); }).is_ok());
    EXPECT_EQ(calls, text.size());
}

TEST(MemoryRemoteFiles, MoveRenamesOrIsSwitchedOff)
{
    memory_remote_files f;
    ASSERT_TRUE(f.put("in/a.csv", bytes_of("a"), t0).is_ok());
    ASSERT_TRUE(f.put("in/b.csv", bytes_of("b"), t0).is_ok());
    EXPECT_TRUE(f.supports(remote_files_op::move));
    EXPECT_TRUE(f.move("in/a.csv", "archive/a.csv").is_ok());
    EXPECT_EQ(f.contents("in/a.csv"), nullptr);
    EXPECT_EQ(*f.contents("archive/a.csv"), bytes_of("a"));
    EXPECT_EQ(f.move("in/a.csv", "x").unwrap_err().code, capability_errc::not_found);
    EXPECT_EQ(f.move("in/b.csv", "archive/a.csv").unwrap_err().code, capability_errc::conflict);
    EXPECT_EQ(f.move("in/b.csv", "in/b.csv").unwrap_err().code, capability_errc::conflict);

    memory_remote_files read_only{false};
    ASSERT_TRUE(read_only.put("in/a.csv", bytes_of("a"), t0).is_ok());
    EXPECT_FALSE(read_only.supports(remote_files_op::move));
    EXPECT_TRUE(read_only.supports(remote_files_op::list) && read_only.supports(remote_files_op::fetch));
    EXPECT_EQ(read_only.move("in/a.csv", "archive/a.csv").unwrap_err().code, capability_errc::unsupported);
    EXPECT_NE(read_only.contents("in/a.csv"), nullptr);
}

TEST(MemoryRemoteFiles, FailNextInjectsOneFailureThenRecovers)
{
    memory_remote_files f;
    ASSERT_TRUE(f.put("in/a.csv", bytes_of("a"), t0).is_ok());
    f.fail_next(capability_errc::timeout, "ETIMEDOUT reading from sftp.example.test");
    const auto first = f.fetch("in/a.csv", [](std::span<const std::byte>) -> capability_result<void> { return ok(); });
    ASSERT_TRUE(first.is_err());
    EXPECT_TRUE(first.unwrap_err().retryable());
    EXPECT_EQ(first.unwrap_err().provider_code, std::string_view{"ETIMEDOUT"});   // the code only, never the rest
    EXPECT_TRUE(f.list("").is_ok());
    f.fail_next(capability_errc::unavailable, "{\"error\":\"down\"}");
    const auto listed = f.list("");
    ASSERT_TRUE(listed.is_err());
    EXPECT_TRUE(listed.unwrap_err().provider_code.empty());
    f.fail_next(capability_errc::unavailable);
    EXPECT_EQ(f.move("in/a.csv", "b").unwrap_err().code, capability_errc::unavailable);
    EXPECT_NE(f.contents("in/a.csv"), nullptr);
}

TEST(MemoryRemoteFiles, ErrorsNeverNameTheFile)
{
    memory_remote_files f;
    const auto r = f.fetch("in/client-7731-statement.csv", [](std::span<const std::byte>) -> capability_result<void> { return ok(); });
    ASSERT_TRUE(r.is_err());
    EXPECT_EQ(r.unwrap_err().code, capability_errc::not_found);
    EXPECT_EQ(r.unwrap_err().message.view().find("7731"), std::string_view::npos);
    EXPECT_TRUE(r.unwrap_err().provider_code.empty());
}

} // namespace
