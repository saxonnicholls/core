//
//  remote_files_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/concepts/remote_files.hpp>: the models and the non-models, the name and prefix rules, the
//  digester, at compile time.
//

#include <snicholls/core/concepts/remote_files.hpp>
#include <snicholls/core/interfaces/remote_files_reference.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

using namespace snicholls;

namespace {

static_assert(remote_files<memory_remote_files>);
static_assert(remote_files<remote_files_interface>);
static_assert(remote_files<remote_files_adapter<memory_remote_files>>);
static_assert(remote_files<remote_files_adapter<memory_remote_files&>>);
static_assert(remote_files<remote_files_ref>);

struct no_move {
    capability_result<std::vector<remote_file>> list(std::string_view);
    capability_result<file_digest> fetch(std::string_view, remote_file_sink);
    bool supports(remote_files_op) const;
};
struct fetch_returns_bytes {
    capability_result<std::vector<remote_file>> list(std::string_view);
    std::vector<std::byte> fetch(std::string_view, remote_file_sink);
    capability_result<void> move(std::string_view, std::string_view);
    bool supports(remote_files_op) const;
};
struct supports_not_const {
    capability_result<std::vector<remote_file>> list(std::string_view);
    capability_result<file_digest> fetch(std::string_view, remote_file_sink);
    capability_result<void> move(std::string_view, std::string_view);
    bool supports(remote_files_op);
};
static_assert(!remote_files<no_move>);   // move is optional at run time, never in the type
static_assert(!remote_files<fetch_returns_bytes>);
static_assert(!remote_files<supports_not_const>);

static_assert(std::is_trivially_copyable_v<remote_file>);
static_assert(std::is_trivially_copyable_v<file_digest>);

static_assert(is_remote_name("APST_20261126.csv"));
static_assert(is_remote_name("outgoing/2026/11/APST_20261126.csv"));
static_assert(is_remote_name(".hidden") && is_remote_name("a/..b") && is_remote_name("a/b."));
static_assert(!is_remote_name(""));
static_assert(!is_remote_name("/etc/passwd") && !is_remote_name("in/") && !is_remote_name("in//a"));
static_assert(!is_remote_name("..") && !is_remote_name("in/../a") && !is_remote_name("./a") && !is_remote_name("in/."));
static_assert(!is_remote_name("in\\a") && !is_remote_name("in/a\tb") && !is_remote_name("in/a\x7f"));
static_assert(is_remote_prefix("") && is_remote_prefix("in/") && is_remote_prefix("in/APST_2026") && is_remote_prefix("in/."));
static_assert(!is_remote_prefix("/in") && !is_remote_prefix("in//") && !is_remote_prefix("../") && !is_remote_prefix("in/.."));
static_assert(!is_remote_prefix("./in"));
static_assert(noexcept(is_remote_name(std::string_view{})) && noexcept(validate_remote_name(std::string_view{})));

constexpr bool digester_agrees()
{
    constexpr std::array<std::byte, 3> abc{std::byte{'a'}, std::byte{'b'}, std::byte{'c'}};
    file_digester d;
    d.update(std::span{abc}.first(1));
    d.update(std::span{abc}.subspan(1));
    const file_digest split = d.finish();
    // FIPS 180-4's example: SHA-256("abc")
    return split == file_digester::of(abc) && split.size == 3 &&
           split.sha256 == *digest<32>::from_hex("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
}
static_assert(digester_agrees());

TEST(RemoteFilesNames, TheLengthLimitIsInclusive)
{
    const std::string longest(max_remote_name_length, 'a');
    EXPECT_TRUE(is_remote_name(longest));
    EXPECT_FALSE(is_remote_name(longest + "a"));
    EXPECT_TRUE(is_remote_prefix(longest));
    EXPECT_FALSE(is_remote_prefix(longest + "a"));
    EXPECT_FALSE(is_remote_name(std::string_view{"a\0b", 3}));
}

TEST(RemoteFilesNames, ValidationNamesTheRuleNeverTheName)
{
    const auto r = validate_remote_name("in/../secret-client-name.csv");
    ASSERT_TRUE(r.is_err());
    EXPECT_EQ(r.unwrap_err().code, capability_errc::invalid_argument);
    EXPECT_EQ(r.unwrap_err().message, std::string_view{"not a file name in the drop"});
    EXPECT_TRUE(validate_remote_name("in/a.csv").is_ok());
    EXPECT_EQ(validate_remote_prefix("/in").unwrap_err().code, capability_errc::invalid_argument);
    EXPECT_TRUE(validate_remote_prefix("").is_ok());
}

TEST(RemoteFilesDigest, OfNoBytesIsTheEmptyDigest)
{
    const auto d = file_digester::of({});
    EXPECT_EQ(d.size, 0u);
    EXPECT_EQ(d.sha256, *digest<32>::from_hex("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"));
}

} // namespace
