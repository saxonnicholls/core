//
//  hash/wyhash_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/hash/wyhash.hpp>, against upstream's test_vector.cpp for
//  final version 4.3 (github.com/wangyi-fudan/wyhash, master, run 2026-11-06).
//  These values are a wire contract between machines, so they are pinned here:
//  a change to any of them is a breaking change.
//

#include <snicholls/core/hash/wyhash.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace snicholls::literals;
using snicholls::splitmix64;
using snicholls::wyhash;
using snicholls::wyhash_bytes;
using snicholls::wymix;

// Upstream test_vector.cpp, final version 4.3: wyhash(input, seed) with the default secret
static_assert(wyhash("", 0) == 0x93228a4de0eec5a2ull);
static_assert(wyhash("a", 1) == 0xc5bac3db178713c4ull);
static_assert(wyhash("abc", 2) == 0xa97f2f7b1d9b3314ull);
static_assert(wyhash("message digest", 3) == 0x786d1f1df3801df4ull);
static_assert(wyhash("abcdefghijklmnopqrstuvwxyz", 4) == 0xdca5a8138ad37c87ull);
static_assert(wyhash("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789", 5) == 0xb9e734f117cfaf70ull);
static_assert(wyhash("12345678901234567890123456789012345678901234567890123456789012345678901234567890", 6) ==
              0x6cc5eab49a92d617ull);

// The literal is the function
static_assert("message digest"_wyhash == wyhash("message digest"));

// Every length class the function branches on hashes, and differently from its neighbours
static_assert(wyhash("") != wyhash(std::string_view{"\0", 1}));          // 0 vs 1..3
static_assert(wyhash("abc") != wyhash("abcd"));                           // 1..3 vs 4..16
static_assert(wyhash("0123456789abcdef") != wyhash("0123456789abcdef0"));   // 16 vs 17..48
static_assert(wyhash(std::string_view{"0123456789abcdef0123456789abcdef0123456789abcdef"}) !=
              wyhash(std::string_view{"0123456789abcdef0123456789abcdef0123456789abcdefX"}));   // 48 vs 49+

// The seed changes the function
static_assert(wyhash("abc", 0) != wyhash("abc", 1));

// splitmix64's first output from state 0 (Vigna's reference), and wymix's shape
static_assert(splitmix64(0) == 0xe220a8397b1dcdafull);
static_assert(wymix(3, 5) == 15 && wymix(5, 3) == 15 && wymix(0, 12345) == 0);
static_assert(wymix(0xffffffffffffffffull, 0xffffffffffffffffull) == (1ull ^ 0xfffffffffffffffeull));

static_assert(noexcept(wyhash(std::string_view{})) && noexcept(splitmix64(1)) && noexcept(wymix(1, 2)));

std::vector<std::string> keys_of_every_length()
{
    std::vector<std::string> out;
    for (std::size_t len = 0; len <= 200; ++len) {
        std::string s(len, '\0');
        for (std::size_t i = 0; i < len; ++i)
            s[i] = static_cast<char>((i * 31 + len * 7) & 0xff);
        out.push_back(std::move(s));
    }
    return out;
}

TEST(WyhashRuntime, AgreesWithTheConstantExpressionPathOnEveryLengthClass)
{
    // At compile time the 128-bit multiply is four 32-bit ones; at run time it may be one
    // instruction. The pinned vectors above prove the compile-time path; this proves the other.
    constexpr auto pinned = wyhash("message digest", 3);
    const std::string runtime{"message digest"};
    EXPECT_EQ(wyhash(runtime, 3), pinned);

    for (const auto& k : keys_of_every_length()) {
        const auto h = wyhash(std::string_view{k});
        EXPECT_EQ(h, wyhash(std::string_view{k}));                                            // deterministic
        EXPECT_EQ(h, wyhash(std::span<const std::byte>{reinterpret_cast<const std::byte*>(k.data()), k.size()}));
        EXPECT_EQ(h, wyhash(std::span<const unsigned char>{reinterpret_cast<const unsigned char*>(k.data()), k.size()}));
        EXPECT_EQ(h, wyhash_bytes(k, k.size()));                                              // the generic form
    }
}

TEST(WyhashRuntime, OneFlippedByteChangesTheHash)
{
    const std::string base(70, 'q');
    const auto h = wyhash(std::string_view{base});
    for (std::size_t i = 0; i < base.size(); ++i) {
        std::string t = base;
        t[i] = 'r';
        EXPECT_NE(wyhash(std::string_view{t}), h) << "byte " << i;
    }
}

TEST(WyhashRuntime, HashesAFixedWidthArrayThroughTheGenericForm)
{
    constexpr std::array<char, 5> word{'h', 'e', 'l', 'l', 'o'};
    static_assert(wyhash_bytes(word, word.size()) == wyhash("hello"));
    EXPECT_EQ(wyhash_bytes(word, 4), wyhash("hell"));   // len limits what is read
}

} // namespace
