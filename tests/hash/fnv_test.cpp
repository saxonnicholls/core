//
//  hash/fnv_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/hash/fnv.hpp>, against the published FNV-1a test
//  vectors. These values are a wire contract between machines, so they are
//  pinned here: a change to any of them is a breaking change.
//

#include <snicholls/core/hash/fnv.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <string_view>

namespace {

using namespace snicholls::literals;
using snicholls::fnv1a_32;
using snicholls::fnv1a_64;
using snicholls::fnv1a_64_append;

// Published FNV-1a vectors (Fowler, Noll, Vo), checked at compile time
static_assert(fnv1a_64("") == 0xcbf29ce484222325ull);
static_assert(fnv1a_64("a") == 0xaf63dc4c8601ec8cull);
static_assert(fnv1a_64("foobar") == 0x85944171f73967e8ull);
static_assert(fnv1a_32("") == 0x811c9dc5u);
static_assert(fnv1a_32("a") == 0xe40c292cu);
static_assert(fnv1a_32("foobar") == 0xbf9cf968u);

// The literals are the functions
static_assert("foobar"_fnv64 == fnv1a_64("foobar"));
static_assert("foobar"_fnv32 == fnv1a_32("foobar"));

// Hashing in pieces is hashing the whole
static_assert(fnv1a_64("bar", fnv1a_64("foo")) == fnv1a_64("foobar"));

// Appending an integer hashes its little-endian bytes, whatever the host
static_assert(fnv1a_64_append(snicholls::fnv1a_64_basis, std::uint32_t{1}) ==
              fnv1a_64(std::string_view("\x01\x00\x00\x00", 4)));
static_assert(fnv1a_64_append(snicholls::fnv1a_64_basis, std::uint32_t{1}) == 0xad2aca7747985764ull);
static_assert(fnv1a_64_append(snicholls::fnv1a_64_basis, std::uint8_t{1}) !=
              fnv1a_64_append(snicholls::fnv1a_64_basis, std::uint32_t{1}));   // width matters

constexpr std::string_view some_view{"x"};
static_assert(noexcept(fnv1a_64(some_view)));
static_assert(noexcept(fnv1a_64_append(0ull, 1)));

int classify(std::string_view method)
{
    switch (fnv1a_64(method)) {
    case "GET"_fnv64: return 1;
    case "POST"_fnv64: return 2;
    default: return 0;
    }
}

TEST(Fnv, RunTimeEqualsCompileTime)
{
    const std::string s = "foobar";   // a run-time string
    EXPECT_EQ(fnv1a_64(s), 0x85944171f73967e8ull);
    EXPECT_EQ(fnv1a_32(s), 0xbf9cf968u);
}

TEST(Fnv, SwitchOnAString)
{
    EXPECT_EQ(classify("GET"), 1);
    EXPECT_EQ(classify("POST"), 2);
    EXPECT_EQ(classify("BREW"), 0);
}

TEST(Fnv, EmbeddedNullsAreHashed)
{
    EXPECT_NE(fnv1a_64(std::string_view("a\0b", 3)), fnv1a_64("ab"));
}

TEST(Fnv, AppendIsOrderSensitive)
{
    const auto h = snicholls::fnv1a_64_basis;
    EXPECT_NE(fnv1a_64_append(fnv1a_64_append(h, 1), 2), fnv1a_64_append(fnv1a_64_append(h, 2), 1));
}

} // namespace
