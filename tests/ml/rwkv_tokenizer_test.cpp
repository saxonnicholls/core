//
//  rwkv_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/ml/rwkv_tokenizer.hpp>: greedy longest matches, a byte no entry starts with, the escapes of
//  RWKV's vocabulary (\t \n \r \xNN, a backslash before anything else, a trailing backslash), decoding by
//  concatenation, the errors, and both at compile time.
//

#include <snicholls/core/ml/bpe.hpp>
#include <snicholls/core/ml/rwkv_tokenizer.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace ml = snicholls::ml;

constexpr ml::static_token_trie<6> vocab{std::array<std::string_view, 6>{"a", "ab", "abc", "b", "c", "\n\n"}};

constexpr std::size_t encode_code(std::string_view text)
{
    std::array<std::uint32_t, 16> out{};
    const std::size_t n = ml::rwkv_encode(text, vocab.view(), 9, out).unwrap();
    std::size_t code = n;
    for (std::size_t i = 0; i < n; ++i)
        code = code * 10 + out[i];
    return code;
}
static_assert(encode_code("abcab") == (2 * 10 + 2) * 10 + 1);   // "abc", "ab"
static_assert(encode_code("abx") == (2 * 10 + 1) * 10 + 9);     // "ab", unknown
static_assert(encode_code("\n\n\n") == (2 * 10 + 5) * 10 + 9);
static_assert(encode_code("") == 0);

constexpr std::size_t unescaped_size(std::string_view s)
{
    std::array<char, 16> out{};
    return ml::rwkv_unescape(s, out).unwrap();
}
static_assert(unescaped_size("\\n\\x41\\\\") == 3 && unescaped_size("plain") == 5 && unescaped_size("a\\") == 1);

TEST(MlRwkv, UnescapingReadsTheVocabularysEscapes)
{
    std::array<char, 16> out{};
    const auto u = [&](std::string_view s) { return std::string(out.data(), ml::rwkv_unescape(s, out).unwrap()); };
    EXPECT_EQ(u("\\t\\n\\r"), "\t\n\r");
    EXPECT_EQ(u("\\x00\\xff\\xAB"), std::string("\x00\xff\xab", 3));
    EXPECT_EQ(u("\\'\\\\"), "'\\");
    EXPECT_EQ(ml::rwkv_unescape("\\x4", out).unwrap_err(), ml::rwkv_error::bad_escape);
    EXPECT_EQ(ml::rwkv_unescape("\\xzz", out).unwrap_err(), ml::rwkv_error::bad_escape);
    std::array<char, 1> one{};
    EXPECT_EQ(ml::rwkv_unescape("ab", one).unwrap_err(), ml::rwkv_error::output_too_small);
}

TEST(MlRwkv, EncodingIsGreedyAndDecodingConcatenates)
{
    std::array<std::uint32_t, 16> out{};
    const auto n = ml::rwkv_encode("abcabxc", vocab.view(), 9, out).unwrap();
    EXPECT_EQ(std::vector<std::uint32_t>(out.begin(), out.begin() + static_cast<std::ptrdiff_t>(n)), (std::vector<std::uint32_t>{2, 1, 9, 4}));
    std::array<char, 16> text{};
    const std::array<std::uint32_t, 3> known{2, 1, 4};
    const auto m = ml::bpe_decode(known, [](std::uint32_t id) { return vocab.pieces[id]; }, text).unwrap();
    EXPECT_EQ(std::string_view(text.data(), m), "abcabc");
    std::array<std::uint32_t, 1> one{};
    EXPECT_EQ(ml::rwkv_encode("abcab", vocab.view(), 9, one).unwrap_err(), ml::rwkv_error::output_too_small);
}

} // namespace
