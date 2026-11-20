//
//  scrypt_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/hash/scrypt.hpp>: RFC 7914's Salsa20/8 core, BlockMix, ROMix and scrypt vectors (the
//  N = 2^20 one only in Release), Python's hashlib.scrypt on random inputs and parameters, every error, the wipe of
//  the working memory, and a derivation at compile time.
//

#include <snicholls/core/hash/scrypt.hpp>

#include "../support/hash_fixtures.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace {

using namespace snicholls;
using snicholls::test::hex;
using snicholls::test::unhex;

constexpr bool rfc7914_first_vector_at_compile_time()
{
    std::array<std::uint32_t, 32 * (16 + 1 + 1)> work{};
    digest<64> out{};
    return scrypt("", "", scrypt_params{16, 1, 1}, work, out.bytes).is_ok() &&
           out.hex().view() ==
               "77d6576238657b203b19ca42c18a0497f16b4844e3074ae8dfdffa3fede21442fcd0069ded0948f8326a753a0fc81f17e8d3e0fb2e0d3628cf35e20c38d18906";
}
static_assert(rfc7914_first_vector_at_compile_time());
static_assert(scrypt_work_words({16, 1, 1}).unwrap_unchecked() == 32 * 18);
static_assert(scrypt_work_words({1024, 8, 16}).unwrap_unchecked() == 256 * (1024 + 16 + 1));
static_assert(scrypt_work_words({15, 1, 1}).unwrap_err_unchecked() == kdf_error::invalid_cost);
static_assert(scrypt_work_words({1, 1, 1}).unwrap_err_unchecked() == kdf_error::invalid_cost);
static_assert(scrypt_work_words({1u << 16, 1, 1}).unwrap_err_unchecked() == kdf_error::invalid_cost);   // N < 2^(16 r)
static_assert(scrypt_work_words({1u << 15, 1, 1}).is_ok());
static_assert(scrypt_work_words({16, 0, 1}).unwrap_err_unchecked() == kdf_error::invalid_block_parameters);
static_assert(scrypt_work_words({16, 1, 0}).unwrap_err_unchecked() == kdf_error::invalid_block_parameters);
static_assert(scrypt_work_words({16, 1u << 15, 1u << 15}).unwrap_err_unchecked() == kdf_error::invalid_block_parameters);
static_assert(scrypt_work_words({std::uint64_t{1} << 62, 8, 1}).unwrap_err_unchecked() == kdf_error::invalid_cost);   // overflow
static_assert(noexcept(scrypt("p", "s", scrypt_params{}, std::span<std::uint32_t>{}, std::span<std::byte>{})));

std::vector<std::uint32_t> words_of(const std::vector<std::byte>& bytes)
{
    std::vector<std::uint32_t> w(bytes.size() / 4);
    for (std::size_t i = 0; i < w.size(); ++i)
        w[i] = detail::load_le_word<std::uint32_t>(bytes.data() + 4 * i);
    return w;
}

std::string hex_of(const std::vector<std::uint32_t>& w)
{
    std::vector<std::byte> b(4 * w.size());
    for (std::size_t i = 0; i < w.size(); ++i)
        detail::store_le_word(b.data() + 4 * i, w[i]);
    return hex(b);
}

TEST(ScryptVectors, Rfc7914Salsa20BlockMixAndRoMix)
{
    auto salsa = words_of(unhex("7e879a214f3ec9867ca940e641718f26baee555b8c61c1b50df846116dcd3b1dee24f319df9b3d8514121e4b5ac5aa3276021d2909c74829edebc68db8b8c25e"));
    std::uint32_t block[16]{};
    for (std::size_t i = 0; i < 16; ++i) block[i] = salsa[i];
    detail::salsa20_8(block);
    EXPECT_EQ(hex_of({block, block + 16}),
              "a41f859c6608cc993b81cacb020cef05044b2181a2fd337dfd7b1c6396682f29b4393168e3c9e6bcfe6bc5b7a06d96bae424cc102c91745c24ad673dc7618f81");

    const std::string b = "f7ce0b653d2d72a4108cf5abe912ffdd777616dbbb27a70e8204f3ae2d0f6fad89f68f4811d1e87bcc3bd7400a9ffd29094f0184639574f39ae5a1315217bcd7"
                          "894991447213bb226c25b54da86370fbcd984380374666bb8ffcb5bf40c254b067d27c51ce4ad5fed829c90b505a571b7f4d1cad6a523cda770e67bceaaf7e89";
    auto mix = words_of(unhex(b));
    std::vector<std::uint32_t> y(32);
    detail::scrypt_block_mix(mix.data(), y.data(), 1);
    EXPECT_EQ(hex_of(mix), "a41f859c6608cc993b81cacb020cef05044b2181a2fd337dfd7b1c6396682f29b4393168e3c9e6bcfe6bc5b7a06d96bae424cc102c91745c24ad673dc7618f81"
                           "20edc975323881a80540f64c162dcd3c21077cfe5f8d5fe2b1a4168f953678b77d3b3d803b60e4ab920996e59b4d53b65d2a225877d5edf5842cb9f14eefe425");

    auto ro = words_of(unhex(b));
    std::vector<std::uint32_t> v(16 * 32);
    detail::scrypt_ro_mix(ro.data(), v.data(), y.data(), 1, 16);
    EXPECT_EQ(hex_of(ro), "79ccc193629debca047f0b70604bf6b62ce3dd4a9626e355fafc6198e6ea2b46d58413673b99b029d665c357601fb426a0b2f4bba200ee9f0a43d19b571a9c71"
                          "ef1142e65d5a266fddca832ce59faa7cac0b9cf1be2bffca300d01ee387619c4ae12fd4438f203a0e4e1c47ec314861f4e9087cb33396a6873e8f9d2539a4b8e");
}

std::string run_scrypt(std::string_view password, std::string_view salt, scrypt_params params, std::size_t length)
{
    std::vector<std::uint32_t> work(scrypt_work_words(params).unwrap_unchecked());
    std::vector<std::byte> out(length);
    EXPECT_TRUE(scrypt(password, salt, params, work, out));
    return hex(out);
}

TEST(ScryptVectors, Rfc7914Scrypt)
{
    EXPECT_EQ(run_scrypt("", "", {16, 1, 1}, 64),
              "77d6576238657b203b19ca42c18a0497f16b4844e3074ae8dfdffa3fede21442fcd0069ded0948f8326a753a0fc81f17e8d3e0fb2e0d3628cf35e20c38d18906");
    EXPECT_EQ(run_scrypt("password", "NaCl", {1024, 8, 16}, 64),
              "fdbabe1c9d3472007856e7190d01e9fe7c6ad7cbc8237830e77376634b3731622eaf30d92e22a3886ff109279d9830dac727afb94a83ee6d8360cbdfa2cc0640");
    EXPECT_EQ(run_scrypt("pleaseletmein", "SodiumChloride", {16384, 8, 1}, 64),
              "7023bdcb3afd7348461c06cd81fd38ebfda8fbba904f8e3ea9b543f6545da1f2d5432955613f0fcf62d49705242a9af9e61e85dc0d651e40dfcf017b45575887");
#ifdef NDEBUG
    EXPECT_EQ(run_scrypt("pleaseletmein", "SodiumChloride", {1048576, 8, 1}, 64),   // 1 GiB of working memory
              "2101cb9b6a511aaeaddbbe09cf70f881ec568d574a2ffd4dabe5ee9820adaa478e56fd8f4ba5d09ffa1c6d927c40f4c337304049e8a952fbcbf45c6fa77a41a4");
#endif
}

TEST(ScryptOracle, PythonHashlibScryptOnRandomInputs)
{
    const auto salt = test::oracle_key(16);
    for (const auto& c : test::read_oracle("oracle_mac_kdf.txt", "scrypt")) {
        const scrypt_params params{c.params.at(0), static_cast<std::uint32_t>(c.params.at(1)), static_cast<std::uint32_t>(c.params.at(2))};
        std::vector<std::uint32_t> work(scrypt_work_words(params).unwrap_unchecked());
        std::vector<std::byte> out(48);
        ASSERT_TRUE(scrypt(test::oracle_message(c.length), salt, params, work, out));
        ASSERT_EQ(hex(out), c.hex) << "line " << c.line;
    }
}

TEST(ScryptMemory, TheWorkingMemoryIsWipedAndALargerBufferIsUntouchedBeyondTheNeed)
{
    const scrypt_params params{64, 2, 2};
    const std::size_t need = scrypt_work_words(params).unwrap_unchecked();
    std::vector<std::uint32_t> work(need + 8, 0xdeadbeefu);
    std::vector<std::byte> out(32);
    ASSERT_TRUE(scrypt("password", "salt", params, work, out));
    for (std::size_t i = 0; i < need; ++i)
        ASSERT_EQ(work[i], 0u) << i;
    for (std::size_t i = need; i < work.size(); ++i)
        ASSERT_EQ(work[i], 0xdeadbeefu);
}

TEST(ScryptErrors, BadArgumentsAreErrorsAndWriteNothing)
{
    std::vector<std::uint32_t> work(scrypt_work_words({16, 1, 1}).unwrap_unchecked(), 7u);
    std::vector<std::byte> out(16, std::byte{0x5a});
    EXPECT_EQ(scrypt("p", "s", {16, 1, 1}, std::span<std::uint32_t>(work).first(work.size() - 1), out).unwrap_err_unchecked(),
              kdf_error::insufficient_work_memory);
    EXPECT_EQ(scrypt("p", "s", {17, 1, 1}, work, out).unwrap_err_unchecked(), kdf_error::invalid_cost);
    EXPECT_EQ(scrypt("p", "s", {16, 0, 1}, work, out).unwrap_err_unchecked(), kdf_error::invalid_block_parameters);
    EXPECT_EQ(scrypt("p", "s", {16, 1, 1}, work, std::span<std::byte>{}).unwrap_err_unchecked(), kdf_error::empty_output);
    for (const auto w : work)
        EXPECT_EQ(w, 7u);
    for (const auto b : out)
        EXPECT_EQ(b, std::byte{0x5a});
}

} // namespace
