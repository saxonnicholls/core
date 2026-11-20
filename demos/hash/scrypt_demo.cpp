//
//  scrypt_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/hash/scrypt.hpp>: a password hashed with scrypt in working memory the caller owns,
//  RFC 7914's vector, and the parameter errors. Exits non-zero if any value is not the one documented.
//

#include <snicholls/core/hash/scrypt.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace {

//! [scrypt]
// The caller owns the working memory: ask for its size, provide it, reuse it between calls
snicholls::result<snicholls::digest<32>, snicholls::kdf_error> password_key(const std::string& password, const std::string& salt,
                                                                            std::vector<std::uint32_t>& work)
{
    const snicholls::scrypt_params params{1u << 14, 8, 1};   // 16 MiB
    SNICHOLLS_TRY_ASSIGN(const std::size_t words, snicholls::scrypt_work_words(params));
    work.resize(words);
    snicholls::digest<32> key{};
    SNICHOLLS_TRY_VOID(snicholls::scrypt(password, salt, params, work, key.bytes));
    return snicholls::ok(key);
}
//! [scrypt]

} // namespace

int main()
{
    std::printf("snicholls-core scrypt demo\n");
    bool ok = true;

    std::vector<std::uint32_t> work;
    const auto key = password_key("correct horse battery staple", "per-user random salt", work);
    std::printf("  key = %s (working memory %zu KiB, wiped)\n", key ? key.unwrap_unchecked().hex().c_str() : "error", work.size() * 4 / 1024);
    ok &= key.is_ok() && work.size() == 32 * 8 * ((1u << 14) + 2);
    bool wiped = true;
    for (const auto w : work)
        wiped &= w == 0;
    ok &= wiped;

    // RFC 7914's second vector
    std::vector<std::uint32_t> big(snicholls::scrypt_work_words({1024, 8, 16}).unwrap_unchecked());
    std::array<std::byte, 64> dk{};
    ok &= snicholls::scrypt("password", "NaCl", {1024, 8, 16}, big, dk).is_ok();
    ok &= snicholls::digest<64>{dk}.first<8>().hex().view() == "fdbabe1c9d347200";

    // Errors: a cost that is not a power of two, too little memory
    ok &= snicholls::scrypt_work_words({1000, 8, 1}).unwrap_err_unchecked() == snicholls::kdf_error::invalid_cost;
    std::vector<std::uint32_t> small(16);
    ok &= snicholls::scrypt("p", "s", {16, 1, 1}, small, dk).unwrap_err_unchecked() == snicholls::kdf_error::insufficient_work_memory;
    std::printf("  N = 1000 and a 64-byte working memory are refused: %s\n", ok ? "yes" : "no");

    std::printf(ok ? "scrypt demo: ok\n" : "scrypt demo: FAILED\n");
    return ok ? 0 : 1;
}
