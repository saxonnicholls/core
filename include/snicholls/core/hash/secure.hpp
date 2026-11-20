//
//  secure.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `secure_wipe`, which the optimiser may not remove, and `constant_time_equal` for MACs and digests.
/// \ingroup core
///
/// **Overview.** Two small tools every use of a key or a MAC needs.
///
/// - `secure_wipe(span)` zeroes memory that held a secret (a key, a password, a KDF's working memory) just before
///   it goes out of scope or is freed. A plain `memset` or loop there is a dead store the optimiser may delete,
///   because nothing reads the memory afterwards; `secure_wipe` writes through a `volatile` pointer, which it may
///   not, followed by a compiler fence. In a constant expression it is an ordinary zero fill.
/// - `constant_time_equal(a, b)` compares two byte sequences in time that depends only on their length, never on
///   where they first differ, so an attacker who can time a MAC check cannot learn the correct tag a byte at a
///   time. Use it, not `==`, to check a MAC, a password hash or any digest an attacker chooses; `hmac::verify`
///   uses it.
///
/// \snippet hash/secure_demo.cpp secure
///
/// **Relationships.** `hmac::verify` (`hash/hmac.hpp`); `scrypt` (`hash/scrypt.hpp`), `pbkdf2_hmac` and `hkdf_*`
/// (`hash/kdf.hpp`) wipe their intermediate keys with it; `secret<T>` (`secret.hpp`) wipes a string with it when
/// it is destroyed. Group: \ref core.
///
/// **Guarantees.** Both are `constexpr` and `noexcept`. `secure_wipe` writes every byte of the span exactly once;
/// it cannot reach copies the program made elsewhere (registers, spilled stack, a reallocated vector's old
/// buffer), which no library function can. `constant_time_equal` returns `false` at once when the lengths differ
/// (a length is not secret) and otherwise reads every byte of both, accumulating the differences with no branch on
/// their values; the accumulator is read back through a `volatile` so the optimiser cannot turn the loop into an
/// early exit.
///
/// **Errors.** None.
///
/// **Examples.** Above; `demos/hash/secure_demo.cpp`.
///
/// **Performance.** Not benchmarked: both are one pass over the bytes, and neither is on a hot path (a wipe ends a
/// key's life, a comparison ends a MAC check).
///
/// **Tests and demos.** `tests/hash/secure_test.cpp`, `demos/hash/secure_demo.cpp`.
///
/// **Dependencies.** `<cstddef>`, `<span>`, `<type_traits>`, `hash/digest.hpp`, `detail/hash_engine.hpp`.
///
/// **Known issues.** Constant time is a property of the generated code, which the standard does not specify; this
/// is the pattern libsodium and BoringSSL use for portable code, not a proof. Without platform headers there is no
/// `explicit_bzero`, `memset_s` or `SecureZeroMemory`; the volatile loop is what they do.
///
/// **Status.** New in 0.17.0. Experimental.

#ifndef snicholls_core_hash_secure_hpp
#define snicholls_core_hash_secure_hpp

#include <snicholls/core/detail/hash_engine.hpp>
#include <snicholls/core/hash/digest.hpp>

#include <cstddef>
#include <span>
#include <type_traits>

namespace snicholls {

/// \brief Zeroes `s` in a way the optimiser may not remove, for memory that held a secret.
/// \ingroup core
/// \tparam T A trivially copyable element type (bytes, words, a `digest<N>`).
/// \tparam Extent The span's extent.
/// \param s The memory.
template <typename T, std::size_t Extent>
    requires(std::is_trivially_copyable_v<T> && !std::is_const_v<T>)
constexpr void secure_wipe(std::span<T, Extent> s) noexcept
{
    if (std::is_constant_evaluated()) {
        for (T& x : s)
            x = T{};
        return;
    }
    if (!s.empty())
        detail::volatile_zero(static_cast<void*>(s.data()), s.size_bytes());
}

/// \brief Zeroes a digest (a derived key, a MAC) in a way the optimiser may not remove.
/// \ingroup core
/// \param d The digest.
template <std::size_t N>
constexpr void secure_wipe(digest<N>& d) noexcept
{
    secure_wipe(std::span<std::byte, N>(d.bytes));
}

/// \brief Whether `a` and `b` hold the same bytes, in time independent of where they differ.
/// \ingroup core
/// \param a One sequence (any `hash_input`: bytes, text, a digest).
/// \param b The other.
/// \return `true` when the lengths and every byte are equal.
template <hash_input A, hash_input B>
[[nodiscard]] constexpr bool constant_time_equal(const A& a, const B& b) noexcept
{
    const std::size_t n = detail::input_size(a);
    if (n != detail::input_size(b))
        return false;
    const auto* pa = detail::input_data(a);
    const auto* pb = detail::input_data(b);
    unsigned difference = 0;
    for (std::size_t i = 0; i < n; ++i)
        difference |= static_cast<unsigned>(detail::octet_of(pa[i]) ^ detail::octet_of(pb[i]));
    if (std::is_constant_evaluated())
        return difference == 0;
    return detail::volatile_is_zero(difference);
}

} // namespace snicholls

#endif /* snicholls_core_hash_secure_hpp */
