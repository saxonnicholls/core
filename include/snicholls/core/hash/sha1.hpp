//
//  sha1.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief SHA-1 (FIPS 180-4), usable at compile time; for the protocols that still name it, never for security.
/// \ingroup core
///
/// **Overview.** SHA-1 with the incremental interface of every hash in core (`update`, then `finish`) and a
/// one-shot `hash`, returning a 20-byte `digest<20>`. **SHA-1 is broken for collision resistance** (a chosen-prefix
/// collision costs a few tens of thousands of GPU-days, and identical-prefix collisions are public): never use it
/// for a signature, a certificate, an integrity check against an adversary or a password. It is here because
/// protocols fix it: RFC 9562's name-based (version 5) UUIDs (`uuid.hpp` uses it), the WebSocket handshake
/// (RFC 6455), Git's object names, RFC 6070's PBKDF2-HMAC-SHA1 and TOTP / HOTP (RFC 6238 / 4226), where
/// HMAC-SHA1 remains sound because HMAC does not rely on collision resistance.
///
/// \snippet hash/sha1_demo.cpp sha1
///
/// **Relationships.** `md_hash` and `digest<N>` (`hash/digest.hpp`); `sha256` (`hash/sha2.hpp`) for anything
/// new; `hmac<sha1>` (`hash/hmac.hpp`). Group: \ref core.
///
/// **Guarantees.** Every member is `constexpr` and `noexcept`; no allocation; a hasher is a trivially copyable
/// value. Complexity: one 80-round compression per 64-byte block.
///
/// **Errors.** None.
///
/// **Examples.** Above; `demos/hash/sha1_demo.cpp`.
///
/// **Performance.** `benchmarks/hash/sha1_benchmark.cpp`, against a hand-written loop and OpenSSL (when
/// installed), after a fairness check. Release, Apple clang 17, Xeon W-3245 (a shared machine), 2026-11-20, 16 KiB / 64-byte messages: core 346 / 168
/// MB/s, the hand loop 249 / 137, OpenSSL 1,130 / 138 (its SIMD message schedule).
///
/// **Tests and demos.** `tests/hash/sha1_test.cpp` (FIPS 180-4 examples, NIST's byte-oriented short-message and
/// Monte Carlo vectors, Python's `hashlib` on random lengths, every split), `demos/hash/sha1_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<cstddef>`, `<cstdint>`, `hash/digest.hpp`.
///
/// **Known issues.** See the warning above: SHA-1 is kept for compatibility only. No collision-detection
/// variant (Git's sha1dc) is provided.
///
/// **Status.** New in 0.17.0 (it was internal, `detail/sha1.hpp`, behind `uuid`). Experimental.

#ifndef snicholls_core_hash_sha1_hpp
#define snicholls_core_hash_sha1_hpp

#include <snicholls/core/detail/hash_engine.hpp>
#include <snicholls/core/hash/digest.hpp>

#include <array>
#include <cstddef>
#include <cstdint>

namespace snicholls {

namespace detail {

    struct sha1_policy
    {
        using word = std::uint32_t;
        using state = std::array<word, 5>;
        static constexpr std::size_t block_size = 64;
        static constexpr std::size_t length_size = 8;
        static constexpr std::size_t digest_size = 20;
        static constexpr bool big_endian = true;
        static constexpr state initial{0x67452301u, 0xefcdab89u, 0x98badcfeu, 0x10325476u, 0xc3d2e1f0u};

        // Five rounds at a time with the roles of a..e rotated through the calls; each run of 20 rounds has its
        // own function and constant, so the selection is outside the rounds.
        template <typename F>
        static constexpr void twenty(word& a, word& b, word& c, word& d, word& e, word (&w)[16], std::size_t t0, word k,
                                     F f) noexcept
        {
            const auto round = [&](word x0, word& x1, word x2, word x3, word& x4, std::size_t t) {
                if (t >= 16)
                    w[t & 15] = rotl<word>(w[(t + 13) & 15] ^ w[(t + 8) & 15] ^ w[(t + 2) & 15] ^ w[t & 15], 1);
                x4 = static_cast<word>(x4 + rotl<word>(x0, 5) + f(x1, x2, x3) + k + w[t & 15]);
                x1 = rotl<word>(x1, 30);
            };
            for (std::size_t t = t0; t < t0 + 20; t += 5) {
                round(a, b, c, d, e, t);
                round(e, a, b, c, d, t + 1);
                round(d, e, a, b, c, t + 2);
                round(c, d, e, a, b, t + 3);
                round(b, c, d, e, a, t + 4);
            }
        }

        static constexpr void compress(state& h, const word (&block)[16]) noexcept
        {
            word w[16]{};
            for (std::size_t i = 0; i < 16; ++i)
                w[i] = block[i];
            word a = h[0], b = h[1], c = h[2], d = h[3], e = h[4];
            twenty(a, b, c, d, e, w, 0, 0x5a827999u, [](word x, word y, word z) { return static_cast<word>((x & y) | (~x & z)); });
            twenty(a, b, c, d, e, w, 20, 0x6ed9eba1u, [](word x, word y, word z) { return static_cast<word>(x ^ y ^ z); });
            twenty(a, b, c, d, e, w, 40, 0x8f1bbcdcu, [](word x, word y, word z) { return static_cast<word>((x & y) | (x & z) | (y & z)); });
            twenty(a, b, c, d, e, w, 60, 0xca62c1d6u, [](word x, word y, word z) { return static_cast<word>(x ^ y ^ z); });
            h[0] += a;
            h[1] += b;
            h[2] += c;
            h[3] += d;
            h[4] += e;
        }
    };

} // namespace detail

/// \brief SHA-1 (FIPS 180-4): 20 bytes of output over 64-byte blocks. Broken for collisions; compatibility only.
/// \ingroup core
using sha1 = md_hash<detail::sha1_policy>;

} // namespace snicholls

#endif /* snicholls_core_hash_sha1_hpp */
