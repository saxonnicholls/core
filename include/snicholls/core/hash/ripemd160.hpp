//
//  ripemd160.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief RIPEMD-160 (Dobbertin, Bosselaers and Preneel, 1996), usable at compile time: Bitcoin's `hash160`.
/// \ingroup core
///
/// **Overview.** RIPEMD-160 with the incremental interface of every hash in core and a one-shot `hash`, returning
/// a `digest<20>`. It survives in one place that matters: `hash160(x) = RIPEMD-160(SHA-256(x))` names Bitcoin
/// addresses (P2PKH, P2SH, P2WPKH) and the BIP-32 key fingerprint, and Cosmos derives account addresses the same
/// way. Two parallel lines of 80 rounds over 64-byte blocks, little-endian words. For anything new use SHA-256 or
/// BLAKE2.
///
/// \snippet hash/ripemd160_demo.cpp ripemd160
///
/// **Relationships.** `md_hash` and `digest<N>` (`hash/digest.hpp`); `sha256` (`hash/sha2.hpp`) for `hash160`.
/// Group: \ref core.
///
/// **Guarantees.** Every member is `constexpr` and `noexcept`; no allocation; a hasher is a trivially copyable
/// value. Complexity: one compression (160 steps) per block.
///
/// **Errors.** None.
///
/// **Examples.** Above; `demos/hash/ripemd160_demo.cpp`.
///
/// **Performance.** `benchmarks/hash/ripemd160_benchmark.cpp`, against a hand-written loop and OpenSSL (when it
/// provides the algorithm), after a fairness check; figures in BENCHMARKS.md.
///
/// **Tests and demos.** `tests/hash/ripemd160_test.cpp` (the designers' published vectors, including a million
/// `a`, Python's `hashlib` on random lengths, every split), `demos/hash/ripemd160_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<cstddef>`, `<cstdint>`, `hash/digest.hpp`.
///
/// **Known issues.** OpenSSL 3 moved RIPEMD-160 to its legacy provider, so some systems' `hashlib` lack it; the
/// committed fixture does not depend on it at test time.
///
/// **Status.** New in 0.17.0. Experimental.

#ifndef snicholls_core_hash_ripemd160_hpp
#define snicholls_core_hash_ripemd160_hpp

#include <snicholls/core/detail/hash_engine.hpp>
#include <snicholls/core/hash/digest.hpp>

#include <array>
#include <cstddef>
#include <cstdint>

namespace snicholls {

namespace detail {

    struct ripemd160_policy
    {
        using word = std::uint32_t;
        using state = std::array<word, 5>;
        static constexpr std::size_t block_size = 64;
        static constexpr std::size_t length_size = 8;
        static constexpr std::size_t digest_size = 20;
        static constexpr bool big_endian = false;
        static constexpr state initial{0x67452301u, 0xefcdab89u, 0x98badcfeu, 0x10325476u, 0xc3d2e1f0u};

        // The message word and rotation of each step, left line then right line.
        static constexpr std::uint8_t rl[80] = {0, 1, 2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 7,  4,  13, 1,
                                                10, 6, 15, 3,  12, 0,  9,  5,  2,  14, 11, 8,  3,  10, 14, 4,  9,  15, 8,  1,
                                                2,  7, 0,  6,  13, 11, 5,  12, 1,  9,  11, 10, 0,  8,  12, 4,  13, 3,  7,  15,
                                                14, 5, 6,  2,  4,  0,  5,  9,  7,  12, 2,  10, 14, 1,  3,  8,  11, 6,  15, 13};
        static constexpr std::uint8_t rr[80] = {5,  14, 7,  0, 9, 2,  11, 4,  13, 6,  15, 8,  1,  10, 3,  12, 6,  11, 3,  7,
                                                0,  13, 5,  10, 14, 15, 8,  12, 4,  9,  1,  2,  15, 5,  1,  3,  7,  14, 6,  9,
                                                11, 8,  12, 2,  10, 0,  4,  13, 8,  6,  4,  1,  3,  11, 15, 0,  5,  12, 2,  13,
                                                9,  7,  10, 14, 12, 15, 10, 4,  1,  5,  8,  7,  6,  2,  13, 14, 0,  3,  9,  11};
        static constexpr std::uint8_t sl[80] = {11, 14, 15, 12, 5,  8,  7,  9,  11, 13, 14, 15, 6,  7,  9,  8,  7,  6,  8,  13,
                                                11, 9,  7,  15, 7,  12, 15, 9,  11, 7,  13, 12, 11, 13, 6,  7,  14, 9,  13, 15,
                                                14, 8,  13, 6,  5,  12, 7,  5,  11, 12, 14, 15, 14, 15, 9,  8,  9,  14, 5,  6,
                                                8,  6,  5,  12, 9,  15, 5,  11, 6,  8,  13, 12, 5,  12, 13, 14, 11, 8,  5,  6};
        static constexpr std::uint8_t sr[80] = {8,  9,  9,  11, 13, 15, 15, 5,  7,  7,  8,  11, 14, 14, 12, 6,  9,  13, 15, 7,
                                                12, 8,  9,  11, 7,  7,  12, 7,  6,  15, 13, 11, 9,  7,  15, 11, 8,  6,  6,  14,
                                                12, 13, 5,  14, 13, 13, 7,  5,  15, 5,  8,  11, 14, 14, 6,  14, 6,  9,  12, 9,
                                                12, 5,  15, 8,  8,  5,  12, 9,  12, 5,  14, 6,  8,  13, 6,  5,  15, 13, 11, 11};

        static constexpr word f(std::size_t j, word x, word y, word z) noexcept
        {
            return j < 16 ? x ^ y ^ z : j < 32 ? (x & y) | (~x & z) : j < 48 ? (x | ~y) ^ z : j < 64 ? (x & z) | (y & ~z) : x ^ (y | ~z);
        }

        static constexpr void compress(state& h, const word (&x)[16]) noexcept
        {
            constexpr word kl[5] = {0x00000000u, 0x5a827999u, 0x6ed9eba1u, 0x8f1bbcdcu, 0xa953fd4eu};
            constexpr word kr[5] = {0x50a28be6u, 0x5c4dd124u, 0x6d703ef3u, 0x7a6d76e9u, 0x00000000u};
            word al = h[0], bl = h[1], cl = h[2], dl = h[3], el = h[4];
            word ar = h[0], br = h[1], cr = h[2], dr = h[3], er = h[4];
            for (std::size_t j = 0; j < 80; ++j) {
                word t = rotl<word>(al + f(j, bl, cl, dl) + x[rl[j]] + kl[j / 16], sl[j]) + el;
                al = el;
                el = dl;
                dl = rotl<word>(cl, 10);
                cl = bl;
                bl = t;
                t = rotl<word>(ar + f(79 - j, br, cr, dr) + x[rr[j]] + kr[j / 16], sr[j]) + er;
                ar = er;
                er = dr;
                dr = rotl<word>(cr, 10);
                cr = br;
                br = t;
            }
            const word t = h[1] + cl + dr;
            h[1] = h[2] + dl + er;
            h[2] = h[3] + el + ar;
            h[3] = h[4] + al + br;
            h[4] = h[0] + bl + cr;
            h[0] = t;
        }
    };

} // namespace detail

/// \brief RIPEMD-160: 20 bytes of output over 64-byte blocks, little-endian words.
/// \ingroup core
using ripemd160 = md_hash<detail::ripemd160_policy>;

} // namespace snicholls

#endif /* snicholls_core_hash_ripemd160_hpp */
