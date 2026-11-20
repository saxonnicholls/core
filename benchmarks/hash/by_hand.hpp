//
//  by_hand.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  The hand-written equivalents the hash benchmarks compare core against: each algorithm as it is usually written
//  for run time only, from its specification, with memcpy loads and the compiler's byte swap, no constexpr and no
//  templates. The arithmetic of every round is the specification's, in the same order as core's. Also the
//  informational OpenSSL comparison (EVP), compiled only when CMake found OpenSSL; core never depends on it.
//

#ifndef snicholls_core_benchmarks_hash_by_hand_hpp
#define snicholls_core_benchmarks_hash_by_hand_hpp

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#if defined(_MSC_VER) && !defined(__clang__)
#include <stdlib.h>
#endif

#if SNICHOLLS_CORE_BENCH_OPENSSL
#include <openssl/evp.h>
#endif

namespace by_hand {

inline std::uint32_t rotr32(std::uint32_t x, unsigned n) { return (x >> n) | (x << (32 - n)); }
inline std::uint32_t rotl32(std::uint32_t x, unsigned n) { return (x << n) | (x >> (32 - n)); }
inline std::uint64_t rotr64(std::uint64_t x, unsigned n) { return (x >> n) | (x << (64 - n)); }
inline std::uint64_t rotl64(std::uint64_t x, unsigned n) { return (x << n) | (x >> (64 - n)); }

#if defined(_MSC_VER) && !defined(__clang__)
inline std::uint32_t bswap32(std::uint32_t v) { return _byteswap_ulong(v); }
inline std::uint64_t bswap64(std::uint64_t v) { return _byteswap_uint64(v); }
#else
inline std::uint32_t bswap32(std::uint32_t v) { return __builtin_bswap32(v); }
inline std::uint64_t bswap64(std::uint64_t v) { return __builtin_bswap64(v); }
#endif

inline std::uint32_t be32(const unsigned char* p)
{
    std::uint32_t v;
    std::memcpy(&v, p, 4);
    return bswap32(v);
}
inline std::uint64_t be64(const unsigned char* p)
{
    std::uint64_t v;
    std::memcpy(&v, p, 8);
    return bswap64(v);
}
inline std::uint32_t le32(const unsigned char* p)
{
    std::uint32_t v;
    std::memcpy(&v, p, 4);
    return v;
}
inline std::uint64_t le64(const unsigned char* p)
{
    std::uint64_t v;
    std::memcpy(&v, p, 8);
    return v;
}

using bytes = std::vector<unsigned char>;

// The benchmarks' data: messages of `size` bytes, 256 KiB in all (at least one), from a fixed xorshift stream
inline std::vector<bytes> corpus(std::size_t size)
{
    std::vector<bytes> out((256 * 1024) / size == 0 ? 1 : (256 * 1024) / size, bytes(size));
    std::uint64_t x = 0x243f6a8885a308d3ull;
    for (auto& m : out)
        for (auto& c : m) {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            c = static_cast<unsigned char>(x);
        }
    return out;
}

inline std::int64_t corpus_bytes(const std::vector<bytes>& c) { return static_cast<std::int64_t>(c.size() * c.front().size()); }

// Merkle-Damgard padding around a block function: whole blocks from the input, then the padded tail.
template <std::size_t Block, std::size_t LengthBytes, bool BigEndian, typename F>
void md_pad(const unsigned char* p, std::size_t n, F&& compress)
{
    const std::size_t whole = n / Block * Block;
    for (std::size_t i = 0; i < whole; i += Block)
        compress(p + i);
    unsigned char tail[2 * Block] = {};
    const std::size_t rest = n - whole;
    std::memcpy(tail, p + whole, rest);
    tail[rest] = 0x80;
    const std::size_t total = rest + 1 + LengthBytes <= Block ? Block : 2 * Block;
    const std::uint64_t bits = static_cast<std::uint64_t>(n) * 8;
    for (std::size_t i = 0; i < 8; ++i)
        tail[BigEndian ? total - 1 - i : total - 8 + i] = static_cast<unsigned char>(bits >> (8 * i));
    compress(tail);
    if (total == 2 * Block)
        compress(tail + Block);
}

// ---- SHA-256 -----------------------------------------------------------------------------------------------
inline constexpr std::uint32_t k256[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be,
    0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa,
    0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85,
    0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f,
    0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

inline void sha256_block(std::uint32_t* h, const unsigned char* p)
{
    std::uint32_t w[64];
    for (int i = 0; i < 16; ++i)
        w[i] = be32(p + 4 * i);
    for (int i = 16; i < 64; ++i) {
        const std::uint32_t s0 = rotr32(w[i - 15], 7) ^ rotr32(w[i - 15], 18) ^ (w[i - 15] >> 3);
        const std::uint32_t s1 = rotr32(w[i - 2], 17) ^ rotr32(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    std::uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
    for (int i = 0; i < 64; ++i) {
        const std::uint32_t t1 = hh + (rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25)) + ((e & f) ^ (~e & g)) + k256[i] + w[i];
        const std::uint32_t t2 = (rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
        hh = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
}

inline std::array<unsigned char, 32> sha256(const unsigned char* p, std::size_t n)
{
    std::uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    md_pad<64, 8, true>(p, n, [&h](const unsigned char* b) { sha256_block(h, b); });
    std::array<unsigned char, 32> out{};
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 4; ++j)
            out[4 * i + j] = static_cast<unsigned char>(h[i] >> (24 - 8 * j));
    return out;
}

// ---- SHA-512 -----------------------------------------------------------------------------------------------
inline constexpr std::uint64_t k512[80] = {
    0x428a2f98d728ae22, 0x7137449123ef65cd, 0xb5c0fbcfec4d3b2f, 0xe9b5dba58189dbbc, 0x3956c25bf348b538, 0x59f111f1b605d019,
    0x923f82a4af194f9b, 0xab1c5ed5da6d8118, 0xd807aa98a3030242, 0x12835b0145706fbe, 0x243185be4ee4b28c, 0x550c7dc3d5ffb4e2,
    0x72be5d74f27b896f, 0x80deb1fe3b1696b1, 0x9bdc06a725c71235, 0xc19bf174cf692694, 0xe49b69c19ef14ad2, 0xefbe4786384f25e3,
    0x0fc19dc68b8cd5b5, 0x240ca1cc77ac9c65, 0x2de92c6f592b0275, 0x4a7484aa6ea6e483, 0x5cb0a9dcbd41fbd4, 0x76f988da831153b5,
    0x983e5152ee66dfab, 0xa831c66d2db43210, 0xb00327c898fb213f, 0xbf597fc7beef0ee4, 0xc6e00bf33da88fc2, 0xd5a79147930aa725,
    0x06ca6351e003826f, 0x142929670a0e6e70, 0x27b70a8546d22ffc, 0x2e1b21385c26c926, 0x4d2c6dfc5ac42aed, 0x53380d139d95b3df,
    0x650a73548baf63de, 0x766a0abb3c77b2a8, 0x81c2c92e47edaee6, 0x92722c851482353b, 0xa2bfe8a14cf10364, 0xa81a664bbc423001,
    0xc24b8b70d0f89791, 0xc76c51a30654be30, 0xd192e819d6ef5218, 0xd69906245565a910, 0xf40e35855771202a, 0x106aa07032bbd1b8,
    0x19a4c116b8d2d0c8, 0x1e376c085141ab53, 0x2748774cdf8eeb99, 0x34b0bcb5e19b48a8, 0x391c0cb3c5c95a63, 0x4ed8aa4ae3418acb,
    0x5b9cca4f7763e373, 0x682e6ff3d6b2b8a3, 0x748f82ee5defb2fc, 0x78a5636f43172f60, 0x84c87814a1f0ab72, 0x8cc702081a6439ec,
    0x90befffa23631e28, 0xa4506cebde82bde9, 0xbef9a3f7b2c67915, 0xc67178f2e372532b, 0xca273eceea26619c, 0xd186b8c721c0c207,
    0xeada7dd6cde0eb1e, 0xf57d4f7fee6ed178, 0x06f067aa72176fba, 0x0a637dc5a2c898a6, 0x113f9804bef90dae, 0x1b710b35131c471b,
    0x28db77f523047d84, 0x32caab7b40c72493, 0x3c9ebe0a15c9bebc, 0x431d67c49c100d4c, 0x4cc5d4becb3e42b6, 0x597f299cfc657e2a,
    0x5fcb6fab3ad6faec, 0x6c44198c4a475817};

inline void sha512_block(std::uint64_t* h, const unsigned char* p)
{
    std::uint64_t w[80];
    for (int i = 0; i < 16; ++i)
        w[i] = be64(p + 8 * i);
    for (int i = 16; i < 80; ++i) {
        const std::uint64_t s0 = rotr64(w[i - 15], 1) ^ rotr64(w[i - 15], 8) ^ (w[i - 15] >> 7);
        const std::uint64_t s1 = rotr64(w[i - 2], 19) ^ rotr64(w[i - 2], 61) ^ (w[i - 2] >> 6);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    std::uint64_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
    for (int i = 0; i < 80; ++i) {
        const std::uint64_t t1 = hh + (rotr64(e, 14) ^ rotr64(e, 18) ^ rotr64(e, 41)) + ((e & f) ^ (~e & g)) + k512[i] + w[i];
        const std::uint64_t t2 = (rotr64(a, 28) ^ rotr64(a, 34) ^ rotr64(a, 39)) + ((a & b) ^ (a & c) ^ (b & c));
        hh = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
}

inline std::array<unsigned char, 64> sha512(const unsigned char* p, std::size_t n)
{
    std::uint64_t h[8] = {0x6a09e667f3bcc908, 0xbb67ae8584caa73b, 0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
                          0x510e527fade682d1, 0x9b05688c2b3e6c1f, 0x1f83d9abfb41bd6b, 0x5be0cd19137e2179};
    md_pad<128, 16, true>(p, n, [&h](const unsigned char* b) { sha512_block(h, b); });
    std::array<unsigned char, 64> out{};
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j)
            out[8 * i + j] = static_cast<unsigned char>(h[i] >> (56 - 8 * j));
    return out;
}

// ---- SHA-1 -------------------------------------------------------------------------------------------------
inline std::array<unsigned char, 20> sha1(const unsigned char* p, std::size_t n)
{
    std::uint32_t h[5] = {0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476, 0xc3d2e1f0};
    md_pad<64, 8, true>(p, n, [&h](const unsigned char* blk) {
        std::uint32_t w[80];
        for (int i = 0; i < 16; ++i)
            w[i] = be32(blk + 4 * i);
        for (int i = 16; i < 80; ++i)
            w[i] = rotl32(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
        std::uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4];
        for (int i = 0; i < 80; ++i) {
            std::uint32_t f, k;
            if (i < 20) { f = (b & c) | (~b & d); k = 0x5a827999; }
            else if (i < 40) { f = b ^ c ^ d; k = 0x6ed9eba1; }
            else if (i < 60) { f = (b & c) | (b & d) | (c & d); k = 0x8f1bbcdc; }
            else { f = b ^ c ^ d; k = 0xca62c1d6; }
            const std::uint32_t t = rotl32(a, 5) + f + e + k + w[i];
            e = d;
            d = c;
            c = rotl32(b, 30);
            b = a;
            a = t;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e;
    });
    std::array<unsigned char, 20> out{};
    for (int i = 0; i < 5; ++i)
        for (int j = 0; j < 4; ++j)
            out[4 * i + j] = static_cast<unsigned char>(h[i] >> (24 - 8 * j));
    return out;
}

// ---- RIPEMD-160 --------------------------------------------------------------------------------------------
inline std::array<unsigned char, 20> ripemd160(const unsigned char* p, std::size_t n)
{
    static constexpr int rl[80] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 7, 4, 13, 1, 10, 6, 15, 3, 12, 0, 9, 5, 2, 14, 11, 8,
                                   3, 10, 14, 4, 9, 15, 8, 1, 2, 7, 0, 6, 13, 11, 5, 12, 1, 9, 11, 10, 0, 8, 12, 4, 13, 3, 7, 15, 14, 5, 6, 2,
                                   4, 0, 5, 9, 7, 12, 2, 10, 14, 1, 3, 8, 11, 6, 15, 13};
    static constexpr int rr[80] = {5, 14, 7, 0, 9, 2, 11, 4, 13, 6, 15, 8, 1, 10, 3, 12, 6, 11, 3, 7, 0, 13, 5, 10, 14, 15, 8, 12, 4, 9, 1, 2,
                                   15, 5, 1, 3, 7, 14, 6, 9, 11, 8, 12, 2, 10, 0, 4, 13, 8, 6, 4, 1, 3, 11, 15, 0, 5, 12, 2, 13, 9, 7, 10, 14,
                                   12, 15, 10, 4, 1, 5, 8, 7, 6, 2, 13, 14, 0, 3, 9, 11};
    static constexpr unsigned sl[80] = {11, 14, 15, 12, 5, 8, 7, 9, 11, 13, 14, 15, 6, 7, 9, 8, 7, 6, 8, 13, 11, 9, 7, 15, 7, 12, 15, 9, 11, 7, 13, 12,
                                        11, 13, 6, 7, 14, 9, 13, 15, 14, 8, 13, 6, 5, 12, 7, 5, 11, 12, 14, 15, 14, 15, 9, 8, 9, 14, 5, 6, 8, 6, 5, 12,
                                        9, 15, 5, 11, 6, 8, 13, 12, 5, 12, 13, 14, 11, 8, 5, 6};
    static constexpr unsigned sr[80] = {8, 9, 9, 11, 13, 15, 15, 5, 7, 7, 8, 11, 14, 14, 12, 6, 9, 13, 15, 7, 12, 8, 9, 11, 7, 7, 12, 7, 6, 15, 13, 11,
                                        9, 7, 15, 11, 8, 6, 6, 14, 12, 13, 5, 14, 13, 13, 7, 5, 15, 5, 8, 11, 14, 14, 6, 14, 6, 9, 12, 9, 12, 5, 15, 8,
                                        8, 5, 12, 9, 12, 5, 14, 6, 8, 13, 6, 5, 15, 13, 11, 11};
    static constexpr std::uint32_t kl[5] = {0x00000000, 0x5a827999, 0x6ed9eba1, 0x8f1bbcdc, 0xa953fd4e};
    static constexpr std::uint32_t kr[5] = {0x50a28be6, 0x5c4dd124, 0x6d703ef3, 0x7a6d76e9, 0x00000000};
    const auto f = [](int j, std::uint32_t x, std::uint32_t y, std::uint32_t z) -> std::uint32_t {
        if (j < 16) return x ^ y ^ z;
        if (j < 32) return (x & y) | (~x & z);
        if (j < 48) return (x | ~y) ^ z;
        if (j < 64) return (x & z) | (y & ~z);
        return x ^ (y | ~z);
    };
    std::uint32_t h[5] = {0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476, 0xc3d2e1f0};
    md_pad<64, 8, false>(p, n, [&](const unsigned char* blk) {
        std::uint32_t x[16];
        for (int i = 0; i < 16; ++i)
            x[i] = le32(blk + 4 * i);
        std::uint32_t al = h[0], bl = h[1], cl = h[2], dl = h[3], el = h[4];
        std::uint32_t ar = h[0], br = h[1], cr = h[2], dr = h[3], er = h[4];
        for (int j = 0; j < 80; ++j) {
            std::uint32_t t = rotl32(al + f(j, bl, cl, dl) + x[rl[j]] + kl[j / 16], sl[j]) + el;
            al = el; el = dl; dl = rotl32(cl, 10); cl = bl; bl = t;
            t = rotl32(ar + f(79 - j, br, cr, dr) + x[rr[j]] + kr[j / 16], sr[j]) + er;
            ar = er; er = dr; dr = rotl32(cr, 10); cr = br; br = t;
        }
        const std::uint32_t t = h[1] + cl + dr;
        h[1] = h[2] + dl + er; h[2] = h[3] + el + ar; h[3] = h[4] + al + br; h[4] = h[0] + bl + cr; h[0] = t;
    });
    std::array<unsigned char, 20> out{};
    std::memcpy(out.data(), h, 20);   // little-endian host
    return out;
}

// ---- Keccak ------------------------------------------------------------------------------------------------
inline void keccak_f(std::uint64_t* a)
{
    static constexpr std::uint64_t rc[24] = {
        0x0000000000000001, 0x0000000000008082, 0x800000000000808a, 0x8000000080008000, 0x000000000000808b, 0x0000000080000001,
        0x8000000080008081, 0x8000000000008009, 0x000000000000008a, 0x0000000000000088, 0x0000000080008009, 0x000000008000000a,
        0x000000008000808b, 0x800000000000008b, 0x8000000000008089, 0x8000000000008003, 0x8000000000008002, 0x8000000000000080,
        0x000000000000800a, 0x800000008000000a, 0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008};
    static constexpr unsigned rho[24] = {1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14, 27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44};
    static constexpr int pi[24] = {10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1};
    for (int round = 0; round < 24; ++round) {
        std::uint64_t c[5];
        for (int x = 0; x < 5; ++x)
            c[x] = a[x] ^ a[x + 5] ^ a[x + 10] ^ a[x + 15] ^ a[x + 20];
        for (int x = 0; x < 5; ++x) {
            const std::uint64_t d = c[(x + 4) % 5] ^ rotl64(c[(x + 1) % 5], 1);
            for (int y = 0; y < 25; y += 5)
                a[y + x] ^= d;
        }
        std::uint64_t t = a[1];
        for (int i = 0; i < 24; ++i) {
            const std::uint64_t next = a[pi[i]];
            a[pi[i]] = rotl64(t, rho[i]);
            t = next;
        }
        for (int y = 0; y < 25; y += 5) {
            const std::uint64_t r0 = a[y], r1 = a[y + 1], r2 = a[y + 2], r3 = a[y + 3], r4 = a[y + 4];
            a[y] = r0 ^ (~r1 & r2); a[y + 1] = r1 ^ (~r2 & r3); a[y + 2] = r2 ^ (~r3 & r4); a[y + 3] = r3 ^ (~r4 & r0); a[y + 4] = r4 ^ (~r0 & r1);
        }
        a[0] ^= rc[round];
    }
}

inline std::array<unsigned char, 32> keccak_256_family(const unsigned char* p, std::size_t n, unsigned char pad)
{
    constexpr std::size_t rate = 136;
    std::uint64_t a[25] = {};
    for (; n >= rate; p += rate, n -= rate) {
        for (std::size_t i = 0; i < rate / 8; ++i)
            a[i] ^= le64(p + 8 * i);
        keccak_f(a);
    }
    unsigned char last[rate] = {};
    std::memcpy(last, p, n);
    last[n] ^= pad;
    last[rate - 1] ^= 0x80;
    for (std::size_t i = 0; i < rate / 8; ++i)
        a[i] ^= le64(last + 8 * i);
    keccak_f(a);
    std::array<unsigned char, 32> out{};
    std::memcpy(out.data(), a, 32);   // little-endian host
    return out;
}

// ---- BLAKE2 ------------------------------------------------------------------------------------------------
inline constexpr unsigned char sigma[12][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
    {11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4}, {7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8},
    {9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13}, {2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9},
    {12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11}, {13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10},
    {6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5}, {10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0},
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3}};

inline std::array<unsigned char, 64> blake2b_512(const unsigned char* p, std::size_t n)
{
    static constexpr std::uint64_t iv[8] = {0x6a09e667f3bcc908, 0xbb67ae8584caa73b, 0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
                                            0x510e527fade682d1, 0x9b05688c2b3e6c1f, 0x1f83d9abfb41bd6b, 0x5be0cd19137e2179};
    std::uint64_t h[8];
    for (int i = 0; i < 8; ++i) h[i] = iv[i];
    h[0] ^= 0x01010000 ^ 64;
    const auto compress = [&](const unsigned char* blk, std::uint64_t t, bool last) {
        std::uint64_t m[16], v[16];
        for (int i = 0; i < 16; ++i) m[i] = le64(blk + 8 * i);
        for (int i = 0; i < 8; ++i) { v[i] = h[i]; v[i + 8] = iv[i]; }
        v[12] ^= t;
        if (last) v[14] = ~v[14];
        const auto g = [&](int a, int b, int c, int d, std::uint64_t x, std::uint64_t y) {
            v[a] = v[a] + v[b] + x; v[d] = rotr64(v[d] ^ v[a], 32); v[c] = v[c] + v[d]; v[b] = rotr64(v[b] ^ v[c], 24);
            v[a] = v[a] + v[b] + y; v[d] = rotr64(v[d] ^ v[a], 16); v[c] = v[c] + v[d]; v[b] = rotr64(v[b] ^ v[c], 63);
        };
        for (int r = 0; r < 12; ++r) {
            const unsigned char* s = sigma[r];
            g(0, 4, 8, 12, m[s[0]], m[s[1]]); g(1, 5, 9, 13, m[s[2]], m[s[3]]); g(2, 6, 10, 14, m[s[4]], m[s[5]]); g(3, 7, 11, 15, m[s[6]], m[s[7]]);
            g(0, 5, 10, 15, m[s[8]], m[s[9]]); g(1, 6, 11, 12, m[s[10]], m[s[11]]); g(2, 7, 8, 13, m[s[12]], m[s[13]]); g(3, 4, 9, 14, m[s[14]], m[s[15]]);
        }
        for (int i = 0; i < 8; ++i) h[i] ^= v[i] ^ v[i + 8];
    };
    std::uint64_t t = 0;
    for (; n > 128; p += 128, n -= 128) {
        t += 128;
        compress(p, t, false);
    }
    unsigned char last[128] = {};
    std::memcpy(last, p, n);
    compress(last, t + n, true);
    std::array<unsigned char, 64> out{};
    std::memcpy(out.data(), h, 64);
    return out;
}

inline std::array<unsigned char, 32> blake2s_256(const unsigned char* p, std::size_t n)
{
    static constexpr std::uint32_t iv[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    std::uint32_t h[8];
    for (int i = 0; i < 8; ++i) h[i] = iv[i];
    h[0] ^= 0x01010000 ^ 32;
    const auto compress = [&](const unsigned char* blk, std::uint64_t t, bool last) {
        std::uint32_t m[16], v[16];
        for (int i = 0; i < 16; ++i) m[i] = le32(blk + 4 * i);
        for (int i = 0; i < 8; ++i) { v[i] = h[i]; v[i + 8] = iv[i]; }
        v[12] ^= static_cast<std::uint32_t>(t);
        v[13] ^= static_cast<std::uint32_t>(t >> 32);
        if (last) v[14] = ~v[14];
        const auto g = [&](int a, int b, int c, int d, std::uint32_t x, std::uint32_t y) {
            v[a] = v[a] + v[b] + x; v[d] = rotr32(v[d] ^ v[a], 16); v[c] = v[c] + v[d]; v[b] = rotr32(v[b] ^ v[c], 12);
            v[a] = v[a] + v[b] + y; v[d] = rotr32(v[d] ^ v[a], 8); v[c] = v[c] + v[d]; v[b] = rotr32(v[b] ^ v[c], 7);
        };
        for (int r = 0; r < 10; ++r) {
            const unsigned char* s = sigma[r];
            g(0, 4, 8, 12, m[s[0]], m[s[1]]); g(1, 5, 9, 13, m[s[2]], m[s[3]]); g(2, 6, 10, 14, m[s[4]], m[s[5]]); g(3, 7, 11, 15, m[s[6]], m[s[7]]);
            g(0, 5, 10, 15, m[s[8]], m[s[9]]); g(1, 6, 11, 12, m[s[10]], m[s[11]]); g(2, 7, 8, 13, m[s[12]], m[s[13]]); g(3, 4, 9, 14, m[s[14]], m[s[15]]);
        }
        for (int i = 0; i < 8; ++i) h[i] ^= v[i] ^ v[i + 8];
    };
    std::uint64_t t = 0;
    for (; n > 64; p += 64, n -= 64) {
        t += 64;
        compress(p, t, false);
    }
    unsigned char last[64] = {};
    std::memcpy(last, p, n);
    compress(last, t + n, true);
    std::array<unsigned char, 32> out{};
    std::memcpy(out.data(), h, 32);
    return out;
}

// ---- BLAKE3, as the official reference implementation writes it -------------------------------------------
namespace b3 {
inline constexpr std::uint32_t iv[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
inline constexpr int perm[16] = {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8};

inline void compress(const std::uint32_t cv[8], const std::uint32_t block[16], std::uint64_t counter, std::uint32_t len,
                     std::uint32_t flags, std::uint32_t out[16])
{
    std::uint32_t s[16] = {cv[0], cv[1], cv[2], cv[3], cv[4], cv[5], cv[6], cv[7], iv[0], iv[1], iv[2], iv[3],
                           static_cast<std::uint32_t>(counter), static_cast<std::uint32_t>(counter >> 32), len, flags};
    std::uint32_t m[16];
    std::memcpy(m, block, 64);
    const auto g = [&s](int a, int b, int c, int d, std::uint32_t x, std::uint32_t y) {
        s[a] = s[a] + s[b] + x; s[d] = rotr32(s[d] ^ s[a], 16); s[c] = s[c] + s[d]; s[b] = rotr32(s[b] ^ s[c], 12);
        s[a] = s[a] + s[b] + y; s[d] = rotr32(s[d] ^ s[a], 8); s[c] = s[c] + s[d]; s[b] = rotr32(s[b] ^ s[c], 7);
    };
    for (int r = 0; r < 7; ++r) {
        g(0, 4, 8, 12, m[0], m[1]); g(1, 5, 9, 13, m[2], m[3]); g(2, 6, 10, 14, m[4], m[5]); g(3, 7, 11, 15, m[6], m[7]);
        g(0, 5, 10, 15, m[8], m[9]); g(1, 6, 11, 12, m[10], m[11]); g(2, 7, 8, 13, m[12], m[13]); g(3, 4, 9, 14, m[14], m[15]);
        if (r < 6) {
            std::uint32_t t[16];
            for (int i = 0; i < 16; ++i) t[i] = m[perm[i]];
            std::memcpy(m, t, 64);
        }
    }
    for (int i = 0; i < 8; ++i) {
        out[i] = s[i] ^ s[i + 8];
        out[i + 8] = s[i + 8] ^ cv[i];
    }
}

inline std::array<unsigned char, 32> hash(const unsigned char* p, std::size_t n)
{
    std::uint32_t stack[54][8];
    int depth = 0;
    std::uint64_t chunk = 0;
    std::uint32_t out[16];
    const auto parent = [&](const std::uint32_t* l, const std::uint32_t* r, std::uint32_t flags, std::uint32_t* cv) {
        std::uint32_t block[16];
        std::memcpy(block, l, 32);
        std::memcpy(block + 8, r, 32);
        compress(iv, block, 0, 64, 4 | flags, out);
        std::memcpy(cv, out, 32);
    };
    // Every chunk except the last is complete: its chaining value goes on the stack, merging completed subtrees
    std::size_t done = 0;
    while (n - done > 1024) {
        std::uint32_t cv[8];
        std::memcpy(cv, iv, 32);
        for (int b = 0; b < 16; ++b) {
            std::uint32_t block[16];
            for (int i = 0; i < 16; ++i) block[i] = le32(p + done + 64 * b + 4 * i);
            compress(cv, block, chunk, 64, (b == 0 ? 1u : 0u) | (b == 15 ? 2u : 0u), out);
            std::memcpy(cv, out, 32);
        }
        ++chunk;
        for (std::uint64_t total = chunk; (total & 1) == 0; total >>= 1) {
            parent(stack[--depth], cv, 0, cv);
        }
        std::memcpy(stack[depth++], cv, 32);
        done += 1024;
    }
    // The last chunk, then the parents up the right edge, the root flagged
    const std::size_t rest = n - done;
    std::uint32_t cv[8];
    std::memcpy(cv, iv, 32);
    const std::size_t blocks = rest == 0 ? 1 : (rest + 63) / 64;
    std::uint32_t block[16] = {};
    std::uint32_t len = 0, flags = 0;
    for (std::size_t b = 0; b < blocks; ++b) {
        unsigned char buf[64] = {};
        len = static_cast<std::uint32_t>(rest - 64 * b < 64 ? rest - 64 * b : 64);
        std::memcpy(buf, p + done + 64 * b, len);
        for (int i = 0; i < 16; ++i) block[i] = le32(buf + 4 * i);
        flags = (b == 0 ? 1u : 0u) | (b + 1 == blocks ? 2u : 0u);
        if (b + 1 < blocks) {
            compress(cv, block, chunk, 64, flags, out);
            std::memcpy(cv, out, 32);
        }
    }
    std::uint32_t node_cv[8], node_block[16];
    std::uint64_t node_counter = chunk;
    std::memcpy(node_cv, cv, 32);
    std::memcpy(node_block, block, 64);
    while (depth > 0) {
        compress(node_cv, node_block, node_counter, len, flags, out);
        std::memcpy(node_block, stack[--depth], 32);
        std::memcpy(node_block + 8, out, 32);
        std::memcpy(node_cv, iv, 32);
        node_counter = 0;
        len = 64;
        flags = 4;
    }
    compress(node_cv, node_block, node_counter, len, flags | 8, out);
    std::array<unsigned char, 32> r{};
    std::memcpy(r.data(), out, 32);
    return r;
}
} // namespace b3

// ---- CRC, slicing-by-8 and byte at a time ------------------------------------------------------------------
template <typename T, T Poly>
struct crc_table
{
    T t[8][256];
    crc_table()
    {
        for (unsigned b = 0; b < 256; ++b) {
            T c = static_cast<T>(b);
            for (int k = 0; k < 8; ++k) c = (c & 1) ? static_cast<T>((c >> 1) ^ Poly) : static_cast<T>(c >> 1);
            t[0][b] = c;
        }
        for (int k = 1; k < 8; ++k)
            for (unsigned b = 0; b < 256; ++b)
                t[k][b] = static_cast<T>((t[k - 1][b] >> 8) ^ t[0][t[k - 1][b] & 0xff]);
    }
};

template <typename T, T Poly>
T crc_sliced(const unsigned char* p, std::size_t n)
{
    static const crc_table<T, Poly> table;
    const auto& t = table.t;
    T c = static_cast<T>(~T{0});
    for (; n >= 8; p += 8, n -= 8) {
        const std::uint64_t w = static_cast<std::uint64_t>(c) ^ le64(p);
        c = static_cast<T>(t[7][w & 0xff] ^ t[6][(w >> 8) & 0xff] ^ t[5][(w >> 16) & 0xff] ^ t[4][(w >> 24) & 0xff] ^
                           t[3][(w >> 32) & 0xff] ^ t[2][(w >> 40) & 0xff] ^ t[1][(w >> 48) & 0xff] ^ t[0][w >> 56]);
    }
    for (std::size_t i = 0; i < n; ++i) c = static_cast<T>((c >> 8) ^ t[0][(c ^ p[i]) & 0xff]);
    return static_cast<T>(~c);
}

template <typename T, T Poly>
T crc_bytewise(const unsigned char* p, std::size_t n)
{
    static const crc_table<T, Poly> table;
    T c = static_cast<T>(~T{0});
    for (std::size_t i = 0; i < n; ++i) c = static_cast<T>((c >> 8) ^ table.t[0][(c ^ p[i]) & 0xff]);
    return static_cast<T>(~c);
}

// ---- HMAC and PBKDF2 over the hand SHA-2, the key's states primed once --------------------------------------
template <std::size_t Block, std::size_t Out, typename Word, std::size_t LengthBytes, typename BlockFn>
struct hmac_sha2
{
    Word inner[8], outer[8];
    BlockFn block;

    // A key no longer than the block (the benchmarks' passwords are), zero-padded
    hmac_sha2(const Word* iv, const unsigned char* key, std::size_t klen, BlockFn f) : block(f)
    {
        unsigned char k[Block] = {};
        std::memcpy(k, key, klen);
        unsigned char pad[Block];
        for (std::size_t i = 0; i < Block; ++i) pad[i] = static_cast<unsigned char>(k[i] ^ 0x36);
        std::memcpy(inner, iv, sizeof inner);
        block(inner, pad);
        for (std::size_t i = 0; i < Block; ++i) pad[i] = static_cast<unsigned char>(k[i] ^ 0x5c);
        std::memcpy(outer, iv, sizeof outer);
        block(outer, pad);
    }

    // The tag of a message that, with the primed block, fits the hash's last block: `Out` bytes (PBKDF2's U_i)
    void mac_short(const unsigned char* msg, std::size_t n, unsigned char* tag) const
    {
        Word s[8];
        std::memcpy(s, inner, sizeof s);
        finish(s, msg, n, Block, tag);
        std::memcpy(s, outer, sizeof s);
        unsigned char inner_tag[Out];
        std::memcpy(inner_tag, tag, Out);
        finish(s, inner_tag, Out, Block, tag);
    }

    void finish(Word* s, const unsigned char* msg, std::size_t n, std::size_t already, unsigned char* tag) const
    {
        unsigned char last[2 * Block] = {};
        std::memcpy(last, msg, n);
        last[n] = 0x80;
        const std::size_t total = n + 1 + LengthBytes <= Block ? Block : 2 * Block;
        const std::uint64_t bits = static_cast<std::uint64_t>(already + n) * 8;
        for (std::size_t i = 0; i < 8; ++i) last[total - 1 - i] = static_cast<unsigned char>(bits >> (8 * i));
        block(s, last);
        if (total == 2 * Block) block(s, last + Block);
        for (std::size_t i = 0; i < Out / sizeof(Word); ++i)
            for (std::size_t j = 0; j < sizeof(Word); ++j)
                tag[sizeof(Word) * i + j] = static_cast<unsigned char>(s[i] >> (8 * (sizeof(Word) - 1 - j)));
    }
};

// PBKDF2 for a password no longer than the block and a salt that fits one block with the index
template <typename Hmac, std::size_t Out>
bytes pbkdf2(const Hmac& m, const unsigned char* salt, std::size_t slen, std::uint32_t iterations, std::size_t dk)
{
    bytes out;
    for (std::uint32_t block = 1; out.size() < dk; ++block) {
        unsigned char s[256];
        std::memcpy(s, salt, slen);
        s[slen] = static_cast<unsigned char>(block >> 24);
        s[slen + 1] = static_cast<unsigned char>(block >> 16);
        s[slen + 2] = static_cast<unsigned char>(block >> 8);
        s[slen + 3] = static_cast<unsigned char>(block);
        unsigned char u[Out], t[Out];
        m.mac_short(s, slen + 4, u);
        std::memcpy(t, u, Out);
        for (std::uint32_t i = 1; i < iterations; ++i) {
            unsigned char next[Out];
            m.mac_short(u, Out, next);
            std::memcpy(u, next, Out);
            for (std::size_t j = 0; j < Out; ++j) t[j] ^= u[j];
        }
        out.insert(out.end(), t, t + Out);
    }
    out.resize(dk);
    return out;
}

inline constexpr std::uint32_t sha256_iv[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
inline constexpr std::uint64_t sha512_iv[8] = {0x6a09e667f3bcc908, 0xbb67ae8584caa73b, 0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
                                               0x510e527fade682d1, 0x9b05688c2b3e6c1f, 0x1f83d9abfb41bd6b, 0x5be0cd19137e2179};
using hmac256 = hmac_sha2<64, 32, std::uint32_t, 8, void (*)(std::uint32_t*, const unsigned char*)>;
using hmac512 = hmac_sha2<128, 64, std::uint64_t, 16, void (*)(std::uint64_t*, const unsigned char*)>;

// ---- scrypt, over a caller's working memory like core's -----------------------------------------------------
inline void salsa(std::uint32_t* b)
{
    std::uint32_t x[16];
    std::memcpy(x, b, 64);
    for (int i = 0; i < 8; i += 2) {
        x[4] ^= rotl32(x[0] + x[12], 7); x[8] ^= rotl32(x[4] + x[0], 9); x[12] ^= rotl32(x[8] + x[4], 13); x[0] ^= rotl32(x[12] + x[8], 18);
        x[9] ^= rotl32(x[5] + x[1], 7); x[13] ^= rotl32(x[9] + x[5], 9); x[1] ^= rotl32(x[13] + x[9], 13); x[5] ^= rotl32(x[1] + x[13], 18);
        x[14] ^= rotl32(x[10] + x[6], 7); x[2] ^= rotl32(x[14] + x[10], 9); x[6] ^= rotl32(x[2] + x[14], 13); x[10] ^= rotl32(x[6] + x[2], 18);
        x[3] ^= rotl32(x[15] + x[11], 7); x[7] ^= rotl32(x[3] + x[15], 9); x[11] ^= rotl32(x[7] + x[3], 13); x[15] ^= rotl32(x[11] + x[7], 18);
        x[1] ^= rotl32(x[0] + x[3], 7); x[2] ^= rotl32(x[1] + x[0], 9); x[3] ^= rotl32(x[2] + x[1], 13); x[0] ^= rotl32(x[3] + x[2], 18);
        x[6] ^= rotl32(x[5] + x[4], 7); x[7] ^= rotl32(x[6] + x[5], 9); x[4] ^= rotl32(x[7] + x[6], 13); x[5] ^= rotl32(x[4] + x[7], 18);
        x[11] ^= rotl32(x[10] + x[9], 7); x[8] ^= rotl32(x[11] + x[10], 9); x[9] ^= rotl32(x[8] + x[11], 13); x[10] ^= rotl32(x[9] + x[8], 18);
        x[12] ^= rotl32(x[15] + x[14], 7); x[13] ^= rotl32(x[12] + x[15], 9); x[14] ^= rotl32(x[13] + x[12], 13); x[15] ^= rotl32(x[14] + x[13], 18);
    }
    for (int i = 0; i < 16; ++i) b[i] += x[i];
}

inline void block_mix(std::uint32_t* b, std::uint32_t* y, std::size_t r)
{
    std::uint32_t t[16];
    std::memcpy(t, b + (2 * r - 1) * 16, 64);
    for (std::size_t i = 0; i < 2 * r; ++i) {
        for (int k = 0; k < 16; ++k) t[k] ^= b[i * 16 + k];
        salsa(t);
        std::memcpy(y + ((i % 2 == 0) ? i / 2 : r + i / 2) * 16, t, 64);
    }
    std::memcpy(b, y, 128 * r);
}

// ROMix of each lane of b in place, through v and y: the memory-hard middle of scrypt (both PBKDF2 ends are the
// same calls in core and here, so the comparison is of this part)
inline void ro_mix(std::uint32_t* x, std::uint32_t* v, std::uint32_t* y, std::size_t r, std::uint64_t n)
{
    const std::size_t words = 32 * r;
    for (std::uint64_t i = 0; i < n; ++i) {
        std::memcpy(v + i * words, x, 4 * words);
        block_mix(x, y, r);
    }
    for (std::uint64_t i = 0; i < n; ++i) {
        const std::uint64_t j = (x[(2 * r - 1) * 16] | (std::uint64_t{x[(2 * r - 1) * 16 + 1]} << 32)) & (n - 1);
        for (std::size_t k = 0; k < words; ++k) x[k] ^= v[j * words + k];
        block_mix(x, y, r);
    }
}

#if SNICHOLLS_CORE_BENCH_OPENSSL
// One EVP digest of `n` bytes into `out`; `md` from EVP_get_digestbyname or EVP_MD_fetch.
inline void openssl_digest(const EVP_MD* md, const unsigned char* p, std::size_t n, unsigned char* out)
{
    unsigned len = 0;
    EVP_Digest(p, n, out, &len, md, nullptr);
}
#endif

} // namespace by_hand

#endif /* snicholls_core_benchmarks_hash_by_hand_hpp */
