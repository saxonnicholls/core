//
//  sha1.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

// SHA-1 (FIPS 180-4), constexpr, for name-based (version 5) UUIDs only.
//
//   snicholls::detail::sha1 h;  h.update(bytes);  h.update(more);  std::array<std::uint8_t, 20> d = h.finish();
//   constexpr auto d = snicholls::detail::sha1_digest("abc");
//
// SHA-1 is broken for collision resistance: never use it for signatures, integrity or passwords. RFC 9562 fixes it
// for UUIDv5, where it only spreads a name into 122 bits. Not public API.
//
// ts-moveables has a private, run-time-only SHA-1 in http/websocket.hpp; this one is constexpr.

#ifndef snicholls_core_detail_sha1_hpp
#define snicholls_core_detail_sha1_hpp

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace snicholls::detail {

class sha1
{
public:
    using digest_type = std::array<std::uint8_t, 20>;

    constexpr sha1() noexcept = default;

    constexpr sha1& update(std::span<const std::uint8_t> data) noexcept
    {
        for (const std::uint8_t b : data)
            push(b);
        return *this;
    }

    constexpr sha1& update(std::span<const std::byte> data) noexcept
    {
        for (const std::byte b : data)
            push(static_cast<std::uint8_t>(b));
        return *this;
    }

    constexpr sha1& update(std::string_view text) noexcept
    {
        for (const char c : text)
            push(static_cast<std::uint8_t>(c));
        return *this;
    }

    // Pads, processes the last block(s) and returns the digest. The object must not be used afterwards.
    constexpr digest_type finish() noexcept
    {
        const std::uint64_t bit_length = total_ * 8U;
        push(0x80);
        while (used_ != 56)
            push(0x00);
        for (int shift = 56; shift >= 0; shift -= 8)
            push(static_cast<std::uint8_t>(bit_length >> shift));
        digest_type out{};
        for (std::size_t i = 0; i < 5; ++i) {
            out[4 * i + 0] = static_cast<std::uint8_t>(h_[i] >> 24);
            out[4 * i + 1] = static_cast<std::uint8_t>(h_[i] >> 16);
            out[4 * i + 2] = static_cast<std::uint8_t>(h_[i] >> 8);
            out[4 * i + 3] = static_cast<std::uint8_t>(h_[i]);
        }
        return out;
    }

private:
    static constexpr std::uint32_t rotl(std::uint32_t x, int n) noexcept { return (x << n) | (x >> (32 - n)); }

    constexpr void push(std::uint8_t b) noexcept
    {
        block_[used_++] = b;
        ++total_;
        if (used_ == 64) {
            compress();
            used_ = 0;
        }
    }

    constexpr void compress() noexcept
    {
        std::uint32_t w[80]{};
        for (std::size_t i = 0; i < 16; ++i)
            w[i] = (std::uint32_t{block_[4 * i]} << 24) | (std::uint32_t{block_[4 * i + 1]} << 16) |
                   (std::uint32_t{block_[4 * i + 2]} << 8) | std::uint32_t{block_[4 * i + 3]};
        for (std::size_t i = 16; i < 80; ++i)
            w[i] = rotl(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);

        std::uint32_t a = h_[0], b = h_[1], c = h_[2], d = h_[3], e = h_[4];
        for (std::size_t i = 0; i < 80; ++i) {
            std::uint32_t f = 0, k = 0;
            if (i < 20) {
                f = (b & c) | (~b & d);
                k = 0x5A827999U;
            } else if (i < 40) {
                f = b ^ c ^ d;
                k = 0x6ED9EBA1U;
            } else if (i < 60) {
                f = (b & c) | (b & d) | (c & d);
                k = 0x8F1BBCDCU;
            } else {
                f = b ^ c ^ d;
                k = 0xCA62C1D6U;
            }
            const std::uint32_t t = rotl(a, 5) + f + e + k + w[i];
            e = d;
            d = c;
            c = rotl(b, 30);
            b = a;
            a = t;
        }
        h_[0] += a;
        h_[1] += b;
        h_[2] += c;
        h_[3] += d;
        h_[4] += e;
    }

    std::uint32_t h_[5]{0x67452301U, 0xEFCDAB89U, 0x98BADCFEU, 0x10325476U, 0xC3D2E1F0U};
    std::uint8_t block_[64]{};
    std::size_t used_ = 0;
    std::uint64_t total_ = 0;
};

constexpr sha1::digest_type sha1_digest(std::string_view text) noexcept
{
    sha1 h;
    h.update(text);
    return h.finish();
}

} // namespace snicholls::detail

#endif /* snicholls_core_detail_sha1_hpp */
