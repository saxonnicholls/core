//
//  mul_div.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::detail::mul_div`: the exact quotient and remainder of a 256 x 256-bit product by a 256-bit divisor.
///
/// The product is formed in 512 bits and divided with Knuth's algorithm D on 32-bit digits, so nothing is lost before
/// the caller rounds: money conversion multiplies an amount by a rate's numerator and divides by its denominator.
/// When both factors fit in 64 bits the product is one 128-bit multiply and the division is `uint256`'s, which is one
/// hardware division when the product and the divisor fit in 64 bits; the results are bit-identical to the general path.
/// constexpr and allocation-free. Internal: `money/rate.hpp` is the public face.

#ifndef snicholls_core_detail_mul_div_hpp
#define snicholls_core_detail_mul_div_hpp

#include <snicholls/core/detail/wide_mul.hpp>
#include <snicholls/core/uint256.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>

namespace snicholls::detail {

    /// The quotient and remainder of `a * b / d`, or `std::nullopt` when `d` is zero or the quotient needs more than
    /// 256 bits.
    [[nodiscard]] constexpr std::optional<std::pair<uint256, uint256>> mul_div(const uint256& a, const uint256& b,
                                                                             const uint256& d) noexcept
    {
        if (d.is_zero())
            return std::nullopt;
        // the fast path: two 64-bit factors give a 128-bit product, divided by uint256's own division (one hardware
        // division when the product and the divisor fit in 64 bits); the quotient always fits, the bits are the same
        if (((a.limb(1) | a.limb(2) | a.limb(3)) | (b.limb(1) | b.limb(2) | b.limb(3))) == 0) {
            const auto w = umul128(a.limb(0), b.limb(0));
            return uint256::checked_divmod(uint256::from_limbs(w.low, w.high), d);
        }
        // the 512-bit product, as eight 64-bit limbs
        std::array<std::uint64_t, 8> p{};
        for (std::size_t i = 0; i < 4; ++i) {
            std::uint64_t carry = 0;
            for (std::size_t j = 0; j < 4; ++j) {
                const auto w = umul128(a.limb(i), b.limb(j));
                std::uint64_t lo = w.low + carry;
                std::uint64_t hi = w.high + (lo < carry ? 1u : 0u);
                const std::uint64_t sum = p[i + j] + lo;
                hi += sum < lo ? 1u : 0u;
                p[i + j] = sum;
                carry = hi;
            }
            p[i + 4] = carry;
        }
        // to 32-bit digits: u (16 + 1 for normalisation) and v (8)
        std::array<std::uint32_t, 17> u{};
        for (std::size_t i = 0; i < 8; ++i) {
            u[2 * i] = static_cast<std::uint32_t>(p[i]);
            u[2 * i + 1] = static_cast<std::uint32_t>(p[i] >> 32);
        }
        std::array<std::uint32_t, 8> v{};
        for (std::size_t i = 0; i < 4; ++i) {
            v[2 * i] = static_cast<std::uint32_t>(d.limb(i));
            v[2 * i + 1] = static_cast<std::uint32_t>(d.limb(i) >> 32);
        }
        int n = 8;
        while (v[static_cast<std::size_t>(n - 1)] == 0)
            --n;
        int m = 16;
        while (m > 0 && u[static_cast<std::size_t>(m - 1)] == 0)
            --m;
        std::array<std::uint32_t, 16> q{};
        std::array<std::uint32_t, 8> r{};
        if (m < n) {   // the product is smaller than the divisor
            for (std::size_t i = 0; i < 8; ++i)
                r[i] = u[i];
        } else if (n == 1) {   // a one-digit divisor: short division
            std::uint64_t rem = 0;
            for (int i = m - 1; i >= 0; --i) {
                const std::uint64_t cur = (rem << 32) | u[static_cast<std::size_t>(i)];
                q[static_cast<std::size_t>(i)] = static_cast<std::uint32_t>(cur / v[0]);
                rem = cur % v[0];
            }
            r[0] = static_cast<std::uint32_t>(rem);
        } else {   // Knuth, TAOCP vol. 2, 4.3.1, algorithm D
            int s = 0;
            for (std::uint32_t top = v[static_cast<std::size_t>(n - 1)]; (top & 0x80000000u) == 0; top <<= 1)
                ++s;
            std::array<std::uint32_t, 8> vn{};
            for (int i = n - 1; i > 0; --i)
                vn[static_cast<std::size_t>(i)] = static_cast<std::uint32_t>(
                    (static_cast<std::uint64_t>(v[static_cast<std::size_t>(i)]) << s) |
                    (s == 0 ? 0 : static_cast<std::uint64_t>(v[static_cast<std::size_t>(i - 1)]) >> (32 - s)));
            vn[0] = static_cast<std::uint32_t>(static_cast<std::uint64_t>(v[0]) << s);
            std::array<std::uint32_t, 17> un{};
            un[static_cast<std::size_t>(m)] =
                s == 0 ? 0u : static_cast<std::uint32_t>(static_cast<std::uint64_t>(u[static_cast<std::size_t>(m - 1)]) >> (32 - s));
            for (int i = m - 1; i > 0; --i)
                un[static_cast<std::size_t>(i)] = static_cast<std::uint32_t>(
                    (static_cast<std::uint64_t>(u[static_cast<std::size_t>(i)]) << s) |
                    (s == 0 ? 0 : static_cast<std::uint64_t>(u[static_cast<std::size_t>(i - 1)]) >> (32 - s)));
            un[0] = static_cast<std::uint32_t>(static_cast<std::uint64_t>(u[0]) << s);
            const std::uint64_t base = 1ull << 32;
            for (int j = m - n; j >= 0; --j) {
                const auto jn = static_cast<std::size_t>(j + n);
                const std::uint64_t num = (static_cast<std::uint64_t>(un[jn]) << 32) | un[jn - 1];
                std::uint64_t qhat = num / vn[static_cast<std::size_t>(n - 1)];
                std::uint64_t rhat = num % vn[static_cast<std::size_t>(n - 1)];
                while (qhat >= base || qhat * vn[static_cast<std::size_t>(n - 2)] > ((rhat << 32) | un[jn - 2])) {
                    --qhat;
                    rhat += vn[static_cast<std::size_t>(n - 1)];
                    if (rhat >= base)
                        break;
                }
                std::int64_t borrow = 0;
                std::uint64_t carry = 0;
                for (int i = 0; i < n; ++i) {
                    const auto idx = static_cast<std::size_t>(i);
                    const std::uint64_t prod = qhat * vn[idx] + carry;
                    carry = prod >> 32;
                    const std::int64_t t = static_cast<std::int64_t>(un[idx + static_cast<std::size_t>(j)]) - borrow -
                                           static_cast<std::int64_t>(prod & 0xffffffffu);
                    un[idx + static_cast<std::size_t>(j)] = static_cast<std::uint32_t>(t);
                    borrow = t < 0 ? 1 : 0;
                }
                const std::int64_t t = static_cast<std::int64_t>(un[jn]) - borrow - static_cast<std::int64_t>(carry);
                un[jn] = static_cast<std::uint32_t>(t);
                q[static_cast<std::size_t>(j)] = static_cast<std::uint32_t>(qhat);
                if (t < 0) {   // added back: qhat was one too large
                    q[static_cast<std::size_t>(j)] -= 1;
                    std::uint64_t c = 0;
                    for (int i = 0; i < n; ++i) {
                        const auto idx = static_cast<std::size_t>(i) + static_cast<std::size_t>(j);
                        const std::uint64_t sum = static_cast<std::uint64_t>(un[idx]) + vn[static_cast<std::size_t>(i)] + c;
                        un[idx] = static_cast<std::uint32_t>(sum);
                        c = sum >> 32;
                    }
                    un[jn] = static_cast<std::uint32_t>(un[jn] + c);
                }
            }
            for (int i = 0; i < n; ++i) {
                const auto idx = static_cast<std::size_t>(i);
                r[idx] = static_cast<std::uint32_t>(
                    (static_cast<std::uint64_t>(un[idx]) >> s) |
                    (s == 0 ? 0 : static_cast<std::uint64_t>(un[idx + 1]) << (32 - s)));
            }
        }
        for (std::size_t i = 8; i < 16; ++i)
            if (q[i] != 0)
                return std::nullopt;   // the quotient needs more than 256 bits
        const auto limb = [](const auto& digits, std::size_t i) {
            return digits[2 * i] | (static_cast<std::uint64_t>(digits[2 * i + 1]) << 32);
        };
        return std::pair{uint256::from_limbs(limb(q, 0), limb(q, 1), limb(q, 2), limb(q, 3)),
                         uint256::from_limbs(limb(r, 0), limb(r, 1), limb(r, 2), limb(r, 3))};
    }

} // namespace snicholls::detail

#endif // snicholls_core_detail_mul_div_hpp
