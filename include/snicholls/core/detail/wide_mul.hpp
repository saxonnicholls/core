//
//  wide_mul.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  New in core: the 64 x 64 -> 128-bit multiply that wyhash.hpp and uint256.hpp share, written once.
//

/// \file
/// \brief `snicholls::detail::umul128`: the full 128-bit product of two 64-bit integers, constexpr and portable.
/// \ingroup core

#ifndef snicholls_core_detail_wide_mul_hpp
#define snicholls_core_detail_wide_mul_hpp

#include <cstdint>
#include <type_traits>

namespace snicholls::detail {

    // The two halves of a 128-bit product.
    struct wide_product {
        std::uint64_t low;
        std::uint64_t high;
    };

    // a * b as 128 bits. In a constant expression, and on a compiler without a 128-bit integer
    // (MSVC), four 32 x 32 multiplies; at run time with __int128 (GCC, Clang) one instruction.
    // The two paths give the same bits, which the tests prove.
    constexpr wide_product umul128(std::uint64_t a, std::uint64_t b) noexcept
    {
#if defined(__SIZEOF_INT128__)
        if (!std::is_constant_evaluated()) {
            __extension__ using u128 = unsigned __int128;
            const u128 r = static_cast<u128>(a) * static_cast<u128>(b);
            return {static_cast<std::uint64_t>(r), static_cast<std::uint64_t>(r >> 64)};
        }
#endif
        const std::uint64_t a_hi = a >> 32, a_lo = static_cast<std::uint32_t>(a);
        const std::uint64_t b_hi = b >> 32, b_lo = static_cast<std::uint32_t>(b);
        const std::uint64_t hh = a_hi * b_hi, hl = a_hi * b_lo, lh = a_lo * b_hi, ll = a_lo * b_lo;
        // The 2^32 column: three 32-bit values, so it fits in 64 bits with its carry.
        const std::uint64_t mid = (ll >> 32) + static_cast<std::uint32_t>(hl) + static_cast<std::uint32_t>(lh);
        const std::uint64_t low = (mid << 32) | static_cast<std::uint32_t>(ll);
        const std::uint64_t high = hh + (hl >> 32) + (lh >> 32) + (mid >> 32);
        return {low, high};
    }

} // namespace snicholls::detail

#endif /* snicholls_core_detail_wide_mul_hpp */
