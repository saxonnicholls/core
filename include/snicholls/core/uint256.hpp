//
//  uint256.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  The limbs, the never-throwing parsers and the exact decimal rendering are the source's. Core makes
//  everything constexpr and portable (no __int128: the 128-bit products come from detail/wide_mul.hpp),
//  completes the arithmetic (multiply, divide, remainder, shifts, bit operations, comparisons) and replaces
//  the bool-plus-out-parameter calls with std::optional.
//

/// \file
/// \brief `snicholls::uint256`: a 256-bit unsigned integer, constexpr, exact, for amounts that do not fit a `uint64_t`.
/// \ingroup core
///
/// **Overview.** Every EVM token amount is a 256-bit unsigned integer; one token with 18
/// decimals is 10^18 base units, and a `std::uint64_t` overflows at 18.4 of them. A
/// `double` holds 15 to 17 significant digits and silently rounds the rest. `uint256` is the
/// native type: four 64-bit limbs, little-endian, every operation `constexpr`, and arithmetic
/// that behaves like the built-in unsigned types (modulo 2^256) with checked forms beside it
/// that report overflow instead of wrapping:
///
/// \snippet uint256_demo.cpp amounts
///
/// A token amount as people write it goes through `parse_units("250.12", 6)` and
/// `format_units(6)`: 6 decimals for USDT and USDC, 18 for ETH, any count up to 77. Digits past the
/// token's decimals are accepted only when they are zeros (Binance's eight-place strings for a
/// 6-decimal token), so nothing is ever rounded away.
///
/// Parsing never throws: `from_decimal` and `from_hex` return `std::nullopt` on any bad
/// input or overflow, because a money path must be able to refuse bad input without
/// unwinding through a half-completed transaction. Rendering is exact: `to_string` is the
/// full decimal, `to_hex` the minimal `0x` form, `to_abi_word` the 64-nibble word an ABI
/// encoder wants.
///
/// **Relationships.** `detail/wide_mul.hpp` for the 128-bit products; `hash_combine` for
/// `std::hash`; `wyhash` where a `uint256` is a key across machines. Scaling to human units
/// is the caller's, from the amount and its decimals together (`pow10`). Group: \ref core.
///
/// **Guarantees.** A literal type of exactly 32 bytes, trivially copyable, standard layout.
/// Every operation is `constexpr`; everything but division and the string renderings is
/// `noexcept`. `+`, `-`, `*`, `<<`, `>>` wrap modulo 2^256 like `unsigned`; the `checked_`
/// forms return `std::nullopt` instead. Division by zero is a precondition violation
/// (asserted); `checked_div` and `checked_divmod` return `std::nullopt` for it. Shifts by
/// 256 or more give 0. Decimal and hexadecimal round trips are exact.
///
/// **Errors.** None thrown by the arithmetic. `to_string`, `to_hex` and `to_abi_word`
/// allocate and can throw `std::bad_alloc`. The parsers and the checked operations report
/// through `std::optional`: this header deliberately does not include `result.hpp`, so
/// that a numeric vocabulary type stays light; wrap the optional in a `result` at the call
/// site where the reason matters.
///
/// **Performance.** Benchmarked in `benchmarks/uint256_benchmark.cpp`: addition and
/// multiplication against the same limb loops written out by hand, after a fairness check;
/// division and `to_string` alone, since their hand-written equivalents are the same
/// algorithms. Release, Apple clang 17, Xeon W-3245, 2026-11-06, 4,096 pairs of every width:
/// addition 2.0 ns through the class and 2.0 ns by hand, multiplication 4.3 ns and 4.4 ns,
/// `divmod` 45 ns (Knuth D, base 2^32), `to_string` 540 ns (nine digits per 32-bit division,
/// plus the string building). Zero overhead on the operators.
///
/// **Tests and demos.** `tests/uint256_test.cpp`, `demos/uint256_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<bit>`, `<cassert>`, `<compare>`, `<cstddef>`, `<cstdint>`,
/// `<functional>`, `<optional>`, `<ostream>`, `<string>`, `<string_view>`, `<utility>`,
/// `detail/wide_mul.hpp` and `hash/hash_combine.hpp`. `print(std::ostream&) const` makes it a
/// `snicholls::printable` for `acout.hpp` without including it.
///
/// **Known issues.** Construction is from `std::uint64_t` only; a negative `int` converts
/// to a large value as it would for `unsigned`. There is no conversion to or from floating
/// point: scale to human units by division and `to_string`, which is exact, and convert the
/// text if a `double` is really wanted.
///
/// **Status.** New in 0.5.0.

#ifndef snicholls_core_uint256_hpp
#define snicholls_core_uint256_hpp

#include <snicholls/core/detail/wide_mul.hpp>
#include <snicholls/core/hash/hash_combine.hpp>

#include <array>
#include <bit>
#include <cassert>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <utility>

namespace snicholls {

    /// \brief A 256-bit unsigned integer: four 64-bit limbs, little-endian, constexpr throughout.
    /// \ingroup core
    class uint256 {
    public:
        /// \brief One limb.
        using limb_type = std::uint64_t;
        /// \brief The number of limbs.
        static constexpr std::size_t limb_count = 4;

        /// \brief Zero.
        constexpr uint256() noexcept = default;

        /// \brief The value of a 64-bit integer. Implicit, so `uint256 x = 5;` and `x + 1` read naturally.
        /// \param low The value.
        constexpr uint256(std::uint64_t low) noexcept : limbs_{{low, 0, 0, 0}} {}   // NOLINT(google-explicit-constructor)

        /// \brief The value with these limbs, least significant first.
        /// \param l0 Bits 0 to 63.
        /// \param l1 Bits 64 to 127.
        /// \param l2 Bits 128 to 191.
        /// \param l3 Bits 192 to 255.
        /// \return The value.
        [[nodiscard]] static constexpr uint256 from_limbs(std::uint64_t l0, std::uint64_t l1 = 0, std::uint64_t l2 = 0,
                                                          std::uint64_t l3 = 0) noexcept
        {
            uint256 v;
            v.limbs_ = {{l0, l1, l2, l3}};
            return v;
        }

        /// \brief 2^256 - 1.
        /// \return The largest value.
        [[nodiscard]] static constexpr uint256 max() noexcept { return from_limbs(~0ull, ~0ull, ~0ull, ~0ull); }

        /// \brief 10^n: the scale of a token with `n` decimals.
        /// \pre `n <= 77` (10^77 is the largest power of ten that fits).
        /// \param n The exponent.
        /// \return The power.
        [[nodiscard]] static constexpr uint256 pow10(unsigned n) noexcept
        {
            assert(n <= 77 && "snicholls::uint256::pow10: 10^n does not fit in 256 bits");
            uint256 v{1};
            for (unsigned i = 0; i < n; ++i)
                v.mul_small(10);
            return v;
        }

        // ---- observers ------------------------------------------------------------------------------------------

        /// \brief The limbs, least significant first.
        /// \return The four limbs.
        [[nodiscard]] constexpr const std::array<std::uint64_t, 4>& limbs() const noexcept { return limbs_; }

        /// \brief Limb `i`.
        /// \pre `i < 4`.
        /// \param i The index, 0 the least significant.
        /// \return The limb.
        [[nodiscard]] constexpr std::uint64_t limb(std::size_t i) const noexcept { return limbs_[i]; }

        /// \brief Whether the value is 0.
        /// \return true for 0.
        [[nodiscard]] constexpr bool is_zero() const noexcept { return (limbs_[0] | limbs_[1] | limbs_[2] | limbs_[3]) == 0; }

        /// \brief true when the value is not 0, like an integer.
        /// \return `!is_zero()`.
        [[nodiscard]] constexpr explicit operator bool() const noexcept { return !is_zero(); }

        /// \brief The number of bits needed to represent the value: 0 for 0, 256 for `max()`.
        /// \return The width.
        [[nodiscard]] constexpr int bit_width() const noexcept
        {
            for (int i = 3; i >= 0; --i)
                if (limbs_[static_cast<std::size_t>(i)] != 0)
                    return 64 * i + (64 - std::countl_zero(limbs_[static_cast<std::size_t>(i)]));
            return 0;
        }

        /// \brief Bit `i` of the value.
        /// \pre `i < 256`.
        /// \param i The bit, 0 the least significant.
        /// \return Whether it is set.
        [[nodiscard]] constexpr bool bit(std::size_t i) const noexcept { return (limbs_[i / 64] >> (i % 64)) & 1u; }

        /// \brief The value as a `std::uint64_t`, when it fits.
        /// \return The value, or `std::nullopt` when any bit above 63 is set.
        [[nodiscard]] constexpr std::optional<std::uint64_t> to_uint64() const noexcept
        {
            if ((limbs_[1] | limbs_[2] | limbs_[3]) != 0)
                return std::nullopt;
            return limbs_[0];
        }

        /// \brief The low 64 bits, truncating.
        /// \return Limb 0.
        [[nodiscard]] constexpr std::uint64_t low64() const noexcept { return limbs_[0]; }

        // ---- comparison -----------------------------------------------------------------------------------------

        /// \brief Equality, limb by limb.
        /// \param a One value.
        /// \param b The other.
        /// \return Whether they are equal.
        friend constexpr bool operator==(const uint256& a, const uint256& b) noexcept = default;

        /// \brief Ordering, most significant limb first.
        /// \param a One value.
        /// \param b The other.
        /// \return The ordering.
        friend constexpr std::strong_ordering operator<=>(const uint256& a, const uint256& b) noexcept
        {
            for (int i = 3; i >= 0; --i) {
                const auto idx = static_cast<std::size_t>(i);
                if (a.limbs_[idx] != b.limbs_[idx])
                    return a.limbs_[idx] <=> b.limbs_[idx];
            }
            return std::strong_ordering::equal;
        }

        // ---- arithmetic, modulo 2^256 ---------------------------------------------------------------------------

        /// \brief Add, wrapping.
        /// \param other The addend.
        /// \return `*this`.
        constexpr uint256& operator+=(const uint256& other) noexcept
        {
            add_with_carry(other);
            return *this;
        }

        /// \brief Subtract, wrapping.
        /// \param other The subtrahend.
        /// \return `*this`.
        constexpr uint256& operator-=(const uint256& other) noexcept
        {
            sub_with_borrow(other);
            return *this;
        }

        /// \brief Multiply, keeping the low 256 bits.
        /// \param other The multiplier.
        /// \return `*this`.
        constexpr uint256& operator*=(const uint256& other) noexcept
        {
            const auto full = mul_full(*this, other);
            limbs_ = {{full[0], full[1], full[2], full[3]}};
            return *this;
        }

        /// \brief Divide.
        /// \pre `!other.is_zero()`.
        /// \param other The divisor.
        /// \return `*this`.
        constexpr uint256& operator/=(const uint256& other)
        {
            *this = divmod(other).first;
            return *this;
        }

        /// \brief Remainder.
        /// \pre `!other.is_zero()`.
        /// \param other The divisor.
        /// \return `*this`.
        constexpr uint256& operator%=(const uint256& other)
        {
            *this = divmod(other).second;
            return *this;
        }

        /// \brief Bitwise and.
        /// \param other The other operand.
        /// \return `*this`.
        constexpr uint256& operator&=(const uint256& other) noexcept
        {
            for (std::size_t i = 0; i < 4; ++i)
                limbs_[i] &= other.limbs_[i];
            return *this;
        }

        /// \brief Bitwise or.
        /// \param other The other operand.
        /// \return `*this`.
        constexpr uint256& operator|=(const uint256& other) noexcept
        {
            for (std::size_t i = 0; i < 4; ++i)
                limbs_[i] |= other.limbs_[i];
            return *this;
        }

        /// \brief Bitwise xor.
        /// \param other The other operand.
        /// \return `*this`.
        constexpr uint256& operator^=(const uint256& other) noexcept
        {
            for (std::size_t i = 0; i < 4; ++i)
                limbs_[i] ^= other.limbs_[i];
            return *this;
        }

        /// \brief Shift left; 256 or more gives 0.
        /// \param n The shift.
        /// \return `*this`.
        constexpr uint256& operator<<=(unsigned n) noexcept
        {
            if (n >= 256) {
                limbs_ = {};
                return *this;
            }
            const std::size_t limb_shift = n / 64;
            const unsigned bit_shift = n % 64;
            std::array<std::uint64_t, 4> out{};
            for (std::size_t i = 0; i < 4; ++i) {
                if (i < limb_shift)
                    continue;
                const std::size_t from = i - limb_shift;
                out[i] = limbs_[from] << bit_shift;
                if (bit_shift != 0 && from > 0)
                    out[i] |= limbs_[from - 1] >> (64 - bit_shift);
            }
            limbs_ = out;
            return *this;
        }

        /// \brief Shift right; 256 or more gives 0.
        /// \param n The shift.
        /// \return `*this`.
        constexpr uint256& operator>>=(unsigned n) noexcept
        {
            if (n >= 256) {
                limbs_ = {};
                return *this;
            }
            const std::size_t limb_shift = n / 64;
            const unsigned bit_shift = n % 64;
            std::array<std::uint64_t, 4> out{};
            for (std::size_t i = 0; i < 4; ++i) {
                const std::size_t from = i + limb_shift;
                if (from >= 4)
                    continue;
                out[i] = limbs_[from] >> bit_shift;
                if (bit_shift != 0 && from + 1 < 4)
                    out[i] |= limbs_[from + 1] << (64 - bit_shift);
            }
            limbs_ = out;
            return *this;
        }

        /// \brief Increment, wrapping.
        /// \return `*this`.
        constexpr uint256& operator++() noexcept { return *this += uint256{1}; }
        /// \brief Decrement, wrapping.
        /// \return `*this`.
        constexpr uint256& operator--() noexcept { return *this -= uint256{1}; }
        /// \brief Post-increment, wrapping.
        /// \return The old value.
        constexpr uint256 operator++(int) noexcept
        {
            const uint256 old = *this;
            ++*this;
            return old;
        }
        /// \brief Post-decrement, wrapping.
        /// \return The old value.
        constexpr uint256 operator--(int) noexcept
        {
            const uint256 old = *this;
            --*this;
            return old;
        }

        /// \name The binary operators, from the compound ones
        ///@{
        /// \brief The binary form of the compound operator of the same name: wrapping for `+ - *` and the
        ///        shifts; `/` and `%` need a non-zero divisor.
        /// \param a The left operand, by value.
        /// \param b The right operand.
        /// \return The result.
        friend constexpr uint256 operator+(uint256 a, const uint256& b) noexcept { return a += b; }
        friend constexpr uint256 operator-(uint256 a, const uint256& b) noexcept { return a -= b; }
        friend constexpr uint256 operator*(uint256 a, const uint256& b) noexcept { return a *= b; }
        friend constexpr uint256 operator/(uint256 a, const uint256& b) { return a /= b; }
        friend constexpr uint256 operator%(uint256 a, const uint256& b) { return a %= b; }
        friend constexpr uint256 operator&(uint256 a, const uint256& b) noexcept { return a &= b; }
        friend constexpr uint256 operator|(uint256 a, const uint256& b) noexcept { return a |= b; }
        friend constexpr uint256 operator^(uint256 a, const uint256& b) noexcept { return a ^= b; }
        /// \brief Shift left, 256 or more giving 0.
        /// \param a The value, by value.
        /// \param n The shift.
        /// \return The result.
        friend constexpr uint256 operator<<(uint256 a, unsigned n) noexcept { return a <<= n; }
        /// \brief Shift right, 256 or more giving 0.
        /// \param a The value, by value.
        /// \param n The shift.
        /// \return The result.
        friend constexpr uint256 operator>>(uint256 a, unsigned n) noexcept { return a >>= n; }
        ///@}

        /// \brief Bitwise not.
        /// \param a The value.
        /// \return Its complement.
        friend constexpr uint256 operator~(uint256 a) noexcept
        {
            for (auto& l : a.limbs_)
                l = ~l;
            return a;
        }

        // ---- checked arithmetic ---------------------------------------------------------------------------------

        /// \brief `a + b`, or `std::nullopt` on overflow.
        /// \param a One addend.
        /// \param b The other.
        /// \return The sum when it fits.
        [[nodiscard]] static constexpr std::optional<uint256> checked_add(uint256 a, const uint256& b) noexcept
        {
            if (a.add_with_carry(b))
                return std::nullopt;
            return a;
        }

        /// \brief `a - b`, or `std::nullopt` when `b > a`.
        /// \param a The minuend.
        /// \param b The subtrahend.
        /// \return The difference when it is not negative.
        [[nodiscard]] static constexpr std::optional<uint256> checked_sub(uint256 a, const uint256& b) noexcept
        {
            if (a.sub_with_borrow(b))
                return std::nullopt;
            return a;
        }

        /// \brief `a * b`, or `std::nullopt` on overflow.
        /// \param a One factor.
        /// \param b The other.
        /// \return The product when it fits.
        [[nodiscard]] static constexpr std::optional<uint256> checked_mul(const uint256& a, const uint256& b) noexcept
        {
            const auto full = mul_full(a, b);
            if ((full[4] | full[5] | full[6] | full[7]) != 0)
                return std::nullopt;
            return from_limbs(full[0], full[1], full[2], full[3]);
        }

        /// \brief The quotient and remainder of `*this / divisor`.
        /// \pre `!divisor.is_zero()` (asserted).
        /// \param divisor The divisor.
        /// \return `{quotient, remainder}`.
        [[nodiscard]] constexpr std::pair<uint256, uint256> divmod(const uint256& divisor) const
        {
            assert(!divisor.is_zero() && "snicholls::uint256: division by zero");
            return divmod_knuth(*this, divisor);
        }

        /// \brief `a / b`, or `std::nullopt` when `b` is 0.
        /// \param a The dividend.
        /// \param b The divisor.
        /// \return The quotient.
        [[nodiscard]] static constexpr std::optional<uint256> checked_div(const uint256& a, const uint256& b) noexcept
        {
            if (b.is_zero())
                return std::nullopt;
            return divmod_knuth(a, b).first;
        }

        /// \brief `{a / b, a % b}`, or `std::nullopt` when `b` is 0.
        /// \param a The dividend.
        /// \param b The divisor.
        /// \return The quotient and remainder.
        [[nodiscard]] static constexpr std::optional<std::pair<uint256, uint256>> checked_divmod(const uint256& a,
                                                                                                 const uint256& b) noexcept
        {
            if (b.is_zero())
                return std::nullopt;
            return divmod_knuth(a, b);
        }

        // ---- text ----------------------------------------------------------------------------------------------

        /// \brief The value of a decimal string: digits only, no sign, no separators.
        /// \param text The digits.
        /// \return The value, or `std::nullopt` when the text is empty, has a non-digit, or exceeds 2^256 - 1.
        [[nodiscard]] static constexpr std::optional<uint256> from_decimal(std::string_view text) noexcept
        {
            if (text.empty())
                return std::nullopt;
            uint256 acc;
            for (const char c : text) {
                if (c < '0' || c > '9')
                    return std::nullopt;
                if (acc.mul_small(10) != 0)
                    return std::nullopt;
                if (acc.add_small(static_cast<std::uint64_t>(c - '0')) != 0)
                    return std::nullopt;
            }
            return acc;
        }

        /// \brief The value of a hexadecimal string, with or without `0x`, either case, leading zeros allowed.
        /// \param text The digits.
        /// \return The value, or `std::nullopt` when there are no digits, a non-hex character, or more than 64 significant nibbles.
        [[nodiscard]] static constexpr std::optional<uint256> from_hex(std::string_view text) noexcept
        {
            if (text.size() >= 2 && text[0] == '0' && (text[1] == 'x' || text[1] == 'X'))
                text.remove_prefix(2);
            if (text.empty())
                return std::nullopt;
            while (text.size() > 1 && text[0] == '0')
                text.remove_prefix(1);
            if (text.size() > 64)
                return std::nullopt;
            uint256 acc;
            for (const char c : text) {
                std::uint64_t d = 0;
                if (c >= '0' && c <= '9')
                    d = static_cast<std::uint64_t>(c - '0');
                else if (c >= 'a' && c <= 'f')
                    d = static_cast<std::uint64_t>(c - 'a' + 10);
                else if (c >= 'A' && c <= 'F')
                    d = static_cast<std::uint64_t>(c - 'A' + 10);
                else
                    return std::nullopt;
                acc <<= 4;
                acc.limbs_[0] |= d;
            }
            return acc;
        }

        /// \brief The value in decimal.
        /// \return The digits, `"0"` for 0.
        /// \throws std::bad_alloc.
        [[nodiscard]] std::string to_string() const
        {
            if (is_zero())
                return "0";
            std::string out;
            uint256 v = *this;
            while (!v.is_zero()) {
                const std::uint32_t chunk = v.divmod_small(1'000'000'000u);   // nine digits at a time
                std::string digits = std::to_string(chunk);
                if (!v.is_zero())
                    digits.insert(0, 9 - digits.size(), '0');
                out.insert(0, digits);
            }
            return out;
        }

        /// \brief The base units in a decimal amount of a token with `decimals` decimals: ethers' `parseUnits`.
        ///
        /// `parse_units("10000.5", 6)` is 10,000,500,000 (USDT and USDC have 6 decimals; ETH 18). Digits beyond
        /// `decimals` are accepted only when they are zeros, so Binance's eight-place strings (`"0.00100000"`) parse
        /// for a 6-decimal token, but nothing is ever rounded away.
        /// \param text     Digits with at most one `.`: `"12"`, `"12.5"`, `".5"`, `"12."`; no sign, no separators.
        /// \param decimals The token's decimals, at most 77.
        /// \return The base units, or `std::nullopt` for malformed text, a non-zero digit past `decimals`, or overflow.
        [[nodiscard]] static constexpr std::optional<uint256> parse_units(std::string_view text, unsigned decimals) noexcept
        {
            if (decimals > 77)
                return std::nullopt;
            const std::size_t point = text.find('.');
            const std::string_view whole = text.substr(0, point);
            const std::string_view fraction = point == std::string_view::npos ? std::string_view{} : text.substr(point + 1);
            if (whole.empty() && fraction.empty())
                return std::nullopt;
            uint256 acc;
            const auto push = [&acc](char c) {
                return c >= '0' && c <= '9' && acc.mul_small(10) == 0 && acc.add_small(static_cast<std::uint64_t>(c - '0')) == 0;
            };
            for (const char c : whole)
                if (!push(c))
                    return std::nullopt;
            for (std::size_t i = 0; i < fraction.size(); ++i) {
                const char c = fraction[i];
                if (i < decimals) {
                    if (!push(c))
                        return std::nullopt;
                } else if (c != '0') {
                    return std::nullopt;   // a digit the token cannot hold; never rounded
                }
            }
            for (std::size_t i = fraction.size(); i < decimals; ++i)
                if (acc.mul_small(10) != 0)
                    return std::nullopt;
            return acc;
        }

        /// \brief How `format_units` writes the fraction.
        enum class fraction_digits {
            all,       ///< Exactly `decimals` digits: `"9997.500000"`.
            trimmed,   ///< Trailing zeros removed, and the point with them: `"9997.5"`, `"10000"`.
        };

        /// \brief The value as a decimal amount of a token with `decimals` decimals: ethers' `formatUnits`.
        /// \param decimals The token's decimals; 0 gives the integer.
        /// \param digits   All of the fraction's digits, or trimmed.
        /// \return The text, exact.
        /// \throws std::bad_alloc.
        [[nodiscard]] std::string format_units(unsigned decimals, fraction_digits digits = fraction_digits::all) const
        {
            std::string text = to_string();
            if (decimals == 0)
                return text;
            if (text.size() <= decimals)
                text.insert(0, decimals + 1 - text.size(), '0');
            text.insert(text.size() - decimals, 1, '.');
            if (digits == fraction_digits::trimmed) {
                text.erase(text.find_last_not_of('0') + 1);
                if (text.back() == '.')
                    text.pop_back();
            }
            return text;
        }

        /// \brief The value in hexadecimal: `0x`, lowercase, no leading zeros, `"0x0"` for 0.
        /// \return The text.
        /// \throws std::bad_alloc.
        [[nodiscard]] std::string to_hex() const
        {
            const std::string word = to_abi_word();
            const std::size_t first = word.find_first_not_of('0');
            return "0x" + (first == std::string::npos ? std::string{"0"} : word.substr(first));
        }

        /// \brief Write the value in decimal to a stream: the `snicholls::printable` member, so `acout{} << amount` works.
        /// \param os The stream.
        /// \throws std::bad_alloc.
        void print(std::ostream& os) const { os << to_string(); }

        /// \brief The value as one ABI word: 64 lowercase hex digits, zero-padded, no prefix.
        /// \return The text.
        /// \throws std::bad_alloc.
        [[nodiscard]] std::string to_abi_word() const
        {
            constexpr const char* hex = "0123456789abcdef";
            std::string out(64, '0');
            for (std::size_t i = 0; i < 4; ++i)
                for (std::size_t n = 0; n < 16; ++n)
                    out[63 - (i * 16 + n)] = hex[(limbs_[i] >> (4 * n)) & 0xfu];
            return out;
        }

    private:
        // Add with carry out: returns 1 on overflow.
        constexpr std::uint64_t add_with_carry(const uint256& other) noexcept
        {
            std::uint64_t carry = 0;
            for (std::size_t i = 0; i < 4; ++i) {
                const std::uint64_t sum = limbs_[i] + other.limbs_[i];
                std::uint64_t c = sum < limbs_[i] ? 1u : 0u;
                const std::uint64_t with_carry = sum + carry;
                c += with_carry < sum ? 1u : 0u;
                limbs_[i] = with_carry;
                carry = c;
            }
            return carry;
        }

        // Subtract with borrow out: returns 1 when other > *this.
        constexpr std::uint64_t sub_with_borrow(const uint256& other) noexcept
        {
            std::uint64_t borrow = 0;
            for (std::size_t i = 0; i < 4; ++i) {
                const std::uint64_t diff = limbs_[i] - other.limbs_[i];
                std::uint64_t b = limbs_[i] < other.limbs_[i] ? 1u : 0u;
                const std::uint64_t with_borrow = diff - borrow;
                b += diff < borrow ? 1u : 0u;
                limbs_[i] = with_borrow;
                borrow = b;
            }
            return borrow;
        }

        // Multiply by a 64-bit value in place; returns the limb that fell off the top (0 when it fits).
        constexpr std::uint64_t mul_small(std::uint64_t m) noexcept
        {
            std::uint64_t carry = 0;
            for (std::size_t i = 0; i < 4; ++i) {
                const detail::wide_product p = detail::umul128(limbs_[i], m);
                const std::uint64_t low = p.low + carry;
                limbs_[i] = low;
                carry = p.high + (low < carry ? 1u : 0u);
            }
            return carry;
        }

        // Add a 64-bit value in place; returns the carry out.
        constexpr std::uint64_t add_small(std::uint64_t v) noexcept { return add_with_carry(uint256{v}); }

        // Divide in place by a 32-bit value; returns the remainder.
        constexpr std::uint32_t divmod_small(std::uint32_t d) noexcept
        {
            std::uint64_t rem = 0;
            for (int i = 3; i >= 0; --i) {
                const auto idx = static_cast<std::size_t>(i);
                const std::uint64_t high = (rem << 32) | (limbs_[idx] >> 32);
                const std::uint64_t qh = high / d;
                rem = high % d;
                const std::uint64_t low = (rem << 32) | (limbs_[idx] & 0xffffffffu);
                const std::uint64_t ql = low / d;
                rem = low % d;
                limbs_[idx] = (qh << 32) | ql;
            }
            return static_cast<std::uint32_t>(rem);
        }

        // The full 512-bit product, as eight limbs.
        static constexpr std::array<std::uint64_t, 8> mul_full(const uint256& a, const uint256& b) noexcept
        {
            std::array<std::uint64_t, 8> r{};
            for (std::size_t i = 0; i < 4; ++i) {
                std::uint64_t carry = 0;
                for (std::size_t j = 0; j < 4; ++j) {
                    const detail::wide_product p = detail::umul128(a.limbs_[i], b.limbs_[j]);
                    std::uint64_t low = r[i + j] + p.low;
                    std::uint64_t c = low < p.low ? 1u : 0u;
                    const std::uint64_t with_carry = low + carry;
                    c += with_carry < low ? 1u : 0u;
                    r[i + j] = with_carry;
                    carry = p.high + c;   // p.high <= 2^64 - 2, so no overflow
                }
                r[i + 4] = carry;
            }
            return r;
        }

        // Knuth's algorithm D (TAOCP 4.3.1), base 2^32, in the form of Hacker's Delight's divmnu.
        static constexpr std::pair<uint256, uint256> divmod_knuth(const uint256& dividend, const uint256& divisor) noexcept
        {
            std::array<std::uint32_t, 8> u{}, v{};
            for (std::size_t i = 0; i < 4; ++i) {
                u[2 * i] = static_cast<std::uint32_t>(dividend.limbs_[i]);
                u[2 * i + 1] = static_cast<std::uint32_t>(dividend.limbs_[i] >> 32);
                v[2 * i] = static_cast<std::uint32_t>(divisor.limbs_[i]);
                v[2 * i + 1] = static_cast<std::uint32_t>(divisor.limbs_[i] >> 32);
            }
            int n = 8;
            while (n > 1 && v[static_cast<std::size_t>(n - 1)] == 0)
                --n;
            int u_digits = 8;
            while (u_digits > 0 && u[static_cast<std::size_t>(u_digits - 1)] == 0)
                --u_digits;
            if (u_digits < n)
                return {uint256{}, dividend};

            std::array<std::uint32_t, 8> q{}, r{};
            if (n == 1) {   // short division
                std::uint64_t k = 0;
                for (int j = u_digits - 1; j >= 0; --j) {
                    const auto idx = static_cast<std::size_t>(j);
                    const std::uint64_t cur = (k << 32) | u[idx];
                    q[idx] = static_cast<std::uint32_t>(cur / v[0]);
                    k = cur % v[0];
                }
                r[0] = static_cast<std::uint32_t>(k);
                return {from_digits(q), from_digits(r)};
            }

            const int m = u_digits - n;
            const int s = std::countl_zero(v[static_cast<std::size_t>(n - 1)]);   // normalise so vn[n-1] has its top bit set
            std::array<std::uint32_t, 8> vn{};
            std::array<std::uint32_t, 9> un{};
            for (int i = n - 1; i > 0; --i)
                vn[static_cast<std::size_t>(i)] = static_cast<std::uint32_t>(
                    (static_cast<std::uint64_t>(v[static_cast<std::size_t>(i)]) << s) |
                    (static_cast<std::uint64_t>(v[static_cast<std::size_t>(i - 1)]) >> (32 - s)));
            vn[0] = v[0] << s;
            un[static_cast<std::size_t>(u_digits)] = static_cast<std::uint32_t>(
                static_cast<std::uint64_t>(u[static_cast<std::size_t>(u_digits - 1)]) >> (32 - s));
            for (int i = u_digits - 1; i > 0; --i)
                un[static_cast<std::size_t>(i)] = static_cast<std::uint32_t>(
                    (static_cast<std::uint64_t>(u[static_cast<std::size_t>(i)]) << s) |
                    (static_cast<std::uint64_t>(u[static_cast<std::size_t>(i - 1)]) >> (32 - s)));
            un[0] = u[0] << s;

            constexpr std::uint64_t base = std::uint64_t{1} << 32;
            for (int j = m; j >= 0; --j) {
                const auto jn = static_cast<std::size_t>(j + n);
                const std::uint64_t numerator = (static_cast<std::uint64_t>(un[jn]) << 32) | un[jn - 1];
                std::uint64_t qhat = numerator / vn[static_cast<std::size_t>(n - 1)];
                std::uint64_t rhat = numerator % vn[static_cast<std::size_t>(n - 1)];
                while (qhat >= base ||
                       qhat * vn[static_cast<std::size_t>(n - 2)] > ((rhat << 32) | un[jn - 2])) {
                    --qhat;
                    rhat += vn[static_cast<std::size_t>(n - 1)];
                    if (rhat >= base)
                        break;
                }
                // multiply and subtract
                std::int64_t k = 0;
                std::int64_t t = 0;
                for (int i = 0; i < n; ++i) {
                    const auto ij = static_cast<std::size_t>(i + j);
                    const std::uint64_t p = qhat * vn[static_cast<std::size_t>(i)];
                    t = static_cast<std::int64_t>(un[ij]) - k - static_cast<std::int64_t>(p & 0xffffffffu);
                    un[ij] = static_cast<std::uint32_t>(t);
                    k = static_cast<std::int64_t>(p >> 32) - (t >> 32);
                }
                t = static_cast<std::int64_t>(un[jn]) - k;
                un[jn] = static_cast<std::uint32_t>(t);
                q[static_cast<std::size_t>(j)] = static_cast<std::uint32_t>(qhat);
                if (t < 0) {   // qhat was one too large: add back
                    --q[static_cast<std::size_t>(j)];
                    std::uint64_t carry = 0;
                    for (int i = 0; i < n; ++i) {
                        const auto ij = static_cast<std::size_t>(i + j);
                        const std::uint64_t sum = static_cast<std::uint64_t>(un[ij]) + vn[static_cast<std::size_t>(i)] + carry;
                        un[ij] = static_cast<std::uint32_t>(sum);
                        carry = sum >> 32;
                    }
                    un[jn] = static_cast<std::uint32_t>(un[jn] + carry);
                }
            }
            // the remainder is un, un-normalised
            for (int i = 0; i < n; ++i) {
                const auto idx = static_cast<std::size_t>(i);
                r[idx] = static_cast<std::uint32_t>((static_cast<std::uint64_t>(un[idx]) >> s) |
                                                    (static_cast<std::uint64_t>(un[idx + 1]) << (32 - s)));
            }
            return {from_digits(q), from_digits(r)};
        }

        static constexpr uint256 from_digits(const std::array<std::uint32_t, 8>& d) noexcept
        {
            return from_limbs(d[0] | (static_cast<std::uint64_t>(d[1]) << 32), d[2] | (static_cast<std::uint64_t>(d[3]) << 32),
                              d[4] | (static_cast<std::uint64_t>(d[5]) << 32), d[6] | (static_cast<std::uint64_t>(d[7]) << 32));
        }

        std::array<std::uint64_t, 4> limbs_{};
    };

} // namespace snicholls

/// \brief `std::hash` of a `snicholls::uint256`: `hash_combine` over the limbs.
/// \ingroup core
template <>
struct std::hash<snicholls::uint256> {
    /// \brief The hash.
    /// \param v The value.
    /// \return The combined limb hashes.
    std::size_t operator()(const snicholls::uint256& v) const noexcept
    {
        std::size_t seed = 0;
        for (const auto limb : v.limbs())
            seed = snicholls::hash_combine(seed, limb);
        return seed;
    }
};

#endif /* snicholls_core_uint256_hpp */
