//
//  money.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::money` and `snicholls::money_of<T>`: an exact amount of one asset, a sign and 256 bits of
/// minor units, with checked arithmetic and exact text.
/// \ingroup core
///
/// **Overview.** An amount is a whole number of minor units of one asset: cents, wei, satoshi. Neither a `double`,
/// which rounds, nor an `int64`, which overflows at 9.2 ETH in wei, can hold every amount a ledger meets, so the
/// magnitude is a `uint256`: every raw on-chain amount is representable exactly. The sign is separate, so debits
/// and credits share one type and the whole uint256 range is available with either sign. Zero is never negative.
///
/// Two forms:
/// - `money`: the asset is a run-time value (`snicholls::asset`), for data such as a ledger row;
/// - `money_of<T>`: the asset is a type (`money_of<currency::AUD>`, `money_of<native::ETH>`), for code that knows
///   its currency. Mixing two currencies is a compile error, and the asset is not stored at all.
///
/// Arithmetic is exact and checked:
/// - `add` and `subtract` return `result`, never wrap, and work within one asset only (for `money`, a mismatch is an
///   error; for `money_of`, there is no overload);
/// - `multiply` by an integer is checked too;
/// - conversion between assets goes through an explicit `rate` and a named `rounding` mode (`money/rate.hpp`).
///
/// Text is exact at the asset's scale, and no `double` is involved anywhere:
/// - `parse_money(asset, "1000.5")` refuses digits the scale cannot hold unless a rounding mode is given;
/// - `to_chars` writes the minimal form (`"1000.5"`, `"-0.000001"`, `"1"`: viem's `formatUnits`), or every
///   fractional digit on request.
///
/// \snippet money/money_demo.cpp money
///
/// **Relationships.** `money/asset.hpp`, `money/rate.hpp`, `money/iso4217.hpp`, `money/native.hpp`, `uint256.hpp`,
/// `result.hpp`.
///
/// `decimal.hpp` remains for bounded values within 18 digits (a price on a wire, a rate in a message); it is not for
/// balances. The two convert:
/// - `from_decimal(asset, d)` is exact, refusing digits beyond the asset's scale unless a `rounding` mode is named;
/// - `to_decimal(m)` gives `std::nullopt` when the amount needs more than 18 significant digits. Group:
/// \ref core.
///
/// **Implementations.** One representation, `{ sign, uint256 magnitude }`, with the asset beside it (`money`) or in
/// the type (`money_of`).
///
/// **Guarantees.**
/// - constexpr throughout.
/// - Arithmetic and `to_chars` are `noexcept` and allocation-free; only `to_string` allocates.
/// - Both forms are trivially copyable.
/// - `money_of<T>` is the size of its sign plus its `uint256`.
/// - Equality compares asset, sign and magnitude; zero is normalised to non-negative, so `-0 == 0`.
///
/// **Errors.** `money_errc`, returned through `result`:
/// - `asset_mismatch`: two different assets;
/// - `overflow`: a magnitude beyond 2^256 - 1;
/// - `malformed`: text that is not a decimal amount;
/// - `too_many_digits`: more fractional digits than the scale, with no rounding mode;
/// - `buffer_too_small`.
///
/// **Performance.** `benchmarks/money/money_benchmark.cpp`: `add` against hand-written signed uint256 addition, after
/// a fairness check.
///
/// **Tests and demos.** `tests/money/money_test.cpp`, with fixtures from Python's `decimal`
/// (`scripts/generate_money_fixtures.py`, CTest `generated.money_fixtures_up_to_date`); `demos/money/money_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<charconv>`, `<compare>`, `<cstddef>`, `<cstdint>`, `<optional>`, `<ostream>`,
/// `<string>`, `<string_view>`, `money/asset.hpp`, `result.hpp`, `uint256.hpp`.
///
/// **Known issues.** None.
///
/// **Status.** New in 0.22.0.

#ifndef snicholls_core_money_money_hpp
#define snicholls_core_money_money_hpp

#include <snicholls/core/decimal.hpp>
#include <snicholls/core/money/asset.hpp>
#include <snicholls/core/result.hpp>
#include <snicholls/core/uint256.hpp>

#include <array>
#include <charconv>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>

namespace snicholls {

    /// \brief Why a money operation failed.
    enum class money_errc : std::uint8_t {
        asset_mismatch = 1,   ///< the operands are of different assets
        overflow,             ///< the magnitude would exceed 2^256 - 1
        malformed,            ///< the text is not a decimal amount
        too_many_digits,      ///< more fractional digits than the asset's scale, and no rounding mode was given
        buffer_too_small,     ///< `to_chars` was given too little room
    };

    /// \brief How a value that falls between two minor units is rounded. There is no default: every caller names one.
    enum class rounding : std::uint8_t {
        half_even,        ///< to the nearer; a tie to the even neighbour (banker's rounding)
        half_up,          ///< to the nearer; a tie away from zero
        half_down,        ///< to the nearer; a tie toward zero
        toward_zero,      ///< truncate
        away_from_zero,   ///< up in magnitude whenever anything is left
        floor,            ///< toward negative infinity
        ceiling,          ///< toward positive infinity
    };

    namespace detail {
        // The rounding test on r and rest = d - r, of any unsigned type: r > rest is above half, r == rest a tie.
        template <typename U>
        [[nodiscard]] constexpr bool rounds_away_by(const U& r, const U& rest, bool odd, bool negative, rounding mode) noexcept
        {
            switch (mode) {
            case rounding::toward_zero: return false;
            case rounding::away_from_zero: return true;
            case rounding::floor: return negative;
            case rounding::ceiling: return !negative;
            case rounding::half_up: return r >= rest;
            case rounding::half_down: return r > rest;
            case rounding::half_even: return r > rest || (r == rest && odd);
            }
            return false;
        }

        // Whether q + r/d, of the given sign, rounds up in magnitude (to q + 1) under `mode`. 0 <= r < d. When r and d
        // fit in 64 bits the comparisons are on std::uint64_t, with the same answer.
        [[nodiscard]] constexpr bool rounds_away(const uint256& q, const uint256& r, const uint256& d, bool negative,
                                                 rounding mode) noexcept
        {
            if (r.is_zero())
                return false;
            if ((r.limb(1) | r.limb(2) | r.limb(3) | d.limb(1) | d.limb(2) | d.limb(3)) == 0)
                return rounds_away_by(r.limb(0), d.limb(0) - r.limb(0), q.bit(0), negative, mode);
            return rounds_away_by(r, d - r, q.bit(0), negative, mode);
        }
    } // namespace detail

    class money;

    /// \brief An exact amount of a run-time asset: the asset, a sign and a uint256 magnitude of minor units.
    /// \ingroup core
    class money {
    public:
        /// \brief Zero of no asset (an empty code at scale 0); assign a real amount before use.
        constexpr money() noexcept = default;

        /// \brief Zero of `a`.
        /// \param a The asset.
        /// \return The amount.
        [[nodiscard]] static constexpr money zero(const asset& a) noexcept { return from_units(a, 0); }

        /// \brief `units` minor units of `a`, negative if `negative`. A negative zero is normalised to zero.
        /// \param a        The asset.
        /// \param units    The magnitude in minor units.
        /// \param negative The sign.
        /// \return The amount.
        [[nodiscard]] static constexpr money from_units(const asset& a, const uint256& units, bool negative = false) noexcept
        {
            money m;
            m.asset_ = a;
            m.magnitude_ = units;
            m.negative_ = negative && !units.is_zero();
            return m;
        }

        /// \brief The asset.
        /// \return The asset.
        [[nodiscard]] constexpr const asset& asset_of() const noexcept { return asset_; }
        /// \brief The magnitude, in minor units.
        /// \return The magnitude.
        [[nodiscard]] constexpr const uint256& magnitude() const noexcept { return magnitude_; }
        /// \brief Whether the amount is below zero.
        /// \return The sign.
        [[nodiscard]] constexpr bool is_negative() const noexcept { return negative_; }
        /// \brief Whether the amount is zero.
        /// \return Whether the magnitude is zero.
        [[nodiscard]] constexpr bool is_zero() const noexcept { return magnitude_.is_zero(); }
        /// \brief -1, 0 or 1.
        /// \return The sign as an integer.
        [[nodiscard]] constexpr int signum() const noexcept { return is_zero() ? 0 : (negative_ ? -1 : 1); }

        /// \brief The amount with its sign reversed.
        /// \return The negation.
        [[nodiscard]] constexpr money operator-() const noexcept { return from_units(asset_, magnitude_, !negative_); }

        /// \brief Equality: the same asset, sign and magnitude.
        friend constexpr bool operator==(const money&, const money&) noexcept = default;

        /// \brief Write the amount and its code to a stream, `"1000.5 ETH"`: the `snicholls::printable` member.
        /// \param os The stream.
        /// \throws std::bad_alloc.
        void print(std::ostream& os) const;

    private:
        asset asset_{};
        bool negative_ = false;
        uint256 magnitude_{};
    };

    /// \brief An exact amount of a compile-time asset `T`: a sign and a uint256 magnitude, the asset in the type.
    /// \tparam T An `asset_like` type: `currency::AUD`, `native::ETH`, `with_scale<currency::XAU, 18>`, a token.
    /// \ingroup core
    template <asset_like T>
    class money_of {
    public:
        /// \brief The asset type.
        using asset_type = T;

        /// \brief Zero.
        constexpr money_of() noexcept = default;

        /// \brief `units` minor units, negative if `negative`.
        /// \param units    The magnitude.
        /// \param negative The sign.
        /// \return The amount.
        [[nodiscard]] static constexpr money_of from_units(const uint256& units, bool negative = false) noexcept
        {
            money_of m;
            m.magnitude_ = units;
            m.negative_ = negative && !units.is_zero();
            return m;
        }

        /// \brief The asset's value.
        /// \return `asset_v<T>`.
        [[nodiscard]] static constexpr asset asset_of() noexcept { return asset_v<T>; }
        /// \brief The magnitude.
        /// \return The magnitude in minor units.
        [[nodiscard]] constexpr const uint256& magnitude() const noexcept { return magnitude_; }
        /// \brief The sign.
        /// \return Whether below zero.
        [[nodiscard]] constexpr bool is_negative() const noexcept { return negative_; }
        /// \brief Whether zero.
        /// \return Whether the magnitude is zero.
        [[nodiscard]] constexpr bool is_zero() const noexcept { return magnitude_.is_zero(); }

        /// \brief The negation.
        /// \return The amount with its sign reversed.
        [[nodiscard]] constexpr money_of operator-() const noexcept { return from_units(magnitude_, !negative_); }

        /// \brief As a run-time `money`.
        /// \return The same amount, with its asset as a value.
        [[nodiscard]] constexpr money to_money() const noexcept { return money::from_units(asset_of(), magnitude_, negative_); }

        /// \brief From a run-time `money` of the same asset.
        /// \param m The amount.
        /// \return The typed amount, or `asset_mismatch`.
        [[nodiscard]] static constexpr result<money_of, money_errc> from_money(const money& m) noexcept
        {
            if (m.asset_of() != asset_of())
                return err(money_errc::asset_mismatch);
            return ok(from_units(m.magnitude(), m.is_negative()));
        }

        /// \brief Equality.
        friend constexpr bool operator==(const money_of&, const money_of&) noexcept = default;
        /// \brief Order: by value, negative below positive.
        /// \param a An amount.
        /// \param b An amount.
        /// \return The order.
        friend constexpr std::strong_ordering operator<=>(const money_of& a, const money_of& b) noexcept
        {
            if (a.signum() != b.signum())
                return a.signum() <=> b.signum();
            return a.negative_ ? b.magnitude_ <=> a.magnitude_ : a.magnitude_ <=> b.magnitude_;
        }

    private:
        constexpr int signum() const noexcept { return is_zero() ? 0 : (negative_ ? -1 : 1); }
        bool negative_ = false;
        uint256 magnitude_{};
    };

    namespace detail {
        struct signed_magnitude {
            bool negative;
            uint256 magnitude;
        };

        // a + b for sign-magnitude values, or nullopt on overflow
        [[nodiscard]] constexpr std::optional<signed_magnitude> signed_add(bool an, const uint256& a, bool bn,
                                                                         const uint256& b) noexcept
        {
            if (an == bn) {
                const auto sum = uint256::checked_add(a, b);
                if (!sum)
                    return std::nullopt;
                return signed_magnitude{an, *sum};
            }
            if (a >= b)
                return signed_magnitude{an && a != b, a - b};
            return signed_magnitude{bn, b - a};
        }
    } // namespace detail

    /// \brief The sum of two amounts of the same asset.
    /// \param a An amount.
    /// \param b An amount.
    /// \return The sum, or `asset_mismatch`, or `overflow`.
    [[nodiscard]] constexpr result<money, money_errc> add(const money& a, const money& b) noexcept
    {
        if (a.asset_of() != b.asset_of())
            return err(money_errc::asset_mismatch);
        const auto s = detail::signed_add(a.is_negative(), a.magnitude(), b.is_negative(), b.magnitude());
        if (!s)
            return err(money_errc::overflow);
        return ok(money::from_units(a.asset_of(), s->magnitude, s->negative));
    }

    /// \brief `a - b`, for two amounts of the same asset.
    /// \param a An amount.
    /// \param b An amount.
    /// \return The difference, or `asset_mismatch`, or `overflow`.
    [[nodiscard]] constexpr result<money, money_errc> subtract(const money& a, const money& b) noexcept { return add(a, -b); }

    /// \brief The sum of two amounts of `T`.
    /// \param a An amount.
    /// \param b An amount.
    /// \return The sum, or `overflow`.
    template <asset_like T>
    [[nodiscard]] constexpr result<money_of<T>, money_errc> add(const money_of<T>& a, const money_of<T>& b) noexcept
    {
        const auto s = detail::signed_add(a.is_negative(), a.magnitude(), b.is_negative(), b.magnitude());
        if (!s)
            return err(money_errc::overflow);
        return ok(money_of<T>::from_units(s->magnitude, s->negative));
    }

    /// \brief `a - b` for two amounts of `T`.
    /// \param a An amount.
    /// \param b An amount.
    /// \return The difference, or `overflow`.
    template <asset_like T>
    [[nodiscard]] constexpr result<money_of<T>, money_errc> subtract(const money_of<T>& a, const money_of<T>& b) noexcept
    {
        return add(a, -b);
    }

    /// \brief `a * k`, for a whole number `k`.
    /// \param a The amount.
    /// \param k The multiplier.
    /// \return The product, or `overflow`.
    [[nodiscard]] constexpr result<money, money_errc> multiply(const money& a, const uint256& k) noexcept
    {
        const auto p = uint256::checked_mul(a.magnitude(), k);
        if (!p)
            return err(money_errc::overflow);
        return ok(money::from_units(a.asset_of(), *p, a.is_negative()));
    }

    /// \brief `a * k` for an amount of `T`.
    /// \param a The amount.
    /// \param k The multiplier.
    /// \return The product, or `overflow`.
    template <asset_like T>
    [[nodiscard]] constexpr result<money_of<T>, money_errc> multiply(const money_of<T>& a, const uint256& k) noexcept
    {
        const auto p = uint256::checked_mul(a.magnitude(), k);
        if (!p)
            return err(money_errc::overflow);
        return ok(money_of<T>::from_units(*p, a.is_negative()));
    }

    /// \brief Compare two amounts of the same asset.
    /// \param a An amount.
    /// \param b An amount.
    /// \return The order, or `asset_mismatch`.
    [[nodiscard]] constexpr result<std::strong_ordering, money_errc> compare(const money& a, const money& b) noexcept
    {
        if (a.asset_of() != b.asset_of())
            return err(money_errc::asset_mismatch);
        if (a.signum() != b.signum())
            return ok(a.signum() <=> b.signum());
        return ok(a.is_negative() ? b.magnitude() <=> a.magnitude() : a.magnitude() <=> b.magnitude());
    }

    /// \brief The text of an amount: `-`, the whole part, and the fraction.
    /// \param a The amount.
    /// \param text A decimal amount: an optional sign, digits with at most one `.` (`"12"`, `"-12.5"`, `".5"`).
    /// \param mode How to round digits beyond the asset's scale; without one they are an error unless all zeros.
    /// \return The amount, or `malformed`, `too_many_digits` or `overflow`.
    [[nodiscard]] constexpr result<money, money_errc> parse_money(const asset& a, std::string_view text,
                                                                 std::optional<rounding> mode = std::nullopt) noexcept
    {
        bool negative = false;
        if (!text.empty() && (text.front() == '-' || text.front() == '+')) {
            negative = text.front() == '-';
            text.remove_prefix(1);
        }
        const std::size_t point = text.find('.');
        const std::string_view whole = text.substr(0, point);
        const std::string_view fraction = point == std::string_view::npos ? std::string_view{} : text.substr(point + 1);
        if (whole.empty() && fraction.empty())
            return err(money_errc::malformed);
        const auto is_digits = [](std::string_view s) {
            for (const char c : s)
                if (c < '0' || c > '9')
                    return false;
            return true;
        };
        if (!is_digits(whole) || !is_digits(fraction))
            return err(money_errc::malformed);
        const unsigned scale = a.scale();
        uint256 units;
        const auto push = [&units](char c) {
            const auto t = uint256::checked_mul(units, 10);
            if (!t)
                return false;
            const auto s = uint256::checked_add(*t, static_cast<std::uint64_t>(c - '0'));
            if (!s)
                return false;
            units = *s;
            return true;
        };
        for (const char c : whole)
            if (!push(c))
                return err(money_errc::overflow);
        for (std::size_t i = 0; i < fraction.size() && i < scale; ++i)
            if (!push(fraction[i]))
                return err(money_errc::overflow);
        for (std::size_t i = fraction.size(); i < scale; ++i)
            if (!push('0'))
                return err(money_errc::overflow);
        if (fraction.size() > scale) {
            const std::string_view rest = fraction.substr(scale);
            bool any = false;
            for (const char c : rest)
                any = any || c != '0';
            if (any) {
                if (!mode)
                    return err(money_errc::too_many_digits);
                // the rest as a fraction r / d of one minor unit, compared without forming 10^len when it is long
                const std::size_t len = rest.size() > 77 ? 77 : rest.size();
                uint256 r;
                for (std::size_t i = 0; i < len; ++i)
                    r = r * 10 + static_cast<std::uint64_t>(rest[i] - '0');
                const uint256 d = uint256::pow10(static_cast<unsigned>(len));
                bool beyond = false;   // nonzero digits past the 77th make a tie impossible and r slightly larger
                for (std::size_t i = len; i < rest.size(); ++i)
                    beyond = beyond || rest[i] != '0';
                bool away = detail::rounds_away(units, r, d, negative, *mode);
                if (beyond && r == d - r && (*mode == rounding::half_down || *mode == rounding::half_even))
                    away = true;   // above the tie, not on it
                if (away) {
                    const auto s = uint256::checked_add(units, 1);
                    if (!s)
                        return err(money_errc::overflow);
                    units = *s;
                }
            }
        }
        return ok(money::from_units(a, units, negative));
    }

    /// \brief An amount of `a` from an exact `decimal`: exact, or rounded by `mode` when the decimal has more places
    /// than the asset's scale.
    /// \param a    The asset.
    /// \param d    The value, in whole units of `a`.
    /// \param mode How to round digits beyond the scale; without one they are an error.
    /// \return The amount, or `too_many_digits` or `overflow`.
    [[nodiscard]] constexpr result<money, money_errc> from_decimal(const asset& a, const decimal& d,
                                                                  std::optional<rounding> mode = std::nullopt) noexcept
    {
        const bool negative = d.is_negative();
        const std::int64_t m = d.mantissa();
        const uint256 magnitude = m < 0 ? uint256(0ull - static_cast<std::uint64_t>(m)) : uint256(static_cast<std::uint64_t>(m));
        const int shift = d.exponent() + static_cast<int>(a.scale());   // units = magnitude * 10^shift
        if (shift >= 0) {
            const auto units = uint256::checked_mul(magnitude, uint256::pow10(static_cast<unsigned>(shift)));
            if (!units)
                return err(money_errc::overflow);
            return ok(money::from_units(a, *units, negative));
        }
        const uint256 divisor = uint256::pow10(static_cast<unsigned>(-shift));
        auto [q, r] = magnitude.divmod(divisor);
        if (!r.is_zero()) {
            if (!mode)
                return err(money_errc::too_many_digits);
            if (detail::rounds_away(q, r, divisor, negative, *mode))
                q = q + 1;
        }
        return ok(money::from_units(a, q, negative));
    }

    /// \brief An amount as an exact `decimal`, in whole units, when it fits: at most 18 significant digits.
    /// \param m The amount.
    /// \return The decimal, or `std::nullopt` if it needs more digits than a `decimal` holds.
    [[nodiscard]] constexpr std::optional<decimal> to_decimal(const money& m) noexcept
    {
        uint256 units = m.magnitude();
        int exponent = -static_cast<int>(m.asset_of().scale());
        while (!units.is_zero()) {   // trailing zeros belong in the exponent
            const auto [q, r] = units.divmod(10);
            if (!r.is_zero())
                break;
            units = q;
            ++exponent;
        }
        const auto small = units.to_uint64();
        if (!small || *small > 999'999'999'999'999'999ull)
            return std::nullopt;
        const auto mantissa = static_cast<std::int64_t>(*small);
        return decimal::from_parts(m.is_negative() ? -mantissa : mantissa, units.is_zero() ? 0 : exponent);
    }

    /// \brief How `to_chars` writes the fraction.
    enum class money_digits : std::uint8_t {
        minimal,   ///< trailing zeros and a bare point removed: `"1000.5"`, `"1"` (viem's `formatUnits`)
        all,       ///< exactly `scale` fractional digits: `"1000.500000"`
    };

    /// \brief Write an amount's number, without its code, into `[first, last)`.
    /// \param first  The buffer's start.
    /// \param last   One past its end. 80 characters always suffice.
    /// \param m      The amount.
    /// \param digits The fraction's form.
    /// \return The end of the text and `std::errc{}`, or `std::errc::value_too_large` if the buffer is too small.
    constexpr std::to_chars_result to_chars(char* first, char* last, const money& m,
                                            money_digits digits = money_digits::minimal) noexcept
    {
        // the magnitude's decimal digits, least significant first
        std::array<char, 80> rev{};
        std::size_t n = 0;
        uint256 v = m.magnitude();
        const uint256 chunk = uint256::pow10(19);
        do {
            const auto [q, r] = v.divmod(chunk);
            std::uint64_t part = r.low64();
            const bool last_chunk = q.is_zero();
            for (int i = 0; i < 19 && (part != 0 || !last_chunk); ++i) {
                rev[n++] = static_cast<char>('0' + part % 10);
                part /= 10;
            }
            v = q;
        } while (!v.is_zero());
        if (n == 0)
            rev[n++] = '0';
        const std::size_t scale = m.asset_of().scale();
        while (n <= scale)   // at least one digit before the point
            rev[n++] = '0';
        std::size_t frac = scale;
        if (digits == money_digits::minimal)
            while (frac > 0 && rev[scale - frac] == '0')
                --frac;
        const std::size_t whole = n - scale;
        // the text in a local buffer first, then copied with a loop bounded by both ends: GCC's -Wstringop-overflow
        // cannot see that a size check alone keeps the writes inside a small destination
        std::array<char, 82> text{};
        std::size_t size = 0;
        if (m.is_negative())
            text[size++] = '-';
        for (std::size_t i = 0; i < whole; ++i)
            text[size++] = rev[n - 1 - i];
        if (frac > 0) {
            text[size++] = '.';
            for (std::size_t i = 0; i < frac; ++i)
                text[size++] = rev[scale - 1 - i];
        }
        if (static_cast<std::size_t>(last - first) < size)
            return {last, std::errc::value_too_large};
        char* out = first;
        for (std::size_t i = 0; i < size && out != last; ++i)
            *out++ = text[i];
        return {out, std::errc{}};
    }

    /// \brief The amount's number as a string: `"1000.5"`.
    /// \param m      The amount.
    /// \param digits The fraction's form.
    /// \return The text.
    /// \throws std::bad_alloc.
    [[nodiscard]] inline std::string to_string(const money& m, money_digits digits = money_digits::minimal)
    {
        std::array<char, 80> buf{};
        const auto r = to_chars(buf.data(), buf.data() + buf.size(), m, digits);
        return std::string(buf.data(), r.ptr);
    }

    inline void money::print(std::ostream& os) const { os << to_string(*this) << ' ' << asset_.code(); }

} // namespace snicholls

#endif // snicholls_core_money_money_hpp
