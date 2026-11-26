//
//  money_ratio.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief Exact arithmetic between `money` and `finance::ratio`: an amount times or divided by a ratio, as an exact
/// ratio or as money at a stated scale with a named rounding mode; the ratio of two amounts; `_ratio` literals.
/// \ingroup core
///
/// **Overview.** A fund's dealing is `units = amount / NAV per unit` and `amount = units * NAV per unit`, where the NAV
/// per unit is a pure rational, not tied to any asset (1.234567 AUD a unit is `ratio{1234567, 1000000}`). These
/// functions do that arithmetic exactly, with one rounding at the end, where the caller names it:
///
/// - `exact_value(m)`: an amount as a ratio of whole units (`1000.00 AUD` is `1000/1`);
/// - `multiply(m, r)` and `divide(m, r)`: the exact product or quotient, as a ratio of whole units of the result;
/// - `multiply(m, r, out, mode)` and `divide(m, r, out, mode)`: the same, as money of the asset `out`, rounded once to
///   its scale under `mode`; computed in 256 bits, so any amount `money` holds works;
/// - `to_money(r, out, mode)`: a ratio of whole units as money of `out`;
/// - `ratio_of(a, b)`: `a / b` for two amounts of any assets (net assets over units outstanding is the NAV per unit);
/// - `"1.234567"_ratio` (`finance::literals`): a ratio checked by the compiler.
///
/// \snippet finance/money_ratio_demo.cpp fund
///
/// **Relationships.** `finance/ratio.hpp`, `money/money.hpp` (`rounding`), `money/rate.hpp` (an asset-to-asset rate,
/// for currency conversion; a `ratio` carries no assets). Group: \ref core.
///
/// **Implementations / variants.** Exact ratio results (64-bit parts) or money results (256-bit intermediate).
///
/// **Guarantees.** `constexpr` and `noexcept`; no allocation. A money result is the exact value rounded once under the
/// named mode; the sign is the product's.
///
/// **Errors.** `std::nullopt` for a division by zero, an amount whose value does not fit a `ratio` (the ratio forms),
/// or a result beyond 256 bits (the money forms). A malformed `_ratio` literal is a compile error.
///
/// **Performance.** Not benchmarked: a 256-bit multiply and divide (`benchmarks/money/` measures those).
///
/// **Tests and demos.** `tests/finance/money_ratio_test.cpp`, `demos/finance/money_ratio_demo.cpp`.
///
/// **Dependencies.** `<cstddef>`, `<cstdint>`, `<limits>`, `<optional>`, `<string_view>`, `finance/ratio.hpp`, `money/money.hpp`, `uint256.hpp`.
///
/// **Known issues.** The ratio forms keep 64-bit parts: an amount in wei with 18 decimals rarely fits; use the money
/// forms, which do not have that limit.
///
/// **Status.** New in 0.27.0.

#ifndef snicholls_core_finance_money_ratio_hpp
#define snicholls_core_finance_money_ratio_hpp

#include <snicholls/core/finance/ratio.hpp>
#include <snicholls/core/money/money.hpp>
#include <snicholls/core/uint256.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string_view>

namespace snicholls::finance {

    namespace detail {
        inline constexpr std::uint64_t max_part = static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());

        // |n| of a ratio part, never INT64_MIN.
        constexpr std::uint64_t magnitude(std::int64_t v) noexcept
        {
            return static_cast<std::uint64_t>(v < 0 ? -v : v);
        }

        // magnitude * num / den rounded under `mode` with the result's sign, or none past 256 bits.
        constexpr std::optional<uint256> scaled(const uint256& magnitude, const uint256& num, const uint256& den, bool negative,
                                                rounding mode) noexcept
        {
            const auto product = uint256::checked_mul(magnitude, num);
            const auto qr = product ? uint256::checked_divmod(*product, den) : std::nullopt;
            if (!qr)
                return std::nullopt;
            if (!snicholls::detail::rounds_away(qr->first, qr->second, den, negative, mode))
                return qr->first;
            return uint256::checked_add(qr->first, uint256::from_limbs(1));
        }

        // m * (num / den) as money of `out`, rounded once.
        constexpr std::optional<money> apply(const money& m, std::int64_t num, std::int64_t den, const asset& out,
                                             rounding mode) noexcept
        {
            if (den == 0)
                return std::nullopt;
            const bool negative = (m.is_negative() != (num < 0)) != (den < 0) && !m.is_zero() && num != 0;
            const auto top = uint256::checked_mul(uint256::from_limbs(magnitude(num)), uint256::pow10(out.scale()));
            const auto bottom = uint256::checked_mul(uint256::from_limbs(magnitude(den)), uint256::pow10(m.asset_of().scale()));
            const auto v = top && bottom ? scaled(m.magnitude(), *top, *bottom, negative, mode) : std::nullopt;
            if (!v)
                return std::nullopt;
            return money::from_units(out, *v, negative && !v->is_zero());
        }
    } // namespace detail

    /// \brief An amount as an exact ratio of whole units of its asset: `1000.00 AUD` is `1000/1`.
    /// \ingroup core
    /// \param m The amount.
    /// \return The ratio, or `std::nullopt` when the minor units or `10^scale` do not fit in 63 bits.
    [[nodiscard]] constexpr std::optional<ratio> exact_value(const money& m) noexcept
    {
        const auto units = m.magnitude().to_uint64();
        if (!units || *units > detail::max_part || m.asset_of().scale() > 18)
            return std::nullopt;
        const auto n = static_cast<std::int64_t>(*units);
        std::int64_t d = 1;
        for (unsigned i = 0; i < m.asset_of().scale(); ++i)
            d *= 10;
        return ratio{m.is_negative() ? -n : n, d};
    }

    /// \brief `m * r`, exact, in whole units.
    /// \ingroup core
    /// \param m The amount.
    /// \param r The ratio.
    /// \return The product, or `std::nullopt` when it does not fit a ratio.
    [[nodiscard]] constexpr std::optional<ratio> multiply(const money& m, const ratio& r) noexcept
    {
        const auto v = exact_value(m);
        return v ? multiply(*v, r) : std::nullopt;
    }

    /// \brief `m / r`, exact, in whole units: units for an amount at a price per unit.
    /// \ingroup core
    /// \param m The amount.
    /// \param r The divisor.
    /// \return The quotient, or `std::nullopt` for zero or when it does not fit a ratio.
    [[nodiscard]] constexpr std::optional<ratio> divide(const money& m, const ratio& r) noexcept
    {
        const auto v = exact_value(m);
        const auto inv = r.inverse();
        return v && inv ? multiply(*v, *inv) : std::nullopt;
    }

    /// \brief `m * r` as money of `out`, rounded once to its scale under `mode`: an amount for units at a price.
    /// \ingroup core
    /// \param m    The amount.
    /// \param r    The ratio.
    /// \param out  The result's asset (its scale is the rounding's).
    /// \param mode The rounding.
    /// \return The amount, or `std::nullopt` past 256 bits.
    [[nodiscard]] constexpr std::optional<money> multiply(const money& m, const ratio& r, const asset& out, rounding mode) noexcept
    {
        return detail::apply(m, r.numerator(), r.denominator(), out, mode);
    }

    /// \brief `m / r` as money of `out`, rounded once to its scale under `mode`: units for an amount at a price.
    /// \ingroup core
    /// \param m    The amount.
    /// \param r    The divisor.
    /// \param out  The result's asset (its scale is the rounding's).
    /// \param mode The rounding.
    /// \return The amount, or `std::nullopt` for zero or past 256 bits.
    [[nodiscard]] constexpr std::optional<money> divide(const money& m, const ratio& r, const asset& out, rounding mode) noexcept
    {
        return detail::apply(m, r.denominator(), r.numerator(), out, mode);
    }

    /// \brief A ratio of whole units as money of `out`, rounded once to its scale under `mode`.
    /// \ingroup core
    /// \param r    The value.
    /// \param out  The asset.
    /// \param mode The rounding.
    /// \return The amount, or `std::nullopt` past 256 bits.
    [[nodiscard]] constexpr std::optional<money> to_money(const ratio& r, const asset& out, rounding mode) noexcept
    {
        const bool negative = r.is_negative();
        const auto v = detail::scaled(uint256::from_limbs(detail::magnitude(r.numerator())), uint256::pow10(out.scale()),
                                      uint256::from_limbs(static_cast<std::uint64_t>(r.denominator())), negative, mode);
        if (!v)
            return std::nullopt;
        return money::from_units(out, *v, negative && !v->is_zero());
    }

    /// \brief `a / b` for two amounts of any assets, exact, in whole units of each: net assets over units outstanding.
    /// \ingroup core
    /// \param a The numerator.
    /// \param b The denominator.
    /// \return The ratio, or `std::nullopt` for a zero `b` or when it does not fit.
    [[nodiscard]] constexpr std::optional<ratio> ratio_of(const money& a, const money& b) noexcept
    {
        const auto x = exact_value(a), y = exact_value(b);
        const auto inv = y ? y->inverse() : std::nullopt;
        return x && inv ? multiply(*x, *inv) : std::nullopt;
    }

    /// \brief Literals for exact ratios.
    /// \ingroup core
    namespace literals {
        /// \brief `"1.234567"_ratio`, `"3/2"_ratio`, `"4.35%"_ratio`: parsed by the compiler; a malformed one does not
        /// compile.
        /// \param text The text.
        /// \param n    Its length.
        /// \return The ratio.
        consteval ratio operator""_ratio(const char* text, std::size_t n)
        {
            const auto r = ratio::parse(std::string_view{text, n});
            if (!r)
                throw "snicholls::finance::_ratio: not a ratio";   // a compile error
            return *r;
        }
    } // namespace literals

} // namespace snicholls::finance

#endif // snicholls_core_finance_money_ratio_hpp
