//
//  rate.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::rate` and `snicholls::convert`: exact conversion between assets, with a named rounding mode and
/// the exact residual.
/// \ingroup core
///
/// **Overview.** A rate is the price of one whole unit of one asset in whole units of another, as an exact ratio
/// of two `uint256`s:
/// - `rate::parse(ETH, USD, "3000.123456789012345678")` holds 3000123456789012345678 / 10^18 exactly, so a rate is
///   never limited to 18 digits;
/// - `rate::make(from, to, numerator, denominator)` gives any ratio;
/// - `inverse()` swaps the two.
///
/// `convert(amount, rate, mode)` computes the target amount in minor units, `amount * numerator * 10^to.scale /
/// (denominator * 10^from.scale)`. The product is formed in 512 bits and divided once, so nothing is lost before
/// the named rounding mode is applied. It returns:
/// - `converted`, the rounded target amount;
/// - `residual`, the exact signed fraction of one target minor unit between the true value and `converted`.
///
/// A ledger posts `converted`, and either keeps the residual exactly or posts it to a rounding account at a finer
/// scale with `residual.at_scale(target, extra_digits)`.
///
/// Two helpers serve a caller that divides by itself, an entitlement engine rounding one line per holder:
/// - `rounds_away(quotient, fraction, mode)`, the rounding test `convert` applies;
/// - `sum_residuals(parts, denominator)`, the exact signed total of many residuals over one denominator;
/// - `convert_rate(rate, places, mode)`, the rate itself priced to `places` decimals (a NAV per unit), with its
///   residual.
///
/// \snippet money/rate_demo.cpp helpers
///
/// \snippet money/rate_demo.cpp rate
///
/// **Relationships.** `money/money.hpp` (`money`, `rounding`), `money/asset.hpp`, `detail/mul_div.hpp`. Group:
/// \ref core.
///
/// **Implementations.** One.
///
/// **Guarantees.** constexpr, `noexcept`, allocation-free. `converted + residual` is the exact value: the property
/// tests check it for every mode.
///
/// **Errors.** `money_errc`:
/// - `asset_mismatch`: the amount is not of the rate's source asset;
/// - `overflow`: the scaled numerator or denominator, or the result, exceeds 256 bits;
/// - `malformed`: a zero denominator, or malformed rate text.
///
/// **Performance.** `benchmarks/money/rate_benchmark.cpp`: `convert` against the same multiply, divide and round
/// written by hand, after a fairness check. A cash dividend of 0.2534 AUD a share to 100,000 holders, half-even per
/// line with the residual kept per line and summed (Release, Apple clang 17, Xeon W-3245 under a load average of 130 to
/// 180, 2026-11-26, CPU time): `convert` per line 8.8 ms (17.3 ms before `uint256`'s 64-bit fast path),
/// `checked_mul` / `divmod` / `rounds_away` per line 5.0 ms (8.9 ms before), the same in `std::uint64_t` by hand
/// 1.0 ms. The remaining gap is mostly memory: a `residual` is 72 bytes a line, against 8 by hand.
///
/// **Tests and demos.** `tests/money/rate_test.cpp`, with fixtures for every rounding mode from Python's `decimal`
/// (`scripts/generate_money_fixtures.py`); `demos/money/rate_demo.cpp`.
///
/// **Dependencies.** `<cstdint>`, `<span>`, `<string_view>`, `detail/mul_div.hpp`, `money/money.hpp`.
///
/// **Known issues.** None.
///
/// **Status.** New in 0.22.0; `rounds_away`, `sum_residuals` and `convert_rate` in 0.25.0.

#ifndef snicholls_core_money_rate_hpp
#define snicholls_core_money_rate_hpp

#include <snicholls/core/detail/mul_div.hpp>
#include <snicholls/core/money/money.hpp>

#include <cstdint>
#include <span>
#include <string_view>

namespace snicholls {

    /// \brief The price of one whole unit of `from` in whole units of `to`, as an exact ratio.
    /// \ingroup core
    class rate {
    public:
        /// \brief The rate `numerator / denominator`.
        /// \param from        The source asset.
        /// \param to          The target asset.
        /// \param numerator   The numerator.
        /// \param denominator The denominator, not zero.
        /// \return The rate, or `malformed` for a zero denominator.
        [[nodiscard]] static constexpr result<rate, money_errc> make(const asset& from, const asset& to,
                                                                     const uint256& numerator,
                                                                     const uint256& denominator) noexcept
        {
            if (denominator.is_zero())
                return err(money_errc::malformed);
            rate r;
            r.from_ = from;
            r.to_ = to;
            r.numerator_ = numerator;
            r.denominator_ = denominator;
            return ok(r);
        }

        /// \brief The rate written as a decimal: `"3000.123456789012345678"` is that many `to` per `from`.
        /// \param from The source asset.
        /// \param to   The target asset.
        /// \param text Digits with at most one `.`, up to 77 digits after it.
        /// \return The rate, exactly, or `malformed` or `overflow`.
        [[nodiscard]] static constexpr result<rate, money_errc> parse(const asset& from, const asset& to,
                                                                      std::string_view text) noexcept
        {
            const std::size_t point = text.find('.');
            if (text.empty() || text == "." ||
                (point != std::string_view::npos && text.find('.', point + 1) != std::string_view::npos))
                return err(money_errc::malformed);
            const std::size_t places = point == std::string_view::npos ? 0 : text.size() - point - 1;
            if (places > 77)
                return err(money_errc::overflow);
            uint256 n;
            for (const char c : text) {
                if (c == '.')
                    continue;
                if (c < '0' || c > '9')
                    return err(money_errc::malformed);
                const auto t = uint256::checked_mul(n, 10);
                const auto s = t ? uint256::checked_add(*t, static_cast<std::uint64_t>(c - '0')) : std::nullopt;
                if (!s)
                    return err(money_errc::overflow);
                n = *s;
            }
            return make(from, to, n, uint256::pow10(static_cast<unsigned>(places)));
        }

        /// \brief The rate given as a `decimal`, exactly.
        /// \param from The source asset.
        /// \param to   The target asset.
        /// \param d    The price of one `from` in `to`, not negative.
        /// \return The rate, or `malformed` for a negative or zero-denominator value.
        [[nodiscard]] static constexpr result<rate, money_errc> from_decimal(const asset& from, const asset& to,
                                                                             const decimal& d) noexcept
        {
            if (d.is_negative())
                return err(money_errc::malformed);
            const uint256 m = static_cast<std::uint64_t>(d.mantissa());
            const int e = d.exponent();
            if (e >= 0)
                return make(from, to, m * uint256::pow10(static_cast<unsigned>(e)), 1);
            return make(from, to, m, uint256::pow10(static_cast<unsigned>(-e)));
        }

        /// \brief The source asset.
        /// \return It.
        [[nodiscard]] constexpr const asset& from() const noexcept { return from_; }
        /// \brief The target asset.
        /// \return It.
        [[nodiscard]] constexpr const asset& to() const noexcept { return to_; }
        /// \brief The numerator.
        /// \return It.
        [[nodiscard]] constexpr const uint256& numerator() const noexcept { return numerator_; }
        /// \brief The denominator.
        /// \return It.
        [[nodiscard]] constexpr const uint256& denominator() const noexcept { return denominator_; }

        /// \brief The rate the other way: `to` in `from`.
        /// \return The inverse, or `malformed` for a zero rate.
        [[nodiscard]] constexpr result<rate, money_errc> inverse() const noexcept
        {
            return make(to_, from_, denominator_, numerator_);
        }

        /// \brief Equality: the same assets and the same numerator and denominator (not reduced).
        friend constexpr bool operator==(const rate&, const rate&) noexcept = default;

    private:
        constexpr rate() noexcept = default;
        asset from_{};
        asset to_{};
        uint256 numerator_{};
        uint256 denominator_{1};
    };

    /// \brief An exact signed fraction of one minor unit: what rounding left behind.
    struct residual {
        /// \brief Whether the fraction is below zero.
        bool negative = false;
        /// \brief The numerator: less than the denominator for one rounding; a `sum_residuals` total can exceed it.
        uint256 numerator{};
        /// \brief The denominator.
        uint256 denominator{1};

        /// \brief Whether nothing was left behind.
        /// \return Whether the numerator is zero.
        [[nodiscard]] constexpr bool is_zero() const noexcept { return numerator.is_zero(); }

        /// \brief The fraction as an amount at a finer scale, truncated: `extra` more decimal places than `target`.
        /// \param target The asset the fraction is of.
        /// \param extra  The additional places.
        /// \return The amount, of an asset with `target`'s code at `target.scale() + extra` places, or `overflow`.
        [[nodiscard]] constexpr result<money, money_errc> at_scale(const asset& target, unsigned extra) const noexcept
        {
            const auto finer = make_asset(target.code(), target.scale() + extra);
            if (!finer)
                return err(money_errc::overflow);
            const auto qr = detail::mul_div(numerator, uint256::pow10(extra), denominator);
            if (!qr)
                return err(money_errc::overflow);
            return ok(money::from_units(*finer, qr->first, negative));
        }

        /// \brief Equality.
        friend constexpr bool operator==(const residual&, const residual&) noexcept = default;
    };

    /// \brief Whether a signed value `quotient + fraction` rounds up in magnitude, to `quotient + 1`, under `mode`.
    ///
    /// The test `convert` and `round` apply, for a caller that divides by itself: an entitlement engine forms
    /// `n / d` per holder and rounds each line by the policy's mode.
    /// \pre `fraction.numerator < fraction.denominator`.
    /// \param quotient The truncated magnitude; its parity breaks a `half_even` tie.
    /// \param fraction What truncation left, `numerator / denominator`; `negative` is the sign of the whole value.
    /// \param mode     The rounding mode.
    /// \return true when the rounded magnitude is `quotient + 1`, false when it is `quotient`. Always false for a zero
    ///         fraction.
    [[nodiscard]] constexpr bool rounds_away(const uint256& quotient, const residual& fraction, rounding mode) noexcept
    {
        return detail::rounds_away(quotient, fraction.numerator, fraction.denominator, fraction.negative, mode);
    }

    /// \brief The exact sum of signed residuals over one denominator: what a run of roundings left behind in total.
    ///
    /// The total is not reduced and its magnitude can be one or more minor units; a ledger compares it with the
    /// difference between the exact total and the sum of the rounded lines. A zero total is not negative.
    /// \param parts       The residuals, every one over `denominator`.
    /// \param denominator The common denominator, not zero.
    /// \return The sum over `denominator` (zero for no parts), or `malformed` for a zero denominator or a part over
    ///         another denominator, or `overflow` when the positive or the negative parts sum past 256 bits.
    /// \note O(parts.size()), no allocation.
    [[nodiscard]] constexpr result<residual, money_errc> sum_residuals(std::span<const residual> parts,
                                                                       const uint256& denominator) noexcept
    {
        if (denominator.is_zero())
            return err(money_errc::malformed);
        uint256 up{}, down{};   // the positive and the negative parts, summed apart: one carry chain per part
        for (const residual& part : parts) {
            if (part.denominator != denominator)
                return err(money_errc::malformed);
            uint256& side = part.negative ? down : up;
            const auto sum = uint256::checked_add(side, part.numerator);
            if (!sum)
                return err(money_errc::overflow);
            side = *sum;
        }
        return ok(up >= down ? residual{false, up - down, denominator} : residual{true, down - up, denominator});
    }

    /// \brief The result of `convert`.
    struct conversion {
        /// \brief The target amount, rounded.
        money converted;
        /// \brief The exact value minus `converted`, in target minor units; its magnitude is below one.
        snicholls::residual residual;
    };

    /// \brief The result of `convert_rate`.
    struct priced_rate {
        /// \brief The rate rounded to the places asked for: its denominator is `10^places`.
        snicholls::rate rounded;
        /// \brief The exact rate minus `rounded`, in units of `10^-places`, over the rate's own denominator; its
        /// magnitude is below one.
        snicholls::residual residual;
    };

    /// \brief A rate priced to `places` decimal places by `mode`: a fund's NAV per unit to four places, say.
    ///
    /// The same rounding as `convert`, for the same seven modes, applied to the rate itself: the rounded rate is
    /// `round(numerator * 10^places / denominator) / 10^places`, and the residual what rounding left, exactly. A rate is
    /// never negative, so `floor` rounds as `toward_zero` and `ceiling` as `away_from_zero`.
    /// \param r      The rate.
    /// \param places The decimal places, at most 77.
    /// \param mode   The rounding mode.
    /// \return The rounded rate (same assets) and the residual, or `overflow` for more than 77 places or a result
    ///         past 256 bits.
    [[nodiscard]] constexpr result<priced_rate, money_errc> convert_rate(const rate& r, unsigned places, rounding mode) noexcept
    {
        if (places > 77)
            return err(money_errc::overflow);
        const uint256 scale = uint256::pow10(places);
        const auto qr = detail::mul_div(r.numerator(), scale, r.denominator());
        if (!qr)
            return err(money_errc::overflow);
        auto [q, rem] = *qr;
        const uint256& d = r.denominator();
        residual left{false, rem, d};
        if (rounds_away(q, left, mode)) {
            const auto up = uint256::checked_add(q, 1);
            if (!up)
                return err(money_errc::overflow);
            q = *up;
            left = residual{true, d - rem, d};   // the rounded rate overshoots by (d - rem) / d
        }
        return ok(priced_rate{*rate::make(r.from(), r.to(), q, scale), left});
    }

    /// \brief `amount` in `r.to()`, rounded by `mode`.
    /// \param amount An amount of `r.from()`.
    /// \param r      The rate.
    /// \param mode   The rounding mode.
    /// \return The conversion, or `asset_mismatch` or `overflow`.
    [[nodiscard]] constexpr result<conversion, money_errc> convert(const money& amount, const rate& r, rounding mode) noexcept
    {
        if (amount.asset_of() != r.from())
            return err(money_errc::asset_mismatch);
        const auto n = uint256::checked_mul(r.numerator(), uint256::pow10(r.to().scale()));
        const auto d = uint256::checked_mul(r.denominator(), uint256::pow10(r.from().scale()));
        if (!n || !d)
            return err(money_errc::overflow);
        const auto qr = detail::mul_div(amount.magnitude(), *n, *d);
        if (!qr)
            return err(money_errc::overflow);
        auto [q, rem] = *qr;
        residual left{amount.is_negative(), rem, *d};
        if (detail::rounds_away(q, rem, *d, amount.is_negative(), mode)) {
            const auto up = uint256::checked_add(q, 1);
            if (!up)
                return err(money_errc::overflow);
            q = *up;
            left = residual{!amount.is_negative(), *d - rem, *d};   // the rounded value overshoots by (d - r) / d
        }
        if (left.numerator.is_zero())
            left.negative = false;
        return ok(conversion{money::from_units(r.to(), q, amount.is_negative()), left});
    }

    /// \brief `amount` in `To`, rounded by `mode`: the typed form.
    /// \tparam To     The target asset type.
    /// \param amount An amount of the rate's source asset.
    /// \param r      The rate; its target must be `To`.
    /// \param mode   The rounding mode.
    /// \return The converted amount and the residual, or `asset_mismatch` or `overflow`.
    template <asset_like To, asset_like From>
    [[nodiscard]] constexpr result<std::pair<money_of<To>, residual>, money_errc> convert_to(const money_of<From>& amount,
                                                                                         const rate& r,
                                                                                         rounding mode) noexcept
    {
        if (r.to() != asset_v<To>)
            return err(money_errc::asset_mismatch);
        auto c = convert(amount.to_money(), r, mode);
        if (!c)
            return err(*c.err());
        return ok(std::pair{money_of<To>::from_units(c->converted.magnitude(), c->converted.is_negative()), c->residual});
    }

} // namespace snicholls

#endif // snicholls_core_money_rate_hpp
