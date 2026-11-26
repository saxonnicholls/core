//
//  ratio.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `finance::ratio`: an exact ratio of two 64-bit integers, normalised; the one rational of `finance/`, behind
/// `year_fraction`, split and rights ratios, cost apportionments, repo rates and exact returns.
/// \ingroup core
///
/// **Overview.** Finance meets many exact fractions: a day count's 91 / 360, a 3-for-2 split, a 1-for-5 rights issue, a
/// demerger's 13.24% of the cost base, a repo at 4.35% a year, a period's return of 1,250 / 100,000. `ratio` keeps
/// each one as `numerator / denominator`, always in lowest terms with a positive denominator, so equal values have
/// equal members, `==` is member-wise and two sources' ratios compare exactly. It is the only rational in
/// `finance/`: `finance::year_fraction` is this type (`finance/year_fraction.hpp`), so a day-count fraction, a split
/// ratio and a return combine without conversion.
///
/// - `ratio{n, d}` normalises; its preconditions (`d != 0`, neither part `INT64_MIN`) are `assert`ed. `ratio::make`
///   checks them and returns `ratio_errc` instead, for data.
/// - `ratio::parse` reads `"3/2"`, `"7"`, `"-1.25"` and `"13.24%"` exactly.
/// - `add`, `subtract` and `multiply` are exact and checked: `std::nullopt` when the result does not fit in 64 bits.
///   `multiply` cancels across before it multiplies, so a chain of returns or ratios stays small as long as it can.
/// - `<=>` is exact for every pair: cross-multiplied while the products fit in 64 bits, and compared by continued
///   fractions (only divisions) when they would not.
/// - `to_double()` is for display and for code that wants a `double` anyway.
///
/// It is a pure rational, tied to no asset: `finance/money_ratio.hpp` multiplies and divides `money` by it exactly (a
/// fund's units are `amount / NAV per unit`), to a ratio or to money at a stated scale and rounding mode.
///
/// Use it where an exact fraction is the answer; use `snicholls::rate` (`money/rate.hpp`) to convert an amount from
/// one asset to another, and `money` for amounts.
///
/// \snippet finance/ratio_demo.cpp ratio
///
/// **Relationships.** `finance/year_fraction.hpp` (the alias), `finance/money_ratio.hpp` (with `money`), `finance/day_count.hpp`, `finance/returns.hpp`,
/// `concepts/security_provider.hpp` (split ratios, apportionments, repo rates), `checked.hpp`, `result.hpp`. Group:
/// \ref core.
///
/// **Implementations / variants.** One, over `std::int64_t`. The securities library's entitlement arithmetic works in
/// `uint256` and widens a `ratio` into it.
///
/// **Guarantees.** `constexpr` and `noexcept` throughout; trivially copyable; 16 bytes; always normalised
/// (`gcd(numerator, denominator) == 1`, `denominator > 0`, zero is `0/1`). `INT64_MIN` is never a member, so negation
/// never overflows.
///
/// **Errors.** `ratio::make` and `ratio::parse` return `ratio_errc`: `zero_denominator`, `out_of_range` (a part that is
/// `INT64_MIN` or does not fit), `malformed` (text that is not a ratio). The arithmetic returns `std::nullopt` on
/// overflow.
///
/// **Performance.** One `std::gcd` on construction. `benchmarks/finance/ratio_benchmark.cpp`, 4,096 pairs of day-count
/// fractions, Release, Apple clang 17, macOS x86-64, 2026-11-26, CPU time on a loaded machine (load 200, so a record,
/// not a quote): `<=>` 10.0 us against 7.4 us for the bare 64-bit cross-multiplication (the overflow checks);
/// `multiply` 648 us against 358 us for the same cancel-then-multiply unchecked and unnormalised.
///
/// **Tests and demos.** `tests/finance/ratio_test.cpp`, `demos/finance/ratio_demo.cpp`.
///
/// **Dependencies.** `<cassert>`, `<compare>`, `<cstdint>`, `<limits>`, `<numeric>`, `<optional>`, `<string_view>`,
/// `checked.hpp` (which brings `<charconv>` and `<algorithm>`), `result.hpp`.
///
/// **Known issues.** 64 bits: a long chain of unrelated ratios overflows (the arithmetic says so with `std::nullopt`),
/// and `finance/returns.hpp` falls back to `double` there. Not a general arithmetic type: no operators but negation.
///
/// **Status.** New in 0.27.0. `year_fraction` became an alias of it, source-compatible.

#ifndef snicholls_core_finance_ratio_hpp
#define snicholls_core_finance_ratio_hpp

#include <snicholls/core/checked.hpp>
#include <snicholls/core/result.hpp>

#include <cassert>
#include <compare>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <string_view>

namespace snicholls::finance {

    /// \brief Why a ratio could not be made.
    /// \ingroup core
    enum class ratio_errc : std::uint8_t {
        zero_denominator = 1,  ///< `n / 0`
        out_of_range = 2,      ///< a part is `INT64_MIN`, or the value does not fit in 64 bits
        malformed = 3,         ///< text that is not `n`, `n/d`, a decimal or a percentage
    };

    /// \brief An exact ratio `numerator / denominator`, normalised: lowest terms, positive denominator.
    /// \ingroup core
    class ratio {
    public:
        /// \brief Zero.
        constexpr ratio() noexcept = default;

        /// \brief `numerator / denominator`, reduced to lowest terms with a positive denominator.
        /// \param numerator   The numerator, not `INT64_MIN`.
        /// \param denominator The denominator, neither zero nor `INT64_MIN`.
        /// \pre `denominator != 0`, and neither part is `INT64_MIN` (`assert`ed; `make` checks them).
        constexpr ratio(std::int64_t numerator, std::int64_t denominator) noexcept
        {
            assert(denominator != 0 && numerator != min_ && denominator != min_);
            if (denominator < 0) {
                numerator = -numerator;
                denominator = -denominator;
            }
            const std::int64_t g = std::gcd(numerator, denominator);
            numerator_ = g == 0 ? 0 : numerator / g;
            denominator_ = g == 0 ? 1 : denominator / g;
        }

        /// \brief `numerator / denominator`, checked.
        /// \param numerator   The numerator.
        /// \param denominator The denominator.
        /// \return The ratio, or `zero_denominator` or `out_of_range`.
        [[nodiscard]] static constexpr result<ratio, ratio_errc> make(std::int64_t numerator, std::int64_t denominator) noexcept
        {
            if (denominator == 0)
                return err(ratio_errc::zero_denominator);
            if (numerator == min_ || denominator == min_)
                return err(ratio_errc::out_of_range);
            return ok(ratio{numerator, denominator});
        }

        /// \brief Reads `"7"`, `"3/2"`, `"-1.25"` or `"13.24%"` exactly: an optional `-`, digits, then either `/` and
        /// digits, or an optional `.` and digits and an optional `%`.
        /// \param text The text.
        /// \return The ratio, or `malformed`, `zero_denominator` or `out_of_range`.
        [[nodiscard]] static constexpr result<ratio, ratio_errc> parse(std::string_view text) noexcept
        {
            const bool negative = !text.empty() && text.front() == '-';
            if (negative)
                text.remove_prefix(1);
            std::int64_t n = 0, d = 1;
            std::size_t i = 0;
            const auto digits = [&](std::int64_t& v, bool scale) -> result<std::size_t, ratio_errc> {
                const std::size_t start = i;
                for (; i < text.size() && text[i] >= '0' && text[i] <= '9'; ++i) {
                    const auto t = checked_mul<std::int64_t>(v, 10);
                    const auto u = t ? checked_add<std::int64_t>(*t, text[i] - '0') : t;
                    const auto s = scale ? checked_mul<std::int64_t>(d, 10) : std::optional<std::int64_t>{d};
                    if (!u || !s)
                        return err(ratio_errc::out_of_range);
                    v = *u;
                    d = *s;
                }
                return ok(i - start);
            };
            const auto whole = digits(n, false);
            if (!whole)
                return err(whole.unwrap_err_unchecked());
            if (*whole == 0)
                return err(ratio_errc::malformed);
            if (i < text.size() && text[i] == '/') {
                ++i;
                std::int64_t den = 0;
                const auto below = digits(den, false);
                if (!below)
                    return err(below.unwrap_err_unchecked());
                if (*below == 0 || i != text.size())
                    return err(ratio_errc::malformed);
                return make(negative ? -n : n, den);
            }
            if (i < text.size() && text[i] == '.') {
                ++i;
                const auto fraction = digits(n, true);
                if (!fraction)
                    return err(fraction.unwrap_err_unchecked());
                if (*fraction == 0)
                    return err(ratio_errc::malformed);
            }
            if (i < text.size() && text[i] == '%') {
                ++i;
                const auto hundredths = checked_mul<std::int64_t>(d, 100);
                if (!hundredths)
                    return err(ratio_errc::out_of_range);
                d = *hundredths;
            }
            if (i != text.size())
                return err(ratio_errc::malformed);
            return make(negative ? -n : n, d);
        }

        /// \brief The numerator, in lowest terms; negative when the ratio is.
        /// \return It.
        [[nodiscard]] constexpr std::int64_t numerator() const noexcept { return numerator_; }
        /// \brief The denominator, in lowest terms; always positive.
        /// \return It.
        [[nodiscard]] constexpr std::int64_t denominator() const noexcept { return denominator_; }
        /// \brief Whether it is zero.
        /// \return `numerator() == 0`.
        [[nodiscard]] constexpr bool is_zero() const noexcept { return numerator_ == 0; }
        /// \brief Whether it is below zero.
        /// \return `numerator() < 0`.
        [[nodiscard]] constexpr bool is_negative() const noexcept { return numerator_ < 0; }
        /// \brief The nearest `double`, for display: each part converted, then divided.
        /// \return `numerator / denominator`.
        [[nodiscard]] constexpr double to_double() const noexcept
        {
            return static_cast<double>(numerator_) / static_cast<double>(denominator_);
        }
        /// \brief `denominator / numerator`.
        /// \return The inverse, or `std::nullopt` for zero.
        [[nodiscard]] constexpr std::optional<ratio> inverse() const noexcept
        {
            if (numerator_ == 0)
                return std::nullopt;
            return ratio{denominator_, numerator_};
        }

        /// \brief The negation; never overflows.
        /// \param r The ratio.
        /// \return `-r`.
        [[nodiscard]] friend constexpr ratio operator-(const ratio& r) noexcept { return ratio{-r.numerator_, r.denominator_}; }

        /// \brief Equality: the same number.
        /// \return Whether the two are equal.
        friend constexpr bool operator==(const ratio&, const ratio&) noexcept = default;
        /// \brief Order by value, exact for every pair.
        /// \param a The left operand.
        /// \param b The right operand.
        /// \return The order of `a` and `b`.
        friend constexpr std::strong_ordering operator<=>(const ratio& a, const ratio& b) noexcept
        {
            // Cross-multiplied while the products fit (one multiply each, the usual case); otherwise
            // compare a/b with c/d (b, d > 0) by their continued fractions: whole parts first, then the remainders'
            // inverses, with the order flipping at each inversion. Every step divides; nothing is multiplied.
            const auto left = checked_mul(a.numerator_, b.denominator_), right = checked_mul(b.numerator_, a.denominator_);
            if (left && right)
                return *left <=> *right;
            std::int64_t p = a.numerator_, q = a.denominator_, r = b.numerator_, s = b.denominator_;
            bool flipped = false;
            for (;;) {
                const std::int64_t x = floor_div(p, q), y = floor_div(r, s);
                const std::int64_t pr = p - x * q, rr = r - y * s;   // 0 <= remainder < denominator
                if (x != y || pr == 0 || rr == 0) {
                    const std::strong_ordering o = x != y ? x <=> y : pr <=> rr;   // a zero remainder is the smaller
                    return flipped ? 0 <=> o : o;
                }
                p = q;
                q = pr;
                r = s;
                s = rr;
                flipped = !flipped;
            }
        }

    private:
        static constexpr std::int64_t min_ = std::numeric_limits<std::int64_t>::min();
        static constexpr std::int64_t floor_div(std::int64_t p, std::int64_t q) noexcept
        {
            const std::int64_t d = p / q;
            return (p % q != 0 && p < 0) ? d - 1 : d;
        }

        std::int64_t numerator_ = 0;
        std::int64_t denominator_ = 1;
    };

    /// \brief `a * b`, exact: cancelled across first, so it fits whenever the reduced product does.
    /// \ingroup core
    /// \param a One.
    /// \param b The other.
    /// \return The product, or `std::nullopt` when it does not fit in 64 bits.
    [[nodiscard]] constexpr std::optional<ratio> multiply(const ratio& a, const ratio& b) noexcept
    {
        const std::int64_t g1 = std::gcd(a.numerator(), b.denominator()), g2 = std::gcd(b.numerator(), a.denominator());
        const auto n = checked_mul<std::int64_t>(a.numerator() / g1, b.numerator() / g2);
        const auto d = checked_mul<std::int64_t>(a.denominator() / g2, b.denominator() / g1);
        if (!n || !d)
            return std::nullopt;
        const auto r = ratio::make(*n, *d);
        return r ? std::optional<ratio>{*r} : std::nullopt;
    }

    /// \brief `a + b`, exact.
    /// \ingroup core
    /// \param a One.
    /// \param b The other.
    /// \return The sum, or `std::nullopt` when it does not fit in 64 bits.
    [[nodiscard]] constexpr std::optional<ratio> add(const ratio& a, const ratio& b) noexcept
    {
        const std::int64_t g = std::gcd(a.denominator(), b.denominator());
        const auto left = checked_mul<std::int64_t>(a.numerator(), b.denominator() / g);
        const auto right = checked_mul<std::int64_t>(b.numerator(), a.denominator() / g);
        const auto n = left && right ? checked_add<std::int64_t>(*left, *right) : std::nullopt;
        const auto d = checked_mul<std::int64_t>(a.denominator(), b.denominator() / g);
        if (!n || !d)
            return std::nullopt;
        const auto r = ratio::make(*n, *d);
        return r ? std::optional<ratio>{*r} : std::nullopt;
    }

    /// \brief `a - b`, exact.
    /// \ingroup core
    /// \param a One.
    /// \param b The other.
    /// \return The difference, or `std::nullopt` when it does not fit in 64 bits.
    [[nodiscard]] constexpr std::optional<ratio> subtract(const ratio& a, const ratio& b) noexcept { return add(a, -b); }

} // namespace snicholls::finance

#endif // snicholls_core_finance_ratio_hpp
