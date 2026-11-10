//
//  strong.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `strong<T, Tag>`: a value with its own type, so prices, quantities and notionals cannot be mixed.
/// \ingroup core
///
/// **Overview.** `using price = strong<double, struct price_tag>;` gives a type that holds a `double` and
/// is nothing else: it compares and adds with another `price`, scales by a plain number, and refuses a
/// `quantity` or a bare `double` where a `price` is expected. It costs nothing: the same size and the same
/// code as the `double`.
///
/// \snippet strong_demo.cpp strong
///
/// Multiplying two different quantities (price times quantity is a notional) is the caller's to state, with
/// a function: core does not guess units.
///
/// **Relationships.** `meta/assignable.hpp`; `acout.hpp` prints one (its `print` member); a `std::hash` is
/// provided. Group: \ref core.
///
/// **Guarantees.** `constexpr` and `noexcept` when `T`'s operations are; `sizeof(strong<T, Tag>) == sizeof(T)`;
/// trivially copyable when `T` is; construction is `explicit`.
///
/// **Errors.** None.
///
/// **Performance.** The same as `T`'s.
///
/// **Tests and demos.** `tests/strong_test.cpp`, `demos/strong_demo.cpp`.
///
/// **Dependencies.** `<compare>`, `<cstddef>`, `<functional>`, `<ostream>`, `<utility>`.
///
/// **Known issues.** No units algebra: a product of two strong types is a function the caller writes.
///
/// **Status.** New in 0.5.0.

#ifndef snicholls_core_strong_hpp
#define snicholls_core_strong_hpp

#include <compare>
#include <cstddef>
#include <functional>
#include <ostream>
#include <utility>

namespace snicholls {

    /// \brief A `T` with its own type, distinguished by `Tag`.
    /// \ingroup core
    /// \tparam T   The value type.
    /// \tparam Tag Any type; usually declared inline: `strong<double, struct price_tag>`.
    template <typename T, typename Tag>
    struct strong {
        T value{};   ///< The value.

        /// \brief Zero.
        constexpr strong() = default;
        /// \brief From a `T`; explicit, so a bare number never becomes a price by accident.
        /// \param v The value.
        constexpr explicit strong(T v) noexcept(std::is_nothrow_move_constructible_v<T>) : value(std::move(v)) {}

        /// \brief Compared as the values are.
        friend constexpr auto operator<=>(const strong&, const strong&) = default;

        /// \brief The sum of two of the same kind.
        /// \param a One.
        /// \param b The other.
        /// \return The sum.
        friend constexpr strong operator+(const strong& a, const strong& b) { return strong(a.value + b.value); }
        /// \brief The difference of two of the same kind.
        /// \param a One.
        /// \param b The other.
        /// \return The difference.
        friend constexpr strong operator-(const strong& a, const strong& b) { return strong(a.value - b.value); }
        /// \brief The negation.
        /// \param a The value.
        /// \return Its negation.
        friend constexpr strong operator-(const strong& a) { return strong(-a.value); }
        /// \brief Scaled by a plain number.
        /// \param a The value.
        /// \param k The factor.
        /// \return The product.
        friend constexpr strong operator*(const strong& a, const T& k) { return strong(a.value * k); }
        /// \brief Scaled by a plain number.
        /// \param k The factor.
        /// \param a The value.
        /// \return The product.
        friend constexpr strong operator*(const T& k, const strong& a) { return strong(k * a.value); }
        /// \brief Divided by a plain number.
        /// \param a The value.
        /// \param k The divisor.
        /// \return The quotient.
        friend constexpr strong operator/(const strong& a, const T& k) { return strong(a.value / k); }
        /// \brief The ratio of two of the same kind: a plain number.
        /// \param a One.
        /// \param b The other.
        /// \return `a / b`.
        friend constexpr T operator/(const strong& a, const strong& b) { return a.value / b.value; }
        /// \brief Add another of the same kind.
        /// \param b The other.
        /// \return `*this`.
        constexpr strong& operator+=(const strong& b) { value += b.value; return *this; }
        /// \brief Subtract another of the same kind.
        /// \param b The other.
        /// \return `*this`.
        constexpr strong& operator-=(const strong& b) { value -= b.value; return *this; }

        /// \brief The `snicholls::printable` member: the value, as it streams.
        /// \param os The stream.
        void print(std::ostream& os) const { os << value; }
    };

} // namespace snicholls

/// \cond SNICHOLLS_INTERNAL
template <typename T, typename Tag>
struct std::hash<snicholls::strong<T, Tag>> {
    std::size_t operator()(const snicholls::strong<T, Tag>& s) const noexcept(noexcept(std::hash<T>{}(s.value))) { return std::hash<T>{}(s.value); }
};
/// \endcond

#endif /* snicholls_core_strong_hpp */
