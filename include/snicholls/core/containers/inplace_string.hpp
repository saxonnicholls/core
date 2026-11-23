//
//  inplace_string.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::inplace_string<N>`: a string of at most `N` characters stored in place, trivially copyable and
/// structural, refused rather than truncated when too long.
/// \ingroup core
///
/// **Overview.** Identifiers that cross a wire or sit in a record have a bounded length set by a contract: an
/// order id, a contract name, a venue symbol. `inplace_string<N>` holds up to `N` characters inside itself, so a
/// record that holds one is trivially copyable, can be written as bytes and compared without following a
/// pointer, and is a structural type, usable as a template argument. Run-time text becomes one only through
/// `make`, which returns `result` and refuses a text longer than `N` with `inplace_string_errc::too_long`: an identifier cut
/// short is a different identifier, so it is never truncated. A string literal known to fit is checked by the
/// compiler (`inplace_string<8>{"ETH-PERP"}`, a compile error when it does not fit).
///
/// \snippet containers/inplace_string_demo.cpp inplace_string
///
/// **Relationships.** `containers/inplace_vector.hpp` (the same idea for elements), `fixed_string.hpp` (a string
/// whose length is part of its type, for template arguments of exactly one text), `hash/fnv.hpp` (the hash).
/// Group: \ref core.
///
/// **Implementations.** One, for `char`.
///
/// **Guarantees.**
/// - constexpr and `noexcept` throughout; no allocation, ever.
/// - Layout: `N` characters then the length, in the smallest unsigned type that holds `N` (one byte up to 255):
///   `sizeof(inplace_string<255>) == 256`, `sizeof(inplace_string<64>) == 65`; alignment 1 up to 255.
/// - Trivially copyable, standard layout and structural. The characters past the length are always zero, so two
///   equal strings have equal members (and, up to 255, no padding: they are equal byte for byte), and the
///   member-wise equivalence of template arguments agrees with `==`.
/// - `==` and `<=>` compare the text (as `std::string_view` does), also across capacities and against a
///   `std::string_view`; `hash()` is FNV-1a 64 of the text, the same on every machine, and `std::hash` is
///   specialised.
///
/// **Errors.** `make` returns `inplace_string_errc::too_long` when the text has more than `N` characters.
///
/// **Performance.** `benchmarks/containers/inplace_string_benchmark.cpp`: 4,096 contract ids of 8 to 40 characters,
/// Release, Xeon W-3245, 2026-11-23. `inplace_string<64>::make`: 50.9 us against 42.4 us for the same check and
/// copy written by hand with Apple clang 17 (macOS), 22.5 us against 23.8 us with GCC 13 (Ubuntu 24.04 in Docker).
/// `==` of neighbours: 3.95 us against 3.96 us (Apple clang).
///
/// **Tests and demos.** `tests/containers/inplace_string_test.cpp`, `demos/containers/inplace_string_demo.cpp`.
///
/// **Dependencies.** `<algorithm>`, `<compare>`, `<concepts>`, `<cstddef>`, `<cstdint>`, `<functional>`, `<string_view>`,
/// `<type_traits>`, `<utility>`, `hash/fnv.hpp`, `result.hpp`.
///
/// **Known issues.** No `c_str()`: the characters are not followed by a terminator (that byte is the length's).
/// Text only, `char`; for bytes use `containers::inplace_vector<std::byte, N>`.
///
/// **Status.** New in 0.23.0.

#ifndef snicholls_core_containers_inplace_string_hpp
#define snicholls_core_containers_inplace_string_hpp

#include <snicholls/core/hash/fnv.hpp>
#include <snicholls/core/result.hpp>

#include <algorithm>
#include <compare>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string_view>
#include <type_traits>
#include <utility>

namespace snicholls {

    /// \brief Why an `inplace_string` was refused.
    /// \ingroup core
    enum class inplace_string_errc : std::uint8_t {
        too_long = 1,   ///< the text has more characters than the capacity
    };

    namespace detail {
        struct inplace_string_fits {};

        template <std::size_t N>
        using inplace_string_size_t =
            std::conditional_t<(N <= 0xFF), std::uint8_t, std::conditional_t<(N <= 0xFFFF), std::uint16_t, std::uint32_t>>;
    } // namespace detail

    /// \brief A string of at most `N` characters stored in place.
    /// \ingroup core
    /// \tparam N The capacity, at least 1.
    template <std::size_t N>
    struct inplace_string {
        static_assert(N >= 1 && N <= 0xFFFFFFFFu, "snicholls::inplace_string: the capacity must be 1 to 2^32 - 1");

        /// \brief The type of the length.
        using size_type = detail::inplace_string_size_t<N>;

        /// \brief The characters, zero past the length. Public for structural (template argument) use; read-only.
        char chars_[N]{};
        /// \brief The length. Public for structural use; read-only.
        size_type size_{};

        /// \brief The empty string.
        constexpr inplace_string() noexcept = default;

        /// \brief A string literal that fits, checked by the compiler.
        /// \tparam M The literal's size with its terminator, at most `N + 1`.
        /// \param literal The literal.
        template <std::size_t M>
            requires(M <= N + 1)
        consteval inplace_string(const char (&literal)[M]) noexcept   // NOLINT(google-explicit-constructor)
        {
            for (std::size_t i = 0; i + 1 < M; ++i)
                chars_[i] = literal[i];
            size_ = static_cast<size_type>(M - 1);
        }

        /// \cond
        // make's constructor: the length is already checked
        constexpr inplace_string(std::string_view text, detail::inplace_string_fits) noexcept
            : size_(static_cast<size_type>(text.size()))
        {
            std::copy_n(text.data(), text.size(), chars_);   // memmove at run time
        }
        /// \endcond

        /// \brief A copy of `text`, or `too_long` when it has more than `N` characters.
        /// \param text The text.
        /// \return The string, or the error.
        [[nodiscard]] static constexpr result<inplace_string, inplace_string_errc> make(std::string_view text) noexcept
        {
            if (text.size() > N)
                return err(inplace_string_errc::too_long);
            // constructed in place in the result, so the characters are written once
            return result<inplace_string, inplace_string_errc>(std::in_place, text, detail::inplace_string_fits{});
        }

        /// \brief The text.
        /// \return A view of the characters, valid while this string lives and is unchanged.
        [[nodiscard]] constexpr std::string_view view() const noexcept { return {chars_, size_}; }
        /// \brief The characters.
        /// \return A pointer to the first; not followed by a terminator.
        [[nodiscard]] constexpr const char* data() const noexcept { return chars_; }
        /// \brief The number of characters.
        /// \return It.
        [[nodiscard]] constexpr std::size_t size() const noexcept { return size_; }
        /// \brief Whether the string is empty.
        /// \return `true` when it is.
        [[nodiscard]] constexpr bool empty() const noexcept { return size_ == 0; }
        /// \brief The capacity.
        /// \return `N`.
        [[nodiscard]] static constexpr std::size_t capacity() noexcept { return N; }
        /// \brief FNV-1a 64 of the text: the same on every machine.
        /// \return The hash.
        [[nodiscard]] constexpr std::uint64_t hash() const noexcept { return fnv1a_64(view()); }

        /// \brief Equality of the text, across capacities.
        /// \tparam M The other capacity.
        /// \param a The left operand.
        /// \param b The right operand.
        /// \return Whether the texts are equal.
        template <std::size_t M>
        friend constexpr bool operator==(const inplace_string& a, const inplace_string<M>& b) noexcept
        {
            return a.view() == b.view();
        }
        /// \brief Order of the text, across capacities.
        /// \tparam M The other capacity.
        /// \param a The left operand.
        /// \param b The right operand.
        /// \return The order.
        template <std::size_t M>
        friend constexpr std::strong_ordering operator<=>(const inplace_string& a, const inplace_string<M>& b) noexcept
        {
            return a.view() <=> b.view();
        }
        /// \brief Equality with a view.
        /// \param a The string.
        /// \param b The view.
        /// \return Whether the texts are equal.
        friend constexpr bool operator==(const inplace_string& a, std::string_view b) noexcept { return a.view() == b; }
        /// \brief Order against a view.
        /// \param a The string.
        /// \param b The view.
        /// \return The order.
        friend constexpr std::strong_ordering operator<=>(const inplace_string& a, std::string_view b) noexcept
        {
            return a.view() <=> b;
        }
    };

} // namespace snicholls

/// \cond
template <std::size_t N>
struct std::hash<snicholls::inplace_string<N>> {
    constexpr std::size_t operator()(const snicholls::inplace_string<N>& s) const noexcept { return static_cast<std::size_t>(s.hash()); }
};
/// \endcond

#endif // snicholls_core_containers_inplace_string_hpp
