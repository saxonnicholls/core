//
//  http_header.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::http_header`, a header field (HTTP, a webhook's), and `find_header`, the case-insensitive lookup.
/// \ingroup core
///
/// **Overview.** A header is a name and a value. `find_header` finds the first by name ignoring ASCII case, as RFC 9110
/// requires, with `ascii_iequals` (no locale). The HTTP handler contract (`concepts/http_handler.hpp`) carries headers
/// on its requests and responses; the payment provider contract (`concepts/payment_provider.hpp`) reads a webhook's
/// signature header with `find_header`. Core does no networking: these are the values, not a transport.
///
/// \snippet http_header_demo.cpp http_header
///
/// **Relationships.** `concepts/http_handler.hpp`, `concepts/payment_provider.hpp`. Group: \ref core.
///
/// **Implementations / variants.** One.
///
/// **Guarantees.** `ascii_iequals` and `find_header` are `constexpr`, `noexcept`, allocation-free and locale-free;
/// O(total length). `http_header` owns its strings: copying one may throw `std::bad_alloc`.
///
/// **Errors.** None; a missing header is `std::nullopt`.
///
/// **Performance.** Not benchmarked on its own; measured inside `benchmarks/interfaces/http_handler_benchmark.cpp`.
///
/// **Tests and demos.** `tests/http_header_test.cpp`, `demos/http_header_demo.cpp`.
///
/// **Dependencies.** `<cstddef>`, `<optional>`, `<span>`, `<string>`, `<string_view>`.
///
/// **Known issues.** Names are compared, not validated: a name with a space or a colon is found like any other.
///
/// **Status.** New in 0.23.0.

#ifndef snicholls_core_http_header_hpp
#define snicholls_core_http_header_hpp

#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <string_view>

namespace snicholls {

    /// \brief One header field.
    /// \ingroup core
    struct http_header {
        std::string name;   ///< the field name, any case
        std::string value;  ///< the field value, as received

        /// \brief Equality, case-sensitive on both members.
        friend bool operator==(const http_header&, const http_header&) = default;
    };

    /// \brief Whether two strings are equal ignoring ASCII case (no locale).
    /// \ingroup core
    /// \param a One string.
    /// \param b The other.
    /// \return `true` when equal ignoring ASCII case.
    [[nodiscard]] constexpr bool ascii_iequals(std::string_view a, std::string_view b) noexcept
    {
        if (a.size() != b.size())
            return false;
        constexpr auto lower = [](char c) noexcept { return (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c; };
        for (std::size_t i = 0; i < a.size(); ++i)
            if (lower(a[i]) != lower(b[i]))
                return false;
        return true;
    }

    /// \brief The value of the first header named `name`, ignoring case.
    /// \ingroup core
    /// \param headers The headers.
    /// \param name    The name to find.
    /// \return A view of the value (valid while `headers` is), or `std::nullopt`.
    [[nodiscard]] constexpr std::optional<std::string_view> find_header(std::span<const http_header> headers,
                                                                        std::string_view name) noexcept
    {
        for (const auto& h : headers)
            if (ascii_iequals(h.name, name))
                return std::string_view{h.value};
        return std::nullopt;
    }

} // namespace snicholls

#endif // snicholls_core_http_header_hpp
