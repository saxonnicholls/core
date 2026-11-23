//
//  named_conformance.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::check_named`: the conformance kit of the `has_name` capability.
/// \ingroup core
///
/// **Overview.** Drives any `has_name` implementation, a type or an object behind `named_interface`, through the
/// contract and says which check failed: the name is not empty and is the same on every call. No test framework is
/// included; a test asserts that the result is empty.
///
/// \snippet interfaces/named_conformance_demo.cpp conformance
///
/// **Relationships.** `concepts/named.hpp`, `interfaces/named.hpp`. Group: \ref core.
///
/// **Implementations.** Not applicable.
///
/// **Guarantees.** constexpr, `noexcept`.
///
/// **Errors.** The returned message names the failed check.
///
/// **Performance.** Not applicable.
///
/// **Tests and demos.** `tests/interfaces/named_conformance_test.cpp`, `demos/interfaces/named_conformance_demo.cpp`.
///
/// **Dependencies.** `<optional>`, `<string_view>`, `concepts/named.hpp`.
///
/// **Known issues.** None.
///
/// **Status.** New in 0.22.0.

#ifndef snicholls_core_interfaces_named_conformance_hpp
#define snicholls_core_interfaces_named_conformance_hpp

#include <snicholls/core/concepts/named.hpp>

#include <optional>
#include <string_view>

namespace snicholls {

    /// \brief Check `t` against the `has_name` contract.
    /// \param t The implementation.
    /// \return `std::nullopt` if it conforms, or the failed check.
    template <has_name T>
    [[nodiscard]] constexpr std::optional<std::string_view> check_named(const T& t) noexcept
    {
        const std::string_view first = t.name();
        if (first.empty())
            return "named: name() is empty";
        if (std::string_view{t.name()} != first)
            return "named: name() changes between calls";
        return std::nullopt;
    }

} // namespace snicholls

#endif // snicholls_core_interfaces_named_conformance_hpp
