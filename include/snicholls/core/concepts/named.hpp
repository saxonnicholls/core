//
//  named.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::has_name`: anything that answers `name()` with a string view, a type or an object.
/// \ingroup core
///
/// **Overview.** The smallest capability: a thing that has a name. It is the concept half of a pair; the pure
/// virtual half is `snicholls::named_interface` (`interfaces/named.hpp`). The concept is written over an object
/// expression, `t.name()`, so it is modelled both by a compile-time type whose `name()` is a `static constexpr`
/// member function (`struct AUD { static constexpr std::string_view name() noexcept { ... } };`, called as
/// `AUD{}.name()`) and by any object behind the virtual interface. Larger capabilities (`asset_like`,
/// `chain_like`) are built from it.
///
/// \snippet concepts/named_demo.cpp named
///
/// **Relationships.** `interfaces/named.hpp` (the virtual twin and the adapters), `interfaces/named_conformance.hpp`
/// (the kit), `concepts/asset.hpp`, `concepts/chain.hpp`. Group: \ref core.
///
/// **Implementations.** Every currency, native asset and chain type; `named_interface` and its adapter.
///
/// **Guarantees.** A concept: no code, no cost.
///
/// **Errors.** None.
///
/// **Performance.** Not applicable.
///
/// **Tests and demos.** `tests/concepts/named_test.cpp`, `demos/concepts/named_demo.cpp`.
///
/// **Dependencies.** `<concepts>`, `<string_view>`.
///
/// **Known issues.** None.
///
/// **Status.** New in 0.22.0.

#ifndef snicholls_core_concepts_named_hpp
#define snicholls_core_concepts_named_hpp

#include <concepts>
#include <string_view>

namespace snicholls {

    /// \brief A type whose objects answer `name()` with something that converts to `std::string_view`.
    template <typename T>
    concept has_name = requires(const T& t) {
        { t.name() } -> std::convertible_to<std::string_view>;
    };

} // namespace snicholls

#endif // snicholls_core_concepts_named_hpp
