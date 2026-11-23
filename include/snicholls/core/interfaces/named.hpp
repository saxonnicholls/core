//
//  named.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::named_interface`: the pure virtual twin of the `has_name` concept, and the adapters between them.
/// \ingroup core
///
/// **Overview.** `has_name` (`concepts/named.hpp`) is for static dispatch; `named_interface` is the same operation
/// behind a virtual call, for code that chooses at run time (a list of assets of different types, a plug-in).
/// `named_model<T>` adapts any `has_name` type to the interface, and the interface models the concept, so the two
/// forms are interchangeable.
///
/// \snippet interfaces/named_demo.cpp interface
///
/// **Relationships.** `concepts/named.hpp`, `interfaces/named_conformance.hpp`, `interfaces/asset.hpp`. Group:
/// \ref core.
///
/// **Implementations.** `named_model<T>` for every `has_name` T.
///
/// **Guarantees.** `name()` is `noexcept`; the adapter holds its `T` by value and is usable in constant
/// expressions when `T` is.
///
/// **Errors.** None.
///
/// **Performance.** One virtual call; not benchmarked.
///
/// **Tests and demos.** `tests/interfaces/named_test.cpp`, `demos/interfaces/named_demo.cpp`.
///
/// **Dependencies.** `<string_view>`, `concepts/named.hpp`.
///
/// **Known issues.** None.
///
/// **Status.** New in 0.22.0.

#ifndef snicholls_core_interfaces_named_hpp
#define snicholls_core_interfaces_named_hpp

#include <snicholls/core/concepts/named.hpp>

#include <string_view>

namespace snicholls {

    /// \brief A thing with a name, chosen at run time.
    class named_interface {
    public:
        /// \brief Virtual, so an implementation can be deleted through the interface.
        constexpr virtual ~named_interface() = default;
        /// \brief The name.
        /// \return A view valid while the object lives.
        [[nodiscard]] constexpr virtual std::string_view name() const noexcept = 0;
    };

    /// \brief Any `has_name` type behind `named_interface`.
    /// \tparam T The type, held by value.
    template <has_name T>
    class named_model final : public named_interface {
    public:
        /// \brief Adapt a default-constructed `T` (a compile-time type with static members).
        constexpr named_model() = default;
        /// \brief Adapt this value.
        /// \param t The value.
        constexpr explicit named_model(T t) : t_(t) {}
        /// \brief The adapted value's name.
        /// \return Its name.
        [[nodiscard]] constexpr std::string_view name() const noexcept override { return t_.name(); }

    private:
        T t_{};
    };

    static_assert(has_name<named_interface>, "snicholls::named_interface must model snicholls::named");

} // namespace snicholls

#endif // snicholls_core_interfaces_named_hpp
