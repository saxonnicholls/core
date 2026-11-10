//
//  basic_type_registry.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Moved from base-encode-decode (BaseEncodeDecode/utils/basic_type_registry.hpp
//  at c3a2ed0). The definitions are unchanged but for the names, which follow
//  core's lower_snake convention: BasicTypeRegistry -> basic_type_registry,
//  Named -> named, nameOf -> name_of. The guard, include path and
//  documentation are core's.
//

/// \file
/// \brief `snicholls::basic_type_registry` and `snicholls::named`: construct-and-dispatch by name, with no codec.
/// \ingroup core
///
/// \overview
/// The codec-independent core of the type registry: construct-and-dispatch by name with **no** `std::variant`,
/// **no** common base class and **no** switch statement. Each type is bound to a compile-time string; a run-time
/// name is matched against those names with a short-circuiting fold expression, and on the first match the
/// payload is deserialised with a caller-supplied deserialiser and handed to a generic callable. It depends only
/// on `fixed_string.hpp` - no encoder, no SIMD - so it can be used with any format (JSON, protobuf, your own
/// bytes).
///
/// \relations
/// - `snicholls::basic_fixed_string` names the bindings; `snicholls::overloaded` makes a per-type handler.
/// - base-encode-decode's `TypeRegistry` (`utils/type_registry.hpp`) extends this one with the Base64 defaults
///   (`Registry::serialize(obj)`, `Registry::dispatch(name, base64, handler)`); that part stays there because it
///   is a codec.
/// - Module: \ref core.
///
/// \impls
/// None: a single implementation. The set of bindings is closed and fixed at compile time.
///
/// \contracts
/// - Every member is static; the registry is never instantiated. `count`, `names`, `contains` and `name_of` are
///   compile-time (`contains` also works at run time).
/// - `dispatch` tries the bindings in declaration order and stops at the first name that matches: at most one
///   deserialisation and one handler call. On no match neither is called.
/// - Lookup is a linear chain of string comparisons, O(bindings x name length).
/// - Thread safety: no state, so as safe as the handler and deserialiser passed in.
///
/// \errors
/// `name_of<T>()` for an unregistered `T` is a compile error. `dispatch` returns `false` for an unknown name and
/// lets whatever the deserialiser or handler throws propagate.
///
/// \examples
/// \snippet basic_type_registry_demo.cpp dispatch
///
/// \perf
/// Not benchmarked.
///
/// \testing
/// - Tests: tests/basic_type_registry_test.cpp.
/// - Demo: demos/basic_type_registry_demo.cpp.
///
/// \deps
/// `<array>`, `<cstddef>`, `<string_view>`, `<type_traits>` and `snicholls/core/fixed_string.hpp`.
///
/// \gotchas
/// - `snicholls::named` is a name-to-type binding, not a concept for types that declare their own `name()`.
/// - Two bindings with the same name are not rejected: the first one wins in `dispatch` and `name_of`.
///
/// \status
/// Stable.

#ifndef snicholls_core_basic_type_registry_hpp
#define snicholls_core_basic_type_registry_hpp

#include <array>
#include <cstddef>
#include <string_view>
#include <type_traits>

#include <snicholls/core/fixed_string.hpp>

namespace snicholls {

    /// \brief Binds a compile-time name to a type: `named<"object1", Object1>`.
    /// \ingroup core
    /// \tparam Name The name, a string literal.
    /// \tparam T    The bound type.
    template<basic_fixed_string Name, typename T>
    struct named {
        using type = T;                                            ///< The bound type.
        static constexpr std::string_view name = Name.view();      ///< The bound name.
    };

    /// \brief A closed set of (name, type) bindings, all known at compile time.
    /// \ingroup core
    /// \tparam Bindings `named<"name", T>` bindings.
    template<typename... Bindings>
    struct basic_type_registry {
        /// \brief The number of bindings.
        static constexpr size_t count = sizeof...(Bindings);
        /// \brief The registered names, in declaration order.
        static constexpr std::array<std::string_view, count> names{ Bindings::name... };

        /// \brief Is `name` one of the registered names? (run time or compile time)
        /// \param name The name to look up.
        /// \return Whether a binding has exactly that name.
        static constexpr bool contains(std::string_view name) {
            return ((name == Bindings::name) || ...);
        }

        /// \brief The registered name for a type.
        ///
        /// Referencing an unregistered type is a compile error (the consteval throw cannot be a constant expression).
        /// \tparam T A registered type.
        /// \return Its name.
        template<typename T>
        static consteval std::string_view name_of() {
            std::string_view result{};
            const bool found = ((std::is_same_v<typename Bindings::type, T>
                                     ? (result = Bindings::name, true) : false) || ...);
            if (!found) {
                throw "snicholls::basic_type_registry::name_of<T>: type is not registered";
            }
            return result;
        }

        /// \brief Match `name`, deserialise the bound type from `serialized`, and invoke `handler(object)`.
        ///
        /// The fold short-circuits at the first match - no switch, no variant, no base class.
        /// \param name        The run-time type name.
        /// \param serialized  The payload, passed to the deserialiser unchanged.
        /// \param handler     Called once with the deserialised object; usually generic or an `overloaded` set.
        /// \param deserialize Maps `(std::type_identity<T>, std::string_view)` to a `T`.
        /// \return `true` iff a registered type matched.
        /// \throws Whatever `deserialize` or `handler` throws.
        template<typename Handler, typename Deserialize>
        static bool dispatch(std::string_view name, std::string_view serialized,
                             Handler&& handler, Deserialize&& deserialize) {
            return (try_one<Bindings>(name, serialized, handler, deserialize) || ...);
        }

    private:
        template<typename Binding, typename Handler, typename Deserialize>
        static bool try_one(std::string_view name, std::string_view serialized,
                           Handler& handler, Deserialize& deserialize) {
            if (name != Binding::name) {
                return false;
            }
            handler(deserialize(std::type_identity<typename Binding::type>{}, serialized));
            return true;
        }
    };
}

#endif /* snicholls_core_basic_type_registry_hpp */
