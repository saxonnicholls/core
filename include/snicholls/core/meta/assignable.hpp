//
//  assignable.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `non_narrowing_assignable<V, E>`: `V = E` compiles, does not narrow a number, and never puts a number in a string.
/// \ingroup core
///
/// **Overview.** `std::is_assignable` says yes to `int = 3.7` (it truncates) and to `std::string = 3`
/// (`basic_string::operator=(char)` turns the number into the character `'\x03'`). A setter, a parser
/// or a configuration loader wants neither. `non_narrowing_assignable` is assignability with the
/// brace-initialisation rules for arithmetic types, and with arithmetic into strings refused.
///
/// \snippet meta/assignable_demo.cpp assignable
/// \snippet meta/assignable_demo.cpp assignable_use
///
/// **Relationships.** `meta/concepts.hpp`; `reflect/struct_tuple.hpp` applies the same brace rules
/// to whole aggregates. Group: \ref core.
///
/// **Guarantees.** A concept: compile time only.
///
/// **Errors.** None.
///
/// **Performance.** Compile time only.
///
/// **Tests and demos.** `tests/meta/assignable_test.cpp`, `demos/meta/assignable_demo.cpp`.
///
/// **Dependencies.** `<string>`, `<type_traits>`, `<utility>`, `meta/concepts.hpp`.
///
/// **Known issues.** A character counts as arithmetic, so `std::string = 'x'` is refused too: write
/// `std::string(1, 'x')`. A constant expression that fits (`unsigned char = 3`) is still refused, because
/// the concept sees types, not values.
///
/// **Status.** Stable.

#ifndef snicholls_core_meta_assignable_hpp
#define snicholls_core_meta_assignable_hpp

#include <snicholls/core/meta/concepts.hpp>

#include <string>
#include <type_traits>
#include <utility>

namespace snicholls {

    /// \brief `V` can be assigned from `E` without narrowing (brace rules for arithmetic types), and a number is never
    /// assigned to a string.
    /// \ingroup core
    template <typename V, typename E>
    concept non_narrowing_assignable =
        std::is_assignable_v<V&, E> && !(specialisation_of<std::remove_cv_t<V>, std::basic_string> && std::is_arithmetic_v<std::remove_cvref_t<E>>) &&
        (!(std::is_arithmetic_v<V> && std::is_arithmetic_v<std::remove_cvref_t<E>>) || requires { V{std::declval<E>()}; });

} // namespace snicholls

#endif /* snicholls_core_meta_assignable_hpp */
