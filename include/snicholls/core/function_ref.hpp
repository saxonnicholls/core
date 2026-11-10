//
//  function_ref.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  The definition is unchanged; it is public here, in namespace snicholls, and the guard and documentation
//  are core's.
//

/// \file
/// \brief `snicholls::function_ref<R(Args...)>`: a non-owning, non-allocating reference to a callable.
/// \ingroup core
///
/// **Overview.** `function_ref<R(Args...)>` is the shape of C++26 `std::function_ref`:
/// a type-erased reference to any callable invocable as `R(Args...)`, built from two
/// pointers. It is what a function takes when it wants "something callable" without
/// becoming a template and without the allocation and copy of `std::function`:
///
/// \snippet function_ref_demo.cpp parameter
///
/// Use it for parameters - a visitor passed down a call, a callback invoked before
/// the call returns. Do not use it for storage: it refers to the callable it was
/// built from and does not keep it alive.
///
/// **Relationships.** Replaces a `template <typename F>` parameter where code size or
/// compile time matters, and `std::function` where the callable does not need to be
/// owned. `snicholls::overloaded` builds a callable it can refer to.
///
/// **Guarantees.** Two pointers, trivially copyable; copying a `function_ref` copies
/// the reference, never the callable, so every copy calls the same object and sees
/// the same state. Construction is `noexcept` and allocates nothing. The call is one
/// indirect call, which throws nothing of its own and lets whatever the callable
/// throws propagate. A result convertible to `R` is
/// converted; with `R = void` any result is discarded.
///
/// **Errors.** None of its own. A callable not invocable as `R(Args...)` does not
/// convert (the constructor is constrained), so the mistake is a compile error.
///
/// **Performance.** Not benchmarked.
///
/// **Tests and demos.** `tests/function_ref_test.cpp`, `demos/function_ref_demo.cpp`.
///
/// **Dependencies.** `<memory>` (for `std::addressof`), `<type_traits>`, `<utility>`.
///
/// **Known issues.** The referenced callable must outlive the `function_ref`. Built
/// from a temporary (a lambda written in the argument list) it is valid for that
/// call only, which is exactly the parameter case; stored past the full expression it
/// dangles. A function pointer is referred to by the address of the pointer object,
/// so the same applies to a pointer temporary. A plain function name (a function
/// type, not a pointer) is not accepted.
///
/// **Status.** Stable.

#ifndef snicholls_core_function_ref_hpp
#define snicholls_core_function_ref_hpp

#include <memory>
#include <type_traits>
#include <utility>

namespace snicholls {

/// \brief A non-owning reference to a callable; declared for every signature, defined for `R(Args...)`.
/// \tparam Signature A function type `R(Args...)`.
template <typename Signature>
class function_ref;

/// \brief A non-owning reference to any callable invocable as `R(Args...)`.
/// \ingroup core
/// \tparam R    The result type; any result convertible to it is accepted, and discarded when it is `void`.
/// \tparam Args The parameter types.
template <typename R, typename... Args>
class function_ref<R(Args...)>
{
public:
    /// \brief Refer to `f`. Implicit, like `std::function_ref`, so a callable passes straight to a parameter.
    /// \tparam F A callable invocable as `R(Args...)` through an lvalue, other than a `function_ref` or a function type.
    /// \param f The callable; it must outlive this `function_ref`.
    template <typename F>
        requires(!std::is_same_v<std::remove_cvref_t<F>, function_ref> && !std::is_function_v<std::remove_reference_t<F>> &&
                 std::is_invocable_r_v<R, F&, Args...>)
    constexpr function_ref(F&& f) noexcept   // NOLINT(google-explicit-constructor): implicit, like std::function_ref
        : object_(const_cast<void*>(static_cast<const void*>(std::addressof(f))))
        , call_(&invoke<std::remove_reference_t<F>>)
    {
    }

    /// \brief Call the referenced callable.
    /// \param args The arguments, forwarded.
    /// \return Its result, converted to `R`.
    /// \throws Whatever the callable throws.
    constexpr R operator()(Args... args) const { return call_(object_, std::forward<Args>(args)...); }

private:
    template <typename F>
    static constexpr R invoke(void* object, Args... args)
    {
        F& f = *static_cast<F*>(object);
        if constexpr (std::is_void_v<R>)
            static_cast<void>(f(std::forward<Args>(args)...));
        else
            return f(std::forward<Args>(args)...);
    }

    void* object_;
    R (*call_)(void*, Args...);
};

} // namespace snicholls

#endif /* snicholls_core_function_ref_hpp */
