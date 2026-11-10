//
//  callable_traits.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::callable_traits`: the result, arguments and arity of a callable with one signature.
/// \ingroup core
///
/// **Overview.** Given a callable type, say what it takes and what it returns: for a function
/// type, a function pointer or reference, a pointer to member function with any `const`,
/// `volatile`, `&`, `&&` or `noexcept` qualification, a `std::function`, or a class with
/// exactly one non-template `operator()` (every non-generic lambda). The arguments come as a
/// `meta::type_list`, so the type-level algorithms apply to them:
///
/// \snippet callable_traits_demo.cpp traits
///
/// `single_signature_callable<F>` is the concept: true exactly when `callable_traits<F>` is
/// defined. A generic lambda or an overloaded `operator()` has no single signature, and
/// `invoke_with` uses this concept to say so with a message.
///
/// **Relationships.** `snicholls::invoke_with` reads parameter lists through it;
/// `snicholls::function_ref` and `overloaded` are what it is usually pointed at;
/// `meta::type_list` carries the arguments. Group: \ref core.
///
/// **Guarantees.** Compile time only. `callable_traits<F>` strips `const`, `volatile` and
/// references from `F` first, so the traits of a `const` lambda object are the lambda's.
/// `result_type` and `arguments` are the declared types, references and all; `arity` counts
/// the arguments. For a pointer to member function `class_type` is the class and `is_member`
/// is true; for a function object `is_member` is false and `class_type` is the object's type.
///
/// **Errors.** None. `callable_traits<F>` for a type without a single signature is an empty
/// struct, so naming its members is a compile error and the concept is false, never a hard
/// error inside a constraint.
///
/// **Performance.** Not applicable.
///
/// **Tests and demos.** `tests/meta/callable_traits_test.cpp`, `demos/meta/callable_traits_demo.cpp`.
///
/// **Dependencies.** `<cstddef>`, `<type_traits>` and `meta/type_list.hpp`.
///
/// **Known issues.** A C-variadic function (`int(const char*, ...)`) has no traits. A class
/// whose `operator()` is itself inherited through `using` from two bases is ambiguous and has
/// none either.
///
/// **Status.** New in 0.5.0.

#ifndef snicholls_core_meta_callable_traits_hpp
#define snicholls_core_meta_callable_traits_hpp

#include <snicholls/core/meta/type_list.hpp>

#include <cstddef>
#include <type_traits>

namespace snicholls {

    namespace detail {

        // What every defined callable_traits exposes.
        template <typename R, bool Noexcept, typename... Args>
        struct signature_traits {
            using result_type = R;
            using arguments = meta::type_list<Args...>;
            using signature = R(Args...);
            static constexpr std::size_t arity = sizeof...(Args);
            static constexpr bool is_noexcept = Noexcept;
            template <std::size_t I>
            using argument_t = meta::at_t<arguments, I>;
        };

    } // namespace detail

    /// \brief The result, arguments and arity of a callable with one signature; empty for any other type.
    /// \ingroup core
    ///
    /// Members when defined: `result_type`, `arguments` (a `meta::type_list`), `signature`
    /// (`R(Args...)`), `arity`, `is_noexcept`, `argument_t<I>`, `is_member`, and `class_type`
    /// for a pointer to member function or a function object. Defined for function types,
    /// pointers and references to functions, pointers to member functions with any
    /// combination of `const`, `volatile`, `&`, `&&` and `noexcept` (where `class_type` is
    /// the class and `is_member` is true), and classes with one non-template `operator()`.
    /// \tparam F The callable type; cv and references are stripped.
    template <typename F>
    struct callable_traits {};

    /// \brief A cv-qualified or reference type: the traits of the unqualified type.
    /// \tparam F The type, with its qualification.
    template <typename F>
        requires(!std::is_same_v<F, std::remove_cvref_t<F>>)
    struct callable_traits<F> : callable_traits<std::remove_cvref_t<F>> {};

    /// \brief A function type.
    /// \tparam R    The result type.
    /// \tparam Args The parameter types.
    template <typename R, typename... Args>
    struct callable_traits<R(Args...)> : detail::signature_traits<R, false, Args...> {
        /// \brief Not a pointer to member function.
        static constexpr bool is_member = false;
    };

    /// \brief A `noexcept` function type.
    /// \tparam R    The result type.
    /// \tparam Args The parameter types.
    template <typename R, typename... Args>
    struct callable_traits<R(Args...) noexcept> : detail::signature_traits<R, true, Args...> {
        /// \brief Not a pointer to member function.
        static constexpr bool is_member = false;
    };

    /// \brief A pointer to function: the function type's traits.
    /// \tparam R    The result type.
    /// \tparam Args The parameter types.
    template <typename R, typename... Args>
    struct callable_traits<R (*)(Args...)> : callable_traits<R(Args...)> {};

    /// \brief A pointer to `noexcept` function: the function type's traits.
    /// \tparam R    The result type.
    /// \tparam Args The parameter types.
    template <typename R, typename... Args>
    struct callable_traits<R (*)(Args...) noexcept> : callable_traits<R(Args...) noexcept> {};

#ifndef SNICHOLLS_DOXYGEN
    // Pointers to member functions, every qualification: the signature's traits plus class_type, and
    // is_member true. Generated, so Doxygen sees the description on the primary template instead.
#define SNICHOLLS_CALLABLE_TRAITS_MEMBER(qualifiers, is_noexcept_value)                                                   \
    template <typename C, typename R, typename... Args>                                                                 \
    struct callable_traits<R (C::*)(Args...) qualifiers> : detail::signature_traits<R, is_noexcept_value, Args...> {    \
        using class_type = C;                                                                                           \
        static constexpr bool is_member = true;                                                                         \
    };
    SNICHOLLS_CALLABLE_TRAITS_MEMBER(, false)
    SNICHOLLS_CALLABLE_TRAITS_MEMBER(const, false)
    SNICHOLLS_CALLABLE_TRAITS_MEMBER(volatile, false)
    SNICHOLLS_CALLABLE_TRAITS_MEMBER(const volatile, false)
    SNICHOLLS_CALLABLE_TRAITS_MEMBER(&, false)
    SNICHOLLS_CALLABLE_TRAITS_MEMBER(const&, false)
    SNICHOLLS_CALLABLE_TRAITS_MEMBER(volatile&, false)
    SNICHOLLS_CALLABLE_TRAITS_MEMBER(const volatile&, false)
    SNICHOLLS_CALLABLE_TRAITS_MEMBER(&&, false)
    SNICHOLLS_CALLABLE_TRAITS_MEMBER(const&&, false)
    SNICHOLLS_CALLABLE_TRAITS_MEMBER(volatile&&, false)
    SNICHOLLS_CALLABLE_TRAITS_MEMBER(const volatile&&, false)
    SNICHOLLS_CALLABLE_TRAITS_MEMBER(noexcept, true)
    SNICHOLLS_CALLABLE_TRAITS_MEMBER(const noexcept, true)
    SNICHOLLS_CALLABLE_TRAITS_MEMBER(volatile noexcept, true)
    SNICHOLLS_CALLABLE_TRAITS_MEMBER(const volatile noexcept, true)
    SNICHOLLS_CALLABLE_TRAITS_MEMBER(& noexcept, true)
    SNICHOLLS_CALLABLE_TRAITS_MEMBER(const& noexcept, true)
    SNICHOLLS_CALLABLE_TRAITS_MEMBER(volatile& noexcept, true)
    SNICHOLLS_CALLABLE_TRAITS_MEMBER(const volatile& noexcept, true)
    SNICHOLLS_CALLABLE_TRAITS_MEMBER(&& noexcept, true)
    SNICHOLLS_CALLABLE_TRAITS_MEMBER(const&& noexcept, true)
    SNICHOLLS_CALLABLE_TRAITS_MEMBER(volatile&& noexcept, true)
    SNICHOLLS_CALLABLE_TRAITS_MEMBER(const volatile&& noexcept, true)
#undef SNICHOLLS_CALLABLE_TRAITS_MEMBER
#endif

    /// \brief A class with exactly one non-template `operator()` (a lambda, a function object, a `std::function`):
    ///        the `operator()`'s traits, but called as an object, not through a member pointer.
    /// \tparam F The class type, unqualified.
    template <typename F>
        requires(std::is_class_v<F> && std::is_same_v<F, std::remove_cvref_t<F>> && requires { &F::operator(); })
    struct callable_traits<F> : callable_traits<decltype(&F::operator())> {
        /// \brief The object's type.
        using class_type = F;
        /// \brief Called as `f(args...)`, not as a pointer to member function.
        static constexpr bool is_member = false;
    };

    /// \brief A callable type with exactly one signature: `callable_traits<F>` is defined.
    /// \ingroup core
    /// \tparam F The callable type; cv and references are stripped.
    template <typename F>
    concept single_signature_callable = requires { typename callable_traits<F>::signature; };

    /// \brief The result type of a single-signature callable.
    /// \ingroup core
    /// \tparam F The callable type.
    template <single_signature_callable F>
    using callable_result_t = typename callable_traits<F>::result_type;

    /// \brief The arguments of a single-signature callable, as a `meta::type_list`.
    /// \ingroup core
    /// \tparam F The callable type.
    template <single_signature_callable F>
    using callable_arguments_t = typename callable_traits<F>::arguments;

    /// \brief The number of arguments of a single-signature callable.
    /// \ingroup core
    /// \tparam F The callable type.
    template <single_signature_callable F>
    inline constexpr std::size_t callable_arity_v = callable_traits<F>::arity;

} // namespace snicholls

#endif /* snicholls_core_meta_callable_traits_hpp */
