//
//  result.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::result<V, E>`: Rust's `Result`, zero overhead, rethrowable, with `SNICHOLLS_TRY` as `?`.
/// \ingroup core
///
/// | Rust | snicholls |
/// |---|---|
/// | `Ok(v)`, `Err(e)`, `Ok(())` | `snicholls::ok(v)`, `snicholls::err(e)`, `snicholls::ok()` |
/// | `Result<T, E>`, `Result<(), E>` | `snicholls::result<T, E>`, `snicholls::result<void, E>` |
/// | `r?` | `SNICHOLLS_TRY(r)` (GCC/Clang), `SNICHOLLS_TRY_ASSIGN(auto x, r);` / `SNICHOLLS_TRY_VOID(r);` (portable) |
/// | `r.context("...")` (anyhow) | `r.context("loading {}", id)` (needs `snicholls/core/traced.hpp`) |
/// | `catch_unwind` | `snicholls::try_invoke(f)`, `snicholls::result_from_exceptions(f, map)` |
///
/// What happens on an error is chosen per call, not per type:
///
/// | Call | On an error |
/// |---|---|
/// | `unwrap()`, `expect(msg)` | follows the program's policy, `SNICHOLLS_RESULT_THROW` (throw `bad_result_access<E>`, or print and terminate) |
/// | `value_or_throw()` | always throws, the error as its own exception (`snicholls::throw_error`) |
/// | `unwrap_or_terminate()` | always prints the error and calls `std::terminate`; `noexcept` |
/// | `unwrap_unchecked()`, `*r` | not checked: `is_ok()` is a precondition |
/// | `unwrap_or(v)`, `unwrap_or_else(f)`, `unwrap_or_default()` | a fallback value |
/// | `SNICHOLLS_TRY(r)` and its portable forms | returns the error to the caller |
///
/// A policy parameter on the type (`result<V, E, Policy>`) was considered and rejected: it splits one vocabulary type
/// into several that do not convert, and the choice spreads into every signature that names a result.
///
/// The full design and the API table are in [RESULT_PLAN.md](RESULT_PLAN.md). Standard library only: it knows
/// nothing about any consumer's objects or fields.

#ifndef snicholls_core_result_hpp
#define snicholls_core_result_hpp

#include <cassert>
#include <compare>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <format>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>
#include <version>

#if defined(__cpp_lib_source_location)
#include <source_location>
#endif

#if __has_include(<expected>)
#include <expected>
#endif

/// \brief Unwrap-on-error policy: 1 throws `snicholls::bad_result_access`, 0 prints and calls `std::terminate`.
/// \ingroup core
///
/// 1 is the default when exceptions are on, 0 under `-fno-exceptions`. It is a whole-program switch: set it for the
/// whole program (`-DSNICHOLLS_RESULT_THROW=0`), never per file. Per call site, `value_or_throw()` and
/// `unwrap_or_terminate()` choose explicitly whatever the policy.
///
/// The two policies put `snicholls::result` in different inline namespaces (`snicholls::result_throws` /
/// `snicholls::result_terminates`). Mixing them is a link error wherever a result appears in a signature shared by
/// translation units built with different policies; otherwise (a function that only uses results inside its body,
/// an inline function, a template) it is an ODR violation that nothing diagnoses. On MSVC a
/// `#pragma detect_mismatch("snicholls_result_throw", ...)` makes any mix a link error; GCC and Clang have no
/// equivalent, so the whole-program rule stands.
#ifndef SNICHOLLS_RESULT_THROW
#if defined(__cpp_exceptions) || defined(__EXCEPTIONS) || defined(SNICHOLLS_DOXYGEN)
#define SNICHOLLS_RESULT_THROW 1
#else
#define SNICHOLLS_RESULT_THROW 0
#endif
#endif

#if SNICHOLLS_RESULT_THROW && !(defined(__cpp_exceptions) || defined(__EXCEPTIONS) || defined(SNICHOLLS_DOXYGEN))
#error "snicholls: SNICHOLLS_RESULT_THROW=1 needs exceptions; build with exceptions or define SNICHOLLS_RESULT_THROW=0"
#endif

// MSVC: every object file records its policy, and the linker refuses to mix them (LNK2038).
#if defined(_MSC_VER) && !defined(__clang__)
#if SNICHOLLS_RESULT_THROW
#pragma detect_mismatch("snicholls_result_throw", "1")
#else
#pragma detect_mismatch("snicholls_result_throw", "0")
#endif
#endif

/// \brief 1 when the compiler has exceptions (`try_invoke` can catch, `rethrow()` can throw), else 0.
/// \ingroup core
#if defined(__cpp_exceptions) || defined(__EXCEPTIONS)
#define SNICHOLLS_RESULT_HAS_EXCEPTIONS 1
#else
#define SNICHOLLS_RESULT_HAS_EXCEPTIONS 0
#endif

/// \brief 1 when `SNICHOLLS_TRY(expr)` (a GNU statement expression) is available (GCC, Clang); 0 on MSVC.
/// \ingroup core
#if defined(__GNUC__) || defined(__clang__)
#define SNICHOLLS_RESULT_HAS_STATEMENT_EXPRESSIONS 1
#else
#define SNICHOLLS_RESULT_HAS_STATEMENT_EXPRESSIONS 0
#endif

/// \brief `[[no_unique_address]]`, spelled so MSVC honours it too (it ignores the standard spelling).
/// \ingroup core
#if defined(_MSC_VER) && !defined(__clang__)
#define SNICHOLLS_RESULT_NO_UNIQUE_ADDRESS [[msvc::no_unique_address]]
#else
#define SNICHOLLS_RESULT_NO_UNIQUE_ADDRESS [[no_unique_address]]
#endif

namespace snicholls {

#if defined(__cpp_lib_source_location)
/// \brief `std::source_location` where the library has it (every C++20 library core supports).
/// \ingroup core
using source_location = std::source_location;
#else
/// \brief Stand-in for `std::source_location` on a library without it: every location is empty (line 0).
/// \ingroup core
struct source_location
{
    /// \brief An empty location.
    /// \return the location.
    [[nodiscard]] static constexpr source_location current() noexcept { return {}; }
    /// \brief The file: empty.
    /// \return "".
    [[nodiscard]] constexpr const char* file_name() const noexcept { return ""; }
    /// \brief The function: empty.
    /// \return "".
    [[nodiscard]] constexpr const char* function_name() const noexcept { return ""; }
    /// \brief The line: 0.
    /// \return 0.
    [[nodiscard]] constexpr std::uint_least32_t line() const noexcept { return 0; }
    /// \brief The column: 0.
    /// \return 0.
    [[nodiscard]] constexpr std::uint_least32_t column() const noexcept { return 0; }
};
#endif

template <class E>
class traced;   // snicholls/core/traced.hpp: the opt-in error with a source location, context chain, fields and stack.

// -------------------------------------------------------------------------------------------------------------------
// ok / err tags
// -------------------------------------------------------------------------------------------------------------------

/// \brief The success tag returned by `snicholls::ok(v)`; converts to any result whose V it constructs.
/// \ingroup core
/// \tparam V the value type (`void` for `snicholls::ok()`).
template <class V>
struct ok_t
{
    /// \brief The value.
    V value;
};

/// \brief The success tag of `snicholls::result<void, E>`, returned by `snicholls::ok()`.
/// \ingroup core
template <>
struct ok_t<void>
{
};

/// \brief The error tag returned by `snicholls::err(e)`; converts to any result whose E it constructs.
/// \ingroup core
/// \tparam E the error type.
template <class E>
struct err_t
{
    /// \brief The error.
    E error;
};

/// \brief Tag to construct the error of a result in place: `result<V, E>(snicholls::err_in_place, args...)`.
/// \ingroup core
struct err_in_place_t
{
    /// \brief Constructs it.
    explicit err_in_place_t() = default;
};

/// \brief The `snicholls::err_in_place_t` value. (The value is constructed in place with `std::in_place`.)
/// \ingroup core
inline constexpr err_in_place_t err_in_place{};

/// \brief Rust's `Ok(v)`: a success tag holding `v` (decayed, moved or copied).
/// \ingroup core
/// \param v the value.
/// \return `snicholls::ok_t<std::decay_t<V>>{v}`.
template <class V>
[[nodiscard]] constexpr ok_t<std::decay_t<V>> ok(V&& v) noexcept(std::is_nothrow_constructible_v<std::decay_t<V>, V>)
{
    return ok_t<std::decay_t<V>>{std::forward<V>(v)};
}

/// \brief Rust's `Ok(())`: the success of a `snicholls::result<void, E>`.
/// \ingroup core
/// \return `snicholls::ok_t<void>{}`.
[[nodiscard]] constexpr ok_t<void> ok() noexcept { return {}; }

/// \brief Rust's `Err(e)`: an error tag holding `e` (decayed, moved or copied).
/// \ingroup core
/// \param e the error.
/// \return `snicholls::err_t<std::decay_t<E>>{e}`.
template <class E>
[[nodiscard]] constexpr err_t<std::decay_t<E>> err(E&& e) noexcept(std::is_nothrow_constructible_v<std::decay_t<E>, E>)
{
    return err_t<std::decay_t<E>>{std::forward<E>(e)};
}

// -------------------------------------------------------------------------------------------------------------------
// Customisation points
// -------------------------------------------------------------------------------------------------------------------

/// \brief Opt-in niche for `snicholls::result<V, E>` with an empty E: a value of V that is never a valid success.
/// \ingroup core
///
/// Specialise it with `static constexpr V value` (an invalid V, e.g. `-1` for an index type). Then
/// `result<V, E>` for any empty, trivially default-constructible E has exactly `sizeof(V)`: the discriminant is
/// "v == value". Constructing `ok(value)` is a precondition violation and terminates. Nothing in core specialises it:
/// a raw pointer or integer has no invalid value (`nullptr` and `-1` are valid successes), so a niche is sound only
/// for a type whose invariant excludes one value.
///
/// The state *is* the value: a niche result whose V has been moved from reads as an error if the moved-from V
/// equals the niche value (for example a handle whose move constructor leaves the source at `-1`, the niche).
/// Do not use a niche result after moving its value out (`std::move(r).unwrap()`), only assign to it or destroy it.
/// \tparam V the value type.
template <class V>
struct result_niche
{
};

/// \brief Opt-in conversion of an error to an exception, used by `rethrow()`, `rethrow_if_err()` and
///        `value_or_throw()`.
/// \ingroup core
///
/// Specialise it with `[[noreturn]] static void raise(const E&)` to throw your own exception type for E. Without a
/// specialisation: `std::exception_ptr` is rethrown, a `std::exception` is thrown as itself (its dynamic type is
/// the static type E), `std::error_code` becomes `std::system_error`, a `snicholls::traced<X>` raises its X, and
/// anything else becomes `snicholls::bad_result_access<E>`.
/// \tparam E the error type.
template <class E>
struct error_exception
{
};

/// \brief Converts a propagated error of type From into the enclosing function's error type To (`SNICHOLLS_TRY`).
/// \ingroup core
///
/// The primary template constructs To from From. `snicholls/core/traced.hpp` specialises it so that a traced error gains a
/// "propagated" frame (the expression text and the `SNICHOLLS_TRY` location), and a plain error entering a function
/// that returns a traced one is wrapped with that location. Rust's `From` in `?`.
/// \tparam To the error type of the enclosing function.
/// \tparam From the error type of the expression.
template <class To, class From>
struct error_conversion
{
    /// \brief Converts the error.
    /// \param e the error.
    /// \return `To(std::move(e))`.
    static constexpr To convert(From&& e, const char* /*expr*/, source_location /*where*/) noexcept(
        std::is_nothrow_constructible_v<To, From>)
    {
        return To(std::move(e));
    }
};

/// \brief True for `snicholls::traced<X>`.
/// \ingroup core
/// \tparam E the type.
template <class E>
struct is_traced : std::false_type
{
};

/// \brief True for `snicholls::traced<X>`.
/// \ingroup core
/// \tparam X the traced error's inner type.
template <class X>
struct is_traced<traced<X>> : std::true_type
{
};

/// \brief `snicholls::is_traced<E>::value`.
/// \ingroup core
/// \tparam E the type.
template <class E>
inline constexpr bool is_traced_v = is_traced<std::remove_cvref_t<E>>::value;

namespace result_detail {

template <class E>
struct traced_of
{
    using type = traced<E>;
};

template <class X>
struct traced_of<traced<X>>
{
    using type = traced<X>;
};

template <class T>
concept string_viewable = std::convertible_to<const T&, std::string_view>;

template <class T>
concept has_what = requires(const T& t) {
    { t.what() } -> std::convertible_to<const char*>;
};

template <class T>
concept has_message = requires(const T& t) {
    { t.message() } -> std::convertible_to<std::string>;
};

template <class T>
concept std_formattable = std::is_default_constructible_v<std::formatter<std::remove_cvref_t<T>, char>>;

template <class T>
concept adl_error_text = requires(const T& t) {
    { snicholls_error_text(t) } -> std::convertible_to<std::string>;
};

template <class E>
concept has_custom_raise = requires(const E& e) { error_exception<E>::raise(e); };

template <class V>
concept has_niche = requires {
    { result_niche<V>::value } -> std::convertible_to<const V&>;
};

} // namespace result_detail

/// \brief `snicholls::traced<E>` for a plain E, and E itself when it is already traced.
/// \ingroup core
/// \tparam E the error type.
template <class E>
using traced_t = typename result_detail::traced_of<std::remove_cvref_t<E>>::type;

/// \brief One line of text describing an error, for messages (`unwrap`, `expect`, logging).
/// \ingroup core
///
/// In order: an ADL `snicholls_error_text(e)` (the customisation point; `snicholls::traced` provides one), `what()` for
/// exceptions, the message of a `std::exception_ptr`, `message()` (`std::error_code`), the text of a string-like
/// error, `std::format("{}", e)` when E is formattable, the integer of an enum, else `"error"`.
/// Never call it on a secret: it reveals whatever the error holds.
/// \param e the error.
/// \return the text.
/// \throws std::bad_alloc.
template <class E>
[[nodiscard]] std::string error_text(const E& e)
{
    using T = std::remove_cvref_t<E>;
    if constexpr (result_detail::adl_error_text<T>)
        return std::string(snicholls_error_text(e));
    else if constexpr (result_detail::has_what<T>)
        return std::string(e.what());
    else if constexpr (std::is_same_v<T, std::exception_ptr>)
    {
        if (!e)
            return "null exception_ptr";
#if SNICHOLLS_RESULT_HAS_EXCEPTIONS
        try
        {
            std::rethrow_exception(e);
        }
        catch (const std::exception& x)
        {
            return std::string(x.what());
        }
        catch (...)
        {
            return "unknown exception";
        }
#else
        return "exception";
#endif
    }
    else if constexpr (result_detail::string_viewable<T>)
        return std::string(std::string_view(e));
    else if constexpr (result_detail::has_message<T>)
        return std::string(e.message());
    else if constexpr (result_detail::std_formattable<T>)
        return std::format("{}", e);
    else if constexpr (std::is_enum_v<T>)
        return "error " + std::to_string(static_cast<std::underlying_type_t<T>>(e));
    else
        return "error";
}

// -------------------------------------------------------------------------------------------------------------------
// bad_result_access
// -------------------------------------------------------------------------------------------------------------------

/// \brief Thrown by `unwrap()` / `expect()` on an error (`SNICHOLLS_RESULT_THROW=1`), carrying the error.
/// \ingroup core
///
/// `bad_result_access<void>` is the base of every `bad_result_access<E>` and derives from `std::logic_error` (as the
/// older `Result::unwrap` implementations do, so existing `catch (const std::logic_error&)` still works). It is thrown
/// alone by `unwrap_err()` on a success and by `unwrap()` when E is not copyable (an exception object must be).
/// \tparam E the error type, or void.
template <class E = void>
class bad_result_access;

/// \brief The base of every `snicholls::bad_result_access<E>`; derives from `std::logic_error`.
/// \ingroup core
template <>
class bad_result_access<void> : public std::logic_error
{
public:
    /// \brief Constructs it.
    /// \param what the message.
    explicit bad_result_access(const std::string& what) : std::logic_error(what) {}
};

/// \brief `snicholls::bad_result_access` carrying the error E that `unwrap()` / `expect()` found.
/// \ingroup core
/// \tparam E the error type (copy constructible).
template <class E>
#ifdef SNICHOLLS_DOXYGEN
class bad_result_access : public std::logic_error   // Doxygen misreads the real base as recursive (docs/Doxyfile.in)
#else
class bad_result_access : public bad_result_access<void>
#endif
{
public:
    /// \brief Constructs it.
    /// \param what the message.
    /// \param error the error.
    bad_result_access(const std::string& what, E error) : bad_result_access<void>(what), error_(std::move(error)) {}

    /// \name The error.
    ///@{
    /// \brief The error.
    /// \return the error.
    [[nodiscard]] const E& error() const& noexcept { return error_; }
    [[nodiscard]] E& error() & noexcept { return error_; }
    [[nodiscard]] E&& error() && noexcept { return std::move(error_); }
    ///@}

private:
    E error_;
};

/// \brief Throws E as an exception: `snicholls::error_exception<E>::raise`, else the defaults described there.
/// \ingroup core
/// \param e the error.
/// \throws the exception E converts to. Without exceptions, prints the error and terminates.
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(push)
#pragma warning(disable : 4702)   // the closing std::terminate() is unreachable whenever every branch throws
#endif
template <class E>
[[noreturn]] void throw_error(const E& e)
{
    using T = std::remove_cvref_t<E>;
#if SNICHOLLS_RESULT_HAS_EXCEPTIONS
    if constexpr (result_detail::has_custom_raise<T>)
        error_exception<T>::raise(e);
    else if constexpr (std::is_same_v<T, std::exception_ptr>)
    {
        if (e)
            std::rethrow_exception(e);
        throw bad_result_access<void>("snicholls::result: rethrow of a null exception_ptr");
    }
    else if constexpr (std::is_base_of_v<std::exception, T> && std::is_copy_constructible_v<T>)
        throw e;
    else if constexpr (std::is_same_v<T, std::error_code>)
        throw std::system_error(e);
    else if constexpr (std::is_copy_constructible_v<T>)
        throw bad_result_access<T>("snicholls::result: error: " + error_text(e), e);
    else
        throw bad_result_access<void>("snicholls::result: error: " + error_text(e));
#else
    std::fputs("snicholls::result: error: ", stderr);
    std::fputs(error_text(e).c_str(), stderr);
    std::fputs("\n", stderr);
    std::terminate();
#endif
    // Every branch above either throws or terminates; this keeps [[noreturn]] honest for the compiler.
    std::terminate();
}
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(pop)
#endif

#if SNICHOLLS_RESULT_THROW
inline namespace result_throws {
#else
inline namespace result_terminates {
#endif

template <class V, class E>
class result;

namespace result_impl {

/// The niche-violation path: a precondition violation, so it terminates under both policies.
[[noreturn]] inline void niche_violation() noexcept
{
    std::fputs("snicholls::result: ok() of the niche value of V (snicholls::result_niche) is a precondition violation\n", stderr);
    std::terminate();
}

/// Unwrap / expect on an error: throws bad_result_access<E> (policy throw) or prints and terminates.
template <class E>
[[noreturn]] void unwrap_failed(std::string_view what, const E& e)
{
    std::string text(what);
    text += ": ";
    text += snicholls::error_text(e);
#if SNICHOLLS_RESULT_THROW
    if constexpr (std::is_copy_constructible_v<E>)
        throw bad_result_access<E>(text, e);
    else
        throw bad_result_access<void>(text);
#else
    std::fputs("snicholls::result: ", stderr);
    std::fputs(text.c_str(), stderr);
    std::fputs("\n", stderr);
    std::terminate();
#endif
}

/// unwrap_or_terminate on an error: prints and terminates whatever the policy.
template <class E>
[[noreturn]] void terminate_failed(std::string_view what, const E& e) noexcept
{
    std::fputs("snicholls::result: ", stderr);
    std::fwrite(what.data(), 1, what.size(), stderr);
    std::fputs(": ", stderr);
    const std::string text = snicholls::error_text(e);   // noexcept: if formatting the error throws, that terminates too
    std::fputs(text.c_str(), stderr);
    std::fputs("\n", stderr);
    std::terminate();
}

/// unwrap_err / expect_err on a success.
[[noreturn]] inline void unwrap_err_failed(std::string_view what)
{
#if SNICHOLLS_RESULT_THROW
    throw bad_result_access<void>(std::string(what));
#else
    std::fputs("snicholls::result: ", stderr);
    std::fwrite(what.data(), 1, what.size(), stderr);
    std::fputs("\n", stderr);
    std::terminate();
#endif
}

inline constexpr bool unwrap_noexcept = !SNICHOLLS_RESULT_THROW;

struct void_value
{
    friend constexpr bool operator==(void_value, void_value) noexcept = default;
    friend constexpr std::strong_ordering operator<=>(void_value, void_value) noexcept = default;
};

template <class T>
inline constexpr bool trivially_copyable_member =
    std::is_trivially_copy_constructible_v<T> && std::is_trivially_copy_assignable_v<T> &&
    std::is_trivially_destructible_v<T>;

template <class T>
inline constexpr bool trivially_movable_member =
    std::is_trivially_move_constructible_v<T> && std::is_trivially_move_assignable_v<T> &&
    std::is_trivially_destructible_v<T>;

template <class V, class E>
inline constexpr bool trivial_copy_ctor = std::is_trivially_copy_constructible_v<V> && std::is_trivially_copy_constructible_v<E>;
template <class V, class E>
inline constexpr bool trivial_move_ctor = std::is_trivially_move_constructible_v<V> && std::is_trivially_move_constructible_v<E>;
template <class V, class E>
inline constexpr bool trivial_copy_assign = trivially_copyable_member<V> && trivially_copyable_member<E>;
template <class V, class E>
inline constexpr bool trivial_move_assign = trivially_movable_member<V> && trivially_movable_member<E>;

/// Tag: build a storage from another result's state (converting constructors), one construction, no temporary.
struct convert_t
{
};

template <class V, class E>
inline constexpr bool use_niche = result_detail::has_niche<V> && std::is_empty_v<E> && std::is_trivially_default_constructible_v<E> &&
                                  std::is_trivially_copyable_v<E>;

/// Union + discriminant. The destructor and the assignments are trivial exactly when they are for both V and E
/// (P0848 constrained members). Copy and move constructors live in the layers below instead: Clang treats a class
/// that declares any non-trivial copy/move constructor - even one whose constraint is not satisfied - as non-trivial
/// for the purposes of calls, and returns it through memory instead of registers (RESULT_PLAN.md, layout).
template <class V, class E>
class storage_union
{
public:
    template <class... A>
    constexpr explicit storage_union(std::in_place_t,
                                      A&&... a) noexcept(std::is_nothrow_constructible_v<V, A...>)
        : v_(std::forward<A>(a)...), ok_(true)
    {
    }

    template <class... A>
    constexpr explicit storage_union(err_in_place_t, A&&... a) noexcept(std::is_nothrow_constructible_v<E, A...>)
        : e_(std::forward<A>(a)...), ok_(false)
    {
    }

    template <class FV, class FE>
    constexpr storage_union(convert_t, bool ok, FV&& make_value, FE&& make_error) : ok_(ok)
    {
        if (ok)
            std::construct_at(std::addressof(v_), std::forward<FV>(make_value)());
        else
            std::construct_at(std::addressof(e_), std::forward<FE>(make_error)());
    }

    constexpr ~storage_union()
        requires(std::is_trivially_destructible_v<V> && std::is_trivially_destructible_v<E>)
    = default;
    constexpr ~storage_union() { destroy(); }

    // Trivial when V and E are, deleted otherwise (the layers above then supply them). Declared because the
    // assignments below would otherwise delete the copy and suppress the move.
    constexpr storage_union(const storage_union&) = default;
    constexpr storage_union(storage_union&&) = default;

    constexpr storage_union& operator=(const storage_union&)
        requires trivial_copy_assign<V, E>
    = default;
    constexpr storage_union& operator=(const storage_union& o) noexcept(
        std::is_nothrow_copy_constructible_v<V> && std::is_nothrow_copy_assignable_v<V> &&
        std::is_nothrow_copy_constructible_v<E> && std::is_nothrow_copy_assignable_v<E>)
        requires(std::is_copy_constructible_v<V> && std::is_copy_assignable_v<V> && std::is_copy_constructible_v<E> &&
                 std::is_copy_assignable_v<E> && !trivial_copy_assign<V, E> &&
                 (std::is_nothrow_move_constructible_v<V> || std::is_nothrow_move_constructible_v<E>))
    {
        if (ok_ && o.ok_)
            v_ = o.v_;
        else if (!ok_ && !o.ok_)
            e_ = o.e_;
        else if (o.ok_)
            reinit_value(o.v_);
        else
            reinit_error(o.e_);
        return *this;
    }

    constexpr storage_union& operator=(storage_union&&)
        requires trivial_move_assign<V, E>
    = default;
    constexpr storage_union& operator=(storage_union&& o) noexcept(
        std::is_nothrow_move_constructible_v<V> && std::is_nothrow_move_assignable_v<V> &&
        std::is_nothrow_move_constructible_v<E> && std::is_nothrow_move_assignable_v<E>)
        requires(std::is_move_constructible_v<V> && std::is_move_assignable_v<V> && std::is_move_constructible_v<E> &&
                 std::is_move_assignable_v<E> && !trivial_move_assign<V, E> &&
                 (std::is_nothrow_move_constructible_v<V> || std::is_nothrow_move_constructible_v<E>))
    {
        if (ok_ && o.ok_)
            v_ = std::move(o.v_);
        else if (!ok_ && !o.ok_)
            e_ = std::move(o.e_);
        else if (o.ok_)
            reinit_value(std::move(o.v_));
        else
            reinit_error(std::move(o.e_));
        return *this;
    }

    [[nodiscard]] constexpr bool has_value() const noexcept { return ok_; }
    [[nodiscard]] constexpr V& val() & noexcept { return v_; }
    [[nodiscard]] constexpr const V& val() const& noexcept { return v_; }
    [[nodiscard]] constexpr V&& val() && noexcept { return std::move(v_); }
    [[nodiscard]] constexpr const V&& val() const&& noexcept { return std::move(v_); }
    [[nodiscard]] constexpr E& err() & noexcept { return e_; }
    [[nodiscard]] constexpr const E& err() const& noexcept { return e_; }
    [[nodiscard]] constexpr E&& err() && noexcept { return std::move(e_); }
    [[nodiscard]] constexpr const E&& err() const&& noexcept { return std::move(e_); }

    /// Switch to the value state (strong guarantee: on an exception the old error is restored).
    template <class... A>
    constexpr void reinit_value(A&&... a)
    {
        if (ok_)
        {
            // Assign, never destroy-then-construct: if constructing the new value threw after the old one was
            // destroyed, ok_ would still say "value" and the destructor would destroy it a second time.
            if constexpr (sizeof...(A) == 1 && (std::is_assignable_v<V&, A> && ...))
                ((v_ = std::forward<A>(a)), ...);
            else
                v_ = V(std::forward<A>(a)...);
            return;
        }
        if constexpr (std::is_nothrow_constructible_v<V, A...>)
        {
            std::destroy_at(std::addressof(e_));
            std::construct_at(std::addressof(v_), std::forward<A>(a)...);
        }
        else if constexpr (std::is_nothrow_move_constructible_v<V>)
        {
            V tmp(std::forward<A>(a)...);
            std::destroy_at(std::addressof(e_));
            std::construct_at(std::addressof(v_), std::move(tmp));
        }
        else
        {
            E saved(std::move(e_));
            std::destroy_at(std::addressof(e_));
#if SNICHOLLS_RESULT_HAS_EXCEPTIONS
            try
            {
                std::construct_at(std::addressof(v_), std::forward<A>(a)...);
            }
            catch (...)
            {
                std::construct_at(std::addressof(e_), std::move(saved));
                throw;
            }
#else
            std::construct_at(std::addressof(v_), std::forward<A>(a)...);
#endif
        }
        ok_ = true;
    }

    /// Switch to the error state (strong guarantee: on an exception the old value is restored).
    template <class... A>
    constexpr void reinit_error(A&&... a)
    {
        if (!ok_)
        {
            // Assign, as reinit_value does: destroy-then-construct would destroy the error twice if construction threw.
            if constexpr (sizeof...(A) == 1 && (std::is_assignable_v<E&, A> && ...))
                ((e_ = std::forward<A>(a)), ...);
            else
                e_ = E(std::forward<A>(a)...);
            return;
        }
        if constexpr (std::is_nothrow_constructible_v<E, A...>)
        {
            std::destroy_at(std::addressof(v_));
            std::construct_at(std::addressof(e_), std::forward<A>(a)...);
        }
        else if constexpr (std::is_nothrow_move_constructible_v<E>)
        {
            E tmp(std::forward<A>(a)...);
            std::destroy_at(std::addressof(v_));
            std::construct_at(std::addressof(e_), std::move(tmp));
        }
        else
        {
            V saved(std::move(v_));
            std::destroy_at(std::addressof(v_));
#if SNICHOLLS_RESULT_HAS_EXCEPTIONS
            try
            {
                std::construct_at(std::addressof(e_), std::forward<A>(a)...);
            }
            catch (...)
            {
                std::construct_at(std::addressof(v_), std::move(saved));
                throw;
            }
#else
            std::construct_at(std::addressof(e_), std::forward<A>(a)...);
#endif
        }
        ok_ = false;
    }

private:
    constexpr void destroy() noexcept
    {
        if (ok_)
            std::destroy_at(std::addressof(v_));
        else
            std::destroy_at(std::addressof(e_));
    }

    union
    {
        V v_;
        E e_;
    };
    bool ok_;
};


/// Copy constructor layer: implicit (trivial, or deleted) unless V or E needs a hand-written one.
template <class V, class E,
          bool Implicit = trivial_copy_ctor<V, E> || !(std::is_copy_constructible_v<V> && std::is_copy_constructible_v<E>)>
class copy_ctor_layer : public storage_union<V, E>
{
public:
    using storage_union<V, E>::storage_union;
};

template <class V, class E>
class copy_ctor_layer<V, E, false> : public storage_union<V, E>
{
    using base = storage_union<V, E>;

public:
    using base::base;

    constexpr copy_ctor_layer(const copy_ctor_layer& o) noexcept(
        std::is_nothrow_copy_constructible_v<V> && std::is_nothrow_copy_constructible_v<E>)
        : base(convert_t{}, o.has_value(), [&]() -> const V& { return o.val(); }, [&]() -> const E& { return o.err(); })
    {
    }
    constexpr copy_ctor_layer(copy_ctor_layer&&) = default;
    constexpr copy_ctor_layer& operator=(const copy_ctor_layer&) = default;
    constexpr copy_ctor_layer& operator=(copy_ctor_layer&&) = default;
};

/// Move constructor layer: implicit (trivial, or falling back to the copy) unless V or E needs a hand-written one.
template <class V, class E,
          bool Implicit = trivial_move_ctor<V, E> || !(std::is_move_constructible_v<V> && std::is_move_constructible_v<E>)>
class move_ctor_layer : public copy_ctor_layer<V, E>
{
public:
    using copy_ctor_layer<V, E>::copy_ctor_layer;
};

template <class V, class E>
class move_ctor_layer<V, E, false> : public copy_ctor_layer<V, E>
{
    using base = copy_ctor_layer<V, E>;

public:
    using base::base;

    constexpr move_ctor_layer(const move_ctor_layer&) = default;
    constexpr move_ctor_layer(move_ctor_layer&& o) noexcept(
        std::is_nothrow_move_constructible_v<V> && std::is_nothrow_move_constructible_v<E>)
        : base(convert_t{}, o.has_value(), [&]() -> V&& { return std::move(o).val(); },
               [&]() -> E&& { return std::move(o).err(); })
    {
    }
    constexpr move_ctor_layer& operator=(const move_ctor_layer&) = default;
    constexpr move_ctor_layer& operator=(move_ctor_layer&&) = default;
};

/// The general layout (Niche = false): union + bool, every special member trivial exactly when it is for V and E.
template <class V, class E, bool Niche = use_niche<V, E>>
class result_storage : public move_ctor_layer<V, E>
{
public:
    using move_ctor_layer<V, E>::move_ctor_layer;
};

/// Niche storage: V alone (E is empty and lives in no bytes); "error" is V == result_niche<V>::value.
template <class V, class E>
class result_storage<V, E, true>
{
public:
    template <class... A>
    constexpr explicit result_storage(std::in_place_t,
                                      A&&... a) noexcept(std::is_nothrow_constructible_v<V, A...>)
        : v_(std::forward<A>(a)...)
    {
        if (!has_value())
            niche_violation();
    }

    template <class... A>
    constexpr explicit result_storage(err_in_place_t, A&&... a) noexcept(std::is_nothrow_constructible_v<E, A...>)
        : v_(result_niche<V>::value), e_(std::forward<A>(a)...)
    {
    }

    template <class FV, class FE>
    constexpr result_storage(convert_t, bool ok, FV&& make_value, FE&& make_error)
        : v_(ok ? V(std::forward<FV>(make_value)()) : V(result_niche<V>::value))
    {
        if (!ok)
            e_ = E(std::forward<FE>(make_error)());
        else if (!has_value())
            niche_violation();
    }

    [[nodiscard]] constexpr bool has_value() const noexcept { return !(v_ == result_niche<V>::value); }
    [[nodiscard]] constexpr V& val() & noexcept { return v_; }
    [[nodiscard]] constexpr const V& val() const& noexcept { return v_; }
    [[nodiscard]] constexpr V&& val() && noexcept { return std::move(v_); }
    [[nodiscard]] constexpr const V&& val() const&& noexcept { return std::move(v_); }
    [[nodiscard]] constexpr E& err() & noexcept { return e_; }
    [[nodiscard]] constexpr const E& err() const& noexcept { return e_; }
    [[nodiscard]] constexpr E&& err() && noexcept { return std::move(e_); }
    [[nodiscard]] constexpr const E&& err() const&& noexcept { return std::move(e_); }

    template <class... A>
    constexpr void reinit_value(A&&... a)
    {
        v_ = V(std::forward<A>(a)...);
        if (!has_value())
            niche_violation();
    }

    template <class... A>
    constexpr void reinit_error(A&&... a)
    {
        v_ = result_niche<V>::value;
        e_ = E(std::forward<A>(a)...);
    }

private:
    V v_;
    SNICHOLLS_RESULT_NO_UNIQUE_ADDRESS E e_{};
};

template <class R>
struct is_result : std::false_type
{
};

template <class V, class E>
struct is_result<result<V, E>> : std::true_type
{
};

template <class T>
inline constexpr bool is_result_v = is_result<std::remove_cvref_t<T>>::value;

template <class T>
inline constexpr bool valid_value = std::is_same_v<T, void> ||
                                    (std::is_object_v<T> && !std::is_array_v<T> && !std::is_const_v<T> &&
                                     !std::is_volatile_v<T>);

template <class T>
inline constexpr bool valid_error = std::is_object_v<T> && !std::is_array_v<T> && !std::is_const_v<T> &&
                                    !std::is_volatile_v<T>;

/// A format string that also records where it was written (for `context`).
template <class... A>
struct located_format
{
    std::format_string<A...> fmt;
    source_location where;

    template <class S>
        requires std::convertible_to<const S&, std::string_view>
    consteval located_format(const S& s, source_location w = source_location::current()) : fmt(s), where(w)
    {
    }
};

/// What SNICHOLLS_TRY returns from the enclosing function: converts to any result<V2, E2> via error_conversion.
template <class E>
struct propagated_error
{
    E error;
    const char* expr;
    source_location where;

    template <class V2, class E2>
    constexpr operator result<V2, E2>() &&   // NOLINT(google-explicit-constructor)
    {
        return result<V2, E2>(err_in_place, error_conversion<E2, E>::convert(std::move(error), expr, where));
    }
};

template <class R>
constexpr auto propagate(R&& r, const char* expr, source_location where)
{
    using E = typename std::remove_cvref_t<R>::error_type;
    return propagated_error<E>{std::forward<R>(r).unwrap_err_unchecked(), expr, where};
}

template <class T>
constexpr void hash_mix(std::size_t& seed, const T& v) noexcept(noexcept(std::hash<T>{}(v)))
{
    // Boost.ContainerHash's combine step (the same constant as snicholls::hash_combine, without its finaliser).
    seed ^= std::hash<T>{}(v) + static_cast<std::size_t>(0x9e3779b97f4a7c15ULL) + (seed << 6) + (seed >> 2);
}

} // namespace result_impl

/// \brief Rust's `Result<V, E>`: either a success value V or an error E, with zero runtime overhead.
/// \ingroup core
///
/// \overview
/// A function that can fail returns `snicholls::result<V, E>`: `return snicholls::ok(v);` or `return snicholls::err(e);`. The
/// caller tests it (`is_ok`, `if (r)`), transforms it (`map`, `and_then`, `or_else` ...), propagates it with
/// `SNICHOLLS_TRY` (Rust's `?`), or unwraps it (`unwrap`, `expect`, `unwrap_or` ...). The whole of Rust's `Result` API is
/// here, in Rust's names (`and_` / `or_` because `and` / `or` are C++ keywords), with ref-qualified overloads so
/// values move out without copies. `snicholls::result<void, E>` is the `Result<(), E>` specialisation.
///
/// Use it for expected failures that the caller should handle: parsing, validation, lookups, I/O. Use exceptions
/// for failures nobody nearby can handle, and `try_invoke` to turn one into the other at a boundary. Errors are any
/// object type: an enum or `std::error_code` costs nothing; `snicholls::traced<E>` (`snicholls/core/traced.hpp`) adds a source
/// location, a context chain, fields and an optional stack trace for logging, paying only on the error path.
///
/// \relations
/// - `snicholls::ok`, `snicholls::err`, `snicholls::ok_t`, `snicholls::err_t`: the constructors (Rust's `Ok` / `Err`).
/// - `snicholls::traced`, `snicholls::error_sink`: detailed errors and logging (`snicholls/core/traced.hpp`); `context()`,
///   `with_field()`, `with_stack()`, `trace()` and `log_err()` here are their entry points.
/// - `snicholls::try_invoke`, `snicholls::result_from_exceptions`, `rethrow()`, `value_or_throw()`, `snicholls::throw_error`,
///   `snicholls::error_exception`: the bridge to and from exceptions.
/// - `SNICHOLLS_TRY`, `SNICHOLLS_TRY_ASSIGN`, `SNICHOLLS_TRY_VOID`: propagation.
/// - `std::expected` (C++23): `to_expected()` and a constructor, under `__cpp_lib_expected`.
/// - Module: \ref core.
///
/// \impls
/// One class, three layouts chosen at compile time:
/// | V, E | Layout | Size |
/// |---|---|---|
/// | general | `union { V; E; }` + `bool` | that of the hand-written tagged union |
/// | `void`, E | `union { empty; E; }` + `bool` | that of `struct { E; bool; }` |
/// | V with a `snicholls::result_niche`, E empty | V alone | `sizeof(V)` |
///
/// \contracts
/// - **Layout:** no allocation, no virtual function, no RTTI. Trivially copyable / destructible (and each special
///   member trivial) exactly when it is for both V and E. Proven by static_asserts in tests/result_test.cpp.
/// - **noexcept** exactly when the operations performed are (copies, moves, the callables). `unwrap`, `expect`,
///   `unwrap_err` and `expect_err` are `noexcept` under `SNICHOLLS_RESULT_THROW=0`, where they terminate instead.
/// - **constexpr:** every operation except the error paths of unwrap/expect/rethrow.
/// - **Assignment** between the two states gives the strong guarantee (std::expected's rule); it exists when V or
///   E is nothrow move constructible.
/// - Not thread-safe beyond the usual value-type rules: concurrent const access is safe.
/// - \pre `operator*`, `operator->`, `unwrap_unchecked()` need `is_ok()`; `unwrap_err_unchecked()` needs
///   `is_err()` (asserted in debug builds, undefined behaviour otherwise, as in Rust).
///
/// \errors
/// - `unwrap()` / `expect(msg)` on an error throw `snicholls::bad_result_access<E>` (`bad_result_access<void>` when E is
///   not copyable); `unwrap_err()` / `expect_err(msg)` on a success throw `snicholls::bad_result_access<void>`. With
///   `SNICHOLLS_RESULT_THROW=0` each prints its message and calls `std::terminate`.
/// - `rethrow()`, `rethrow_if_err()`, `value_or_throw()` throw the error as its own exception
///   (`snicholls::throw_error`).
/// - Everything else throws only what the callables and the copies / moves of V and E throw.
///
/// \examples
/// \snippet result_demo.cpp parse
/// \snippet result_demo.cpp try
/// \snippet result_demo.cpp rethrow
///
/// \perf
/// Not yet benchmarked in Release (the numbers wait for a quiet machine). benchmarks/result_benchmark.cpp compares
/// it with a hand-written value-plus-bool struct, an error-code out-parameter and exceptions, on the success path
/// and on error paths at several error rates, after a fairness check; its smoke run (CTest
/// `benchmark.result_benchmark`) proves it compiles and agrees. See [RESULT_PLAN.md](RESULT_PLAN.md).
///
/// \testing
/// - Tests: tests/result_test.cpp (API, layout, noexcept, constexpr, move-only, void, rethrow),
///   tests/result_terminate_test.cpp (the `SNICHOLLS_RESULT_THROW=0` policy, death tests).
/// - Demo: demos/result_demo.cpp.
///
/// \deps
/// CMake target `snicholls::core`; standard library only (core). Written to move unchanged into a dependency-free
/// foundation library.
///
/// \gotchas
/// - `SNICHOLLS_TRY` returns a proxy that converts to the enclosing function's `snicholls::result`: the function needs a
///   declared result return type (not `auto`).
/// - `SNICHOLLS_TRY(expr)` is a GNU statement expression (GCC, Clang); elsewhere use `SNICHOLLS_TRY_ASSIGN` / `SNICHOLLS_TRY_VOID`.
/// - There is no implicit construction from a bare V or E (as some `Result` types have): write `snicholls::ok(v)`
///   / `snicholls::err(e)`. A result whose V and E are similar types (`result<double, int>`) is never ambiguous.
/// - `rethrow()` of a `snicholls::traced<X>` throws X: the context chain is not in the exception. Log first
///   (`log_err`).
/// - Equality and hashing of `snicholls::traced` errors use the inner error only (the context is diagnostics).
/// - The `&&` overloads that hand out the value or the error return it by value, moved out (`std::move(r).unwrap()`
///   is a `V`, not a `V&&`), so `for (auto x : make().unwrap())` and `const auto& v = make().unwrap();` are safe.
///   On an lvalue they return references, as before.
/// - A niche result (`snicholls::result_niche`) reads a moved-from V that equals the niche value as an error.
///
/// \status
/// Experimental, pre-1.0. Design: [RESULT_PLAN.md](RESULT_PLAN.md).
///
/// \tparam V the value type: an object type (not an array, reference or cv-qualified type), or `void`.
/// \tparam E the error type: an object type (not an array, reference or cv-qualified type).
template <class V, class E>
class result : private result_impl::result_storage<std::conditional_t<std::is_void_v<V>, result_impl::void_value, V>, E>
{
    static_assert(result_impl::valid_value<V>,
                  "snicholls::result<V, E>: V must be void or a non-array, non-reference, unqualified object type");
    static_assert(result_impl::valid_error<E>,
                  "snicholls::result<V, E>: E must be a non-array, non-reference, unqualified object type");
    static_assert(!std::is_same_v<std::remove_cv_t<V>, std::in_place_t> && !std::is_same_v<V, err_in_place_t>,
                  "snicholls::result<V, E>: V must not be a tag type");

    static constexpr bool is_void = std::is_void_v<V>;
    using stored = std::conditional_t<is_void, result_impl::void_value, V>;
    using base = result_impl::result_storage<stored, E>;

    template <class, class>
    friend class result;

public:
    /// \brief The value type V.
    using value_type = V;
    /// \brief The error type E.
    using error_type = E;
    /// \brief This result with another value type.
    /// \tparam U the value type.
    template <class U>
    using rebind = result<U, E>;

    // ---- construction ----------------------------------------------------------------------------------------------

    /// \brief Constructs a success from `snicholls::ok(u)`; explicit when U does not convert to V.
    /// \param o the success tag.
    template <class U>
        requires(!is_void && std::is_constructible_v<stored, U>)
    constexpr explicit(!std::is_convertible_v<U, stored>)
        result(ok_t<U> o) noexcept(std::is_nothrow_constructible_v<stored, U>)
        : base(std::in_place, std::move(o.value))
    {
    }

    /// \brief Constructs the success of a `result<void, E>` from `snicholls::ok()`.
    /// \param o the success tag.
    constexpr result(ok_t<void> o) noexcept
        requires is_void
        : base(std::in_place)
    {
        (void)o;
    }

    /// \brief Constructs an error from `snicholls::err(g)`; explicit when G does not convert to E.
    /// \param e the error tag.
    template <class G>
        requires std::is_constructible_v<E, G>
    constexpr explicit(!std::is_convertible_v<G, E>) result(err_t<G> e) noexcept(std::is_nothrow_constructible_v<E, G>)
        : base(err_in_place, std::move(e.error))
    {
    }

    /// \brief Constructs the value in place.
    /// \param a the constructor arguments of V.
    template <class... A>
        requires std::is_constructible_v<stored, A...>
    constexpr explicit result(std::in_place_t, A&&... a) noexcept(std::is_nothrow_constructible_v<stored, A...>)
        : base(std::in_place, std::forward<A>(a)...)
    {
    }

    /// \brief Constructs the error in place.
    /// \param a the constructor arguments of E.
    template <class... A>
        requires std::is_constructible_v<E, A...>
    constexpr explicit result(err_in_place_t, A&&... a) noexcept(std::is_nothrow_constructible_v<E, A...>)
        : base(err_in_place, std::forward<A>(a)...)
    {
    }

    /// \brief Converts from a result of other types; explicit when either does not convert implicitly.
    ///
    /// Not available from a plain error into a `snicholls::traced` one (the location would be this header's): use
    /// `trace()`, `context()` or `SNICHOLLS_TRY`, which record the caller's location.
    /// \param o the other result.
    template <class U, class G>
        requires(!std::is_same_v<result<U, G>, result> && (std::is_void_v<U> == is_void) &&
                 (is_void || std::is_constructible_v<stored, const U&>) && std::is_constructible_v<E, const G&> &&
                 (!is_traced_v<E> || is_traced_v<G>))
    constexpr explicit(!(is_void || std::is_convertible_v<const U&, stored>) || !std::is_convertible_v<const G&, E>)
        result(const result<U, G>& o) noexcept(
            (is_void || std::is_nothrow_constructible_v<stored, const U&>) && std::is_nothrow_constructible_v<E, const G&>)
        : base(result_impl::convert_t{}, o.is_ok(), [&]() -> decltype(auto) { return o.stored_value(); },
               [&]() -> decltype(auto) { return o.err_ref(); })
    {
    }

    /// \brief Converts from an rvalue result of other types; explicit when either does not convert implicitly.
    /// \param o the other result.
    template <class U, class G>
        requires(!std::is_same_v<result<U, G>, result> && (std::is_void_v<U> == is_void) &&
                 (is_void || std::is_constructible_v<stored, U>) && std::is_constructible_v<E, G> &&
                 (!is_traced_v<E> || is_traced_v<G>))
    constexpr explicit(!(is_void || std::is_convertible_v<U, stored>) || !std::is_convertible_v<G, E>)
        result(result<U, G>&& o) noexcept((is_void || std::is_nothrow_constructible_v<stored, U>) &&
                                          std::is_nothrow_constructible_v<E, G>)
        : base(result_impl::convert_t{}, o.is_ok(), [&]() -> decltype(auto) { return std::move(o).stored_value(); },
               [&]() -> decltype(auto) { return std::move(o).err_ref(); })
    {
    }

#if defined(__cpp_lib_expected)
    /// \brief Converts from `std::expected<V, E>` (C++23).
    /// \param x the expected.
    template <class U = V>
    constexpr explicit result(const std::expected<U, E>& x)
        : base(result_impl::convert_t{}, x.has_value(), [&]() -> decltype(auto) { return expected_value(x); },
               [&]() -> decltype(auto) { return x.error(); })
    {
    }
#endif

#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(push)
#pragma warning(disable : 4702)   // unreachable only for a type whose construction always throws, which MSVC inlines
#endif
    /// \brief Assigns a success.
    /// \param o the success tag.
    /// \return `*this`.
    template <class U>
        requires(!is_void && std::is_constructible_v<stored, U> && std::is_assignable_v<stored&, U>)
    constexpr result& operator=(ok_t<U> o)
    {
        base::reinit_value(std::move(o.value));
        return *this;
    }

    /// \brief Assigns an error.
    /// \param e the error tag.
    /// \return `*this`.
    template <class G>
        requires(std::is_constructible_v<E, G> && std::is_assignable_v<E&, G>)
    constexpr result& operator=(err_t<G> e)
    {
        base::reinit_error(std::move(e.error));
        return *this;
    }
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(pop)
#endif

    // ---- observers -------------------------------------------------------------------------------------------------

    /// \brief Rust's `is_ok`.
    /// \return true for a success.
    [[nodiscard]] constexpr bool is_ok() const noexcept { return base::has_value(); }

    /// \brief Rust's `is_err`.
    /// \return true for an error.
    [[nodiscard]] constexpr bool is_err() const noexcept { return !base::has_value(); }

    /// \brief True for a success.
    /// \return `is_ok()`.
    [[nodiscard]] constexpr explicit operator bool() const noexcept { return base::has_value(); }

    /// \brief Rust's `is_ok_and`: a success whose value satisfies `f`.
    /// \param f a predicate on `const V&` (no argument for void).
    /// \return `is_ok() && f(value)`.
    template <class F>
    [[nodiscard]] constexpr bool is_ok_and(F&& f) const& noexcept(noexcept(call_value(std::declval<F>(), *this)))
    {
        return is_ok() && static_cast<bool>(call_value(std::forward<F>(f), *this));
    }

    /// \brief Rust's `is_err_and`: an error that satisfies `f`.
    /// \param f a predicate on `const E&`.
    /// \return `is_err() && f(error)`.
    template <class F>
    [[nodiscard]] constexpr bool is_err_and(F&& f) const& noexcept(std::is_nothrow_invocable_v<F, const E&>)
    {
        return is_err() && static_cast<bool>(std::invoke(std::forward<F>(f), base::err()));
    }

    /// \name Rust's ok(): the value as a std::optional
    ///@{
    /// \brief Rust's `ok`: the value, or `std::nullopt` for an error. Not for `result<void, E>`.
    /// \return the value as a `std::optional`.
    [[nodiscard]] constexpr std::optional<stored> ok() const& noexcept(std::is_nothrow_copy_constructible_v<stored>)
        requires(!is_void && std::is_copy_constructible_v<stored>)
    {
        return is_ok() ? std::optional<stored>(base::val()) : std::nullopt;
    }
    [[nodiscard]] constexpr std::optional<stored> ok() && noexcept(std::is_nothrow_move_constructible_v<stored>)
        requires(!is_void)
    {
        return is_ok() ? std::optional<stored>(std::move(*this).stored_value()) : std::nullopt;
    }
    ///@}

    /// \name Rust's err(): the error as a std::optional
    ///@{
    /// \brief Rust's `err`: the error, or `std::nullopt` for a success.
    /// \return the error as a `std::optional`.
    [[nodiscard]] constexpr std::optional<E> err() const& noexcept(std::is_nothrow_copy_constructible_v<E>)
        requires std::is_copy_constructible_v<E>
    {
        return is_err() ? std::optional<E>(base::err()) : std::nullopt;
    }
    [[nodiscard]] constexpr std::optional<E> err() && noexcept(std::is_nothrow_move_constructible_v<E>)
    {
        return is_err() ? std::optional<E>(std::move(*this).err_ref()) : std::nullopt;
    }
    ///@}

    /// \name Unchecked access (Rust's unwrap_unchecked)
    ///@{
    /// \brief The value, unchecked. \pre `is_ok()`. Not for `result<void, E>` (use `unwrap_unchecked()`).
    /// \return the value.
    [[nodiscard]] constexpr stored& operator*() & noexcept
        requires(!is_void)
    {
        assert(is_ok());
        return base::val();
    }
    [[nodiscard]] constexpr const stored& operator*() const& noexcept
        requires(!is_void)
    {
        assert(is_ok());
        return base::val();
    }
    [[nodiscard]] constexpr stored operator*() && noexcept(std::is_nothrow_move_constructible_v<stored>)
        requires(!is_void)
    {
        assert(is_ok());
        return std::move(*this).stored_value();
    }
    [[nodiscard]] constexpr stored operator*() const&& noexcept(std::is_nothrow_copy_constructible_v<stored>)
        requires(!is_void)
    {
        assert(is_ok());
        return base::val();
    }
    ///@}

    /// \name Member access, unchecked
    ///@{
    /// \brief Member access to the value, unchecked. \pre `is_ok()`.
    /// \return a pointer to the value.
    [[nodiscard]] constexpr stored* operator->() noexcept
        requires(!is_void)
    {
        assert(is_ok());
        return std::addressof(base::val());
    }
    [[nodiscard]] constexpr const stored* operator->() const noexcept
        requires(!is_void)
    {
        assert(is_ok());
        return std::addressof(base::val());
    }
    ///@}

    /// \name Rust's unwrap_unchecked
    ///@{
    /// \brief The value, unchecked (void for `result<void, E>`). \pre `is_ok()`.
    /// \return the value.
    [[nodiscard]] constexpr decltype(auto) unwrap_unchecked() & noexcept { return checked_ok(*this); }
    [[nodiscard]] constexpr decltype(auto) unwrap_unchecked() const& noexcept { return checked_ok(*this); }
    [[nodiscard]] constexpr decltype(auto) unwrap_unchecked() && noexcept(std::is_nothrow_move_constructible_v<stored>)
    {
        return checked_ok(std::move(*this));
    }
    ///@}

    /// \name Rust's unwrap_err_unchecked
    ///@{
    /// \brief The error, unchecked. \pre `is_err()`.
    /// \return the error.
    [[nodiscard]] constexpr E& unwrap_err_unchecked() & noexcept
    {
        assert(is_err());
        return base::err();
    }
    [[nodiscard]] constexpr const E& unwrap_err_unchecked() const& noexcept
    {
        assert(is_err());
        return base::err();
    }
    [[nodiscard]] constexpr E unwrap_err_unchecked() && noexcept(std::is_nothrow_move_constructible_v<E>)
    {
        assert(is_err());
        return std::move(*this).err_ref();
    }
    ///@}

    /// \name Rust's iter(): the value as a range of zero or one element
    ///@{
    /// \brief Rust's `iter`: a span over the value (one element) or nothing (an error). Not for void.
    /// \return the span.
    [[nodiscard]] constexpr std::span<const stored> iter() const& noexcept
        requires(!is_void)
    {
        return is_ok() ? std::span<const stored>(std::addressof(base::val()), 1) : std::span<const stored>();
    }
    /// \brief Rust's `iter_mut`: a mutable span over the value (one element) or nothing.
    /// \return the span.
    [[nodiscard]] constexpr std::span<stored> iter_mut() & noexcept
        requires(!is_void)
    {
        return is_ok() ? std::span<stored>(std::addressof(base::val()), 1) : std::span<stored>();
    }
    /// \brief Deleted: a span into a temporary result would dangle (a range-for does not extend its life).
    void iter() const&& = delete;
    /// \brief Deleted: a span into a temporary result would dangle.
    void iter_mut() && = delete;
    ///@}

    // ---- unwrap family ---------------------------------------------------------------------------------------------

    /// \name Rust's unwrap
    ///@{
    /// \brief Rust's `unwrap`: the value (moved out of an rvalue), or the policy's panic on an error.
    /// \return the value (void for `result<void, E>`).
    /// \throws snicholls::bad_result_access<E> on an error (`SNICHOLLS_RESULT_THROW=1`).
    constexpr decltype(auto) unwrap() & noexcept(result_impl::unwrap_noexcept)
    {
        if (is_err()) [[unlikely]]
            result_impl::unwrap_failed("called unwrap() on an err", base::err());
        return checked_ok(*this);
    }
    constexpr decltype(auto) unwrap() const& noexcept(result_impl::unwrap_noexcept)
    {
        if (is_err()) [[unlikely]]
            result_impl::unwrap_failed("called unwrap() on an err", base::err());
        return checked_ok(*this);
    }
    constexpr decltype(auto) unwrap() && noexcept(result_impl::unwrap_noexcept &&
                                                   std::is_nothrow_move_constructible_v<stored>)
    {
        if (is_err()) [[unlikely]]
            result_impl::unwrap_failed("called unwrap() on an err", base::err());
        return checked_ok(std::move(*this));
    }
    ///@}

    /// \name Rust's expect
    ///@{
    /// \brief Rust's `expect`: as `unwrap`, with `msg` leading the panic message ("msg: error text").
    /// \param msg what was expected.
    /// \return the value (void for `result<void, E>`).
    /// \throws snicholls::bad_result_access<E> on an error (`SNICHOLLS_RESULT_THROW=1`).
    constexpr decltype(auto) expect(std::string_view msg) & noexcept(result_impl::unwrap_noexcept)
    {
        if (is_err()) [[unlikely]]
            result_impl::unwrap_failed(msg, base::err());
        return checked_ok(*this);
    }
    constexpr decltype(auto) expect(std::string_view msg) const& noexcept(result_impl::unwrap_noexcept)
    {
        if (is_err()) [[unlikely]]
            result_impl::unwrap_failed(msg, base::err());
        return checked_ok(*this);
    }
    constexpr decltype(auto) expect(std::string_view msg) && noexcept(result_impl::unwrap_noexcept &&
                                                                       std::is_nothrow_move_constructible_v<stored>)
    {
        if (is_err()) [[unlikely]]
            result_impl::unwrap_failed(msg, base::err());
        return checked_ok(std::move(*this));
    }
    ///@}

    /// \name Rust's unwrap_err
    ///@{
    /// \brief Rust's `unwrap_err`: the error, or the policy's panic on a success.
    /// \return the error.
    /// \throws snicholls::bad_result_access<void> on a success (`SNICHOLLS_RESULT_THROW=1`).
    constexpr E& unwrap_err() & noexcept(result_impl::unwrap_noexcept)
    {
        if (is_ok()) [[unlikely]]
            result_impl::unwrap_err_failed("called unwrap_err() on an ok");
        return base::err();
    }
    constexpr const E& unwrap_err() const& noexcept(result_impl::unwrap_noexcept)
    {
        if (is_ok()) [[unlikely]]
            result_impl::unwrap_err_failed("called unwrap_err() on an ok");
        return base::err();
    }
    constexpr E unwrap_err() && noexcept(result_impl::unwrap_noexcept && std::is_nothrow_move_constructible_v<E>)
    {
        if (is_ok()) [[unlikely]]
            result_impl::unwrap_err_failed("called unwrap_err() on an ok");
        return std::move(*this).err_ref();
    }
    ///@}

    /// \name Rust's expect_err
    ///@{
    /// \brief Rust's `expect_err`: as `unwrap_err`, with `msg` as the panic message.
    /// \param msg what was expected.
    /// \return the error.
    /// \throws snicholls::bad_result_access<void> on a success (`SNICHOLLS_RESULT_THROW=1`).
    constexpr E& expect_err(std::string_view msg) & noexcept(result_impl::unwrap_noexcept)
    {
        if (is_ok()) [[unlikely]]
            result_impl::unwrap_err_failed(msg);
        return base::err();
    }
    constexpr const E& expect_err(std::string_view msg) const& noexcept(result_impl::unwrap_noexcept)
    {
        if (is_ok()) [[unlikely]]
            result_impl::unwrap_err_failed(msg);
        return base::err();
    }
    constexpr E expect_err(std::string_view msg) && noexcept(result_impl::unwrap_noexcept &&
                                                              std::is_nothrow_move_constructible_v<E>)
    {
        if (is_ok()) [[unlikely]]
            result_impl::unwrap_err_failed(msg);
        return std::move(*this).err_ref();
    }
    ///@}

    /// \name Rust's unwrap_or
    ///@{
    /// \brief Rust's `unwrap_or`: the value, or `fallback` for an error. Not for void.
    /// \param fallback the value to return for an error.
    /// \return the value or the fallback.
    template <class U>
        requires(!is_void && std::is_constructible_v<stored, U>)
    [[nodiscard]] constexpr stored unwrap_or(U&& fallback) const& noexcept(
        std::is_nothrow_copy_constructible_v<stored> && std::is_nothrow_constructible_v<stored, U>)
    {
        return is_ok() ? base::val() : static_cast<stored>(std::forward<U>(fallback));
    }
    template <class U>
        requires(!is_void && std::is_constructible_v<stored, U>)
    [[nodiscard]] constexpr stored unwrap_or(U&& fallback) && noexcept(
        std::is_nothrow_move_constructible_v<stored> && std::is_nothrow_constructible_v<stored, U>)
    {
        return is_ok() ? std::move(*this).stored_value() : static_cast<stored>(std::forward<U>(fallback));
    }
    ///@}

    /// \name Rust's unwrap_or_else
    ///@{
    /// \brief Rust's `unwrap_or_else`: the value, or `f(error)` for an error. Not for void.
    /// \param f computes the fallback from the error.
    /// \return the value or `f(error)`.
    template <class F>
        requires(!is_void)
    [[nodiscard]] constexpr stored unwrap_or_else(F&& f) const& noexcept(
        std::is_nothrow_copy_constructible_v<stored> && std::is_nothrow_invocable_v<F, const E&> &&
        std::is_nothrow_constructible_v<stored, std::invoke_result_t<F, const E&>>)
    {
        if (is_ok())
            return base::val();
        return static_cast<stored>(std::invoke(std::forward<F>(f), base::err()));
    }
    template <class F>
        requires(!is_void)
    [[nodiscard]] constexpr stored unwrap_or_else(F&& f) && noexcept(
        std::is_nothrow_move_constructible_v<stored> && std::is_nothrow_invocable_v<F, E&&> &&
        std::is_nothrow_constructible_v<stored, std::invoke_result_t<F, E&&>>)
    {
        if (is_ok())
            return std::move(*this).stored_value();
        return static_cast<stored>(std::invoke(std::forward<F>(f), std::move(*this).err_ref()));
    }
    ///@}

    /// \name Rust's unwrap_or_default
    ///@{
    /// \brief Rust's `unwrap_or_default`: the value, or `V{}` for an error. Not for void.
    /// \return the value or `V{}`.
    [[nodiscard]] constexpr stored unwrap_or_default() const& noexcept(
        std::is_nothrow_copy_constructible_v<stored> && std::is_nothrow_default_constructible_v<stored>)
        requires(!is_void && std::is_default_constructible_v<stored>)
    {
        return is_ok() ? base::val() : stored{};
    }
    [[nodiscard]] constexpr stored unwrap_or_default() && noexcept(
        std::is_nothrow_move_constructible_v<stored> && std::is_nothrow_default_constructible_v<stored>)
        requires(!is_void && std::is_default_constructible_v<stored>)
    {
        return is_ok() ? std::move(*this).stored_value() : stored{};
    }
    ///@}

    // ---- monadic operations ----------------------------------------------------------------------------------------

    /// \name Rust's map
    ///@{
    /// \brief Rust's `map`: `ok(f(value))`, or the error unchanged. For void, `f()` takes no argument.
    /// \param f the transformation; may return void.
    /// \return `snicholls::result<U, E>` where U is what `f` returns.
    template <class F>
    [[nodiscard]] constexpr auto map(F&& f) & noexcept(map_noexcept<result&, F>())
    {
        return map_impl(*this, std::forward<F>(f));
    }
    template <class F>
    [[nodiscard]] constexpr auto map(F&& f) const& noexcept(map_noexcept<const result&, F>())
    {
        return map_impl(*this, std::forward<F>(f));
    }
    template <class F>
    [[nodiscard]] constexpr auto map(F&& f) && noexcept(map_noexcept<result&&, F>())
    {
        return map_impl(std::move(*this), std::forward<F>(f));
    }
    ///@}

    /// \name Rust's map_err
    ///@{
    /// \brief Rust's `map_err`: the value unchanged, or `err(f(error))`.
    /// \param f the error transformation.
    /// \return `snicholls::result<V, G>` where G is what `f` returns.
    template <class F>
    [[nodiscard]] constexpr auto map_err(F&& f) & noexcept(map_err_noexcept<result&, F>())
    {
        return map_err_impl(*this, std::forward<F>(f));
    }
    template <class F>
    [[nodiscard]] constexpr auto map_err(F&& f) const& noexcept(map_err_noexcept<const result&, F>())
    {
        return map_err_impl(*this, std::forward<F>(f));
    }
    template <class F>
    [[nodiscard]] constexpr auto map_err(F&& f) && noexcept(map_err_noexcept<result&&, F>())
    {
        return map_err_impl(std::move(*this), std::forward<F>(f));
    }
    ///@}

    /// \name Rust's map_or
    ///@{
    /// \brief Rust's `map_or`: `f(value)`, or `fallback` for an error.
    /// \param fallback the result for an error.
    /// \param f the transformation.
    /// \return `f(value)` or `fallback`, as U.
    template <class U, class F>
    [[nodiscard]] constexpr U map_or(U fallback, F&& f) const& noexcept(
        noexcept(call_value(std::declval<F>(), std::declval<const result&>())) &&
        std::is_nothrow_constructible_v<U, call_t<const result&, F>> && std::is_nothrow_move_constructible_v<U>)
    {
        if (is_ok())
            return static_cast<U>(call_value(std::forward<F>(f), *this));
        return fallback;
    }
    template <class U, class F>
    [[nodiscard]] constexpr U map_or(U fallback, F&& f) && noexcept(
        noexcept(call_value(std::declval<F>(), std::declval<result&&>())) &&
        std::is_nothrow_constructible_v<U, call_t<result&&, F>> && std::is_nothrow_move_constructible_v<U>)
    {
        if (is_ok())
            return static_cast<U>(call_value(std::forward<F>(f), std::move(*this)));
        return fallback;
    }
    ///@}

    /// \name Rust's map_or_else
    ///@{
    /// \brief Rust's `map_or_else`: `f(value)`, or `fallback(error)` for an error.
    /// \param fallback computes the result from the error.
    /// \param f the transformation.
    /// \return `f(value)` or `fallback(error)`.
    template <class D, class F>
    [[nodiscard]] constexpr auto map_or_else(D&& fallback, F&& f) const& noexcept(
        noexcept(call_value(std::declval<F>(), std::declval<const result&>())) &&
        std::is_nothrow_invocable_v<D, const E&> &&
        std::is_nothrow_constructible_v<std::remove_cvref_t<call_t<const result&, F>>, call_t<const result&, F>> &&
        std::is_nothrow_constructible_v<std::remove_cvref_t<call_t<const result&, F>>, std::invoke_result_t<D, const E&>>)
    {
        using U = std::remove_cvref_t<decltype(call_value(std::forward<F>(f), *this))>;
        if (is_ok())
            return static_cast<U>(call_value(std::forward<F>(f), *this));
        return static_cast<U>(std::invoke(std::forward<D>(fallback), base::err()));
    }
    template <class D, class F>
    [[nodiscard]] constexpr auto map_or_else(D&& fallback, F&& f) && noexcept(
        noexcept(call_value(std::declval<F>(), std::declval<result&&>())) && std::is_nothrow_invocable_v<D, E&&> &&
        std::is_nothrow_constructible_v<std::remove_cvref_t<call_t<result&&, F>>, call_t<result&&, F>> &&
        std::is_nothrow_constructible_v<std::remove_cvref_t<call_t<result&&, F>>, std::invoke_result_t<D, E&&>>)
    {
        using U = std::remove_cvref_t<decltype(call_value(std::forward<F>(f), std::move(*this)))>;
        if (is_ok())
            return static_cast<U>(call_value(std::forward<F>(f), std::move(*this)));
        return static_cast<U>(std::invoke(std::forward<D>(fallback), std::move(*this).err_ref()));
    }
    ///@}

    /// \name Rust's and_then
    ///@{
    /// \brief Rust's `and_then`: `f(value)` (a `snicholls::result<U, E>`), or the error unchanged.
    /// \param f the next step; returns `snicholls::result<U, E>` with the same E.
    /// \return what `f` returns, or the error.
    template <class F>
    [[nodiscard]] constexpr auto and_then(F&& f) & noexcept(and_then_noexcept<result&, F>())
    {
        return and_then_impl(*this, std::forward<F>(f));
    }
    template <class F>
    [[nodiscard]] constexpr auto and_then(F&& f) const& noexcept(and_then_noexcept<const result&, F>())
    {
        return and_then_impl(*this, std::forward<F>(f));
    }
    template <class F>
    [[nodiscard]] constexpr auto and_then(F&& f) && noexcept(and_then_noexcept<result&&, F>())
    {
        return and_then_impl(std::move(*this), std::forward<F>(f));
    }
    ///@}

    /// \name Rust's or_else
    ///@{
    /// \brief Rust's `or_else`: the value unchanged, or `f(error)` (a `snicholls::result<V, G>`).
    /// \param f the recovery; returns `snicholls::result<V, G>` with the same V.
    /// \return the value, or what `f` returns.
    template <class F>
    [[nodiscard]] constexpr auto or_else(F&& f) & noexcept(or_else_noexcept<result&, F>())
    {
        return or_else_impl(*this, std::forward<F>(f));
    }
    template <class F>
    [[nodiscard]] constexpr auto or_else(F&& f) const& noexcept(or_else_noexcept<const result&, F>())
    {
        return or_else_impl(*this, std::forward<F>(f));
    }
    template <class F>
    [[nodiscard]] constexpr auto or_else(F&& f) && noexcept(or_else_noexcept<result&&, F>())
    {
        return or_else_impl(std::move(*this), std::forward<F>(f));
    }
    ///@}

    /// \name Rust's and
    ///@{
    /// \brief Rust's `and`: `next` for a success, else this error.
    /// \param next the result to return for a success.
    /// \return `next` or the error.
    template <class U>
    [[nodiscard]] constexpr result<U, E> and_(result<U, E> next) const& noexcept(
        std::is_nothrow_move_constructible_v<result<U, E>> && std::is_nothrow_copy_constructible_v<E>)
    {
        if (is_ok())
            return next;
        return result<U, E>(err_in_place, base::err());
    }
    template <class U>
    [[nodiscard]] constexpr result<U, E> and_(result<U, E> next) && noexcept(
        std::is_nothrow_move_constructible_v<result<U, E>> && std::is_nothrow_move_constructible_v<E>)
    {
        if (is_ok())
            return next;
        return result<U, E>(err_in_place, std::move(*this).err_ref());
    }
    ///@}

    /// \name Rust's or
    ///@{
    /// \brief Rust's `or`: this success, else `other`.
    /// \param other the result to return for an error.
    /// \return the success or `other`.
    template <class G>
    [[nodiscard]] constexpr result<V, G> or_(result<V, G> other) const& noexcept(
        std::is_nothrow_move_constructible_v<result<V, G>> && nothrow_ok_like<const result&>)
    {
        if (is_ok())
            return ok_like<G>(*this);
        return other;
    }
    template <class G>
    [[nodiscard]] constexpr result<V, G> or_(result<V, G> other) && noexcept(
        std::is_nothrow_move_constructible_v<result<V, G>> && nothrow_ok_like<result&&>)
    {
        if (is_ok())
            return ok_like<G>(std::move(*this));
        return other;
    }
    ///@}

    /// \name Rust's inspect
    ///@{
    /// \brief Rust's `inspect`: calls `f(const V&)` (`f()` for void) on a success; returns this result.
    /// \param f the observer.
    /// \return `*this` (moved for an rvalue).
    template <class F>
    constexpr result& inspect(F&& f) & noexcept(noexcept(call_value(std::declval<F>(), std::as_const(*this))))
    {
        if (is_ok())
            call_value(std::forward<F>(f), std::as_const(*this));
        return *this;
    }
    template <class F>
    constexpr const result& inspect(F&& f) const& noexcept(noexcept(call_value(std::declval<F>(), *this)))
    {
        if (is_ok())
            call_value(std::forward<F>(f), *this);
        return *this;
    }
    template <class F>
    [[nodiscard]] constexpr result inspect(F&& f) && noexcept(
        noexcept(call_value(std::declval<F>(), std::as_const(*this))) && std::is_nothrow_move_constructible_v<result>)
    {
        if (is_ok())
            call_value(std::forward<F>(f), std::as_const(*this));
        return std::move(*this);
    }
    ///@}

    /// \name Rust's inspect_err
    ///@{
    /// \brief Rust's `inspect_err`: calls `f(const E&)` on an error; returns this result.
    /// \param f the observer.
    /// \return `*this` (moved for an rvalue).
    template <class F>
    constexpr result& inspect_err(F&& f) & noexcept(std::is_nothrow_invocable_v<F, const E&>)
    {
        if (is_err())
            std::invoke(std::forward<F>(f), std::as_const(base::err()));
        return *this;
    }
    template <class F>
    constexpr const result& inspect_err(F&& f) const& noexcept(std::is_nothrow_invocable_v<F, const E&>)
    {
        if (is_err())
            std::invoke(std::forward<F>(f), base::err());
        return *this;
    }
    template <class F>
    [[nodiscard]] constexpr result inspect_err(F&& f) && noexcept(
        std::is_nothrow_invocable_v<F, const E&> && std::is_nothrow_move_constructible_v<result>)
    {
        if (is_err())
            std::invoke(std::forward<F>(f), std::as_const(base::err()));
        return std::move(*this);
    }
    ///@}

    /// \name Rust's flatten
    ///@{
    /// \brief Rust's `flatten`: `result<result<U, E>, E>` to `result<U, E>`.
    /// \return the inner result, or the outer error.
    [[nodiscard]] constexpr auto flatten() const& noexcept(
        std::is_nothrow_copy_constructible_v<stored> && std::is_nothrow_copy_constructible_v<E>)
        requires(result_impl::is_result_v<stored>)
    {
        using inner = stored;
        static_assert(std::is_same_v<typename inner::error_type, E>, "snicholls::result::flatten: the error types differ");
        if (is_ok())
            return inner(base::val());
        return inner(err_in_place, base::err());
    }
    [[nodiscard]] constexpr auto flatten() && noexcept(
        std::is_nothrow_move_constructible_v<stored> && std::is_nothrow_move_constructible_v<E>)
        requires(result_impl::is_result_v<stored>)
    {
        using inner = stored;
        static_assert(std::is_same_v<typename inner::error_type, E>, "snicholls::result::flatten: the error types differ");
        if (is_ok())
            return inner(std::move(*this).stored_value());
        return inner(err_in_place, std::move(*this).err_ref());
    }
    ///@}

    /// \name Rust's transpose
    ///@{
    /// \brief Rust's `transpose`: `result<std::optional<U>, E>` to `std::optional<result<U, E>>`.
    /// \return `nullopt` for `ok(nullopt)`, else the value or error wrapped.
    [[nodiscard]] constexpr auto transpose() const&
        requires(!is_void && requires { typename stored::value_type; } &&
                 std::is_same_v<stored, std::optional<typename stored::value_type>>)
    {
        using U = typename stored::value_type;
        using out = std::optional<result<U, E>>;
        if (is_err())
            return out(result<U, E>(err_in_place, base::err()));
        if (!base::val().has_value())
            return out();
        return out(result<U, E>(std::in_place, *base::val()));
    }
    [[nodiscard]] constexpr auto transpose() &&
        requires(!is_void && requires { typename stored::value_type; } &&
                 std::is_same_v<stored, std::optional<typename stored::value_type>>)
    {
        using U = typename stored::value_type;
        using out = std::optional<result<U, E>>;
        if (is_err())
            return out(result<U, E>(err_in_place, std::move(*this).err_ref()));
        if (!base::val().has_value())
            return out();
        return out(result<U, E>(std::in_place, std::move(*base::val())));
    }
    ///@}

    // ---- exceptions --------------------------------------------------------------------------------------------

    /// \brief Throws the error as its own exception (`snicholls::throw_error`). \pre `is_err()`.
    /// \throws the error's exception; `snicholls::bad_result_access<void>` if called on a success.
    [[noreturn]] void rethrow() const
    {
        if (is_ok()) [[unlikely]]
            result_impl::unwrap_err_failed("called rethrow() on an ok");
        throw_error(base::err());
    }

    /// \brief Throws the error as its own exception if there is one (`rethrow_if_err` in other libraries).
    /// \throws the error's exception (`snicholls::throw_error`).
    void rethrow_if_err() const
    {
        if (is_err()) [[unlikely]]
            throw_error(base::err());
    }

    /// \name The value, or the original exception
    ///@{
    /// \brief The value; an error is thrown as its own exception (`snicholls::throw_error`), not as bad_result_access.
    /// \return the value (void for `result<void, E>`).
    /// \throws the error's exception.
    constexpr decltype(auto) value_or_throw() &
    {
        if (is_err()) [[unlikely]]
            throw_error(base::err());
        return checked_ok(*this);
    }
    constexpr decltype(auto) value_or_throw() const&
    {
        if (is_err()) [[unlikely]]
            throw_error(base::err());
        return checked_ok(*this);
    }
    constexpr decltype(auto) value_or_throw() &&
    {
        if (is_err()) [[unlikely]]
            throw_error(base::err());
        return checked_ok(std::move(*this));
    }
    ///@}

    /// \name unwrap_or_terminate: the value, or print the error and terminate, whatever the policy
    ///@{
    /// \brief The value; an error is printed ("called unwrap_or_terminate() on an err: <error text>") and
    ///        `std::terminate` is called, under either `SNICHOLLS_RESULT_THROW` policy. The counterpart of
    ///        `value_or_throw()`, for the call sites where a failure is a bug and must never become an exception.
    ///        Like every `&&` overload, the rvalue one returns the value by value.
    ///
    /// \snippet result_demo.cpp terminate
    /// \return the value (void for `result<void, E>`).
    [[nodiscard]] constexpr decltype(auto) unwrap_or_terminate() & noexcept
    {
        if (is_err()) [[unlikely]]
            result_impl::terminate_failed("called unwrap_or_terminate() on an err", base::err());
        return checked_ok(*this);
    }
    [[nodiscard]] constexpr decltype(auto) unwrap_or_terminate() const& noexcept
    {
        if (is_err()) [[unlikely]]
            result_impl::terminate_failed("called unwrap_or_terminate() on an err", base::err());
        return checked_ok(*this);
    }
    [[nodiscard]] constexpr decltype(auto) unwrap_or_terminate() && noexcept
    {
        if (is_err()) [[unlikely]]
            result_impl::terminate_failed("called unwrap_or_terminate() on an err", base::err());
        return checked_ok(std::move(*this));
    }
    ///@}

    /// \brief Explicitly discards the result (`ignore_err` in other libraries): silences `[[nodiscard]]`.
    constexpr void ignore_err() const noexcept {}

    // ---- traced errors (snicholls/core/traced.hpp) -----------------------------------------------------------------------

    /// \name anyhow's context: add a context frame
    ///@{
    /// \brief Adds a context frame "message" at the caller's location to an error (anyhow's `context`).
    ///
    /// A plain E becomes `snicholls::traced<E>` first, located here. Does nothing (no formatting, no allocation) for a
    /// success. Needs `snicholls/core/traced.hpp`.
    /// \param fmt a `std::format` string; its location is recorded.
    /// \param args the format arguments.
    /// \return `snicholls::result<V, snicholls::traced_t<E>>`.
    /// \throws std::bad_alloc; std::format_error when an argument is invalid for its format at run time, e.g. a
    ///         negative dynamic width (error path only).
    template <class... A>
    [[nodiscard]] constexpr result<V, traced_t<E>> context(
        result_impl::located_format<std::type_identity_t<A>...> fmt, A&&... args) &&
    {
        if (is_ok())
            return ok_like<traced_t<E>>(std::move(*this));
        auto t = make_traced(std::move(*this).err_ref(), fmt.where);
        t.add_context(std::format(fmt.fmt, std::forward<A>(args)...), fmt.where);
        return result<V, traced_t<E>>(err_in_place, std::move(t));
    }
    template <class... A>
    [[nodiscard]] constexpr result<V, traced_t<E>> context(
        result_impl::located_format<std::type_identity_t<A>...> fmt, A&&... args) const&
    {
        return result(*this).context(fmt, std::forward<A>(args)...);
    }
    ///@}

    /// \brief Records a key-value field on an error (traced; the value is formatted, secrets masked).
    /// \param key the field name.
    /// \param value the value (string-like, formattable, or a secret with `reveal()`, which is masked).
    /// \param where the caller's location (if E becomes traced here).
    /// \return `snicholls::result<V, snicholls::traced_t<E>>`.
    /// \throws std::bad_alloc (error path only).
    template <class T>
    [[nodiscard]] constexpr result<V, traced_t<E>> with_field(
        std::string_view key, const T& value, source_location where = source_location::current()) &&
    {
        if (is_ok())
            return ok_like<traced_t<E>>(std::move(*this));
        auto t = make_traced(std::move(*this).err_ref(), where);
        t.add_field(key, value);
        return result<V, traced_t<E>>(err_in_place, std::move(t));
    }

    /// \brief Captures a stack trace into an error (traced), through the hook installed with
    ///        `snicholls::set_stack_trace_hook` (empty if none). Costs nothing for a success.
    /// \param where the caller's location (if E becomes traced here).
    /// \return `snicholls::result<V, snicholls::traced_t<E>>`.
    /// \throws std::bad_alloc (error path only).
    [[nodiscard]] constexpr result<V, traced_t<E>> with_stack(
        source_location where = source_location::current()) &&
    {
        if (is_ok())
            return ok_like<traced_t<E>>(std::move(*this));
        auto t = make_traced(std::move(*this).err_ref(), where);
        t.capture_stack();
        return result<V, traced_t<E>>(err_in_place, std::move(t));
    }

    /// \brief Makes the error a `snicholls::traced<E>` located at the caller (identity if already traced).
    /// \param where the caller's location.
    /// \return `snicholls::result<V, snicholls::traced_t<E>>`.
    [[nodiscard]] constexpr result<V, traced_t<E>> trace(source_location where = source_location::current()) &&
    {
        if (is_ok())
            return ok_like<traced_t<E>>(std::move(*this));
        return result<V, traced_t<E>>(err_in_place, make_traced(std::move(*this).err_ref(), where));
    }

    /// \name Log an error through a sink, keep the result
    ///@{
    /// \brief Logs an error through `sink.log_error(level, error, where)`; returns this result.
    ///
    /// `snicholls::error_sink` (`snicholls/core/traced.hpp`) is the interface; any type with that member works. A success costs
    /// one branch.
    /// \param sink the sink.
    /// \param level the level (`snicholls::error_level`).
    /// \param where the caller's location (recorded for errors that carry none).
    /// \return `*this` (moved for an rvalue).
    template <class Sink, class Level>
        requires requires(Sink& s, Level l, const E& e, source_location w) { s.log_error(l, e, w); }
    constexpr result& log_err(Sink& sink, Level level, source_location where = source_location::current()) &
    {
        if (is_err()) [[unlikely]]
            sink.log_error(level, std::as_const(base::err()), where);
        return *this;
    }
    template <class Sink, class Level>
        requires requires(Sink& s, Level l, const E& e, source_location w) { s.log_error(l, e, w); }
    constexpr const result& log_err(Sink& sink, Level level,
                                    source_location where = source_location::current()) const&
    {
        if (is_err()) [[unlikely]]
            sink.log_error(level, base::err(), where);
        return *this;
    }
    template <class Sink, class Level>
        requires requires(Sink& s, Level l, const E& e, source_location w) { s.log_error(l, e, w); }
    [[nodiscard]] constexpr result log_err(Sink& sink, Level level,
                                           source_location where = source_location::current()) &&
    {
        if (is_err()) [[unlikely]]
            sink.log_error(level, std::as_const(base::err()), where);
        return std::move(*this);
    }
    ///@}

    /// \name Log an error at the sink's default level, keep the result
    ///@{
    /// \brief Logs an error through `sink.log_error(error, where)` (the sink's default level).
    /// \param sink the sink.
    /// \param where the caller's location.
    /// \return `*this` (moved for an rvalue).
    template <class Sink>
        requires requires(Sink& s, const E& e, source_location w) { s.log_error(e, w); }
    constexpr result& log_err(Sink& sink, source_location where = source_location::current()) &
    {
        if (is_err()) [[unlikely]]
            sink.log_error(std::as_const(base::err()), where);
        return *this;
    }
    template <class Sink>
        requires requires(Sink& s, const E& e, source_location w) { s.log_error(e, w); }
    constexpr const result& log_err(Sink& sink, source_location where = source_location::current()) const&
    {
        if (is_err()) [[unlikely]]
            sink.log_error(base::err(), where);
        return *this;
    }
    template <class Sink>
        requires requires(Sink& s, const E& e, source_location w) { s.log_error(e, w); }
    [[nodiscard]] constexpr result log_err(Sink& sink, source_location where = source_location::current()) &&
    {
        if (is_err()) [[unlikely]]
            sink.log_error(std::as_const(base::err()), where);
        return std::move(*this);
    }
    ///@}

#if defined(__cpp_lib_expected)
    /// \brief Converts to `std::expected<V, E>` (C++23).
    /// \return the expected.
    [[nodiscard]] constexpr auto to_expected() const&
    {
        using X = std::expected<V, E>;
        if (is_err())
            return X(std::unexpect, base::err());
        if constexpr (is_void)
            return X();
        else
            return X(std::in_place, base::val());
    }
#endif

    // ---- comparison, swap ------------------------------------------------------------------------------------------

    /// \brief Equal when both are successes with equal values, or both errors with equal errors.
    /// \param a a result.
    /// \param b a result.
    /// \return the result.
    template <class U, class G>
        requires(std::is_void_v<U> == is_void)
    [[nodiscard]] friend constexpr bool operator==(const result& a, const result<U, G>& b) noexcept(
        noexcept(a.unwrap_err_unchecked() == b.unwrap_err_unchecked()) && result::template values_nothrow_equal<U, G>())
    {
        if (a.is_ok() != b.is_ok())
            return false;
        if (a.is_err())
            return static_cast<bool>(a.unwrap_err_unchecked() == b.unwrap_err_unchecked());
        if constexpr (is_void)
            return true;
        else
            return static_cast<bool>(a.stored_value() == b.unwrap_unchecked());
    }

    /// \brief True for a success equal to `o.value`.
    /// \param a the result.
    /// \param o the success tag.
    /// \return the result.
    template <class U>
        requires(!is_void)
    [[nodiscard]] friend constexpr bool operator==(const result& a, const ok_t<U>& o) noexcept(
        noexcept(a.stored_value() == o.value))
    {
        return a.is_ok() && static_cast<bool>(a.stored_value() == o.value);
    }

    /// \brief True for the success of a `result<void, E>`.
    /// \param a the result.
    /// \return the result.
    [[nodiscard]] friend constexpr bool operator==(const result& a, const ok_t<void>&) noexcept
        requires is_void
    {
        return a.is_ok();
    }

    /// \brief True for an error equal to `e.error`.
    /// \param a the result.
    /// \param e the error tag.
    /// \return the result.
    template <class G>
    [[nodiscard]] friend constexpr bool operator==(const result& a, const err_t<G>& e) noexcept(
        noexcept(a.err_ref() == e.error))
    {
        return a.is_err() && static_cast<bool>(a.err_ref() == e.error);
    }

    /// \brief Rust's derived `Ord`: every success orders before every error; then by value or by error.
    /// \param a a result.
    /// \param b a result.
    /// \return the common comparison category of V's and E's `<=>`.
    [[nodiscard]] friend constexpr auto operator<=>(const result& a, const result& b)
        requires std::three_way_comparable<stored> && std::three_way_comparable<E>
    {
        using cat = std::common_comparison_category_t<std::compare_three_way_result_t<stored>,
                                                      std::compare_three_way_result_t<E>>;
        if (a.is_ok() != b.is_ok())
            return static_cast<cat>(a.is_ok() ? std::strong_ordering::less : std::strong_ordering::greater);
        return a.is_ok() ? static_cast<cat>(a.stored_value() <=> b.stored_value())
                         : static_cast<cat>(a.err_ref() <=> b.err_ref());
    }

    /// \brief Swaps two results.
    /// \param a a result.
    /// \param b a result.
    friend constexpr void swap(result& a, result& b) noexcept(
        std::is_nothrow_move_constructible_v<result> && std::is_nothrow_move_assignable_v<result>)
        requires std::is_move_constructible_v<result> && std::is_move_assignable_v<result>
    {
        result t(std::move(a));
        a = std::move(b);
        b = std::move(t);
    }

    /// \brief The hash of the state and the value or error (for `std::hash`).
    /// \return the hash.
    [[nodiscard]] std::size_t hash() const noexcept(noexcept(std::hash<stored>{}(std::declval<const stored&>())) &&
                                                    noexcept(std::hash<E>{}(std::declval<const E&>())))
    {
        // Cast explicitly: on a 32-bit size_t the 64-bit constants are truncated, by design and without a warning.
        std::size_t seed = is_ok() ? static_cast<std::size_t>(0x51ED270B27ULL) : static_cast<std::size_t>(0xE7037ED1A0B428DBULL);
        if (is_ok())
            result_impl::hash_mix(seed, base::val());
        else
            result_impl::hash_mix(seed, base::err());
        return seed;
    }

private:
    // Whether comparing the values of this result and a result<U, G> is noexcept. A void result has no value to
    // compare, so the expression is never formed for it (forming it did not compile: unwrap_unchecked() is void).
    template <class U, class G>
    static constexpr bool values_nothrow_equal() noexcept
    {
        if constexpr (is_void)
            return true;
        else
            return noexcept(std::declval<const stored&>() == std::declval<const result<U, G>&>().unwrap_unchecked());
    }

    [[nodiscard]] constexpr const stored& stored_value() const& noexcept { return base::val(); }
    [[nodiscard]] constexpr stored& stored_value() & noexcept { return base::val(); }
    [[nodiscard]] constexpr stored&& stored_value() && noexcept { return std::move(*this).base::val(); }
    [[nodiscard]] constexpr const E& err_ref() const& noexcept { return base::err(); }
    [[nodiscard]] constexpr E&& err_ref() && noexcept { return std::move(*this).base::err(); }

#if defined(__cpp_lib_expected)
    template <class U>
    static constexpr stored expected_value(const std::expected<U, E>& x)
    {
        if constexpr (is_void)
            return stored{};
        else
            return stored(*x);
    }
#endif

    // The value as unwrap returns it: void for result<void, E>, else a reference of the right category.
    template <class Self>
    static constexpr decltype(auto) checked_ok(Self&& self) noexcept(!std::is_rvalue_reference_v<Self&&> ||
                                                                      std::is_nothrow_move_constructible_v<stored>)
    {
        assert(self.is_ok());
        if constexpr (is_void)
            return;
        else if constexpr (std::is_rvalue_reference_v<Self&&>)
            return stored(std::move(self).stored_value());   // by value: a reference into a temporary would dangle
        else
            return (self.stored_value());
    }

    // f(value) for V, f() for void.
    template <class F, class Self>
    static constexpr decltype(auto) call_value(F&& f, Self&& self) noexcept(
        is_void ? std::is_nothrow_invocable_v<F>
                : std::is_nothrow_invocable_v<F, decltype(std::declval<Self>().stored_value())>)
    {
        if constexpr (is_void)
            return std::invoke(std::forward<F>(f));
        else
            return std::invoke(std::forward<F>(f), std::forward<Self>(self).stored_value());
    }

    // This success as a result<V, G>.
    template <class G, class Self>
    static constexpr result<V, G> ok_like(Self&& self)
    {
        if constexpr (is_void)
            return result<V, G>(std::in_place);
        else
            return result<V, G>(std::in_place, std::forward<Self>(self).stored_value());
    }

    template <class X>
    static constexpr traced_t<E> make_traced(X&& e, source_location where)
    {
        if constexpr (is_traced_v<E>)
            return traced_t<E>(std::forward<X>(e));
        else
            return traced_t<E>(std::forward<X>(e), where);
    }

    template <class Self>
    using err_fwd_t = decltype(std::declval<Self>().err_fwd());
    template <class Self, class F>
    using call_t = decltype(call_value(std::declval<F>(), std::declval<Self>()));

    template <class Self>
    static constexpr bool nothrow_ok_like = is_void || std::is_nothrow_constructible_v<
                                                           stored, decltype(std::declval<Self>().stored_value())>;

    template <class Self, class F>
    static constexpr bool map_noexcept() noexcept
    {
        using U = std::remove_cv_t<call_t<Self, F>>;
        constexpr bool base_ok = noexcept(call_value(std::declval<F>(), std::declval<Self>())) &&
                                 std::is_nothrow_constructible_v<E, err_fwd_t<Self>>;
        if constexpr (std::is_void_v<U>)
            return base_ok;
        else
            return base_ok && std::is_nothrow_constructible_v<U, call_t<Self, F>>;
    }

    template <class Self, class F>
    static constexpr bool map_err_noexcept() noexcept
    {
        using G = std::remove_cv_t<std::invoke_result_t<F, err_fwd_t<Self>>>;
        return std::is_nothrow_invocable_v<F, err_fwd_t<Self>> && std::is_nothrow_constructible_v<G, std::invoke_result_t<F, err_fwd_t<Self>>> &&
               nothrow_ok_like<Self>;
    }

    template <class Self, class F>
    static constexpr bool and_then_noexcept() noexcept
    {
        return noexcept(call_value(std::declval<F>(), std::declval<Self>())) &&
               std::is_nothrow_constructible_v<E, err_fwd_t<Self>>;
    }

    template <class Self, class F>
    static constexpr bool or_else_noexcept() noexcept
    {
        return std::is_nothrow_invocable_v<F, err_fwd_t<Self>> && nothrow_ok_like<Self>;
    }

    template <class Self, class F>
    static constexpr auto map_impl(Self&& self, F&& f) noexcept(map_noexcept<Self, F>())
    {
        using U = std::remove_cv_t<decltype(call_value(std::forward<F>(f), std::forward<Self>(self)))>;
        using R = result<U, E>;
        if (self.is_ok())
        {
            if constexpr (std::is_void_v<U>)
            {
                call_value(std::forward<F>(f), std::forward<Self>(self));
                return R(std::in_place);
            }
            else
                return R(std::in_place, call_value(std::forward<F>(f), std::forward<Self>(self)));
        }
        return R(err_in_place, std::forward<Self>(self).err_fwd());
    }

    template <class Self, class F>
    static constexpr auto map_err_impl(Self&& self, F&& f) noexcept(map_err_noexcept<Self, F>())
    {
        using G = std::remove_cv_t<std::invoke_result_t<F, decltype(std::forward<Self>(self).err_fwd())>>;
        using R = result<V, G>;
        if (self.is_ok())
            return ok_like<G>(std::forward<Self>(self));
        return R(err_in_place, std::invoke(std::forward<F>(f), std::forward<Self>(self).err_fwd()));
    }

    template <class Self, class F>
    static constexpr auto and_then_impl(Self&& self, F&& f) noexcept(and_then_noexcept<Self, F>())
    {
        using R = std::remove_cvref_t<decltype(call_value(std::forward<F>(f), std::forward<Self>(self)))>;
        static_assert(result_impl::is_result_v<R>, "snicholls::result::and_then: f must return a snicholls::result");
        static_assert(std::is_same_v<typename R::error_type, E>,
                      "snicholls::result::and_then: f must return a snicholls::result with the same error type");
        if (self.is_ok())
            return R(call_value(std::forward<F>(f), std::forward<Self>(self)));
        return R(err_in_place, std::forward<Self>(self).err_fwd());
    }

    template <class Self, class F>
    static constexpr auto or_else_impl(Self&& self, F&& f) noexcept(or_else_noexcept<Self, F>())
    {
        using R = std::remove_cvref_t<std::invoke_result_t<F, decltype(std::forward<Self>(self).err_fwd())>>;
        static_assert(result_impl::is_result_v<R>, "snicholls::result::or_else: f must return a snicholls::result");
        static_assert(std::is_same_v<typename R::value_type, V>,
                      "snicholls::result::or_else: f must return a snicholls::result with the same value type");
        if (self.is_ok())
            return ok_like<typename R::error_type>(std::forward<Self>(self));
        return R(std::invoke(std::forward<F>(f), std::forward<Self>(self).err_fwd()));
    }

    [[nodiscard]] constexpr E& err_fwd() & noexcept { return base::err(); }
    [[nodiscard]] constexpr const E& err_fwd() const& noexcept { return base::err(); }
    [[nodiscard]] constexpr E&& err_fwd() && noexcept { return std::move(*this).base::err(); }
};

// -------------------------------------------------------------------------------------------------------------------
// Exceptions -> results
// -------------------------------------------------------------------------------------------------------------------

/// \brief Runs `f(args...)` and returns its value as a success, or the exception it threw as
///        `err(std::exception_ptr)` (Rust's `catch_unwind`, for exceptions).
/// \ingroup core
///
/// Catches everything, including `std::bad_alloc`. If f returns a reference, the referred value is copied. Without
/// exceptions it just calls f (nothing can be thrown).
/// \param f the callable.
/// \param args its arguments.
/// \return `snicholls::result<R, std::exception_ptr>`, R being f's decayed return type (void allowed).
template <class F, class... A>
[[nodiscard]] auto try_invoke(F&& f, A&&... args) noexcept
    -> result<std::remove_cvref_t<std::invoke_result_t<F, A...>>, std::exception_ptr>
{
    using R = std::remove_cvref_t<std::invoke_result_t<F, A...>>;
    using out = result<R, std::exception_ptr>;
#if SNICHOLLS_RESULT_HAS_EXCEPTIONS
    try
    {
#endif
        if constexpr (std::is_void_v<R>)
        {
            std::invoke(std::forward<F>(f), std::forward<A>(args)...);
            return out(std::in_place);
        }
        else
            return out(std::in_place, std::invoke(std::forward<F>(f), std::forward<A>(args)...));
#if SNICHOLLS_RESULT_HAS_EXCEPTIONS
    }
    catch (...)
    {
        return out(err_in_place, std::current_exception());
    }
#endif
}

/// \brief As `snicholls::try_invoke(f)`.
/// \ingroup core
/// \param f the callable (no arguments).
/// \return `snicholls::result<R, std::exception_ptr>`.
template <class F>
[[nodiscard]] auto result_from_exceptions(F&& f) noexcept
{
    return try_invoke(std::forward<F>(f));
}

/// \brief Runs `f()`; an exception becomes `err(map(std::current_exception()))`.
/// \ingroup core
///
/// `map` converts the exception to your error type, typically by rethrowing and catching what it knows.
/// \param f the callable (no arguments).
/// \param map `std::exception_ptr` -> the error type.
/// \return `snicholls::result<R, G>`, G being what `map` returns.
/// \throws what `map` throws.
template <class F, class M>
[[nodiscard]] auto result_from_exceptions(F&& f, M&& map)
    -> result<std::remove_cvref_t<std::invoke_result_t<F>>, std::remove_cvref_t<std::invoke_result_t<M, std::exception_ptr>>>
{
    using R = std::remove_cvref_t<std::invoke_result_t<F>>;
    using G = std::remove_cvref_t<std::invoke_result_t<M, std::exception_ptr>>;
    using out = result<R, G>;
    std::exception_ptr caught;
#if SNICHOLLS_RESULT_HAS_EXCEPTIONS
    try
    {
#endif
        if constexpr (std::is_void_v<R>)
        {
            std::invoke(std::forward<F>(f));
            return out(std::in_place);
        }
        else
            return out(std::in_place, std::invoke(std::forward<F>(f)));
#if SNICHOLLS_RESULT_HAS_EXCEPTIONS
    }
    catch (...)
    {
        caught = std::current_exception();
    }
    return out(err_in_place, std::invoke(std::forward<M>(map), std::move(caught)));
#endif
}

#if defined(__cpp_lib_expected)
/// \brief Converts a `std::expected<V, E>` to a `snicholls::result<V, E>` (C++23).
/// \ingroup core
/// \param x the expected.
/// \return the result.
template <class V, class E>
[[nodiscard]] constexpr result<V, E> to_result(const std::expected<V, E>& x)
{
    return result<V, E>(x);
}
#endif

} // inline namespace result_throws / result_terminates

} // namespace snicholls

/// \brief `std::hash` of a `snicholls::result<V, E>`: enabled when both `std::hash<V>` and `std::hash<E>` are.
/// \ingroup core
/// \tparam V the value type.
/// \tparam E the error type.
template <class V, class E>
    requires(std::is_default_constructible_v<
                 std::hash<std::conditional_t<std::is_void_v<V>, snicholls::result_impl::void_value, V>>> &&
             std::is_default_constructible_v<std::hash<E>>)
struct std::hash<snicholls::result<V, E>>
{
    /// \brief The hash.
    /// \param r the result.
    /// \return `r.hash()`.
    [[nodiscard]] std::size_t operator()(const snicholls::result<V, E>& r) const noexcept(noexcept(r.hash()))
    {
        return r.hash();
    }
};

/// \brief `std::hash` of the unit value of `snicholls::result<void, E>`.
/// \ingroup core
template <>
struct std::hash<snicholls::result_impl::void_value>
{
    /// \brief The hash (a constant).
    /// \return 0.
    [[nodiscard]] std::size_t operator()(snicholls::result_impl::void_value) const noexcept { return 0; }
};

// -------------------------------------------------------------------------------------------------------------------
// Propagation: Rust's `?`
// -------------------------------------------------------------------------------------------------------------------

/// \brief Rust's `?` (GCC, Clang): the value of `expr`, or return its error from the enclosing function.
/// \ingroup core
///
/// `auto px = SNICHOLLS_TRY(parse_price(text));`. The error converts to the enclosing function's error type through
/// `snicholls::error_conversion` (Rust's `From`): a `snicholls::traced` error gains a "propagated" context frame with the
/// expression text and this location; a plain error entering a traced function is wrapped, located here. `expr` is
/// evaluated once; an rvalue's value is moved out, an lvalue's copied. The enclosing function must declare a
/// `snicholls::result` return type. A GNU statement expression (`__extension__` keeps `-Wpedantic` quiet); without one
/// (`SNICHOLLS_RESULT_HAS_STATEMENT_EXPRESSIONS` is 0, e.g. MSVC) use `SNICHOLLS_TRY_ASSIGN` / `SNICHOLLS_TRY_VOID`.
/// \param ... the expression (may contain commas).
#if SNICHOLLS_RESULT_HAS_STATEMENT_EXPRESSIONS
#define SNICHOLLS_TRY(...) \
    SNICHOLLS_RESULT_TRY_EXPR_(SNICHOLLS_RESULT_CAT(snicholls_try_result_, __COUNTER__), #__VA_ARGS__, __VA_ARGS__)
#endif

// The TRY macros name their temporary uniquely (__COUNTER__, which GCC, Clang and MSVC all have), so a TRY nested
// in another's expression, or two on one line, never shadow each other (-Wshadow). The expression text is taken
// before expansion, by the public macro, so the "propagated" frame shows what the caller wrote.
#define SNICHOLLS_RESULT_TRY_EXPR_(name, text, ...)                                                                    \
    __extension__({                                                                                                    \
        auto&& name = (__VA_ARGS__);                                                                                   \
        if (!name.is_ok()) [[unlikely]]                                                                                \
            return ::snicholls::result_impl::propagate(static_cast<decltype(name)&&>(name), text,                      \
                                                       ::snicholls::source_location::current());                      \
        static_cast<decltype(name)&&>(name).unwrap_unchecked();                                                        \
    })

#define SNICHOLLS_RESULT_CAT_(a, b) a##b
#define SNICHOLLS_RESULT_CAT(a, b) SNICHOLLS_RESULT_CAT_(a, b)

/// \brief Portable `?` as a statement: declares `decl` from the value of `expr`, or returns its error.
/// \ingroup core
///
/// `SNICHOLLS_TRY_ASSIGN(auto px, parse_price(text));` - see `SNICHOLLS_TRY` for the conversion rules. One per line.
/// \param decl the declaration (or an assignable lvalue) receiving the value.
/// \param ... the expression (a `snicholls::result` with a non-void value).
#define SNICHOLLS_TRY_ASSIGN(decl, ...)                                                                                     \
    SNICHOLLS_RESULT_TRY_ASSIGN_(SNICHOLLS_RESULT_CAT(snicholls_try_result_, __COUNTER__), decl, #__VA_ARGS__, __VA_ARGS__)
#define SNICHOLLS_RESULT_TRY_ASSIGN_(name, decl, text, ...)                                                            \
    auto&& name = (__VA_ARGS__);                                                                                       \
    if (!name.is_ok()) [[unlikely]]                                                                                    \
        return ::snicholls::result_impl::propagate(static_cast<decltype(name)&&>(name), text,                          \
                                                   ::snicholls::source_location::current());                          \
    decl = static_cast<decltype(name)&&>(name).unwrap_unchecked()

/// \brief Portable `?` as a statement, discarding the value: returns the error of `expr` if there is one.
/// \ingroup core
///
/// `SNICHOLLS_TRY_VOID(validate(order));` - the familiar `RETURN_IF_ERR`. See `SNICHOLLS_TRY` for the conversion rules.
/// \param ... the expression (any `snicholls::result`).
#define SNICHOLLS_TRY_VOID(...)                                                                                             \
    SNICHOLLS_RESULT_TRY_VOID_(SNICHOLLS_RESULT_CAT(snicholls_try_result_, __COUNTER__), #__VA_ARGS__, __VA_ARGS__)
#define SNICHOLLS_RESULT_TRY_VOID_(name, text, ...)                                                                    \
    do                                                                                                                 \
    {                                                                                                                  \
        auto&& name = (__VA_ARGS__);                                                                                   \
        if (!name.is_ok()) [[unlikely]]                                                                                \
            return ::snicholls::result_impl::propagate(static_cast<decltype(name)&&>(name), text,                      \
                                                       ::snicholls::source_location::current());                      \
    } while (false)

#endif /* snicholls_core_result_hpp */
