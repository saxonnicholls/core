//
//  secret.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief Secrets: values that must never be printed or logged.
/// \ingroup core
///
/// Secrets: values that must never be printed or logged.
///
/// \code
///   snicholls::secret<std::string> key{"..."};
///   std::cout << key;              // ********
///   connect(key.reveal());         // the only way to the value - easy to grep for
/// \endcode
///
/// is_secret_key(name) is the naming heuristic config uses to mask values (PASSWORD, SECRET, TOKEN, ..., *_KEY),
/// and mask_url_credentials() hides the password in "scheme://user:password\@host" values.

#ifndef snicholls_core_secret_hpp
#define snicholls_core_secret_hpp

#include <algorithm>
#include <cstddef>
#include <ostream>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace snicholls {

/// \brief The masked text.
inline constexpr std::string_view masked_text = "********";

/// \brief Secrets: values that must never be printed or logged.
/// \ingroup core
///
/// \overview
/// The header's description above gives the usage and the rules in full.
///
/// \relations
/// - See the file description above for the related types and functions.
/// - Group: \ref core.
///
/// \impls
/// None: a single implementation.
///
/// \contracts
/// Stated in the file description above where they apply (thread safety, exceptions, lifetime); otherwise the usual value-type rules: no internal synchronisation.
///
/// \errors
/// See the `\throws` of the members; none otherwise.
///
/// \examples
/// No demo of its own; the tests show every use (tests/secret_test.cpp).
///
/// \perf
/// Not benchmarked.
///
/// \testing
/// - Tests: tests/secret_test.cpp.
/// - Demo: none of its own.
///
/// \deps
/// CMake target `snicholls::core`; standard library only (app core).
///
/// \gotchas
/// None known.
///
/// \status
/// Stable, pre-1.0.
template <typename T>
class secret
{
public:
    /// \brief The stored object type.
    using value_type = T;

    /// \brief Constructs it.
    secret() = default;
    /// \brief Constructs it.
    /// \param value the value.
    explicit secret(T value) noexcept(std::is_nothrow_move_constructible_v<T>) : value_(std::move(value)) {}

    /// \brief Constructs it.
    secret(const secret&) = default;
    /// \brief Constructs it.
    secret(secret&&) noexcept(std::is_nothrow_move_constructible_v<T>) = default;
    /// \brief Assigns.
    /// \return the result.
    secret& operator=(const secret&) = default;
    /// \brief Assigns.
    /// \return the result.
    secret& operator=(secret&&) noexcept(std::is_nothrow_move_assignable_v<T>) = default;

    /// \brief Best effort: overwrite string storage before it is released.
    ///
    /// (the optimiser may not elide volatile writes).
    ~secret() { wipe(); }

    /// \brief The reveal.
    /// \return the reveal.
    [[nodiscard]] const T& reveal() const& noexcept { return value_; }
    /// \brief The reveal.
    /// \return the reveal.
    [[nodiscard]] T& reveal() & noexcept { return value_; }

    /// \brief Equality.
    /// \param l the l.
    /// \param r the r.
    /// \return the result.
    friend bool operator==(const secret& l, const secret& r) noexcept(noexcept(l.value_ == r.value_))
    {
        return l.value_ == r.value_;
    }

    /// \brief Writes the value to a stream.
    /// \param os the output stream.
    /// \return the result.
    friend std::ostream& operator<<(std::ostream& os, const secret&) { return os << masked_text; }

private:
    void wipe() noexcept
    {
        if constexpr (std::is_same_v<T, std::string>)
        {
            volatile char* p = value_.data();
            for (std::size_t i = 0; i < value_.size(); ++i)
                p[i] = '\0';
        }
    }

    T value_{};
};

template <typename T>
struct is_secret : std::false_type {};

template <typename T>
struct is_secret<secret<T>> : std::true_type {};

/// \brief Whether the type is secret (compile-time constant).
template <typename T>
inline constexpr bool is_secret_v = is_secret<std::remove_cvref_t<T>>::value;

namespace detail {

constexpr char ascii_upper(char c) noexcept
{
    return (c >= 'a' && c <= 'z') ? static_cast<char>(c - 'a' + 'A') : c;
}

} // namespace detail

/// \brief True for configuration keys that conventionally hold secrets.
///
/// Case-insensitive. False positives are harmless (a value is masked); false negatives are not, so the list errs on the side of masking.
/// \ingroup core
/// \param key the key.
/// \return the result.
[[nodiscard]] inline bool is_secret_key(std::string_view key)
{
    std::string upper(key);
    std::transform(upper.begin(), upper.end(), upper.begin(), detail::ascii_upper);
    constexpr std::string_view markers[] = {"PASSWORD", "PASSWD", "PASSPHRASE", "SECRET", "TOKEN", "CREDENTIAL",
                                            "PRIVATE",  "AUTH",   "API_KEY",    "APIKEY", "ACCESS_KEY", "COOKIE",
                                            "SESSION",  "SIGNATURE"};
    for (const std::string_view marker : markers)
        if (upper.find(marker) != std::string::npos)
            return true;
    return upper == "KEY" || upper.ends_with("_KEY") || upper.ends_with("_PWD");
}

/// \brief "postgres://user:pw\@host/db" -> "postgres://user:********\@host/db".
///
/// Values without credentials are unchanged.
/// \ingroup core
/// \param value the value.
/// \return the result.
[[nodiscard]] inline std::string mask_url_credentials(std::string_view value)
{
    const std::size_t scheme = value.find("://");
    if (scheme == std::string_view::npos)
        return std::string(value);
    const std::size_t authority = scheme + 3;
    const std::size_t end = value.find_first_of("/?#", authority);
    const std::string_view host_part = value.substr(authority, end == std::string_view::npos ? value.npos : end - authority);
    const std::size_t at = host_part.rfind('@');
    if (at == std::string_view::npos)
        return std::string(value);
    const std::size_t colon = host_part.find(':');
    if (colon == std::string_view::npos || colon > at)
        return std::string(value);
    std::string out(value.substr(0, authority + colon + 1));
    out += masked_text;
    out += value.substr(authority + at);
    return out;
}

} // namespace snicholls

#endif /* snicholls_core_secret_hpp */
