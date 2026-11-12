//
//  version.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief Which version of snicholls-core you are compiling against.
/// \ingroup core
///
/// CMake knows the version and exports it for `find_package`, but a header-only
/// library is also consumed by people who copy the headers into a tree with no
/// build system at all, and they have no CMake to ask. Without these macros
/// there is no way for them to find out what they have, or to write code that
/// works against two versions:
///
/// \code
/// #if SNICHOLLS_CORE_VERSION >= SNICHOLLS_CORE_VERSION_CHECK(0, 2, 0)
///     // use something that only exists from 0.2 onwards
/// #endif
/// \endcode
///
/// A consumer that needs a minimum says so once, where the compiler can see it:
///
/// \code
/// SNICHOLLS_CORE_REQUIRE_VERSION(0, 5, 0);
/// \endcode
///
/// That is the compile-time half of the diamond rules in CMakeLists.txt; the
/// CMake half is `snicholls_core_require(x.y.z)`. The number here, the
/// `project(... VERSION)` in CMakeLists.txt and the release tag are three
/// copies of one fact, checked against each other by `scripts/check_version.py`
/// on every CI job. Three places holding one number is exactly how a number
/// goes quietly wrong.
///
/// \snippet version_demo.cpp version

#ifndef snicholls_core_version_hpp
#define snicholls_core_version_hpp

#include <string_view>

/// \addtogroup core
/// @{

/// \brief Major version of snicholls-core.
#define SNICHOLLS_CORE_VERSION_MAJOR 0
/// \brief Minor version of snicholls-core.
#define SNICHOLLS_CORE_VERSION_MINOR 8
/// \brief Patch version of snicholls-core.
#define SNICHOLLS_CORE_VERSION_PATCH 0

/// \brief Encode a version as one integer the preprocessor can compare.
///
/// Two decimal digits each for minor and patch, which is plenty and keeps the
/// number readable: 1.0.0 is 10000, 1.2.3 is 10203.
/// \param major Major version.
/// \param minor Minor version, 0 to 99.
/// \param patch Patch version, 0 to 99.
#define SNICHOLLS_CORE_VERSION_CHECK(major, minor, patch) ((major) * 10000 + (minor) * 100 + (patch))

/// \brief This version of snicholls-core, encoded by #SNICHOLLS_CORE_VERSION_CHECK.
#define SNICHOLLS_CORE_VERSION \
    SNICHOLLS_CORE_VERSION_CHECK(SNICHOLLS_CORE_VERSION_MAJOR, SNICHOLLS_CORE_VERSION_MINOR, SNICHOLLS_CORE_VERSION_PATCH)

/// \brief Implementation detail of #SNICHOLLS_CORE_VERSION_STRING: stringise a token.
/// \param x The token to stringise, after expansion.
#define SNICHOLLS_CORE_VERSION_STRINGIFY_(x) #x
/// \brief Implementation detail of #SNICHOLLS_CORE_VERSION_STRING: expand, then stringise.
/// \param x The macro to expand and stringise.
#define SNICHOLLS_CORE_VERSION_STRINGIFY(x) SNICHOLLS_CORE_VERSION_STRINGIFY_(x)

/// \brief This version of snicholls-core as a string literal, `"major.minor.patch"`.
#define SNICHOLLS_CORE_VERSION_STRING                                  \
    SNICHOLLS_CORE_VERSION_STRINGIFY(SNICHOLLS_CORE_VERSION_MAJOR) "." \
    SNICHOLLS_CORE_VERSION_STRINGIFY(SNICHOLLS_CORE_VERSION_MINOR) "." \
    SNICHOLLS_CORE_VERSION_STRINGIFY(SNICHOLLS_CORE_VERSION_PATCH)

/// \brief Fail to compile unless snicholls-core is at least `major.minor.patch`.
///
/// Put one in a header of yours that depends on a core feature, and a build
/// against an older core stops with a message that names the number rather
/// than deep inside a template. No trailing semicolon in the expansion: write
/// `SNICHOLLS_CORE_REQUIRE_VERSION(0, 5, 0);` like any declaration.
/// \param major Minimum major version.
/// \param minor Minimum minor version.
/// \param patch Minimum patch version.
#define SNICHOLLS_CORE_REQUIRE_VERSION(major, minor, patch)                                     \
    static_assert(SNICHOLLS_CORE_VERSION >= SNICHOLLS_CORE_VERSION_CHECK(major, minor, patch),  \
                  "snicholls-core " SNICHOLLS_CORE_VERSION_STRING " is older than the " #major \
                  "." #minor "." #patch " this code requires")

/// @}

namespace snicholls {

/// \brief The library's own metadata. Everything else in snicholls-core lives in `snicholls` itself.
namespace core {

/// \addtogroup core
/// @{

/// \brief Major version, the same as #SNICHOLLS_CORE_VERSION_MAJOR; usable in a `static_assert`.
/// \return The major version number.
constexpr int version_major() noexcept { return SNICHOLLS_CORE_VERSION_MAJOR; }
/// \brief Minor version, the same as #SNICHOLLS_CORE_VERSION_MINOR.
/// \return The minor version number.
constexpr int version_minor() noexcept { return SNICHOLLS_CORE_VERSION_MINOR; }
/// \brief Patch version, the same as #SNICHOLLS_CORE_VERSION_PATCH.
/// \return The patch version number.
constexpr int version_patch() noexcept { return SNICHOLLS_CORE_VERSION_PATCH; }
/// \brief The encoded version, the same as #SNICHOLLS_CORE_VERSION.
/// \return major * 10000 + minor * 100 + patch.
constexpr int version() noexcept { return SNICHOLLS_CORE_VERSION; }
/// \brief The version as text, `"major.minor.patch"`, the same as #SNICHOLLS_CORE_VERSION_STRING.
/// \return A view of a string literal; valid for the life of the program.
constexpr std::string_view version_string() noexcept { return SNICHOLLS_CORE_VERSION_STRING; }

/// @}

} // namespace core
} // namespace snicholls

#endif /* snicholls_core_version_hpp */
