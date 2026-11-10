//
//  version_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/version.hpp>. Prints the version every way the
//  header offers it and exits non-zero if the ways disagree, so this is also a
//  test (CTest label `demo`). The marked region is pulled into the Doxygen page
//  for version.hpp with \snippet, so it is real compiled code, not a copy that
//  can rot.
//

#include <snicholls/core/version.hpp>

#include <cstdio>
#include <string>
#include <string_view>

namespace {

bool check(bool ok, const char* what)
{
    std::printf("  %s  %s\n", ok ? "ok  " : "FAIL", what);
    return ok;
}

} // namespace

int main()
{
    std::printf("snicholls-core version demo\n");

    //! [version]
    // Three views of one number: macros for the preprocessor, constexpr
    // functions for templates and static_assert, a string for people.
    std::printf("  version            %s\n", SNICHOLLS_CORE_VERSION_STRING);
    std::printf("  major.minor.patch  %d.%d.%d\n",
                snicholls::core::version_major(),
                snicholls::core::version_minor(),
                snicholls::core::version_patch());
    std::printf("  encoded            %d\n", snicholls::core::version());

    // Gate code on a minimum at compile time, in either language
#if SNICHOLLS_CORE_VERSION >= SNICHOLLS_CORE_VERSION_CHECK(0, 1, 0)
    std::printf("  at least 0.1.0     yes, says the preprocessor\n");
#endif
    static_assert(snicholls::core::version() >= SNICHOLLS_CORE_VERSION_CHECK(0, 1, 0));
    SNICHOLLS_CORE_REQUIRE_VERSION(0, 1, 0);
    //! [version]

    bool ok = true;

    const std::string from_parts = std::to_string(snicholls::core::version_major()) + "."
                                 + std::to_string(snicholls::core::version_minor()) + "."
                                 + std::to_string(snicholls::core::version_patch());
    ok &= check(from_parts == snicholls::core::version_string(),
                "the string agrees with the components");
    ok &= check(snicholls::core::version()
                    == snicholls::core::version_major() * 10000
                         + snicholls::core::version_minor() * 100
                         + snicholls::core::version_patch(),
                "the encoded number agrees with the components");
    ok &= check(std::string_view{SNICHOLLS_CORE_VERSION_STRING} == snicholls::core::version_string(),
                "the macro string agrees with the function");

    std::printf(ok ? "version demo: ok\n" : "version demo: FAILED\n");
    return ok ? 0 : 1;
}
