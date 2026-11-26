//
//  random_source_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  The random_source_like concept: a function that takes any source of secure random bytes.
//

#include <snicholls/core/concepts/random_source.hpp>
#include <snicholls/core/interfaces/random_source_reference.hpp>

#include <array>
#include <cstddef>
#include <cstdio>

namespace {
int failures = 0;
void check(bool ok, const char* what)
{
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++failures;
}
} // namespace

//! [concept]
// A 128-bit session token from any random source; the caller's failure is ours, never a weaker fallback
template <typename Source>
std::errc session_token(Source& source, std::array<std::byte, 16>& token) noexcept
{
    static_assert(snicholls::is_random_source_like_v<Source>, "a cryptographic random source");
    return source.fill(token.data(), token.size());
}
//! [concept]

int main()
{
    std::puts("random_source_like");
    snicholls::test_random_source tests{2026};   // in production: ts-moveables' operating-system source
    std::array<std::byte, 16> token{};
    check(!snicholls::random_failed(session_token(tests, token)), "a token from the test source");
    snicholls::unavailable_random_source down;
    check(snicholls::random_failed(session_token(down, token)), "an unavailable source is an error, never a fallback");
    std::printf("%s\n", failures == 0 ? "all checks passed" : "FAILED");
    return failures == 0 ? 0 : 1;
}
