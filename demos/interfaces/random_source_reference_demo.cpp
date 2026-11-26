//
//  random_source_reference_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  The test doubles: a reproducible source for tests, and one that is always unavailable.
//

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

int main()
{
    using namespace snicholls;
    std::puts("random_source test doubles");
    //! [reference]
    test_random_source a{1234}, b{1234};   // NOT cryptographic: the seed fixes every byte, for reproducible tests
    std::array<std::byte, 24> x{}, y{};
    (void)a.fill(x.data(), x.size());
    (void)b.fill(y.data(), y.size());
    unavailable_random_source unseeded;    // proves a caller's failure path fails closed
    //! [reference]
    check(x == y, "the same seed gives the same bytes");
    check(snicholls::random_failed(unseeded.fill(x.data(), x.size())), "the unavailable source always fails");
    std::printf("%s\n", failures == 0 ? "all checks passed" : "FAILED");
    return failures == 0 ? 0 : 1;
}
