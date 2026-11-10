//
//  failing_new.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Allocation-failure injection for the tests that prove a strong guarantee.
//  tests/support/failing_new.cpp replaces the global operator new of the test
//  executable it is linked into (snicholls_core_allocation_tests only, never
//  the main test binary): off by default, and while a failing_new guard is
//  alive, the Nth allocation from its construction throws std::bad_alloc.
//

#ifndef snicholls_core_tests_support_failing_new_hpp
#define snicholls_core_tests_support_failing_new_hpp

namespace snicholls_test {

// Arms the injection: the nth operator new from now (1-based) throws std::bad_alloc; later ones succeed.
void fail_nth_allocation(long n) noexcept;
// Disarms it.
void stop_failing_allocations() noexcept;

// RAII: armed for the guard's lifetime.
struct failing_new
{
    explicit failing_new(long n) noexcept { fail_nth_allocation(n); }
    ~failing_new() { stop_failing_allocations(); }
    failing_new(const failing_new&) = delete;
    failing_new& operator=(const failing_new&) = delete;
};

} // namespace snicholls_test

#endif /* snicholls_core_tests_support_failing_new_hpp */
