//
//  failing_new.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  The replaced global operator new / delete behind failing_new.hpp. Linked
//  into snicholls_core_allocation_tests only.
//

#include "failing_new.hpp"

#include <atomic>
#include <cstdlib>
#include <new>

namespace {

std::atomic<long> countdown{0};   // 0: off; n > 0: the allocation that takes it to 0 throws

void* allocate(std::size_t n)
{
    if (countdown.load(std::memory_order_relaxed) > 0 && countdown.fetch_sub(1, std::memory_order_relaxed) == 1)
        throw std::bad_alloc();
    if (void* p = std::malloc(n ? n : 1))
        return p;
    throw std::bad_alloc();
}

} // namespace

namespace snicholls_test {

void fail_nth_allocation(long n) noexcept { countdown.store(n, std::memory_order_relaxed); }
void stop_failing_allocations() noexcept { countdown.store(0, std::memory_order_relaxed); }

} // namespace snicholls_test

void* operator new(std::size_t n) { return allocate(n); }
void* operator new[](std::size_t n) { return allocate(n); }
void* operator new(std::size_t n, const std::nothrow_t&) noexcept
{
    try { return allocate(n); } catch (...) { return nullptr; }
}
void* operator new[](std::size_t n, const std::nothrow_t&) noexcept
{
    try { return allocate(n); } catch (...) { return nullptr; }
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
void operator delete(void* p, const std::nothrow_t&) noexcept { std::free(p); }
void operator delete[](void* p, const std::nothrow_t&) noexcept { std::free(p); }
