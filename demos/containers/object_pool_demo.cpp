//
//  object_pool_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/containers/object_pool.hpp>.
//

// Demo: snicholls::containers::object_pool - chunked slots, stable addresses, O(1) create/destroy with LIFO reuse, no
// per-object heap allocation. Exits non-zero if a result is not as printed.

#include <cstddef>
#include <cstdio>
#include <memory_resource>
#include <vector>

#include <snicholls/core/containers/object_pool.hpp>

namespace {
int failures = 0;
void expect(bool ok, const char* what)
{
    std::printf("  %-56s %s\n", what, ok ? "ok" : "FAILED");
    if (!ok) ++failures;
}
struct order
{
    long id;
    long quantity;
    order(long i, long q) noexcept : id(i), quantity(q) {}
};
} // namespace

int main()
{
    std::printf("object_pool<order, 1024>\n");
    //! [pool]
    snicholls::containers::object_pool<order, 1024> pool;
    order* a = pool.create(1, 100);   // constructed in place in a pooled slot
    order* b = pool.create(2, 200);
    pool.destroy(a);                  // slot goes back to the free list
    order* c = pool.create(3, 300);   // ... and is reused first
    //! [pool]
    expect(c == a && b->quantity == 200, "a released slot is reused first (LIFO)");
    expect(pool.live() == 2 && pool.capacity() == 1024, "one chunk of 1024 slots, two live objects");
    std::vector<order*> many;
    for (long i = 0; i < 5000; ++i) many.push_back(pool.create(i, i));
    expect(pool.capacity() == 5120 && many[4999]->id == 4999 && b->id == 2, "grows by chunks; addresses never move");
    pool.reserve(10000);
    expect(pool.capacity() >= 10000, "reserve");
    pool.release_all();
    expect(pool.live() == 0, "release_all keeps the chunks for reuse");

    std::printf("object_pool<order, 64> over a monotonic_buffer_resource\n");
    //! [resource]
    alignas(std::max_align_t) static std::byte arena_bytes[64 * 1024];
    std::pmr::monotonic_buffer_resource arena(arena_bytes, sizeof arena_bytes, std::pmr::null_memory_resource());
    snicholls::containers::object_pool<order, 64> in_arena(&arena);   // chunks come from the arena, not the heap
    order* o = in_arena.create(7, 700);
    //! [resource]
    const auto* raw = reinterpret_cast<const std::byte*>(o);
    expect(raw >= arena_bytes && raw < arena_bytes + sizeof arena_bytes && o->quantity == 700,
           "the object lives inside the caller's arena");
    expect(in_arena.resource() == &arena, "resource() names where chunks come from");
    return failures == 0 ? 0 : 1;
}
