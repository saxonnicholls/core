//
//  index_pool_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/containers/index_pool.hpp>.
//

// Demo: snicholls::containers::index_pool - slots addressed by 32-bit index inside a caller's region, the pool's state
// in the region itself, so the region can be copied, grown or reopened and every index still names the same record.
// Exits non-zero if a result is not as printed.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include <snicholls/core/containers/index_pool.hpp>

namespace {
int failures = 0;
void expect(bool ok, const char* what)
{
    std::printf("  %-60s %s\n", what, ok ? "ok" : "FAILED");
    if (!ok) ++failures;
}
struct order
{
    std::int64_t id;
    std::int32_t quantity;
    std::int32_t price;
};
using pool_t = snicholls::containers::index_pool<order>;

// Aligned storage standing in for a mapping.
struct buffer
{
    std::vector<std::uint64_t> words;
    explicit buffer(std::size_t bytes) : words((bytes + 7) / 8) {}
    std::span<std::byte> span() { return std::as_writable_bytes(std::span(words)); }
};
} // namespace

int main()
{
    std::printf("index_pool<order>: slot %zu bytes, state %zu bytes\n", pool_t::slot_size, pool_t::header_size);
    buffer small(pool_t::bytes_for(4));
    //! [pool]
    auto made = snicholls::containers::index_pool<order>::format(small.span());   // an empty pool in the region
    if (made.is_err()) return 1;
    auto pool = *made;
    std::uint32_t a = *pool.create(order{1, 100, 990});
    std::uint32_t b = *pool.create(order{2, 200, 991});
    pool.destroy(a);                                   // index 0 goes on the free list ...
    std::uint32_t c = *pool.create(order{3, 300, 992});   // ... and is reused first
    //! [pool]
    expect(a == 0 && b == 1 && c == 0, "indices from 0, a freed index reused first");
    expect(pool[b].quantity == 200 && pool[c].id == 3, "pool[i] is the record");
    (void)pool.create(order{4, 400, 993});
    (void)pool.create(order{5, 500, 994});
    expect(pool.create(order{6, 600, 995}).err() == snicholls::containers::index_pool_error::full,
           "a full region says full; nothing is allocated behind your back");

    //! [rebase]
    buffer big(pool_t::bytes_for(1024));
    std::memcpy(big.span().data(), small.span().data(), small.span().size());   // grow: copy, or remap a file
    if (pool.rebase(big.span()).is_err()) return 1;
    std::uint32_t d = *pool.create(order{6, 600, 995});
    //! [rebase]
    expect(pool.capacity() == 1024 && d == 4 && pool[b].id == 2, "after rebase the same indices name the same records");

    auto reopened = pool_t::attach(big.span());   // as a later run of the program would
    expect(reopened.is_ok() && reopened->live() == 5 && (*reopened)[d].quantity == 600,
           "attach finds the pool, its counters and its free list");
    big.span()[0] = std::byte{0};
    expect(pool_t::attach(big.span()).err() == snicholls::containers::index_pool_error::not_formatted,
           "a region that is not a pool is refused");
    return failures == 0 ? 0 : 1;
}
