//
//  flat_id_map_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/containers/flat_id_map.hpp>.
//

// Demo: snicholls::containers::flat_id_map - exchange order ids to pointers in one flat array: linear probing, Fibonacci
// hashing, backward-shift deletion (no tombstones). Exits non-zero if a result is not as printed.

#include <cstdint>
#include <cstdio>

#include <snicholls/core/containers/flat_id_map.hpp>

namespace {
int failures = 0;
void expect(bool ok, const char* what)
{
    std::printf("  %-56s %s\n", what, ok ? "ok" : "FAILED");
    if (!ok) ++failures;
}
} // namespace

int main()
{
    std::printf("flat_id_map<int>\n");
    //! [map]
    snicholls::containers::flat_id_map<int> ids(1024);   // room for 1024 ids before growing
    ids.insert(1'000'001, 7);
    const bool again = ids.insert(1'000'001, 8);    // duplicate: refused, value unchanged
    int* found = ids.find(1'000'001);
    ids.erase(1'000'001);
    //! [map]
    expect(!again && found && *found == 7, "insert, duplicate refused, find");
    expect(!ids.contains(1'000'001) && ids.empty(), "erase");
    for (std::uint64_t i = 0; i < 100000; ++i) ids.insert(i, static_cast<int>(i));
    for (std::uint64_t i = 0; i < 100000; i += 2) ids.erase(i);
    long long sum = 0;   // not long: 2.5e9 overflows a 32-bit long (Windows)
    ids.for_each([&](std::uint64_t, int v) { sum += v; });
    expect(ids.size() == 50000 && sum == 2500000000LL && *ids.find(99999) == 99999, "100k sequential ids, half erased");
    expect(ids.size() * 2 <= ids.capacity(), "load factor stays <= 1/2");
    const auto cap = ids.capacity();
    ids.clear();
    ids.reserve(10);
    expect(ids.empty() && ids.capacity() == cap, "clear keeps the capacity");
    std::printf("  capacity %zu, reserved key %llu\n", ids.capacity(),
                static_cast<unsigned long long>(snicholls::containers::flat_id_map<int>::empty_key));
    return failures == 0 ? 0 : 1;
}
