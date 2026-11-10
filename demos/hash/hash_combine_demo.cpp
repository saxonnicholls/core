//
//  hash_combine_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/hash/hash_combine.hpp>: a hash for a plain struct built
//  from its members' hashes, used to key an unordered_set. Exits non-zero if
//  any result is not the one documented.
//

#include <snicholls/core/hash/hash_combine.hpp>

#include <cstddef>
#include <cstdio>
#include <functional>
#include <string>
#include <unordered_set>

//! [struct]
struct quote_key {
    std::string symbol;
    char side;
    long level;

    bool operator==(const quote_key&) const = default;
};

struct quote_key_hash {
    std::size_t operator()(const quote_key& k) const noexcept
    {
        std::size_t seed = 0;   // every member, in declaration order, from seed 0
        seed = snicholls::hash_combine(seed, std::hash<std::string>{}(k.symbol));
        seed = snicholls::hash_combine(seed, std::hash<char>{}(k.side));
        seed = snicholls::hash_combine(seed, std::hash<long>{}(k.level));
        return seed;
    }
};

std::unordered_set<quote_key, quote_key_hash> seen;
//! [struct]

int main()
{
    std::printf("snicholls-core hash_combine demo\n");
    bool ok = true;

    seen.insert({"ACME", 'B', 100});
    seen.insert({"ACME", 'S', 100});
    seen.insert({"ACME", 'B', 100});   // a duplicate: equal keys hash equal
    std::printf("  distinct keys: %zu\n", seen.size());
    ok &= seen.size() == 2;
    ok &= seen.contains({"ACME", 'S', 100});

    // Order matters along a chain: the same values in another order hash differently
    const std::size_t ab = snicholls::hash_combine(snicholls::hash_combine(0, 1), 2);
    const std::size_t ba = snicholls::hash_combine(snicholls::hash_combine(0, 2), 1);
    std::printf("  chain 1,2: %zx  chain 2,1: %zx\n", ab, ba);
    ok &= ab != ba;

    // constexpr: usable to hash compile-time data
    constexpr std::size_t compile_time = snicholls::hash_combine(0, 42);
    static_assert(compile_time == snicholls::hash_combine(0, 42));
    ok &= compile_time == snicholls::hash_combine(0, 42);

    std::printf(ok ? "hash_combine demo: ok\n" : "hash_combine demo: FAILED\n");
    return ok ? 0 : 1;
}
