//
//  bloom_filter_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/containers/bloom_filter.hpp>: a set of watched
//  addresses in front of the authoritative map, so a stream of events that are
//  mostly not ours is rejected without a lookup. Exits non-zero if a member is
//  ever missed or the false-positive rate is far from its target.
//

#include <snicholls/core/containers/bloom_filter.hpp>

#include <cstddef>
#include <cstdio>
#include <string>
#include <unordered_map>

int main()
{
    std::printf("snicholls-core bloom_filter demo\n");
    bool ok = true;

    //! [watch]
    // The authoritative map: 100,000 watched addresses and who owns each.
    std::unordered_map<std::string, int> owners;
    for (int i = 0; i < 100'000; ++i)
        owners.emplace("0x" + std::to_string(i * 2654435761u), i);

    // The filter in front of it, sized for those keys at a 0.5% false-positive rate.
    snicholls::containers::bloom_filter watched(owners.size(), 0.005);
    for (const auto& [address, owner] : owners)
        watched.add(address);
    std::printf("  %zu addresses, %zu KB, %u hashes per key\n", owners.size(), watched.memory_bytes() / 1024, watched.hashes());

    // A stream of events, half of them ours: "no" is final, "maybe" asks the map.
    std::size_t ours = 0, rejected_by_filter = 0, rejected_by_map = 0;
    for (int i = 0; i < 200'000; ++i) {
        const bool watched_address = i < 100'000;
        const std::string address = (watched_address ? "0x" : "0y") + std::to_string(i * 2654435761u);
        if (!watched.maybe_contains(address)) {
            ++rejected_by_filter;   // certain, and no lookup
            continue;
        }
        if (owners.contains(address))
            ++ours;
        else
            ++rejected_by_map;   // a false positive: one wasted lookup
    }
    //! [watch]

    std::printf("  200,000 events: %zu ours, %zu rejected by the filter, %zu false positives (%.2f%%)\n", ours,
                rejected_by_filter, rejected_by_map, 100.0 * static_cast<double>(rejected_by_map) / 100'000.0);
    std::printf("  estimated false-positive rate for %zu keys: %.3f%%\n", watched.count(),
                100.0 * watched.estimated_false_positive_rate());

    ok &= ours == 100'000;                            // no false negatives: every watched event reached its owner
    ok &= rejected_by_map < 2'000;                    // 0.5% of 100,000 strangers, with room: never more than 2%
    ok &= rejected_by_filter + rejected_by_map == 100'000;

    std::printf(ok ? "bloom_filter demo: ok\n" : "bloom_filter demo: FAILED\n");
    return ok ? 0 : 1;
}
