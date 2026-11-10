//
//  compare_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/meta/compare.hpp>: a generic three-way comparison over a key of several parts, some with only <. Exits non-zero if an ordering is not the one documented.
//

#include <snicholls/core/meta/compare.hpp>

#include <compare>
#include <cstdio>
#include <string>

namespace {

struct legacy_id {   // an old type with only operator<
    int value;
    friend bool operator<(const legacy_id& a, const legacy_id& b) noexcept { return a.value < b.value; }
};

//! [compare]
// A key of two parts, ordered as std::tuple would order them: the first part that differs decides.
struct key {
    std::string venue;
    legacy_id id;
    friend std::weak_ordering operator<=>(const key& a, const key& b)
    {
        if (const auto c = snicholls::synth_three_way(a.venue, b.venue); c != 0)
            return c;
        return snicholls::synth_three_way(a.id, b.id);   // only <, so a weak ordering from two of them
    }
    friend bool operator==(const key& a, const key& b) { return (a <=> b) == 0; }
};
//! [compare]

} // namespace

int main()
{
    std::printf("snicholls-core compare demo\n");
    bool ok = true;

    //! [compare_use]
    const bool earlier = key{"LSE", {7}} < key{"LSE", {9}};
    //! [compare_use]

    std::printf("  LSE/7 before LSE/9: %d\n", earlier);
    ok &= earlier && !(key{"NYSE", {1}} < key{"LSE", {9}});
    ok &= key{"LSE", {7}} == key{"LSE", {7}};

    std::printf(ok ? "compare demo: ok\n" : "compare demo: FAILED\n");
    return ok ? 0 : 1;
}
