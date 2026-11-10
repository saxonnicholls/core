//
//  member_count_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/reflect/member_count.hpp>: the member count of plain
//  aggregate structs, detected at compile time, including the member types that
//  defeat naive counting. Exits non-zero if any count is not the one documented.
//

#include <snicholls/core/reflect/member_count.hpp>

#include <array>
#include <cstddef>
#include <cstdio>
#include <memory>
#include <optional>
#include <string>
#include <variant>

namespace {

//! [count]
struct quote { long id; double px; };
static_assert(snicholls::member_count_v<quote> == 2);

// Greedy constructors, move-only and const members, nested aggregates and std::array each count as one.
struct row {
    std::optional<int> maybe;
    std::variant<int, std::string> either;
    std::unique_ptr<int> owned;
    const int fixed;
    quote nested;
    std::array<int, 3> triple;
};
static_assert(snicholls::member_count_v<row> == 6);
//! [count]

struct empty {};

} // namespace

int main()
{
    std::printf("snicholls-core member_count demo\n");
    bool ok = true;

    const std::size_t counts[] = {
        snicholls::member_count_v<quote>,
        snicholls::member_count_v<row>,
        snicholls::member_count_v<empty>,
        snicholls::member_count_v<const quote&>,   // cv and references are ignored
    };
    std::printf("  quote: %zu, row: %zu, empty: %zu, const quote&: %zu\n", counts[0], counts[1], counts[2], counts[3]);
    ok &= counts[0] == 2 && counts[1] == 6 && counts[2] == 0 && counts[3] == 2;

    // plain_aggregate is the concept member_count requires
    static_assert(snicholls::plain_aggregate<quote> && !snicholls::plain_aggregate<std::string>);
    ok &= snicholls::member_count<row>::value == snicholls::member_count_v<row>;

    std::printf("  counted up to SNICHOLLS_MAX_STRUCT_MEMBERS = %d\n", SNICHOLLS_MAX_STRUCT_MEMBERS);

    std::printf(ok ? "member_count demo: ok\n" : "member_count demo: FAILED\n");
    return ok ? 0 : 1;
}
