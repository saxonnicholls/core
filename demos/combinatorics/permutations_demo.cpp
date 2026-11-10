//
//  permutations_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/combinatorics/permutations.hpp>: tables built at compile time, and a walk that
//  stores none, and the orders printed with each member in its own colour. Exits non-zero if a row is not the
//  one documented.
//

#include <snicholls/core/acout.hpp>
#include <snicholls/core/combinatorics/permutations.hpp>
#include <snicholls/core/display.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <sstream>
#include <string>

namespace legs_demo {

//! [permutations_colour]
// Each member in its own colour, wherever it prints.
enum class leg : std::uint8_t { A, B, C };
constexpr auto attributes(leg l)
{
    using namespace snicholls;
    return per_enumerator(l, [] { return std::array{attrs(colours::blue), attrs(colours::orange), attrs(colours::magenta)}; });
}
//! [permutations_colour]

} // namespace legs_demo

int main()
{
    std::printf("snicholls-core permutations demo\n");
    bool ok = true;

    //! [permutations]
    // The table, built at compile time: the first row is the identity, the last is reversed.
    constexpr auto orders = snicholls::index_permutations<3>();
    static_assert(orders.size() == 6 && orders[5] == std::array<std::size_t, 3>{2, 1, 0});

    // The same for values, and for 2 chosen from 3.
    constexpr auto legs = snicholls::permutations(std::array{'A', 'B', 'C'});
    constexpr auto pairs = snicholls::permutations<2>(std::array{'A', 'B', 'C'});
    static_assert(legs[1] == std::array{'A', 'C', 'B'} && pairs[5] == std::array{'C', 'B'});

    // Where a table would not fit: one row at a time, storing none.
    std::size_t rows = 0;
    snicholls::for_each_permutation<9>([&rows](const std::array<std::size_t, 9>&) { ++rows; });
    //! [permutations]

    std::string shown;
    for (const auto& row : legs) {
        shown.append(row.begin(), row.end());
        shown += ' ';
    }

    // The same orders of enumerators that carry a colour each: A, B and C keep their colours as they move.
    using legs_demo::leg;
    constexpr auto coloured = snicholls::permutations(std::array{leg::A, leg::B, leg::C});
    snicholls::acout out;
    std::ostringstream plain;
    out << "  ";
    for (const auto& row : coloured) {
        for (const leg l : row) {
            out << l;
            snicholls::print_to(plain, l);
        }
        out << ' ';
        plain << ' ';
    }
    out << '\n';
    out.emit();
    ok &= plain.str() == shown;

    std::printf("  %zu rows of 9\n", rows);
    ok &= shown == "ABC ACB BAC BCA CAB CBA ";
    ok &= rows == snicholls::permutation_count_v<9>;
    ok &= pairs.size() == snicholls::permutation_count_v<3, 2>;

    std::printf(ok ? "permutations demo: ok\n" : "permutations demo: FAILED\n");
    return ok ? 0 : 1;
}
