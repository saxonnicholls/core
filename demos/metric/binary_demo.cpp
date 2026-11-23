//
//  binary_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/metric/binary.hpp>: 128-bit codes packed in words compared by hamming and jaccard, the
//  bit_vector concept (unsigned words yes, signed words and std::vector<bool> no), a nearest-code search, and
//  mismatches between strings and categorical vectors. Exits non-zero if a result is not what it documents.
//

#include <snicholls/core/metric/binary.hpp>

#include <snicholls/core/metric/concepts.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string_view>
#include <vector>

namespace mt = snicholls::metric;

int main()
{
    int failures = 0;
    const auto check = [&](bool ok, const char* what, double value) {
        std::printf("  %-56s %10.6f  %s\n", what, value, ok ? "ok" : "FAILED");
        failures += ok ? 0 : 1;
    };

    //! [binary]
    // 128-bit codes (a locality-sensitive hash, say) as two 64-bit words; any unsigned words, any point kind
    using code = std::array<std::uint64_t, 2>;
    const code a{0b1011, 0xFF00};
    const std::vector<std::uint16_t> b{0b0110, 0x0F00};   // the same bits, in narrower words

    // XOR 0b1101, 0xF000: 7 bits differ; AND has 5 bits set, OR 12 (python3 bin().count)
    std::puts("Codes {0b1011, 0xFF00} and {0b0110, 0x0F00}");
    check(mt::hamming{}(a, b) == 7, "hamming = 7 differing bits", double(mt::hamming{}(a, b)));
    check(mt::jaccard{}(a, b) == 1.0 - 5.0 / 12.0, "jaccard = 1 - 5/12", mt::jaccard{}(a, b));

    // The nearest stored code by Hamming distance
    const std::vector<code> stored{{0, 0}, {0b1111, 0xFF00}, {~0ull, ~0ull}};
    std::size_t best = 0;
    for (std::size_t i = 1; i < stored.size(); ++i)
        if (mt::hamming{}(a, stored[i]) < mt::hamming{}(a, stored[best]))
            best = i;
    check(best == 1 && mt::hamming{}(a, stored[1]) == 1, "nearest stored code: {0b1111, 0xFF00}, 1 bit away", double(best));

    // mismatches: the Hamming distance between sequences of any coordinate type
    check(mt::mismatches{}(std::string_view("karolin"), std::string_view("kathrin")) == 3, "mismatches(karolin, kathrin) = 3", 3.0);
    const std::array<int, 4> colours{0, 2, 1, 1}, other{0, 1, 1, 2};
    check(mt::mismatches{}(colours, other) == 2, "mismatches of categorical vectors = 2", 2.0);
    //! [binary]

    std::puts("What is a bit vector");
    check(mt::bit_vector<code> && mt::bit_vector<std::vector<std::uint8_t>> && mt::bit_vector<std::uint32_t[4]>, "unsigned words of any width and kind", 1.0);
    check(!mt::bit_vector<std::array<std::int64_t, 2>> && !mt::bit_vector<std::vector<bool>> && !mt::bit_vector<int>,
          "not signed words, std::vector<bool> or a plain int", 0.0);
    check(mt::true_metric<mt::hamming> && mt::true_metric<mt::jaccard> && mt::true_metric<mt::mismatches>, "all three are true metrics", 1.0);

    std::puts("Edge cases");
    const code none{0, 0};
    check(mt::jaccard{}(none, none) == 0.0 && mt::jaccard{}(none, a) == 1.0, "jaccard: two empty sets 0, empty against non-empty 1", 0.0);
    check(mt::hamming{}(code{~0ull, ~0ull}, none) == 128, "hamming of all ones against zeros = 128", 128.0);

    std::printf("%s\n", failures == 0 ? "all results as documented" : "SOME RESULTS DIFFER");
    return failures == 0 ? 0 : 1;
}
