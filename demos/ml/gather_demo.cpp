//
//  gather_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/ml/gather.hpp>: a batch of two sequences of four token rows; the context rows of each
//  gathered by its index list and the gradient scattered back; rows written to positions; two parts concatenated per
//  sequence and sliced apart, with the slice's backward. Exits non-zero if a result is not what it documents.
//

#include <snicholls/core/ml/gather.hpp>

#include <snicholls/core/linalg/reference.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>

int main()
{
    std::printf("snicholls-core gather demo\n");
    namespace ml = snicholls::ml;
    const auto& backend = snicholls::linalg::reference;
    bool ok = true;

    //! [gather]
    // Two sequences of four rows of two values: row r of sequence s is (10 s + r, -(10 s + r))
    std::array<float, 2 * 4 * 2> tokens{};
    for (std::size_t s = 0; s < 2; ++s)
        for (std::size_t r = 0; r < 4; ++r) {
            tokens[(s * 4 + r) * 2] = float(10 * s + r);
            tokens[(s * 4 + r) * 2 + 1] = -float(10 * s + r);
        }
    // Each sequence's context: positions within the sequence, sorted
    const std::array<std::uint32_t, 4> context{0, 3, 1, 2};
    std::array<float, 4 * 2> rows{};
    ml::gather_rows(backend, std::span<const float>(tokens), context, std::span<float>(rows), 2, 2);   // rows 0, 3 | 11, 12

    // The backward: each gathered row's gradient added back to its position, in a fixed order
    const std::array<float, 4 * 2> drows{1, 1, 2, 2, 3, 3, 4, 4};
    std::array<float, 2 * 4 * 2> dtokens{};
    ml::scatter_add_rows(backend, std::span<const float>(drows), context, std::span<float>(dtokens), 2, 2);
    //! [gather]
    std::printf("  gathered first values: %g %g | %g %g\n", double(rows[0]), double(rows[2]), double(rows[4]), double(rows[6]));
    ok &= rows[0] == 0 && rows[2] == 3 && rows[4] == 11 && rows[6] == 12 && rows[7] == -12;
    ok &= dtokens[3 * 2] == 2 && dtokens[(4 + 1) * 2] == 3 && dtokens[1 * 2] == 0;

    // Rows written to positions (the backward is a gather), then two parts per sequence side by side and apart again
    std::array<float, 2 * 4 * 2> placed{};
    ml::scatter_rows(backend, std::span<const float>(rows), context, std::span<float>(placed), 2, 2);
    ok &= placed[3 * 2] == 3 && placed[(4 + 2) * 2] == 12 && placed[2 * 2] == 0;
    const std::array<float, 2 * 1 * 2> extra{100, 100, 200, 200};
    std::array<float, 2 * 3 * 2> joined{};
    ml::concat_rows(backend, std::span<const float>(rows), std::span<const float>(extra), std::span<float>(joined), 2, 2);
    std::array<float, 2 * 1 * 2> tail{};
    ml::slice_rows(backend, std::span<const float>(joined), 2, 1, std::span<float>(tail), 2, 2);
    std::array<float, 2 * 3 * 2> djoined{};
    ml::slice_rows_backward(backend, std::span<const float>(tail), 2, 1, std::span<float>(djoined), 2, 2);
    std::printf("  joined: %g %g %g | %g %g %g; sliced back %g %g\n", double(joined[0]), double(joined[2]), double(joined[4]), double(joined[6]),
                double(joined[8]), double(joined[10]), double(tail[0]), double(tail[2]));
    ok &= joined[4] == 100 && joined[10] == 200 && tail[0] == 100 && tail[2] == 200 && djoined[4] == 100 && djoined[0] == 0;

    std::printf(ok ? "gather demo: ok\n" : "gather demo: FAILED\n");
    return ok ? 0 : 1;
}
