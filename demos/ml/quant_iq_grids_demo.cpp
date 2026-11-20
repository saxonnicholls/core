//
//  quant_iq_grids_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/ml/quant_iq_grids.hpp>: the IQ codebooks the compiler builds from their compact index
//  lists, one point of each decoded, a packed point found on the grid and one that is not, and the sign table's
//  parity. Exits non-zero if a result is not what it documents.
//

#include <snicholls/core/ml/quant_iq_grids.hpp>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>

int main()
{
    std::printf("snicholls-core quant_iq_grids demo\n");
    namespace ml = snicholls::ml;
    bool ok = true;

    //! [grids]
    // Point 1 of IQ2_XXS's grid: index 2 is level 2 in coordinate 0, so its bytes are 43, 8, 8, ...
    static_assert(ml::iq2xxs_grid_index[1] == 2 && ml::iq2xxs_grid[1] == 0x080808080808082bULL);

    // The quantiser's view of the same point, 2 · level + 1: 5, 1, 1, ...
    constexpr int c0 = ml::iq_grid_coordinate(ml::iq2xxs_grid_index, 2, 1, 0);

    // ggml's kmap: a packed point's grid index, or -1 off the grid
    constexpr int on = ml::iq_grid_find(ml::iq3xxs_grid_index, ml::iq3xxs_grid_index[100]);
    constexpr int off = ml::iq_grid_find(ml::iq3xxs_grid_index, 1);

    // 7 sign bits and their parity: entry 1 (one minus) gains an eighth minus sign
    constexpr std::uint8_t signs = ml::iq2_signs[1];
    //! [grids]

    std::printf("  grid points: IQ2_XXS %zu, IQ2_XS %zu, IQ2_S %zu, IQ1_S %zu, IQ3_XXS %zu, IQ3_S %zu\n", ml::iq2xxs_grid.size(),
                ml::iq2xs_grid.size(), ml::iq2s_grid.size(), ml::iq1s_grid.size(), ml::iq3xxs_grid.size(), ml::iq3s_grid.size());
    std::printf("  IQ2_XXS point 1: %016llx, coordinate 0 as the quantiser sees it: %d\n", static_cast<unsigned long long>(ml::iq2xxs_grid[1]), c0);
    std::printf("  IQ3_XXS: point 100 found at %d, packed 1 found at %d\n", on, off);
    std::printf("  signs[1] = %u (%d minus signs)\n", unsigned(signs), std::popcount(unsigned(signs)));
    std::printf("  IQ4_NL values: %d .. %d\n", int(ml::iq4nl_values[0]), int(ml::iq4nl_values[15]));
    ok &= c0 == 5 && on == 100 && off == -1 && signs == 129 && ml::iq4nl_values[0] == -127;

    std::printf(ok ? "quant_iq_grids demo: ok\n" : "quant_iq_grids demo: FAILED\n");
    return ok ? 0 : 1;
}
