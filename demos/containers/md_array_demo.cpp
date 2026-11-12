//
//  md_array_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/containers/md_array.hpp>: a scenario x tenor x strike grid with
//  compile-time strides, one matrix seen through each layout, and a sub-view of a batch. Exits
//  non-zero if a cell is not the one documented.
//

#include <snicholls/core/containers/md_array.hpp>

#include <array>
#include <cstdio>
#include <utility>

int main()
{
    std::printf("snicholls-core md_array demo\n");
    //! [md_array]
    // 3 scenarios x 4 tenors x 5 strikes, one block, strides known to the compiler.
    snicholls::containers::md_array<double, 3, 4, 5> pnl{};
    snicholls::containers::md_array<double, 3, 4, 5>::for_each_index([&](auto s, auto t, auto k) {
        pnl(s(), t(), k()) = 10.0 * static_cast<double>(s) - static_cast<double>(t) + 0.5 * static_cast<double>(k);
    });
    //! [md_array]
    std::printf("  scenario 2, tenor 3, strike 4: %.1f\n", pnl(2, 3, 4));
    bool ok = pnl(2, 3, 4) == 19.0;

    using namespace snicholls::containers;
    //! [layouts]
    // The same six numbers, read as a 2 x 3 matrix three ways.
    std::array<double, 6> storage{1, 2, 3, 4, 5, 6};
    const md_span<double, 2> rows(storage.data(), {2, 3});                            // row-major: 1 2 3 / 4 5 6
    const md_span<double, 2, layout_left> columns(storage.data(), {2, 3});            // column-major: 1 3 5 / 2 4 6
    const md_span<double, 2, layout_stride> transpose(storage.data(), {3, 2}, {1, 3});   // rows, transposed
    const md_span<const double, 2, layout_stride> any = columns;                      // converts, as std::mdspan's do
    //! [layouts]
    std::printf("  (1, 0): row-major %.0f, column-major %.0f, transpose(0, 1) %.0f, strided %.0f\n", rows(1, 0), columns(1, 0),
                transpose(0, 1), any(1, 0));
    ok = ok && rows(1, 0) == 4 && columns(1, 0) == 2 && transpose(0, 1) == 4 && any(1, 0) == 2 && rows.stride(0) == 3 &&
         columns.stride(1) == 2 && rows.extent(1) == 3 && rows.extents()[0] == 2 && md_span<double, 2>::rank() == 2 &&
         rows.size() == 6 && !rows.empty() && rows.data() == rows.data_handle() && rows.is_exhaustive() &&
         rows.mapping().required_span_size() == 6;

    //! [submdspan]
    // Three 2 x 2 systems, stored one after another; the second system, its first column, and a block of rows.
    std::array<double, 12> batch_storage{};
    for (std::size_t i = 0; i < batch_storage.size(); ++i)
        batch_storage[i] = static_cast<double>(i);
    const md_span<double, 3> batch(batch_storage.data(), {3, 2, 2});
    const auto second = submdspan(batch, 1, full_extent, full_extent);   // md_span<double, 2, layout_stride>
    const auto first_column = submdspan(second, full_extent, 0);
    const auto last_two = submdspan(batch, std::pair<std::size_t, std::size_t>{1, 3}, full_extent, full_extent);
    //! [submdspan]
    std::printf("  system 1 is [[%.0f %.0f] [%.0f %.0f]], its first column (%.0f, %.0f), systems 1..2: %zu of them\n", second(0, 0),
                second(0, 1), second(1, 0), second(1, 1), first_column(0), first_column(1), last_two.extent(0));
    ok = ok && second(1, 1) == 7 && first_column(1) == 6 && last_two(1, 0, 0) == 8 && last_two.extent(0) == 2;
    const md_span<double, 2, layout_stride> empty;
    ok = ok && empty.empty() && layout_stride::mapping<2>({2, 2}, {2, 1}).is_exhaustive();
    std::printf(ok ? "md_array demo: ok\n" : "md_array demo: FAILED\n");
    return ok ? 0 : 1;
}
