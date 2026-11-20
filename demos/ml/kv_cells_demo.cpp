//
//  kv_cells_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/ml/kv_cells.hpp>: two sequences sharing a prompt in one cache, a context shift that drops
//  the oldest tokens and re-rotates the cached keys (the K-shift), and attention over the cells a query sees. Exits
//  non-zero if a result is not what it documents.
//

#include <snicholls/core/ml/kv_cells.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <span>
#include <vector>

int main()
{
    std::printf("snicholls-core kv_cells demo\n");
    namespace ml = snicholls::ml;
    bool ok = true;

    //! [kv_cells]
    // One layer, one key-value head of 4 channels, 8 slots, RoPE on
    constexpr ml::transformer_config c{.d_model = 4, .n_heads = 1, .n_kv_heads = 1, .head_dim = 4, .d_ff = 4, .rope = ml::rope_kind::half};
    std::vector<float> keys(ml::kv_cache_view<c, float>::required_size(1, 8)), values(keys.size());
    const ml::kv_cache_view<c, float> cache(keys, values, 1, 8);
    ml::kv_cells<8> cells;

    // A prompt of three tokens for sequence 0, rotated at their positions, then shared with sequence 1
    auto write = [&](std::size_t slot, ml::kv_pos pos, ml::kv_seq seq, float x) {
        std::array<float, 4> k{x, 1.0f, -x, 0.5f}, v{x, x, x, x};
        std::array<float, 2> cs{}, sn{};
        ml::rope_angles(std::size_t(pos), cs, sn);
        ml::rope(k, cs, sn);
        cache.append(0, slot, k, v);   // the slot, not the position: cells say which position it holds
        cells.place(slot, pos, seq);
    };
    for (std::size_t t = 0; t < 3; ++t)
        write(t, ml::kv_pos(t), 0, float(t + 1));
    cells.seq_cp(0, 1, -1, -1);
    write(cells.find_free(), 3, 0, 4.0f);   // sequence 0 continues; sequence 1 does not see this token
    write(cells.find_free(), 3, 1, 9.0f);   // nor does sequence 0 see this one

    // A query of sequence 1 at position 3 attends to the shared prompt and its own token
    const std::array<float, 4> q{0.2f, 0.1f, -0.3f, 0.4f};
    std::array<float, 4> out{};
    ml::attend_cells(cache, cells, 0, 0, q, out, 1, 3);

    // Context shift for sequence 0: drop position 0, move the rest back by one, re-rotate the keys (a position belongs to
    // the cell, so the shared cells move for sequence 1 too, as in llama.cpp)
    cells.seq_rm(0, 0, 1);
    cells.seq_add(0, 1, -1, -1);
    ml::kv_shift_keys<c>(std::span<float>(keys), 1, cells);
    //! [kv_cells]

    std::printf("  used %zu slots; sequence 0 at positions %d to %d, sequence 1 to %d\n", cells.used(), cells.seq_pos_min(0),
                cells.seq_pos_max(0), cells.seq_pos_max(1));
    // Slot 0 is still sequence 1's (shared), so not freed; sequence 0's positions are now 0 to 2
    ok &= cells.used() == 5 && cells.seq_pos_min(0) == 0 && cells.seq_pos_max(0) == 2 && cells.seq_pos_max(1) == 3;
    std::printf("  sequence 1's query: out %.4f (a mix of 1, 2, 3 and 9)\n", static_cast<double>(out[0]));
    ok &= out[0] > 1.0f && out[0] < 9.0f && !cells.has_shift();
    // Slot 3 held sequence 0's token at position 3, rotated there; it now sits at 2, as if rotated at 2
    std::array<float, 4> fresh{4.0f, 1.0f, -4.0f, 0.5f};
    std::array<float, 2> cs{}, sn{};
    ml::rope_angles(2, cs, sn);
    ml::rope(fresh, cs, sn);
    const auto moved = cache.visible(0, 0, ml::key_range{3, 4}).keys[0];
    std::printf("  K-shift: key at slot 3 %.5f %.5f, rotated afresh at 2 %.5f %.5f\n", static_cast<double>(moved[0]),
                static_cast<double>(moved[1]), static_cast<double>(fresh[0]), static_cast<double>(fresh[1]));
    for (std::size_t i = 0; i < 4; ++i)
        ok &= std::abs(moved[i] - fresh[i]) < 1e-5f;

    std::printf(ok ? "kv_cells demo: ok\n" : "kv_cells demo: FAILED\n");
    return ok ? 0 : 1;
}
