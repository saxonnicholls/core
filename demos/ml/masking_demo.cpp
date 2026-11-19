//
//  masking_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/ml/masking.hpp>: I-JEPA's four target blocks and its context on a 14 x 14 grid, drawn as a
//  picture; a block's size from one draw; V-JEPA's short- and long-range tubes; a random mask; the collation of a
//  batch's context lists by truncation and by padding. Exits non-zero if a result is not what it documents.
//

#include <snicholls/core/ml/masking.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <vector>

int main()
{
    std::printf("snicholls-core masking demo\n");
    namespace ml = snicholls::ml;
    bool ok = true;

    //! [ijepa]
    // I-JEPA's mask for sample 0 of step 5 (the paper's settings): 4 target blocks and a context without them
    const ml::ijepa_mask_config config{};
    std::vector<std::uint32_t> context(196), targets(4 * 196);
    std::array<std::size_t, 4> target_counts{};
    const ml::mask_counts counts = ml::ijepa_masks(config, 2026, 5, 0, context, targets, target_counts);
    // The size every target block of step 5 has, from the step's draw
    const ml::block_extent extent = ml::sample_block_extent(config.grid, config.targets, 0.5, ml::block_rule::ijepa);
    //! [ijepa]
    std::array<char, 196> picture{};
    picture.fill('.');
    for (std::size_t b = 0; b < 4; ++b)
        for (std::size_t i = 0; i < target_counts[b]; ++i)
            picture[targets[b * 196 + i]] = char('1' + b);
    for (std::size_t i = 0; i < counts.context; ++i)
        picture[context[i]] = picture[context[i]] == '.' ? 'c' : '!';
    for (std::size_t y = 0; y < 14; ++y)
        std::printf("    %.*s\n", 14, picture.data() + y * 14);
    std::printf("  context %zu tokens, targets %zu in 4 blocks of %zu; a mid-range draw's block is %zu x %zu\n", counts.context, counts.targets,
                target_counts[0], extent.height, extent.width);
    for (const char c : picture)
        ok &= c != '!';   // context and targets are disjoint
    ok &= counts.context > config.min_keep && target_counts[0] == target_counts[3] && extent.tokens() > 29 && extent.tokens() < 40;

    // V-JEPA's two masks over 8 frame groups: tubes through every frame, the rest the context
    std::vector<std::uint32_t> vctx(8 * 196), vtgt(8 * 196);
    const auto short_range = ml::vjepa_masks(ml::vjepa_short_range, 2026, 5, 0, vctx, vtgt);
    const auto long_range = ml::vjepa_masks(ml::vjepa_long_range, 2026, 5, 0, vctx, vtgt);
    std::printf("  V-JEPA: short range hides %zu of 1568, long range %zu\n", short_range.targets, long_range.targets);
    ok &= short_range.context + short_range.targets == 1568 && long_range.targets > short_range.targets && long_range.context > 0;

    // MAE's random mask: three quarters hidden
    const auto random = ml::random_masks(196, 0.75, 2026, 5, 0, context, targets);
    std::printf("  random: %zu visible, %zu hidden\n", random.context, random.targets);
    ok &= random.context == 49 && random.targets == 147;

    //! [collate]
    // A batch of four: the context counts differ, so cut every list to the smallest (the papers' rule) ...
    std::vector<std::uint32_t> lists(4 * 196);
    std::array<std::size_t, 4> context_counts{}, unused{};
    std::vector<std::uint32_t> scratch(4 * 196);
    for (std::size_t s = 0; s < 4; ++s)
        context_counts[s] = ml::ijepa_masks(config, 2026, 5, s, std::span<std::uint32_t>(lists).subspan(s * 196, 196), scratch, unused).context;
    const std::size_t width = ml::common_count(context_counts);
    std::vector<std::uint32_t> equal(4 * width);
    ml::collate_indices(lists, 196, context_counts, width, equal, {});
    // ... or pad them to the longest and mask the padding (attention_activations::lengths, the losses' lengths)
    const std::size_t longest = std::max(std::max(context_counts[0], context_counts[1]), std::max(context_counts[2], context_counts[3]));
    std::vector<std::uint32_t> padded(4 * longest);
    std::array<std::size_t, 4> lengths{};
    ml::collate_indices(lists, 196, context_counts, longest, padded, lengths);
    //! [collate]
    std::printf("  batch context counts %zu %zu %zu %zu: cut to %zu, or padded to %zu\n", context_counts[0], context_counts[1], context_counts[2],
                context_counts[3], width, longest);
    ok &= lengths == context_counts && equal[0] == lists[0] && padded[longest - 1] == lists[context_counts[0] - 1];

    std::printf(ok ? "masking demo: ok\n" : "masking demo: FAILED\n");
    return ok ? 0 : 1;
}
