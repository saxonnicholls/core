//
//  masking.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::ml` masks for joint embedding predictive architectures: I-JEPA's multi-block sampler, V-JEPA's spatio-temporal tubes, random masks, and the collation of index lists across a batch, deterministic from (seed, step, sample).
/// \ingroup core
///
/// **Overview.** A JEPA hides part of each sample and predicts the representations of the hidden part from the rest
/// (TRAINING.md T11). Which tokens are hidden is the mask; this header samples them on a grid of tokens
/// (`token_grid`: frame groups, rows, columns, raster order, as `ml/patch.hpp` numbers them) and writes, for each
/// sample, the sorted index lists of its **context** tokens (what the context encoder sees) and **target** tokens (what
/// the predictor predicts).
///
/// - `ijepa_masks(config, seed, step, sample, context, targets, target_counts)`: I-JEPA's multi-block mask (Assran et
///   al., 2023, and its code's `MaskCollator`, read 2026-11-19): `M` target blocks (4 by default) of a scale of the
///   grid in `[0.15, 0.2]` and an aspect ratio in `[0.75, 1.5]`, then one context block of scale `[0.85, 1.0]`,
///   square, with every target removed from it. A block is accepted only with more than `min_keep` (10) tokens; if
///   the context fails 20 times in a row one target stops constraining it (the code's `tries`). The block sizes are
///   drawn once per step, shared by the batch, as the code does; the positions per sample.
/// - `vjepa_masks(config, seed, step, sample, context, targets)`: V-JEPA's multi-block mask in space and time (Bardes
///   et al., 2024, and its code's `multiblock3d`, read 2026-11-19): `blocks` boxes of a spatial scale, an aspect ratio
///   and a temporal scale (1 by default: each box spans every frame group, a tube), their union the targets and its
///   complement the context, resampled if the context is empty. V-JEPA trains with two such masks at once, 8 blocks
///   of scale 0.15 ("short range") and 2 of scale 0.7 ("long range"): `vjepa_short_range` and `vjepa_long_range`.
/// - `random_masks(tokens, ratio, seed, step, sample, context, targets)`: a uniformly random fraction `ratio` of the
///   tokens hidden (MAE's random masking, `len_keep = int(tokens · (1 - ratio))`).
/// - `common_count(counts)` and `collate_indices(lists, stride, counts, width, out, lengths)`: the counts differ from
///   sample to sample, and a batch needs one shape. The papers keep the first `common_count` (the smallest count) of
///   each sorted list, so every sample has the same number of tokens (`width = common_count(counts)`); or pass a wider
///   `width` and the lists are padded (with their last index) and `lengths` says how many are real, for the attention
///   layer's key-padding mask (`attention_activations::lengths`) and the losses.
///
/// **Determinism.** Every draw comes from core's `stats::wyrand` streams: `key = wyrand::stream(seed, step)()`, the
/// step's block sizes from `wyrand::stream(key, 0)` and sample `s`'s positions from `wyrand::stream(key, s + 1)`, with
/// `stats::uniform_real` for the scales and `stats::uniform_index` for the positions. The same (seed, step, sample)
/// gives the same mask on every machine, whichever worker samples it and in whatever order.
///
/// **A block's size**, from one uniform draw `u` shared by its ranges as I-JEPA's `_sample_block_size` does: the
/// scale `s = s_min + u (s_max - s_min)`, the aspect ratio `a` and temporal scale likewise; `keep = int(rows · columns
/// · s)`, `h = round(sqrt(keep · a))`, `w = round(sqrt(keep / a))` (rounding halves to even, as Python's `round`),
/// and `frames = max(1, int(frame groups · temporal))`. Under `block_rule::ijepa` `h` and `w` are lowered until they
/// are below the grid's rows and columns and a block's top-left corner is drawn from `[0, rows - h)` (the code's
/// `randint(0, height - h)`); under `block_rule::vjepa` they are at most the grid's and the corner is drawn from `[0,
/// rows - h]`, the first frame from `[0, frame groups - frames]`. Neither paper lets a block be empty; this header
/// keeps `h`, `w` and `frames` at least 1.
///
/// \snippet ml/masking_demo.cpp ijepa
///
/// \snippet ml/masking_demo.cpp collate
///
/// **Relationships.** `stats/random.hpp` (the streams), `ml/gather.hpp` (the lists select rows), `ml/patch.hpp` (the
/// token grid), `ml/jepa.hpp`, `ml/attention_layer.hpp` (`lengths`). TRAINING.md T11. Group: \ref core.
///
/// **Implementations.** One each, on the host: masks are drawn by the data loader, before the batch moves to a device.
///
/// **Guarantees.** `constexpr`, no allocation. Lists are sorted ascending without repeats; a sample's context and
/// targets are disjoint (I-JEPA's unless `allow_overlap`); each I-JEPA target block has its sampled size (so its scale
/// and aspect ratio within their ranges, up to rounding); the union of the lists is the grid for `vjepa_masks` and
/// `random_masks`. O(blocks · tokens) per sample.
///
/// **Errors.** Preconditions, asserted: `context` holds at least the grid's tokens, `targets` (`ijepa_masks`)
/// `blocks · tokens`; at most `max_mask_blocks` blocks; under I-JEPA a target block's size above `min_keep` (else no
/// block could be accepted, as in the paper's code); a ratio in `[0, 1]`.
///
/// **Performance.** Not benchmarked: once per sample in the data loader, O(tokens · blocks).
///
/// **Tests and demos.** `tests/ml/masking_test.cpp`, `demos/ml/masking_demo.cpp`.
///
/// **Dependencies.** `<algorithm>`, `<array>`, `<cassert>`, `<cstddef>`, `<cstdint>`, `<span>`,
/// `snicholls/core/math/constexpr_math.hpp`, `stats/random.hpp` (which includes `<random>` for its concept).
///
/// **Known issues.** The draws are core's, not PyTorch's: the same configuration gives masks of the papers'
/// distribution, not their exact masks. V-JEPA's use of one draw for all three ranges is assumed from I-JEPA's code;
/// with V-JEPA's ranges (a fixed scale, a fixed temporal scale) only the aspect ratio varies, so it does not matter.
///
/// **Status.** Experimental, new in 0.16.0.

#ifndef snicholls_core_ml_masking_hpp
#define snicholls_core_ml_masking_hpp

#include <snicholls/core/math/constexpr_math.hpp>
#include <snicholls/core/stats/random.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <span>

namespace snicholls::ml
{

/// \brief A grid of tokens: frame groups, rows, columns, numbered in raster order.
/// \ingroup core
struct token_grid
{
    std::size_t frames = 1;   ///< The frame groups; 1 for an image.
    std::size_t height = 0;   ///< The rows.
    std::size_t width = 0;    ///< The columns.

    /// \brief The tokens. \return `frames · height · width`.
    [[nodiscard]] constexpr std::size_t tokens() const noexcept { return frames * height * width; }
};

/// \brief The ranges a block's size is drawn from.
/// \ingroup core
struct block_sampler
{
    double scale_min = 0.15;     ///< The smallest fraction of a frame's tokens.
    double scale_max = 0.2;      ///< The largest.
    double aspect_min = 0.75;    ///< The smallest ratio of rows to columns.
    double aspect_max = 1.5;     ///< The largest.
    double temporal_min = 1.0;   ///< The smallest fraction of the frame groups.
    double temporal_max = 1.0;   ///< The largest.
    std::size_t blocks = 4;      ///< The blocks drawn.
};

/// \brief The size of a block, in tokens.
/// \ingroup core
struct block_extent
{
    std::size_t frames = 1;   ///< Frame groups.
    std::size_t height = 1;   ///< Rows.
    std::size_t width = 1;    ///< Columns.

    /// \brief The tokens of the block. \return `frames · height · width`.
    [[nodiscard]] constexpr std::size_t tokens() const noexcept { return frames * height * width; }
};

/// \brief Which paper's rule bounds a block and places it.
enum class block_rule
{
    ijepa,   ///< Rows and columns below the grid's; the corner in `[0, rows - h)`.
    vjepa,   ///< Rows and columns at most the grid's; the corner in `[0, rows - h]`, the first frame likewise.
};

/// \brief The most blocks a mask draws.
inline constexpr std::size_t max_mask_blocks = 16;

/// \brief How many tokens a sample's lists hold.
/// \ingroup core
struct mask_counts
{
    std::size_t context = 0;   ///< The context tokens.
    std::size_t targets = 0;   ///< The target tokens (for `ijepa_masks`, of every block together).
};

/// \brief I-JEPA's settings: the paper's by default (its ViT-H/14 configuration).
/// \ingroup core
struct ijepa_mask_config
{
    token_grid grid{1, 14, 14};                                     ///< The tokens.
    block_sampler targets{0.15, 0.2, 0.75, 1.5, 1.0, 1.0, 4};       ///< The target blocks.
    block_sampler context{0.85, 1.0, 1.0, 1.0, 1.0, 1.0, 1};        ///< The context block: one, square.
    std::size_t min_keep = 10;                                      ///< A block needs more tokens than this.
    bool allow_overlap = false;                                     ///< Whether the context may keep target tokens.
};

/// \brief V-JEPA's settings for one mask.
/// \ingroup core
struct vjepa_mask_config
{
    token_grid grid{8, 14, 14};                                     ///< The tokens: 16 frames in tubelets of 2, 224 / 16.
    block_sampler targets{0.15, 0.15, 0.75, 1.5, 1.0, 1.0, 8};      ///< The blocks, their union the targets.
};

/// \brief V-JEPA's short-range mask: 8 tubes of scale 0.15.
inline constexpr vjepa_mask_config vjepa_short_range{};

/// \brief V-JEPA's long-range mask: 2 tubes of scale 0.7.
inline constexpr vjepa_mask_config vjepa_long_range{{8, 14, 14}, {0.7, 0.7, 0.75, 1.5, 1.0, 1.0, 2}};

namespace detail::masking
{
    // Python's round(): to nearest, halves to even
    constexpr double round_even(double x) noexcept
    {
        const double f = math::floor(x), r = x - f;
        if (r > 0.5)
            return f + 1;
        if (r < 0.5)
            return f;
        return math::floor(f / 2) * 2 == f ? f : f + 1;
    }

    constexpr std::uint64_t key(std::uint64_t seed, std::uint64_t step) noexcept { return stats::wyrand::stream(seed, step)(); }

    struct box
    {
        std::size_t t0, y0, x0;
        block_extent e;
        [[nodiscard]] constexpr bool contains(std::size_t t, std::size_t y, std::size_t x) const noexcept
        {
            return t >= t0 && t < t0 + e.frames && y >= y0 && y < y0 + e.height && x >= x0 && x < x0 + e.width;
        }
    };

    constexpr box place(const token_grid& g, block_extent e, block_rule rule, stats::wyrand& rng) noexcept
    {
        const std::size_t extra = rule == block_rule::vjepa ? 1 : 0;
        const std::size_t t0 = e.frames < g.frames ? std::size_t(stats::uniform_index(rng, g.frames - e.frames + 1)) : 0;
        const std::size_t y0 = std::size_t(stats::uniform_index(rng, g.height - e.height + extra));
        const std::size_t x0 = std::size_t(stats::uniform_index(rng, g.width - e.width + extra));
        return {rule == block_rule::vjepa ? t0 : 0, y0, x0, e};
    }

    // The sorted tokens of b that no box of others[0, n) contains
    template <typename Out>
    constexpr std::size_t write_box(const token_grid& g, const box& b, std::span<const box> others, Out out) noexcept
    {
        std::size_t count = 0;
        for (std::size_t t = b.t0; t < b.t0 + b.e.frames; ++t)
            for (std::size_t y = b.y0; y < b.y0 + b.e.height; ++y)
                for (std::size_t x = b.x0; x < b.x0 + b.e.width; ++x) {
                    bool hidden = false;
                    for (const box& o : others)
                        hidden = hidden || o.contains(t, y, x);
                    if (!hidden) {
                        if (!out.empty())
                            out[count] = std::uint32_t((t * g.height + y) * g.width + x);
                        ++count;
                    }
                }
        return count;
    }
} // namespace detail::masking

/// \brief A block's size from one uniform draw `u`, as the file's overview says.
/// \param g The grid.
/// \param s The ranges.
/// \param u A uniform draw in `[0, 1)`.
/// \param rule Which paper's bounds.
/// \return The size, every side at least 1.
[[nodiscard]] constexpr block_extent sample_block_extent(const token_grid& g, const block_sampler& s, double u, block_rule rule) noexcept
{
    const double scale = s.scale_min + u * (s.scale_max - s.scale_min);
    const double aspect = s.aspect_min + u * (s.aspect_max - s.aspect_min);
    const double temporal = s.temporal_min + u * (s.temporal_max - s.temporal_min);
    const double keep = double(std::size_t(double(g.height * g.width) * scale));
    std::size_t h = std::size_t(detail::masking::round_even(math::sqrt(keep * aspect)));
    std::size_t w = std::size_t(detail::masking::round_even(math::sqrt(keep / aspect)));
    if (rule == block_rule::ijepa) {
        while (h >= g.height && h > 1)
            --h;
        while (w >= g.width && w > 1)
            --w;
    } else {
        h = std::min(h, g.height);
        w = std::min(w, g.width);
    }
    const std::size_t f = std::size_t(double(g.frames) * temporal);
    return {std::clamp<std::size_t>(f, 1, g.frames), std::max<std::size_t>(h, 1), std::max<std::size_t>(w, 1)};
}

/// \brief I-JEPA's multi-block mask for one sample: `M` target blocks and a context block with the targets removed.
/// \param c The settings.
/// \param seed The run's seed.
/// \param step The step: the block sizes are the step's, shared by its batch.
/// \param sample The sample in the step's batch.
/// \param context Receives the context tokens, sorted; at least `c.grid.tokens()` long.
/// \param targets Receives target block `b`'s tokens, sorted, at `targets[b · tokens]`; at least `blocks · tokens`.
/// \param target_counts Receives each block's count; `c.targets.blocks` long.
/// \return The context's count and the targets' total.
constexpr mask_counts ijepa_masks(const ijepa_mask_config& c, std::uint64_t seed, std::uint64_t step, std::uint64_t sample,
                                  std::span<std::uint32_t> context, std::span<std::uint32_t> targets,
                                  std::span<std::size_t> target_counts) noexcept
{
    using detail::masking::box;
    const std::size_t n = c.grid.tokens(), m = c.targets.blocks;
    assert(c.grid.frames == 1 && m <= max_mask_blocks && c.context.blocks == 1 && context.size() >= n && targets.size() >= m * n
           && target_counts.size() == m);
    const std::uint64_t key = detail::masking::key(seed, step);
    stats::wyrand sizes = stats::wyrand::stream(key, 0), rng = stats::wyrand::stream(key, sample + 1);
    const block_extent pe = sample_block_extent(c.grid, c.targets, stats::uniform_real(sizes), block_rule::ijepa);
    const block_extent ee = sample_block_extent(c.grid, c.context, stats::uniform_real(sizes), block_rule::ijepa);
    assert(pe.tokens() > c.min_keep && ee.tokens() > c.min_keep);
    std::array<box, max_mask_blocks> placed{};
    mask_counts counts{};
    for (std::size_t b = 0; b < m; ++b) {
        placed[b] = detail::masking::place(c.grid, pe, block_rule::ijepa, rng);   // a whole block always exceeds min_keep
        target_counts[b] = detail::masking::write_box(c.grid, placed[b], {}, targets.subspan(b * n, n));
        counts.targets += target_counts[b];
    }
    for (std::size_t tries = 0, timeout = 0;;) {
        const box e = detail::masking::place(c.grid, ee, block_rule::ijepa, rng);
        const std::size_t constrain = c.allow_overlap ? 0 : (m > tries ? m - tries : 0);
        const auto others = std::span<const box>(placed).first(constrain);
        if (detail::masking::write_box(c.grid, e, others, std::span<std::uint32_t>{}) > c.min_keep) {
            counts.context = detail::masking::write_box(c.grid, e, others, context);
            return counts;
        }
        if (++timeout % 20 == 0)
            ++tries;
    }
}

/// \brief V-JEPA's multi-block mask for one sample: the union of `blocks` boxes the targets, the rest the context.
/// \param c The settings.
/// \param seed The run's seed.
/// \param step The step: the box size is the step's.
/// \param sample The sample.
/// \param context Receives the context tokens, sorted; at least `c.grid.tokens()` long.
/// \param targets Receives the target tokens, sorted; at least `c.grid.tokens()` long.
/// \return The counts, which add up to the grid's tokens.
constexpr mask_counts vjepa_masks(const vjepa_mask_config& c, std::uint64_t seed, std::uint64_t step, std::uint64_t sample,
                                  std::span<std::uint32_t> context, std::span<std::uint32_t> targets) noexcept
{
    using detail::masking::box;
    [[maybe_unused]] const std::size_t n = c.grid.tokens();
    const std::size_t m = c.targets.blocks;
    assert(m <= max_mask_blocks && context.size() >= n && targets.size() >= n);
    const std::uint64_t key = detail::masking::key(seed, step);
    stats::wyrand sizes = stats::wyrand::stream(key, 0), rng = stats::wyrand::stream(key, sample + 1);
    const block_extent e = sample_block_extent(c.grid, c.targets, stats::uniform_real(sizes), block_rule::vjepa);
    std::array<box, max_mask_blocks> placed{};
    for (;;) {
        for (std::size_t b = 0; b < m; ++b)
            placed[b] = detail::masking::place(c.grid, e, block_rule::vjepa, rng);
        const auto boxes = std::span<const box>(placed).first(m);
        mask_counts counts{};
        for (std::size_t t = 0; t < c.grid.frames; ++t)
            for (std::size_t y = 0; y < c.grid.height; ++y)
                for (std::size_t x = 0; x < c.grid.width; ++x) {
                    bool hidden = false;
                    for (const box& b : boxes)
                        hidden = hidden || b.contains(t, y, x);
                    const auto token = std::uint32_t((t * c.grid.height + y) * c.grid.width + x);
                    if (hidden)
                        targets[counts.targets++] = token;
                    else
                        context[counts.context++] = token;
                }
        if (counts.context > 0)
            return counts;
    }
}

/// \brief A random mask for one sample: `tokens - int(tokens · (1 - ratio))` tokens hidden, uniformly.
/// \param tokens The tokens of a sample.
/// \param ratio The fraction hidden, in `[0, 1]`.
/// \param seed The run's seed.
/// \param step The step.
/// \param sample The sample.
/// \param context Receives the visible tokens, sorted; at least `tokens` long (also the shuffle's working space).
/// \param targets Receives the hidden tokens, sorted; at least `tokens` long.
/// \return The counts.
constexpr mask_counts random_masks(std::size_t tokens, double ratio, std::uint64_t seed, std::uint64_t step, std::uint64_t sample,
                                   std::span<std::uint32_t> context, std::span<std::uint32_t> targets) noexcept
{
    assert(ratio >= 0 && ratio <= 1 && context.size() >= tokens && targets.size() >= tokens);
    stats::wyrand rng = stats::wyrand::stream(detail::masking::key(seed, step), sample + 1);
    const auto all = context.first(tokens);
    for (std::size_t i = 0; i < tokens; ++i)
        all[i] = std::uint32_t(i);
    stats::shuffle(all, rng);
    const auto keep = std::size_t(double(tokens) * (1.0 - ratio));
    std::copy(all.begin() + std::ptrdiff_t(keep), all.end(), targets.begin());
    std::sort(all.begin(), all.begin() + std::ptrdiff_t(keep));
    std::sort(targets.begin(), targets.begin() + std::ptrdiff_t(tokens - keep));
    return {keep, tokens - keep};
}

/// \brief The smallest of a batch's counts: the width every sample is cut to under the papers' rule.
/// \param counts One count per sample.
/// \return The smallest; 0 for no samples.
[[nodiscard]] constexpr std::size_t common_count(std::span<const std::size_t> counts) noexcept
{
    if (counts.empty())
        return 0;
    std::size_t m = counts[0];
    for (const std::size_t c : counts)
        m = c < m ? c : m;
    return m;
}

/// \brief A batch's index lists in one shape: each sample's first `width` indices, padded with its last if it has
/// fewer, and how many are real.
/// \param lists Sample `s`'s list at `lists[s · stride]`, `counts[s]` long, sorted.
/// \param stride The distance between two samples' lists.
/// \param counts Each sample's count, each at least 1.
/// \param width The indices kept per sample: `common_count(counts)` (the papers' rule, no padding), or more.
/// \param out Receives `counts.size() · width` indices.
/// \param lengths Receives `min(counts[s], width)` per sample; or empty.
constexpr void collate_indices(std::span<const std::uint32_t> lists, std::size_t stride, std::span<const std::size_t> counts,
                               std::size_t width, std::span<std::uint32_t> out, std::span<std::size_t> lengths) noexcept
{
    assert(out.size() == counts.size() * width && (lengths.empty() || lengths.size() == counts.size()));
    for (std::size_t s = 0; s < counts.size(); ++s) {
        assert(counts[s] > 0 && counts[s] <= stride && s * stride + counts[s] <= lists.size());
        const std::size_t real = std::min(counts[s], width);
        for (std::size_t j = 0; j < width; ++j)
            out[s * width + j] = lists[s * stride + std::min(j, real - 1)];
        if (!lengths.empty())
            lengths[s] = real;
    }
}

} // namespace snicholls::ml

#endif // snicholls_core_ml_masking_hpp
