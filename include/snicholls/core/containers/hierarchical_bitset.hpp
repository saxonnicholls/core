//
//  hierarchical_bitset.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief A fixed-size bitset with summary levels: next/previous set bit in O(log64 n).
/// \ingroup core

#ifndef snicholls_core_containers_hierarchical_bitset_hpp
#define snicholls_core_containers_hierarchical_bitset_hpp

#include <bit>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace snicholls::containers {

/// \brief A bitset of `n` bits with summary words, so the next or previous set bit is found in O(log64 n).
/// \ingroup core
///
/// \overview
/// Level 0 holds the bits; each higher level holds one bit per non-zero word of the level below, up to a single
/// top word. Finding the next set bit at or after `i` scans at most one word per level up and one per level down
/// (three levels cover 262,144 bits, four 16.7 million). A price ladder uses it to find the next non-empty price
/// level after the best one empties, in constant time for any realistic span, instead of walking the ladder.
///
/// Use it to track occupancy of a dense index range. Not for sparse huge ranges (memory is n / 8 bytes plus ~1/63
/// of that for the summaries).
///
/// \relations
/// - Used by ladder order books (one bit per price level).
/// - Module: \ref core.
///
/// \impls
/// None: a single implementation.
///
/// \contracts
/// - **Thread safety:** none internal.
/// - **Exception safety:** `resize` may throw `std::bad_alloc` (strong); every other operation is `noexcept`.
/// - **Complexity:** set/reset/test O(levels); find_next/find_prev O(levels); levels = ceil(log64 n).
/// - **Invariants:** a summary bit is set exactly when the word below is non-zero.
/// \pre Indices passed to set/reset/test are `< size()`.
///
/// \errors
/// None besides `std::bad_alloc` from `resize`.
///
/// \examples
/// \snippet hierarchical_bitset_demo.cpp bits
///
/// \perf
/// Measured inside a ladder order book.
/// In core, benchmarks/containers/hierarchical_bitset_benchmark.cpp compares it like for like with `std::vector<bool>`, after a fairness
/// check; not yet measured in Release.
///
/// \testing
/// - Tests: tests/containers/hierarchical_bitset_test.cpp (model-checked against `std::vector<bool>`).
/// - Demo: demos/containers/hierarchical_bitset_demo.cpp.
/// - Benchmark: benchmarks/containers/hierarchical_bitset_benchmark.cpp.
///
/// \deps
/// CMake target `snicholls::core`; standard library only.
///
/// \gotchas
/// None known.
///
/// \status
/// Experimental.
class hierarchical_bitset
{
public:
    /// \brief Returned by the find functions when there is no such bit.
    static constexpr std::size_t npos = static_cast<std::size_t>(-1);

    /// \brief An empty bitset of size 0.
    hierarchical_bitset() = default;
    /// \brief A bitset of `n` cleared bits.
    /// \param n the number of bits.
    /// \throws std::bad_alloc on allocation failure.
    explicit hierarchical_bitset(std::size_t n) { resize(n); }

    /// \brief Resize to `n` bits, all cleared.
    /// \param n the number of bits.
    /// \throws std::bad_alloc on allocation failure (bitset unchanged).
    void resize(std::size_t n)
    {
        std::vector<std::vector<std::uint64_t>> levels;
        std::size_t bits = n;
        do {
            const std::size_t words = (bits + 63) / 64;
            levels.emplace_back(words == 0 ? 1 : words, 0);
            bits = words;
        } while (bits > 1);
        levels_ = std::move(levels);
        size_ = n;
    }

    /// \brief Number of bits.
    /// \return the size.
    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    /// \brief Whether any bit is set.
    /// \return `true` when one is.
    [[nodiscard]] bool any() const noexcept { return levels_.empty() ? false : levels_.back()[0] != 0; }

    /// \brief Clear every bit.
    void clear() noexcept
    {
        for (auto& level : levels_)
            for (auto& w : level) w = 0;
    }

    /// \brief The value of bit `i`.
    /// \param i the index.
    /// \return the bit.
    [[nodiscard]] bool test(std::size_t i) const noexcept { return (levels_[0][i >> 6] >> (i & 63)) & 1u; }

    /// \brief Set bit `i`.
    /// \param i the index.
    void set(std::size_t i) noexcept
    {
        for (auto& level : levels_) {
            std::uint64_t& w = level[i >> 6];
            const bool was_zero = w == 0;
            w |= std::uint64_t{1} << (i & 63);
            if (!was_zero) return;
            i >>= 6;
        }
    }

    /// \brief Clear bit `i`.
    /// \param i the index.
    void reset(std::size_t i) noexcept
    {
        for (auto& level : levels_) {
            std::uint64_t& w = level[i >> 6];
            w &= ~(std::uint64_t{1} << (i & 63));
            if (w != 0) return;
            i >>= 6;
        }
    }

    /// \brief The first set bit at or after `i`.
    /// \param i the start index (may be `>= size()`).
    /// \return the index, or `npos`.
    [[nodiscard]] std::size_t find_next(std::size_t i) const noexcept
    {
        if (i >= size_) return npos;
        std::size_t level = 0;
        // Climb: look in the current word at or after the position; if empty, move to the next word via the parent.
        for (;;) {
            const auto& words = levels_[level];
            const std::size_t w = i >> 6;
            if (w >= words.size()) return npos;
            const std::uint64_t bits = words[w] & (~std::uint64_t{0} << (i & 63));
            if (bits) {
                i = (w << 6) | static_cast<std::size_t>(std::countr_zero(bits));
                break;
            }
            if (level + 1 == levels_.size()) return npos;
            i = w + 1;   // next word at this level == next bit at the level above
            ++level;
            if ((i >> 6) >= levels_[level].size()) return npos;
        }
        // Descend: take the lowest set bit of each word below.
        while (level > 0) {
            --level;
            i = (i << 6) | static_cast<std::size_t>(std::countr_zero(levels_[level][i]));
        }
        return i < size_ ? i : npos;
    }

    /// \brief The last set bit at or before `i`.
    /// \param i the start index (clamped to `size() - 1`).
    /// \return the index, or `npos`.
    [[nodiscard]] std::size_t find_prev(std::size_t i) const noexcept
    {
        if (size_ == 0) return npos;
        if (i >= size_) i = size_ - 1;
        std::size_t level = 0;
        for (;;) {
            const auto& words = levels_[level];
            const std::size_t w = i >> 6;
            const unsigned shift = 63 - static_cast<unsigned>(i & 63);
            const std::uint64_t bits = words[w] & (~std::uint64_t{0} >> shift);
            if (bits) {
                i = (w << 6) | (63 - static_cast<std::size_t>(std::countl_zero(bits)));
                break;
            }
            if (level + 1 == levels_.size() || w == 0) return npos;
            i = w - 1;
            ++level;
        }
        while (level > 0) {
            --level;
            i = (i << 6) | (63 - static_cast<std::size_t>(std::countl_zero(levels_[level][i])));
        }
        return i;
    }

private:
    std::vector<std::vector<std::uint64_t>> levels_ = {std::vector<std::uint64_t>(1, 0)};
    std::size_t size_ = 0;
};

} // namespace snicholls::containers

#endif /* snicholls_core_containers_hierarchical_bitset_hpp */
