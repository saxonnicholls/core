//
//  bloom_filter.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  The algorithm is the same: k bits per key from two 64-bit hashes by double hashing (Kirsch and
//  Mitzenmacher), sized for a target false-positive rate. Core hashes with wyhash instead of FNV-1a, stores
//  plain words instead of atomics (the lock-free read side is a wrapper's business), and adds clear, count,
//  merge and the estimated rate.
//

/// \file
/// \brief `snicholls::containers::bloom_filter`: a fixed-size probabilistic set that never says "no" wrongly.
/// \ingroup core

#ifndef snicholls_core_containers_bloom_filter_hpp
#define snicholls_core_containers_bloom_filter_hpp

#include <snicholls/core/hash/wyhash.hpp>

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace snicholls::containers {

    /// \brief The size of a Bloom filter: its bits and the number of hashes per key.
    /// \ingroup core
    struct bloom_parameters {
        /// \brief The number of bits; a multiple of 64.
        std::size_t bits;
        /// \brief The number of bits set per key, 1 to 32.
        std::uint32_t hashes;
    };

    /// \brief The bits and hashes for `expected_items` keys at `false_positive_rate`: `m = -n ln p / (ln 2)^2`, `k = m/n ln 2`.
    /// \ingroup core
    /// \param expected_items    The number of keys that will be added; at least 1.
    /// \param false_positive_rate The rate wanted once that many are in, in (0, 1); outside that, 0.01.
    /// \return The parameters, bits rounded up to a multiple of 64, hashes clamped to 1 to 32.
    [[nodiscard]] inline bloom_parameters bloom_sizing(std::size_t expected_items, double false_positive_rate) noexcept
    {
        if (expected_items < 1)
            expected_items = 1;
        if (!(false_positive_rate > 0.0 && false_positive_rate < 1.0))
            false_positive_rate = 0.01;
        constexpr double ln2 = 0.6931471805599453;
        const double n = static_cast<double>(expected_items);
        const double m = -n * std::log(false_positive_rate) / (ln2 * ln2);
        std::size_t bits = static_cast<std::size_t>(m) + 1;
        bits = ((bits + 63) / 64) * 64;
        const double k = std::round((m / n) * ln2);
        const std::uint32_t hashes = k < 1.0 ? 1u : k > 32.0 ? 32u : static_cast<std::uint32_t>(k);
        return {bits, hashes};
    }

    /// \brief A fixed-size probabilistic set: "definitely not present" in O(k), or "maybe present".
    /// \ingroup core
    ///
    /// \overview
    /// A million watched addresses and a stream of events, nearly all of which are not for
    /// us: the hot path must reject a non-member in constant time with no map lookup. A Bloom
    /// filter sets `k` bits per key in a bit array of `m` bits; a query reads the same `k`
    /// bits and answers "no" if any is clear, which is never wrong, or "maybe" if all are set,
    /// which is wrong with a probability the sizing chose. A "maybe" falls through to the
    /// authoritative container; a "no" costs `k` hashes and `k` bit reads.
    ///
    /// \snippet containers/bloom_filter_demo.cpp watch
    ///
    /// Sizing: `bloom_filter(n, p)` picks `m` and `k` for `n` keys at false-positive rate `p`
    /// (`bloom_sizing`); 1,000,000 keys at 0.5% is 1.7 MB and 8 hashes. The `k` positions come
    /// from two 64-bit values by double hashing, `h1 + i * h2`, where `h1` is `wyhash` of the
    /// key and `h2` is `splitmix64(h1)` made odd, so one hash of the key serves every bit.
    ///
    /// \relations
    /// - `snicholls::wyhash`, the hash; `hierarchical_bitset`, the exact bitset beside it.
    /// - Module: \ref core.
    ///
    /// \impls
    /// One class; the words are plain `std::uint64_t`. A lock-free read side (bits only ever
    /// set, never cleared, so a reader racing a writer of another key sees no false negative)
    /// is a wrapper over `words()` with atomic loads, in a library that has threads.
    ///
    /// \contracts
    /// - No false negatives: a key that was added is always "maybe present".
    /// - `maybe_contains` is `const`, `noexcept`, O(k), allocation-free; `add` is `noexcept`
    ///   once constructed. Not thread-safe: synchronise `add` against readers externally.
    /// - Memory: `bits() / 8` bytes, allocated once by the constructor; `clear` keeps it.
    /// - `merge` requires the same `bits()` and `hashes()` (asserted); the result is the union.
    /// - `estimated_false_positive_rate()` is `(1 - e^(-k n / m))^k` for the `count()` added so far.
    ///
    /// \errors
    /// `std::bad_alloc` from the constructor only.
    ///
    /// \examples
    /// \snippet containers/bloom_filter_demo.cpp watch
    ///
    /// \perf
    /// Benchmarked in `benchmarks/containers/bloom_filter_benchmark.cpp` against the same
    /// filter written by hand over a `std::vector<std::uint64_t>` with the same hashes, on
    /// 4,096 queries half of which are members, after a fairness check; a
    /// `std::unordered_set<std::string>` lookup runs beside them for scale. Release, Apple
    /// clang 17, Xeon W-3245, 2026-11-06, 65,536 keys at 1%: 39.5 ns per query through the
    /// class, 40.0 ns by hand, 54 ns for the exact set. Zero overhead; the cost is the hash
    /// and seven dependent bit reads.
    ///
    /// \testing
    /// - Tests: tests/containers/bloom_filter_test.cpp.
    /// - Demo: demos/containers/bloom_filter_demo.cpp.
    ///
    /// \deps
    /// CMake target `snicholls::core`; standard library only, plus `hash/wyhash.hpp`.
    ///
    /// \gotchas
    /// A filter is sized for `n` keys; adding many more raises the false-positive rate past
    /// `p` (watch `estimated_false_positive_rate()`). Keys are bytes: the same text as a
    /// `std::string_view` and as a span of its bytes is the same key, but `"1"` and the
    /// integer 1 are not.
    ///
    /// \status
    /// New in 0.5.0.
    class bloom_filter {
    public:
        /// \brief An empty filter of 64 bits and one hash: valid, and almost useless. Size it with the other constructors.
        bloom_filter() : bloom_filter(bloom_parameters{64, 1}) {}

        /// \brief A filter sized for `expected_items` keys at `false_positive_rate` (`bloom_sizing`).
        /// \param expected_items      The keys that will be added.
        /// \param false_positive_rate The rate wanted; default 1%.
        /// \throws std::bad_alloc.
        explicit bloom_filter(std::size_t expected_items, double false_positive_rate = 0.01)
            : bloom_filter(bloom_sizing(expected_items, false_positive_rate))
        {
        }

        /// \brief A filter of exactly these parameters; `bits` is rounded up to a multiple of 64, `hashes` clamped to 1 to 32.
        /// \param parameters The size.
        /// \throws std::bad_alloc.
        explicit bloom_filter(bloom_parameters parameters)
            : bits_(((parameters.bits < 64 ? 64 : parameters.bits) + 63) / 64 * 64)
            , hashes_(parameters.hashes < 1 ? 1u : parameters.hashes > 32 ? 32u : parameters.hashes)
            , words_(bits_ / 64, 0)
        {
        }

        /// \name Adding keys
        ///@{
        /// \brief Add the bytes of `key`.
        /// \param key The key.
        void add(std::string_view key) noexcept { add_hash(wyhash(key)); }
        void add(std::span<const std::byte> key) noexcept { add_hash(wyhash(key)); }
        void add(std::span<const unsigned char> key) noexcept { add_hash(wyhash(key)); }
        ///@}

        /// \name Querying keys
        ///@{
        /// \brief Whether `key` may have been added: false is certain, true is probable.
        /// \param key The key.
        /// \return false when the key was never added; true when it was, or with probability about the sizing's rate when it was not.
        [[nodiscard]] bool maybe_contains(std::string_view key) const noexcept { return contains_hash(wyhash(key)); }
        [[nodiscard]] bool maybe_contains(std::span<const std::byte> key) const noexcept { return contains_hash(wyhash(key)); }
        [[nodiscard]] bool maybe_contains(std::span<const unsigned char> key) const noexcept { return contains_hash(wyhash(key)); }
        ///@}

        /// \brief Forget every key; the size is kept.
        void clear() noexcept
        {
            for (auto& w : words_)
                w = 0;
            count_ = 0;
        }

        /// \brief Take the union with another filter of the same size.
        /// \pre `other.bits() == bits() && other.hashes() == hashes()`.
        /// \param other The filter whose keys are added to this one.
        /// \return `*this`.
        bloom_filter& merge(const bloom_filter& other) noexcept
        {
            assert(other.bits_ == bits_ && other.hashes_ == hashes_ && "snicholls::containers::bloom_filter::merge: sizes differ");
            for (std::size_t i = 0; i < words_.size(); ++i)
                words_[i] |= other.words_[i];
            count_ += other.count_;
            return *this;
        }

        /// \brief The number of bits.
        /// \return A multiple of 64.
        [[nodiscard]] std::size_t bits() const noexcept { return bits_; }

        /// \brief The number of bits set per key.
        /// \return 1 to 32.
        [[nodiscard]] std::uint32_t hashes() const noexcept { return hashes_; }

        /// \brief How many keys have been added (duplicates counted again).
        /// \return The count since construction or `clear`.
        [[nodiscard]] std::size_t count() const noexcept { return count_; }

        /// \brief Whether no key has been added.
        /// \return `count() == 0`.
        [[nodiscard]] bool empty() const noexcept { return count_ == 0; }

        /// \brief The bytes the bit array occupies.
        /// \return `bits() / 8`.
        [[nodiscard]] std::size_t memory_bytes() const noexcept { return words_.size() * sizeof(std::uint64_t); }

        /// \brief The false-positive rate expected for the keys added so far: `(1 - e^(-k n / m))^k`.
        /// \return A probability; 0 when empty.
        [[nodiscard]] double estimated_false_positive_rate() const noexcept
        {
            if (count_ == 0)
                return 0.0;
            const double k = hashes_;
            const double fill = 1.0 - std::exp(-k * static_cast<double>(count_) / static_cast<double>(bits_));
            return std::pow(fill, k);
        }

        /// \brief The bit array, one `std::uint64_t` per 64 bits, bit `i` at `words()[i / 64] >> (i % 64)`.
        /// \return The words, for a wrapper that reads them atomically or persists them.
        [[nodiscard]] std::span<const std::uint64_t> words() const noexcept { return words_; }

    private:
        // The k bit positions of a key: h1 + i * h2 (double hashing), h2 odd.
        void add_hash(std::uint64_t h1) noexcept
        {
            const std::uint64_t h2 = splitmix64(h1) | 1u;
            for (std::uint32_t i = 0; i < hashes_; ++i) {
                const std::uint64_t bit = (h1 + i * h2) % bits_;
                words_[bit >> 6] |= std::uint64_t{1} << (bit & 63);
            }
            ++count_;
        }

        [[nodiscard]] bool contains_hash(std::uint64_t h1) const noexcept
        {
            const std::uint64_t h2 = splitmix64(h1) | 1u;
            for (std::uint32_t i = 0; i < hashes_; ++i) {
                const std::uint64_t bit = (h1 + i * h2) % bits_;
                if ((words_[bit >> 6] & (std::uint64_t{1} << (bit & 63))) == 0)
                    return false;
            }
            return true;
        }

        std::size_t bits_;
        std::uint32_t hashes_;
        std::vector<std::uint64_t> words_;
        std::size_t count_ = 0;
    };

} // namespace snicholls::containers

#endif /* snicholls_core_containers_bloom_filter_hpp */
