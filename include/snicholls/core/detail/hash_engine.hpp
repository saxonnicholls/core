//
//  hash_engine.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

// The machinery the hashes in hash/ share: word loads and stores from any one-byte code unit (char, unsigned char,
// char8_t, std::byte), assembled byte by byte so they are allowed in a constant expression and give the same value
// on every host (compilers fuse them into one load and a byte swap); the block feeder of an iterated hash, which
// compresses whole blocks straight from the input and buffers only the rest; and the Merkle-Damgard engine behind
// SHA-1, SHA-2 and RIPEMD-160 (a policy gives the block size, word type, byte order, length field, initial state and
// compression function). Not public API.

#ifndef snicholls_core_detail_hash_engine_hpp
#define snicholls_core_detail_hash_engine_hpp

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace snicholls::detail {

// ---- word loads and stores ------------------------------------------------------------------------------------

template <typename U>
constexpr std::uint8_t octet_of(U u) noexcept
{
    return static_cast<std::uint8_t>(u);
}

template <typename W, typename U>
constexpr W load_le_word(const U* p) noexcept
{
    W w = 0;
    for (std::size_t i = 0; i < sizeof(W); ++i)
        w |= static_cast<W>(static_cast<W>(octet_of(p[i])) << (8 * i));
    return w;
}

template <typename W, typename U>
constexpr W load_be_word(const U* p) noexcept
{
    W w = 0;
    for (std::size_t i = 0; i < sizeof(W); ++i)
        w = static_cast<W>((w << 8) | static_cast<W>(octet_of(p[i])));
    return w;
}

template <typename W, typename B>
constexpr void store_le_word(B* p, W w) noexcept
{
    for (std::size_t i = 0; i < sizeof(W); ++i)
        p[i] = static_cast<B>(static_cast<std::uint8_t>(w >> (8 * i)));
}

template <typename W, typename B>
constexpr void store_be_word(B* p, W w) noexcept
{
    for (std::size_t i = 0; i < sizeof(W); ++i)
        p[sizeof(W) - 1 - i] = static_cast<B>(static_cast<std::uint8_t>(w >> (8 * i)));
}

template <typename W>
constexpr W rotr(W x, unsigned n) noexcept
{
    return static_cast<W>((x >> n) | (x << (sizeof(W) * 8 - n)));
}

template <typename W>
constexpr W rotl(W x, unsigned n) noexcept
{
    return static_cast<W>((x << n) | (x >> (sizeof(W) * 8 - n)));
}

// ---- the block feeder ----------------------------------------------------------------------------------------

// Whole blocks go to `block` straight from the input; a partial block waits in `buf`. `block` is called with a
// pointer to Block code units, either the caller's or buf's.
template <std::size_t Block>
struct block_feeder
{
    std::array<std::uint8_t, Block> buf{};
    std::size_t fill = 0;

    template <typename U, typename F>
    constexpr void feed(const U* p, std::size_t n, F&& block) noexcept
    {
        if (fill != 0) {
            const std::size_t take = n < Block - fill ? n : Block - fill;
            for (std::size_t i = 0; i < take; ++i)
                buf[fill + i] = octet_of(p[i]);
            fill += take;
            p += take;
            n -= take;
            if (fill < Block)
                return;
            block(buf.data());
            fill = 0;
        }
        for (; n >= Block; p += Block, n -= Block)
            block(p);
        for (std::size_t i = 0; i < n; ++i)
            buf[i] = octet_of(p[i]);
        fill = n;
    }
};

// ---- Merkle-Damgard ----------------------------------------------------------------------------------------

// Policy: word (std::uint32_t or std::uint64_t), block_size, length_size (8 or 16), big_endian, state (an array
// of words), initial, digest_size, and compress(state&, const word (&)[16]).
template <typename Policy>
class md_engine
{
public:
    using word = typename Policy::word;
    static constexpr std::size_t block_size = Policy::block_size;
    static constexpr std::size_t digest_size = Policy::digest_size;

    template <typename U>
    constexpr void absorb(const U* p, std::size_t n) noexcept
    {
        total_ += n;
        in_.feed(p, n, [this](const auto* b) { this->compress_from(b); });
    }

    template <typename Out>
    constexpr void finish(Out* out) noexcept
    {
        const std::uint64_t bits_low = total_ << 3;
        const std::uint64_t bits_high = total_ >> 61;
        std::size_t f = in_.fill;
        in_.buf[f++] = 0x80;
        if (f > block_size - Policy::length_size) {
            while (f < block_size)
                in_.buf[f++] = 0;
            compress_from(in_.buf.data());
            f = 0;
        }
        while (f < block_size - 8)
            in_.buf[f++] = 0;
        if constexpr (Policy::big_endian) {
            if constexpr (Policy::length_size == 16)
                store_be_word(in_.buf.data() + block_size - 16, bits_high);
            store_be_word(in_.buf.data() + block_size - 8, bits_low);
        }
        else {
            store_le_word(in_.buf.data() + block_size - 8, bits_low);
        }
        compress_from(in_.buf.data());

        // The state in the policy's byte order, truncated to the digest
        std::array<std::uint8_t, sizeof(typename Policy::state)> bytes{};
        for (std::size_t i = 0; i < state_.size(); ++i) {
            if constexpr (Policy::big_endian)
                store_be_word(bytes.data() + i * sizeof(word), state_[i]);
            else
                store_le_word(bytes.data() + i * sizeof(word), state_[i]);
        }
        for (std::size_t i = 0; i < digest_size; ++i)
            out[i] = static_cast<Out>(bytes[i]);
    }

private:
    template <typename U>
    constexpr void compress_from(const U* b) noexcept
    {
        word w[16]{};
        for (std::size_t i = 0; i < 16; ++i) {
            if constexpr (Policy::big_endian)
                w[i] = load_be_word<word>(b + i * sizeof(word));
            else
                w[i] = load_le_word<word>(b + i * sizeof(word));
        }
        Policy::compress(state_, w);
    }

    typename Policy::state state_ = Policy::initial;
    block_feeder<block_size> in_{};
    std::uint64_t total_ = 0;
};

// ---- run-time barriers ----------------------------------------------------------------------------------------

// Writes zeros through a volatile pointer, which the optimiser may not remove, then a compiler fence so the stores
// are not moved past what follows.
inline void volatile_zero(void* p, std::size_t n) noexcept
{
    volatile unsigned char* q = static_cast<volatile unsigned char*>(p);
    for (std::size_t i = 0; i < n; ++i)
        q[i] = 0;
    std::atomic_signal_fence(std::memory_order_seq_cst);
}

// Reads a value back through a volatile, so the comparison that produced it cannot be turned into an early exit.
inline bool volatile_is_zero(unsigned v) noexcept
{
    volatile unsigned settled = v;
    return settled == 0;
}

} // namespace snicholls::detail

#endif /* snicholls_core_detail_hash_engine_hpp */
