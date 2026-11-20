//
//  blake3.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief BLAKE3: hash, keyed hash, key derivation and extendable output, on one thread, usable at compile time.
/// \ingroup core
///
/// **Overview.** BLAKE3 (O'Connor, Aumasson, Neves, Wilcox-O'Hearn, 2020) is a Merkle tree of 1024-byte chunks,
/// each chunk sixteen 64-byte blocks through a 7-round compression derived from BLAKE2s. One function gives four
/// modes, each a different domain, so their outputs never collide by construction:
///
/// - `blake3::hash(input)`: the 32-byte hash.
/// - `blake3::keyed_hash(key, input)`, or `blake3{key}`: a MAC or PRF under a 32-byte key.
/// - `blake3::derive_key(context, material)`, or `blake3::key_derivation(context)`: a key derived from key
///   material under a hard-coded, globally unique, application-specific context string.
/// - `finish<N>()` and `finish_into(out, seek)`: any number of output bytes from any position (an XOF), the first
///   32 of which are the hash.
///
/// \snippet hash/blake3_demo.cpp blake3
///
/// This is the tree mode run on one thread, with the portable compression: the same outputs as the official
/// implementation, without its SIMD and multithreading. Every function is `constexpr`.
///
/// **Relationships.** `digest<N>` and `hash_input` (`hash/digest.hpp`); `blake2s` (`hash/blake2.hpp`), whose
/// compression it shortens; `hkdf_*` (`hash/kdf.hpp`) for key derivation under a standard. Group: \ref core.
///
/// **Guarantees.** Every member is `constexpr`; the unkeyed constructor, `key_derivation`, `update`, `finish`,
/// `finish_into` and `hash` are `noexcept`. No allocation: the chaining-value stack (54 entries, enough for 2^64
/// bytes) is inside the object, about 1.9 KB. A hasher is a trivially copyable value; `finish` and `finish_into`
/// are `const`, so a hasher can be finished, updated further and finished again. Complexity: one compression per
/// 64-byte block, plus one per parent node (one per chunk, amortised).
///
/// **Errors.** The keyed constructor and `keyed_hash` throw `std::length_error` unless the key is exactly 32
/// bytes; in a constant expression that is a compile error.
///
/// **Examples.** Above; `demos/hash/blake3_demo.cpp`.
///
/// **Performance.** `benchmarks/hash/blake3_benchmark.cpp`: against a hand-written single-threaded reference
/// loop, after a fairness check; figures in BENCHMARKS.md. The official implementation's AVX2 / AVX-512 / NEON
/// paths hash several chunks at once and are several times faster on long inputs.
///
/// **Tests and demos.** `tests/hash/blake3_test.cpp` (the official `test_vectors.json` of all three modes, 131
/// extended bytes each, at every published length from 0 to 102,400; the `blake3` Python package on random
/// lengths; every split; seeking), `demos/hash/blake3_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<cstddef>`, `<cstdint>`, `<span>`, `<stdexcept>`, `hash/digest.hpp`.
///
/// **Known issues.** None known.
///
/// **Status.** New in 0.17.0. Experimental.

#ifndef snicholls_core_hash_blake3_hpp
#define snicholls_core_hash_blake3_hpp

#include <snicholls/core/detail/hash_engine.hpp>
#include <snicholls/core/hash/digest.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>

namespace snicholls {

namespace detail {

    inline constexpr std::uint32_t blake3_iv[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                                   0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};

    enum blake3_flag : std::uint32_t {
        blake3_chunk_start = 1,
        blake3_chunk_end = 2,
        blake3_parent = 4,
        blake3_root = 8,
        blake3_keyed_hash = 16,
        blake3_derive_key_context = 32,
        blake3_derive_key_material = 64,
    };

    using blake3_words = std::array<std::uint32_t, 16>;
    using blake3_cv = std::array<std::uint32_t, 8>;

    constexpr blake3_words blake3_compress(const blake3_cv& cv, const blake3_words& m, std::uint64_t counter,
                                           std::uint32_t block_len, std::uint32_t flags) noexcept
    {
        constexpr std::uint8_t schedule[7][16] = {
            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8},
            {3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1}, {10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6},
            {12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4}, {9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7},
            {11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13}};
        std::uint32_t v[16] = {cv[0], cv[1], cv[2], cv[3], cv[4], cv[5], cv[6], cv[7],
                               blake3_iv[0], blake3_iv[1], blake3_iv[2], blake3_iv[3],
                               static_cast<std::uint32_t>(counter), static_cast<std::uint32_t>(counter >> 32), block_len, flags};
        const auto g = [&v](std::size_t a, std::size_t b, std::size_t c, std::size_t d, std::uint32_t x, std::uint32_t y) {
            v[a] = v[a] + v[b] + x;
            v[d] = rotr<std::uint32_t>(v[d] ^ v[a], 16);
            v[c] = v[c] + v[d];
            v[b] = rotr<std::uint32_t>(v[b] ^ v[c], 12);
            v[a] = v[a] + v[b] + y;
            v[d] = rotr<std::uint32_t>(v[d] ^ v[a], 8);
            v[c] = v[c] + v[d];
            v[b] = rotr<std::uint32_t>(v[b] ^ v[c], 7);
        };
        for (const auto& s : schedule) {
            g(0, 4, 8, 12, m[s[0]], m[s[1]]);
            g(1, 5, 9, 13, m[s[2]], m[s[3]]);
            g(2, 6, 10, 14, m[s[4]], m[s[5]]);
            g(3, 7, 11, 15, m[s[6]], m[s[7]]);
            g(0, 5, 10, 15, m[s[8]], m[s[9]]);
            g(1, 6, 11, 12, m[s[10]], m[s[11]]);
            g(2, 7, 8, 13, m[s[12]], m[s[13]]);
            g(3, 4, 9, 14, m[s[14]], m[s[15]]);
        }
        blake3_words out{};
        for (std::size_t i = 0; i < 8; ++i) {
            out[i] = v[i] ^ v[i + 8];
            out[i + 8] = v[i + 8] ^ cv[i];
        }
        return out;
    }

    template <typename U>
    constexpr blake3_words blake3_load(const U* p) noexcept
    {
        blake3_words m{};
        for (std::size_t i = 0; i < 16; ++i)
            m[i] = load_le_word<std::uint32_t>(p + 4 * i);
        return m;
    }

    constexpr blake3_cv blake3_first8(const blake3_words& w) noexcept
    {
        return {w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]};
    }

    // What a node's compression needs before its flags are final: a chunk's last block, or a parent's two children.
    struct blake3_output
    {
        blake3_cv input_cv{};
        blake3_words block{};
        std::uint64_t counter = 0;
        std::uint32_t block_len = 0;
        std::uint32_t flags = 0;

        constexpr blake3_cv chaining_value() const noexcept
        {
            return blake3_first8(blake3_compress(input_cv, block, counter, block_len, flags));
        }

        template <typename Out>
        constexpr void root_bytes(Out* out, std::size_t n, std::uint64_t seek) const noexcept
        {
            std::uint64_t block_index = seek / 64;
            std::size_t skip = static_cast<std::size_t>(seek % 64);
            std::size_t written = 0;
            while (written < n) {
                const blake3_words w = blake3_compress(input_cv, block, block_index++, block_len, flags | blake3_root);
                for (std::size_t i = skip; i < 64 && written < n; ++i)
                    out[written++] = static_cast<Out>(static_cast<std::uint8_t>(w[i / 4] >> (8 * (i % 4))));
                skip = 0;
            }
        }
    };

    struct blake3_chunk
    {
        blake3_cv cv{};
        std::uint64_t counter = 0;
        std::array<std::uint8_t, 64> block{};
        std::size_t block_len = 0;
        std::size_t blocks_done = 0;
        std::uint32_t flags = 0;

        constexpr std::size_t size() const noexcept { return 64 * blocks_done + block_len; }
        constexpr std::uint32_t start_flag() const noexcept { return blocks_done == 0 ? blake3_chunk_start : 0u; }

        template <typename U>
        constexpr void compress_block(const U* p) noexcept
        {
            cv = blake3_first8(blake3_compress(cv, blake3_load(p), counter, 64, flags | start_flag()));
            ++blocks_done;
        }

        // Absorbs up to the rest of the chunk; the chunk's last block is held back for its CHUNK_END flag.
        template <typename U>
        constexpr void update(const U* p, std::size_t n) noexcept
        {
            while (n != 0) {
                if (block_len == 64) {
                    compress_block(block.data());
                    block_len = 0;
                }
                if (block_len == 0 && n > 64) {
                    compress_block(p);
                    p += 64;
                    n -= 64;
                    continue;
                }
                const std::size_t take = n < 64 - block_len ? n : 64 - block_len;
                for (std::size_t i = 0; i < take; ++i)
                    block[block_len + i] = octet_of(p[i]);
                block_len += take;
                p += take;
                n -= take;
            }
        }

        constexpr blake3_output output() const noexcept
        {
            std::array<std::uint8_t, 64> last{};
            for (std::size_t i = 0; i < block_len; ++i)
                last[i] = block[i];
            return {cv, blake3_load(last.data()), counter, static_cast<std::uint32_t>(block_len),
                    flags | start_flag() | blake3_chunk_end};
        }
    };

    constexpr blake3_output blake3_parent_output(const blake3_cv& left, const blake3_cv& right, const blake3_cv& key,
                                                 std::uint32_t flags) noexcept
    {
        blake3_words block{};
        for (std::size_t i = 0; i < 8; ++i) {
            block[i] = left[i];
            block[i + 8] = right[i];
        }
        return {key, block, 0, 64, flags | blake3_parent};
    }

} // namespace detail

/// \brief BLAKE3: the 32-byte hash, the keyed hash, key derivation and extendable output.
/// \ingroup core
class blake3 : public detail::hash_updates<blake3>, public detail::one_shot_hash<blake3>
{
public:
    /// \brief The block size in bytes (what `hmac` would pad a key to; BLAKE3's own keyed mode needs no HMAC).
    static constexpr std::size_t block_size = 64;
    /// \brief The default digest size in bytes.
    static constexpr std::size_t digest_size = 32;
    /// \brief The key size in bytes, for the keyed mode.
    static constexpr std::size_t key_size = 32;
    /// \brief The digest type.
    using digest_type = digest<32>;

    /// \brief A hasher in the plain hash mode.
    constexpr blake3() noexcept
    {
        for (std::size_t i = 0; i < 8; ++i)
            key_[i] = detail::blake3_iv[i];
        reset_chunk(0);
    }

    /// \brief A hasher in the keyed mode: a MAC or PRF under `key`.
    /// \param key Exactly 32 bytes.
    /// \throws std::length_error unless the key is 32 bytes.
    template <hash_input K>
    constexpr explicit blake3(const K& key) : flags_(detail::blake3_keyed_hash)
    {
        if (detail::input_size(key) != key_size)
            throw std::length_error("snicholls::blake3: the key is exactly 32 bytes");
        const auto* k = detail::input_data(key);
        for (std::size_t i = 0; i < 8; ++i)
            key_[i] = detail::load_le_word<std::uint32_t>(k + 4 * i);
        reset_chunk(0);
    }

    /// \brief A hasher in the key derivation mode: `update` with the key material, then `finish` for the key.
    /// \param context The context string: hard-coded, globally unique, application-specific (for example
    ///                `"example.com 2026-11-20 session tokens v1"`), never variable or secret.
    /// \return The hasher.
    template <hash_input C>
    [[nodiscard]] static constexpr blake3 key_derivation(const C& context) noexcept
    {
        blake3 ctx(context_tag{});
        ctx.update(context);
        const digest<32> context_key = ctx.finish();
        blake3 h(context_tag{});
        h.flags_ = detail::blake3_derive_key_material;
        for (std::size_t i = 0; i < 8; ++i)
            h.key_[i] = detail::load_le_word<std::uint32_t>(context_key.bytes.data() + 4 * i);
        h.reset_chunk(0);
        return h;
    }

    /// \brief One-shot keyed hash.
    /// \param key Exactly 32 bytes.
    /// \param input The input.
    /// \return The 32-byte keyed hash.
    /// \throws std::length_error unless the key is 32 bytes.
    template <hash_input K, hash_input S>
    [[nodiscard]] static constexpr digest_type keyed_hash(const K& key, const S& input)
    {
        blake3 h{key};
        h.update(input);
        return h.finish();
    }

    /// \brief One-shot key derivation.
    /// \param context The context string (see `key_derivation`).
    /// \param material The key material.
    /// \return A 32-byte key; `key_derivation(context).update(material).finish<N>()` for another length.
    template <hash_input C, hash_input S>
    [[nodiscard]] static constexpr digest_type derive_key(const C& context, const S& material) noexcept
    {
        blake3 h = key_derivation(context);
        h.update(material);
        return h.finish();
    }

    /// \brief The first `N` bytes of the extendable output.
    /// \tparam N The length.
    /// \return The bytes.
    template <std::size_t N>
    [[nodiscard]] constexpr digest<N> finish() const noexcept
    {
        digest<N> out{};
        finish_into(out.bytes, 0);
        return out;
    }

    /// \brief The 32-byte output. `const`: the hasher can be updated further and finished again.
    /// \return The digest.
    [[nodiscard]] constexpr digest_type finish() const noexcept 
    {
        digest_type out{};
        finish_into(out.bytes, 0);
        return out;
    }

    /// \brief `out.size()` bytes of the extendable output, starting `seek` bytes in.
    /// \param out Where to write.
    /// \param seek The output position of `out[0]`.
    constexpr void finish_into(std::span<std::byte> out, std::uint64_t seek = 0) const noexcept
    {
        detail::blake3_output o = chunk_.output();
        for (std::size_t i = stack_size_; i-- > 0;)
            o = detail::blake3_parent_output(stack_[i], o.chaining_value(), key_, flags_);
        o.root_bytes(out.data(), out.size(), seek);
    }

private:
    friend detail::hash_updates<blake3>;
    struct context_tag
    {
    };

    constexpr explicit blake3(context_tag) noexcept : flags_(detail::blake3_derive_key_context)
    {
        for (std::size_t i = 0; i < 8; ++i)
            key_[i] = detail::blake3_iv[i];
        reset_chunk(0);
    }

    constexpr void reset_chunk(std::uint64_t counter) noexcept
    {
        chunk_ = detail::blake3_chunk{};
        chunk_.cv = key_;
        chunk_.counter = counter;
        chunk_.flags = flags_;
    }

    // Merges completed subtrees: as many parents as there are trailing zero bits in the chunk count.
    constexpr void push_chunk(detail::blake3_cv cv, std::uint64_t total_chunks) noexcept
    {
        while ((total_chunks & 1) == 0) {
            cv = detail::blake3_parent_output(stack_[--stack_size_], cv, key_, flags_).chaining_value();
            total_chunks >>= 1;
        }
        stack_[stack_size_++] = cv;
    }

    template <typename U>
    constexpr void absorb(const U* p, std::size_t n) noexcept
    {
        while (n != 0) {
            if (chunk_.size() == 1024) {
                const std::uint64_t total = chunk_.counter + 1;
                push_chunk(chunk_.output().chaining_value(), total);
                reset_chunk(total);
            }
            const std::size_t room = 1024 - chunk_.size();
            const std::size_t take = n < room ? n : room;
            chunk_.update(p, take);
            p += take;
            n -= take;
        }
    }

    detail::blake3_cv key_{};
    std::uint32_t flags_ = 0;
    detail::blake3_chunk chunk_{};
    std::array<detail::blake3_cv, 54> stack_{};
    std::size_t stack_size_ = 0;
};

} // namespace snicholls

#endif /* snicholls_core_hash_blake3_hpp */
