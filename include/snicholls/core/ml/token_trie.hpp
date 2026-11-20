//
//  token_trie.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::ml::token_trie`: a tokenizer vocabulary as a trie over its pieces' bytes, held as the ids sorted by piece, so it is built by one sort (at compile time for a fixed vocabulary) and walked without allocation.
/// \ingroup core
///
/// **Overview.** Every tokenizer asks its vocabulary two questions: is this byte string a token, and which tokens
/// are prefixes of the text from here (Unigram's Viterbi lattice, WordPiece's longest match, RWKV's greedy match).
/// A sorted array answers both. With the ids ordered by their pieces' bytes, the pieces sharing a prefix are one
/// contiguous range, and extending the prefix by a byte narrows the range by two binary searches: that is a trie,
/// walked one byte at a time (`cursor`, `advance`, `at`), with no nodes to build or store.
///
/// - `build_token_index(pieces, order, keep)` fills `order` with the ids `keep` accepts, sorted by piece. It is
///   `constexpr`, so `static_token_trie<N>` holds a fixed vocabulary's index computed by the compiler.
/// - `token_trie{pieces, order}` is the view: `find(bytes)` (exact match, O(L log V)), `longest_prefix(text)`, and
///   the cursor walk.
/// - `token_kind` is what a token is, numbered as GGUF stores it.
///
/// The pieces are the caller's: a span of `std::string_view` into a tokenizer file's string table, or a `constexpr`
/// array of literals. Pieces are bytes, compared unsigned.
///
/// \snippet ml/token_trie_demo.cpp token_trie
///
/// **Relationships.** The vocabulary behind `ml/unigram.hpp`, `ml/wordpiece.hpp`, `ml/rwkv.hpp`, and the token
/// lookups of `ml/sentencepiece.hpp` and `ml/bpe_merges.hpp` (any callable will do there; this is the
/// allocation-free one). `containers/perfect_map.hpp` is the faster exact lookup when only `find` is needed and
/// the vocabulary is fixed. Group: \ref core.
///
/// **Implementations.** One. Against a node trie (llama.cpp's `naive_trie`, a `std::map` per node) it stores nothing
/// but the 4-byte ids and costs a binary search per byte instead of a map lookup.
///
/// **Guarantees.** `constexpr`; `noexcept` except `build_token_index`, which only sorts. No allocation; the view
/// does not own the pieces or the order, which must outlive it. Complexity: building O(V log V · L); `find`
/// O(L log V); `advance` O(log V) per byte, shrinking as the range narrows. When pieces repeat, the largest id is
/// the one found (llama.cpp's vocabulary map keeps the last).
///
/// **Errors.** `build_token_index` asserts `order.size() >= pieces.size()`; ids are `std::uint32_t`.
///
/// **Performance.** `benchmarks/ml/token_trie_benchmark.cpp`, about 13,000 pieces (Release, Apple clang 17, Xeon W-3245, 2026-11-19, one thread on a loaded machine (ratios held across runs)): the longest token at
/// every byte of a text, 13 MiB/s against 40 MiB/s for a node trie of `std::map`s (llama.cpp's `naive_trie`); `find`,
/// 11 million lookups/s against 31 million for `std::unordered_map`. The sorted array is about three times slower
/// than either and stores 4 bytes per token instead of a node per byte; a caller that needs the speed and not the
/// compile-time form puts a hash map behind the same lookup callable.
///
/// **Tests and demos.** `tests/ml/token_trie_test.cpp` (against a linear scan on random vocabularies, duplicates,
/// the empty piece, bytes above 0x7F, a compile-time trie), `demos/ml/token_trie_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<cassert>`, `<cstddef>`, `<cstdint>`, `<span>`, `<string_view>`.
///
/// **Known issues.** Each step of a walk is two binary searches over string views (two indirections a probe), which
/// is what makes it slower than a node trie (see Performance).
///
/// **Status.** Experimental, new in 0.17.0.

#ifndef snicholls_core_ml_token_trie_hpp
#define snicholls_core_ml_token_trie_hpp

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace snicholls::ml
{

/// \brief The id a lookup returns for bytes that are not a token.
inline constexpr std::uint32_t no_token = 0xffffffffu;

/// \brief What a token is, numbered as GGUF's `tokenizer.ggml.token_type` and SentencePiece's model numbers them.
/// \ingroup core
enum class token_kind : std::uint8_t
{
    undefined = 0,      ///< No type given.
    normal = 1,         ///< An ordinary piece of text.
    unknown = 2,        ///< The unknown token (`<unk>`, `[UNK]`).
    control = 3,        ///< A control token (`<s>`, `</s>`, `[CLS]`): never produced from text, rendered as nothing.
    user_defined = 4,   ///< A user-defined token: matched in the raw text first, rendered as written.
    unused = 5,         ///< Reserved: in the vocabulary, never produced by SentencePiece.
    byte = 6,           ///< A byte-fallback token, `<0xNN>`.
};

namespace detail::vocab_index
{
    // Heap sort written out, O(n log n) and constexpr: GCC's -Warray-bounds misreads both std::sort's insertion
    // pass and std::make_heap over a fixed array of a few elements
    template <typename Less>
    constexpr void sift_down(std::uint32_t* v, std::size_t i, std::size_t n, Less& less)
    {
        for (std::size_t c = 2 * i + 1; c < n; i = c, c = 2 * i + 1) {
            if (c + 1 < n && less(v[c], v[c + 1]))
                ++c;
            if (!less(v[i], v[c]))
                return;
            const std::uint32_t t = v[i];
            v[i] = v[c];
            v[c] = t;
        }
    }

    template <typename Less>
    constexpr void sort(std::span<std::uint32_t> span, Less& less)
    {
        const std::size_t n = span.size();
        if (n < 2)
            return;
        std::uint32_t* v = span.data();
        for (std::size_t i = n / 2; i-- > 0;)
            sift_down(v, i, n, less);
        for (std::size_t end = n - 1; end > 0; --end) {
            const std::uint32_t t = v[0];
            v[0] = v[end];
            v[end] = t;
            sift_down(v, 0, end, less);
        }
    }
} // namespace detail::vocab_index

/// \brief Fills `order` with the ids that `keep` accepts, sorted by piece (bytes compared unsigned), ties by id
/// descending.
/// \tparam Keep A callable `keep(std::uint32_t id) -> bool`.
/// \param pieces Every token's bytes, by id.
/// \param order At least `pieces.size()` slots.
/// \param keep Which ids belong in the trie.
/// \return The number of ids written: the trie is `order.first(count)`.
template <typename Keep>
constexpr std::size_t build_token_index(std::span<const std::string_view> pieces, std::span<std::uint32_t> order, Keep&& keep)
{
    assert(order.size() >= pieces.size());
    std::size_t n = 0;
    for (std::size_t id = 0; id < pieces.size(); ++id)
        if (keep(static_cast<std::uint32_t>(id)))
            order[n++] = static_cast<std::uint32_t>(id);
    const auto before = [&](std::uint32_t a, std::uint32_t b) { return pieces[a] < pieces[b] || (pieces[a] == pieces[b] && a > b); };
    detail::vocab_index::sort(order.first(n), before);
    return n;
}

/// \brief Fills `order` with every id sorted by piece.
/// \param pieces Every token's bytes, by id.
/// \param order At least `pieces.size()` slots.
/// \return `pieces.size()`.
constexpr std::size_t build_token_index(std::span<const std::string_view> pieces, std::span<std::uint32_t> order)
{
    return build_token_index(pieces, order, [](std::uint32_t) { return true; });
}

/// \brief A vocabulary as a trie: a view of the pieces and of the ids sorted by piece.
/// \ingroup core
struct token_trie
{
    std::span<const std::string_view> pieces;   ///< Every token's bytes, by id.
    std::span<const std::uint32_t> order;       ///< The ids in the trie, sorted by piece (`build_token_index`).

    /// \brief A position in the trie: the range of `order` whose pieces start with the bytes walked so far.
    struct cursor
    {
        std::uint32_t lo;      ///< First index into `order` of the range.
        std::uint32_t hi;      ///< One past its last.
        std::uint32_t depth;   ///< The bytes walked: every piece in the range starts with them.

        /// \brief Whether no piece starts with the bytes walked.
        /// \return True when the range is empty.
        constexpr bool empty() const noexcept { return lo >= hi; }
    };

    /// \brief The root: every piece, nothing walked.
    /// \return The cursor at the root.
    constexpr cursor root() const noexcept { return {0, static_cast<std::uint32_t>(order.size()), 0}; }

    /// \brief Walks one byte down: narrows `c` to the pieces whose next byte is `byte`.
    /// \param c The cursor, updated.
    /// \param byte The next byte.
    /// \return Whether any piece continues with it (otherwise `c` is empty).
    constexpr bool advance(cursor& c, char byte) const noexcept
    {
        const auto b = static_cast<unsigned char>(byte);
        // key(p) = p[depth], or -1 when p ends at depth; non-decreasing over the range
        const auto key = [&](std::uint32_t i) -> int {
            const std::string_view p = pieces[order[i]];
            return p.size() > c.depth ? static_cast<int>(static_cast<unsigned char>(p[c.depth])) : -1;
        };
        std::uint32_t lo = c.lo, hi = c.hi;
        while (lo < hi) {   // first with key >= b
            const std::uint32_t mid = lo + (hi - lo) / 2;
            if (key(mid) < static_cast<int>(b))
                lo = mid + 1;
            else
                hi = mid;
        }
        std::uint32_t end = lo, top = c.hi;
        while (end < top) {   // first with key > b
            const std::uint32_t mid = end + (top - end) / 2;
            if (key(mid) <= static_cast<int>(b))
                end = mid + 1;
            else
                top = mid;
        }
        c = {lo, end, c.depth + 1};
        return lo < end;
    }

    /// \brief The token whose piece is exactly the bytes walked, if any.
    /// \param c A cursor.
    /// \return Its id, or `no_token`.
    constexpr std::uint32_t at(const cursor& c) const noexcept
    {
        if (c.lo < c.hi && pieces[order[c.lo]].size() == c.depth)
            return order[c.lo];
        return no_token;
    }

    /// \brief The token whose piece is `bytes`.
    /// \param bytes Any bytes.
    /// \return Its id, or `no_token`.
    constexpr std::uint32_t find(std::string_view bytes) const noexcept
    {
        std::size_t lo = 0, hi = order.size();
        while (lo < hi) {
            const std::size_t mid = lo + (hi - lo) / 2;
            if (pieces[order[mid]] < bytes)
                lo = mid + 1;
            else
                hi = mid;
        }
        return lo < order.size() && pieces[order[lo]] == bytes ? order[lo] : no_token;
    }

    /// \brief The token for `bytes`, as a vocabulary lookup callable.
    /// \param bytes Any bytes.
    /// \return `find(bytes)`.
    constexpr std::uint32_t operator()(std::string_view bytes) const noexcept { return find(bytes); }

    /// \brief The longest token that is a prefix of `text`.
    /// \param text Any bytes.
    /// \param length Receives the token's length (0 if none).
    /// \return Its id, or `no_token`.
    constexpr std::uint32_t longest_prefix(std::string_view text, std::size_t& length) const noexcept
    {
        std::uint32_t best = no_token;
        length = 0;
        cursor c = root();
        for (std::size_t i = 0; i < text.size() && advance(c, text[i]); ++i)
            if (const std::uint32_t id = at(c); id != no_token) {
                best = id;
                length = i + 1;
            }
        return best;
    }
};

/// \brief A fixed vocabulary and its index, both computed by the compiler.
/// \tparam N The number of tokens.
/// \ingroup core
template <std::size_t N>
struct static_token_trie
{
    std::array<std::string_view, N> pieces{};   ///< Every token's bytes, by id.
    std::array<std::uint32_t, N> order{};       ///< The ids sorted by piece.

    /// \brief Indexes the pieces (every id).
    /// \param p The pieces, by id.
    constexpr explicit static_token_trie(const std::array<std::string_view, N>& p) : pieces(p)
    {
        build_token_index(std::span<const std::string_view>(pieces), std::span<std::uint32_t>(order));
    }

    /// \brief The trie view over this vocabulary.
    /// \return A `token_trie` of `pieces` and `order`.
    constexpr token_trie view() const noexcept { return {pieces, order}; }
};

} // namespace snicholls::ml

#endif // snicholls_core_ml_token_trie_hpp
