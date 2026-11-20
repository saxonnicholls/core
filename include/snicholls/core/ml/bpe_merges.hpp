//
//  bpe_merges.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::ml::merge_encode`: byte-pair encoding driven by a priority over pairs (Hugging Face's and GGUF's merge lists, or SentencePiece's scores), with byte fallback; the merge list as a sorted index; and GPT-2's byte-to-unicode map, built at compile time.
/// \ingroup core
///
/// **Overview.** `ml/bpe.hpp` is tiktoken's merge, where a token's rank decides. Hugging Face's `tokenizer.json`
/// and GGUF instead list merges, pairs `left right` in priority order, and the pair decides: the same bytes may
/// be mergeable from one split and not from another. `merge_encode` is that algorithm, as Hugging Face's
/// `BPE::merge_word` and llama.cpp's `llm_tokenizer_bpe_session` run it:
///
/// 1. Optionally, if the whole piece is a token, that token (`ignore_merges`, which Llama 3's tokenizer sets).
/// 2. Start from the piece's units: its bytes, or its UTF-8 code points.
/// 3. Repeatedly merge the adjacent pair with the best priority, the leftmost on ties, until no adjacent pair
///    has one.
/// 4. Each part that is a token is that token; a part that is not becomes one token per byte (byte fallback: per
///    byte of the part, or with `byte_level` per byte its code points spell in GPT-2's alphabet).
///
/// The priority is the caller's: `priority(left, right) -> Key`, lower is better, `merge_never<Key>` for a pair
/// that does not merge. A merge list gives its rank (`merge_table`); `ml/sentencepiece.hpp` gives minus the score
/// of the merged piece, which is SentencePiece's BPE. The token lookup and the byte fallback are callables too.
///
/// - **Byte-level BPE** (GPT-2, Llama 3, Qwen, DeepSeek, ...): the vocabulary is written in GPT-2's byte-to-unicode
///   alphabet, each byte one printable code point. `byte_level_encode(piece, out)` maps a pre-tokenised piece
///   into that alphabet, `merge_encode` runs on its code points against the vocabulary as stored, and
///   `byte_level_decode` maps decoded pieces back to bytes. The two 256- and 324-entry tables are computed by the
///   compiler (`byte_to_unicode`, `unicode_to_byte`).
/// - **SentencePiece-style BPE** (Gemma 4, `byte_encode = false` in llama.cpp): spaces already escaped to U+2581,
///   code points as units, `<0xNN>` byte tokens for the fallback.
///
/// \snippet ml/bpe_merges_demo.cpp bpe_merges
///
/// **Relationships.** `ml/bpe.hpp` (tiktoken's rank-of-bytes merge, and `bpe_error`), `ml/pretokenize.hpp` (the
/// pieces), `ml/token_trie.hpp` (a token lookup), `ml/sentencepiece.hpp` (the score priority). Group: \ref core.
///
/// **Implementations.** One: candidate pairs in a binary heap ordered by (priority, start), the pair a part
/// starts being current while the part is alive and its pair still ends where it did; O(n log n) for a piece of
/// n units, so a long piece (Gemma 4's whole lines) costs no more per byte than a short one.
///
/// **Guarantees.** `constexpr` when the callables are. No allocation: the caller's output and scratch. Pure.
/// Complexity: O(n log n) heap steps and O(n) priority lookups (at most three per merge).
///
/// **Errors.** `bpe_error::output_too_small` (fewer than `piece.size()` slots), `scratch_too_small` (fewer than
/// `merge_scratch_size(piece.size())` entries), `unknown_bytes` (a part is not a token and one of its bytes has
/// no byte token; `out` may then be partly written). `byte_level_encode` / `byte_level_decode` return
/// `output_too_small`, and `byte_level_decode` `unknown_bytes` for a code point outside the alphabet.
///
/// **Performance.** `benchmarks/ml/bpe_merges_benchmark.cpp`, the Llama 3 fixture's pieces in GPT-2's alphabet
/// (Release, Apple clang 17, Xeon W-3245, 2026-11-19, one thread on a loaded machine (ratios held across runs)): 744,000 pieces/s, against 556,000 for llama.cpp's loop written out (a `std::priority_queue` of bigrams
/// holding `std::string`s, merges in a `std::map`, the vocabulary in a `std::unordered_map`): 1.3 times faster, with
/// no allocation.
///
/// **Tests and demos.** `tests/ml/bpe_merges_test.cpp`: against the merge rule written directly on random pieces
/// and merge lists; ties; the pair that decides; both fallbacks; the byte map's 256 entries; the errors.
/// `tests/ml/tokenizer_test.cpp`: token for token llama.cpp's test vectors (`models/ggml-vocab-*.gguf.inp/.out`)
/// for GPT-2, Llama 3, Qwen 2 and 3.5, DeepSeek LLM and Coder, Falcon, StarCoder, Refact, Command R, MPT and
/// Gemma 4, on committed subsets of those vocabularies (`tests/fixtures/ml/tokenizers/`, written by
/// `scripts/generate_tokenizer_fixtures.cpp`), and decoding back. `demos/ml/bpe_merges_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<cstddef>`, `<cstdint>`, `<limits>`, `<span>`, `<string_view>`,
/// `<type_traits>`, `<utility>`, `ml/bpe.hpp`, `ml/unicode.hpp`.
///
/// **Known issues.** Code-point units decode strictly (`decode_utf8`): an invalid byte is a unit of its own.
/// llama.cpp takes a unit's length from its lead byte alone, and its byte-level path replaces an invalid byte by
/// U+FFFD before encoding; Hugging Face cannot be given invalid UTF-8. A piece is at most 2^32 - 1 bytes.
///
/// **Status.** Experimental, new in 0.17.0.

#ifndef snicholls_core_ml_bpe_merges_hpp
#define snicholls_core_ml_bpe_merges_hpp

#include <snicholls/core/ml/bpe.hpp>
#include <snicholls/core/ml/unicode.hpp>
#include <snicholls/core/ml/token_trie.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string_view>
#include <type_traits>
#include <utility>

namespace snicholls::ml
{

namespace detail::bpe_merges
{
    // GPT-2's bytes_to_unicode: printable Latin-1 bytes map to themselves, the other 68 to U+0100 onwards in order
    consteval std::array<char32_t, 256> make_byte_to_unicode()
    {
        std::array<char32_t, 256> map{};
        char32_t next = 256;
        for (std::size_t b = 0; b < 256; ++b) {
            const bool printable = (b >= 0x21 && b <= 0x7E) || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
            map[b] = printable ? static_cast<char32_t>(b) : next++;
        }
        return map;
    }
} // namespace detail::bpe_merges

/// \brief GPT-2's byte-to-unicode map: the code point each byte is written as in a byte-level vocabulary.
inline constexpr std::array<char32_t, 256> byte_to_unicode = detail::bpe_merges::make_byte_to_unicode();

/// \brief The inverse of `byte_to_unicode` over U+0000 to U+0143: the byte, or -1 for a code point not in the
/// alphabet.
inline constexpr std::array<std::int16_t, 324> unicode_to_byte = [] {
    std::array<std::int16_t, 324> map{};
    for (auto& m : map)
        m = -1;
    for (std::size_t b = 0; b < 256; ++b)
        map[byte_to_unicode[b]] = static_cast<std::int16_t>(b);
    return map;
}();

/// \brief The priority of a pair that never merges: infinity, or the largest value of an integer key.
template <typename Key>
inline constexpr Key merge_never = std::numeric_limits<Key>::has_infinity ? std::numeric_limits<Key>::infinity()
                                                                          : std::numeric_limits<Key>::max();

/// \brief What a piece starts as: its bytes, or its UTF-8 code points (an invalid byte one unit).
/// \ingroup core
enum class merge_units
{
    bytes,         ///< One unit per byte (a vocabulary over raw bytes).
    code_points,   ///< One unit per code point (byte-level vocabularies in GPT-2's alphabet, SentencePiece).
};

/// \brief One scratch entry of `merge_encode`: a candidate pair, or a part's links.
/// \tparam Key The priority type.
/// \ingroup core
template <typename Key>
struct merge_entry
{
    Key key;          ///< The pair's priority.
    std::uint32_t a;  ///< A candidate's start, or a part's next start.
    std::uint32_t b;  ///< A candidate's end, or a part's previous start (and whether it is alive).
};

/// \brief The scratch entries `merge_encode` needs for a piece of `n` bytes: `4n + 1` (links n, heap 3n).
/// \param n The piece's length.
/// \return The number of `merge_entry`s.
constexpr std::size_t merge_scratch_size(std::size_t n) noexcept { return 4 * n + 1; }

/// \brief Options of `merge_encode`.
/// \ingroup core
struct merge_options
{
    merge_units units = merge_units::code_points;   ///< What the piece starts as.
    bool ignore_merges = false;                     ///< Look the whole piece up first (Llama 3's `ignore_merges`).
    bool byte_level = false;                        ///< The piece is in GPT-2's alphabet: fall back per byte it spells.
};

namespace detail::bpe_merges
{
    inline constexpr std::uint32_t dead = 0x80000000u;

    template <typename Key>
    constexpr bool before(const merge_entry<Key>& x, const merge_entry<Key>& y) noexcept
    {
        return x.key < y.key || (x.key == y.key && x.a < y.a);
    }

    template <typename Key>
    constexpr void push(merge_entry<Key>* heap, std::size_t& size, merge_entry<Key> e) noexcept
    {
        std::size_t i = size++;
        while (i > 0) {
            const std::size_t p = (i - 1) / 2;
            if (!before(e, heap[p]))
                break;
            heap[i] = heap[p];
            i = p;
        }
        heap[i] = e;
    }

    template <typename Key>
    constexpr merge_entry<Key> pop(merge_entry<Key>* heap, std::size_t& size) noexcept
    {
        const merge_entry<Key> top = heap[0];
        const merge_entry<Key> e = heap[--size];
        std::size_t i = 0;
        for (;;) {
            std::size_t c = 2 * i + 1;
            if (c >= size)
                break;
            if (c + 1 < size && before(heap[c + 1], heap[c]))
                ++c;
            if (!before(heap[c], e))
                break;
            heap[i] = heap[c];
            i = c;
        }
        if (size > 0)
            heap[i] = e;
        return top;
    }

    // Pushes the candidate of the part at i and its successor, if they merge
    template <typename Key, typename Priority>
    constexpr void candidate(std::string_view piece, Priority& priority, merge_entry<Key>* link, merge_entry<Key>* heap,
                             std::size_t& size, std::uint32_t i)
    {
        const std::uint32_t j = link[i].a;
        if (j >= piece.size()) {
            link[i].key = merge_never<Key>;
            return;
        }
        const std::uint32_t k = link[j].a;
        const Key key = priority(piece.substr(i, j - i), piece.substr(j, k - j));
        link[i].key = key;
        if (key != merge_never<Key>)
            push(heap, size, merge_entry<Key>{key, i, k});
    }
} // namespace detail::bpe_merges

/// \brief Encodes one piece by merging pairs in priority order, with byte fallback.
/// \tparam Priority A callable `priority(std::string_view left, std::string_view right) -> Key`, lower first,
/// `merge_never<Key>` for no merge.
/// \tparam Token A callable `token(std::string_view bytes) -> std::uint32_t`, `no_token` if not a token.
/// \tparam ByteToken A callable `byte_token(unsigned char) -> std::uint32_t`, `no_token` if none.
/// \param piece The piece (pre-tokenised; for byte-level vocabularies, already in GPT-2's alphabet).
/// \param priority The merge priority.
/// \param token The vocabulary lookup.
/// \param byte_token The fallback for a byte of a part that is not a token.
/// \param out Receives the tokens: `piece.size()` slots always suffice.
/// \param scratch At least `merge_scratch_size(piece.size())` entries.
/// \param options Units and `ignore_merges`.
/// \return The number of tokens written, or why not.
template <typename Priority, typename Token, typename ByteToken,
          typename Key = std::remove_cvref_t<std::invoke_result_t<Priority&, std::string_view, std::string_view>>>
constexpr result<std::size_t, bpe_error> merge_encode(std::string_view piece, Priority&& priority, Token&& token, ByteToken&& byte_token,
                                                      std::span<std::uint32_t> out, std::type_identity_t<std::span<merge_entry<Key>>> scratch,
                                                      merge_options options = {})
{
    namespace d = detail::bpe_merges;
    const std::size_t n = piece.size();
    if (n == 0)
        return ok(std::size_t{0});
    if (options.ignore_merges)
        if (const std::uint32_t whole = token(piece); whole != no_token) {
            if (out.empty())
                return err(bpe_error::output_too_small);
            out[0] = whole;
            return ok(std::size_t{1});
        }
    if (out.size() < n)
        return err(bpe_error::output_too_small);
    if (scratch.size() < merge_scratch_size(n))
        return err(bpe_error::scratch_too_small);

    // link[i], for a part starting at byte i: a = the next part's start, b = the previous part's start (n for
    // none) with d::dead set once merged away, key = the priority of the part and its successor
    merge_entry<Key>* link = scratch.data();
    merge_entry<Key>* heap = link + n;
    std::size_t size = 0;
    std::uint32_t prev = static_cast<std::uint32_t>(n);
    for (std::uint32_t i = 0; i < n;) {
        const std::uint32_t len = options.units == merge_units::bytes ? 1u : decode_utf8(piece, i).length;
        link[i] = {merge_never<Key>, i + len, prev};
        prev = i;
        i += len;
    }
    for (std::uint32_t i = 0; i < n; i = link[i].a)
        d::candidate(piece, priority, link, heap, size, i);

    while (size > 0) {
        const merge_entry<Key> e = d::pop(heap, size);
        const std::uint32_t i = e.a;
        if ((link[i].b & d::dead) || link[i].key != e.key || link[i].a >= n || link[link[i].a].a != e.b)
            continue;   // merged away, or its pair has changed since
        const std::uint32_t j = link[i].a, k = link[j].a;
        link[j].b |= d::dead;
        link[i].a = k;
        if (k < n)
            link[k].b = i;
        d::candidate(piece, priority, link, heap, size, i);
        if (const std::uint32_t p = link[i].b; p < n)
            d::candidate(piece, priority, link, heap, size, p);
    }

    std::size_t count = 0;
    for (std::uint32_t i = 0; i < n; i = link[i].a) {
        const std::string_view part = piece.substr(i, link[i].a - i);
        if (const std::uint32_t id = token(part); id != no_token) {
            out[count++] = id;
            continue;
        }
        for (std::size_t k = 0; k < part.size();) {
            // A byte of the part, or with byte_level the byte its next code point spells
            std::uint32_t b = static_cast<unsigned char>(part[k]);
            std::size_t len = 1;
            if (options.byte_level) {
                const utf8_decoded d = decode_utf8(part, k);
                if (!d.valid || d.code_point >= unicode_to_byte.size() || unicode_to_byte[d.code_point] < 0)
                    return err(bpe_error::unknown_bytes);
                b = static_cast<std::uint32_t>(unicode_to_byte[d.code_point]);
                len = d.length;
            }
            const std::uint32_t id = byte_token(static_cast<unsigned char>(b));
            if (id == no_token)
                return err(bpe_error::unknown_bytes);
            out[count++] = id;
            k += len;
        }
    }
    return ok(count);
}

/// \brief A merge list as a sorted index: `rank(left, right)` is the pair's position in the list.
/// \ingroup core
struct merge_table
{
    std::span<const std::pair<std::string_view, std::string_view>> merges;   ///< The merges in priority order.
    std::span<const std::uint32_t> order;   ///< Their indices sorted by (left, right), then rank.

    /// \brief The rank of a pair.
    /// \param left The pair's left side.
    /// \param right Its right side.
    /// \return Its index in the list (the first, if listed twice), or `merge_never<std::uint32_t>`.
    constexpr std::uint32_t rank(std::string_view left, std::string_view right) const noexcept
    {
        const auto less = [&](std::uint32_t m) {
            const auto& [l, r] = merges[m];
            return l < left || (l == left && r < right);
        };
        std::size_t lo = 0, hi = order.size();
        while (lo < hi) {
            const std::size_t mid = lo + (hi - lo) / 2;
            if (less(order[mid]))
                lo = mid + 1;
            else
                hi = mid;
        }
        if (lo < order.size() && merges[order[lo]].first == left && merges[order[lo]].second == right)
            return order[lo];
        return merge_never<std::uint32_t>;
    }

    /// \brief The rank of a pair, as a `merge_encode` priority.
    /// \param left The pair's left side.
    /// \param right Its right side.
    /// \return `rank(left, right)`.
    constexpr std::uint32_t operator()(std::string_view left, std::string_view right) const noexcept { return rank(left, right); }
};

/// \brief Fills `order` with the merges' indices sorted for `merge_table`.
/// \param merges The merges in priority order.
/// \param order At least `merges.size()` slots.
constexpr void build_merge_index(std::span<const std::pair<std::string_view, std::string_view>> merges, std::span<std::uint32_t> order)
{
    for (std::size_t i = 0; i < merges.size(); ++i)
        order[i] = static_cast<std::uint32_t>(i);
    const auto before = [&](std::uint32_t x, std::uint32_t y) { return merges[x] < merges[y] || (merges[x] == merges[y] && x < y); };
    detail::vocab_index::sort(order.first(merges.size()), before);
}

/// \brief Splits a merge as GGUF and `tokenizer.json` (`"left right"`) write it, at the first space after the
/// first byte.
/// \param merge The merge.
/// \return Its two sides; both empty if there is no space.
constexpr std::pair<std::string_view, std::string_view> split_merge(std::string_view merge) noexcept
{
    const std::size_t space = merge.size() > 1 ? merge.find(' ', 1) : std::string_view::npos;
    if (space == std::string_view::npos)
        return {};
    return {merge.substr(0, space), merge.substr(space + 1)};
}

/// \brief Writes bytes in GPT-2's byte-level alphabet (each byte one code point, one or two UTF-8 bytes).
/// \param bytes Any bytes.
/// \param out Receives the UTF-8: `2 * bytes.size()` bytes always suffice.
/// \return The bytes written, or `output_too_small`.
constexpr result<std::size_t, bpe_error> byte_level_encode(std::string_view bytes, std::span<char> out) noexcept
{
    std::size_t at = 0;
    for (const char c : bytes) {
        const char32_t cp = byte_to_unicode[static_cast<unsigned char>(c)];
        const std::size_t len = utf8_length(cp);
        if (out.size() - at < len)
            return err(bpe_error::output_too_small);
        at += encode_utf8(cp, out.data() + at);
    }
    return ok(at);
}

/// \brief Reads text in GPT-2's byte-level alphabet back to bytes (a token's piece to what it means).
/// \param text UTF-8 in the alphabet.
/// \param out Receives the bytes: `text.size()` always suffice.
/// \return The bytes written, `unknown_bytes` for a code point outside the alphabet, or `output_too_small`.
constexpr result<std::size_t, bpe_error> byte_level_decode(std::string_view text, std::span<char> out) noexcept
{
    std::size_t at = 0;
    for (std::size_t i = 0; i < text.size();) {
        const utf8_decoded d = decode_utf8(text, i);
        if (!d.valid || d.code_point >= unicode_to_byte.size() || unicode_to_byte[d.code_point] < 0)
            return err(bpe_error::unknown_bytes);
        if (at == out.size())
            return err(bpe_error::output_too_small);
        out[at++] = static_cast<char>(unicode_to_byte[d.code_point]);
        i += d.length;
    }
    return ok(at);
}

} // namespace snicholls::ml

#endif // snicholls_core_ml_bpe_merges_hpp
