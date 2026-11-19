//
//  unigram.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::ml` SentencePiece Unigram (T5, ALBERT, XLM-R, mT5, nomic-embed): the normaliser a Unigram vocabulary carries (its precompiled charsmap, read in place, and the whitespace rules), and the Viterbi segmentation over the vocabulary's log-probabilities.
/// \ingroup core
///
/// **Overview.** A Unigram model scores every piece with a log-probability and tokenizes text as the segmentation
/// whose scores sum highest. Two steps:
///
/// 1. `unigram_normalize(text, charsmap, user_defined, options, out)`: each prefix of the input is replaced by the
///    longest match in the model's **precompiled charsmap** (SentencePiece's normalisation rules, usually NFKC
///    with case and width folding, compiled into a double array), user-defined pieces pass through untouched,
///    invalid bytes become U+FFFD; then whitespace is escaped to "▁", a "▁" put in front (`add_dummy_prefix`), and
///    runs of spaces merged (`remove_extra_whitespaces`), all as SentencePiece's `Normalizer` does.
///    `precompiled_charsmap` is a view of the charsmap blob as GGUF (`tokenizer.ggml.precompiled_charsmap`) and
///    `tokenizer.json` store it: nothing is decoded or copied.
/// 2. `unigram_encode(normalized, vocab, score, kind, unknown, out, lattice)`: the Viterbi lattice over byte
///    positions, each extended by every vocabulary piece that starts there (a walk of the `token_trie`); a code
///    point no piece covers becomes the unknown token at the vocabulary's lowest score minus 10; consecutive
///    unknowns become one. Scores are summed in `double`, as SentencePiece and llama.cpp sum them, so ties fall
///    the same way.
///
/// Decode with `sentencepiece_decode` (`ml/sentencepiece.hpp`): "▁" is a space, the dummy prefix is dropped.
///
/// \snippet ml/unigram_demo.cpp unigram
///
/// **Relationships.** `ml/token_trie.hpp` (the vocabulary: build it over the normal, user-defined and unused
/// tokens, as SentencePiece does), `ml/sentencepiece.hpp` (decoding, `escaped_space`), `ml/unicode.hpp`.
/// Group: \ref core.
///
/// **Implementations.** One, llama.cpp's `llm_tokenizer_ugm_session` (itself SentencePiece's optimised Viterbi):
/// the best path to each byte offset, extended forward one code point at a time.
///
/// **Guarantees.** `constexpr` when the callables are. No allocation: the caller's output and lattice. Pure.
/// Complexity: normalising O(n · L) for charsmap matches of up to L bytes; encoding O(n · L · log V) for pieces of up
/// to L bytes in a vocabulary of V. The charsmap view checks every index it reads against the blob: a truncated or
/// corrupt blob stops a match, it never reads out of bounds.
///
/// **Errors.** `unigram_error::output_too_small` (normalising: fewer than the normalised bytes, at most
/// `unigram_normalized_size(text.size(), charsmap)`; encoding: fewer than `normalized.size()` slots),
/// `lattice_too_small` (fewer than `normalized.size() + 1` entries), `bad_charsmap` (a blob too short to hold
/// its own header).
///
/// **Performance.** `benchmarks/ml/unigram_benchmark.cpp`: the Viterbi pass against llama.cpp's, written out with
/// its `std::map` trie, on the same vocabulary and text. See BENCHMARKS.md.
///
/// **Tests and demos.** `tests/ml/unigram_test.cpp`: the charsmap view over double arrays built by the tests, the
/// normaliser's whitespace rules, Viterbi against the definition on random texts, unknown runs, the errors.
/// `tests/ml/tokenizer_test.cpp`: token for token llama.cpp on nomic-embed's XLM-R vocabulary
/// (`ggml-vocab-nomic-bert-moe.gguf`), on a committed subset with the charsmap entries its strings reach rebuilt as
/// a small double array (the fixture generator checks, on the whole vocabulary and the whole 237 KB charsmap, that
/// core and llama.cpp agree). `demos/ml/unigram_demo.cpp`.
///
/// **Dependencies.** `<cstddef>`, `<cstdint>`, `<span>`, `<string_view>`, `snicholls/core/result.hpp`,
/// `ml/sentencepiece.hpp`, `ml/token_trie.hpp`, `ml/unicode.hpp`.
///
/// **Known issues.** NFC applied before the charsmap (llama.cpp's `normalizer.nfc`, for vocabularies without a
/// charsmap) is not provided: a caller that needs it composes the text first. SentencePiece's
/// `treat_whitespace_as_suffix` is supported as llama.cpp supports it, by appending the space.
///
/// **Status.** Experimental, new in 0.17.0.

#ifndef snicholls_core_ml_unigram_hpp
#define snicholls_core_ml_unigram_hpp

#include <snicholls/core/ml/sentencepiece.hpp>
#include <snicholls/core/ml/token_trie.hpp>
#include <snicholls/core/ml/unicode.hpp>
#include <snicholls/core/result.hpp>

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace snicholls::ml
{

/// \brief Why a Unigram step failed.
/// \ingroup core
enum class unigram_error
{
    output_too_small,    ///< The output is shorter than the result.
    lattice_too_small,   ///< The lattice has fewer than `normalized.size() + 1` entries.
    bad_charsmap,        ///< The charsmap blob is too short for its own header.
};

/// \brief SentencePiece's precompiled charsmap, viewed in place: a XOR-compressed compact double array of prefixes
/// and the replacement strings they map to.
/// \ingroup core
///
/// The blob is a little-endian `uint32` byte count of the array, the array's packed `uint32` entries, then the
/// NUL-terminated replacements. Each entry packs a BASE (bits 10-30, shifted left 8 more when bit 9 is set), a
/// LEAF flag (bit 8) and a LCHECK label (bits 0-7, with bit 31 set on value entries); a child of node s for byte c
/// is at `BASE(s) ^ c` and holds `LCHECK == c` (Kanda 2018; SentencePiece's Darts-clone format).
struct precompiled_charsmap
{
    std::string_view blob;   ///< The charsmap bytes; empty for none.

    /// \brief Whether the blob holds its header and array.
    /// \return True if it does, or if it is empty (an empty charsmap maps nothing).
    constexpr bool valid() const noexcept { return blob.empty() || (blob.size() >= 4 && 4 + array_bytes() <= blob.size()); }

    /// \brief A match: the replacement and the input bytes it replaces (0 for none).
    struct match_result
    {
        std::string_view replacement;   ///< What the prefix becomes.
        std::size_t consumed;           ///< The input bytes matched; 0 when no prefix matched.
    };

    /// \brief The longest prefix of `text` from `at` that the charsmap maps, and what it becomes.
    /// \param text The input.
    /// \param at A byte offset into it.
    /// \return The replacement and the bytes it replaces, or `consumed == 0`.
    constexpr match_result match(std::string_view text, std::size_t at) const noexcept
    {
        if (blob.size() < 4 || !valid())
            return {{}, 0};
        const std::size_t units = array_bytes() / 4;
        std::uint32_t node = base(unit(0));
        std::size_t best = 0;
        std::uint32_t value = 0;
        for (std::size_t i = at; i < text.size(); ++i) {
            const auto c = static_cast<unsigned char>(text[i]);
            if (c == 0)
                break;
            node ^= c;
            if (node >= units || (unit(node) & 0x800000FFu) != c)
                break;
            const bool leaf = (unit(node) >> 8) & 1u;
            node ^= base(unit(node));
            if (leaf) {
                if (node >= units)
                    break;
                best = i - at + 1;
                value = unit(node) & 0x7FFFFFFFu;
            }
        }
        if (best == 0)
            return {{}, 0};
        const std::string_view strings = blob.substr(4 + array_bytes());
        if (value >= strings.size())
            return {{}, 0};
        const std::size_t end = strings.find('\0', value);
        if (end == std::string_view::npos)
            return {{}, 0};
        return {strings.substr(value, end - value), best};
    }

    /// \brief The longest replacement in the blob, for sizing the normaliser's output.
    /// \return Its length in bytes (0 for an empty or malformed blob).
    constexpr std::size_t max_replacement() const noexcept
    {
        if (blob.size() < 4 || !valid())
            return 0;
        const std::string_view strings = blob.substr(4 + array_bytes());
        std::size_t longest = 0, run = 0;
        for (const char c : strings) {
            run = c == '\0' ? 0 : run + 1;
            longest = run > longest ? run : longest;
        }
        return longest;
    }

private:
    constexpr std::uint32_t read32(std::size_t i) const noexcept
    {
        return static_cast<std::uint32_t>(static_cast<unsigned char>(blob[i])) |
               static_cast<std::uint32_t>(static_cast<unsigned char>(blob[i + 1])) << 8 |
               static_cast<std::uint32_t>(static_cast<unsigned char>(blob[i + 2])) << 16 |
               static_cast<std::uint32_t>(static_cast<unsigned char>(blob[i + 3])) << 24;
    }
    constexpr std::size_t array_bytes() const noexcept { return read32(0); }
    constexpr std::uint32_t unit(std::size_t i) const noexcept { return read32(4 + 4 * i); }
    static constexpr std::uint32_t base(std::uint32_t u) noexcept { return (u >> 10) << ((u & (1u << 9)) >> 6); }
};

/// \brief SentencePiece's whitespace rules for a Unigram normaliser (GGUF's `tokenizer.ggml.*` flags).
/// \ingroup core
struct unigram_normalizer_options
{
    bool add_dummy_prefix = true;             ///< Put a space in front of the text (`add_space_prefix`).
    bool remove_extra_whitespaces = true;     ///< Drop leading and trailing spaces and merge runs of them.
    bool escape_whitespaces = true;           ///< Write spaces as "▁".
    bool treat_whitespace_as_suffix = false;  ///< Put the dummy space after the text instead.
};

/// \brief An upper bound on the bytes `unigram_normalize` writes for `n` bytes of input.
/// \param n The input's length.
/// \param charsmap The charsmap (its longest replacement sets the bound).
/// \return `(max(longest replacement, 3) + 3) * n + 3`.
constexpr std::size_t unigram_normalized_size(std::size_t n, const precompiled_charsmap& charsmap) noexcept
{
    const std::size_t r = charsmap.max_replacement();
    return ((r > 3 ? r : 3) + 3) * n + 3;
}

/// \brief Normalises text as a SentencePiece Unigram model does: charsmap, user-defined pieces, whitespace.
/// \param text The raw text.
/// \param charsmap The model's precompiled charsmap (an empty view maps nothing).
/// \param user_defined A trie of the user-defined pieces (an empty trie for none): matched first, unchanged.
/// \param options The whitespace rules.
/// \param out Receives the normalised text: `unigram_normalized_size(text.size(), charsmap)` bytes suffice.
/// \return The bytes written, or `output_too_small` / `bad_charsmap`.
constexpr result<std::size_t, unigram_error> unigram_normalize(std::string_view text, const precompiled_charsmap& charsmap, const token_trie& user_defined,
                                                               unigram_normalizer_options options, std::span<char> out) noexcept
{
    if (!charsmap.valid())
        return err(unigram_error::bad_charsmap);
    const std::string_view space = options.escape_whitespaces ? escaped_space : std::string_view(" ");
    const bool prepend = !options.treat_whitespace_as_suffix && options.add_dummy_prefix;
    const bool append = options.treat_whitespace_as_suffix && options.add_dummy_prefix;
    const bool merge = options.remove_extra_whitespaces;
    std::size_t at = 0;
    bool fail = false;
    const auto put = [&](std::string_view s) {
        if (out.size() - at < s.size()) {
            fail = true;
            return;
        }
        for (const char c : s)
            out[at++] = c;
    };
    bool prepended = false, in_word = false;
    for (std::size_t i = 0; i < text.size() && !fail;) {
        std::string_view r;
        std::size_t consumed = 0;
        if (!user_defined.order.empty())
            if (user_defined.longest_prefix(text.substr(i), consumed) == no_token)
                consumed = 0;
        if (consumed > 0) {
            r = text.substr(i, consumed);
        } else if (const auto m = charsmap.match(text, i); m.consumed > 0) {
            r = m.replacement;
            consumed = m.consumed;
        } else if (const utf8_decoded d = decode_utf8(text, i); d.valid) {
            r = text.substr(i, d.length);
            consumed = d.length;
        } else {
            r = "\xEF\xBF\xBD";
            consumed = 1;
        }
        for (const char c : r) {
            if (c != ' ') {
                if (!in_word) {
                    in_word = true;
                    if ((prepend && !prepended) || merge) {
                        put(space);
                        prepended = true;
                    }
                }
                put(std::string_view(&c, 1));
            } else {
                in_word = false;
                if (!merge)
                    put(space);
            }
        }
        i += consumed;
    }
    if (append)
        put(space);
    if (fail)
        return err(unigram_error::output_too_small);
    return ok(at);
}

/// \brief The score Unigram gives the unknown token: the lowest score of a normal piece, less 10.
/// \param lowest_normal_score The lowest score among the vocabulary's normal pieces.
/// \return That less `10`.
constexpr double unigram_unknown_score(float lowest_normal_score) noexcept { return static_cast<double>(lowest_normal_score) - 10.0; }

/// \brief One lattice entry: the best segmentation of the text up to a byte offset.
/// \ingroup core
struct unigram_lattice_entry
{
    double score;          ///< The best sum of scores of a segmentation ending here.
    std::uint32_t from;    ///< Where its last token starts.
    std::uint32_t token;   ///< Its last token.
};

/// \brief The unknown token and its score.
/// \ingroup core
struct unigram_unknown
{
    std::uint32_t id;   ///< The unknown token's id.
    double score;       ///< Its score: `unigram_unknown_score(lowest normal score)`.
};

/// \brief Segments normalised text into the pieces whose scores sum highest (Viterbi), as SentencePiece does.
/// \tparam Score A callable `score(std::uint32_t id) -> float` (the piece's log-probability).
/// \tparam Kind A callable `kind(std::uint32_t id) -> token_kind`: user-defined pieces score 0.
/// \param normalized The text after `unigram_normalize`.
/// \param vocab A trie of the normal, user-defined and unused pieces.
/// \param score Each piece's score.
/// \param kind Each piece's kind.
/// \param unknown The unknown token and its score.
/// \param out Receives the tokens: `normalized.size()` slots always suffice.
/// \param lattice At least `normalized.size() + 1` entries.
/// \return The number of tokens written, or why not.
template <typename Score, typename Kind>
constexpr result<std::size_t, unigram_error> unigram_encode(std::string_view normalized, const token_trie& vocab, Score&& score, Kind&& kind,
                                                            unigram_unknown unknown, std::span<std::uint32_t> out,
                                                            std::span<unigram_lattice_entry> lattice)
{
    const std::size_t n = normalized.size();
    if (n == 0)
        return ok(std::size_t{0});
    if (lattice.size() < n + 1)
        return err(unigram_error::lattice_too_small);
    if (out.size() < n)
        return err(unigram_error::output_too_small);
    constexpr double lowest = -1.7976931348623157e308;
    for (std::size_t i = 0; i <= n; ++i)
        lattice[i] = {lowest, 0, unknown.id};
    lattice[0].score = 0;
    for (std::size_t i = 0; i < n;) {
        const std::size_t unit = decode_utf8(normalized, i).length;
        const double here = lattice[i].score;
        bool covered = false;
        auto c = vocab.root();
        for (std::size_t j = i; j < n && vocab.advance(c, normalized[j]); ++j) {
            const std::uint32_t id = vocab.at(c);
            if (id == no_token)
                continue;
            covered = covered || j + 1 - i == unit;
            const double s = here + (kind(id) == token_kind::user_defined ? 0.0 : static_cast<double>(score(id)));
            if (s > lattice[j + 1].score)
                lattice[j + 1] = {s, static_cast<std::uint32_t>(i), id};
        }
        if (!covered && here + unknown.score > lattice[i + unit].score)
            lattice[i + unit] = {here + unknown.score, static_cast<std::uint32_t>(i), unknown.id};
        i += unit;
    }
    // Back from the end, one unknown for a run of them, then reversed
    std::size_t count = 0;
    bool previous_unknown = false;
    for (std::size_t at = n;;) {
        const unigram_lattice_entry& e = lattice[at];
        const bool is_unknown = e.token == unknown.id;
        if (!(previous_unknown && is_unknown))
            out[count++] = e.token;
        previous_unknown = is_unknown;
        if (e.from == 0)
            break;
        at = e.from;
    }
    for (std::size_t a = 0, b = count - 1; a < b; ++a, --b) {
        const std::uint32_t t = out[a];
        out[a] = out[b];
        out[b] = t;
    }
    return ok(count);
}

} // namespace snicholls::ml

#endif // snicholls_core_ml_unigram_hpp
