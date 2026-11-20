//
//  wordpiece.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::ml::wordpiece_encode`: BERT's tokenizer, its basic tokenisation (control characters removed, accents stripped, lowercased, punctuation and CJK ideographs split off) and greedy longest-match-first WordPiece with "##" continuations and [UNK]; and `wordpiece_decode`.
/// \ingroup core
///
/// **Overview.** BERT and its descendants (BGE, MiniLM, E5) tokenize in two steps, done here in one pass over
/// the text with no allocation:
///
/// 1. **Basic tokenisation** (Hugging Face's `BertNormalizer` and `BertPreTokenizer`): drop NUL, U+FFFD and
///    control characters; with `strip_accents`, decompose canonically and drop the nonspacing marks; with
///    `lowercase`, lowercase; split at whitespace, and make each punctuation character (ASCII punctuation or
///    `\p{P}`) and each CJK ideograph a word of its own.
/// 2. **WordPiece**: each word is cut greedily into the longest vocabulary pieces from the left, every piece after
///    the first spelled with the continuation prefix ("##ing"). A word that cannot be cut, or that is longer than
///    `max_word_chars`, is the unknown token.
///
/// The vocabulary convention is an option: Hugging Face's `vocab.txt` writes word-initial pieces bare and
/// continuations with "##" (the defaults); GGUF writes word-initial pieces with "▁" and continuations bare
/// (`word_prefix = "▁"`, `continuation_prefix = ""`, and llama.cpp sets no word limit).
///
/// \snippet ml/wordpiece_demo.cpp wordpiece
///
/// **Relationships.** `ml/unicode.hpp` (categories, `decompose`, `to_lower`), `ml/token_trie.hpp` (the vocabulary;
/// the longest match is one walk of it), `ml/sentencepiece.hpp`'s decoding for comparison. Group: \ref core.
///
/// **Implementations.** One. The normalisation follows Hugging Face's rules, which llama.cpp approximates
/// (llama.cpp strips every mark, `\p{M}`, and keeps unassigned code points; Hugging Face strips only nonspacing
/// marks and drops `\p{C}` including unassigned).
///
/// **Guarantees.** `constexpr` when the callables are. No allocation: the caller's output and word scratch.
/// Complexity: linear in the text times the trie walk (O(L log V) per piece of L bytes).
///
/// **Errors.** `wordpiece_error::output_too_small` (the tokens do not fit), `scratch_too_small` (a normalised word
/// does not fit; `wordpiece_scratch_size(text.size())` bytes always suffice), `unknown_id` (decoding an id that
/// `piece` does not know).
///
/// **Performance.** Not benchmarked separately; the work is `token_trie`'s walk, benchmarked in
/// `benchmarks/ml/token_trie_benchmark.cpp`.
///
/// **Tests and demos.** `tests/ml/wordpiece_test.cpp`: Hugging Face's convention on a small vocabulary; accents,
/// CJK, control characters, U+0130, long words; GGUF's convention; decoding; the errors. `tests/ml/tokenizer_test.cpp`:
/// token for token llama.cpp's test vectors for BGE (bert-bge) on a committed subset of its vocabulary.
/// `demos/ml/wordpiece_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<cstddef>`, `<cstdint>`, `<span>`, `<string_view>`, `snicholls/core/result.hpp`,
/// `ml/token_trie.hpp`, `ml/unicode.hpp`.
///
/// **Known issues.** Strip-accents keeps canonical order without reordering the marks it keeps; only spacing
/// marks with a nonzero combining class (rare, e.g. U+1D165) could differ from NFD. Hugging Face's decoder also
/// removes spaces before punctuation (`cleanup`); `wordpiece_decode` does not.
///
/// **Status.** Experimental, new in 0.17.0.

#ifndef snicholls_core_ml_wordpiece_hpp
#define snicholls_core_ml_wordpiece_hpp

#include <snicholls/core/ml/token_trie.hpp>
#include <snicholls/core/ml/unicode.hpp>
#include <snicholls/core/result.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace snicholls::ml
{

/// \brief Why `wordpiece_encode` or `wordpiece_decode` failed.
/// \ingroup core
enum class wordpiece_error
{
    output_too_small,    ///< The output is shorter than the result.
    scratch_too_small,   ///< A normalised word does not fit in the scratch.
    unknown_id,          ///< Decoding: an id `piece` does not know.
};

/// \brief BERT's tokenizer settings.
/// \ingroup core
struct wordpiece_options
{
    bool lowercase = true;                         ///< Lowercase (Hugging Face's `lowercase`).
    bool strip_accents = true;                     ///< Decompose and drop nonspacing marks.
    bool clean_text = true;                        ///< Drop NUL, U+FFFD and `\p{C}` except tab, CR and LF.
    bool split_cjk = true;                         ///< Make each CJK ideograph a word (`handle_chinese_chars`).
    std::string_view word_prefix = "";             ///< How a word-initial piece is spelled in the vocabulary.
    std::string_view continuation_prefix = "##";   ///< How a continuation piece is spelled.
    std::size_t max_word_chars = 100;              ///< A longer word is the unknown token (0 for no limit).
};

/// \brief The word scratch `wordpiece_encode` needs for `n` bytes of text: `3n + 4` (Hangul decomposes to three
/// times its bytes, nothing more).
/// \param n The text's length.
/// \return The bytes.
constexpr std::size_t wordpiece_scratch_size(std::size_t n) noexcept { return 3 * n + 4; }

/// \brief Whether a code point is a CJK ideograph by BERT's ranges (Hugging Face's, with its U+2B920 bound).
/// \param cp The code point.
/// \return Whether BERT splits it off.
constexpr bool is_bert_cjk(char32_t cp) noexcept
{
    return (cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0x3400 && cp <= 0x4DBF) || (cp >= 0x20000 && cp <= 0x2A6DF) ||
           (cp >= 0x2A700 && cp <= 0x2B73F) || (cp >= 0x2B740 && cp <= 0x2B81F) || (cp >= 0x2B920 && cp <= 0x2CEAF) ||
           (cp >= 0xF900 && cp <= 0xFAFF) || (cp >= 0x2F800 && cp <= 0x2FA1F);
}

/// \brief Whether BERT splits a code point off as punctuation: ASCII punctuation, or `\p{P}`.
/// \param cp The code point.
/// \return Whether it is BERT punctuation.
constexpr bool is_bert_punctuation(char32_t cp) noexcept
{
    if ((cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) || (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126))
        return true;
    return is_punctuation(category(cp));
}

namespace detail::wordpiece
{
    // Greedy longest-match-first over one normalised word; appends the pieces, or the unknown token
    constexpr bool cut(std::string_view word, const token_trie& vocab, std::uint32_t unknown, const wordpiece_options& o,
                       std::span<std::uint32_t> out, std::size_t& count) noexcept
    {
        if (o.max_word_chars != 0) {
            std::size_t chars = 0;
            for (std::size_t i = 0; i < word.size(); i += decode_utf8(word, i).length)
                ++chars;
            if (chars > o.max_word_chars) {
                if (count == out.size())
                    return false;
                out[count++] = unknown;
                return true;
            }
        }
        const std::size_t first = count;
        for (std::size_t start = 0; start < word.size();) {
            auto c = vocab.root();
            bool alive = true;
            for (const char b : start == 0 ? o.word_prefix : o.continuation_prefix)
                if (!(alive = vocab.advance(c, b)))
                    break;
            std::uint32_t best = no_token;
            std::size_t end = start;
            for (std::size_t i = start; alive && i < word.size();) {
                const std::size_t len = decode_utf8(word, i).length;
                for (std::size_t k = 0; k < len && alive; ++k)
                    alive = vocab.advance(c, word[i + k]);
                i += len;
                if (alive)
                    if (const std::uint32_t id = vocab.at(c); id != no_token) {
                        best = id;
                        end = i;
                    }
            }
            if (best == no_token) {   // the word cannot be cut: it is the unknown token
                count = first;
                break;
            }
            if (count == out.size())
                return false;
            out[count++] = best;
            start = end;
        }
        if (count == first) {
            if (count == out.size())
                return false;
            out[count++] = unknown;
        }
        return true;
    }
} // namespace detail::wordpiece

/// \brief Tokenizes text as BERT does: basic tokenisation, then WordPiece.
/// \param text The raw text.
/// \param vocab The vocabulary as a trie.
/// \param unknown The unknown token ([UNK]).
/// \param options Normalisation and the vocabulary's spelling.
/// \param out Receives the tokens.
/// \param scratch Holds one normalised word: `wordpiece_scratch_size(text.size())` bytes always suffice.
/// \return The number of tokens written, or why not.
constexpr result<std::size_t, wordpiece_error> wordpiece_encode(std::string_view text, const token_trie& vocab, std::uint32_t unknown,
                                                                const wordpiece_options& options, std::span<std::uint32_t> out,
                                                                std::span<char> scratch) noexcept
{
    std::size_t count = 0, len = 0;
    const auto flush = [&]() {
        const bool fits = len == 0 || detail::wordpiece::cut(std::string_view(scratch.data(), len), vocab, unknown, options, out, count);
        len = 0;
        return fits;
    };
    const auto append = [&](char32_t cp) {
        if (scratch.size() - len < utf8_length(cp))
            return false;
        len += encode_utf8(cp, scratch.data() + len);
        return true;
    };
    for (std::size_t i = 0; i < text.size();) {
        const utf8_decoded d = decode_utf8(text, i);
        i += d.length;
        const char32_t cp = d.code_point;
        const unicode_category cat = category(cp);
        if (is_whitespace(cp)) {
            if (!flush())
                return err(wordpiece_error::output_too_small);
            continue;
        }
        if (options.clean_text && (cp == 0 || cp == replacement_character || is_other(cat)))
            continue;
        const bool cjk = options.split_cjk && is_bert_cjk(cp);
        std::array<char32_t, max_decomposition> parts{};
        const std::size_t n = options.strip_accents ? decompose(cp, parts) : (parts[0] = cp, std::size_t{1});
        for (std::size_t k = 0; k < n; ++k) {
            const char32_t p = parts[k];
            if (options.strip_accents && category(p) == unicode_category::nonspacing_mark)
                continue;
            const bool alone = cjk || is_bert_punctuation(p);
            if (alone && !flush())
                return err(wordpiece_error::output_too_small);
            bool fits = true;
            if (options.lowercase && p == U'İ')
                fits = append(U'i') && append(U'̇');   // the one unconditional full lowercase mapping
            else
                fits = append(options.lowercase ? to_lower(p) : p);
            if (!fits)
                return err(wordpiece_error::scratch_too_small);
            if (alone && !flush())
                return err(wordpiece_error::output_too_small);
        }
    }
    if (!flush())
        return err(wordpiece_error::output_too_small);
    return ok(count);
}

/// \brief Renders WordPiece tokens as text: words separated by spaces, continuations joined.
/// \tparam Piece A callable `piece(std::uint32_t id) -> std::string_view`, empty for an unknown id.
/// \tparam Skip A callable `skip(std::uint32_t id) -> bool`: tokens not to render ([CLS], [SEP], [PAD]).
/// \param tokens The tokens.
/// \param piece Each token's piece.
/// \param skip Which tokens to leave out.
/// \param options The vocabulary's spelling (`word_prefix`, `continuation_prefix`).
/// \param out Receives the text.
/// \return The bytes written, or `unknown_id` / `output_too_small`.
template <typename Piece, typename Skip>
constexpr result<std::size_t, wordpiece_error> wordpiece_decode(std::span<const std::uint32_t> tokens, Piece&& piece, Skip&& skip,
                                                                const wordpiece_options& options, std::span<char> out)
{
    std::size_t at = 0;
    bool first = true;
    const auto put = [&](std::string_view s) {
        if (out.size() - at < s.size())
            return false;
        for (const char c : s)
            out[at++] = c;
        return true;
    };
    for (const std::uint32_t id : tokens) {
        const std::string_view p = piece(id);
        if (p.empty())
            return err(wordpiece_error::unknown_id);
        if (skip(id))
            continue;
        bool starts_word = true;
        std::string_view body = p;
        if (!options.continuation_prefix.empty() && p.starts_with(options.continuation_prefix)) {
            starts_word = false;
            body = p.substr(options.continuation_prefix.size());
        } else if (!options.word_prefix.empty()) {
            starts_word = p.starts_with(options.word_prefix);
            if (starts_word)
                body = p.substr(options.word_prefix.size());
        }
        if (starts_word && !first && !put(" "))
            return err(wordpiece_error::output_too_small);
        if (!put(body))
            return err(wordpiece_error::output_too_small);
        first = false;
    }
    return ok(at);
}

} // namespace snicholls::ml

#endif // snicholls_core_ml_wordpiece_hpp
