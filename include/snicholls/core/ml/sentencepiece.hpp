//
//  sentencepiece.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::ml` SentencePiece BPE (Llama 1 and 2, Mistral, Phi-3): spaces escaped to "▁" with the dummy prefix, merges by score, byte fallback to `<0xNN>`; and decoding of SentencePiece pieces back to text, shared with Unigram.
/// \ingroup core
///
/// **Overview.** SentencePiece's BPE model has no merge list: a pair of adjacent pieces may merge when their
/// concatenation is in the vocabulary, and the merge with the highest score goes first (the leftmost on ties).
/// Text reaches it with every space written as U+2581 "▁" and, with `add_dummy_prefix`, one more in front, so
/// "Hello world" is "▁Hello▁world". A character no piece covers falls back to its bytes, as the tokens `<0x00>`
/// to `<0xFF>`.
///
/// - `sentencepiece_escape(text, out, add_dummy_prefix)` writes the escaped text (empty text stays empty, as
///   llama.cpp and SentencePiece give no tokens for it).
/// - `sentencepiece_bpe_encode(escaped, token, score, bytes, out, scratch)` merges by score: `merge_encode`
///   (`ml/bpe_merges.hpp`) over code points with the priority minus the merged piece's score. `token` and `score`
///   are the vocabulary's lookups; `bytes` is a `byte_fallback`, the 256 byte tokens found once.
/// - `byte_fallback` / `make_byte_fallback(token)`: each byte's `<0xNN>` token, or, failing that, the token of the
///   byte itself (llama.cpp's rule).
/// - `sentencepiece_decode(tokens, piece, kind, out, options)` renders tokens as text: a normal piece with "▁" as
///   a space, a byte token as its byte, a user-defined token as written, control and unknown tokens as nothing
///   (unless asked), and the dummy prefix's space dropped from the first piece.
///
/// \snippet ml/sentencepiece_demo.cpp sentencepiece
///
/// **Relationships.** `ml/bpe_merges.hpp` (the merge engine), `ml/token_trie.hpp` (a lookup and `token_kind`),
/// `ml/unigram.hpp` (SentencePiece's other model, which decodes the same way), `ml/pretokenize.hpp`
/// (`split_added_tokens`, for user-defined tokens in the raw text). Group: \ref core.
///
/// **Implementations.** One, which is llama.cpp's `llm_tokenizer_spm_session`: a symbol that is not a token is
/// written as bytes, and a merged symbol always is one (llama.cpp does not resegment unused pieces as SentencePiece
/// can, and nor does this).
///
/// **Guarantees.** `constexpr` when the lookups are. No allocation. Complexity: O(n log n) in the escaped text's
/// code points.
///
/// **Errors.** `bpe_error::output_too_small` (escaping needs `sentencepiece_escaped_size(text.size())` bytes,
/// encoding `escaped.size()` tokens, decoding the rendered bytes), `scratch_too_small`, and `unknown_bytes` when a
/// byte has no token or a token's piece is unknown to `piece`.
///
/// **Performance.** Benchmarked through `merge_encode` (`benchmarks/ml/bpe_merges_benchmark.cpp`); the scoring is
/// one lookup per candidate pair.
///
/// **Tests and demos.** `tests/ml/sentencepiece_test.cpp`: escaping, merging by score, ties, byte fallback,
/// decoding, the errors, at compile time. `tests/ml/tokenizer_test.cpp`: token for token llama.cpp's test vectors
/// for Llama (llama-spm) and Phi-3 on committed subsets of their vocabularies, and decoding back to the text.
/// `demos/ml/sentencepiece_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<cstddef>`, `<cstdint>`, `<span>`, `<string_view>`, `ml/bpe_merges.hpp`,
/// `ml/token_trie.hpp`.
///
/// **Known issues.** Hugging Face's decoder also strips spaces around punctuation for some models (`clean_spaces`);
/// that cleanup is not here.
///
/// **Status.** Experimental, new in 0.17.0.

#ifndef snicholls_core_ml_sentencepiece_hpp
#define snicholls_core_ml_sentencepiece_hpp

#include <snicholls/core/ml/bpe_merges.hpp>
#include <snicholls/core/ml/token_trie.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace snicholls::ml
{

/// \brief U+2581 LOWER ONE EIGHTH BLOCK, SentencePiece's escaped space, in UTF-8.
inline constexpr std::string_view escaped_space = "\xE2\x96\x81";

/// \brief The bytes `sentencepiece_escape` may write for `n` bytes of text: `3n + 3`.
/// \param n The text's length.
/// \return The bound.
constexpr std::size_t sentencepiece_escaped_size(std::size_t n) noexcept { return 3 * n + 3; }

/// \brief Escapes spaces to "▁", with one more in front when `add_dummy_prefix` (and the text is not empty).
/// \param text The text.
/// \param out Receives the escaped text: `sentencepiece_escaped_size(text.size())` bytes always suffice.
/// \param add_dummy_prefix Whether to prepend "▁" (SentencePiece's `add_dummy_prefix`, GGUF's `add_space_prefix`).
/// \return The bytes written, or `output_too_small`.
constexpr result<std::size_t, bpe_error> sentencepiece_escape(std::string_view text, std::span<char> out, bool add_dummy_prefix = true) noexcept
{
    std::size_t at = 0;
    const auto put_space = [&]() {
        if (out.size() - at < escaped_space.size())
            return false;
        for (const char c : escaped_space)
            out[at++] = c;
        return true;
    };
    if (text.empty())
        return ok(std::size_t{0});
    if (add_dummy_prefix && !put_space())
        return err(bpe_error::output_too_small);
    for (const char c : text) {
        if (c == ' ') {
            if (!put_space())
                return err(bpe_error::output_too_small);
        } else {
            if (at == out.size())
                return err(bpe_error::output_too_small);
            out[at++] = c;
        }
    }
    return ok(at);
}

/// \brief Each byte's fallback token: `<0xNN>`, or the byte as a piece, or `no_token`.
/// \ingroup core
struct byte_fallback
{
    std::array<std::uint32_t, 256> ids{};   ///< The token for each byte.

    /// \brief The token for a byte.
    /// \param b The byte.
    /// \return Its fallback token, or `no_token`.
    constexpr std::uint32_t operator()(unsigned char b) const noexcept { return ids[b]; }
};

/// \brief Finds the 256 byte tokens: `<0xNN>` (upper-case hex), else the single byte's piece (llama.cpp's
/// `byte_to_token`).
/// \tparam Token A callable `token(std::string_view) -> std::uint32_t`.
/// \param token The vocabulary lookup.
/// \return The fallback table.
template <typename Token>
constexpr byte_fallback make_byte_fallback(Token&& token)
{
    byte_fallback f;
    constexpr char hex[] = "0123456789ABCDEF";
    for (std::size_t b = 0; b < 256; ++b) {
        const char name[6] = {'<', '0', 'x', hex[b >> 4], hex[b & 15], '>'};
        std::uint32_t id = token(std::string_view(name, 6));
        if (id == no_token) {
            const char single = static_cast<char>(b);
            id = token(std::string_view(&single, 1));
        }
        f.ids[b] = id;
    }
    return f;
}

/// \brief Encodes escaped text with SentencePiece's BPE: merges by score, byte fallback.
/// \tparam Token A callable `token(std::string_view) -> std::uint32_t`, `no_token` if not a token.
/// \tparam Score A callable `score(std::uint32_t id) -> float`.
/// \param escaped The text after `sentencepiece_escape`.
/// \param token The vocabulary lookup (llama.cpp's: every token, control tokens included).
/// \param score Each token's score.
/// \param bytes The byte tokens.
/// \param out Receives the tokens: `escaped.size()` slots always suffice.
/// \param scratch At least `merge_scratch_size(escaped.size())` entries.
/// \return The number of tokens written, or why not.
template <typename Token, typename Score>
constexpr result<std::size_t, bpe_error> sentencepiece_bpe_encode(std::string_view escaped, Token&& token, Score&& score, const byte_fallback& bytes,
                                                                  std::span<std::uint32_t> out, std::span<merge_entry<float>> scratch)
{
    // Not lambdas capturing the lookups: MSVC cannot read a captured reference to a function during constant
    // evaluation.
    struct priority
    {
        Token& token;
        Score& score;
        constexpr float operator()(std::string_view left, std::string_view right) const
        {
            const std::uint32_t id = token(std::string_view(left.data(), left.size() + right.size()));
            return id == no_token ? merge_never<float> : -static_cast<float>(score(id));
        }
    };
    return merge_encode(escaped, priority{token, score}, token, bytes, out, scratch);
}

/// \brief Options of `sentencepiece_decode`.
/// \ingroup core
struct sentencepiece_decode_options
{
    bool strip_leading_space = true;   ///< Drop one space from the start of the first piece (the dummy prefix's).
    bool render_special = false;       ///< Write control and unknown tokens' pieces instead of nothing.
};

/// \brief Renders SentencePiece tokens (BPE or Unigram) as text.
/// \tparam Piece A callable `piece(std::uint32_t id) -> std::string_view`, empty for an unknown id.
/// \tparam Kind A callable `kind(std::uint32_t id) -> token_kind`.
/// \param tokens The tokens.
/// \param piece Each token's piece.
/// \param kind Each token's kind.
/// \param out Receives the text.
/// \param options Leading space and special tokens.
/// \return The bytes written, `unknown_bytes` for an id `piece` does not know, or `output_too_small`.
template <typename Piece, typename Kind>
constexpr result<std::size_t, bpe_error> sentencepiece_decode(std::span<const std::uint32_t> tokens, Piece&& piece, Kind&& kind,
                                                              std::span<char> out, sentencepiece_decode_options options = {})
{
    std::size_t at = 0;
    bool first = options.strip_leading_space;
    const auto put = [&](char c) {
        if (at == out.size())
            return false;
        out[at++] = c;
        return true;
    };
    const auto hex = [](char c) -> int {
        if (c >= '0' && c <= '9')
            return c - '0';
        if (c >= 'A' && c <= 'F')
            return c - 'A' + 10;
        if (c >= 'a' && c <= 'f')
            return c - 'a' + 10;
        return -1;
    };
    for (const std::uint32_t id : tokens) {
        const std::string_view p = piece(id);
        if (p.empty())
            return err(bpe_error::unknown_bytes);
        const token_kind k = kind(id);
        if ((k == token_kind::control || k == token_kind::unknown) && !options.render_special)
            continue;
        if (k == token_kind::byte && p.size() == 6 && p.substr(0, 3) == "<0x" && hex(p[3]) >= 0 && hex(p[4]) >= 0) {
            if (!put(static_cast<char>(hex(p[3]) * 16 + hex(p[4]))))
                return err(bpe_error::output_too_small);
            first = false;
            continue;
        }
        const bool escape = k != token_kind::user_defined && k != token_kind::control && k != token_kind::unknown;
        for (std::size_t i = 0; i < p.size();) {
            if (escape && p.substr(i, escaped_space.size()) == escaped_space) {
                i += escaped_space.size();
                if (first && i == escaped_space.size())
                    continue;
                if (!put(' '))
                    return err(bpe_error::output_too_small);
                continue;
            }
            if (!put(p[i++]))
                return err(bpe_error::output_too_small);
        }
        first = false;
    }
    return ok(at);
}

} // namespace snicholls::ml

#endif // snicholls_core_ml_sentencepiece_hpp
