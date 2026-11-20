//
//  tokenizer.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::ml::tokenize` and `detokenize`: a whole tokenizer (added tokens, normalisation, pre-tokenisation, model, byte fallback) over a `tokenizer_vocab`, the non-owning view a GGUF or `tokenizer.json` loader fills.
/// \ingroup core
///
/// **Overview.** The building blocks of `ml/pretokenize.hpp`, `ml/bpe_merges.hpp`, `ml/sentencepiece.hpp`,
/// `ml/unigram.hpp`, `ml/wordpiece.hpp` and `ml/rwkv_tokenizer.hpp` assembled the way llama.cpp's `llama_tokenize` assembles
/// them, for the five model types a tokenizer file can name:
///
/// | `tokenizer_model` | GGUF `tokenizer.ggml.model` | pipeline |
/// |---|---|---|
/// | `sentencepiece_bpe` | `llama` | escape spaces ("▁", dummy prefix), merge by score, `<0xNN>` fallback |
/// | `bpe` | `gpt2`, `gemma4` | pre-tokenise, GPT-2's byte alphabet (or escaped spaces), merge by rank, byte fallback |
/// | `unigram` | `t5` | charsmap and whitespace normaliser, Viterbi, one unknown per run |
/// | `wordpiece` | `bert` | BERT's basic tokenisation, greedy longest match |
/// | `rwkv` | `rwkv` | greedy longest match over raw bytes |
///
/// Before any of them, added tokens (user-defined tokens always, special tokens when the caller puts them in
/// `added`) are cut out of the raw text (`split_added_tokens`) and emitted as themselves. A `sentencepiece_bpe` text
/// fragment gets the dummy prefix when it opens the text or follows an added token, as llama.cpp does.
///
/// **The seam.** `tokenizer_vocab` is what a loader fills from a tokenizer file, all views into memory it owns:
/// the pieces (as stored: GPT-2's alphabet for byte-level BPE, "▁" for SentencePiece, "▁" word starts for GGUF's
/// WordPiece, unescaped bytes for RWKV), scores and kinds by id, the tries (`build_token_index`), the merge index
/// (`split_merge`, `build_merge_index`), the pre-tokenizer (`pretokenizer_for(tokenizer.ggml.pre)`), the flags, the
/// charsmap blob, the unknown token, and the byte fallback (`make_byte_fallback`, `make_byte_level_fallback`).
/// Nothing here reads a file.
///
/// \snippet ml/tokenizer_demo.cpp tokenizer
///
/// **Relationships.** Every tokenizer header above. Group: \ref core.
///
/// **Implementations.** One per model type, each the corresponding header's algorithm.
///
/// **Guarantees.** No allocation: the caller's output and `tokenizer_scratch` (`tokenizer_scratch_need` sizes it
/// for a text length). `tokenize` is `constexpr` when the vocabulary is. Complexity: that of the model (O(n log n)
/// for the merges, O(n · L log V) for the tries).
///
/// **Errors.** `tokenizer_error::output_too_small` (`out` short; `3n + 3` slots suffice for n bytes of text),
/// `scratch_too_small`, `unknown_bytes` (a byte with no token), `bad_charsmap`.
///
/// **Performance.** The models' own benchmarks; the pipeline adds one pass over the text per stage.
///
/// **Tests and demos.** `tests/ml/tokenizer_test.cpp`: token for token llama.cpp's test vectors
/// (`models/ggml-vocab-*.gguf.inp/.out`, plus edge cases: empty text, whitespace runs, emoji, CJK, combining
/// marks, invalid UTF-8) for Llama (SentencePiece), Phi-3, GPT-2, Llama 3, Qwen 2, Qwen 3.5, DeepSeek LLM and
/// Coder, Falcon, StarCoder, Refact, Command R, MPT, Gemma 4, BGE (WordPiece) and nomic-embed (Unigram, XLM-R's
/// vocabulary and charsmap), on committed subsets of those vocabularies (`tests/fixtures/ml/tokenizers/`, written
/// by `scripts/generate_tokenizer_fixtures.cpp` from llama.cpp itself); and every detokenisation back to the text.
/// `demos/ml/tokenizer_demo.cpp`.
///
/// **Dependencies.** `<cstddef>`, `<cstdint>`, `<span>`, `<string_view>`, and the tokenizer headers.
///
/// **Known issues.** A BPE model sees invalid UTF-8 as U+FFFD, as llama.cpp's does (and as Hugging Face sees
/// text decoded lossily), so its tokens decode to the repaired text; the SentencePiece models keep the bytes (byte
/// fallback). Where core and llama.cpp differ (llama.cpp's SentencePiece path cuts an invalid sequence by its lead
/// byte; it treats Phi-3's user-defined "</s>" as a control token; it strips BERT's Hangul to its first jamo), the
/// fixtures record both and the tests check core's. Special
/// tokens' `lstrip`/`rstrip`, Metaspace's `prepend_scheme`, BERT's `[CLS]`/`[SEP]` and BOS/EOS are the caller's.
///
/// **Status.** Experimental, new in 0.17.0.

#ifndef snicholls_core_ml_tokenizer_hpp
#define snicholls_core_ml_tokenizer_hpp

#include <snicholls/core/ml/bpe_merges.hpp>
#include <snicholls/core/ml/pretokenize.hpp>
#include <snicholls/core/ml/rwkv_tokenizer.hpp>
#include <snicholls/core/ml/sentencepiece.hpp>
#include <snicholls/core/ml/token_trie.hpp>
#include <snicholls/core/ml/unigram.hpp>
#include <snicholls/core/ml/wordpiece.hpp>

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace snicholls::ml
{

/// \brief The model a tokenizer file names.
/// \ingroup core
enum class tokenizer_model
{
    sentencepiece_bpe,   ///< SentencePiece BPE (GGUF `llama`): Llama 1 and 2, Mistral, Phi-3.
    bpe,                 ///< Merge-list BPE (GGUF `gpt2`, `gemma4`): GPT-2, Llama 3, Qwen, DeepSeek, Gemma 4.
    unigram,             ///< SentencePiece Unigram (GGUF `t5`): T5, ALBERT, XLM-R.
    wordpiece,           ///< WordPiece (GGUF `bert`): BERT, BGE, MiniLM.
    rwkv,                ///< RWKV's greedy trie tokenizer.
};

/// \brief Why `tokenize` or `detokenize` failed.
/// \ingroup core
enum class tokenizer_error
{
    output_too_small,    ///< The output is too short.
    scratch_too_small,   ///< A scratch span is too short (`tokenizer_scratch_need`).
    unknown_bytes,       ///< A byte has no token, or a token id no piece.
    bad_charsmap,        ///< The Unigram charsmap blob is malformed.
};

/// \brief A tokenizer, as views into a loader's memory.
/// \ingroup core
struct tokenizer_vocab
{
    tokenizer_model model = tokenizer_model::bpe;   ///< The model type.
    std::span<const std::string_view> pieces;       ///< Every token's piece, by id, as the file stores it.
    std::span<const float> scores;                  ///< Scores by id (SentencePiece models); empty otherwise.
    std::span<const token_kind> kinds;              ///< Kinds by id; empty means every token is normal.
    token_trie vocab;                               ///< The lookup trie (Unigram: normal, user-defined, unused).
    token_trie added;                               ///< Tokens cut out of raw text first; empty for none.
    merge_table merges;                             ///< BPE's merge index.
    pretokenizer pre = pretokenizer::gpt2;          ///< BPE's pre-tokenizer.
    bool byte_level = true;                         ///< BPE: pieces in GPT-2's alphabet (false: Gemma 4 style).
    bool ignore_merges = false;                     ///< BPE: a piece that is a token is that token (Llama 3).
    bool escape_whitespaces = false;                ///< BPE: spaces become "▁" before pre-tokenising (Gemma 4).
    bool add_space_prefix = true;                   ///< SentencePiece BPE: the dummy prefix.
    precompiled_charsmap charsmap;                  ///< Unigram's charsmap.
    unigram_normalizer_options normalizer;          ///< Unigram's whitespace rules.
    wordpiece_options wordpiece;                    ///< WordPiece's settings.
    std::uint32_t unknown = no_token;               ///< The unknown token (Unigram, WordPiece, RWKV).
    double unknown_score = 0;                       ///< Unigram: `unigram_unknown_score(lowest normal score)`.
    byte_fallback bytes;                            ///< Each byte's token (SentencePiece BPE, BPE).

    /// \brief A token's kind.
    /// \param id A token id.
    /// \return Its kind (normal when `kinds` is empty or the id is out of range).
    constexpr token_kind kind(std::uint32_t id) const noexcept
    {
        return id < kinds.size() ? kinds[id] : token_kind::normal;
    }
    /// \brief A token's score.
    /// \param id A token id.
    /// \return Its score (0 when `scores` is empty or the id is out of range).
    constexpr float score(std::uint32_t id) const noexcept { return id < scores.size() ? scores[id] : 0.0f; }
    /// \brief A token's piece.
    /// \param id A token id.
    /// \return Its piece, empty for an unknown id.
    constexpr std::string_view piece(std::uint32_t id) const noexcept
    {
        return id < pieces.size() ? pieces[id] : std::string_view{};
    }
};

/// \brief Each byte's token in a byte-level vocabulary: the piece that spells the byte in GPT-2's alphabet.
/// \tparam Token A callable `token(std::string_view) -> std::uint32_t`.
/// \param token The vocabulary lookup.
/// \return The fallback table (`no_token` for a byte the vocabulary lacks).
template <typename Token>
constexpr byte_fallback make_byte_level_fallback(Token&& token)
{
    byte_fallback f;
    for (std::size_t b = 0; b < 256; ++b) {
        char s[4] = {};
        const std::size_t n = encode_utf8(byte_to_unicode[b], s);
        f.ids[b] = token(std::string_view(s, n));
    }
    return f;
}

/// \brief The working space of `tokenize`.
/// \ingroup core
struct tokenizer_scratch
{
    std::span<char> text;                               ///< Escaped, mapped or normalised text.
    std::span<merge_entry<std::uint32_t>> ranked;       ///< BPE's merge scratch.
    std::span<merge_entry<float>> scored;               ///< SentencePiece BPE's merge scratch.
    std::span<unigram_lattice_entry> lattice;           ///< Unigram's lattice.
};

/// \brief The scratch sizes `tokenize` needs for a text of `n` bytes.
/// \ingroup core
struct tokenizer_scratch_need
{
    std::size_t text = 0;      ///< Bytes of `tokenizer_scratch::text`.
    std::size_t ranked = 0;    ///< Entries of `tokenizer_scratch::ranked`.
    std::size_t scored = 0;    ///< Entries of `tokenizer_scratch::scored`.
    std::size_t lattice = 0;   ///< Entries of `tokenizer_scratch::lattice`.

    /// \brief The sizes for `vocab` and `n` bytes of text.
    /// \param vocab The tokenizer.
    /// \param n The text's length in bytes.
    constexpr tokenizer_scratch_need(const tokenizer_vocab& vocab, std::size_t n) noexcept
    {
        switch (vocab.model) {
            case tokenizer_model::sentencepiece_bpe:
                text = sentencepiece_escaped_size(n);
                scored = merge_scratch_size(text);
                break;
            case tokenizer_model::bpe:   // invalid UTF-8 repaired (3n), then escaped or mapped
                text = 3 * n + (vocab.escape_whitespaces ? sentencepiece_escaped_size(3 * n) : 6 * n);
                ranked = merge_scratch_size(vocab.escape_whitespaces ? sentencepiece_escaped_size(3 * n) : 6 * n);
                break;
            case tokenizer_model::unigram:
                text = unigram_normalized_size(n, vocab.charsmap);
                lattice = text + 1;
                break;
            case tokenizer_model::wordpiece:
                text = wordpiece_scratch_size(n);
                break;
            case tokenizer_model::rwkv:
                break;
        }
    }
};

namespace detail::tokenizer
{
    constexpr tokenizer_error from(bpe_error e) noexcept
    {
        return e == bpe_error::output_too_small    ? tokenizer_error::output_too_small
               : e == bpe_error::scratch_too_small ? tokenizer_error::scratch_too_small
                                                   : tokenizer_error::unknown_bytes;
    }
    constexpr tokenizer_error from(unigram_error e) noexcept
    {
        return e == unigram_error::output_too_small    ? tokenizer_error::output_too_small
               : e == unigram_error::lattice_too_small ? tokenizer_error::scratch_too_small
                                                       : tokenizer_error::bad_charsmap;
    }
    constexpr tokenizer_error from(wordpiece_error e) noexcept
    {
        return e == wordpiece_error::scratch_too_small ? tokenizer_error::scratch_too_small : tokenizer_error::output_too_small;
    }
    constexpr tokenizer_error from(rwkv_error) noexcept { return tokenizer_error::output_too_small; }

    // The token lookup of a vocabulary, as a callable
    struct finder
    {
        const token_trie* trie;
        constexpr std::uint32_t operator()(std::string_view s) const noexcept { return trie->find(s); }
    };
    struct scorer
    {
        const tokenizer_vocab* v;
        constexpr float operator()(std::uint32_t id) const noexcept { return v->score(id); }
    };
    struct kinder
    {
        const tokenizer_vocab* v;
        constexpr token_kind operator()(std::uint32_t id) const noexcept { return v->kind(id); }
    };

    // One run of ordinary text (no added tokens) through the model; first_fragment: whether it opens the text or
    // follows an added token
    constexpr result<std::size_t, tokenizer_error> fragment(const tokenizer_vocab& v, std::string_view text, bool first_fragment,
                                                            std::span<std::uint32_t> out, const tokenizer_scratch& s)
    {
        const tokenizer_scratch_need need(v, text.size());
        if (s.text.size() < need.text || s.ranked.size() < need.ranked || s.scored.size() < need.scored ||
            s.lattice.size() < need.lattice)
            return err(tokenizer_error::scratch_too_small);
        switch (v.model) {
            case tokenizer_model::sentencepiece_bpe: {
                const auto e = sentencepiece_escape(text, s.text, v.add_space_prefix && first_fragment);
                if (e.is_err())
                    return err(from(e.unwrap_err()));
                const auto r = sentencepiece_bpe_encode(std::string_view(s.text.data(), e.unwrap()), finder{&v.vocab}, scorer{&v}, v.bytes, out, s.scored);
                if (r.is_err())
                    return err(from(r.unwrap_err()));
                return ok(r.unwrap());
            }
            case tokenizer_model::bpe: {
                // As llama.cpp's split does, an invalid byte becomes U+FFFD; then spaces are escaped (Gemma 4) or
                // each piece is written in GPT-2's alphabet, after the repaired text in the scratch
                std::string_view source = text;
                std::span<char> rest = s.text;
                if (!is_valid_utf8(text)) {
                    const std::size_t n = replace_invalid_utf8(text, s.text);
                    source = std::string_view(s.text.data(), n);
                    rest = s.text.subspan(n);
                }
                if (v.escape_whitespaces) {
                    const auto e = sentencepiece_escape(source, rest, false);
                    if (e.is_err())
                        return err(from(e.unwrap_err()));
                    source = std::string_view(rest.data(), e.unwrap());
                    rest = rest.subspan(e.unwrap());
                }
                std::size_t count = 0;
                bpe_error failure{};
                bool failed = false;
                snicholls::ml::pretokenize(v.pre, source, [&](std::string_view piece) {
                    if (failed)
                        return;
                    merge_options o{merge_units::code_points, v.ignore_merges, v.byte_level};
                    std::string_view units = piece;
                    if (v.byte_level) {
                        const auto m = byte_level_encode(piece, rest);
                        if (m.is_err()) {
                            failed = true;
                            failure = m.unwrap_err();
                            return;
                        }
                        units = std::string_view(rest.data(), m.unwrap());
                    } else if (piece.find_first_not_of('\n') == std::string_view::npos) {
                        o.ignore_merges = true;   // Gemma 4: a run of newlines is one token if there is one
                    }
                    const auto r = merge_encode(units, v.merges, finder{&v.vocab}, v.bytes, out.subspan(count), s.ranked, o);
                    if (r.is_err()) {
                        failed = true;
                        failure = r.unwrap_err();
                        return;
                    }
                    count += r.unwrap();
                });
                if (failed)
                    return err(from(failure));
                return ok(count);
            }
            case tokenizer_model::unigram: {
                const auto n = unigram_normalize(text, v.charsmap, v.added, v.normalizer, s.text);
                if (n.is_err())
                    return err(from(n.unwrap_err()));
                const auto r = unigram_encode(std::string_view(s.text.data(), n.unwrap()), v.vocab, scorer{&v}, kinder{&v},
                                              unigram_unknown{v.unknown, v.unknown_score}, out, s.lattice);
                if (r.is_err())
                    return err(from(r.unwrap_err()));
                return ok(r.unwrap());
            }
            case tokenizer_model::wordpiece: {
                const auto r = wordpiece_encode(text, v.vocab, v.unknown, v.wordpiece, out, s.text);
                if (r.is_err())
                    return err(from(r.unwrap_err()));
                return ok(r.unwrap());
            }
            case tokenizer_model::rwkv: {
                const auto r = rwkv_encode(text, v.vocab, v.unknown, out);
                if (r.is_err())
                    return err(from(r.unwrap_err()));
                return ok(r.unwrap());
            }
        }
        return ok(std::size_t{0});
    }
} // namespace detail::tokenizer

/// \brief Tokenizes text: added tokens first, then the model over each run of text between them.
/// \param vocab The tokenizer.
/// \param text The raw text.
/// \param out Receives the tokens (`3 * text.size() + 3` slots always suffice).
/// \param scratch Working space, sized by `tokenizer_scratch_need(vocab, text.size())`.
/// \return The number of tokens written, or why not.
constexpr result<std::size_t, tokenizer_error> tokenize(const tokenizer_vocab& vocab, std::string_view text, std::span<std::uint32_t> out,
                                                        const tokenizer_scratch& scratch)
{
    std::size_t count = 0;
    bool first = true, failed = false;
    tokenizer_error failure{};
    split_added_tokens(
        text, vocab.added,
        [&](std::string_view run) {
            if (failed)
                return;
            const auto r = detail::tokenizer::fragment(vocab, run, first, out.subspan(count), scratch);
            if (r.is_err()) {
                failed = true;
                failure = r.unwrap_err();
                return;
            }
            count += r.unwrap();
            first = false;
        },
        [&](std::uint32_t id) {
            if (failed)
                return;
            if (count == out.size()) {
                failed = true;
                failure = tokenizer_error::output_too_small;
                return;
            }
            out[count++] = id;
            first = true;
        });
    if (failed)
        return err(failure);
    return ok(count);
}

/// \brief Renders tokens as text: the inverse of `tokenize` up to normalisation.
/// \param vocab The tokenizer.
/// \param tokens The tokens.
/// \param out Receives the text.
/// \param render_special Write control and unknown tokens' pieces instead of nothing.
/// \return The bytes written, or `unknown_bytes` / `output_too_small`.
constexpr result<std::size_t, tokenizer_error> detokenize(const tokenizer_vocab& vocab, std::span<const std::uint32_t> tokens, std::span<char> out,
                                                          bool render_special = false)
{
    const auto piece = [&](std::uint32_t id) { return vocab.piece(id); };
    const auto kind = [&](std::uint32_t id) { return vocab.kind(id); };
    const auto special = [&](std::uint32_t id) {
        const token_kind k = vocab.kind(id);
        return !render_special && (k == token_kind::control || k == token_kind::unknown);
    };
    switch (vocab.model) {
        case tokenizer_model::sentencepiece_bpe:
        case tokenizer_model::unigram: {
            const auto r = sentencepiece_decode(tokens, piece, kind, out,
                                                {vocab.model == tokenizer_model::unigram ? vocab.normalizer.add_dummy_prefix : vocab.add_space_prefix,
                                                 render_special});
            if (r.is_err())
                return err(detail::tokenizer::from(r.unwrap_err()));
            return ok(r.unwrap());
        }
        case tokenizer_model::bpe: {
            if (!vocab.byte_level) {
                const auto r = sentencepiece_decode(tokens, piece, kind, out, {false, render_special});
                if (r.is_err())
                    return err(detail::tokenizer::from(r.unwrap_err()));
                return ok(r.unwrap());
            }
            std::size_t at = 0;
            for (const std::uint32_t id : tokens) {
                const std::string_view p = vocab.piece(id);
                if (p.empty())
                    return err(tokenizer_error::unknown_bytes);
                if (special(id))
                    continue;
                const token_kind k = vocab.kind(id);
                if (k == token_kind::user_defined || k == token_kind::control || k == token_kind::unknown) {
                    if (out.size() - at < p.size())
                        return err(tokenizer_error::output_too_small);
                    for (const char c : p)
                        out[at++] = c;
                    continue;
                }
                const auto r = byte_level_decode(p, out.subspan(at));
                if (r.is_err())
                    return err(detail::tokenizer::from(r.unwrap_err()));
                at += r.unwrap();
            }
            return ok(at);
        }
        case tokenizer_model::wordpiece: {
            const auto r = wordpiece_decode(tokens, piece, special, vocab.wordpiece, out);
            if (r.is_err())
                return err(r.unwrap_err() == wordpiece_error::unknown_id ? tokenizer_error::unknown_bytes : tokenizer_error::output_too_small);
            return ok(r.unwrap());
        }
        case tokenizer_model::rwkv: {
            const auto r = bpe_decode(tokens, piece, out);
            if (r.is_err())
                return err(detail::tokenizer::from(r.unwrap_err()));
            return ok(r.unwrap());
        }
    }
    return ok(std::size_t{0});
}

} // namespace snicholls::ml

#endif // snicholls_core_ml_tokenizer_hpp
