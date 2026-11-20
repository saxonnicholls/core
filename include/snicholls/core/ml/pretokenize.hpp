//
//  pretokenize.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::ml` pre-tokenisers: the split patterns of GPT-2, Llama 3, Qwen 2 and 3.5, GPT-4o (o200k), DeepSeek V3, LLM and Coder, Falcon, StarCoder and Gemma 4 as hand-written `constexpr` matchers over Unicode categories (no regex engine), the splits that compose them, and the added-token split.
/// \ingroup core
///
/// **Overview.** Before a byte-pair or SentencePiece model sees text, the tokenizer cuts it into pieces with a
/// regular expression, and a merge never crosses a cut. Each pattern here is that expression written out as code
/// that walks UTF-8 and asks `ml/unicode.hpp` for categories: what the regex engine would match, alternative by
/// alternative and with its backtracking, decided per code point and without allocating.
///
/// - A **pattern** is a callable `p(text, at) -> std::size_t`: the length in bytes of its match starting at byte
///   `at`, or 0. `gpt2_pattern`, `llama3_pattern`, `qwen2_pattern`, `qwen35_pattern`, `gpt4o_pattern`,
///   `deepseek_v3_pattern` and the smaller ones (`digits_pattern<Max>`, `cjk_pattern`, `newline_pattern`, ...)
///   are documented with the expression each one implements.
/// - `split_isolated(text, pattern, emit)` is a regex split that keeps both the matches and the text between them
///   as pieces (Hugging Face's `Split` with `Isolated`, and llama.cpp's `unicode_regex_split`). `split_sequence`
///   applies several patterns in turn, each to the pieces of the last (Hugging Face's `Sequence`).
/// - `pretokenize(kind, text, emit)` runs a tokenizer family's whole split by name (`pretokenizer`), and
///   `pretokenizer_for(name)` maps GGUF's `tokenizer.ggml.pre` strings to it.
/// - `split_added_tokens(text, added, on_text, on_token)` cuts out the added tokens (special and user-defined
///   tokens matched in the raw text before anything else), leftmost and longest first.
///
/// Every piece is a `std::string_view` into the caller's text.
///
/// \snippet ml/pretokenize_demo.cpp pretokenize
///
/// **Relationships.** `ml/unicode.hpp` (categories), `ml/token_trie.hpp` (the added tokens), and the models that
/// consume the pieces: `ml/bpe.hpp` (tiktoken's merge), `ml/bpe_merges.hpp` (Hugging Face's and GGUF's merges),
/// `ml/sentencepiece.hpp`. Group: \ref core.
///
/// **Implementations.** The patterns follow the expressions in Hugging Face's `tokenizer.json` files and
/// tiktoken, not llama.cpp's adaptations: GPT-4o's case classes are the real `\p{Lu}`/`\p{Ll}`, where llama.cpp
/// approximates them by "not a-z" and "not A-Z", and the case-insensitive contractions accept U+017F (ſ), which
/// folds to s. Where llama.cpp itself runs a sequence of expressions (DeepSeek, Falcon, StarCoder) the sequence
/// is llama.cpp's, which is what its test vectors were checked against.
///
/// **Guarantees.** `constexpr`; no allocation; the pieces cover the text exactly, in order, none empty.
/// Complexity: linear in the text for every pattern but `gpt4o_pattern`, whose case classes backtrack over one
/// letter run (linear in the run). An invalid byte is one unit with U+FFFD's category (other symbol), as llama.cpp
/// decodes it, so a split never stalls.
///
/// **Errors.** `pretokenizer_for` returns `pretokenize_error::unknown_name` for a name it does not know.
///
/// **Performance.** `benchmarks/ml/pretokenize_benchmark.cpp`, the Llama 3 split over the fixtures' texts (Release, Apple clang 17, Xeon W-3245, 2026-11-19, one thread on a loaded machine (ratios held across runs)):
/// 72 MiB/s, against 1.2 MiB/s for `std::regex` on the same expression over text with every code point collapsed to
/// a byte for its category (llama.cpp's general technique, the collapse included): sixty times faster.
///
/// **Tests and demos.** `tests/ml/pretokenize_test.cpp`: every pattern against Python's `regex` module running the
/// original expression, on committed fixtures (`tests/fixtures/ml/pretokenize_fixture.txt`, written by
/// `scripts/generate_pretokenize_fixture.py`), and each family's split through the token-for-token tests of
/// `tests/ml/tokenizer_test.cpp` against llama.cpp's vocabulary test vectors. `demos/ml/pretokenize_demo.cpp`.
///
/// **Dependencies.** `<cstddef>`, `<cstdint>`, `<initializer_list>`, `<span>`, `<string_view>`, `snicholls/core/result.hpp`, `ml/unicode.hpp`,
/// `ml/token_trie.hpp`.
///
/// **Known issues.** Unicode 16.0 categories; Python's `regex` and llama.cpp may be older, which matters only for
/// code points assigned since. Overlapping added tokens are resolved leftmost-longest, as Hugging Face's
/// Aho-Corasick does; llama.cpp splits by one added token at a time, longest first, which differs only when a
/// shorter one starts earlier inside a longer one's match. Added tokens' `lstrip` / `rstrip` are not applied.
///
/// **Status.** Experimental, new in 0.17.0.

#ifndef snicholls_core_ml_pretokenize_hpp
#define snicholls_core_ml_pretokenize_hpp

#include <snicholls/core/ml/token_trie.hpp>
#include <snicholls/core/ml/unicode.hpp>
#include <snicholls/core/result.hpp>

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <span>
#include <string_view>

namespace snicholls::ml
{

namespace detail::pretokenize
{
    // One code point of the text: its value, category and length. Past the end: length 0, category unassigned.
    struct unit
    {
        char32_t cp;
        unicode_category cat;
        std::size_t len;
    };

    constexpr unit at(std::string_view text, std::size_t i) noexcept
    {
        if (i >= text.size())
            return {0, unicode_category::unassigned, 0};
        const utf8_decoded d = decode_utf8(text, i);
        return {d.code_point, category(d.code_point), d.length};
    }

    constexpr bool letter(const unit& u) noexcept { return u.len && is_letter(u.cat); }
    constexpr bool number(const unit& u) noexcept { return u.len && is_number(u.cat); }
    constexpr bool space(const unit& u) noexcept { return u.len && is_whitespace(u.cp); }
    constexpr bool crlf(const unit& u) noexcept { return u.len && (u.cp == '\r' || u.cp == '\n'); }
    constexpr bool letter_or_mark(const unit& u) noexcept { return u.len && (is_letter(u.cat) || is_mark(u.cat)); }
    // [^\s\p{L}\p{N}]
    constexpr bool other(const unit& u) noexcept { return u.len && !space(u) && !letter(u) && !number(u); }
    // [^\r\n\p{L}\p{N}]
    constexpr bool lead(const unit& u) noexcept { return u.len && !crlf(u) && !letter(u) && !number(u); }

    // The bytes of the longest run of units satisfying pred from i (at most max units)
    template <typename Pred>
    constexpr std::size_t run(std::string_view text, std::size_t i, Pred pred, std::size_t max = ~std::size_t{0}) noexcept
    {
        std::size_t n = 0;
        for (std::size_t k = 0; k < max; ++k) {
            const unit u = at(text, i + n);
            if (!pred(u))
                break;
            n += u.len;
        }
        return n;
    }

    // [A]?[B]+ with a greedy optional: the bytes matched at i, or 0
    template <typename A, typename B>
    constexpr std::size_t optional_then_run(std::string_view text, std::size_t i, A a, B b) noexcept
    {
        const unit c = at(text, i);
        if (a(c))
            if (const std::size_t n = run(text, i + c.len, b); n > 0)
                return c.len + n;
        return run(text, i, b);
    }

    // 's|'t|'re|'ve|'m|'ll|'d, case-sensitive or (?i:...): the bytes matched at i, or 0
    constexpr bool folds_to(char32_t cp, char lower, bool fold) noexcept
    {
        if (cp == static_cast<char32_t>(lower))
            return true;
        return fold && (cp == static_cast<char32_t>(lower - 'a' + 'A') || (lower == 's' && cp == U'ſ'));
    }

    constexpr std::size_t contraction(std::string_view text, std::size_t i, bool fold) noexcept
    {
        if (i >= text.size() || text[i] != '\'')
            return 0;
        const unit a = at(text, i + 1);
        for (const char c : {'s', 't'})
            if (folds_to(a.cp, c, fold) && a.len)
                return 1 + a.len;
        const unit b = at(text, i + 1 + a.len);
        const auto pair = [&](char x, char y) { return a.len && b.len && folds_to(a.cp, x, fold) && folds_to(b.cp, y, fold); };
        if (pair('r', 'e') || pair('v', 'e'))
            return 1 + a.len + b.len;
        if (folds_to(a.cp, 'm', fold) && a.len)
            return 1 + a.len;
        if (pair('l', 'l'))
            return 1 + a.len + b.len;
        if (folds_to(a.cp, 'd', fold) && a.len)
            return 1 + a.len;
        return 0;
    }

    // ' ?' then a run of pred, then a run of tail: the bytes matched at i, or 0
    template <typename Pred, typename Tail>
    constexpr std::size_t space_run_tail(std::string_view text, std::size_t i, Pred pred, Tail tail) noexcept
    {
        const std::size_t sp = (i < text.size() && text[i] == ' ' && pred(at(text, i + 1))) ? 1 : 0;
        const std::size_t n = run(text, i + sp, pred);
        if (n == 0)
            return 0;
        return sp + n + run(text, i + sp + n, tail);
    }

    // \s*[\r\n]+ : up to and including the last \r or \n of the whitespace run at i, or 0
    constexpr std::size_t whitespace_to_newline(std::string_view text, std::size_t i) noexcept
    {
        std::size_t n = 0, end = 0;
        for (unit u = at(text, i); space(u); u = at(text, i + n)) {
            n += u.len;
            if (crlf(u))
                end = n;
        }
        return end;
    }

    // \s+(?!\S)|\s+ : the whitespace run at i, less its last unit when a non-space follows and the run is longer
    // than one unit
    constexpr std::size_t whitespace(std::string_view text, std::size_t i) noexcept
    {
        std::size_t n = 0, last = 0, units = 0;
        for (unit u = at(text, i); space(u); u = at(text, i + n)) {
            last = u.len;
            n += u.len;
            ++units;
        }
        if (units > 1 && i + n < text.size())
            return n - last;
        return n;
    }

    constexpr bool is_crlf_or_slash(const unit& u) noexcept { return u.len && (u.cp == '\r' || u.cp == '\n' || u.cp == '/'); }
    constexpr bool none(const unit&) noexcept { return false; }
} // namespace detail::pretokenize

/// \brief GPT-2's pattern (also MPT, OLMo, and the second step of Falcon and StarCoder):
/// \verbatim 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+ \endverbatim
/// \ingroup core
struct gpt2_pattern
{
    /// \brief The match at a byte offset.
    /// \param text The text (a piece being split).
    /// \param at A byte offset into it.
    /// \return The length in bytes of the pattern's match starting at `at`, or 0 if it does not match there.
    constexpr std::size_t operator()(std::string_view text, std::size_t at) const noexcept
    {
        namespace d = detail::pretokenize;
        if (const std::size_t n = d::contraction(text, at, false))
            return n;
        if (const std::size_t n = d::space_run_tail(text, at, d::letter, d::none))
            return n;
        if (const std::size_t n = d::space_run_tail(text, at, d::number, d::none))
            return n;
        if (const std::size_t n = d::space_run_tail(text, at, d::other, d::none))
            return n;
        return d::whitespace(text, at);
    }
};

/// \brief Llama 3's pattern (also DBRX, Smaug, ChatGLM 4), digits at most `MaxDigits` at a time:
/// \verbatim (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+ \endverbatim
/// With `MaxDigits == 1` it is Qwen 2's pattern (`qwen2_pattern`).
/// \tparam MaxDigits The digits in one piece: 3 for Llama 3, 1 for Qwen 2.
/// \ingroup core
template <std::size_t MaxDigits>
struct llama3_like_pattern
{
    /// \brief The match at a byte offset.
    /// \param text The text (a piece being split).
    /// \param at A byte offset into it.
    /// \return The length in bytes of the pattern's match starting at `at`, or 0 if it does not match there.
    constexpr std::size_t operator()(std::string_view text, std::size_t at) const noexcept
    {
        namespace d = detail::pretokenize;
        if (const std::size_t n = d::contraction(text, at, true))
            return n;
        if (const std::size_t n = d::optional_then_run(text, at, d::lead, d::letter))
            return n;
        if (const std::size_t n = d::run(text, at, d::number, MaxDigits))
            return n;
        if (const std::size_t n = d::space_run_tail(text, at, d::other, d::crlf))
            return n;
        if (const std::size_t n = d::whitespace_to_newline(text, at))
            return n;
        return d::whitespace(text, at);
    }
};

/// \brief Llama 3's pattern; see `llama3_like_pattern`.
using llama3_pattern = llama3_like_pattern<3>;
/// \brief Qwen 2's pattern (also StableLM 2): Llama 3's with one digit per piece.
using qwen2_pattern = llama3_like_pattern<1>;

/// \brief Qwen 3.5's pattern: Qwen 2's with marks inside letter runs:
/// \verbatim (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+ \endverbatim
/// \ingroup core
struct qwen35_pattern
{
    /// \brief The match at a byte offset.
    /// \param text The text (a piece being split).
    /// \param at A byte offset into it.
    /// \return The length in bytes of the pattern's match starting at `at`, or 0 if it does not match there.
    constexpr std::size_t operator()(std::string_view text, std::size_t at) const noexcept
    {
        namespace d = detail::pretokenize;
        if (const std::size_t n = d::contraction(text, at, true))
            return n;
        if (const std::size_t n = d::optional_then_run(text, at, d::lead, d::letter_or_mark))
            return n;
        if (const std::size_t n = d::run(text, at, d::number, 1))
            return n;
        constexpr auto other = [](const d::unit& u) { return d::other(u) && !is_mark(u.cat); };
        if (const std::size_t n = d::space_run_tail(text, at, other, d::crlf))
            return n;
        if (const std::size_t n = d::whitespace_to_newline(text, at))
            return n;
        return d::whitespace(text, at);
    }
};

/// \brief GPT-4o's pattern (o200k_base; also MiniMax M2):
/// \verbatim [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)? \endverbatim
/// \verbatim |[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)? \endverbatim
/// \verbatim |\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+ \endverbatim
/// \ingroup core
struct gpt4o_pattern
{
    /// \brief The match at a byte offset.
    /// \param text The text (a piece being split).
    /// \param at A byte offset into it.
    /// \return The length in bytes of the pattern's match starting at `at`, or 0 if it does not match there.
    constexpr std::size_t operator()(std::string_view text, std::size_t at) const noexcept
    {
        namespace d = detail::pretokenize;
        // Each word alternative with its optional lead: with the lead if that matches, else without
        for (const auto word : {&gpt4o_pattern::cased_lower, &gpt4o_pattern::cased_upper}) {
            const d::unit c = d::at(text, at);
            if (d::lead(c))
                if (const std::size_t n = word(text, at + c.len))
                    return c.len + n;
            if (const std::size_t n = word(text, at))
                return n;
        }
        if (const std::size_t n = d::run(text, at, d::number, 3))
            return n;
        if (const std::size_t n = d::space_run_tail(text, at, d::other, d::is_crlf_or_slash))
            return n;
        if (const std::size_t n = d::whitespace_to_newline(text, at))
            return n;
        return d::whitespace(text, at);
    }

private:
    static constexpr bool upper(const detail::pretokenize::unit& u) noexcept { return u.len && is_uppercase_class(u.cat); }
    static constexpr bool lower(const detail::pretokenize::unit& u) noexcept { return u.len && is_lowercase_class(u.cat); }

    // [Upper]*[Lower]+(contraction)? at i: [Upper]* backtracks to the last position from which [Lower]+ matches
    static constexpr std::size_t cased_lower(std::string_view text, std::size_t i) noexcept
    {
        namespace d = detail::pretokenize;
        // The ends of the units of the maximal upper run, to backtrack over
        std::size_t candidate = d::run(text, i, lower) > 0 ? i : ~std::size_t{0};   // [Upper]* empty
        for (std::size_t p = i;;) {
            const d::unit u = d::at(text, p);
            if (!upper(u))
                break;
            p += u.len;
            if (lower(d::at(text, p)))
                candidate = p;   // the longest [Upper]* after which a lower unit follows
        }
        if (candidate == ~std::size_t{0})
            return 0;
        const std::size_t end = candidate + d::run(text, candidate, lower);
        return end - i + d::contraction(text, end, true);
    }

    // [Upper]+[Lower]*(contraction)? at i
    static constexpr std::size_t cased_upper(std::string_view text, std::size_t i) noexcept
    {
        namespace d = detail::pretokenize;
        const std::size_t u = d::run(text, i, upper);
        if (u == 0)
            return 0;
        const std::size_t end = i + u + d::run(text, i + u, lower);
        return end - i + d::contraction(text, end, true);
    }
};

/// \brief `\p{N}{1,Max}`: up to `Max` numbers (`\p{N}` alone with `Max == 1`; `\p{N}+` with the default).
/// \tparam Max The most units in one match.
/// \ingroup core
template <std::size_t Max = ~std::size_t{0}>
struct digits_pattern
{
    /// \brief The match at a byte offset.
    /// \param text The text (a piece being split).
    /// \param at A byte offset into it.
    /// \return The length in bytes of the pattern's match starting at `at`, or 0 if it does not match there.
    constexpr std::size_t operator()(std::string_view text, std::size_t at) const noexcept
    {
        return detail::pretokenize::run(text, at, detail::pretokenize::number, Max);
    }
};

/// \brief `[\r\n]`: one carriage return or line feed.
/// \ingroup core
struct crlf_pattern
{
    /// \brief The match at a byte offset.
    /// \param text The text (a piece being split).
    /// \param at A byte offset into it.
    /// \return The length in bytes of the pattern's match starting at `at`, or 0 if it does not match there.
    constexpr std::size_t operator()(std::string_view text, std::size_t at) const noexcept
    {
        return at < text.size() && (text[at] == '\r' || text[at] == '\n') ? 1 : 0;
    }
};

/// \brief `[^\n]+|[\n]+`: runs of line feeds and runs of everything else (Gemma 4, whose model sees whole lines).
/// \ingroup core
struct newline_pattern
{
    /// \brief The match at a byte offset.
    /// \param text The text (a piece being split).
    /// \param at A byte offset into it.
    /// \return The length in bytes of the pattern's match starting at `at`, or 0 if it does not match there.
    constexpr std::size_t operator()(std::string_view text, std::size_t at) const noexcept
    {
        if (at >= text.size())
            return 0;
        const bool nl = text[at] == '\n';
        std::size_t i = at;
        while (i < text.size() && (text[i] == '\n') == nl)
            ++i;
        return i - at;
    }
};

/// \brief A run of code points in fixed ranges, optionally after one whitespace: `\s?[ranges]+`, or `[ranges]+`.
/// \ingroup core
struct ranges_pattern
{
    std::span<const char32_t> ranges;   ///< Inclusive ranges as pairs: first, last, first, last, ...
    bool leading_space = false;          ///< Whether one whitespace may come first (`\s?`).

    /// \brief Whether `cp` is in one of the ranges.
    /// \param cp The code point.
    /// \return Whether a range holds it.
    constexpr bool contains(char32_t cp) const noexcept
    {
        for (std::size_t i = 0; i + 1 < ranges.size(); i += 2)
            if (cp >= ranges[i] && cp <= ranges[i + 1])
                return true;
        return false;
    }

    /// \brief The match at a byte offset.
    /// \param text The text (a piece being split).
    /// \param at A byte offset into it.
    /// \return The length in bytes of the pattern's match starting at `at`, or 0 if it does not match there.
    constexpr std::size_t operator()(std::string_view text, std::size_t at) const noexcept
    {
        namespace d = detail::pretokenize;
        const auto in = [this](const d::unit& u) { return u.len && contains(u.cp); };
        if (leading_space)
            return d::optional_then_run(text, at, d::space, in);
        return d::run(text, at, in);
    }
};

namespace detail::pretokenize
{
    // [一-龥぀-ゟ゠-ヿ]: CJK unified ideographs to U+9FA5, hiragana, katakana (DeepSeek V3)
    inline constexpr char32_t cjk_kana[] = {0x4E00, 0x9FA5, 0x3040, 0x309F, 0x30A0, 0x30FF};
    // [一-龥ࠀ-一가-퟿]: as written in DeepSeek LLM and Coder's tokenizers, U+0800 to U+9FA5 and Hangul
    inline constexpr char32_t cjk_wide[] = {0x4E00, 0x9FA5, 0x0800, 0x4E00, 0xAC00, 0xD7FF};
    // DeepSeek LLM's letters: [A-Za-zµÀ-ÖØ-öø-ƺ ... 𞤀-𞥃], the cased letters of the alphabetic scripts
    inline constexpr char32_t deepseek_letters[] = {
        0x41, 0x5A, 0x61, 0x7A, 0xB5, 0xB5, 0xC0, 0xD6, 0xD8, 0xF6, 0xF8, 0x1BA, 0x1BC, 0x1BF, 0x1C4, 0x293,
        0x295, 0x2AF, 0x370, 0x373, 0x376, 0x377, 0x37B, 0x37D, 0x37F, 0x37F, 0x386, 0x386, 0x388, 0x38A,
        0x38C, 0x38C, 0x38E, 0x3A1, 0x3A3, 0x3F5, 0x3F7, 0x481, 0x48A, 0x52F, 0x531, 0x556, 0x10A0, 0x10C5,
        0x13A0, 0x13F5, 0x13F8, 0x13FD, 0x1C90, 0x1CBA, 0x1CBD, 0x1CBF, 0x1D00, 0x1D2B, 0x1D6B, 0x1D77,
        0x1D79, 0x1D9A, 0x1E00, 0x1F15, 0x1F18, 0x1F1D, 0x1F20, 0x1F45, 0x1F48, 0x1F4D, 0x1F50, 0x1F57,
        0x1F59, 0x1F59, 0x1F5B, 0x1F5B, 0x1F5D, 0x1F5D, 0x1F5F, 0x1F7D, 0x1F80, 0x1FB4, 0x1FB6, 0x1FBC,
        0x1FBE, 0x1FBE, 0x1FC2, 0x1FC4, 0x1FC6, 0x1FCC, 0x1FD0, 0x1FD3, 0x1FD6, 0x1FDB, 0x1FE0, 0x1FEC,
        0x1FF2, 0x1FF4, 0x1FF6, 0x1FFC, 0x2102, 0x2102, 0x2107, 0x2107, 0x210A, 0x2113, 0x2115, 0x2115,
        0x2119, 0x211D, 0x2124, 0x2124, 0x2126, 0x2126, 0x2128, 0x2128, 0x212A, 0x212D, 0x212F, 0x2134,
        0x2139, 0x2139, 0x213C, 0x213F, 0x2145, 0x2149, 0x214E, 0x214E, 0x2183, 0x2184, 0x2C00, 0x2C7B,
        0x2C7E, 0x2CE4, 0x2CEB, 0x2CEE, 0x2CF2, 0x2CF3, 0xA640, 0xA66D, 0xA680, 0xA69B, 0xA722, 0xA76F,
        0xA771, 0xA787, 0xA78B, 0xA78E, 0xAB70, 0xABBF, 0xFB00, 0xFB06, 0xFB13, 0xFB17, 0xFF21, 0xFF3A,
        0xFF41, 0xFF5A, 0x10400, 0x1044F, 0x104B0, 0x104D3, 0x104D8, 0x104FB, 0x10C80, 0x10CB2, 0x10CC0,
        0x10CF2, 0x118A0, 0x118DF, 0x1E900, 0x1E943};
    // DeepSeek LLM's punctuation: [!-/:-~！-／：-～‘-‟　-。]
    inline constexpr char32_t deepseek_punctuation[] = {0x21, 0x2F, 0x3A, 0x7E, 0xFF01, 0xFF0F, 0xFF1A, 0xFF5E,
                                                        0x2018, 0x201F, 0x3000, 0x3002};
} // namespace detail::pretokenize

/// \brief `[一-龥぀-ゟ゠-ヿ]+`: ideographs and kana (DeepSeek V3's second split).
inline constexpr ranges_pattern cjk_kana_pattern{detail::pretokenize::cjk_kana};
/// \brief `[一-龥ࠀ-一가-퟿]+` as DeepSeek LLM and Coder write it (U+0800 to U+9FA5, and Hangul).
inline constexpr ranges_pattern cjk_wide_pattern{detail::pretokenize::cjk_wide};
/// \brief DeepSeek LLM's `\s?[A-Za-zµÀ-ÖØ-öø-ƺ...]+`.
inline constexpr ranges_pattern deepseek_letters_pattern{detail::pretokenize::deepseek_letters, true};
/// \brief DeepSeek LLM's `\s?[!-/:-~！-／：-～‘-‟　-。]+`.
inline constexpr ranges_pattern deepseek_punctuation_pattern{detail::pretokenize::deepseek_punctuation, true};

/// \brief DeepSeek V3's third split (also Hunyuan Dense):
/// \verbatim [!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+ \endverbatim
/// \ingroup core
struct deepseek_v3_pattern
{
    /// \brief The match at a byte offset.
    /// \param text The text (a piece being split).
    /// \param at A byte offset into it.
    /// \return The length in bytes of the pattern's match starting at `at`, or 0 if it does not match there.
    constexpr std::size_t operator()(std::string_view text, std::size_t at) const noexcept
    {
        namespace d = detail::pretokenize;
        constexpr auto ascii_letter = [](const d::unit& u) { return u.len && ((u.cp | 0x20) >= 'a' && (u.cp | 0x20) <= 'z'); };
        constexpr auto ascii_punct = [](const d::unit& u) {
            return u.len && u.cp < 0x7F && u.cp > 0x20 && !((u.cp | 0x20) >= 'a' && (u.cp | 0x20) <= 'z') && !(u.cp >= '0' && u.cp <= '9');
        };
        if (ascii_punct(d::at(text, at)))
            if (const std::size_t n = d::run(text, at + 1, ascii_letter))
                return 1 + n;
        constexpr auto punct_symbol = [](const d::unit& u) { return u.len && (is_punctuation(u.cat) || is_symbol(u.cat)); };
        constexpr auto lead = [](const d::unit& u) { return u.len && !d::crlf(u) && !is_letter(u.cat) && !is_punctuation(u.cat) && !is_symbol(u.cat); };
        if (const std::size_t n = d::optional_then_run(text, at, lead, d::letter_or_mark))
            return n;
        if (const std::size_t n = d::space_run_tail(text, at, punct_symbol, d::crlf))
            return n;
        if (const std::size_t n = d::whitespace_to_newline(text, at))
            return n;
        return d::whitespace(text, at);
    }
};

/// \brief `\s?\p{L}+` (DeepSeek Coder's second split) or, with `Punctuation`, `\s?\p{P}+` (its third).
/// \tparam Punctuation `\p{P}` instead of `\p{L}`.
/// \ingroup core
template <bool Punctuation>
struct space_category_pattern
{
    /// \brief The match at a byte offset.
    /// \param text The text (a piece being split).
    /// \param at A byte offset into it.
    /// \return The length in bytes of the pattern's match starting at `at`, or 0 if it does not match there.
    constexpr std::size_t operator()(std::string_view text, std::size_t at) const noexcept
    {
        namespace d = detail::pretokenize;
        constexpr auto in = [](const d::unit& u) { return u.len && (Punctuation ? is_punctuation(u.cat) : is_letter(u.cat)); };
        return d::optional_then_run(text, at, d::space, in);
    }
};

/// \brief `\s+$`: trailing whitespace at the end of the piece (DeepSeek LLM's fourth split).
/// \ingroup core
struct trailing_space_pattern
{
    /// \brief The match at a byte offset.
    /// \param text The text (a piece being split).
    /// \param at A byte offset into it.
    /// \return The length in bytes of the pattern's match starting at `at`, or 0 if it does not match there.
    constexpr std::size_t operator()(std::string_view text, std::size_t at) const noexcept
    {
        const std::size_t n = detail::pretokenize::run(text, at, detail::pretokenize::space);
        return at + n == text.size() ? n : 0;
    }
};

/// \brief Falcon's first split, punctuation and the ASCII symbols `$ + < = > ^ ~ |` and the backtick:
/// \verbatim [\p{P}\$\+<=>\^~\|`]+ \endverbatim
/// \ingroup core
struct falcon_punctuation_pattern
{
    /// \brief The match at a byte offset.
    /// \param text The text (a piece being split).
    /// \param at A byte offset into it.
    /// \return The length in bytes of the pattern's match starting at `at`, or 0 if it does not match there.
    constexpr std::size_t operator()(std::string_view text, std::size_t at) const noexcept
    {
        namespace d = detail::pretokenize;
        constexpr auto in = [](const d::unit& u) {
            if (!u.len)
                return false;
            switch (u.cp) {
                case '$': case '+': case '<': case '=': case '>': case '^': case '~': case '|': case '`':
                    return true;
                default:
                    return is_punctuation(u.cat);
            }
        };
        return d::run(text, at, in);
    }
};

/// \brief Falcon's third split, `[0-9][0-9][0-9]`.
/// \ingroup core
struct three_ascii_digits_pattern
{
    /// \brief The match at a byte offset.
    /// \param text The text (a piece being split).
    /// \param at A byte offset into it.
    /// \return The length in bytes of the pattern's match starting at `at`, or 0 if it does not match there.
    constexpr std::size_t operator()(std::string_view text, std::size_t at) const noexcept
    {
        if (at + 3 > text.size())
            return 0;
        for (std::size_t i = at; i < at + 3; ++i)
            if (text[i] < '0' || text[i] > '9')
                return 0;
        return 3;
    }
};

/// \brief A regex split that keeps everything: each match is a piece, and so is the text between matches.
/// \tparam Pattern A callable `(std::string_view text, std::size_t at) -> std::size_t`, the match length at `at`.
/// \tparam Emit A callable `emit(std::string_view piece)`.
/// \param text The text (a piece of an earlier split, or the whole input).
/// \param pattern The pattern.
/// \param emit Receives the pieces in order; none is empty, and together they are `text`.
template <typename Pattern, typename Emit>
constexpr void split_isolated(std::string_view text, const Pattern& pattern, Emit&& emit)
{
    std::size_t gap = 0, at = 0;
    while (at < text.size()) {
        const std::size_t n = pattern(text, at);
        if (n == 0) {
            at += decode_utf8(text, at).length;
            continue;
        }
        if (gap < at)
            emit(text.substr(gap, at - gap));
        emit(text.substr(at, n));
        at += n;
        gap = at;
    }
    if (gap < text.size())
        emit(text.substr(gap));
}

/// \brief Several isolated splits in turn, each applied to every piece of the one before (Hugging Face's
/// `Sequence` of `Split`s, llama.cpp's list of expressions).
/// \param text The text.
/// \param emit Receives the final pieces.
/// \param first The first pattern.
/// \param rest The patterns after it.
template <typename Emit, typename First, typename... Rest>
constexpr void split_sequence(std::string_view text, Emit&& emit, const First& first, const Rest&... rest)
{
    if constexpr (sizeof...(Rest) == 0)
        split_isolated(text, first, emit);
    else
        split_isolated(text, first, [&](std::string_view piece) { split_sequence(piece, emit, rest...); });
}

/// \brief The pre-tokenizer families `pretokenize` runs.
/// \ingroup core
enum class pretokenizer
{
    gpt2,             ///< GPT-2, MPT, OLMo: `gpt2_pattern`.
    llama3,           ///< Llama 3, DBRX, Smaug, ChatGLM 4: `llama3_pattern`.
    qwen2,            ///< Qwen 2, StableLM 2: `qwen2_pattern`.
    qwen35,           ///< Qwen 3.5: `qwen35_pattern`.
    gpt4o,            ///< GPT-4o, o200k_base, MiniMax M2: `gpt4o_pattern`.
    deepseek_v3,      ///< DeepSeek V3 and R1, Hunyuan Dense: `\p{N}{1,3}`, then `cjk_kana_pattern`, then `deepseek_v3_pattern`.
    deepseek_llm,     ///< DeepSeek LLM: `[\r\n]`, letters, punctuation, `\s+$`, `cjk_wide_pattern`, `\p{N}+`.
    deepseek_coder,   ///< DeepSeek Coder: `[\r\n]`, `\s?\p{L}+`, `\s?\p{P}+`, `cjk_wide_pattern`, `\p{N}`.
    falcon,           ///< Falcon: punctuation, `gpt2_pattern`, `[0-9][0-9][0-9]`.
    starcoder,        ///< StarCoder, Refact, Command R, SmolLM: `\p{N}`, then `gpt2_pattern`.
    newlines,         ///< Gemma 4: `newline_pattern` (the model sees whole lines).
};

/// \brief Why a pre-tokenizer could not be chosen.
/// \ingroup core
enum class pretokenize_error
{
    unknown_name,   ///< The name is not one `pretokenizer_for` knows.
};

/// \brief Runs a family's whole split.
/// \tparam Emit A callable `emit(std::string_view piece)`.
/// \param kind The family.
/// \param text The text.
/// \param emit Receives the pieces in order.
template <typename Emit>
constexpr void pretokenize(pretokenizer kind, std::string_view text, Emit&& emit)
{
    switch (kind) {
        case pretokenizer::gpt2:
            return split_isolated(text, gpt2_pattern{}, emit);
        case pretokenizer::llama3:
            return split_isolated(text, llama3_pattern{}, emit);
        case pretokenizer::qwen2:
            return split_isolated(text, qwen2_pattern{}, emit);
        case pretokenizer::qwen35:
            return split_isolated(text, qwen35_pattern{}, emit);
        case pretokenizer::gpt4o:
            return split_isolated(text, gpt4o_pattern{}, emit);
        case pretokenizer::deepseek_v3:
            return split_sequence(text, emit, digits_pattern<3>{}, cjk_kana_pattern, deepseek_v3_pattern{});
        case pretokenizer::deepseek_llm:
            return split_sequence(text, emit, crlf_pattern{}, deepseek_letters_pattern, deepseek_punctuation_pattern,
                                  trailing_space_pattern{}, cjk_wide_pattern, digits_pattern<>{});
        case pretokenizer::deepseek_coder:
            return split_sequence(text, emit, crlf_pattern{}, space_category_pattern<false>{}, space_category_pattern<true>{},
                                  cjk_wide_pattern, digits_pattern<1>{});
        case pretokenizer::falcon:
            return split_sequence(text, emit, falcon_punctuation_pattern{}, gpt2_pattern{}, three_ascii_digits_pattern{});
        case pretokenizer::starcoder:
            return split_sequence(text, emit, digits_pattern<1>{}, gpt2_pattern{});
        case pretokenizer::newlines:
            return split_isolated(text, newline_pattern{}, emit);
    }
}

/// \brief The family for a GGUF `tokenizer.ggml.pre` name (llama.cpp's names).
/// \param name The name, e.g. "llama-bpe", "qwen2", "deepseek-v3", "gpt-4o".
/// \return The family, or `pretokenize_error::unknown_name`.
constexpr result<pretokenizer, pretokenize_error> pretokenizer_for(std::string_view name) noexcept
{
    struct entry
    {
        std::string_view name;
        pretokenizer kind;
    };
    constexpr entry table[] = {
        {"gpt-2", pretokenizer::gpt2},               {"mpt", pretokenizer::gpt2},
        {"olmo", pretokenizer::gpt2},                {"jais", pretokenizer::gpt2},
        {"trillion", pretokenizer::gpt2},            {"granite-docling", pretokenizer::gpt2},
        {"roberta-bpe", pretokenizer::gpt2},         {"jina-v1-en", pretokenizer::gpt2},
        {"jina-v2-code", pretokenizer::gpt2},        {"llama-bpe", pretokenizer::llama3},
        {"llama3", pretokenizer::llama3},            {"dbrx", pretokenizer::llama3},
        {"smaug-bpe", pretokenizer::llama3},         {"chatglm-bpe", pretokenizer::llama3},
        {"qwen2", pretokenizer::qwen2},              {"stablelm2", pretokenizer::qwen2},
        {"hunyuan", pretokenizer::qwen2},            {"qwen35", pretokenizer::qwen35},
        {"gpt-4o", pretokenizer::gpt4o},             {"minimax-m2", pretokenizer::gpt4o},
        {"deepseek-v3", pretokenizer::deepseek_v3},  {"hunyuan-dense", pretokenizer::deepseek_v3},
        {"deepseek-llm", pretokenizer::deepseek_llm}, {"deepseek-coder", pretokenizer::deepseek_coder},
        {"falcon", pretokenizer::falcon},            {"starcoder", pretokenizer::starcoder},
        {"refact", pretokenizer::starcoder},         {"command-r", pretokenizer::starcoder},
        {"smollm", pretokenizer::starcoder},         {"codeshell", pretokenizer::starcoder},
        {"exaone", pretokenizer::starcoder},         {"minerva-7b", pretokenizer::starcoder},
        {"gemma4", pretokenizer::newlines},
    };
    for (const entry& e : table)
        if (e.name == name)
            return ok(e.kind);
    return err(pretokenize_error::unknown_name);
}

/// \brief Cuts the added tokens (special and user-defined) out of raw text, leftmost and longest first.
/// \tparam OnText A callable `on_text(std::string_view)` for the text between added tokens.
/// \tparam OnToken A callable `on_token(std::uint32_t id)` for each added token.
/// \param text The raw text.
/// \param added A trie of the added tokens only (`build_token_index` with a filter).
/// \param on_text Receives each run of ordinary text, never empty.
/// \param on_token Receives each added token's id.
template <typename OnText, typename OnToken>
constexpr void split_added_tokens(std::string_view text, const token_trie& added, OnText&& on_text, OnToken&& on_token)
{
    std::size_t gap = 0, at = 0;
    while (at < text.size()) {
        std::size_t n = 0;
        const std::uint32_t id = added.order.empty() ? no_token : added.longest_prefix(text.substr(at), n);
        if (id == no_token || n == 0) {
            ++at;
            continue;
        }
        if (gap < at)
            on_text(text.substr(gap, at - gap));
        on_token(id);
        at += n;
        gap = at;
    }
    if (gap < text.size())
        on_text(text.substr(gap));
}

} // namespace snicholls::ml

#endif // snicholls_core_ml_pretokenize_hpp
