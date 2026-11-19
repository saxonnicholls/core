//
//  grammar.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::ml` grammar-constrained decoding: a GBNF compiler (at run time from a string, at compile time from a `fixed_string` into fixed-size tables), the pushdown matcher and the token mask over a vocabulary.
/// \ingroup core
///
/// **Overview.** GBNF is llama.cpp's grammar language (`grammars/README.md`): rules `name ::= ...`, alternatives
/// `|`, sequences, literals `"..."` with the escapes `\n \t \r \\ \" \[ \] \- \xHH \uHHHH \UHHHHHHHH`, character
/// classes `[a-z0-9_]` and `[^...]`, `.` (any character), grouping `( )`, the repetitions `*`, `+`, `?`, `{m}`,
/// `{m,}` and `{m,n}`, comments `#`, and token terminals `<[id]>` / `!<[id]>`. A grammar constrains generation:
/// before each token is sampled, the tokens that cannot continue a sentence of the grammar are masked out.
///
/// - `parse_grammar(text, root)`: parses GBNF into a `grammar`, a flat table of `grammar_element`s (llama.cpp's
///   `llama_gretype` encoding, value for value) and each rule's first element, or returns a `grammar_error` with the
///   byte position and a message. Repetitions are rewritten into helper rules exactly as llama.cpp's
///   `handle_repetitions` does, so the rules, their numbering and their names (`root_1`, ...) are llama.cpp's.
///   Undefined rules, a missing root and left recursion are errors.
/// - `compile_grammar<"...">()`: the same parse in a `consteval` function, run twice (once to size, once to fill), into
///   a `fixed_grammar<Elements, Rules>` of `std::array`s: a grammar written in the program is a compile-time automaton
///   table, and a malformed one is a compile error.
/// - `grammar_matcher`: the set of pushdown stacks over a `grammar_view` (either form), each a list of element
///   indices whose top is a terminal (llama.cpp's `llama_grammar_advance_stack`). `accept_code_point`,
///   `accept_string` and `accept_token` advance it (UTF-8 decoded with a partial sequence carried across tokens, as
///   llama.cpp's `decode_utf8` / `llama_partial_utf8`); `complete()` says whether the text so far is a sentence.
/// - The token mask: `allows(token, piece)` for one token, `token_mask` over a vocabulary into `bool`s, and
///   `token_bitmask` into 32-bit words (bit `t % 32` of word `t / 32`), which `apply_token_bitmask` turns into
///   `-infinity` logits. An end-of-generation token is allowed exactly when some stack is empty (the grammar is
///   complete); an empty piece never is.
/// - `grammar_accepts(g, text)`: whether a whole string is a sentence; `constexpr`, so a small grammar can be checked
///   in a `static_assert`.
///
/// When not to use it: for JSON that must follow a schema, write the schema and convert it (`ml/json_schema.hpp`).
///
/// \snippet ml/grammar_demo.cpp grammar
///
/// \snippet ml/grammar_demo.cpp grammar_mask
///
/// **Relationships.** `ml/json_schema.hpp` writes GBNF from a JSON schema; `ml/sampling.hpp` samples from the masked
/// logits; `result.hpp` carries the parse error; `fixed_string.hpp` carries a compile-time grammar. Group: \ref core.
///
/// **Implementations.** One parser for both forms; the matcher reads either through `grammar_view`. Semantics
/// follow llama.cpp at commit 39039aa: `src/llama-grammar.cpp` (`llama_grammar_parser::parse_sequence`,
/// `parse_alternates`, `parse_rule`, `parse`, `parse_char`, `parse_space`, `decode_utf8`,
/// `llama_grammar_match_char`, `llama_grammar_match_partial_char`, `llama_grammar_advance_stack`,
/// `llama_grammar_detect_left_recursion`, `llama_grammar_accept_token`, `llama_grammar_accept_str`,
/// `llama_grammar_apply_impl`, `llama_grammar_reject_candidates`), checked against `tests/test-grammar-parser.cpp` and
/// `tests/test-grammar-integration.cpp` (`tests/ml/grammar_test.cpp`).
///
/// **Guarantees.** Everything is `constexpr`; `compile_grammar` is `consteval`. A `grammar_matcher` holds spans into the
/// grammar, which must outlive it. The token mask agrees with `accept_token`: a token the mask allows can be accepted.
/// Complexity: parsing is linear in the text plus the rules the repetitions generate (at most 2,000 per repetition, as
/// llama.cpp); the mask is O(vocabulary · piece length · stacks).
///
/// **Errors.** `parse_grammar` returns `grammar_error{position, message}`: a syntax error at the byte where it was
/// found; an undefined rule at its first reference; a missing root at the end of the text; left recursion at the first
/// mention of the rule found to recurse. `compile_grammar` fails to compile on any of them. Preconditions, asserted:
/// mask spans of the vocabulary's size.
///
/// **Performance.** Not benchmarked: the mask is a per-token simulation of the stacks, which a serving stack runs once
/// per generated token on the CPU while the device computes the next logits; llama.cpp's own reject pass groups the
/// candidates by code point instead, which is faster on large vocabularies but gives the same mask.
///
/// **Tests and demos.** `tests/ml/grammar_test.cpp`, `demos/ml/grammar_demo.cpp`.
///
/// **Dependencies.** `<algorithm>`, `<array>`, `<cassert>`, `<concepts>`, `<cstddef>`, `<cstdint>`, `<limits>`,
/// `<span>`, `<string>`, `<string_view>`, `<type_traits>`, `<utility>`, `<vector>`, `snicholls/core/fixed_string.hpp`,
/// `snicholls/core/result.hpp` (which includes `<format>`: heavier than it looks).
///
/// **Known issues.** Named token terminals (`<think>`) need a tokenizer and are an error: write `<[id]>`. llama.cpp
/// left-factors rules after parsing (fewer parallel stacks, the same language); this header does not, so its stacks
/// can be more numerous on grammars with many alternatives sharing a prefix. `{m,n}` with `n < m` is an error here
/// (llama.cpp loops). At a token terminal reached in the middle of a piece, llama.cpp's reject pass compares the
/// token's id while its `accept_token` drops the stack; the mask here follows `accept_token`, so the two never
/// disagree. Compile-time grammars are limited by the compiler's evaluation steps: a grammar the size of
/// `json.gbnf` compiles, matching long strings in a `static_assert` may not.
///
/// **Status.** Experimental, new in 0.17.0.

#ifndef snicholls_core_ml_grammar_hpp
#define snicholls_core_ml_grammar_hpp

#include <snicholls/core/fixed_string.hpp>
#include <snicholls/core/result.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace snicholls::ml
{

/// \brief The kind of a grammar element, with llama.cpp's `llama_gretype` values.
/// \ingroup core
enum class grammar_element_type : std::uint8_t
{
    end = 0,               ///< The end of a rule.
    alt = 1,               ///< The start of another alternative of the rule.
    rule_ref = 2,          ///< A reference to rule `value`.
    chr = 3,               ///< The code point `value`, or the first of a class.
    chr_not = 4,           ///< The first of an inverted class `[^...]`.
    chr_range_upper = 5,   ///< Makes the preceding character a range up to `value`, inclusive.
    chr_alt = 6,           ///< Another character of the class begun before it.
    chr_any = 7,           ///< Any code point (`.`).
    token = 8,             ///< The token with id `value` (`<[id]>`).
    token_not = 9,         ///< Any token but `value` (`!<[id]>`).
};

/// \brief One element of a compiled grammar: a type and a code point, rule id or token id.
/// \ingroup core
struct grammar_element
{
    grammar_element_type type = grammar_element_type::end;   ///< The kind.
    std::uint32_t value = 0;                                  ///< The code point, rule id or token id; 0 otherwise.

    /// \brief Element-wise equality.
    /// \return Whether type and value are equal.
    friend constexpr bool operator==(const grammar_element&, const grammar_element&) noexcept = default;
};

/// \brief Why a grammar did not parse: the byte position in the text and a message.
/// \ingroup core
struct grammar_error
{
    std::size_t position = 0;     ///< The byte offset where the error was found.
    std::string_view message{};   ///< A fixed message (a string literal).

    /// \brief Equality of position and message.
    /// \return Whether both are equal.
    friend constexpr bool operator==(const grammar_error&, const grammar_error&) noexcept = default;
};

/// \brief A compiled grammar of either form, as the matcher reads it: every rule's elements end to end, where each
/// rule starts, and the root rule.
/// \ingroup core
struct grammar_view
{
    std::span<const grammar_element> elements{};   ///< Every rule's elements, each ending with `end`.
    std::span<const std::uint32_t> rules{};        ///< The index in `elements` of each rule's first element, by rule id.
    std::uint32_t root = 0;                        ///< The id of the start rule.

    /// \brief The elements of rule `id`, its `end` included.
    /// \param id A rule id, below `rules.size()`.
    /// \return A span of the rule.
    [[nodiscard]] constexpr std::span<const grammar_element> rule(std::uint32_t id) const
    {
        assert(id < rules.size());
        std::size_t last = rules[id];
        while (elements[last].type != grammar_element_type::end)
            ++last;
        return elements.subspan(rules[id], last + 1 - rules[id]);
    }
};

/// \brief A grammar parsed at run time (`parse_grammar`): the flat element table, the rules' starts and their names.
/// \ingroup core
struct grammar
{
    std::vector<grammar_element> elements{};   ///< Every rule's elements end to end, by rule id.
    std::vector<std::uint32_t> rules{};        ///< The first element of each rule.
    std::vector<std::string> names{};          ///< Each rule's name by id; generated rules are named `<rule>_<id>`.
    std::uint32_t root = 0;                    ///< The id of the start rule.

    /// \brief The view the matcher reads; valid while this grammar lives.
    /// \return The view.
    [[nodiscard]] constexpr grammar_view view() const noexcept { return {elements, rules, root}; }

    /// \brief The id of a rule by name.
    /// \param name The rule's name.
    /// \return Its id, or the number of rules when there is none.
    [[nodiscard]] constexpr std::uint32_t find(std::string_view name) const
    {
        for (std::size_t i = 0; i < names.size(); ++i)
            if (names[i] == name)
                return static_cast<std::uint32_t>(i);
        return static_cast<std::uint32_t>(names.size());
    }
};

/// \brief A grammar compiled at compile time (`compile_grammar`): the same table in `std::array`s.
/// \ingroup core
/// \tparam Elements The elements of every rule together.
/// \tparam Rules The rules.
template <std::size_t Elements, std::size_t Rules>
struct fixed_grammar
{
    std::array<grammar_element, Elements> elements{};   ///< Every rule's elements end to end, by rule id.
    std::array<std::uint32_t, Rules> rules{};           ///< The first element of each rule.
    std::uint32_t root = 0;                             ///< The id of the start rule.

    /// \brief The view the matcher reads; valid while this object lives.
    /// \return The view.
    [[nodiscard]] constexpr grammar_view view() const noexcept { return {elements, rules, root}; }
};

namespace detail::grammar
{
    inline constexpr std::size_t npos = static_cast<std::size_t>(-1);
    inline constexpr std::uint64_t no_max = std::numeric_limits<std::uint64_t>::max();
    inline constexpr std::uint64_t max_repetition = 2000;   // llama.cpp's MAX_REPETITION_THRESHOLD
    inline constexpr std::uint32_t recursion_limit = 256;   // llama.cpp's GRAMMAR_RECURSIVE_LIMIT

    constexpr bool is_digit(char c) noexcept { return '0' <= c && c <= '9'; }
    constexpr bool is_word(char c) noexcept { return ('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z') || c == '-' || is_digit(c); }

    constexpr std::string decimal(std::uint64_t v)
    {
        std::string s;
        do {
            s.insert(s.begin(), static_cast<char>('0' + v % 10));
            v /= 10;
        } while (v != 0);
        return s;
    }

    constexpr bool end_of_sequence(grammar_element e) noexcept
    {
        return e.type == grammar_element_type::end || e.type == grammar_element_type::alt;
    }

    struct parser
    {
        std::string_view src;
        std::vector<std::string> names{};         // symbol names in the order first seen (llama.cpp's symbol_ids keys)
        std::vector<std::uint32_t> ids{};         // and their ids
        std::vector<std::size_t> first_seen{};    // the byte where each symbol first appeared
        std::vector<std::vector<grammar_element>> rules{};
        grammar_error error{};
        bool failed = false;

        constexpr char at(std::size_t i) const noexcept { return i < src.size() ? src[i] : '\0'; }

        constexpr std::size_t fail(std::size_t pos, std::string_view message)
        {
            if (!failed) {
                failed = true;
                error = {pos, message};
            }
            return npos;
        }

        constexpr std::uint32_t symbol_id(std::string_view name, std::size_t pos)
        {
            for (std::size_t i = 0; i < names.size(); ++i)
                if (names[i] == name)
                    return ids[i];
            const auto id = static_cast<std::uint32_t>(names.size());
            names.emplace_back(name);
            ids.push_back(id);
            first_seen.push_back(pos);
            return id;
        }

        // llama.cpp's generate_symbol_id: base + '_' + the next id, assigned into the map (an existing name is overwritten)
        constexpr std::uint32_t generate_symbol_id(const std::string& base, std::size_t pos)
        {
            const auto id = static_cast<std::uint32_t>(names.size());
            std::string name = base;
            name += '_';
            name += decimal(id);
            for (std::size_t i = 0; i < names.size(); ++i)
                if (names[i] == name) {
                    ids[i] = id;
                    return id;
                }
            names.push_back(std::move(name));
            ids.push_back(id);
            first_seen.push_back(pos);
            return id;
        }

        constexpr void add_rule(std::uint32_t id, const std::vector<grammar_element>& rule)
        {
            if (rules.size() <= id)
                rules.resize(id + 1);
            rules[id] = rule;
        }

        constexpr std::size_t parse_space(std::size_t pos, bool newline_ok) const
        {
            while (at(pos) == ' ' || at(pos) == '\t' || at(pos) == '#' || (newline_ok && (at(pos) == '\r' || at(pos) == '\n'))) {
                if (at(pos) == '#') {
                    while (at(pos) != '\0' && at(pos) != '\r' && at(pos) != '\n')
                        ++pos;
                } else {
                    ++pos;
                }
            }
            return pos;
        }

        constexpr std::size_t parse_name(std::size_t pos)
        {
            std::size_t end = pos;
            while (is_word(at(end)))
                ++end;
            return end == pos ? fail(pos, "expecting a name") : end;
        }

        constexpr std::size_t parse_int(std::size_t pos)
        {
            std::size_t end = pos;
            while (is_digit(at(end)))
                ++end;
            return end == pos ? fail(pos, "expecting an integer") : end;
        }

        constexpr std::uint64_t int_value(std::size_t from, std::size_t to) const
        {
            std::uint64_t v = 0;
            for (std::size_t i = from; i < to; ++i)
                v = v > (no_max - 9) / 10 ? no_max - 1 : v * 10 + static_cast<std::uint64_t>(src[i] - '0');   // saturates
            return v;
        }

        struct code_point
        {
            std::uint32_t value;
            std::size_t next;
        };

        // llama.cpp's decode_utf8(const char *): assumes valid UTF-8, stops at the end
        constexpr code_point decode(std::size_t pos) const
        {
            constexpr int lookup[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4};
            const auto first = static_cast<std::uint8_t>(at(pos));
            const int len = lookup[first >> 4];
            const auto mask = static_cast<std::uint8_t>((1u << (8 - len)) - 1);
            std::uint32_t value = first & mask;
            const std::size_t end = pos + static_cast<std::size_t>(len);
            std::size_t p = pos + 1;
            for (; p < end && at(p) != '\0'; ++p)
                value = (value << 6) + (static_cast<std::uint8_t>(at(p)) & 0x3fu);
            return {value, p};
        }

        constexpr code_point parse_hex(std::size_t pos, std::size_t size)
        {
            std::uint32_t value = 0;
            std::size_t p = pos;
            for (; p < pos + size && at(p) != '\0'; ++p) {
                const char c = at(p);
                value <<= 4;
                if ('a' <= c && c <= 'f')
                    value += static_cast<std::uint32_t>(c - 'a' + 10);
                else if ('A' <= c && c <= 'F')
                    value += static_cast<std::uint32_t>(c - 'A' + 10);
                else if ('0' <= c && c <= '9')
                    value += static_cast<std::uint32_t>(c - '0');
                else
                    break;
            }
            if (p != pos + size)
                return {0, fail(pos, "expecting hexadecimal digits")};
            return {value, p};
        }

        constexpr code_point parse_char(std::size_t pos)
        {
            if (at(pos) == '\\') {
                switch (at(pos + 1)) {
                case 'x': return parse_hex(pos + 2, 2);
                case 'u': return parse_hex(pos + 2, 4);
                case 'U': return parse_hex(pos + 2, 8);
                case 't': return {'\t', pos + 2};
                case 'r': return {'\r', pos + 2};
                case 'n': return {'\n', pos + 2};
                case '\\':
                case '"':
                case '[':
                case ']':
                case '-': return {static_cast<std::uint32_t>(at(pos + 1)), pos + 2};
                default: return {0, fail(pos, "unknown escape")};
                }
            }
            if (at(pos) != '\0')
                return decode(pos);
            return {0, fail(pos, "unexpected end of input")};
        }

        constexpr code_point parse_token(std::size_t pos)
        {
            if (at(pos) != '<')
                return {0, fail(pos, "expecting '<'")};
            ++pos;
            if (at(pos) != '[')
                return {0, fail(pos, "named tokens need a vocabulary: write <[id]>")};
            ++pos;
            const std::size_t end = parse_int(pos);
            if (end == npos)
                return {0, npos};
            const std::uint64_t id = int_value(pos, end);
            if (id > std::numeric_limits<std::uint32_t>::max())
                return {0, fail(pos, "token id is too big")};
            if (at(end) != ']')
                return {0, fail(end, "expecting ']'")};
            if (at(end + 1) != '>')
                return {0, fail(end + 1, "expecting '>'")};
            return {static_cast<std::uint32_t>(id), end + 2};
        }

        constexpr std::size_t parse_alternates(std::size_t pos, const std::string& rule_name, std::uint32_t rule_id, bool nested,
                                               std::uint32_t depth)
        {
            std::vector<grammar_element> rule;
            pos = parse_sequence(pos, rule_name, rule, nested, depth);
            while (pos != npos && at(pos) == '|') {
                rule.push_back({grammar_element_type::alt, 0});
                pos = parse_sequence(parse_space(pos + 1, true), rule_name, rule, nested, depth);
            }
            if (pos == npos)
                return npos;
            rule.push_back({grammar_element_type::end, 0});
            add_rule(rule_id, rule);
            return pos;
        }

        // S{m,n} -> S (m times) S'(n-m), S'(x) ::= S S'(x-1) |, S'(1) ::= S |; S{m,} -> S (m times) S', S' ::= S S' |
        constexpr bool repeat(std::vector<grammar_element>& rule, std::size_t last_sym_start, std::uint64_t& n_prev_rules,
                              const std::string& rule_name, std::uint64_t min_times, std::uint64_t max_times, std::size_t pos)
        {
            const bool unbounded = max_times == no_max;
            if (last_sym_start == rule.size()) {
                fail(pos, "expecting an item before */+/?/{");
                return false;
            }
            if (!unbounded && max_times < min_times) {
                fail(pos, "a repetition's maximum is below its minimum");
                return false;
            }
            const std::vector<grammar_element> prev(rule.begin() + static_cast<std::ptrdiff_t>(last_sym_start), rule.end());
            std::uint64_t total = 1;
            if (!unbounded && max_times > 0)
                total = max_times;
            else if (min_times > 0)
                total = min_times;
            if (n_prev_rules * total > max_repetition) {
                fail(pos, "the repetition generates more than 2000 rules");
                return false;
            }
            if (min_times == 0)
                rule.resize(last_sym_start);
            else
                for (std::uint64_t i = 1; i < min_times; ++i)
                    rule.insert(rule.end(), prev.begin(), prev.end());
            std::uint32_t last_rec = 0;
            const std::uint64_t n_opt = unbounded ? 1 : max_times - min_times;
            std::vector<grammar_element> rec = prev;
            for (std::uint64_t i = 0; i < n_opt; ++i) {
                rec.resize(prev.size());
                const std::uint32_t id = generate_symbol_id(rule_name, pos);
                if (i > 0 || unbounded)
                    rec.push_back({grammar_element_type::rule_ref, unbounded ? id : last_rec});
                rec.push_back({grammar_element_type::alt, 0});
                rec.push_back({grammar_element_type::end, 0});
                add_rule(id, rec);
                last_rec = id;
            }
            if (n_opt > 0)
                rule.push_back({grammar_element_type::rule_ref, last_rec});
            n_prev_rules *= total;
            return true;
        }

        constexpr std::size_t parse_sequence(std::size_t pos, const std::string& rule_name, std::vector<grammar_element>& rule, bool nested,
                                             std::uint32_t depth)
        {
            std::size_t last_sym_start = rule.size();
            std::uint64_t n_prev_rules = 1;
            while (pos != npos && at(pos) != '\0') {
                const char c = at(pos);
                if (c == '"') {   // a literal
                    ++pos;
                    last_sym_start = rule.size();
                    n_prev_rules = 1;
                    while (at(pos) != '"') {
                        if (at(pos) == '\0')
                            return fail(pos, "unexpected end of input");
                        const auto ch = parse_char(pos);
                        if (ch.next == npos)
                            return npos;
                        pos = ch.next;
                        rule.push_back({grammar_element_type::chr, ch.value});
                    }
                    pos = parse_space(pos + 1, nested);
                } else if (c == '[') {   // a character class
                    ++pos;
                    auto start_type = grammar_element_type::chr;
                    if (at(pos) == '^') {
                        ++pos;
                        start_type = grammar_element_type::chr_not;
                    }
                    last_sym_start = rule.size();
                    n_prev_rules = 1;
                    while (at(pos) != ']') {
                        if (at(pos) == '\0')
                            return fail(pos, "unexpected end of input");
                        const auto ch = parse_char(pos);
                        if (ch.next == npos)
                            return npos;
                        pos = ch.next;
                        rule.push_back({last_sym_start < rule.size() ? grammar_element_type::chr_alt : start_type, ch.value});
                        if (at(pos) == '-' && at(pos + 1) != ']') {
                            if (at(pos + 1) == '\0')
                                return fail(pos + 1, "unexpected end of input");
                            const auto upper = parse_char(pos + 1);
                            if (upper.next == npos)
                                return npos;
                            pos = upper.next;
                            rule.push_back({grammar_element_type::chr_range_upper, upper.value});
                        }
                    }
                    pos = parse_space(pos + 1, nested);
                } else if (c == '<' || c == '!') {   // a token
                    auto type = grammar_element_type::token;
                    if (c == '!') {
                        type = grammar_element_type::token_not;
                        ++pos;
                    }
                    const auto t = parse_token(pos);
                    if (t.next == npos)
                        return npos;
                    last_sym_start = rule.size();
                    n_prev_rules = 1;
                    rule.push_back({type, t.value});
                    pos = parse_space(t.next, nested);
                } else if (is_word(c)) {   // a rule reference
                    const std::size_t name_end = parse_name(pos);
                    const std::uint32_t ref = symbol_id(src.substr(pos, name_end - pos), pos);
                    pos = parse_space(name_end, nested);
                    last_sym_start = rule.size();
                    n_prev_rules = 1;
                    rule.push_back({grammar_element_type::rule_ref, ref});
                } else if (c == '(') {   // a group, as a synthesised rule
                    if (depth >= recursion_limit)
                        return fail(pos, "grammar nesting is deeper than 256");
                    const std::size_t open = pos;
                    pos = parse_space(pos + 1, true);
                    const std::size_t n_before = names.size();
                    const std::uint32_t sub = generate_symbol_id(rule_name, open);
                    pos = parse_alternates(pos, rule_name, sub, true, depth + 1);
                    if (pos == npos)
                        return npos;
                    n_prev_rules = std::max<std::uint64_t>(1, names.size() - n_before);
                    last_sym_start = rule.size();
                    rule.push_back({grammar_element_type::rule_ref, sub});
                    if (at(pos) != ')')
                        return fail(pos, "expecting ')'");
                    pos = parse_space(pos + 1, nested);
                } else if (c == '.') {   // any character
                    last_sym_start = rule.size();
                    n_prev_rules = 1;
                    rule.push_back({grammar_element_type::chr_any, 0});
                    pos = parse_space(pos + 1, nested);
                } else if (c == '*' || c == '+' || c == '?') {
                    const std::size_t at_op = pos;
                    pos = parse_space(pos + 1, nested);
                    const std::uint64_t lo = c == '+' ? 1 : 0, hi = c == '?' ? 1 : no_max;
                    if (!repeat(rule, last_sym_start, n_prev_rules, rule_name, lo, hi, at_op))
                        return npos;
                } else if (c == '{') {
                    const std::size_t at_op = pos;
                    pos = parse_space(pos + 1, nested);
                    if (!is_digit(at(pos)))
                        return fail(pos, "expecting an integer");
                    std::size_t end = parse_int(pos);
                    const std::uint64_t min_times = int_value(pos, end);
                    pos = parse_space(end, nested);
                    std::uint64_t max_times = no_max;
                    if (at(pos) == '}') {
                        max_times = min_times;
                        pos = parse_space(pos + 1, nested);
                    } else if (at(pos) == ',') {
                        pos = parse_space(pos + 1, nested);
                        if (is_digit(at(pos))) {
                            end = parse_int(pos);
                            max_times = int_value(pos, end);
                            pos = parse_space(end, nested);
                        }
                        if (at(pos) != '}')
                            return fail(pos, "expecting '}'");
                        pos = parse_space(pos + 1, nested);
                    } else {
                        return fail(pos, "expecting ','");
                    }
                    if (min_times > max_repetition)
                        return fail(at_op, "the repetition count exceeds 2000");
                    if (max_times != no_max && max_times > max_repetition)
                        max_times = no_max;
                    if (!repeat(rule, last_sym_start, n_prev_rules, rule_name, min_times, max_times, at_op))
                        return npos;
                } else {
                    break;
                }
            }
            return pos;
        }

        constexpr std::size_t parse_rule(std::size_t pos)
        {
            const std::size_t name_end = parse_name(pos);
            if (name_end == npos)
                return npos;
            const std::string name(src.substr(pos, name_end - pos));
            const std::uint32_t id = symbol_id(name, pos);
            pos = parse_space(name_end, false);
            if (!(at(pos) == ':' && at(pos + 1) == ':' && at(pos + 2) == '='))
                return fail(pos, "expecting ::=");
            pos = parse_alternates(parse_space(pos + 3, true), name, id, false, 0);
            if (pos == npos)
                return npos;
            if (at(pos) == '\r')
                pos += at(pos + 1) == '\n' ? 2 : 1;
            else if (at(pos) == '\n')
                ++pos;
            else if (at(pos) != '\0')
                return fail(pos, "expecting a newline or the end");
            return parse_space(pos, true);
        }

        constexpr std::size_t symbol_position(std::uint32_t id) const
        {
            for (std::size_t i = 0; i < ids.size(); ++i)
                if (ids[i] == id)
                    return first_seen[i];
            return 0;
        }

        // llama_grammar_detect_left_recursion; `state` is 0 unvisited, 1 in progress, 2 visited; may_be_empty per rule
        constexpr int left_recursive(std::size_t index, std::vector<char>& state, std::vector<char>& may_be_empty, std::uint32_t depth) const
        {
            if (state[index] == 1)
                return 1;
            if (depth > recursion_limit)
                return 2;
            state[index] = 1;
            const auto& rule = rules[index];
            bool at_start = true;
            for (const auto& e : rule) {
                if (end_of_sequence(e)) {
                    if (at_start) {
                        may_be_empty[index] = 1;
                        break;
                    }
                    at_start = true;
                } else {
                    at_start = false;
                }
            }
            bool recurse = true;
            for (const auto& e : rule) {
                if (e.type == grammar_element_type::rule_ref && recurse) {
                    const int r = left_recursive(e.value, state, may_be_empty, depth + 1);
                    if (r != 0)
                        return r;
                    if (!may_be_empty[e.value])
                        recurse = false;
                } else {
                    recurse = end_of_sequence(e);
                }
            }
            state[index] = 2;
            return 0;
        }

        constexpr result<snicholls::ml::grammar, grammar_error> parse(std::string_view root)
        {
            std::size_t pos = parse_space(0, true);
            while (pos != npos && at(pos) != '\0')
                pos = parse_rule(pos);
            if (pos == npos)
                return err(error);
            if (rules.empty())
                return err(grammar_error{pos, "the grammar has no rules"});
            for (std::size_t i = 0; i < rules.size(); ++i) {
                if (rules[i].empty())
                    return err(grammar_error{symbol_position(static_cast<std::uint32_t>(i)), "undefined rule"});
                for (const auto& e : rules[i])
                    if (e.type == grammar_element_type::rule_ref && (e.value >= rules.size() || rules[e.value].empty()))
                        return err(grammar_error{symbol_position(e.value), "undefined rule"});
            }
            std::uint32_t root_id = 0;
            bool has_root = false;
            for (std::size_t i = 0; i < names.size(); ++i)
                if (names[i] == root) {
                    root_id = ids[i];
                    has_root = true;
                }
            if (!has_root)
                return err(grammar_error{src.size(), "the grammar has no root rule"});
            std::vector<char> state(rules.size(), 0), may_be_empty(rules.size(), 0);
            for (std::size_t i = 0; i < rules.size(); ++i) {
                if (state[i] == 2)
                    continue;
                const int r = left_recursive(i, state, may_be_empty, 0);
                if (r == 1)
                    return err(grammar_error{symbol_position(static_cast<std::uint32_t>(i)), "left recursion"});
                if (r == 2)
                    return err(grammar_error{symbol_position(static_cast<std::uint32_t>(i)), "rules are chained more than 256 deep"});
            }
            snicholls::ml::grammar g;
            g.root = root_id;
            g.names.resize(rules.size());
            for (std::size_t i = 0; i < names.size(); ++i)
                if (ids[i] < rules.size())
                    g.names[ids[i]] = names[i];
            for (const auto& rule : rules) {
                g.rules.push_back(static_cast<std::uint32_t>(g.elements.size()));
                g.elements.insert(g.elements.end(), rule.begin(), rule.end());
            }
            return ok(std::move(g));
        }
    };
} // namespace detail::grammar

/// \brief Parses GBNF text into a grammar, with llama.cpp's rules, numbering and names.
/// \param text The grammar.
/// \param root The start rule's name.
/// \return The grammar, or where and why it does not parse.
[[nodiscard]] constexpr result<grammar, grammar_error> parse_grammar(std::string_view text, std::string_view root = "root")
{
    detail::grammar::parser p{text};
    return p.parse(root);
}

namespace detail::grammar
{
    struct shape
    {
        bool ok = false;
        std::size_t elements = 0;
        std::size_t rules = 0;
    };

    constexpr shape measure(std::string_view text, std::string_view root)
    {
        const auto g = parse_grammar(text, root);
        if (g.is_err())
            return {};
        return {true, (*g).elements.size(), (*g).rules.size()};
    }
} // namespace detail::grammar

/// \brief Compiles a GBNF grammar written in the program into fixed-size tables, at compile time: a malformed
/// grammar does not compile.
/// \tparam Text The grammar, a string literal.
/// \tparam Root The start rule's name.
/// \return The compiled grammar, element for element the one `parse_grammar` gives.
template <basic_fixed_string Text, basic_fixed_string Root = basic_fixed_string("root")>
    requires std::same_as<typename decltype(Text)::value_type, char> && std::same_as<typename decltype(Root)::value_type, char>
consteval auto compile_grammar()
{
    constexpr auto s = detail::grammar::measure(Text.view(), Root.view());
    static_assert(s.ok, "snicholls::ml::compile_grammar: the grammar does not parse; parse_grammar gives the position and the reason");
    fixed_grammar<s.elements, s.rules> out{};
    if (s.ok) {
        const auto parsed = parse_grammar(Text.view(), Root.view());
        const auto& g = *parsed;
        for (std::size_t i = 0; i < s.elements; ++i)
            out.elements[i] = g.elements[i];
        for (std::size_t i = 0; i < s.rules; ++i)
            out.rules[i] = g.rules[i];
        out.root = g.root;
    }
    return out;
}

/// \brief A UTF-8 sequence cut by the end of a token: its bits so far and the bytes still to come (llama.cpp's
/// `llama_partial_utf8`).
/// \ingroup core
struct grammar_utf8_state
{
    std::uint32_t value = 0;   ///< The code point's bits so far, unshifted.
    int remaining = 0;         ///< Continuation bytes still to come; -1 after an invalid sequence.

    /// \brief Equality.
    /// \return Whether both fields are equal.
    friend constexpr bool operator==(const grammar_utf8_state&, const grammar_utf8_state&) noexcept = default;
};

/// \brief Decodes UTF-8 from a partial state as llama.cpp's `decode_utf8(std::string, llama_partial_utf8)`: the
/// whole code points, and the state an unfinished last sequence leaves.
/// \param text The bytes.
/// \param start The state the previous text left.
/// \return The code points and the new state; an invalid sequence gives no code points and `remaining == -1`.
[[nodiscard]] constexpr std::pair<std::vector<std::uint32_t>, grammar_utf8_state> decode_utf8(std::string_view text, grammar_utf8_state start)
{
    constexpr int lookup[] = {1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 3, 4};
    std::vector<std::uint32_t> out;
    std::size_t pos = 0;
    std::uint32_t value = start.value;
    int remaining = start.remaining;
    while (pos < text.size() && text[pos] != '\0' && remaining > 0) {
        const auto b = static_cast<std::uint8_t>(text[pos]);
        if ((b >> 6) != 2)
            return {{}, {0, -1}};
        value = (value << 6) + (b & 0x3fu);
        ++pos;
        --remaining;
    }
    if (start.remaining > 0 && remaining == 0)
        out.push_back(value);
    while (pos < text.size() && text[pos] != '\0') {
        const auto first = static_cast<std::uint8_t>(text[pos]);
        remaining = lookup[first >> 4] - 1;
        if (remaining < 0)
            return {{}, {0, remaining}};
        value = first & static_cast<std::uint8_t>((1u << (7 - remaining)) - 1);
        ++pos;
        while (pos < text.size() && text[pos] != '\0' && remaining > 0) {
            value = (value << 6) + (static_cast<std::uint8_t>(text[pos]) & 0x3fu);
            ++pos;
            --remaining;
        }
        if (remaining == 0)
            out.push_back(value);
    }
    return {std::move(out), {value, remaining}};
}

/// \brief The pushdown stacks of a grammar over the text accepted so far, and the token mask that follows from them.
/// \ingroup core
///
/// Each stack is a list of element indices whose top is a terminal; an empty stack means the text so far is a
/// complete sentence. The matcher holds a `grammar_view`: the grammar must outlive it.
class grammar_matcher
{
public:
    /// \brief One pushdown stack: element indices, the top last.
    using stack = std::vector<std::uint32_t>;

    /// \brief Starts at the root rule: one stack per alternative, each advanced to its first terminal.
    /// \param g The grammar.
    constexpr explicit grammar_matcher(grammar_view g) : g_(g)
    {
        assert(g.root < g.rules.size());
        std::uint32_t pos = g.rules[g.root];
        while (true) {
            stack s;
            if (!detail::grammar::end_of_sequence(g_.elements[pos]))
                s.push_back(pos);
            advance(s, stacks_);
            while (!detail::grammar::end_of_sequence(g_.elements[pos]))
                ++pos;
            if (g_.elements[pos].type != grammar_element_type::alt)
                break;
            ++pos;
        }
    }

    /// \brief Starts at the root rule of a run-time grammar.
    /// \param g The grammar, which must outlive the matcher.
    constexpr explicit grammar_matcher(const grammar& g) : grammar_matcher(g.view()) {}
    /// \brief Deleted: the matcher would outlive a temporary grammar.
    grammar_matcher(const grammar&&) = delete;

    /// \brief Starts at the root rule of a compile-time grammar.
    /// \tparam E The elements.
    /// \tparam R The rules.
    /// \param g The grammar, which must outlive the matcher.
    template <std::size_t E, std::size_t R>
    constexpr explicit grammar_matcher(const fixed_grammar<E, R>& g) : grammar_matcher(g.view())
    {
    }
    /// \brief Deleted: the matcher would outlive a temporary grammar.
    /// \tparam E The elements.
    /// \tparam R The rules.
    template <std::size_t E, std::size_t R>
    grammar_matcher(const fixed_grammar<E, R>&&) = delete;

    /// \brief Accepts one code point (llama.cpp's `llama_grammar_accept`): stacks topped by a token terminal drop out.
    /// \param c The code point.
    /// \return Whether any stack survives.
    constexpr bool accept_code_point(std::uint32_t c)
    {
        std::vector<stack> next;
        for (const auto& s : stacks_)
            accept_chr(s, c, next);
        stacks_ = std::move(next);
        return !stacks_.empty();
    }

    /// \brief Accepts UTF-8 text, carrying an unfinished sequence to the next call (llama.cpp's `llama_grammar_accept_str`);
    /// stacks that no completion of an unfinished last sequence can satisfy drop out, as the mask has it.
    /// \param text The bytes.
    /// \return Whether any stack survives.
    constexpr bool accept_string(std::string_view text)
    {
        const auto [cps, state] = decode_utf8(text, partial_);
        for (const std::uint32_t c : cps)
            if (!accept_code_point(c))
                break;
        std::vector<stack> kept;
        for (auto& s : stacks_)
            if (keeps_partial(s, state))
                kept.push_back(std::move(s));
        stacks_ = std::move(kept);
        partial_ = state;
        return !stacks_.empty();
    }

    /// \brief Accepts a token (llama.cpp's `llama_grammar_accept_token`): a stack topped by a token terminal takes the
    /// id, any other takes the piece's code points; as in the mask, a stack that cannot take some completion of an
    /// unfinished last sequence drops out (llama.cpp keeps it until the next token), so a token is accepted exactly when
    /// `allows` says it may follow.
    /// \param token The token id.
    /// \param piece Its text.
    /// \return Whether any stack survives.
    constexpr bool accept_token(std::uint32_t token, std::string_view piece)
    {
        const auto [cps, state] = decode_utf8(piece, partial_);
        std::vector<stack> out;
        for (const auto& s : stacks_) {
            if (s.empty())
                continue;
            const grammar_element top = g_.elements[s.back()];
            if (is_token(top)) {
                if (!(cps.empty() && state.remaining != 0) && match_token(top, token)) {
                    stack n(s.begin(), s.end() - 1);
                    if (!detail::grammar::end_of_sequence(g_.elements[s.back() + 1]))
                        n.push_back(s.back() + 1);
                    advance(n, out);
                }
                continue;
            }
            for (auto& survivor : consume(s, cps))
                if (keeps_partial(survivor, state))
                    add_unique(out, std::move(survivor));
        }
        stacks_ = std::move(out);
        partial_ = state;
        return !stacks_.empty();
    }

    /// \brief Accepts the end of generation: allowed exactly when the text is complete; changes nothing.
    /// \return `complete()`.
    [[nodiscard]] constexpr bool accept_end() const noexcept { return complete(); }

    /// \brief Whether the text so far is a whole sentence: some stack is empty (llama.cpp's test for allowing an end-of-generation token).
    /// \return The answer.
    [[nodiscard]] constexpr bool complete() const noexcept
    {
        return std::any_of(stacks_.begin(), stacks_.end(), [](const stack& s) { return s.empty(); });
    }

    /// \brief Whether no continuation is possible: every stack has died.
    /// \return `stacks().empty()`.
    [[nodiscard]] constexpr bool failed() const noexcept { return stacks_.empty(); }

    /// \brief The current stacks.
    /// \return The stacks.
    [[nodiscard]] constexpr const std::vector<stack>& stacks() const noexcept { return stacks_; }

    /// \brief The unfinished UTF-8 sequence carried from the last token.
    /// \return The state.
    [[nodiscard]] constexpr grammar_utf8_state utf8_state() const noexcept { return partial_; }

    /// \brief Whether a (non-end) token can follow: some stack accepts its whole piece, an unfinished last sequence
    /// included (llama.cpp's `llama_grammar_reject_candidates` for one candidate); an empty piece never can.
    /// \param token The token id, for token terminals.
    /// \param piece Its text.
    /// \return Whether the token is allowed.
    [[nodiscard]] constexpr bool allows(std::uint32_t token, std::string_view piece) const
    {
        if (piece.empty() || piece[0] == '\0')
            return false;
        const auto [cps, state] = decode_utf8(piece, partial_);
        for (const auto& s : stacks_) {
            if (s.empty()) {
                if (cps.empty() && state.remaining == 0)
                    return true;
                continue;
            }
            const grammar_element top = g_.elements[s.back()];
            if (is_token(top)) {
                if (cps.empty() ? state.remaining == 0 : match_token(top, token))
                    return true;
                continue;
            }
            for (const auto& t : consume(s, cps))
                if (keeps_partial(t, state))
                    return true;
        }
        return false;
    }

    /// \brief The token mask over a vocabulary given by callables.
    /// \tparam PieceOf A callable `std::uint32_t -> std::string_view`.
    /// \tparam IsEnd A callable `std::uint32_t -> bool`, true for end-of-generation tokens.
    /// \param piece_of The piece of each token.
    /// \param is_end Whether a token ends generation.
    /// \param allowed Receives, per token id, whether it may follow.
    template <typename PieceOf, typename IsEnd>
        requires std::convertible_to<std::invoke_result_t<PieceOf&, std::uint32_t>, std::string_view>
                 && std::convertible_to<std::invoke_result_t<IsEnd&, std::uint32_t>, bool>
    constexpr void token_mask(PieceOf&& piece_of, IsEnd&& is_end, std::span<bool> allowed) const
    {
        const bool end_ok = complete();
        for (std::size_t t = 0; t < allowed.size(); ++t) {
            const auto id = static_cast<std::uint32_t>(t);
            allowed[t] = is_end(id) ? end_ok : allows(id, piece_of(id));
        }
    }

    /// \brief The token mask over a vocabulary of pieces.
    /// \param vocabulary The piece of each token id.
    /// \param end_tokens The end-of-generation token ids.
    /// \param allowed Receives, per token id, whether it may follow; the vocabulary's size.
    constexpr void token_mask(std::span<const std::string_view> vocabulary, std::span<const std::uint32_t> end_tokens,
                              std::span<bool> allowed) const
    {
        assert(allowed.size() == vocabulary.size());
        token_mask([&](std::uint32_t t) { return vocabulary[t]; },
                   [&](std::uint32_t t) { return std::find(end_tokens.begin(), end_tokens.end(), t) != end_tokens.end(); }, allowed);
    }

    /// \brief The token mask as bits: bit `t % 32` of word `t / 32` set when token `t` may follow.
    /// \param vocabulary The piece of each token id.
    /// \param end_tokens The end-of-generation token ids.
    /// \param words Receives `ceil(vocabulary / 32)` words.
    constexpr void token_bitmask(std::span<const std::string_view> vocabulary, std::span<const std::uint32_t> end_tokens,
                                 std::span<std::uint32_t> words) const
    {
        assert(words.size() == (vocabulary.size() + 31) / 32);
        std::fill(words.begin(), words.end(), 0u);
        const bool end_ok = complete();
        for (std::size_t t = 0; t < vocabulary.size(); ++t) {
            const auto id = static_cast<std::uint32_t>(t);
            const bool end = std::find(end_tokens.begin(), end_tokens.end(), id) != end_tokens.end();
            if (end ? end_ok : allows(id, vocabulary[t]))
                words[t / 32] |= 1u << (t % 32);
        }
    }

private:
    static constexpr bool is_token(grammar_element e) noexcept
    {
        return e.type == grammar_element_type::token || e.type == grammar_element_type::token_not;
    }

    static constexpr bool match_token(grammar_element e, std::uint32_t token) noexcept
    {
        return e.type == grammar_element_type::token ? e.value == token : e.value != token;
    }

    static constexpr void add_unique(std::vector<stack>& out, stack s)
    {
        if (std::find(out.begin(), out.end(), s) == out.end())
            out.push_back(std::move(s));
    }

    // llama_grammar_match_char: whether c satisfies the class at pos, and the element after the class
    constexpr std::pair<bool, std::uint32_t> match_char(std::uint32_t pos, std::uint32_t c) const
    {
        const auto e = g_.elements;
        bool found = false;
        const bool positive = e[pos].type == grammar_element_type::chr || e[pos].type == grammar_element_type::chr_any;
        assert(positive || e[pos].type == grammar_element_type::chr_not);
        do {
            if (e[pos + 1].type == grammar_element_type::chr_range_upper) {
                found = found || (e[pos].value <= c && c <= e[pos + 1].value);
                pos += 2;
            } else if (e[pos].type == grammar_element_type::chr_any) {
                found = true;
                pos += 1;
            } else {
                found = found || e[pos].value == c;
                pos += 1;
            }
        } while (e[pos].type == grammar_element_type::chr_alt);
        return {found == positive, pos};
    }

    // llama_grammar_match_partial_char: whether some completion of the partial sequence can satisfy the class at pos
    constexpr bool match_partial_char(std::uint32_t pos, grammar_utf8_state partial) const
    {
        const auto e = g_.elements;
        const bool positive = e[pos].type == grammar_element_type::chr || e[pos].type == grammar_element_type::chr_any;
        const int n = partial.remaining;
        if (n < 0 || (n == 1 && partial.value < 2))
            return false;
        std::uint32_t low = partial.value << (n * 6);
        const std::uint32_t high = low | ((1u << (n * 6)) - 1);
        if (low == 0) {
            if (n == 2)
                low = 1u << 11;
            else if (n == 3)
                low = 1u << 16;
        }
        do {
            if (e[pos + 1].type == grammar_element_type::chr_range_upper) {
                if (e[pos].value <= high && low <= e[pos + 1].value)
                    return positive;
                pos += 2;
            } else if (e[pos].type == grammar_element_type::chr_any) {
                return true;
            } else {
                if (low <= e[pos].value && e[pos].value <= high)
                    return positive;
                pos += 1;
            }
        } while (e[pos].type == grammar_element_type::chr_alt);
        return !positive;
    }

    // llama_grammar_advance_stack: expand rule references until every stack is topped by a terminal (or empty)
    constexpr void advance(const stack& start, std::vector<stack>& out) const
    {
        std::vector<stack> todo{start}, seen;
        while (!todo.empty()) {
            stack cur = std::move(todo.back());
            todo.pop_back();
            if (std::find(seen.begin(), seen.end(), cur) != seen.end())
                continue;
            seen.push_back(cur);
            if (cur.empty()) {
                add_unique(out, cur);
                continue;
            }
            const std::uint32_t pos = cur.back();
            const grammar_element e = g_.elements[pos];
            switch (e.type) {
            case grammar_element_type::rule_ref: {
                std::uint32_t sub = g_.rules[e.value];
                while (true) {
                    stack next(cur.begin(), cur.end() - 1);
                    if (!detail::grammar::end_of_sequence(g_.elements[pos + 1]))
                        next.push_back(pos + 1);
                    if (!detail::grammar::end_of_sequence(g_.elements[sub]))
                        next.push_back(sub);
                    todo.push_back(std::move(next));
                    while (!detail::grammar::end_of_sequence(g_.elements[sub]))
                        ++sub;
                    if (g_.elements[sub].type != grammar_element_type::alt)
                        break;
                    ++sub;
                }
                break;
            }
            case grammar_element_type::chr:
            case grammar_element_type::chr_not:
            case grammar_element_type::chr_any:
            case grammar_element_type::token:
            case grammar_element_type::token_not:
                add_unique(out, cur);
                break;
            default:
                assert(false && "a stack is never left on the end of an alternative or inside a class");
                break;
            }
        }
    }

    // llama_grammar_accept_chr
    constexpr void accept_chr(const stack& s, std::uint32_t c, std::vector<stack>& out) const
    {
        if (s.empty() || is_token(g_.elements[s.back()]))
            return;
        const auto [matched, after] = match_char(s.back(), c);
        if (!matched)
            return;
        stack n(s.begin(), s.end() - 1);
        if (!detail::grammar::end_of_sequence(g_.elements[after]))
            n.push_back(after);
        advance(n, out);
    }

    // Whether a stack can take some completion of an unfinished sequence (any stack, when none is unfinished)
    constexpr bool keeps_partial(const stack& s, grammar_utf8_state state) const
    {
        return state.remaining == 0 || (!s.empty() && !is_token(g_.elements[s.back()]) && match_partial_char(s.back(), state));
    }

    // The stacks that one stack becomes after the code points
    constexpr std::vector<stack> consume(const stack& s, const std::vector<std::uint32_t>& cps) const
    {
        std::vector<stack> cur{s};
        for (const std::uint32_t c : cps) {
            std::vector<stack> next;
            for (const auto& t : cur)
                accept_chr(t, c, next);
            cur = std::move(next);
            if (cur.empty())
                break;
        }
        return cur;
    }

    grammar_view g_;
    std::vector<stack> stacks_{};
    grammar_utf8_state partial_{};
};

/// \brief Whether a whole string is a sentence of a grammar: accept its code points, then ask `complete()`.
/// \param g The grammar.
/// \param text UTF-8 text.
/// \return The answer; an unfinished last UTF-8 sequence is not a sentence.
[[nodiscard]] constexpr bool grammar_accepts(grammar_view g, std::string_view text)
{
    grammar_matcher m(g);
    return m.accept_string(text) && m.utf8_state().remaining == 0 && m.complete();
}

/// \brief Whether a whole string is a sentence of a compile-time grammar.
/// \tparam E The elements.
/// \tparam R The rules.
/// \param g The grammar.
/// \param text UTF-8 text.
/// \return The answer.
template <std::size_t E, std::size_t R>
[[nodiscard]] constexpr bool grammar_accepts(const fixed_grammar<E, R>& g, std::string_view text)
{
    return grammar_accepts(g.view(), text);
}

/// \brief Whether a whole string is a sentence of a run-time grammar.
/// \param g The grammar.
/// \param text UTF-8 text.
/// \return The answer.
[[nodiscard]] constexpr bool grammar_accepts(const grammar& g, std::string_view text) { return grammar_accepts(g.view(), text); }

/// \brief Sets the logit of every token whose bit is clear to `-infinity`, in place: the mask applied before sampling.
/// \tparam T A floating-point type.
/// \param words The mask from `token_bitmask`.
/// \param logits The logits, at most `32 · words.size()`.
template <std::floating_point T>
constexpr void apply_token_bitmask(std::span<const std::uint32_t> words, std::span<T> logits) noexcept
{
    assert(logits.size() <= 32 * words.size());
    for (std::size_t t = 0; t < logits.size(); ++t)
        if (((words[t / 32] >> (t % 32)) & 1u) == 0)
            logits[t] = -std::numeric_limits<T>::infinity();
}

} // namespace snicholls::ml

#endif // snicholls_core_ml_grammar_hpp
