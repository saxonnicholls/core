//
//  json_schema.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::ml` JSON schema to GBNF: a practical subset of JSON Schema written as the grammar llama.cpp writes for it, so that generation emits only JSON that matches the schema.
/// \ingroup core
///
/// **Overview.** Structured output (a tool call, an extraction into fields) is a JSON schema; `json_schema_to_gbnf`
/// turns the schema's text into GBNF, which `ml/grammar.hpp` compiles and masks tokens with. The grammar is llama.cpp's
/// (`common/json-schema-to-grammar.cpp`): the same rule names, the `space` rule, the primitive rules (`value`,
/// `object`, `array`, `string`, `char`, `number`, `integer`, `integral-part`, `decimal-part`, `boolean`, `null`) and
/// the rules sorted by name, so its output can be compared with llama.cpp's line for line.
///
/// Supported: `type` (one or a list: a list is an `anyOf` of the schema with each type), `const`, `enum`,
/// `anyOf` / `oneOf`, `allOf` (merging the properties of object schemas, intersecting enums, intersecting integer
/// ranges, as llama.cpp does), objects (`properties` in their written order, `required`, `additionalProperties` as a
/// boolean or a schema, an object without `properties` accepting any object), arrays (`items` as a schema or a tuple,
/// `prefixItems`, `minItems`, `maxItems`), strings (`minLength`, `maxLength`, `format` `date`, `time`, `date-time` and
/// `uuid`; other formats are annotations), integers (`minimum`, `maximum`, `exclusiveMinimum`, `exclusiveMaximum`,
/// as llama.cpp's digit-by-digit ranges), numbers, booleans, null, the empty schema (any value), and `$ref` to
/// anywhere in the same document (`#/$defs/...`, `#/definitions/...`), recursive ones included. Annotations
/// (`title`, `description`, `default`, `examples`, `$schema`, `$id`, ...) are ignored.
///
/// - `json_schema_to_gbnf(schema)`: the GBNF text, or a `json_schema_error` naming the keyword, its path (`#/...`) and,
///   for a syntax error, the byte.
/// - `json_schema_grammar(schema)`: the schema straight to a compiled `grammar`.
/// - `json_schema_options{.strict = false}`: llama.cpp's leniency instead, ignoring what is not implemented.
/// - `compile_json_schema<"...">()`: at compile time, into a `fixed_grammar`, for a schema written in the program.
///
/// \snippet ml/json_schema_demo.cpp json_schema
///
/// **Relationships.** `ml/grammar.hpp` compiles and matches the output. Group: \ref core.
///
/// **Implementations.** One, ported from llama.cpp at commit 39039aa: `common/json-schema-to-grammar.cpp`
/// (`build_repetition`, `build_min_max_int`, `PRIMITIVE_RULES`, `STRING_FORMAT_RULES`, `format_literal`,
/// `gbnf_format_rule_name`, `common_chat_schema_converter::_add_rule`, `_not_strings`, `_resolve_ref`,
/// `_build_object_rule`, `_add_primitive`, `_visit_all_of`, `visit`, `format_grammar`) and `common/json-schema.cpp`
/// (`common_chat_schema_builder`: which keywords make which kind of schema, in its order), checked against the expected
/// grammars of `tests/test-json-schema-to-grammar.cpp` (`tests/ml/json_schema_test.cpp`).
///
/// **Guarantees.** `constexpr` throughout (`compile_json_schema` is `consteval`); no exceptions. The output always
/// parses with `parse_grammar` (tested). Complexity: linear in the schema, plus the integer ranges' digits.
///
/// **Errors.** A `json_schema_error` for: malformed JSON (with its byte); a schema that is not an object; an unknown
/// `type`; an empty or non-array `enum`, `anyOf`, `oneOf` or `allOf`; a negative or non-integer count; an unresolved
/// or external `$ref`; an `allOf` whose integer ranges do not meet; nesting deeper than 128; and the keywords this
/// subset does not implement, each named: `pattern`, `patternProperties`, `propertyNames`, `not`, `if`, `contains`,
/// `uniqueItems`, `multipleOf`, `minProperties`, `maxProperties`, `dependentRequired`, `dependentSchemas`,
/// `unevaluatedProperties`, `unevaluatedItems`, and `minimum` / `maximum` / `exclusiveMinimum` / `exclusiveMaximum`
/// on a `number` (which llama.cpp ignores, so that its grammar admits numbers the schema rejects). Not strict, those
/// are ignored as llama.cpp ignores them, and a `pattern` accepts any string.
///
/// **Performance.** Not benchmarked: a schema is converted once per request, its cost is small beside a forward pass.
///
/// **Tests and demos.** `tests/ml/json_schema_test.cpp`, `demos/ml/json_schema_demo.cpp`.
///
/// **Dependencies.** `<algorithm>`, `<array>`, `<cstddef>`, `<cstdint>`, `<limits>`, `<string>`, `<string_view>`,
/// `<utility>`, `<vector>`, `snicholls/core/ml/grammar.hpp`, `snicholls/core/result.hpp`.
///
/// **Known issues.** Numbers in `const` and `enum` are written as they appear in the schema; llama.cpp re-prints them
/// through its JSON library (`1e2` becomes `100.0`), so the two can differ for numbers not in canonical form.
/// `pattern` (a regular expression) is not converted: llama.cpp converts a subset of regular expressions, or accepts any
/// string with a warning; here it is an error. As in llama.cpp, the key rule of `additionalProperties` beside
/// `properties` writes the property names' characters into classes unescaped, and refuses some keys that are proper
/// prefixes of the declared names.
///
/// **Status.** Experimental, new in 0.17.0.

#ifndef snicholls_core_ml_json_schema_hpp
#define snicholls_core_ml_json_schema_hpp

#include <snicholls/core/fixed_string.hpp>
#include <snicholls/core/ml/grammar.hpp>
#include <snicholls/core/result.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace snicholls::ml
{

/// \brief Why a schema could not be converted: the byte of a JSON syntax error, the schema's path and a message
/// naming the keyword.
/// \ingroup core
struct json_schema_error
{
    std::size_t position = 0;   ///< For malformed JSON, the byte; otherwise 0.
    std::string path{};         ///< The schema's JSON pointer, `#` for the root, `#/properties/a` below; empty for malformed JSON.
    std::string message{};      ///< What is wrong, naming the keyword.

    /// \brief Equality of every field.
    /// \return Whether they are equal.
    friend constexpr bool operator==(const json_schema_error&, const json_schema_error&) = default;
};

/// \brief How strictly `json_schema_to_gbnf` treats what this subset does not implement.
/// \ingroup core
struct json_schema_options
{
    /// \brief True: an unimplemented keyword is an error naming it. False: as llama.cpp, it is ignored (the grammar then
    /// admits more than the schema), and a `pattern` gives any string.
    bool strict = true;
};

namespace detail::json_schema
{
    inline constexpr int recursion_limit = 128;   // llama.cpp's COMMON_RECURSIVE_LIMIT

    // ---- a small JSON reader: objects keep their keys in written order (llama.cpp's ordered JSON)

    enum class json_kind : std::uint8_t { null, boolean, number, string, array, object };

    struct value
    {
        json_kind kind = json_kind::null;
        bool boolean = false;
        std::string text{};                // a string's characters (UTF-8) or a number as written
        std::vector<value> items{};        // array elements, or object values
        std::vector<std::string> keys{};   // object keys, parallel to items

        constexpr const value* get(std::string_view key) const
        {
            if (kind != json_kind::object)
                return nullptr;
            for (std::size_t i = 0; i < keys.size(); ++i)
                if (keys[i] == key)
                    return &items[i];
            return nullptr;
        }
        constexpr bool has(std::string_view key) const { return get(key) != nullptr; }
        constexpr bool is_integer() const
        {
            return kind == json_kind::number && text.find_first_of(".eE") == std::string::npos;
        }
        constexpr bool is_true() const { return kind == json_kind::boolean && boolean; }
    };

    constexpr void append_utf8(std::string& out, std::uint32_t c)
    {
        if (c < 0x80) {
            out += static_cast<char>(c);
        } else if (c < 0x800) {
            out += static_cast<char>(0xc0 | (c >> 6));
            out += static_cast<char>(0x80 | (c & 0x3f));
        } else if (c < 0x10000) {
            out += static_cast<char>(0xe0 | (c >> 12));
            out += static_cast<char>(0x80 | ((c >> 6) & 0x3f));
            out += static_cast<char>(0x80 | (c & 0x3f));
        } else {
            out += static_cast<char>(0xf0 | (c >> 18));
            out += static_cast<char>(0x80 | ((c >> 12) & 0x3f));
            out += static_cast<char>(0x80 | ((c >> 6) & 0x3f));
            out += static_cast<char>(0x80 | (c & 0x3f));
        }
    }

    struct reader
    {
        std::string_view s;
        std::size_t pos = 0;
        std::size_t error_at = 0;
        std::string_view error{};

        constexpr bool fail(std::string_view message)
        {
            if (error.empty()) {
                error_at = pos;
                error = message;
            }
            return false;
        }
        constexpr char peek() const { return pos < s.size() ? s[pos] : '\0'; }
        constexpr void space()
        {
            while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\t' || s[pos] == '\n' || s[pos] == '\r'))
                ++pos;
        }
        constexpr bool word(std::string_view w)
        {
            if (s.substr(pos, w.size()) != w)
                return fail("invalid literal");
            pos += w.size();
            return true;
        }
        constexpr int hex(char c) const
        {
            return ('0' <= c && c <= '9') ? c - '0' : ('a' <= c && c <= 'f') ? c - 'a' + 10 : ('A' <= c && c <= 'F') ? c - 'A' + 10 : -1;
        }
        constexpr bool hex4(std::uint32_t& out)
        {
            out = 0;
            for (int i = 0; i < 4; ++i, ++pos) {
                const int h = hex(peek());
                if (h < 0)
                    return fail("expecting four hexadecimal digits");
                out = out * 16 + static_cast<std::uint32_t>(h);
            }
            return true;
        }
        constexpr bool string(std::string& out)
        {
            ++pos;   // the opening quote
            while (true) {
                if (pos >= s.size())
                    return fail("unterminated string");
                const char c = s[pos];
                if (c == '"') {
                    ++pos;
                    return true;
                }
                if (static_cast<unsigned char>(c) < 0x20)
                    return fail("control character in a string");
                if (c != '\\') {
                    out += c;
                    ++pos;
                    continue;
                }
                ++pos;
                const char e = peek();
                ++pos;
                switch (e) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u': {
                    std::uint32_t c1 = 0;
                    if (!hex4(c1))
                        return false;
                    if (c1 >= 0xd800 && c1 < 0xdc00) {   // a surrogate pair
                        std::uint32_t c2 = 0;
                        if (peek() != '\\' || (pos + 1 < s.size() && s[pos + 1] != 'u'))
                            return fail("unpaired surrogate");
                        pos += 2;
                        if (!hex4(c2))
                            return false;
                        if (c2 < 0xdc00 || c2 > 0xdfff)
                            return fail("unpaired surrogate");
                        c1 = 0x10000 + ((c1 - 0xd800) << 10) + (c2 - 0xdc00);
                    } else if (c1 >= 0xdc00 && c1 <= 0xdfff) {
                        return fail("unpaired surrogate");
                    }
                    append_utf8(out, c1);
                    break;
                }
                default: --pos; return fail("invalid escape");
                }
            }
        }
        constexpr bool digits()
        {
            const std::size_t start = pos;
            while ('0' <= peek() && peek() <= '9')
                ++pos;
            return pos != start || fail("expecting a digit");
        }
        constexpr bool number(value& v)
        {
            const std::size_t start = pos;
            if (peek() == '-')
                ++pos;
            if (peek() == '0')
                ++pos;
            else if (!digits())
                return false;
            if (peek() == '.') {
                ++pos;
                if (!digits())
                    return false;
            }
            if (peek() == 'e' || peek() == 'E') {
                ++pos;
                if (peek() == '+' || peek() == '-')
                    ++pos;
                if (!digits())
                    return false;
            }
            v.kind = json_kind::number;
            v.text = std::string(s.substr(start, pos - start));
            return true;
        }
        constexpr bool parse(value& v, int depth)
        {
            if (depth > 512)
                return fail("JSON nested more than 512 deep");
            space();
            const char c = peek();
            if (c == '{') {
                v.kind = json_kind::object;
                ++pos;
                space();
                if (peek() == '}') {
                    ++pos;
                    return true;
                }
                while (true) {
                    space();
                    if (peek() != '"')
                        return fail("expecting a key");
                    std::string key;
                    if (!string(key))
                        return false;
                    space();
                    if (peek() != ':')
                        return fail("expecting ':'");
                    ++pos;
                    value item;
                    if (!parse(item, depth + 1))
                        return false;
                    bool replaced = false;   // a repeated key keeps its place and takes the last value
                    for (std::size_t i = 0; i < v.keys.size(); ++i)
                        if (v.keys[i] == key) {
                            v.items[i] = std::move(item);
                            replaced = true;
                            break;
                        }
                    if (!replaced) {
                        v.keys.push_back(std::move(key));
                        v.items.push_back(std::move(item));
                    }
                    space();
                    if (peek() == ',') {
                        ++pos;
                        continue;
                    }
                    if (peek() == '}') {
                        ++pos;
                        return true;
                    }
                    return fail("expecting ',' or '}'");
                }
            }
            if (c == '[') {
                v.kind = json_kind::array;
                ++pos;
                space();
                if (peek() == ']') {
                    ++pos;
                    return true;
                }
                while (true) {
                    value item;
                    if (!parse(item, depth + 1))
                        return false;
                    v.items.push_back(std::move(item));
                    space();
                    if (peek() == ',') {
                        ++pos;
                        continue;
                    }
                    if (peek() == ']') {
                        ++pos;
                        return true;
                    }
                    return fail("expecting ',' or ']'");
                }
            }
            if (c == '"') {
                v.kind = json_kind::string;
                return string(v.text);
            }
            if (c == 't') {
                v.kind = json_kind::boolean;
                v.boolean = true;
                return word("true");
            }
            if (c == 'f') {
                v.kind = json_kind::boolean;
                return word("false");
            }
            if (c == 'n') {
                v.kind = json_kind::null;
                return word("null");
            }
            if (c == '-' || ('0' <= c && c <= '9'))
                return number(v);
            return fail("expecting a JSON value");
        }
    };

    // nlohmann::json's compact dump, as llama.cpp prints a const or an enum value (numbers as written)
    constexpr void dump_string(std::string& out, std::string_view s)
    {
        constexpr char digits[] = "0123456789abcdef";
        out += '"';
        for (const char c : s) {
            switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    out += "\\u00";
                    out += digits[(static_cast<unsigned char>(c) >> 4) & 0xf];
                    out += digits[static_cast<unsigned char>(c) & 0xf];
                } else {
                    out += c;
                }
            }
        }
        out += '"';
    }

    constexpr void dump(std::string& out, const value& v)
    {
        switch (v.kind) {
        case json_kind::null: out += "null"; break;
        case json_kind::boolean: out += v.boolean ? "true" : "false"; break;
        case json_kind::number: out += v.text; break;
        case json_kind::string: dump_string(out, v.text); break;
        case json_kind::array:
            out += '[';
            for (std::size_t i = 0; i < v.items.size(); ++i) {
                if (i > 0)
                    out += ',';
                dump(out, v.items[i]);
            }
            out += ']';
            break;
        case json_kind::object:
            out += '{';
            for (std::size_t i = 0; i < v.items.size(); ++i) {
                if (i > 0)
                    out += ',';
                dump_string(out, v.keys[i]);
                out += ':';
                dump(out, v.items[i]);
            }
            out += '}';
            break;
        }
    }

    // ---- helpers of json-schema-to-grammar.cpp

    constexpr std::string decimal(std::int64_t v)
    {
        if (v == std::numeric_limits<std::int64_t>::min())
            return "-9223372036854775808";
        const bool negative = v < 0;
        std::uint64_t u = negative ? static_cast<std::uint64_t>(-v) : static_cast<std::uint64_t>(v);
        std::string s;
        do {
            s.insert(s.begin(), static_cast<char>('0' + u % 10));
            u /= 10;
        } while (u != 0);
        if (negative)
            s.insert(s.begin(), '-');
        return s;
    }

    constexpr std::int64_t to_int(std::string_view s)
    {
        std::int64_t v = 0;
        for (const char c : s)
            v = v * 10 + (c - '0');
        return v;
    }

    constexpr std::string repeat(std::string_view s, std::size_t n)
    {
        std::string out;
        for (std::size_t i = 0; i < n; ++i)
            out += s;
        return out;
    }

    constexpr std::string join(const std::vector<std::string>& parts, std::string_view separator)
    {
        std::string out;
        for (std::size_t i = 0; i < parts.size(); ++i) {
            if (i > 0)
                out += separator;
            out += parts[i];
        }
        return out;
    }

    inline constexpr int no_max_items = std::numeric_limits<int>::max();

    constexpr std::string build_repetition(const std::string& item, int min_items, int max_items, const std::string& separator = "")
    {
        const bool has_max = max_items != no_max_items;
        if (max_items == 0)
            return "";
        if (min_items == 0 && max_items == 1)
            return item + "?";
        if (separator.empty()) {
            if (min_items == 1 && !has_max)
                return item + "+";
            if (min_items == 0 && !has_max)
                return item + "*";
            return item + "{" + decimal(min_items) + "," + (has_max ? decimal(max_items) : std::string()) + "}";
        }
        std::string result = item + " "
                             + build_repetition("(" + separator + " " + item + ")", min_items == 0 ? 0 : min_items - 1,
                                                has_max ? max_items - 1 : max_items);
        if (min_items == 0)
            result = "(" + result + ")?";
        return result;
    }

    inline constexpr std::int64_t no_min = std::numeric_limits<std::int64_t>::min();
    inline constexpr std::int64_t no_max = std::numeric_limits<std::int64_t>::max();

    constexpr void digit_range(std::string& out, char from, char to)
    {
        out += '[';
        out += from;
        if (from != to) {
            out += '-';
            out += to;
        }
        out += ']';
    }

    constexpr void more_digits(std::string& out, int min_digits, int max_digits)
    {
        out += "[0-9]";
        if (min_digits == max_digits && min_digits == 1)
            return;
        out += '{';
        out += decimal(min_digits);
        if (max_digits != min_digits) {
            out += ',';
            if (max_digits != std::numeric_limits<int>::max())
                out += decimal(max_digits);
        }
        out += '}';
    }

    constexpr void uniform_range(std::string& out, std::string_view from, std::string_view to)
    {
        std::size_t i = 0;
        while (i < from.size() && i < to.size() && from[i] == to[i])
            ++i;
        if (i > 0) {
            out += '"';
            out += from.substr(0, i);
            out += '"';
        }
        if (i < from.size() && i < to.size()) {
            if (i > 0)
                out += ' ';
            const std::size_t sub_len = from.size() - i - 1;
            if (sub_len > 0) {
                const std::string from_sub(from.substr(i + 1)), to_sub(to.substr(i + 1));
                const std::string zeros = repeat("0", sub_len), nines = repeat("9", sub_len);
                bool to_reached = false;
                out += '(';
                if (from_sub == zeros) {
                    digit_range(out, from[i], static_cast<char>(to[i] - 1));
                    out += ' ';
                    more_digits(out, static_cast<int>(sub_len), static_cast<int>(sub_len));
                } else {
                    out += '[';
                    out += from[i];
                    out += "] (";
                    uniform_range(out, from_sub, nines);
                    out += ')';
                    if (from[i] < to[i] - 1) {
                        out += " | ";
                        if (to_sub == nines) {
                            digit_range(out, static_cast<char>(from[i] + 1), to[i]);
                            to_reached = true;
                        } else {
                            digit_range(out, static_cast<char>(from[i] + 1), static_cast<char>(to[i] - 1));
                        }
                        out += ' ';
                        more_digits(out, static_cast<int>(sub_len), static_cast<int>(sub_len));
                    }
                }
                if (!to_reached) {
                    out += " | ";
                    digit_range(out, to[i], to[i]);
                    out += ' ';
                    uniform_range(out, zeros, to_sub);
                }
                out += ')';
            } else {
                out += '[';
                out += from[i];
                out += '-';
                out += to[i];
                out += ']';
            }
        }
    }

    // llama.cpp's build_min_max_int: the integers in [min_value, max_value] digit by digit, without leading zeros
    constexpr void build_min_max_int(std::string& out, std::int64_t min_value, std::int64_t max_value, int decimals_left = 16,
                                     bool top_level = true)
    {
        const bool has_min = min_value != no_min, has_max = max_value != no_max;
        if (has_min && has_max) {
            if (min_value < 0 && max_value < 0) {
                out += "\"-\" (";
                build_min_max_int(out, -max_value, -min_value, decimals_left, true);
                out += ')';
                return;
            }
            if (min_value < 0) {
                out += "\"-\" (";
                build_min_max_int(out, 0, -min_value, decimals_left, true);
                out += ") | ";
                min_value = 0;
            }
            std::string min_s = decimal(min_value);
            const std::string max_s = decimal(max_value);
            for (std::size_t digits = min_s.size(); digits < max_s.size(); ++digits) {
                uniform_range(out, min_s, repeat("9", digits));
                min_s = "1" + repeat("0", digits);
                out += " | ";
            }
            uniform_range(out, min_s, max_s);
            return;
        }
        const int less_decimals = std::max(decimals_left - 1, 1);
        if (has_min) {
            if (min_value < 0) {
                out += "\"-\" (";
                build_min_max_int(out, no_min, -min_value, decimals_left, false);
                out += ") | [0] | [1-9] ";
                more_digits(out, 0, decimals_left - 1);
            } else if (min_value == 0) {
                if (top_level) {
                    out += "[0] | [1-9] ";
                    more_digits(out, 0, less_decimals);
                } else {
                    more_digits(out, 1, decimals_left);
                }
            } else if (min_value <= 9) {
                const char c = static_cast<char>('0' + min_value);
                const char range_start = top_level ? '1' : '0';
                if (c > range_start) {
                    digit_range(out, range_start, static_cast<char>(c - 1));
                    out += ' ';
                    more_digits(out, 1, less_decimals);
                    out += " | ";
                }
                digit_range(out, c, '9');
                out += ' ';
                more_digits(out, 0, less_decimals);
            } else {
                const std::string min_s = decimal(min_value);
                const int len = static_cast<int>(min_s.size());
                const char c = min_s[0];
                if (c > '1') {
                    digit_range(out, top_level ? '1' : '0', static_cast<char>(c - 1));
                    out += ' ';
                    more_digits(out, len, less_decimals);
                    out += " | ";
                }
                digit_range(out, c, c);
                out += " (";
                build_min_max_int(out, to_int(std::string_view(min_s).substr(1)), no_max, less_decimals, false);
                out += ')';
                if (c < '9') {
                    out += " | ";
                    digit_range(out, static_cast<char>(c + 1), '9');
                    out += ' ';
                    more_digits(out, len - 1, less_decimals);
                }
            }
            return;
        }
        if (has_max) {
            if (max_value >= 0) {
                if (top_level) {
                    out += "\"-\" [1-9] ";
                    more_digits(out, 0, less_decimals);
                    out += " | ";
                }
                build_min_max_int(out, 0, max_value, decimals_left, true);
            } else {
                out += "\"-\" (";
                build_min_max_int(out, -max_value, no_max, decimals_left, false);
                out += ')';
            }
        }
    }

    inline constexpr std::string_view space_rule = R"(| " " | "\n"{1,2} [ \t]{0,20})";

    struct builtin
    {
        std::string_view name;
        std::string_view content;
        std::array<std::string_view, 6> deps;
    };

    // PRIMITIVE_RULES and STRING_FORMAT_RULES
    inline constexpr std::array<builtin, 18> builtins{{
        {"boolean", R"(("true" | "false"))", {}},
        {"decimal-part", "[0-9]{1,16}", {}},
        {"integral-part", "[0] | [1-9] [0-9]{0,15}", {}},
        {"number", R"(("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?)", {"integral-part", "decimal-part"}},
        {"integer", R"(("-"? integral-part))", {"integral-part"}},
        {"value", "object | array | string | number | boolean | null", {"object", "array", "string", "number", "boolean", "null"}},
        {"object", R"("{" space ( string ":" space value ("," space string ":" space value)* )? space "}")", {"string", "value"}},
        {"array", R"("[" space ( value ("," space value)* )? space "]")", {"value"}},
        {"uuid", R"("\"" [0-9a-fA-F]{8} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{12} "\"")", {}},
        {"char", R"([^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4}))", {}},
        {"string", R"("\"" char* "\"")", {"char"}},
        {"null", R"("null")", {}},
        {"date", R"([0-9]{4} "-" ( "0" [1-9] | "1" [0-2] ) "-" ( "0" [1-9] | [1-2] [0-9] | "3" [0-1] ))", {}},
        {"time",
         R"(([01] [0-9] | "2" [0-3]) ":" [0-5] [0-9] ":" [0-5] [0-9] ( "." [0-9]{3} )? ( "Z" | ( "+" | "-" ) ( [01] [0-9] | "2" [0-3] ) ":" [0-5] [0-9] ))",
         {}},
        {"date-time", R"(date "T" time)", {"date", "time"}},
        {"date-string", R"("\"" date "\"")", {"date"}},
        {"time-string", R"("\"" time "\"")", {"time"}},
        {"date-time-string", R"("\"" date-time "\"")", {"date-time"}},
    }};

    constexpr const builtin& find_builtin(std::string_view name)
    {
        for (const auto& b : builtins)
            if (b.name == name)
                return b;
        return builtins[0];   // never: every name looked up is one of the table's
    }

    constexpr bool is_reserved(std::string_view name)
    {
        return name == "root" || std::any_of(builtins.begin(), builtins.end(), [&](const builtin& b) { return b.name == name; });
    }

    constexpr std::string format_literal(std::string_view literal)
    {
        std::string out = "\"";
        for (const char c : literal) {
            switch (c) {
            case '\r': out += "\\r"; break;
            case '\n': out += "\\n"; break;
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            default: out += c;
            }
        }
        return out + "\"";
    }

    constexpr std::string format_rule_name(std::string_view name)
    {
        std::string out;
        bool in_invalid = false;
        for (const char c : name) {
            if (('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z') || ('0' <= c && c <= '9') || c == '-') {
                out += c;
                in_invalid = false;
            } else if (!in_invalid) {
                out += '-';
                in_invalid = true;
            }
        }
        return out;
    }

    constexpr std::string constant(const value& v)
    {
        std::string s;
        dump(s, v);
        return format_literal(s);
    }

    // A bound as an integer: a fraction rounded inwards (common_chat_schema_builder::get_bound)
    constexpr bool integer_bound(const value& v, bool round_up, std::int64_t& out)
    {
        if (v.kind != json_kind::number)
            return false;
        std::string_view t = v.text;
        const bool negative = !t.empty() && t[0] == '-';
        if (negative)
            t.remove_prefix(1);
        std::string digits;
        std::int64_t point = 0, exponent = 0;
        bool seen_point = false, nonzero_fraction = false;
        std::size_t i = 0;
        for (; i < t.size() && t[i] != 'e' && t[i] != 'E'; ++i) {
            if (t[i] == '.') {
                seen_point = true;
                continue;
            }
            digits += t[i];
            if (seen_point)
                ++point;
        }
        if (i < t.size()) {
            ++i;
            const bool eneg = t[i] == '-';
            if (t[i] == '+' || t[i] == '-')
                ++i;
            for (; i < t.size(); ++i) {
                exponent = exponent * 10 + (t[i] - '0');
                if (exponent > 40)
                    return false;
            }
            if (eneg)
                exponent = -exponent;
        }
        const std::int64_t shift = exponent - point;   // value = digits · 10^shift
        std::uint64_t whole = 0;
        const std::int64_t n = static_cast<std::int64_t>(digits.size());
        const std::int64_t keep = shift >= 0 ? n : n + shift;   // the digits left of the point
        for (std::int64_t k = 0; k < n; ++k) {
            if (k < keep) {
                if (whole > (std::numeric_limits<std::uint64_t>::max() - 9) / 10)
                    return false;
                whole = whole * 10 + static_cast<std::uint64_t>(digits[static_cast<std::size_t>(k)] - '0');
            } else if (digits[static_cast<std::size_t>(k)] != '0') {
                nonzero_fraction = true;
            }
        }
        for (std::int64_t k = 0; k < shift; ++k) {
            if (whole > std::numeric_limits<std::uint64_t>::max() / 10)
                return false;
            whole *= 10;
        }
        if (whole > static_cast<std::uint64_t>(no_max))
            return false;
        std::int64_t r = negative ? -static_cast<std::int64_t>(whole) : static_cast<std::int64_t>(whole);
        if (nonzero_fraction && (round_up != negative))   // ceil of a positive or floor of a negative moves away from zero
            r += negative ? -1 : 1;
        out = r;
        return true;
    }

    enum class kind : std::uint8_t { invalid, ref, any_of, all_of, constant, enumeration, object, array, string, integer, number, boolean, null, any };

    struct classified
    {
        kind k = kind::invalid;
        std::string message{};
    };

    constexpr bool has_format(const value& s)
    {
        const value* f = s.get("format");
        if (f == nullptr || f->kind != json_kind::string)
            return false;
        const std::string_view v = f->text;
        return v == "date" || v == "time" || v == "date-time" || v == "uuid" || (v.size() == 5 && v.substr(0, 4) == "uuid" && v[4] >= '1' && v[4] <= '5');
    }

    // common_chat_schema_builder::build_node_impl's dispatch: which kind of schema this is
    constexpr classified classify(const value& s)
    {
        if (s.kind != json_kind::object)
            return {kind::invalid, "schema must be an object"};
        if (s.has("$ref"))
            return {kind::ref, {}};
        if (s.has("oneOf") || s.has("anyOf"))
            return {kind::any_of, {}};
        const value* type = s.get("type");
        if (type != nullptr && type->kind == json_kind::array)
            return type->items.empty() ? classified{kind::invalid, "type must not be empty"} : classified{kind::any_of, {}};
        if (s.has("const"))
            return {kind::constant, {}};
        if (s.has("enum"))
            return {kind::enumeration, {}};
        if (type != nullptr && type->kind != json_kind::string)
            return {kind::invalid, "type must be a string or an array of strings"};
        const std::string_view name = type != nullptr ? std::string_view(type->text) : std::string_view();
        const value* additional = s.get("additionalProperties");
        const bool has_properties = s.has("properties") || (additional != nullptr && !additional->is_true());
        if (name.empty()) {
            if (has_properties)
                return {kind::object, {}};
            if (s.has("allOf"))
                return {kind::all_of, {}};
            if (s.has("items") || s.has("prefixItems"))
                return {kind::array, {}};
            if (s.has("pattern") || s.has("minLength") || s.has("maxLength") || has_format(s))
                return {kind::string, {}};
            return {kind::any, {}};
        }
        if (name == "object")
            return {!has_properties && s.has("allOf") ? kind::all_of : kind::object, {}};
        if (name == "string")
            return {s.has("allOf") ? kind::all_of : kind::string, {}};
        if (name == "array")
            return {kind::array, {}};
        if (name == "integer")
            return {kind::integer, {}};
        if (name == "number")
            return {kind::number, {}};
        if (name == "boolean")
            return {kind::boolean, {}};
        if (name == "null")
            return {kind::null, {}};
        return {kind::invalid, "unrecognized type " + std::string(name)};
    }

    inline constexpr std::array<std::string_view, 14> unsupported_keywords{
        "pattern", "patternProperties", "propertyNames", "not", "if", "contains", "uniqueItems", "multipleOf",
        "minProperties", "maxProperties", "dependentRequired", "dependentSchemas", "unevaluatedProperties", "unevaluatedItems"};

    struct converter
    {
        const value& root;
        bool strict = true;
        std::vector<std::string> names{};   // the rules, in insertion order (sorted when printed)
        std::vector<std::string> bodies{};
        std::vector<std::string> resolving{};
        json_schema_error error{};
        bool failed = false;
        int depth = 0;

        constexpr std::string fail(const std::string& path, std::string message)
        {
            if (!failed) {
                failed = true;
                error = {0, path, std::move(message)};
            }
            return {};
        }

        constexpr const std::string* rule(std::string_view name) const
        {
            for (std::size_t i = 0; i < names.size(); ++i)
                if (names[i] == name)
                    return &bodies[i];
            return nullptr;
        }

        constexpr void set_rule(const std::string& name, const std::string& body)
        {
            for (std::size_t i = 0; i < names.size(); ++i)
                if (names[i] == name) {
                    bodies[i] = body;
                    return;
                }
            names.push_back(name);
            bodies.push_back(body);
        }

        // _add_rule: a rule under its escaped name, or the name with the first free number when another rule has it
        constexpr std::string add_rule(const std::string& name, const std::string& body)
        {
            const std::string esc = format_rule_name(name);
            const std::string* existing = rule(esc);
            if (existing == nullptr || *existing == body) {
                set_rule(esc, body);
                return esc;
            }
            int i = 0;
            while (true) {
                const std::string candidate = esc + decimal(i);
                const std::string* r = rule(candidate);
                if (r == nullptr || *r == body)
                    break;
                ++i;
            }
            const std::string key = esc + decimal(i);
            set_rule(key, body);
            return key;
        }

        constexpr std::string add_primitive(const std::string& name, const builtin& b)
        {
            std::string n = add_rule(name, std::string(b.content));
            for (const std::string_view dep : b.deps)
                if (!dep.empty() && rule(dep) == nullptr)
                    add_primitive(std::string(dep), find_builtin(dep));
            return n;
        }

        constexpr std::string primitive(std::string_view type) { return add_primitive(std::string(type), find_builtin(type)); }

        constexpr std::string visit_primitive(const std::string& rule_name, std::string_view type)
        {
            return add_primitive(rule_name == "root" ? "root" : std::string(type), find_builtin(type));
        }

        // a JSON pointer into the document; nullptr when it does not resolve
        constexpr const value* resolve(std::string_view ref) const
        {
            const value* target = &root;
            std::string_view rest = ref.substr(1);   // past '#'
            if (!rest.empty())
                rest.remove_prefix(1);               // past the first '/'
            while (true) {
                const std::size_t slash = rest.find('/');
                const std::string_view sel = rest.substr(0, slash);
                if (target->kind == json_kind::object && target->has(sel)) {
                    target = target->get(sel);
                } else if (target->kind == json_kind::array) {
                    std::size_t idx = 0;
                    bool number = !sel.empty();
                    for (const char c : sel) {
                        number = number && '0' <= c && c <= '9';
                        idx = idx * 10 + static_cast<std::size_t>(c - '0');
                    }
                    if (!number || idx >= target->items.size())
                        return nullptr;
                    target = &target->items[idx];
                } else {
                    return nullptr;
                }
                if (slash == std::string_view::npos)
                    return target;
                rest.remove_prefix(slash + 1);
            }
        }

        // the alternatives of an anyOf / oneOf, or of a list of types (each a copy of the schema with one type)
        constexpr std::vector<value> alternatives(const value& s, const std::string& path, std::string& alt_path)
        {
            std::vector<value> out;
            const value* type = s.get("type");
            if (!s.has("oneOf") && !s.has("anyOf") && type != nullptr && type->kind == json_kind::array) {
                for (const value& t : type->items) {
                    value alt = s;
                    for (std::size_t i = 0; i < alt.keys.size(); ++i)
                        if (alt.keys[i] == "type")
                            alt.items[i] = t;
                    out.push_back(std::move(alt));
                }
                alt_path = path + "/type";
                return out;
            }
            const std::string key = s.has("oneOf") ? "oneOf" : "anyOf";
            const value* alts = s.get(key);
            alt_path = path + "/" + key;
            if (alts->kind != json_kind::array) {
                fail(alt_path, key + " must be an array of schemas");
                return out;
            }
            if (alts->items.empty()) {
                fail(alt_path, key + " must not be empty");
                return out;
            }
            return alts->items;
        }

        constexpr bool count(const value& s, std::string_view key, const std::string& path, int fallback, int& out)
        {
            const value* v = s.get(key);
            out = fallback;
            if (v == nullptr)
                return true;
            if (!v->is_integer() || (!v->text.empty() && v->text[0] == '-') || v->text.size() > 9) {
                fail(path, std::string(key) + " must be a non-negative integer");
                return false;
            }
            out = static_cast<int>(to_int(v->text));
            return true;
        }

        constexpr bool integer_range(const value& s, const std::string& path, std::int64_t& lo, std::int64_t& hi)
        {
            lo = no_min;
            hi = no_max;
            const auto bound = [&](std::string_view key, bool round_up, std::int64_t offset, std::int64_t& out) {
                const value* v = s.get(key);
                std::int64_t b = 0;
                if (!integer_bound(*v, round_up, b)) {
                    fail(path, std::string(key) + " must be a number in the 64-bit integers");
                    return false;
                }
                out = b + offset;
                return true;
            };
            if (s.has("minimum")) {
                if (!bound("minimum", true, 0, lo))
                    return false;
            } else if (s.has("exclusiveMinimum")) {
                if (!bound("exclusiveMinimum", false, 1, lo))
                    return false;
            }
            if (s.has("maximum")) {
                if (!bound("maximum", false, 0, hi))
                    return false;
            } else if (s.has("exclusiveMaximum")) {
                if (!bound("exclusiveMaximum", true, -1, hi))
                    return false;
            }
            return true;
        }

        constexpr std::string integer_rule(std::int64_t lo, std::int64_t hi, const std::string& rule_name)
        {
            if (lo == no_min && hi == no_max)
                return visit_primitive(rule_name, "integer");
            std::string out = "(";
            build_min_max_int(out, lo, hi);
            out += ')';
            return add_rule(rule_name, out);
        }

        constexpr std::string resolve_ref(const value& s, const std::string& path)
        {
            const value* r = s.get("$ref");
            if (r->kind != json_kind::string)
                return fail(path, "$ref must be a string");
            const std::string& ref = r->text;
            if (ref.compare(0, 2, "#/") != 0 && ref != "#")
                return fail(path, "unsupported $ref " + ref + ", only references into the same document are supported");
            const std::size_t hash = ref.find('#');
            std::string ref_name = "ref" + format_rule_name(std::string_view(ref).substr(hash + 1));
            if (rule(ref_name) == nullptr && std::find(resolving.begin(), resolving.end(), ref) == resolving.end()) {
                const value* target = resolve(ref);
                if (target == nullptr)
                    return fail(path, "cannot resolve $ref " + ref);
                resolving.push_back(ref);
                ref_name = visit(*target, ref_name, ref);
                resolving.erase(std::find(resolving.begin(), resolving.end(), ref));
            }
            return ref_name;
        }

        // _not_strings: any JSON string but the given ones, through a trie of their code points
        constexpr std::string not_strings(const std::vector<std::string>& strings)
        {
            struct node
            {
                std::vector<std::pair<std::uint32_t, std::size_t>> children;   // sorted by code point (llama.cpp's std::map)
                bool pattern = false;
            };
            std::vector<node> trie(1);
            for (const auto& s : strings) {
                const auto cps = decode_utf8(s, {}).first;
                std::size_t cur = 0;
                for (const std::uint32_t c : cps) {
                    auto& ch = trie[cur].children;
                    auto it = std::lower_bound(ch.begin(), ch.end(), c, [](const auto& p, std::uint32_t v) { return p.first < v; });
                    if (it == ch.end() || it->first != c) {
                        const std::size_t child = trie.size();
                        it = ch.insert(it, {c, child});
                        trie.emplace_back();   // invalidates `ch` and `it`: not used after
                        cur = child;
                    } else {
                        cur = it->second;
                    }
                }
                trie[cur].pattern = true;
            }
            const std::string char_rule = primitive("char");
            std::string out = "[\"] ( ";
            bool ok = true;
            const auto walk = [&](const auto& self, std::size_t idx, int level) -> void {
                if (level >= recursion_limit) {
                    ok = false;
                    return;
                }
                std::string rejects;
                bool first = true;
                for (const auto& [cpt, child] : trie[idx].children) {
                    std::string c;
                    append_utf8(c, cpt);
                    rejects += c;
                    if (!first)
                        out += " | ";
                    first = false;
                    out += "[" + c + "]";
                    if (!trie[child].children.empty()) {
                        out += " (";
                        self(self, child, level + 1);
                        out += ")";
                    } else {
                        out += " " + char_rule + "+";
                    }
                }
                if (!trie[idx].children.empty())
                    out += " | [^\"" + rejects + "] " + char_rule + "*";
            };
            walk(walk, 0, 0);
            if (!ok)
                return fail("#", "a property name is longer than 128 characters");
            out += " )";
            if (!trie[0].pattern)
                out += "?";
            out += " [\"]";
            return out;
        }

        struct property
        {
            std::string name;
            value schema;
            std::string path;
        };

        enum class extra : std::uint8_t { none, any, schema };

        constexpr std::string build_object_rule(const std::vector<property>& properties, const std::vector<std::string>& required,
                                                const std::string& name, extra additional, const value* additional_schema,
                                                const std::string& path)
        {
            std::vector<std::string> required_props, optional_props, prop_names;
            std::vector<std::pair<std::string, std::string>> kv;   // property -> its kv rule
            const auto kv_rule = [&](const std::string& p) -> std::string {
                for (const auto& [k, v] : kv)
                    if (k == p)
                        return v;
                return {};
            };
            const std::string prefix = name + (name.empty() ? "" : "-");
            for (const auto& p : properties) {
                const std::string prop_rule = visit(p.schema, prefix + p.name, p.path);
                if (failed)
                    return {};
                std::string key;
                dump_string(key, p.name);
                kv.emplace_back(p.name, add_rule(prefix + p.name + "-kv", format_literal(key) + " space \":\" space " + prop_rule));
                if (std::find(required.begin(), required.end(), p.name) != required.end())
                    required_props.push_back(p.name);
                else
                    optional_props.push_back(p.name);
                prop_names.push_back(p.name);
            }
            if (additional != extra::none) {
                const std::string sub_name = prefix + "additional";
                const std::string value_rule = additional == extra::schema && classify(*additional_schema).k != kind::any
                                                   ? visit(*additional_schema, sub_name + "-value", path + "/additionalProperties")
                                                   : primitive("value");
                if (failed)
                    return {};
                const std::string key_rule = prop_names.empty() ? primitive("string") : add_rule(sub_name + "-k", not_strings(prop_names));
                kv.emplace_back("*", add_rule(sub_name + "-kv", key_rule + " \":\" space " + value_rule));
                optional_props.emplace_back("*");
            }
            if (required_props.empty() && optional_props.empty())
                return "\"{\" space \"}\"";
            std::string out = "\"{\" space ";
            for (std::size_t i = 0; i < required_props.size(); ++i) {
                if (i > 0)
                    out += " \",\" space ";
                out += kv_rule(required_props[i]);
            }
            if (!optional_props.empty()) {
                out += " (";
                if (!required_props.empty())
                    out += " \",\" space ( ";
                const auto rest = [&](const auto& self, std::size_t from, bool first_is_optional) -> std::string {
                    if (from >= optional_props.size())
                        return {};
                    const std::string& k = optional_props[from];
                    const std::string kv_name = kv_rule(k);
                    const std::string comma_ref = "( \",\" space " + kv_name + " )";
                    std::string res = first_is_optional ? comma_ref + (k == "*" ? "*" : "?") : kv_name + (k == "*" ? " " + comma_ref + "*" : "");
                    if (from + 1 < optional_props.size())
                        res += " " + add_rule(prefix + k + "-rest", self(self, from + 1, true));
                    return res;
                };
                for (std::size_t i = 0; i < optional_props.size(); ++i) {
                    if (i > 0)
                        out += " | ";
                    out += rest(rest, i, false);
                }
                if (!required_props.empty())
                    out += " )";
                out += " )?";
            }
            out += " space \"}\"";
            return out;
        }

        constexpr std::string visit_object(const value& s, const std::string& name, const std::string& rule_name, const std::string& path)
        {
            std::vector<std::string> required;
            if (const value* r = s.get("required"); r != nullptr && r->kind == json_kind::array)
                for (const value& n : r->items)
                    if (n.kind == json_kind::string)
                        required.push_back(n.text);
            std::vector<property> properties;
            if (const value* props = s.get("properties")) {
                if (props->kind != json_kind::object)
                    return fail(path, "properties must be an object");
                for (std::size_t i = 0; i < props->keys.size(); ++i)
                    properties.push_back({props->keys[i], props->items[i], path + "/properties/" + props->keys[i]});
            }
            extra additional = extra::none;
            const value* additional_schema = s.get("additionalProperties");
            if (additional_schema != nullptr) {
                if (additional_schema->kind == json_kind::boolean)
                    additional = additional_schema->boolean ? extra::any : extra::none;
                else if (additional_schema->kind == json_kind::object)
                    additional = extra::schema;
                else
                    return fail(path, "additionalProperties must be a boolean or a schema");
            } else if (!s.has("properties")) {
                additional = extra::any;   // {"type": "object"} on its own accepts any object
            }
            const bool any_extra = additional == extra::any || (additional == extra::schema && classify(*additional_schema).k == kind::any);
            if (properties.empty() && any_extra)
                return add_rule(rule_name, primitive("object"));
            const std::string body = build_object_rule(properties, required, name, additional, additional_schema, path);
            return failed ? std::string() : add_rule(rule_name, body);
        }

        constexpr std::string visit_all_of(const value& s, const std::string& name, const std::string& rule_name, const std::string& path)
        {
            const value* children = s.get("allOf");
            if (children->kind != json_kind::array)
                return fail(path + "/allOf", "allOf must be an array of schemas");
            if (children->items.empty())
                return fail(path + "/allOf", "allOf must not be empty");
            std::vector<std::string> required;
            std::vector<property> properties;
            std::vector<std::pair<std::string, std::size_t>> enum_values;   // sorted by literal (llama.cpp's std::map)
            std::vector<std::pair<value, std::string>> leaves;
            const auto add_component = [&](const auto& self, const value& comp, bool is_required, int level, const std::string& at) -> void {
                if (failed)
                    return;
                if (level >= recursion_limit) {
                    fail(at, "the schema is nested more than 128 deep");
                    return;
                }
                const auto c = classify(comp);
                if (c.k == kind::ref) {
                    const value* r = comp.get("$ref");
                    const value* target = r->kind == json_kind::string ? resolve(r->text) : nullptr;
                    if (target != nullptr)
                        self(self, *target, is_required, level + 1, r->text);
                } else if (c.k == kind::object) {
                    if (const value* props = comp.get("properties"); props != nullptr && props->kind == json_kind::object)
                        for (std::size_t i = 0; i < props->keys.size(); ++i) {
                            properties.push_back({props->keys[i], props->items[i], at + "/properties/" + props->keys[i]});
                            if (is_required)
                                required.push_back(props->keys[i]);
                        }
                } else if (c.k == kind::enumeration) {
                    const value* e = comp.get("enum");
                    if (e->kind == json_kind::array)
                        for (const value& v : e->items) {
                            const std::string lit = constant(v);
                            auto it = std::lower_bound(enum_values.begin(), enum_values.end(), lit,
                                                       [](const auto& p, const std::string& l) { return p.first < l; });
                            if (it != enum_values.end() && it->first == lit)
                                ++it->second;
                            else
                                enum_values.insert(it, {lit, 1});
                        }
                } else if (is_required && c.k != kind::any) {
                    leaves.emplace_back(comp, at);
                }
            };
            for (std::size_t i = 0; i < children->items.size(); ++i) {
                const value& child = children->items[i];
                const std::string at = path + "/allOf/" + decimal(static_cast<std::int64_t>(i));
                if (classify(child).k == kind::any_of) {
                    std::string alt_path;
                    for (const value& alt : alternatives(child, at, alt_path))
                        add_component(add_component, alt, false, 0, alt_path);
                } else {
                    add_component(add_component, child, true, 0, at);
                }
            }
            if (failed)
                return {};
            if (!enum_values.empty()) {
                std::vector<std::string> intersection;
                for (const auto& [lit, n] : enum_values)
                    if (n == children->items.size())
                        intersection.push_back(lit);
                if (!intersection.empty())
                    return add_rule(rule_name, "(" + join(intersection, " | ") + ")");
            }
            if (properties.empty() && !leaves.empty()) {
                if (leaves.size() == 1)
                    return visit(leaves[0].first, name, leaves[0].second);
                bool all_numeric = true, any_integer = false;
                std::int64_t lo = no_min, hi = no_max;
                for (const auto& [leaf, at] : leaves) {
                    const auto k = classify(leaf).k;
                    if (k == kind::integer) {
                        std::int64_t a = 0, b = 0;
                        if (!integer_range(leaf, at, a, b))
                            return {};
                        any_integer = true;
                        lo = std::max(lo, a);
                        hi = std::min(hi, b);
                    } else if (k != kind::number) {
                        all_numeric = false;
                    }
                }
                if (all_numeric) {
                    if (lo > hi)
                        return fail(path + "/allOf", "allOf has an empty numeric range");
                    if (any_integer)
                        return integer_rule(lo, hi, is_reserved(name) ? name + "-" : name.empty() ? "root" : name);
                    return visit(leaves[0].first, name, leaves[0].second);
                }
            }
            const std::string body = build_object_rule(properties, required, name, extra::none, nullptr, path);
            return failed ? std::string() : add_rule(rule_name, body);
        }

        constexpr std::string visit(const value& s, const std::string& name, const std::string& path)
        {
            if (failed)
                return {};
            if (++depth > recursion_limit)
                return fail(path, "the schema is nested more than 128 deep");
            std::string out = visit_impl(s, name, path);
            --depth;
            return out;
        }

        constexpr std::string visit_impl(const value& s, const std::string& name, const std::string& path)
        {
            const std::string rule_name = is_reserved(name) ? name + "-" : name.empty() ? "root" : name;
            const std::string sub_name = name + (name.empty() ? "" : "-");
            const auto c = classify(s);
            if (c.k == kind::invalid)
                return fail(path, c.message);
            for (const std::string_view k : unsupported_keywords)
                if (strict && s.has(k))
                    return fail(path, "unsupported keyword " + std::string(k));
            switch (c.k) {
            case kind::ref: {
                const std::string target = resolve_ref(s, path);
                return failed ? std::string() : add_rule(rule_name, target);
            }
            case kind::any_of: {
                std::string alt_path;
                const auto alts = alternatives(s, path, alt_path);
                std::vector<std::string> rules;
                for (std::size_t i = 0; i < alts.size(); ++i) {
                    const std::string index = decimal(static_cast<std::int64_t>(i));
                    rules.push_back(visit(alts[i], name + (name.empty() ? "alternative-" : "-") + index, alt_path + "/" + index));
                }
                return failed ? std::string() : add_rule(rule_name, join(rules, " | "));
            }
            case kind::all_of: return visit_all_of(s, name, rule_name, path);
            case kind::constant: return add_rule(rule_name, constant(*s.get("const")));
            case kind::enumeration: {
                const value* e = s.get("enum");
                if (e->kind != json_kind::array || e->items.empty())
                    return fail(path, "enum must be a non-empty array");
                std::vector<std::string> values;
                for (const value& v : e->items)
                    values.push_back(constant(v));
                return add_rule(rule_name, "(" + join(values, " | ") + ")");
            }
            case kind::object: return visit_object(s, name, rule_name, path);
            case kind::array: {
                if (const value* items = s.get(s.has("items") ? "items" : "prefixItems"); items != nullptr && items->kind == json_kind::array) {
                    const std::string key = s.has("items") ? "items" : "prefixItems";
                    std::string rule = "\"[\" space ";
                    for (std::size_t i = 0; i < items->items.size(); ++i) {
                        if (i > 0)
                            rule += " \",\" space ";
                        const std::string index = decimal(static_cast<std::int64_t>(i));
                        rule += visit(items->items[i], sub_name + "tuple-" + index, path + "/" + key + "/" + index);
                    }
                    rule += " space \"]\"";
                    return failed ? std::string() : add_rule(rule_name, rule);
                }
                const value* items = s.get(s.has("items") ? "items" : "prefixItems");
                int min_items = 0, max_items = -1;
                if (!count(s, "minItems", path, 0, min_items) || !count(s, "maxItems", path, -1, max_items))
                    return {};
                if ((items == nullptr || classify(*items).k == kind::any) && min_items == 0 && max_items < 0)
                    return visit_primitive(rule_name, "array");
                const value any_item{json_kind::object, false, {}, {}, {}};   // no items: any value, as {}
                const std::string item_rule = visit(items == nullptr ? any_item : *items, sub_name + "item",
                                                    path + (s.has("items") ? "/items" : "/prefixItems"));
                if (failed)
                    return {};
                return add_rule(rule_name, "\"[\" space " + build_repetition(item_rule, min_items, max_items < 0 ? no_max_items : max_items, "\",\" space")
                                               + " space \"]\"");
            }
            case kind::string: {
                if (s.has("pattern"))   // only when not strict: llama.cpp's fallback for a pattern it cannot convert
                    return add_rule(rule_name, primitive("string"));
                const value* f = s.get("format");
                const std::string_view format = f != nullptr && f->kind == json_kind::string && has_format(s) ? std::string_view(f->text) : "";
                if (!format.empty() && format.substr(0, 4) == "uuid")
                    return visit_primitive(rule_name, "uuid");
                if (!format.empty())
                    return add_rule(rule_name, primitive(std::string(format) + "-string"));
                int min_length = 0, max_length = -1;
                if (!count(s, "minLength", path, 0, min_length) || !count(s, "maxLength", path, -1, max_length))
                    return {};
                if (min_length > 0 || max_length >= 0) {
                    const std::string char_rule = primitive("char");
                    return add_rule(rule_name, "\"\\\"\" " + build_repetition(char_rule, min_length, max_length < 0 ? no_max_items : max_length)
                                                   + " \"\\\"\"");
                }
                return visit_primitive(rule_name, "string");
            }
            case kind::integer: {
                std::int64_t lo = 0, hi = 0;
                if (!integer_range(s, path, lo, hi))
                    return {};
                return integer_rule(lo, hi, rule_name);
            }
            case kind::number:
                for (const std::string_view k : {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"})
                    if (strict && s.has(k))
                        return fail(path, "unsupported keyword " + std::string(k) + " on a number");
                return visit_primitive(rule_name, "number");
            case kind::boolean: return visit_primitive(rule_name, "boolean");
            case kind::null: return visit_primitive(rule_name, "null");
            case kind::any: return add_rule(rule_name, primitive("value"));
            case kind::invalid: break;
            }
            return {};
        }

        // format_grammar: the rules sorted by name, one a line
        constexpr std::string format_grammar() const
        {
            std::vector<std::size_t> order(names.size());
            for (std::size_t i = 0; i < order.size(); ++i)
                order[i] = i;
            std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return names[a] < names[b]; });
            std::string out;
            for (const std::size_t i : order)
                out += names[i] + " ::= " + bodies[i] + "\n";
            return out;
        }
    };
} // namespace detail::json_schema

/// \brief Converts a JSON schema to GBNF, as llama.cpp's `json_schema_to_grammar` writes it.
/// \param schema The schema's JSON text.
/// \param options Strict (the default) or lenient as llama.cpp.
/// \return The grammar's text (rules sorted by name, root `root`), or what could not be converted.
[[nodiscard]] constexpr result<std::string, json_schema_error> json_schema_to_gbnf(std::string_view schema, json_schema_options options = {})
{
    namespace js = detail::json_schema;
    js::value root;
    js::reader r{schema};
    if (!r.parse(root, 0))
        return err(json_schema_error{r.error_at, {}, std::string(r.error)});
    r.space();
    if (r.pos != schema.size()) {
        r.fail("trailing characters after the schema");
        return err(json_schema_error{r.error_at, {}, std::string(r.error)});
    }
    js::converter c{root, options.strict};
    c.set_rule("space", std::string(js::space_rule));
    c.visit(root, "", "#");
    if (c.failed)
        return err(std::move(c.error));
    return ok(c.format_grammar());
}

/// \brief Converts a JSON schema straight to a compiled grammar.
/// \param schema The schema's JSON text.
/// \param options Strict (the default) or lenient as llama.cpp.
/// \return The grammar, or the conversion's error (a parse failure of the generated text, which the tests rule out,
/// is reported with path `#` and the parser's message).
[[nodiscard]] constexpr result<grammar, json_schema_error> json_schema_grammar(std::string_view schema, json_schema_options options = {})
{
    auto text = json_schema_to_gbnf(schema, options);
    if (text.is_err())
        return err(std::move(text).unwrap_err());
    auto g = parse_grammar(*text);
    if (g.is_err())
        return err(json_schema_error{g.unwrap_err().position, "#", std::string(g.unwrap_err().message)});
    return ok(std::move(*g));
}

namespace detail::json_schema
{
    constexpr detail::grammar::shape measure(std::string_view schema)
    {
        const auto g = json_schema_grammar(schema);
        if (g.is_err())
            return {};
        return {true, (*g).elements.size(), (*g).rules.size()};
    }

    // By value, so the view points into this call's copy (see detail::grammar::measure_fixed)
    template <std::size_t N>
    constexpr detail::grammar::shape measure_fixed(basic_fixed_string<char, N> schema)
    {
        return measure(schema.view());
    }
} // namespace detail::json_schema

/// \brief Compiles a JSON schema written in the program into a grammar's fixed-size tables, at compile time: a schema
/// that does not convert does not compile.
/// \tparam Schema The schema's JSON text, a string literal.
/// \return The compiled grammar, element for element `json_schema_grammar`'s.
template <basic_fixed_string Schema>
    requires std::same_as<typename decltype(Schema)::value_type, char>
consteval auto compile_json_schema()
{
    constexpr auto s = detail::json_schema::measure_fixed(Schema);
    static_assert(s.ok, "snicholls::ml::compile_json_schema: the schema does not convert; json_schema_to_gbnf gives the keyword and path");
    fixed_grammar<s.elements, s.rules> out{};
    if (s.ok) {
        const auto schema = Schema;   // a local copy (see measure_fixed)
        const auto parsed = json_schema_grammar(schema.view());
        const auto& g = *parsed;
        for (std::size_t i = 0; i < s.elements; ++i)
            out.elements[i] = g.elements[i];
        for (std::size_t i = 0; i < s.rules; ++i)
            out.rules[i] = g.rules[i];
        out.root = g.root;
    }
    return out;
}

} // namespace snicholls::ml

#endif // snicholls_core_ml_json_schema_hpp
