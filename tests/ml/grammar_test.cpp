//
//  grammar_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/ml/grammar.hpp>: the parser's element tables against llama.cpp's
//  tests/test-grammar-parser.cpp, the strings llama.cpp's tests/test-grammar-integration.cpp accepts and rejects
//  (both at commit 39039aa, MIT; the grammars and strings are copied from them), the example grammars
//  grammars/json.gbnf and grammars/arithmetic.gbnf, parse errors and their positions, the compile-time grammar equal
//  to the run-time one element for element, matching in a static_assert, and the token mask over a small vocabulary
//  (UTF-8 split across tokens, end of generation, agreement with accept_token).
//

#include <snicholls/core/ml/grammar.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace ml = snicholls::ml;
using type = ml::grammar_element_type;

// grammars/json.gbnf at llama.cpp 39039aa, verbatim
constexpr std::string_view json_gbnf = R"g(root   ::= object
value  ::= object | array | string | number | ("true" | "false" | "null") ws

object ::=
  "{" ws (
            string ":" ws value
    ("," ws string ":" ws value)*
  )? "}" ws

array  ::=
  "[" ws (
            value
    ("," ws value)*
  )? "]" ws

string ::=
  "\"" (
    [^"\\\x7F\x00-\x1F] |
    "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4}) # escapes
  )* "\"" ws

number ::= ("-"? ([0-9] | [1-9] [0-9]{0,15})) ("." [0-9]+)? ([eE] [-+]? [0-9] [1-9]{0,15})? ws

# Optional space: by convention, applied in this grammar after literal chars when allowed
ws ::= | " " | "\n" [ \t]{0,20}
)g";

// grammars/arithmetic.gbnf at llama.cpp 39039aa, verbatim
constexpr std::string_view arithmetic_gbnf = R"g(root  ::= (expr "=" ws term "\n")+
expr  ::= term ([-+*/] term)*
term  ::= ident | num | "(" ws expr ")" ws
ident ::= [a-z] [a-z0-9_]* ws
num   ::= [0-9]+ ws
ws    ::= [ \t\n]*
)g";

ml::grammar parse(std::string_view text, std::string_view root = "root")
{
    auto g = ml::parse_grammar(text, root);
    EXPECT_TRUE(g.is_ok()) << text << "\nerror at " << (g.is_err() ? g.unwrap_err().position : 0) << ": "
                           << (g.is_err() ? g.unwrap_err().message : "");
    return g.is_ok() ? std::move(*g) : ml::grammar{};
}

// test-grammar-parser.cpp's verify_parsing: the symbols and the rules' elements, in order
void verify_parsing(std::string_view text, const std::vector<std::pair<std::string, std::uint32_t>>& symbols,
                    const std::vector<ml::grammar_element>& elements)
{
    const auto g = parse(text);
    ASSERT_EQ(g.names.size(), symbols.size()) << text;
    for (const auto& [name, id] : symbols) {
        ASSERT_LT(id, g.names.size());
        EXPECT_EQ(g.names[id], name) << text;
    }
    EXPECT_EQ(g.elements, elements) << text;
}

// test-grammar-integration.cpp's token(): a 0xff marker and a big-endian id
std::string token(std::uint32_t id)
{
    return std::string{static_cast<char>(0xff), static_cast<char>((id >> 24) & 0xff), static_cast<char>((id >> 16) & 0xff),
                       static_cast<char>((id >> 8) & 0xff), static_cast<char>(id & 0xff)};
}

// test-grammar-integration.cpp's parse_tokens + match_string: a token per marker, a token (id 0) per code point
struct piece
{
    std::uint32_t id;
    std::string text;
};

std::vector<piece> pieces(const std::string& input)
{
    std::vector<piece> out;
    for (std::size_t i = 0; i < input.size();) {
        const auto b = static_cast<unsigned char>(input[i]);
        if (b == 0xff) {
            const std::uint32_t v = (std::uint32_t(static_cast<unsigned char>(input[i + 1])) << 24)
                                    | (std::uint32_t(static_cast<unsigned char>(input[i + 2])) << 16)
                                    | (std::uint32_t(static_cast<unsigned char>(input[i + 3])) << 8)
                                    | std::uint32_t(static_cast<unsigned char>(input[i + 4]));
            out.push_back({v, "<[" + std::to_string(v) + "]>"});
            i += 5;
        } else {
            const std::size_t n = b < 0x80 ? 1 : b < 0xe0 ? 2 : b < 0xf0 ? 3 : 4;
            out.push_back({0, input.substr(i, n)});
            i += n;
        }
    }
    return out;
}

// match_string, and the mask agrees with accept_token at every step
bool match_string(const ml::grammar& g, const std::string& input)
{
    ml::grammar_matcher m(g);
    for (const auto& p : pieces(input)) {
        const bool allowed = m.allows(p.id, p.text);
        const bool accepted = m.accept_token(p.id, p.text);
        EXPECT_EQ(allowed, accepted) << "mask and accept_token disagree on '" << p.text << "' in '" << input << "'";
        if (!accepted)
            return false;
    }
    return m.complete();
}

void test_grammar(std::string_view text, const std::vector<std::string>& passing, const std::vector<std::string>& failing)
{
    const auto g = parse(text);
    for (const auto& s : passing)
        EXPECT_TRUE(match_string(g, s)) << "should match: '" << s << "'\n" << text;
    for (const auto& s : failing)
        EXPECT_FALSE(match_string(g, s)) << "should not match: '" << s << "'\n" << text;
}

ml::grammar_error error_of(std::string_view text, std::string_view root = "root")
{
    const auto g = ml::parse_grammar(text, root);
    EXPECT_TRUE(g.is_err()) << text;
    return g.is_err() ? g.unwrap_err() : ml::grammar_error{};
}

// ---- compile time

constexpr auto tiny = ml::compile_grammar<R"g(root ::= ("a" | "bc")+ [0-9]? # comment
)g">();
static_assert(ml::grammar_accepts(tiny, "abca") && ml::grammar_accepts(tiny, "bc7") && ml::grammar_accepts(tiny, "a"));
static_assert(!ml::grammar_accepts(tiny, "") && !ml::grammar_accepts(tiny, "b") && !ml::grammar_accepts(tiny, "a77"));

constexpr auto arithmetic = ml::compile_grammar<R"g(root  ::= (expr "=" ws term "\n")+
expr  ::= term ([-+*/] term)*
term  ::= ident | num | "(" ws expr ")" ws
ident ::= [a-z] [a-z0-9_]* ws
num   ::= [0-9]+ ws
ws    ::= [ \t\n]*
)g">();
static_assert(arithmetic.rules.size() == 13 && arithmetic.root == 0);
static_assert(ml::grammar_accepts(arithmetic, "a=1\n") && !ml::grammar_accepts(arithmetic, "a=\n"));

constexpr auto custom_root = ml::compile_grammar<R"g(foobar ::= "foo" "bar")g", "foobar">();
static_assert(ml::grammar_accepts(custom_root, "foobar") && !ml::grammar_accepts(custom_root, "foo"));

// UTF-8 in a class at compile time: one code point, not its bytes
constexpr auto hiragana = ml::compile_grammar<"root ::= [ぁ-ゟ]+">();
static_assert(ml::grammar_accepts(hiragana, "ひらがな") && !ml::grammar_accepts(hiragana, "カ"));

// Parse errors and their positions at compile time
static_assert(ml::parse_grammar("root ::= [a").is_err() && ml::parse_grammar("root ::= [a").unwrap_err().position == 11);
static_assert(ml::parse_grammar(R"g(root ::= "a" | root "a")g").unwrap_err().message == "left recursion");
static_assert(ml::parse_grammar("root ::= x").unwrap_err() == ml::grammar_error{9, "undefined rule"});

static_assert(sizeof(ml::grammar_element) == 8 && std::is_trivially_copyable_v<ml::grammar_element>);

// ---- the parser, against test-grammar-parser.cpp

TEST(GrammarParse, RepetitionsAsLlamaCpp)
{
    verify_parsing(R"g(
        root ::= "a"{0,5000}
    )g",
                   {{"root", 0}, {"root_1", 1}},
                   {{type::rule_ref, 1}, {type::end, 0}, {type::chr, 'a'}, {type::rule_ref, 1}, {type::alt, 0}, {type::end, 0}});
    verify_parsing(R"g(
        root ::= "a"{3,5000}
    )g",
                   {{"root", 0}, {"root_1", 1}},
                   {{type::chr, 'a'}, {type::chr, 'a'}, {type::chr, 'a'}, {type::rule_ref, 1}, {type::end, 0},
                    {type::chr, 'a'}, {type::rule_ref, 1}, {type::alt, 0}, {type::end, 0}});
    verify_parsing(R"g(
        root  ::= "a"
    )g",
                   {{"root", 0}}, {{type::chr, 'a'}, {type::end, 0}});
    verify_parsing(R"g(
        root  ::= "a" | [bdx-z] | [^1-3]
    )g",
                   {{"root", 0}},
                   {{type::chr, 'a'}, {type::alt, 0}, {type::chr, 'b'}, {type::chr_alt, 'd'}, {type::chr_alt, 'x'},
                    {type::chr_range_upper, 'z'}, {type::alt, 0}, {type::chr_not, '1'}, {type::chr_range_upper, '3'}, {type::end, 0}});
    verify_parsing(R"g(
        root  ::= a+
        a     ::= "a"
    )g",
                   {{"a", 1}, {"root", 0}, {"root_2", 2}},
                   {{type::rule_ref, 1}, {type::rule_ref, 2}, {type::end, 0}, {type::chr, 'a'}, {type::end, 0},
                    {type::rule_ref, 1}, {type::rule_ref, 2}, {type::alt, 0}, {type::end, 0}});
    verify_parsing(R"g(
        root  ::= "a"+
    )g",
                   {{"root", 0}, {"root_1", 1}},
                   {{type::chr, 'a'}, {type::rule_ref, 1}, {type::end, 0}, {type::chr, 'a'}, {type::rule_ref, 1}, {type::alt, 0}, {type::end, 0}});
    verify_parsing(R"g(
        root  ::= a?
        a     ::= "a"
    )g",
                   {{"a", 1}, {"root", 0}, {"root_2", 2}},
                   {{type::rule_ref, 2}, {type::end, 0}, {type::chr, 'a'}, {type::end, 0}, {type::rule_ref, 1}, {type::alt, 0}, {type::end, 0}});
    verify_parsing(R"g(
        root  ::= "a"?
    )g",
                   {{"root", 0}, {"root_1", 1}},
                   {{type::rule_ref, 1}, {type::end, 0}, {type::chr, 'a'}, {type::alt, 0}, {type::end, 0}});
    verify_parsing(R"g(
        root  ::= a*
        a     ::= "a"
    )g",
                   {{"a", 1}, {"root", 0}, {"root_2", 2}},
                   {{type::rule_ref, 2}, {type::end, 0}, {type::chr, 'a'}, {type::end, 0}, {type::rule_ref, 1}, {type::rule_ref, 2},
                    {type::alt, 0}, {type::end, 0}});
    verify_parsing(R"g(
        root  ::= "a"{2}
    )g",
                   {{"root", 0}}, {{type::chr, 'a'}, {type::chr, 'a'}, {type::end, 0}});
    verify_parsing(R"g(
        root  ::= "a"{2,}
    )g",
                   {{"root", 0}, {"root_1", 1}},
                   {{type::chr, 'a'}, {type::chr, 'a'}, {type::rule_ref, 1}, {type::end, 0}, {type::chr, 'a'}, {type::rule_ref, 1},
                    {type::alt, 0}, {type::end, 0}});
    verify_parsing(R"g(
        root  ::= "a"{ 4}
    )g",
                   {{"root", 0}}, {{type::chr, 'a'}, {type::chr, 'a'}, {type::chr, 'a'}, {type::chr, 'a'}, {type::end, 0}});
    verify_parsing(R"g(
        root  ::= "a"{2,4}
    )g",
                   {{"root", 0}, {"root_1", 1}, {"root_2", 2}},
                   {{type::chr, 'a'}, {type::chr, 'a'}, {type::rule_ref, 2}, {type::end, 0}, {type::chr, 'a'}, {type::alt, 0}, {type::end, 0},
                    {type::chr, 'a'}, {type::rule_ref, 1}, {type::alt, 0}, {type::end, 0}});
}

TEST(GrammarParse, GroupsAndClassesAsLlamaCpp)
{
    verify_parsing(R"g(
        root  ::= (expr "=" term "\n")+
        expr  ::= term ([-+*/] term)*
        term  ::= [0-9]+
    )g",
                   {{"expr", 2}, {"expr_5", 5}, {"expr_6", 6}, {"root", 0}, {"root_1", 1}, {"root_4", 4}, {"term", 3}, {"term_7", 7}},
                   {// root
                    {type::rule_ref, 1}, {type::rule_ref, 4}, {type::end, 0},
                    // root_1
                    {type::rule_ref, 2}, {type::chr, '='}, {type::rule_ref, 3}, {type::chr, '\n'}, {type::end, 0},
                    // expr
                    {type::rule_ref, 3}, {type::rule_ref, 6}, {type::end, 0},
                    // term
                    {type::chr, '0'}, {type::chr_range_upper, '9'}, {type::rule_ref, 7}, {type::end, 0},
                    // root_4
                    {type::rule_ref, 1}, {type::rule_ref, 4}, {type::alt, 0}, {type::end, 0},
                    // expr_5
                    {type::chr, '-'}, {type::chr_alt, '+'}, {type::chr_alt, '*'}, {type::chr_alt, '/'}, {type::rule_ref, 3}, {type::end, 0},
                    // expr_6
                    {type::rule_ref, 5}, {type::rule_ref, 6}, {type::alt, 0}, {type::end, 0},
                    // term_7
                    {type::chr, '0'}, {type::chr_range_upper, '9'}, {type::rule_ref, 7}, {type::alt, 0}, {type::end, 0}});
}

TEST(GrammarParse, ArithmeticExampleAsLlamaCpp)
{
    // test-grammar-parser.cpp's second expression grammar: arithmetic.gbnf without its first rule's leading spaces
    verify_parsing(R"g(
        root  ::= (expr "=" ws term "\n")+
        expr  ::= term ([-+*/] term)*
        term  ::= ident | num | "(" ws expr ")" ws
        ident ::= [a-z] [a-z0-9_]* ws
        num   ::= [0-9]+ ws
        ws    ::= [ \t\n]*
    )g",
                   {{"expr", 2}, {"expr_6", 6}, {"expr_7", 7}, {"ident", 8}, {"ident_10", 10}, {"num", 9}, {"num_11", 11}, {"root", 0},
                    {"root_1", 1}, {"root_5", 5}, {"term", 4}, {"ws", 3}, {"ws_12", 12}},
                   {{type::rule_ref, 1}, {type::rule_ref, 5}, {type::end, 0},
                    {type::rule_ref, 2}, {type::chr, '='}, {type::rule_ref, 3}, {type::rule_ref, 4}, {type::chr, '\n'}, {type::end, 0},
                    {type::rule_ref, 4}, {type::rule_ref, 7}, {type::end, 0},
                    {type::rule_ref, 12}, {type::end, 0},
                    {type::rule_ref, 8}, {type::alt, 0}, {type::rule_ref, 9}, {type::alt, 0}, {type::chr, '('}, {type::rule_ref, 3},
                    {type::rule_ref, 2}, {type::chr, ')'}, {type::rule_ref, 3}, {type::end, 0},
                    {type::rule_ref, 1}, {type::rule_ref, 5}, {type::alt, 0}, {type::end, 0},
                    {type::chr, '-'}, {type::chr_alt, '+'}, {type::chr_alt, '*'}, {type::chr_alt, '/'}, {type::rule_ref, 4}, {type::end, 0},
                    {type::rule_ref, 6}, {type::rule_ref, 7}, {type::alt, 0}, {type::end, 0},
                    {type::chr, 'a'}, {type::chr_range_upper, 'z'}, {type::rule_ref, 10}, {type::rule_ref, 3}, {type::end, 0},
                    {type::chr, '0'}, {type::chr_range_upper, '9'}, {type::rule_ref, 11}, {type::rule_ref, 3}, {type::end, 0},
                    {type::chr, 'a'}, {type::chr_range_upper, 'z'}, {type::chr_alt, '0'}, {type::chr_range_upper, '9'}, {type::chr_alt, '_'},
                    {type::rule_ref, 10}, {type::alt, 0}, {type::end, 0},
                    {type::chr, '0'}, {type::chr_range_upper, '9'}, {type::rule_ref, 11}, {type::alt, 0}, {type::end, 0},
                    {type::chr, ' '}, {type::chr_alt, '\t'}, {type::chr_alt, '\n'}, {type::rule_ref, 12}, {type::alt, 0}, {type::end, 0}});
}

TEST(GrammarParse, TokensAsLlamaCpp)
{
    verify_parsing(R"g(
        root  ::= <[1000]> !<[1001]> <[1001]>
    )g",
                   {{"root", 0}}, {{type::token, 1000}, {type::token_not, 1001}, {type::token, 1001}, {type::end, 0}});
}

TEST(GrammarParse, EscapesAndUtf8)
{
    const auto g = parse(R"g(root ::= "\x41\u00e9\U0001F600\t\r\n\\\"" [\[\]\-] "é")g");
    EXPECT_EQ(g.elements, (std::vector<ml::grammar_element>{{type::chr, 'A'}, {type::chr, 0xe9}, {type::chr, 0x1f600}, {type::chr, '\t'},
                                                             {type::chr, '\r'}, {type::chr, '\n'}, {type::chr, '\\'}, {type::chr, '"'},
                                                             {type::chr, '['}, {type::chr_alt, ']'}, {type::chr_alt, '-'},
                                                             {type::chr, 0xe9}, {type::end, 0}}));
    EXPECT_EQ(parse("root ::= .").elements, (std::vector<ml::grammar_element>{{type::chr_any, 0}, {type::end, 0}}));
    EXPECT_EQ(parse("root ::= ").elements, (std::vector<ml::grammar_element>{{type::end, 0}}));   // the empty sentence
}

TEST(GrammarParse, TheCompileTimeGrammarIsTheRunTimeOneElementForElement)
{
    const auto g = parse(arithmetic_gbnf);
    ASSERT_EQ(g.elements.size(), arithmetic.elements.size());
    ASSERT_EQ(g.rules.size(), arithmetic.rules.size());
    EXPECT_TRUE(std::equal(g.elements.begin(), g.elements.end(), arithmetic.elements.begin()));
    EXPECT_TRUE(std::equal(g.rules.begin(), g.rules.end(), arithmetic.rules.begin()));
    EXPECT_EQ(g.root, arithmetic.root);
    constexpr auto json = ml::compile_grammar<R"g(root   ::= object
value  ::= object | array | string | number | ("true" | "false" | "null") ws

object ::=
  "{" ws (
            string ":" ws value
    ("," ws string ":" ws value)*
  )? "}" ws

array  ::=
  "[" ws (
            value
    ("," ws value)*
  )? "]" ws

string ::=
  "\"" (
    [^"\\\x7F\x00-\x1F] |
    "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4}) # escapes
  )* "\"" ws

number ::= ("-"? ([0-9] | [1-9] [0-9]{0,15})) ("." [0-9]+)? ([eE] [-+]? [0-9] [1-9]{0,15})? ws

# Optional space: by convention, applied in this grammar after literal chars when allowed
ws ::= | " " | "\n" [ \t]{0,20}
)g">();
    const auto j = parse(json_gbnf);
    ASSERT_EQ(j.elements.size(), json.elements.size());
    EXPECT_TRUE(std::equal(j.elements.begin(), j.elements.end(), json.elements.begin()));
    EXPECT_TRUE(std::equal(j.rules.begin(), j.rules.end(), json.rules.begin()));
    EXPECT_TRUE(ml::grammar_accepts(json, R"g({"a": [1, 2.5e3, true, null, "x\u00e9"]})g"));
    EXPECT_EQ(ml::grammar_view(json.view()).rule(0).size(), 2u);   // root ::= object
}

// ---- parse errors (test-grammar-parser.cpp's verify_failure, test-grammar-integration.cpp's failures)

TEST(GrammarErrors, FailAsLlamaCppAtTheirPositions)
{
    EXPECT_EQ(error_of(R"g(root ::= "a"{,}")g").message, "expecting an integer");
    EXPECT_EQ(error_of(R"g(root ::= "a"{,}")g").position, 13u);
    EXPECT_EQ(error_of(R"g(root ::= "a"{,10}")g").position, 13u);
    EXPECT_EQ(error_of(R"g(root ::= (((((([^x]*){0,99}){0,99}){0,99}){0,99}){0,99}){0,99})g").message,
              "the repetition generates more than 2000 rules");
    EXPECT_EQ(error_of(R"g(root ::= "a"{5000})g").message, "the repetition count exceeds 2000");
    EXPECT_EQ(error_of(R"g(root ::= "a"{5000,})g").message, "the repetition count exceeds 2000");
    EXPECT_EQ(error_of(R"g(root ::= "a"{5000,6000})g").message, "the repetition count exceeds 2000");
    EXPECT_EQ(error_of(R"g(root ::= "a"{3,2})g").message, "a repetition's maximum is below its minimum");
    std::string deep = "root ::= ";
    for (int i = 0; i < 300; ++i)
        deep += "(";
    deep += "\"x\"";
    for (int i = 0; i < 300; ++i)
        deep += ")";
    EXPECT_EQ(error_of(deep).message, "grammar nesting is deeper than 256");
    EXPECT_EQ(error_of(deep).position, 9u + 256u);

    EXPECT_EQ(error_of("root ::= \"abc").message, "unexpected end of input");
    EXPECT_EQ(error_of("root ::= \"\\q\"").message, "unknown escape");
    EXPECT_EQ(error_of("root ::= \"\\x4\"").message, "expecting hexadecimal digits");
    EXPECT_EQ(error_of("root := \"a\"").message, "expecting ::=");
    EXPECT_EQ(error_of("root ::= \"a\" )").message, "expecting a newline or the end");
    EXPECT_EQ(error_of("root ::= (\"a\"").message, "expecting ')'");
    EXPECT_EQ(error_of("root ::= *").message, "expecting an item before */+/?/{");
    EXPECT_EQ(error_of("root ::= <think>").message, "named tokens need a vocabulary: write <[id]>");
    EXPECT_EQ(error_of("root ::= <[99999999999]>").message, "token id is too big");
    EXPECT_EQ(error_of("").message, "the grammar has no rules");
}

TEST(GrammarErrors, MissingRootAndReferences)
{
    // test_failure_missing_root: no root rule
    const std::string missing_root = R"g(
        rot ::= expr
        expr ::= term ("+" term)*
        term ::= number
        number ::= [0-9]+)g";
    EXPECT_EQ(error_of(missing_root), (ml::grammar_error{missing_root.size(), "the grammar has no root rule"}));
    // test_failure_missing_reference: at the first reference
    const std::string missing_ref = R"g(root ::= expr
        expr ::= term ("+" term)*
        term ::= numero
        number ::= [0-9]+)g";
    EXPECT_EQ(error_of(missing_ref), (ml::grammar_error{missing_ref.find("numero"), "undefined rule"}));
    // test_failure_missing_root_symbol and test_custom_root_symbol_check
    EXPECT_TRUE(ml::parse_grammar(R"g(root ::= "foobar")g", "nonexistent").is_err());
    EXPECT_TRUE(ml::parse_grammar(R"g(foobar ::= "foobar")g", "root").is_err());
    EXPECT_TRUE(ml::parse_grammar(R"g(foobar ::= "foobar")g", "foobar").is_ok());
}

TEST(GrammarErrors, LeftRecursionIsRejected)
{
    EXPECT_EQ(error_of(R"g(root ::= "a" | root "a")g").message, "left recursion");
    EXPECT_EQ(error_of(R"g(
        root ::= asdf
        asdf ::= "a" | asdf "a"
        )g").message,
              "left recursion");
    EXPECT_EQ(error_of(R"g(
        root ::= asdf
        asdf ::= "a" | foo "b"
        foo ::= "c" | asdf "d" | "e")g").message,
              "left recursion");
    EXPECT_EQ(error_of(R"g(
        root ::= asdf
        asdf ::= "a" | foo "b"
        foo ::= "c" | empty asdf "d" | "e"
        empty ::= "blah" | )g").message,
              "left recursion");
    const auto chain = [](int n) {
        std::string s = "root ::= r0\n";
        for (int i = 0; i < n; i++)
            s += "r" + std::to_string(i) + " ::= r" + std::to_string(i + 1) + "\n";
        return s + "r" + std::to_string(n) + " ::= \"a\"\n";
    };
    EXPECT_TRUE(ml::parse_grammar(chain(100)).is_ok());
    EXPECT_EQ(error_of(chain(1000)).message, "rules are chained more than 256 deep");
    EXPECT_TRUE(ml::parse_grammar(R"g(root ::= "a" root | "")g").is_ok());   // right recursion is fine
}

// ---- matching, against test-grammar-integration.cpp

TEST(GrammarMatch, SimpleGrammarsAsLlamaCpp)
{
    test_grammar(R"g(root ::= "ab" | "abc" | "abd" | "abe")g", {"ab", "abc", "abd", "abe"}, {"", "a", "abb", "abcc"});
    test_grammar(R"g(root ::= "abc" | "ab" | "abd" | "abe")g", {"ab", "abc", "abd", "abe"}, {"", "a", "abb", "abcc"});
    test_grammar("root ::= \"ok\" | choice\nchoice ::= a | b | c | d\na ::= \"x\" a\nb ::= \"x\" b\nc ::= \"x\" c\nd ::= \"x\" d",
                 {"ok"}, {"", "x", "xx", "okx"});
    test_grammar(R"g(
            root ::= expr
            expr ::= term ("+" term)*
            term ::= number
            number ::= [0-9]+)g",
                 {"42", "1+2+3+4+5", "123+456"}, {"+", "/ 3", "1+2+3+4+5+", "12a45"});
}

TEST(GrammarMatch, TokensAsLlamaCpp)
{
    test_grammar(R"g(
            root ::= <[10]> content <[11]>
            content ::= (!<[11]>)*)g",
                 {token(10) + "hello world" + token(11),
                  token(10) + "text with " + token(12) + " other tokens " + token(13) + " mixed in" + token(11), token(10) + token(11),
                  token(10) + token(12) + token(13) + token(14) + token(15) + token(11), token(10) + "a" + token(11)},
                 {token(10) + "missing end token", token(10), "missing start token" + token(11), token(10) + token(11) + token(11),
                  token(11) + "wrong order" + token(10)});
    test_grammar(R"g(
            root ::= reasoning+ content tool-call*
            reasoning ::= <[10]> (!<[11]>)* <[11]>
            content ::= <[20]> (!<[21]>)* <[21]>
            tool-call ::= <[12]> name <[13]> args <[14]>
            name ::= (!<[13]>)+
            args ::= (!<[14]>)*)g",
                 {token(10) + "I am thinking" + token(11) + token(20) + "hello world!" + token(21) + token(12) + "search" + token(13)
                      + "query=test" + token(14),
                  token(10) + "reasoning 1" + token(11) + token(10) + "reasoning 2" + token(11) + token(20) + token(21) + token(12) + "tool"
                      + token(13) + token(14),
                  token(10) + token(11) + token(20) + "content" + token(21)},
                 {token(20) + "content only" + token(21), token(10) + "no closing reasoning",
                  token(10) + token(11) + token(20) + "no closing content",
                  token(10) + token(11) + token(20) + token(21) + token(12) + "incomplete tool",
                  token(10) + token(11) + token(11) + token(20) + token(21)});
}

TEST(GrammarMatch, MediumComplexityAsLlamaCpp)
{
    test_grammar(R"g(
            root ::= expression
            expression ::= term ws (("+"|"-") ws term)*
            term ::= factor ws (("*"|"/") ws factor)*
            factor ::= number | variable | "(" expression ")" | function-call
            number ::= [0-9]+
            variable ::= [a-zA-Z_][a-zA-Z0-9_]*
            function-call ::= variable ws "(" (expression ("," ws expression)*)? ")"
            ws ::= [ \t\n\r]?)g",
                 {"42", "1*2*3*4*5", "x", "x+10", "x1+y2", "(a+b)*(c-d)", "func()", "func(x,y+2)", "a*(b+c)-d/e", "f(g(x),h(y,z))",
                  "x + 10", "x1 + y2", "(a + b) * (c - d)", "func(x, y + 2)", "a * (b + c) - d / e", "f(g(x), h(y, z))", "123+456",
                  "123*456*789-123/456+789*123",
                  "123+456*789-123/456+789*123-456/789+123*456-789/123+456*789-123/456+789*123-456"},
                 {"+", "/ 3x", "x + + y", "a * / b", "func(,)", "func(x y)", "(a + b", "x + y)", "a + b * (c - d", "42 +", "x +",
                  "x + 10 +", "(a + b) * (c - d", "func(", "func(x, y + 2", "a * (b + c) - d /", "f(g(x), h(y, z)",
                  "123+456*789-123/456+789*123-456/789+123*456-789/123+456*789-123/456+789*123-456/"});
}

TEST(GrammarMatch, SpecialCharactersAsLlamaCpp)
{
    test_grammar(R"g(
            root ::= ... "abc" ...
            )g",
                 {"abcabcabc", "aaaabcccc", "🔵🟠✅abc❌🟠🔵"},
                 {"aaabcccc", "aaaaabcccc", "aaaabccc", "aaaabccccc", "🔵🟠✅❌abc❌✅🟠🔵", "🔵🟠abc🟠🔵"});
    test_grammar(R"g(root ::= [ab] "x" | [ac] "x" | [ad] "x" | [ae] "x")g", {"ax", "bx", "cx", "dx", "ex"}, {"", "a", "fx", "axx"});
    test_grammar(R"g(root ::= [a-b] "x" | [a-c] "x" | [a-d] "x" | [a-e] "x")g", {"ax", "bx", "cx", "dx", "ex"}, {"", "a", "fx", "axx"});
    test_grammar("root ::= a | b | c | d\na ::= \"p\" [a] \"x\"\nb ::= \"p\" [ab] \"x\"\nc ::= \"p\" [ac] \"x\"\nd ::= \"p\" [ad] \"x\"",
                 {"pax", "pbx", "pcx", "pdx"}, {"", "px", "pex", "paxx"});
}

TEST(GrammarMatch, QuantifiersAsLlamaCpp)
{
    test_grammar(R"g(root ::= "a"*)g", {"", "a", "aaaaa", "aaaaaaaaaaaaaaaaaa", std::string(62, 'a')},
                 {"b", "ab", "aab", "ba", std::string(61, 'a') + "b"});
    test_grammar(R"g(root ::= "a"+)g", {"a", "aaaaa", "aaaaaaaaaaaaaaaaaa", std::string(62, 'a')},
                 {"", "b", "ab", "aab", "ba", std::string(61, 'a') + "b"});
    test_grammar(R"g(root ::= "a"?)g", {"", "a"}, {"b", "ab", "aa", "ba"});
    test_grammar(R"g(
            root ::= cons+ vowel* cons? (vowel cons)*
            vowel ::= [aeiouy]
            cons ::= [bcdfghjklmnpqrstvwxyz]
            )g",
                 {"yes", "no", "noyes", "crwth", "four", "bryyyy"}, {"yess", "yesno", "forty", "catyyy"});
    test_grammar(R"g(
            root ::= [ab]{4}
        )g",
                 {"aaaa", "bbbb", "abab"}, {"a", "b", "aaaaa"});
    test_grammar(R"g(
            root ::= [ab]{4,}
        )g",
                 {"aaaa", "aaaaab", "bbbb", "ababab"}, {"", "aba"});
    test_grammar(R"g(
            root ::= [ab]{0,4}
        )g",
                 {"", "a", "aa", "aaa", "aaab"}, {"aaaaa"});
    test_grammar(R"g(
            root ::= ("0x" [A-F0-9]{2} " "?){3,5}
        )g",
                 {"0xFF 0x12 0xAB", "0xFF 0x12 0xAB 0x00 0x00"}, {"", "0xFF", "0xFF 0x12", "0xFF 0x12 0xAB 0x00 0x00 0x00"});
    test_grammar(R"g(
            root ::= ( [x]* )*
        )g",
                 {"", "x", "xx"}, {"y", "yy"});
}

TEST(GrammarMatch, ExampleGrammars)
{
    // json.gbnf and arithmetic.gbnf; the strings are this test's own
    test_grammar(json_gbnf,
                 {R"g({})g", R"g({"a": 1})g", R"g({"key": [1, -2, 3.25, 1e5, "s", {"n": null}], "t": true, "f": false})g",
                  "{\"x\":\n  \"\\u00e9\\n\"}", R"g({"é": "ü"})g"},
                 {"", "[]", R"g({"a"})g", R"g({"a": 01x})g", R"g({'a': 1})g", R"g({"a": [1,]})g", "{\"a\": \"\x01\"}", R"g({"a": tru})g"});
    test_grammar(arithmetic_gbnf, {"a=1\n", "x = (y)\nz=4\n", "y+2*3 = (z)\n", "foo_1 = 42\n"}, {"", "a=1", "a=b+1\n", "a==1\n", "A=1\n"});
}

TEST(GrammarMatch, AcceptStringCarriesPartialUtf8)
{
    const auto g = parse(R"g(root ::= "é" "x")g");
    ml::grammar_matcher m(g);
    EXPECT_TRUE(m.accept_string("\xc3"));
    EXPECT_EQ(m.utf8_state(), (ml::grammar_utf8_state{0x03, 1}));
    EXPECT_FALSE(m.complete());
    EXPECT_TRUE(m.accept_string("\xa9x"));
    EXPECT_TRUE(m.complete());
    EXPECT_FALSE(m.accept_string("y"));
    EXPECT_TRUE(m.failed());
    EXPECT_FALSE(ml::grammar_accepts(parse(R"g(root ::= "é")g"), "\xc3"));   // an unfinished sequence is no sentence
}

TEST(GrammarMatch, DecodeUtf8AsLlamaCpp)
{
    auto [a, sa] = ml::decode_utf8_partial("a\xc3\xa9\xe2\x82", {});
    EXPECT_EQ(a, (std::vector<std::uint32_t>{'a', 0xe9}));
    EXPECT_EQ(sa, (ml::grammar_utf8_state{0x82, 1}));
    auto [b, sb] = ml::decode_utf8_partial("\xac!", sa);   // completes the euro sign
    EXPECT_EQ(b, (std::vector<std::uint32_t>{0x20ac, '!'}));
    EXPECT_EQ(sb.remaining, 0);
    EXPECT_EQ(ml::decode_utf8_partial("\xa9", {}).second.remaining, -1);         // a continuation byte first: invalid
    EXPECT_TRUE(ml::decode_utf8_partial("x\xa9", {}).first.empty());            // and the whole piece is invalid
    EXPECT_EQ(ml::decode_utf8_partial("a", {0x03, 1}).second.remaining, -1);    // not the continuation expected
    EXPECT_EQ(ml::decode_utf8_partial("", {}).first.size(), 0u);
}

// ---- the token mask

TEST(GrammarMask, Utf8SplitAcrossTokensAndEndOfGeneration)
{
    const auto g = parse(R"g(root ::= "é" | "ab")g");
    const std::array<std::string_view, 9> vocab{"a", "b", "ab", "\xc3", "\xa9", "\xc3\xa9", "x", "", "</s>"};
    const std::array<std::uint32_t, 1> eos{8};
    std::array<bool, 9> allowed{};
    ml::grammar_matcher m(g);
    m.token_mask(vocab, eos, allowed);
    EXPECT_EQ(allowed, (std::array<bool, 9>{true, false, true, true, false, true, false, false, false}));

    // After the first byte of é only its continuation may follow
    ASSERT_TRUE(m.accept_token(3, vocab[3]));
    m.token_mask(vocab, eos, allowed);
    EXPECT_EQ(allowed, (std::array<bool, 9>{false, false, false, false, true, false, false, false, false}));
    ASSERT_TRUE(m.accept_token(4, vocab[4]));
    m.token_mask(vocab, eos, allowed);
    EXPECT_EQ(allowed, (std::array<bool, 9>{false, false, false, false, false, false, false, false, true}));   // only the end
    EXPECT_TRUE(m.accept_end());

    // The same mask as bits, and applied to logits
    ml::grammar_matcher fresh(g);
    std::array<std::uint32_t, 1> bits{};
    fresh.token_bitmask(vocab, eos, bits);
    EXPECT_EQ(bits[0], 0b000101101u);
    std::array<float, 9> logits{1, 2, 3, 4, 5, 6, 7, 8, 9};
    ml::apply_token_bitmask(std::span<const std::uint32_t>(bits), std::span<float>(logits));
    EXPECT_EQ(logits[0], 1.0f);
    EXPECT_EQ(logits[1], -std::numeric_limits<float>::infinity());
    EXPECT_EQ(logits[5], 6.0f);
    EXPECT_EQ(logits[8], -std::numeric_limits<float>::infinity());
}

TEST(GrammarMask, PartialSequencesThatCannotSatisfyAClassAreMasked)
{
    // [a-z] cannot be completed by a two-byte lead; an overlong lead (0xc0, 0xc1) never; [^a] by any lead
    const auto lower = parse("root ::= [a-z]");
    ml::grammar_matcher m(lower);
    EXPECT_TRUE(m.allows(0, "q"));
    EXPECT_FALSE(m.allows(0, "\xc3"));
    const auto any_but_a = parse("root ::= [^a]");
    ml::grammar_matcher n(any_but_a);
    EXPECT_TRUE(n.allows(0, "\xc3"));
    EXPECT_FALSE(n.allows(0, "\xc1"));
    EXPECT_TRUE(n.allows(0, "\xf0\x9f"));   // a four-byte lead and one continuation
    const auto dot = parse("root ::= .");
    EXPECT_TRUE(ml::grammar_matcher(dot).allows(0, "\xe2"));
    // A range above the lead's span: \u4e00-\u9fff from 0xe4 (U+4000..U+4FFF) yes, from 0xe0 (U+0800..U+0FFF) no
    const auto cjk = parse("root ::= [\\u4e00-\\u9fff]");
    EXPECT_TRUE(ml::grammar_matcher(cjk).allows(0, "\xe4"));
    EXPECT_FALSE(ml::grammar_matcher(cjk).allows(0, "\xe0"));
}

TEST(GrammarMask, TheMaskAgreesWithAcceptTokenOverAVocabulary)
{
    // Every token the mask allows is accepted, and every token it masks is refused, along a JSON sentence
    const auto g = parse(json_gbnf);
    const std::array<std::string_view, 14> vocab{"{", "}", "\"", "a", "\": ", "1", "2.", "5", ", ", "[", "]", "tr", "ue", "\xe2\x82"};
    const std::array<std::uint32_t, 1> eos{99};
    const std::array<std::uint32_t, 11> sentence{0, 2, 3, 4, 9, 5, 8, 11, 12, 10, 1};   // {"a": [1, true]}
    ml::grammar_matcher m(g);
    for (const std::uint32_t next : sentence) {
        std::array<bool, 14> allowed{};
        m.token_mask(vocab, eos, allowed);
        for (std::uint32_t t = 0; t < vocab.size(); ++t) {
            ml::grammar_matcher probe = m;
            EXPECT_EQ(allowed[t], probe.accept_token(t, vocab[t])) << "token '" << vocab[t] << "' before " << next;
        }
        ASSERT_TRUE(allowed[next]);
        ASSERT_TRUE(m.accept_token(next, vocab[next]));
    }
    EXPECT_TRUE(m.complete());
    // With callables: piece by id, end token by predicate
    std::array<bool, 3> by_callable{};
    m.token_mask([](std::uint32_t t) { return t == 0 ? std::string_view(" ") : std::string_view("x"); },
                 [](std::uint32_t t) { return t == 2; }, std::span<bool>(by_callable));
    EXPECT_EQ(by_callable, (std::array<bool, 3>{true, false, true}));   // json.gbnf's trailing ws, then the end
}

TEST(GrammarMask, TokenTerminalsMatchByIdOnly)
{
    const auto g = parse("root ::= <[7]> \"x\"");
    ml::grammar_matcher m(g);
    EXPECT_TRUE(m.allows(7, "anything"));
    EXPECT_FALSE(m.allows(8, "<[7]>"));
    ASSERT_TRUE(m.accept_token(7, "anything"));
    EXPECT_TRUE(m.allows(0, "x"));
    EXPECT_FALSE(m.allows(0, "xx"));
}

} // namespace
