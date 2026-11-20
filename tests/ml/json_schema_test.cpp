//
//  json_schema_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/ml/json_schema.hpp>: the grammar written for every schema of llama.cpp's
//  tests/test-json-schema-to-grammar.cpp this subset supports, compared line for line with llama.cpp's expected
//  output; the strings llama.cpp's tests/test-grammar-integration.cpp has each schema's grammar accept and reject
//  (both at commit 39039aa, MIT: the schemas, grammars and strings below are copied from them); errors naming the
//  keyword and its path; the JSON reader; lenient mode; recursive references; and a schema compiled at compile time.
//

#include <snicholls/core/ml/json_schema.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace ml = snicholls::ml;

struct grammar_case
{
    bool success;
    const char* name;
    const char* schema;
    const char* expected;
};

struct string_case
{
    const char* name;
    bool strict;
    const char* schema;
    std::vector<std::string> passing;
    std::vector<std::string> failing;
};

// test-json-schema-to-grammar.cpp's trim: no surrounding space, no indentation
std::string trim(std::string_view source)
{
    std::string out;
    bool line_start = true;
    for (const char c : source) {
        if (line_start && (c == ' ' || c == '\t'))
            continue;
        line_start = c == '\n';
        out += c;
    }
    const auto first = out.find_first_not_of(" \n\r\t");
    if (first == std::string::npos)
        return {};
    const auto last = out.find_last_not_of(" \n\r\t");
    return out.substr(first, last + 1 - first);
}

// test-json-schema-to-grammar.cpp at llama.cpp 39039aa: every case this subset supports (all but the regular
// expressions), its name, schema and expected grammar copied verbatim (MIT)
const std::vector<grammar_case> llama_grammar_cases = {
    {true, "min 0",
     R"__({
            "type": "integer",
            "minimum": 0
        })__",
     R"__(
            root ::= ([0] | [1-9] [0-9]{0,15})
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )__"},
    {true, "min 1",
     R"__({
            "type": "integer",
            "minimum": 1
        })__",
     R"__(
            root ::= ([1-9] [0-9]{0,15})
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )__"},
    {true, "min 3",
     R"__({
            "type": "integer",
            "minimum": 3
        })__",
     R"__(
            root ::= ([1-2] [0-9]{1,15} | [3-9] [0-9]{0,15})
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )__"},
    {true, "min 9",
     R"__({
            "type": "integer",
            "minimum": 9
        })__",
     R"__(
            root ::= ([1-8] [0-9]{1,15} | [9] [0-9]{0,15})
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )__"},
    {true, "min 10",
     R"__({
            "type": "integer",
            "minimum": 10
        })__",
     R"__(
            root ::= ([1] ([0-9]{1,15}) | [2-9] [0-9]{1,15})
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )__"},
    {true, "min 25",
     R"__({
            "type": "integer",
            "minimum": 25
        })__",
     R"__(
            root ::= ([1] [0-9]{2,15} | [2] ([0-4] [0-9]{1,14} | [5-9] [0-9]{0,14}) | [3-9] [0-9]{1,15})
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )__"},
    {true, "max 30",
     R"__({
            "type": "integer",
            "maximum": 30
        })__",
     R"__(
            root ::= ("-" [1-9] [0-9]{0,15} | [0-9] | ([1-2] [0-9] | [3] "0"))
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )__"},
    {true, "min -5",
     R"__({
            "type": "integer",
            "minimum": -5
        })__",
     R"__(
            root ::= ("-" ([0-5]) | [0] | [1-9] [0-9]{0,15})
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )__"},
    {true, "min -123",
     R"__({
            "type": "integer",
            "minimum": -123
        })__",
     R"__(
            root ::= ("-" ([0-9] | ([1-8] [0-9] | [9] [0-9]) | "1" ([0-1] [0-9] | [2] [0-3])) | [0] | [1-9] [0-9]{0,15})
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )__"},
    {true, "max -5",
     R"__({
            "type": "integer",
            "maximum": -5
        })__",
     R"__(
            root ::= ("-" ([0-4] [0-9]{1,15} | [5-9] [0-9]{0,15}))
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )__"},
    {true, "max 1",
     R"__({
            "type": "integer",
            "maximum": 1
        })__",
     R"__(
            root ::= ("-" [1-9] [0-9]{0,15} | [0-1])
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )__"},
    {true, "max 100",
     R"__({
            "type": "integer",
            "maximum": 100
        })__",
     R"__(
            root ::= ("-" [1-9] [0-9]{0,15} | [0-9] | ([1-8] [0-9] | [9] [0-9]) | "100")
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )__"},
    {true, "min 0 max 23",
     R"__({
            "type": "integer",
            "minimum": 0,
            "maximum": 23
        })__",
     R"__(
            root ::= ([0-9] | ([1] [0-9] | [2] [0-3]))
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )__"},
    {true, "min 15 max 300",
     R"__({
            "type": "integer",
            "minimum": 15,
            "maximum": 300
        })__",
     R"__(
            root ::= (([1] ([5-9]) | [2-9] [0-9]) | ([1-2] [0-9]{2} | [3] "00"))
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )__"},
    {true, "min 5 max 30",
     R"__({
            "type": "integer",
            "minimum": 5,
            "maximum": 30
        })__",
     R"__(
            root ::= ([5-9] | ([1-2] [0-9] | [3] "0"))
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )__"},
    {true, "min -123 max 42",
     R"__({
            "type": "integer",
            "minimum": -123,
            "maximum": 42
        })__",
     R"__(
            root ::= ("-" ([0-9] | ([1-8] [0-9] | [9] [0-9]) | "1" ([0-1] [0-9] | [2] [0-3])) | [0-9] | ([1-3] [0-9] | [4] [0-2]))
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )__"},
    {true, "min -10 max 10",
     R"__({
            "type": "integer",
            "minimum": -10,
            "maximum": 10
        })__",
     R"__(
            root ::= ("-" ([0-9] | "10") | [0-9] | "10")
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )__"},
    {false, "unknown type",
     R"__({
            "type": "kaboom"
        })__",
     R"__()__"},
    {false, "invalid type",
     R"__({
            "type": 123
        })__",
     R"__()__"},
    {true, "empty schema (any value)",
     R"__({})__",
     R"__(
            array ::= "[" space ( value ("," space value)* )? space "]"
            boolean ::= ("true" | "false")
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            null ::= "null"
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?
            object ::= "{" space ( string ":" space value ("," space string ":" space value)* )? space "}"
            root ::= value
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\""
            value ::= object | array | string | number | boolean | null
        )__"},
    {true, "exotic formats",
     R"__({
            "items": [
                { "format": "date" },
                { "format": "uuid" },
                { "format": "time" },
                { "format": "date-time" }
            ]
        })__",
     R"__(
            date ::= [0-9]{4} "-" ( "0" [1-9] | "1" [0-2] ) "-" ( "0" [1-9] | [1-2] [0-9] | "3" [0-1] )
            date-string ::= "\"" date "\""
            date-time ::= date "T" time
            date-time-string ::= "\"" date-time "\""
            root ::= "[" space tuple-0 "," space uuid "," space tuple-2 "," space tuple-3 space "]"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            time ::= ([01] [0-9] | "2" [0-3]) ":" [0-5] [0-9] ":" [0-5] [0-9] ( "." [0-9]{3} )? ( "Z" | ( "+" | "-" ) ( [01] [0-9] | "2" [0-3] ) ":" [0-5] [0-9] )
            time-string ::= "\"" time "\""
            tuple-0 ::= date-string
            tuple-2 ::= time-string
            tuple-3 ::= date-time-string
            uuid ::= "\"" [0-9a-fA-F]{8} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{12} "\""
        )__"},
    {true, "string",
     R"__({
            "type": "string"
        })__",
     R"__(
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            root ::= "\"" char* "\""
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )__"},
    {true, "string w/ min length 1",
     R"__({
            "type": "string",
            "minLength": 1
        })__",
     R"__(
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            root ::= "\"" char+ "\""
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )__"},
    {true, "string w/ min length 3",
     R"__({
            "type": "string",
            "minLength": 3
        })__",
     R"__(
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            root ::= "\"" char{3,} "\""
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )__"},
    {true, "string w/ max length",
     R"__({
            "type": "string",
            "maxLength": 3
        })__",
     R"__(
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            root ::= "\"" char{0,3} "\""
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )__"},
    {true, "string w/ min & max length",
     R"__({
            "type": "string",
            "minLength": 1,
            "maxLength": 4
        })__",
     R"__(
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            root ::= "\"" char{1,4} "\""
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )__"},
    {true, "boolean",
     R"__({
            "type": "boolean"
        })__",
     R"__(
            root ::= ("true" | "false")
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )__"},
    {true, "integer",
     R"__({
            "type": "integer"
        })__",
     R"__(
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            root ::= ("-"? integral-part)
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )__"},
    {true, "string const",
     R"__({
            "const": "foo"
        })__",
     R"__(
            root ::= "\"foo\""
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )__"},
    {true, "non-string const",
     R"__({
            "const": 123
        })__",
     R"__(
            root ::= "123"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )__"},
    {true, "non-string enum",
     R"__({
            "enum": ["red", "amber", "green", null, 42, ["foo"]]
        })__",
     R"__(
            root ::= ("\"red\"" | "\"amber\"" | "\"green\"" | "null" | "42" | "[\"foo\"]")
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )__"},
    {true, "string array",
     R"__({
            "type": "array",
            "prefixItems": { "type": "string" }
        })__",
     R"__(
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            root ::= "[" space (string ("," space string)*)? space "]"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\""
        )__"},
    {true, "nullable string array",
     R"__({
            "type": ["array", "null"],
            "prefixItems": { "type": "string" }
        })__",
     R"__(
            alternative-0 ::= "[" space (string ("," space string)*)? space "]"
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            null ::= "null"
            root ::= alternative-0 | null
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\""
        )__"},
    {true, "tuple1",
     R"__({
            "prefixItems": [{ "type": "string" }]
        })__",
     R"__(
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            root ::= "[" space string space "]"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\""
        )__"},
    {true, "tuple2",
     R"__({
            "prefixItems": [{ "type": "string" }, { "type": "number" }]
        })__",
     R"__(
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?
            root ::= "[" space string "," space number space "]"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\""
        )__"},
    {true, "array with empty items",
     R"__({
            "type": "array",
            "items": {}
        })__",
     R"__(
            array ::= "[" space ( value ("," space value)* )? space "]"
            boolean ::= ("true" | "false")
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            null ::= "null"
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?
            object ::= "{" space ( string ":" space value ("," space string ":" space value)* )? space "}"
            root ::= "[" space ( value ("," space value)* )? space "]"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\""
            value ::= object | array | string | number | boolean | null
        )__"},
    {true, "array with empty items and prefixItems",
     R"__({
            "type": "array",
            "items": {},
            "prefixItems": { "type": "string" }
        })__",
     R"__(
            array ::= "[" space ( value ("," space value)* )? space "]"
            boolean ::= ("true" | "false")
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            null ::= "null"
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?
            object ::= "{" space ( string ":" space value ("," space string ":" space value)* )? space "}"
            root ::= "[" space ( value ("," space value)* )? space "]"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\""
            value ::= object | array | string | number | boolean | null
        )__"},
    {true, "number",
     R"__({
            "type": "number"
        })__",
     R"__(
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            root ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )__"},
    {true, "minItems",
     R"__({
            "items": {
                "type": "boolean"
            },
            "minItems": 2
        })__",
     R"__(
            boolean ::= ("true" | "false")
            root ::= "[" space boolean ("," space boolean)+ space "]"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )__"},
    {true, "maxItems 0",
     R"__({
            "items": {
                "type": "boolean"
            },
            "maxItems": 0
        })__",
     R"__(
            boolean ::= ("true" | "false")
            root ::= "[" space  space "]"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )__"},
    {true, "maxItems 1",
     R"__({
            "items": {
                "type": "boolean"
            },
            "maxItems": 1
        })__",
     R"__(
            boolean ::= ("true" | "false")
            root ::= "[" space boolean? space "]"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )__"},
    {true, "maxItems 2",
     R"__({
            "items": {
                "type": "boolean"
            },
            "maxItems": 2
        })__",
     R"__(
            boolean ::= ("true" | "false")
            root ::= "[" space (boolean ("," space boolean)?)? space "]"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )__"},
    {true, "min + maxItems",
     R"__({
            "items": {
                "type": ["number", "integer"]
            },
            "minItems": 3,
            "maxItems": 5
        })__",
     R"__(
            decimal-part ::= [0-9]{1,16}
            integer ::= ("-"? integral-part)
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            item ::= number | integer
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?
            root ::= "[" space item ("," space item){2,4} space "]"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )__"},
    {true, "min + max items with min + max values across zero",
     R"__({
            "items": {
                "type": "integer",
                "minimum": -12,
                "maximum": 207
            },
            "minItems": 3,
            "maxItems": 5
        })__",
     R"__(
            item ::= ("-" ([0-9] | "1" [0-2]) | [0-9] | ([1-8] [0-9] | [9] [0-9]) | ([1] [0-9]{2} | [2] "0" [0-7]))
            root ::= "[" space item ("," space item){2,4} space "]"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )__"},
    {true, "min + max items with min + max values",
     R"__({
            "items": {
                "type": "integer",
                "minimum": 12,
                "maximum": 207
            },
            "minItems": 3,
            "maxItems": 5
        })__",
     R"__(
            item ::= (([1] ([2-9]) | [2-9] [0-9]) | ([1] [0-9]{2} | [2] "0" [0-7]))
            root ::= "[" space item ("," space item){2,4} space "]"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )__"},
    {true, "required props in original order",
     R"__({
            "type": "object",
            "properties": {
                "b": {"type": "string"},
                "c": {"type": "string"},
                "a": {"type": "string"}
            },
            "required": [
                "a",
                "b",
                "c"
            ],
            "additionalProperties": false,
            "definitions": {}
        })__",
     R"__(
            a-kv ::= "\"a\"" space ":" space string
            b-kv ::= "\"b\"" space ":" space string
            c-kv ::= "\"c\"" space ":" space string
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            root ::= "{" space b-kv "," space c-kv "," space a-kv space "}"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\""
        )__"},
    {true, "1 optional prop",
     R"__({
            "properties": {
                "a": {
                "type": "string"
                }
            },
            "additionalProperties": false
        })__",
     R"__(
            a-kv ::= "\"a\"" space ":" space string
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            root ::= "{" space  (a-kv )? space "}"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\""
        )__"},
    {true, "N optional props",
     R"__({
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "string"},
                "c": {"type": "string"}
            },
            "additionalProperties": false
        })__",
     R"__(
            a-kv ::= "\"a\"" space ":" space string
            a-rest ::= ( "," space b-kv )? b-rest
            b-kv ::= "\"b\"" space ":" space string
            b-rest ::= ( "," space c-kv )?
            c-kv ::= "\"c\"" space ":" space string
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            root ::= "{" space  (a-kv a-rest | b-kv b-rest | c-kv )? space "}"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\""
        )__"},
    {true, "required + optional props each in original order",
     R"__({
            "properties": {
                "b": {"type": "string"},
                "a": {"type": "string"},
                "d": {"type": "string"},
                "c": {"type": "string"}
            },
            "required": ["a", "b"],
            "additionalProperties": false
        })__",
     R"__(
            a-kv ::= "\"a\"" space ":" space string
            b-kv ::= "\"b\"" space ":" space string
            c-kv ::= "\"c\"" space ":" space string
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            d-kv ::= "\"d\"" space ":" space string
            d-rest ::= ( "," space c-kv )?
            root ::= "{" space b-kv "," space a-kv ( "," space ( d-kv d-rest | c-kv ) )? space "}"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\""
        )__"},
    {true, "additional props",
     R"__({
            "type": "object",
            "additionalProperties": {"type": "array", "items": {"type": "number"}}
        })__",
     R"__(
            additional-kv ::= string ":" space additional-value
            additional-value ::= "[" space (number ("," space number)*)? space "]"
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?
            root ::= "{" space  (additional-kv ( "," space additional-kv )* )? space "}"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\""
        )__"},
    {true, "additional props (true)",
     R"__({
            "type": "object",
            "additionalProperties": true
        })__",
     R"__(
            array ::= "[" space ( value ("," space value)* )? space "]"
            boolean ::= ("true" | "false")
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            null ::= "null"
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?
            object ::= "{" space ( string ":" space value ("," space string ":" space value)* )? space "}"
            root ::= object
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\""
            value ::= object | array | string | number | boolean | null
        )__"},
    {true, "additional props (implicit)",
     R"__({
            "type": "object"
        })__",
     R"__(
            array ::= "[" space ( value ("," space value)* )? space "]"
            boolean ::= ("true" | "false")
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            null ::= "null"
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?
            object ::= "{" space ( string ":" space value ("," space string ":" space value)* )? space "}"
            root ::= object
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\""
            value ::= object | array | string | number | boolean | null
        )__"},
    {true, "empty w/o additional props",
     R"__({
            "type": "object",
            "additionalProperties": false
        })__",
     R"__(
            root ::= "{" space "}"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )__"},
    {true, "required + additional props",
     R"__({
            "type": "object",
            "properties": {
                "a": {"type": "number"}
            },
            "required": ["a"],
            "additionalProperties": {"type": "string"}
        })__",
     R"__(
            a-kv ::= "\"a\"" space ":" space number
            additional-k ::= ["] ( [a] char+ | [^"a] char* )? ["]
            additional-kv ::= additional-k ":" space string
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?
            root ::= "{" space a-kv ( "," space ( additional-kv ( "," space additional-kv )* ) )? space "}"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\""
        )__"},
    {true, "optional + additional props",
     R"__({
            "type": "object",
            "properties": {
                "a": {"type": "number"}
            },
            "additionalProperties": {"type": "number"}
        })__",
     R"__(
            a-kv ::= "\"a\"" space ":" space number
            a-rest ::= ( "," space additional-kv )*
            additional-k ::= ["] ( [a] char+ | [^"a] char* )? ["]
            additional-kv ::= additional-k ":" space number
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?
            root ::= "{" space  (a-kv a-rest | additional-kv ( "," space additional-kv )* )? space "}"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )__"},
    {true, "required + optional + additional props",
     R"__({
            "type": "object",
            "properties": {
                "and": {"type": "number"},
                "also": {"type": "number"}
            },
            "required": ["and"],
            "additionalProperties": {"type": "number"}
        })__",
     R"__(
            additional-k ::= ["] ( [a] ([l] ([s] ([o] char+ | [^"o] char*) | [^"s] char*) | [n] ([d] char+ | [^"d] char*) | [^"ln] char*) | [^"a] char* )? ["]
            additional-kv ::= additional-k ":" space number
            also-kv ::= "\"also\"" space ":" space number
            also-rest ::= ( "," space additional-kv )*
            and-kv ::= "\"and\"" space ":" space number
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?
            root ::= "{" space and-kv ( "," space ( also-kv also-rest | additional-kv ( "," space additional-kv )* ) )? space "}"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )__"},
    {true, "optional props with empty name",
     R"__({
            "properties": {
                "": {"type": "integer"},
                "a": {"type": "integer"}
            },
            "additionalProperties": {"type": "integer"}
        })__",
     R"__(
            -kv ::= "\"\"" space ":" space root
            -rest ::= ( "," space a-kv )? a-rest
            a-kv ::= "\"a\"" space ":" space integer
            a-rest ::= ( "," space additional-kv )*
            additional-k ::= ["] ( [a] char+ | [^"a] char* ) ["]
            additional-kv ::= additional-k ":" space integer
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            integer ::= ("-"? integral-part)
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            root ::= ("-"? integral-part)
            root0 ::= "{" space  (-kv -rest | a-kv a-rest | additional-kv ( "," space additional-kv )* )? space "}"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )__"},
    {true, "optional props with nested names",
     R"__({
            "properties": {
                "a": {"type": "integer"},
                "aa": {"type": "integer"}
            },
            "additionalProperties": {"type": "integer"}
        })__",
     R"__(
            a-kv ::= "\"a\"" space ":" space integer
            a-rest ::= ( "," space aa-kv )? aa-rest
            aa-kv ::= "\"aa\"" space ":" space integer
            aa-rest ::= ( "," space additional-kv )*
            additional-k ::= ["] ( [a] ([a] char+ | [^"a] char*) | [^"a] char* )? ["]
            additional-kv ::= additional-k ":" space integer
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            integer ::= ("-"? integral-part)
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            root ::= "{" space  (a-kv a-rest | aa-kv aa-rest | additional-kv ( "," space additional-kv )* )? space "}"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )__"},
    {true, "optional props with common prefix",
     R"__({
            "properties": {
                "ab": {"type": "integer"},
                "ac": {"type": "integer"}
            },
            "additionalProperties": {"type": "integer"}
        })__",
     R"__(
            ab-kv ::= "\"ab\"" space ":" space integer
            ab-rest ::= ( "," space ac-kv )? ac-rest
            ac-kv ::= "\"ac\"" space ":" space integer
            ac-rest ::= ( "," space additional-kv )*
            additional-k ::= ["] ( [a] ([b] char+ | [c] char+ | [^"bc] char*) | [^"a] char* )? ["]
            additional-kv ::= additional-k ":" space integer
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            integer ::= ("-"? integral-part)
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            root ::= "{" space  (ab-kv ab-rest | ac-kv ac-rest | additional-kv ( "," space additional-kv )* )? space "}"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )__"},
    {true, "top-level $ref",
     R"__({
            "$ref": "#/definitions/foo",
            "definitions": {
                "foo": {
                    "type": "object",
                    "properties": {
                        "a": {
                            "type": "string"
                        }
                    },
                    "required": [
                        "a"
                    ],
                    "additionalProperties": false
                }
            }
        })__",
     R"__(
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            ref-definitions-foo ::= "{" space ref-definitions-foo-a-kv space "}"
            ref-definitions-foo-a-kv ::= "\"a\"" space ":" space string
            root ::= ref-definitions-foo
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\""
        )__"},
    {true, "anyOf",
     R"__({
            "anyOf": [
                {"$ref": "#/definitions/foo"},
                {"$ref": "#/definitions/bar"}
            ],
            "definitions": {
                "foo": {
                    "properties": {"a": {"type": "number"}}
                },
                "bar": {
                    "properties": {"b": {"type": "number"}}
                }
            },
            "type": "object"
        })__",
     R"__(
            alternative-0 ::= ref-definitions-foo
            alternative-1 ::= ref-definitions-bar
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?
            ref-definitions-bar ::= "{" space  (ref-definitions-bar-b-kv )? space "}"
            ref-definitions-bar-b-kv ::= "\"b\"" space ":" space number
            ref-definitions-foo ::= "{" space  (ref-definitions-foo-a-kv )? space "}"
            ref-definitions-foo-a-kv ::= "\"a\"" space ":" space number
            root ::= alternative-0 | alternative-1
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )__"},
    {true, "anyOf $ref",
     R"__({
            "properties": {
                "a": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "number"}
                    ]
                },
                "b": {
                    "anyOf": [
                        {"$ref": "#/properties/a/anyOf/0"},
                        {"type": "boolean"}
                    ]
                }
            },
            "type": "object"
        })__",
     R"__(
            a ::= string | number
            a-kv ::= "\"a\"" space ":" space a
            a-rest ::= ( "," space b-kv )?
            b ::= b-0 | boolean
            b-0 ::= string
            b-kv ::= "\"b\"" space ":" space b
            boolean ::= ("true" | "false")
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?
            root ::= "{" space  (a-kv a-rest | b-kv )? space "}"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\""
        )__"},
    {true, "mix of allOf, anyOf and $ref (similar to https://json.schemastore.org/tsconfig.json)",
     R"__({
            "allOf": [
                {"$ref": "#/definitions/foo"},
                {"$ref": "#/definitions/bar"},
                {
                "anyOf": [
                    {"$ref": "#/definitions/baz"},
                    {"$ref": "#/definitions/bam"}
                ]
                }
            ],
            "definitions": {
                "foo": {
                    "properties": {"a": {"type": "number"}}
                },
                "bar": {
                    "properties": {"b": {"type": "number"}}
                },
                "bam": {
                    "properties": {"c": {"type": "number"}}
                },
                "baz": {
                    "properties": {"d": {"type": "number"}}
                }
            },
            "type": "object"
        })__",
     R"__(
            a-kv ::= "\"a\"" space ":" space number
            b-kv ::= "\"b\"" space ":" space number
            c-kv ::= "\"c\"" space ":" space number
            d-kv ::= "\"d\"" space ":" space number
            d-rest ::= ( "," space c-kv )?
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?
            root ::= "{" space a-kv "," space b-kv ( "," space ( d-kv d-rest | c-kv ) )? space "}"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )__"},
    {true, "allOf with enum schema",
     R"__({
            "allOf": [
                {"$ref": "#/definitions/foo"}
            ],
            "definitions": {
                "foo": {
                    "type": "string",
                    "enum": ["a", "b"]
                }
            }
        })__",
     R"__(
            root ::= ("\"a\"" | "\"b\"")
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )__"},
    {true, "allOf with multiple enum schemas",
     R"__({
            "allOf": [
                {"$ref": "#/definitions/foo"},
                {"$ref": "#/definitions/bar"}
            ],
            "definitions": {
                "foo": {
                    "type": "string",
                    "enum": ["a", "b", "c"]
                },
                "bar": {
                    "type": "string",
                    "enum": ["b", "c", "d"]
                }
            }
        })__",
     R"__(
            root ::= ("\"b\"" | "\"c\"")
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )__"},
    {false, "allOf with an empty integer range",
     R"__({
            "allOf": [
                {"type": "integer", "minimum": 9},
                {"type": "integer", "maximum": 5}
            ]
        })__",
     R"__()__"},
    {true, "allOf with plain integer and number schemas",
     R"__({
            "allOf": [
                {"type": "integer"},
                {"type": "number", "minimum": 0, "maximum": 100}
            ]
        })__",
     R"__(
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            root ::= ("-"? integral-part)
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )__"},
    {true, "allOf with a single integer schema",
     R"__({
            "allOf": [
                {"type": "integer"}
            ]
        })__",
     R"__(
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            root ::= ("-"? integral-part)
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )__"},
    {true, "allOf with bounded integer and number schemas",
     R"__({
            "allOf": [
                {"type": "integer", "minimum": 0, "maximum": 100},
                {"type": "number"}
            ]
        })__",
     R"__(
            root ::= ([0-9] | ([1-8] [0-9] | [9] [0-9]) | "100")
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )__"},
    {true, "conflicting names",
     R"__({
            "type": "object",
            "properties": {
                "number": {
                "type": "object",
                "properties": {
                    "number": {
                    "type": "object",
                        "properties": {
                            "root": {
                                "type": "number"
                            }
                        },
                        "required": [
                            "root"
                        ],
                        "additionalProperties": false
                    }
                },
                "required": [
                    "number"
                ],
                "additionalProperties": false
                }
            },
            "required": [
                "number"
            ],
            "additionalProperties": false,
            "definitions": {}
        })__",
     R"__(
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?
            number- ::= "{" space number-number-kv space "}"
            number-kv ::= "\"number\"" space ":" space number-
            number-number ::= "{" space number-number-root-kv space "}"
            number-number-kv ::= "\"number\"" space ":" space number-number
            number-number-root-kv ::= "\"root\"" space ":" space number
            root ::= "{" space number-kv space "}"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )__"},
    {true, "description only (no type) treated as unconstrained",
     R"__({"description": "The 0-based index of the last line to be retrieved (inclusive). If None, read until the end of the file."})__",
     R"__(
            array ::= "[" space ( value ("," space value)* )? space "]"
            boolean ::= ("true" | "false")
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            null ::= "null"
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?
            object ::= "{" space ( string ":" space value ("," space string ":" space value)* )? space "}"
            root ::= value
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\""
            value ::= object | array | string | number | boolean | null
        )__"},
    {true, "literal string with escapes",
     R"__({
            "properties": {
                "code": {
                    "const": " \r \n \" \\ ",
                    "description": "Generated code",
                    "title": "Code",
                    "type": "string"
                }
            },
            "required": [
                "code"
            ],
            "title": "DecoderResponse",
            "type": "object"
        })__",
     R"__(
            code ::= "\" \\r \\n \\\" \\\\ \""
            code-kv ::= "\"code\"" space ":" space code
            root ::= "{" space code-kv space "}"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )__"},
    {false, "allOf with a $ref cycle",
     R"__({
            "allOf": [{"$ref": "#/$defs/a"}],
            "$defs": {"a": {"$ref": "#/$defs/a"}}
        })__",
     R"__()__"},
};

// test-grammar-integration.cpp at llama.cpp 39039aa: the test_schema cases this subset supports, the strings each
// grammar must accept and reject copied verbatim (MIT); `strict` false where llama.cpp ignores a keyword
const std::vector<string_case> llama_string_cases = {
    {"min 0", true, R"__({
            "type": "integer",
            "minimum": 0
        })__",
     {R"__(0)__", R"__(10)__", R"__(12)__", R"__(10000)__"},
     {R"__(-1)__", R"__(-10)__", R"__(-10000)__", R"__(-100000000000000000000000000000000)__", R"__(100000000000000000000000000000000)__", R"__(00)__", R"__(01)__", R"__(-0)__"}},
    {"min 2", true, R"__({
            "type": "integer",
            "minimum": 2
        })__",
     {R"__(2)__", R"__(3)__", R"__(4)__", R"__(10)__", R"__(20)__", R"__(1234567890000000)__"},
     {R"__(0)__", R"__(1)__", R"__(-1)__", R"__(-100)__", R"__(0)__", R"__(1)__", R"__(01)__", R"__(02)__", R"__(12345678900000000)__"}},
    {"min 456", true, R"__({
            "type": "integer",
            "minimum": 456
        })__",
     {R"__(456)__", R"__(4560)__", R"__(457)__", R"__(460)__", R"__(500)__"},
     {R"__(455)__", R"__(356)__", R"__(50)__", R"__(050)__", R"__(-1)__", R"__(-456)__"}},
    {"min -123", true, R"__({
            "type": "integer",
            "minimum": -123
        })__",
     {R"__(-123)__", R"__(-122)__", R"__(-11)__", R"__(-1)__", R"__(0)__", R"__(1)__", R"__(123)__", R"__(1234)__", R"__(2345)__"},
     {R"__(-1234)__", R"__(-124)__"}},
    {"max 9999", true, R"__({
            "type": "integer",
            "maximum": 9999
        })__",
     {R"__(-99999)__", R"__(0)__", R"__(9999)__"},
     {R"__(10000)__", R"__(99991)__"}},
    {"max -9999", true, R"__({
            "type": "integer",
            "maximum": -9999
        })__",
     {R"__(-10000)__", R"__(-9999)__"},
     {R"__(-9998)__", R"__(0)__", R"__(9999)__"}},
    {"min 5 max 30", true, R"__({
            "type": "integer",
            "minimum": 5,
            "maximum": 30
        })__",
     {R"__(5)__", R"__(10)__", R"__(30)__"},
     {R"__(05)__", R"__(4)__", R"__(-1)__", R"__(31)__", R"__(123)__", R"__(0123)__"}},
    {"min 1 max 900719925474091", true, R"__({
            "type": "integer",
            "exclusiveMinimum": 0,
            "maximum": 900719925474091
        })__",
     {R"__(1)__", R"__(2)__", R"__(10)__", R"__(900719925474090)__", R"__(900719925474091)__"},
     {R"__(0)__", R"__(01)__", R"__(900719925474092)__", R"__(9007199254740910)__"}},
    {"min -1 max 1", true, R"__({
            "type": "integer",
            "minimum": -1,
            "maximum": 1
        })__",
     {R"__(-1)__", R"__(0)__", R"__(1)__"},
     {R"__(-11)__", R"__(-10)__", R"__(-2)__", R"__(2)__", R"__(10)__", R"__(11)__"}},
    {"min -123 max 42", true, R"__({
            "type": "integer",
            "minimum": -123,
            "maximum": 42
        })__",
     {R"__(-123)__", R"__(-122)__", R"__(-13)__", R"__(-11)__", R"__(-2)__", R"__(-1)__", R"__(0)__", R"__(1)__", R"__(5)__", R"__(10)__", R"__(39)__", R"__(40)__", R"__(42)__"},
     {R"__(-0123)__", R"__(-124)__", R"__(-1123)__", R"__(-200)__", R"__(43)__", R"__(123)__", R"__(0123)__"}},
    {"exclusive min / max", true, R"__({
            "type": "integer",
            "exclusiveMinimum": 0,
            "exclusiveMaximum": 10000
        })__",
     {R"__(1)__", R"__(9999)__"},
     {R"__(0)__", R"__(01)__", R"__(10000)__", R"__(99999)__"}},
    {"empty schema (any value)", true, R"__(
            {}
        )__",
     {R"__({})__", R"__({"foo": "bar"})__", R"__([])__", R"__(null)__", R"__("")__", R"__(true)__"},
     {R"__()__", R"__({"foo"})__", R"__(foo)__"}},
    {"exotic formats (list)", true, R"__({
            "items": [
                { "format": "date" },
                { "format": "uuid" },
                { "format": "time" },
                { "format": "date-time" }
            ]
        })__",
     {R"__(["2012-04-23", "12345678-1234-1234-1234-1234567890ab", "18:25:43.511Z", "2012-04-23T18:25:43.511Z"])__"},
     {R"__(["foo", "bar"])__", R"__(["12345678-1234-1234-1234-1234567890ab"])__"}},
    {"string", true, R"__({
            "type": "string"
        })__",
     {R"__("foo")__", R"__("bar")__", R"__("")__"},
     {R"__({})__", R"__("foo": "bar")__"}},
    {"string w/ min length 1", true, R"__({
            "type": "string",
            "minLength": 1
        })__",
     {R"__("foo")__", R"__("bar")__"},
     {R"__("")__", R"__({})__", R"__("foo": "bar")__"}},
    {"string w/ min length 3", true, R"__({
                "type": "string",
                "minLength": 3
        })__",
     {R"__("foo")__", R"__("bar")__", R"__("foobar")__"},
     {R"__("")__", R"__("f")__", R"__("fo")__"}},
    {"string w/ max length", true, R"__({
            "type": "string",
            "maxLength": 3
        })__",
     {R"__("foo")__", R"__("bar")__", R"__("")__", R"__("f")__", R"__("fo")__"},
     {R"__("foobar")__"}},
    {"string w/ min & max length", true, R"__({
            "type": "string",
            "minLength": 1,
            "maxLength": 4
        })__",
     {R"__("foo")__", R"__("bar")__", R"__("f")__", R"__("barf")__"},
     {R"__("")__", R"__("barfo")__", R"__("foobar")__"}},
    {"boolean", true, R"__({
            "type": "boolean"
        })__",
     {R"__(true)__", R"__(false)__"},
     {R"__("")__", R"__("true")__", R"__(True)__", R"__(FALSE)__"}},
    {"integer", true, R"__({
            "type": "integer"
        })__",
     {R"__(0)__", R"__(12345)__", R"__(1234567890123456)__"},
     {R"__()__", R"__(01)__", R"__(007)__", R"__(12345678901234567  )__"}},
    {"string const", true, R"__({
            "const": "foo"
        })__",
     {R"__("foo")__"},
     {R"__(foo)__", R"__("bar")__"}},
    {"non-string const", true, R"__({
            "const": true
        })__",
     {R"__(true)__"},
     {R"__()__", R"__(foo)__", R"__("true")__"}},
    {"non-string const", true, R"__({
            "enum": ["red", "amber", "green", null, 42, ["foo"]]
        })__",
     {R"__("red")__", R"__(null)__", R"__(42)__", R"__(["foo"])__"},
     {R"__()__", R"__(420)__", R"__(true)__", R"__(foo)__"}},
    {"", true, R"__(
            {
                "type": ["array", "null"],
                "items": { "type": "string" }
            }
        )__",
     {R"__(null)__", R"__([])__", R"__(["123"])__", R"__(["foo", "bar"])__"},
     {R"__()__", R"__([123])__", R"__("foo")__", R"__(["foo", 42])__"}},
    {"min+max items", true, R"__({
            "items": {
                "type": ["number", "integer"]
            },
            "minItems": 3,
            "maxItems": 5
        })__",
     {R"__([1, 2, 3])__", R"__([1, 2, 3, 4])__", R"__([1, 2, 3, 4, 5])__"},
     {R"__([1, 2])__", R"__([1, 2, 3, 4, 5, 6])__", R"__(1)__"}},
    {"object properties", true, R"__({
            "type": "object",
            "properties": {
                "number": { "type": "number" },
                "street_name": { "type": "string" },
                "street_type": { "enum": ["Street", "Avenue", "Boulevard"] }
            }
        })__",
     {R"__({ "number": 1600, "street_name": "Pennsylvania", "street_type":"Avenue"})__", R"__({ "street_name": "Pennsylvania" })__", R"__({ "number": 1600, "street_name": "Pennsylvania" })__", R"__({})__", R"__({ "number": 1600, "street_name": "Pennsylvania", "street_type": "Avenue" })__"},
     {R"__({ "number": "1600", "street_name": "Pennsylvania", "street_type":"Avenue"})__", R"__({ "street_name": "Pennsylvania", "number": 1600 })__", R"__({ "number": "1600", "street_name": "Pennsylvania", "street_type":"Avenue"})__", R"__({ "number": 1600, "street_name": "Pennsylvania", "street_type":"Avenue", "direction":"NW"})__"}},
    {"additional properties can't override other properties", true, R"__({
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"}
            },
            "additionalProperties": true
        })__",
     {R"__({"a": 42})__", R"__({"c": ""})__", R"__({"a": 42, "c": ""})__", R"__({"a_": ""})__"},
     {R"__()__", R"__({"a": ""})__", R"__({"a": "", "b": ""})__"}},
    {"object properties, additionalProperties: true", true, R"__({
            "type": "object",
            "properties": {
                "number": { "type": "number" },
                "street_name": { "type": "string" },
                "street_type": { "enum": ["Street", "Avenue", "Boulevard"] }
            },
            "additionalProperties": true
        })__",
     {R"__({})__", R"__({"number":1600,"street_name":"Pennsylvania","street_type":"Avenue"})__", R"__({ "street_name": "Pennsylvania" })__", R"__({ "number": 1600, "street_name": "Pennsylvania" })__", R"__({ "number": 1600, "street_name": "Pennsylvania", "street_type":"Avenue", "direction":"NW"})__", R"__({ "number": 1600, "street_name": "Pennsylvania", "street_type": "Avenue" })__"},
     {R"__({ "number": "1600", "street_name": "Pennsylvania", "street_type":"Avenue"})__", R"__({ "street_name": "Pennsylvania", "number": 1600, "street_type":"Avenue"})__"}},
    {"required + optional props each in original order", true, R"__({
            "type": "object",
            "properties": {
                "number": { "type": "number" },
                "street_name": { "type": "string" },
                "street_type": { "enum": ["Street", "Avenue", "Boulevard"] }
            },
            "additionalProperties": false
        })__",
     {R"__({ "street_name": "Pennsylvania" })__", R"__({ "number": 1600, "street_type":"Avenue"})__", R"__({ "number": 1600, "street_name": "Pennsylvania" })__", R"__({ "number": 1600, "street_name": "Pennsylvania", "street_type":"Avenue"})__", R"__({ "number": 1600, "street_name": "Pennsylvania", "street_type": "Avenue" })__"},
     {R"__({ "street_type": "Avenue", "number": 1600 })__", R"__({ "number": 1600, "street_name": "Pennsylvania", "street_type": "Avenue", "direction": "NW" })__"}},
    {"required + optional props each in original order", true, R"__({
            "properties": {
                "b": {"type": "string"},
                "a": {"type": "string"},
                "d": {"type": "string"},
                "c": {"type": "string"}
            },
            "required": ["a", "b"],
            "additionalProperties": false
        })__",
     {R"__({"b": "foo", "a": "bar"})__", R"__({"b":"foo","a":"bar","d":"qux"})__", R"__({"b":"foo", "a":"bar", "d":"qux", "c":"baz"})__"},
     {R"__({"a": "foo", "b": "bar"})__", R"__({"b": "bar"})__", R"__({"a": "foo", "c": "baz"})__", R"__({"a":"foo", "b":"bar", "c":"baz", "d":"qux"})__"}},
    {"required props", false, R"__({
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "https://example.com/product.schema.json",
            "title": "Product",
            "description": "A product from Acme's catalog",
            "type": "object",
            "properties": {
                "productId": {
                "description": "The unique identifier for a product",
                "type": "integer"
                },
                "productName": {
                "description": "Name of the product",
                "type": "string"
                },
                "price": {
                "description": "The price of the product",
                "type": "number",
                "exclusiveMinimum": 0
                },
                "tags": {
                "description": "Tags for the product",
                "type": "array",
                "items": {
                    "type": "string"
                },
                "minItems": 1,
                "uniqueItems": true
                },
                "dimensions": {
                "type": "object",
                "properties": {
                    "length": {
                    "type": "number"
                    },
                    "width": {
                    "type": "number"
                    },
                    "height": {
                    "type": "number"
                    }
                },
                "required": [ "length", "width", "height" ]
                }
            },
            "required": [ "productId", "productName", "price" ]
        })__",
     {R"__({"productId": 1, "productName": "A green door", "price": 12.50})__", R"__({"productId": 1, "productName": "A green door", "price": 12.50, "tags": ["home", "green"]})__", R"__({"productId": 1, "productName": "A green door", "price": 12.50, "tags": ["home", "green"], "dimensions": {"length": 785, "width": 250.5, "height": -0.359}})__"},
     {R"__({})__", R"__({"productName": "A green door", "price": 12.50, "productId": 1})__", R"__({"productId": 1, "productName": "A green door"})__", R"__({"productName": "A green door", "price": 12.50})__", R"__({"productId": 1, "productName": "A green door", "price": 12.50, "tags": []})__", R"__({"productId": 1, "productName": "A green door", "price": 12.50, "dimensions": {"length": 785, "width": 250.5, "height": -0.359}, "tags": ["home", "green"]})__"}},
};

TEST(JsonSchemaGrammar, WritesLlamaCppsGrammarLineForLine)
{
    for (const auto& c : llama_grammar_cases) {
        const auto g = ml::json_schema_to_gbnf(c.schema);
        if (!c.success) {
            EXPECT_TRUE(g.is_err()) << c.name;
            continue;
        }
        ASSERT_TRUE(g.is_ok()) << c.name << ": " << g.unwrap_err().path << ": " << g.unwrap_err().message;
        EXPECT_EQ(trim(*g), trim(c.expected)) << c.name;
        EXPECT_TRUE(ml::parse_grammar(*g).is_ok()) << c.name;   // and the output always parses
    }
}

TEST(JsonSchemaGrammar, AcceptsAndRejectsLlamaCppsStrings)
{
    for (const auto& c : llama_string_cases) {
        const auto g = ml::json_schema_grammar(c.schema, {.strict = c.strict});
        ASSERT_TRUE(g.is_ok()) << c.name << ": " << g.unwrap_err().path << ": " << g.unwrap_err().message;
        for (const auto& s : c.passing)
            EXPECT_TRUE(ml::grammar_accepts(*g, s)) << c.name << " should accept " << s;
        for (const auto& s : c.failing)
            EXPECT_FALSE(ml::grammar_accepts(*g, s)) << c.name << " should reject " << s;
    }
}

ml::json_schema_error error_of(std::string_view schema)
{
    const auto g = ml::json_schema_to_gbnf(schema);
    EXPECT_TRUE(g.is_err()) << schema;
    return g.is_err() ? g.unwrap_err() : ml::json_schema_error{};
}

TEST(JsonSchemaErrors, NameTheKeywordAndItsPath)
{
    EXPECT_EQ(error_of(R"({"type": "string", "pattern": "^a+$"})"), (ml::json_schema_error{0, "#", "unsupported keyword pattern"}));
    EXPECT_EQ(error_of(R"({"properties": {"a": {"type": "array", "uniqueItems": true}}})"),
              (ml::json_schema_error{0, "#/properties/a", "unsupported keyword uniqueItems"}));
    EXPECT_EQ(error_of(R"({"anyOf": [{"type": "integer"}, {"not": {"type": "null"}}]})"),
              (ml::json_schema_error{0, "#/anyOf/1", "unsupported keyword not"}));
    EXPECT_EQ(error_of(R"({"type": "number", "minimum": 0})"), (ml::json_schema_error{0, "#", "unsupported keyword minimum on a number"}));
    EXPECT_EQ(error_of(R"({"type": "kaboom"})").message, "unrecognized type kaboom");
    EXPECT_EQ(error_of(R"({"type": 123})").message, "type must be a string or an array of strings");
    EXPECT_EQ(error_of(R"({"type": []})").message, "type must not be empty");
    EXPECT_EQ(error_of(R"({"enum": []})").message, "enum must be a non-empty array");
    EXPECT_EQ(error_of(R"({"anyOf": []})"), (ml::json_schema_error{0, "#/anyOf", "anyOf must not be empty"}));
    EXPECT_EQ(error_of(R"({"items": {"type": "string"}, "minItems": -1})").message, "minItems must be a non-negative integer");
    EXPECT_EQ(error_of(R"({"type": "string", "maxLength": 1.5})").message, "maxLength must be a non-negative integer");
    EXPECT_EQ(error_of(R"({"$ref": "#/$defs/missing"})").message, "cannot resolve $ref #/$defs/missing");
    EXPECT_EQ(error_of(R"({"$ref": "https://example.com/schema"})").message,
              "unsupported $ref https://example.com/schema, only references into the same document are supported");
    EXPECT_EQ(error_of(R"({"properties": {"a": 1}})"), (ml::json_schema_error{0, "#/properties/a", "schema must be an object"}));
    EXPECT_EQ(error_of(R"({"allOf": [{"type": "integer", "minimum": 9}, {"type": "integer", "maximum": 5}]})").message,
              "allOf has an empty numeric range");
    EXPECT_EQ(error_of("[1, 2]").message, "schema must be an object");
    // A property name longer than llama.cpp's recursion limit, beside additionalProperties (its "long property name" case)
    EXPECT_EQ(error_of(R"({"type": "object", "properties": {")" + std::string(1000, 'a') + R"(": {"type": "string"}}, "additionalProperties": true})")
                  .message,
              "a property name is longer than 128 characters");
    // A $ref cycle that never reaches a schema
    EXPECT_FALSE(ml::json_schema_to_gbnf(R"({"$ref": "#/$defs/a", "$defs": {"a": {"$ref": "#/$defs/a"}}})").is_ok()
                 && ml::json_schema_grammar(R"({"$ref": "#/$defs/a", "$defs": {"a": {"$ref": "#/$defs/a"}}})").is_ok());
}

TEST(JsonSchemaErrors, MalformedJsonAtItsByte)
{
    EXPECT_EQ(error_of(R"({"type": "string",})"), (ml::json_schema_error{18, "", "expecting a key"}));
    EXPECT_EQ(error_of(R"({"type" "string"})"), (ml::json_schema_error{8, "", "expecting ':'"}));
    EXPECT_EQ(error_of(R"({"type": "str)"), (ml::json_schema_error{13, "", "unterminated string"}));
    EXPECT_EQ(error_of(R"({"const": 01})").message, "expecting ',' or '}'");
    EXPECT_EQ(error_of(R"({"const": "\x"})").message, "invalid escape");
    EXPECT_EQ(error_of(R"({"const": "\ud800"})").message, "unpaired surrogate");
    EXPECT_EQ(error_of(R"({} {})").message, "trailing characters after the schema");
    EXPECT_EQ(error_of("").message, "expecting a JSON value");
    EXPECT_EQ(error_of("{\"const\": \"a\nb\"}").message, "control character in a string");
}

TEST(JsonSchemaReader, StringsAndNumbersAsWritten)
{
    // Escapes decode to UTF-8 and are written back as nlohmann::json writes them (non-ASCII raw, controls as \u00xx)
    const auto g = ml::json_schema_to_gbnf(R"({"enum": ["é😀", "a\"b\\c", "\u0001\t", -0.5e+3, false]})");
    ASSERT_TRUE(g.is_ok());
    EXPECT_EQ(trim(*g), trim(R"(root ::= ("\"é😀\"" | "\"a\\\"b\\\\c\"" | "\"\\u0001\\t\"" | "-0.5e+3" | "false")
                                space ::= | " " | "\n"{1,2} [ \t]{0,20})"));
    const auto m = ml::json_schema_grammar(R"({"enum": ["é😀", "a\"b\\c", -0.5e+3]})");
    ASSERT_TRUE(m.is_ok());
    EXPECT_TRUE(ml::grammar_accepts(*m, "\"é😀\""));
    EXPECT_TRUE(ml::grammar_accepts(*m, R"("a\"b\\c")"));
    EXPECT_TRUE(ml::grammar_accepts(*m, "-0.5e+3"));
    EXPECT_FALSE(ml::grammar_accepts(*m, "-500"));
    // A repeated key keeps its place and takes the last value
    EXPECT_EQ(trim(*ml::json_schema_to_gbnf(R"({"type": "string", "type": "boolean"})")),
              trim(R"(root ::= ("true" | "false")
                      space ::= | " " | "\n"{1,2} [ \t]{0,20})"));
}

TEST(JsonSchemaBounds, FractionalIntegerBoundsRoundInwards)
{
    const auto g = ml::json_schema_grammar(R"({"type": "integer", "minimum": 2.5, "exclusiveMaximum": 7.5e0})");
    ASSERT_TRUE(g.is_ok());
    for (const char* s : {"3", "4", "7"})
        EXPECT_TRUE(ml::grammar_accepts(*g, s)) << s;
    for (const char* s : {"2", "8", "-3"})
        EXPECT_FALSE(ml::grammar_accepts(*g, s)) << s;
    const auto n = ml::json_schema_grammar(R"({"type": "integer", "minimum": -2.5, "maximum": -0.5})");
    ASSERT_TRUE(n.is_ok());
    EXPECT_TRUE(ml::grammar_accepts(*n, "-2") && ml::grammar_accepts(*n, "-1"));
    EXPECT_FALSE(ml::grammar_accepts(*n, "-3") || ml::grammar_accepts(*n, "0"));
    const auto e = ml::json_schema_grammar(R"({"type": "integer", "minimum": 1e2})");
    ASSERT_TRUE(e.is_ok());
    EXPECT_TRUE(ml::grammar_accepts(*e, "100") && !ml::grammar_accepts(*e, "99"));
}

TEST(JsonSchemaLenient, IgnoresWhatStrictRejectsAsLlamaCpp)
{
    const auto g = ml::json_schema_to_gbnf(R"({"type": "string", "pattern": "^[a-z]+$"})", {.strict = false});
    ASSERT_TRUE(g.is_ok());
    EXPECT_EQ(trim(*g), trim(R"(char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
                                root ::= string
                                space ::= | " " | "\n"{1,2} [ \t]{0,20}
                                string ::= "\"" char* "\"")"));
    EXPECT_TRUE(ml::json_schema_to_gbnf(R"({"type": "array", "uniqueItems": true, "items": {"type": "number", "minimum": 0}})", {.strict = false})
                    .is_ok());
}

TEST(JsonSchemaRefs, RecursiveSchemasConvertAndMatch)
{
    // A linked list through $defs: the reference resolves to a rule that names itself
    const auto g = ml::json_schema_grammar(R"({
        "$ref": "#/$defs/node",
        "$defs": {"node": {"type": "object", "properties": {"v": {"type": "integer"}, "next": {"anyOf": [{"$ref": "#/$defs/node"}, {"type": "null"}]}},
                           "required": ["v", "next"], "additionalProperties": false}}})");
    ASSERT_TRUE(g.is_ok()) << g.unwrap_err().message;
    EXPECT_TRUE(ml::grammar_accepts(*g, R"({"v": 1, "next": {"v": 2, "next": null}})"));
    EXPECT_FALSE(ml::grammar_accepts(*g, R"({"v": 1, "next": {"v": 2}})"));
    EXPECT_FALSE(ml::grammar_accepts(*g, R"({"v": 1, "next": null, "w": 3})"));
}

// A schema written in the program, compiled to a grammar's tables at compile time
constexpr auto colour = ml::compile_json_schema<R"({"enum": ["red", "green", 7]})">();
static_assert(ml::grammar_accepts(colour, "\"red\"") && ml::grammar_accepts(colour, "7") && !ml::grammar_accepts(colour, "\"blue\""));
static_assert(ml::json_schema_to_gbnf(R"({"const": true})").is_ok());
static_assert(ml::json_schema_to_gbnf(R"({"type": "kaboom"})").unwrap_err().message == "unrecognized type kaboom");

TEST(JsonSchemaCompileTime, IsTheRunTimeGrammarElementForElement)
{
    constexpr auto point = ml::compile_json_schema<R"({"type": "object", "properties": {"x": {"type": "integer"}, "y": {"type": "boolean"}},
                                                       "required": ["x"], "additionalProperties": false})">();
    const auto g = ml::json_schema_grammar(R"({"type": "object", "properties": {"x": {"type": "integer"}, "y": {"type": "boolean"}},
                                                       "required": ["x"], "additionalProperties": false})");
    ASSERT_TRUE(g.is_ok());
    ASSERT_EQ(g->elements.size(), point.elements.size());
    EXPECT_TRUE(std::equal(g->elements.begin(), g->elements.end(), point.elements.begin()));
    EXPECT_TRUE(std::equal(g->rules.begin(), g->rules.end(), point.rules.begin()));
    EXPECT_TRUE(ml::grammar_accepts(point, R"({"x": -12, "y": true})"));
    EXPECT_TRUE(ml::grammar_accepts(point, R"({ "x": 0 })"));
    EXPECT_FALSE(ml::grammar_accepts(point, R"({"y": true})"));
}

} // namespace
