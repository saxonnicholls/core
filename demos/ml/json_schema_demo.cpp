//
//  json_schema_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/ml/json_schema.hpp>: a tool call's JSON schema written as GBNF, the grammar matching
//  and refusing candidate outputs, an unsupported keyword named with its path, llama.cpp's lenient mode, and a schema
//  compiled at compile time. Exits non-zero if a result is not what it documents.
//

#include <snicholls/core/ml/json_schema.hpp>

#include <cstdio>
#include <string>

int main()
{
    std::printf("snicholls-core json_schema demo\n");
    namespace ml = snicholls::ml;
    bool ok = true;

    //! [json_schema]
    // The arguments of a weather tool: a city, and optionally a unit from a fixed set
    constexpr const char* schema = R"({
        "type": "object",
        "properties": {
            "city": {"type": "string", "minLength": 1},
            "unit": {"enum": ["celsius", "fahrenheit"]}
        },
        "required": ["city"],
        "additionalProperties": false
    })";
    const auto gbnf = ml::json_schema_to_gbnf(schema);   // llama.cpp's grammar for it, rules sorted by name
    const auto g = ml::json_schema_grammar(schema);      // or straight to a compiled grammar

    const bool good = ml::grammar_accepts(*g, R"({"city": "Oslo", "unit": "celsius"})");
    const bool extra = ml::grammar_accepts(*g, R"({"city": "Oslo", "wind": 3})");

    // What the subset does not implement is an error naming the keyword and where it is
    const auto strict = ml::json_schema_to_gbnf(R"({"properties": {"zip": {"type": "string", "pattern": "^[0-9]{5}$"}}})");
    // llama.cpp's leniency instead: the pattern is ignored and any string is accepted
    const auto lenient = ml::json_schema_to_gbnf(R"({"properties": {"zip": {"type": "string", "pattern": "^[0-9]{5}$"}}})", {.strict = false});

    // A schema written in the program, compiled to grammar tables by the compiler
    constexpr auto yes_no = ml::compile_json_schema<R"({"type": "boolean"})">();
    static_assert(ml::grammar_accepts(yes_no, "true") && !ml::grammar_accepts(yes_no, "yes"));
    //! [json_schema]

    std::printf("%s", gbnf->c_str());
    ok &= gbnf.is_ok() && g.is_ok() && gbnf->find("root ::= \"{\" space city-kv") != std::string::npos;
    std::printf("  {\"city\": \"Oslo\", \"unit\": \"celsius\"}: %s; an extra property: %s\n", good ? "accepted" : "rejected",
                extra ? "accepted" : "rejected");
    ok &= good && !extra;
    std::printf("  strict: %s: %s\n", strict.unwrap_err().path.c_str(), strict.unwrap_err().message.c_str());
    ok &= strict.is_err() && strict.unwrap_err().path == "#/properties/zip" && strict.unwrap_err().message == "unsupported keyword pattern";
    ok &= lenient.is_ok() && lenient->find("zip ::= string") != std::string::npos;
    std::printf("  lenient: zip is any string; compile-time boolean grammar: %zu elements\n", yes_no.elements.size());

    std::printf(ok ? "json_schema demo: ok\n" : "json_schema demo: FAILED\n");
    return ok ? 0 : 1;
}
