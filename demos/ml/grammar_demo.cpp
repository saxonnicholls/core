//
//  grammar_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/ml/grammar.hpp>: a GBNF grammar parsed at run time and the same grammar compiled at
//  compile time, strings matched against it, a parse error with its position, and the token mask over a small
//  vocabulary with a character split across two tokens, applied to logits as a bit mask. Exits non-zero if a
//  result is not what it documents.
//

#include <snicholls/core/ml/grammar.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <span>
#include <string_view>

int main()
{
    std::printf("snicholls-core grammar demo\n");
    namespace ml = snicholls::ml;
    bool ok = true;

    //! [grammar]
    // A yes/no answer with an optional reason, as GBNF (llama.cpp's grammar language)
    constexpr std::string_view text = R"(root   ::= answer (" because " reason)?
answer ::= "yes" | "no"
reason ::= [a-z ]+
)";
    const auto parsed = ml::parse_grammar(text);   // at run time: a result
    const ml::grammar& g = *parsed;

    // The same grammar at compile time, into std::array tables; a malformed one would not compile
    constexpr auto fixed = ml::compile_grammar<R"(root   ::= answer (" because " reason)?
answer ::= "yes" | "no"
reason ::= [a-z ]+
)">();
    static_assert(ml::grammar_accepts(fixed, "yes"));

    const bool yes = ml::grammar_accepts(g, "no because it rains");
    const bool shout = ml::grammar_accepts(fixed, "YES");

    // A malformed grammar: where and why
    const auto bad = ml::parse_grammar(R"(root ::= "yes" | [a-z)");
    //! [grammar]

    std::printf("  %zu rules, %zu elements; 'answer' is rule %u; the compile-time tables match: %s\n", g.rules.size(), g.elements.size(),
                g.find("answer"), fixed.elements.size() == g.elements.size() ? "yes" : "no");
    ok &= parsed.is_ok() && g.find("answer") == 1 && g.find("nothing") == g.names.size() && g.view().rule(1).size() == 7;
    ok &= fixed.elements.size() == g.elements.size() && fixed.root == g.root;
    std::printf("  'no because it rains': %s, 'YES': %s\n", yes ? "accepted" : "rejected", shout ? "accepted" : "rejected");
    ok &= yes && !shout && ml::grammar_accepts(g.view(), "yes because x");
    std::printf("  malformed: '%.*s' at byte %zu\n", static_cast<int>(bad.unwrap_err().message.size()), bad.unwrap_err().message.data(),
                bad.unwrap_err().position);
    ok &= bad.is_err() && bad.unwrap_err().position == 21;

    //! [grammar_mask]
    // A vocabulary in which é (0xc3 0xa9) can come whole or split over two tokens; token 5 ends generation
    const auto cafe = ml::parse_grammar(R"(root ::= "caf" ("é" | "e"))");
    const std::array<std::string_view, 6> vocab{"caf", "\xc3\xa9", "\xc3", "\xa9", "e", "</s>"};
    const std::array<std::uint32_t, 1> end{5};

    ml::grammar_matcher m(*cafe);
    m.accept_token(0, vocab[0]);   // "caf"
    std::array<bool, 6> allowed{};
    m.token_mask(vocab, end, allowed);   // é whole, its first byte, or e; not a lone continuation byte, not the end

    m.accept_token(2, vocab[2]);   // the first byte of é: the matcher carries the unfinished sequence
    std::array<std::uint32_t, 1> bits{};
    m.token_bitmask(vocab, end, bits);   // only the continuation byte may follow
    std::array<float, 6> logits{0.5f, 1.0f, 2.0f, 0.1f, 3.0f, 0.0f};
    ml::apply_token_bitmask(std::span<const std::uint32_t>(bits), std::span<float>(logits));

    m.accept_token(3, vocab[3]);
    const bool done = m.accept_end();   // the sentence is complete: the end token is allowed
    //! [grammar_mask]

    std::printf("  after 'caf': allowed %d%d%d%d%d%d\n", allowed[0], allowed[1], allowed[2], allowed[3], allowed[4], allowed[5]);
    ok &= allowed == std::array<bool, 6>{false, true, true, false, true, false};
    std::printf("  after 0xc3: bits %#x, masked logits %g %g %g %g %g %g\n", bits[0], static_cast<double>(logits[0]), static_cast<double>(logits[1]),
                static_cast<double>(logits[2]), static_cast<double>(logits[3]), static_cast<double>(logits[4]), static_cast<double>(logits[5]));
    ok &= bits[0] == 0b001000u && std::isinf(logits[0]) && logits[3] == 0.1f && std::isinf(logits[4]);
    std::printf("  complete after 0xa9: %s, stacks %zu, failed %s\n", done ? "yes" : "no", m.stacks().size(), m.failed() ? "yes" : "no");
    ok &= done && m.complete() && !m.failed() && m.utf8_state().remaining == 0;

    // The lower-level steps: one code point, a string carrying an unfinished sequence, the decoder itself
    ml::grammar_matcher n(*cafe);
    ok &= n.accept_string("ca") && n.accept_code_point('f') && n.accept_string("\xc3") && n.utf8_state().remaining == 1;
    ok &= n.allows(3, "\xa9") && !n.allows(4, "e");
    std::array<bool, 6> by_callable{};
    n.token_mask([&](std::uint32_t t) { return vocab[t]; }, [](std::uint32_t t) { return t == 5; }, std::span<bool>(by_callable));
    ok &= by_callable == std::array<bool, 6>{false, false, false, true, false, false};
    const auto [cps, state] = ml::decode_utf8("caf\xc3", {});
    ok &= cps.size() == 3 && state.remaining == 1;
    std::printf("  step by step: %s\n", ok ? "as documented" : "NOT as documented");

    std::printf(ok ? "grammar demo: ok\n" : "grammar demo: FAILED\n");
    return ok ? 0 : 1;
}
