//
//  unicode_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/ml/unicode.hpp>: decoding text code point by code point with each one's category, at
//  compile time and at run time; invalid bytes; White_Space; lowercase; canonical decomposition and accent
//  stripping; repairing a text. Exits non-zero if a result is not what it documents.
//

#include <snicholls/core/ml/unicode.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <string_view>

namespace {

namespace ml = snicholls::ml;

//! [unicode]
// Count the letters of a text at compile time: decode, classify, move on by the code point's length
constexpr std::size_t letters(std::string_view text)
{
    std::size_t n = 0;
    for (std::size_t i = 0; i < text.size();) {
        const ml::utf8_decoded d = ml::decode_utf8(text, i);
        n += ml::is_letter(ml::category(d.code_point));
        i += d.length;
    }
    return n;
}
static_assert(letters("Grüße, 世界! 42") == 7);
// An invalid byte is one byte of U+FFFD, never a stall
static_assert(!ml::decode_utf8("\xC0\xAF", 0).valid && ml::decode_utf8("\xC0\xAF", 0).length == 1);
// Accent stripping as BERT does it: decompose, drop the nonspacing marks
constexpr char32_t base_of(char32_t cp)
{
    std::array<char32_t, ml::max_decomposition> parts{};
    ml::decompose(cp, parts);
    return parts[0];
}
static_assert(base_of(U'é') == U'e' && base_of(U'Ǖ') == U'U' && ml::to_lower(U'Ä') == U'ä');
//! [unicode]

const char* name(ml::unicode_category c)
{
    constexpr const char* names[] = {"Cn", "Lu", "Ll", "Lt", "Lm", "Lo", "Mn", "Mc", "Me", "Nd", "Nl", "No", "Pc", "Pd", "Ps",
                                     "Pe", "Pi", "Pf", "Po", "Sm", "Sc", "Sk", "So", "Zs", "Zl", "Zp", "Cc", "Cf", "Cs", "Co"};
    return names[static_cast<int>(c)];
}

} // namespace

int main()
{
    std::printf("snicholls-core unicode demo (Unicode %s)\n", snicholls::detail::unicode_tables::version);
    bool ok = true;
    const std::string_view text = "Ǆa é\xCC\x81 ½ 中 🙂\xE2\x80\x8D";
    std::string cats;
    for (std::size_t i = 0; i < text.size();) {
        const auto d = ml::decode_utf8(text, i);
        cats += name(ml::category(d.code_point));
        cats += ' ';
        i += d.length;
    }
    std::printf("  categories: %s\n", cats.c_str());
    ok &= cats == "Lu Ll Zs Ll Mn Zs No Zs Lo Zs So Cf ";
    ok &= ml::is_whitespace(0x3000) && !ml::is_whitespace(0x200B);
    std::printf("  U+3000 is whitespace, U+200B is not\n");

    std::array<char32_t, ml::max_decomposition> parts{};
    const std::size_t n = ml::decompose(U'한', parts);
    std::printf("  U+D55C decomposes to %zu jamo: U+%04X U+%04X U+%04X\n", n, static_cast<unsigned>(parts[0]),
                static_cast<unsigned>(parts[1]), static_cast<unsigned>(parts[2]));
    ok &= n == 3 && parts[0] == 0x1112 && ml::combining_class(0x0301) == 230;

    std::string repaired(30, '\0');
    repaired.resize(ml::replace_invalid_utf8("a\x80" "b", std::span<char>(repaired)));
    std::printf("  \"a\\x80b\" repaired: %zu bytes, valid: %s\n", repaired.size(), ml::is_valid_utf8(repaired) ? "yes" : "no");
    ok &= repaired == "a\xEF\xBF\xBD" "b" && !ml::is_valid_utf8("a\x80");

    std::printf(ok ? "unicode demo: ok\n" : "unicode demo: FAILED\n");
    return ok ? 0 : 1;
}
