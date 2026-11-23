//
//  phone_number_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/phone_number.hpp>.
//

// snicholls::phone_number - parse the ways people write a number, refuse what is not E.164, canonical text out,
// country code and national number, the checked literal, ordering and hashing.
#include <array>
#include <cstdio>
#include <functional>
#include <iostream>
#include <set>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_set>

#include <snicholls/core/phone_number.hpp>

namespace {
int failures = 0;
void check(bool ok, const char* what)
{
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++failures;
}
} // namespace

//! [demo]
int main()
{
    using namespace snicholls::literals;
    using snicholls::phone_errc;
    using snicholls::phone_number;

    std::puts("Parse: one number, many spellings, one value");
    constexpr phone_number mobile = "+61412345678"_e164;   // checked at compile time
    for (const char* text : {"+61 412 345 678", "+61-412-345-678", "(+61) 412345678", "00 61 412 345 678",
                             "+61.412.345.678", "+61/412/345/678"}) {
        const auto r = snicholls::parse_phone_number(text);
        check(r.is_ok() && *r == mobile, text);
    }

    std::puts("Refused, with the reason as a value (no exceptions)");
    struct refusal
    {
        const char* text;
        phone_errc why;
    };
    for (const refusal& r : {refusal{"", phone_errc::empty},
                             refusal{"0412 345 678", phone_errc::missing_country_code},
                             refusal{"+1-800-FLOWERS", phone_errc::invalid_character},
                             refusal{"(+61 412 345 678", phone_errc::unbalanced_parentheses},
                             refusal{"+999 1234 5678", phone_errc::unknown_country_code},
                             refusal{"+44 (0)20 7946 0000", phone_errc::trunk_prefix},
                             refusal{"+61 4", phone_errc::too_short},
                             refusal{"+61 412 345 678 901 234", phone_errc::too_long}}) {
        const auto p = snicholls::parse_phone_number(r.text);
        check(!p.is_ok() && p.err() == r.why, (std::string("refuses \"") + r.text + "\"").c_str());
    }

    std::puts("Parts and text");
    constexpr phone_number rome = "+39 06 6988 4444"_e164;   // the leading 0 is part of an Italian number
    std::cout << "  " << snicholls::to_text(rome).view() << ": country code " << rome.country_code() << ", national "
              << rome.national_number().view() << '\n';
    check(rome.country_code() == 39 && rome.country_code_length() == 2 && rome.national_number().view() == "0669884444" &&
              rome.national_number_length() == 10 && rome.digit_count() == 12 && rome.digit(2) == '0',
          "country_code, national_number, digit_count, digit");
    std::array<char, snicholls::phone_number_max_text_size> buf{};
    const auto [end, ec] = snicholls::to_chars(buf.data(), buf.data() + buf.size(), rome);
    check(ec == std::errc{} && std::string_view(buf.data(), static_cast<std::size_t>(end - buf.data())) == "+390669884444",
          "to_chars into a buffer");
    check(snicholls::to_chars(buf.data(), buf.data() + 4, rome).ec == std::errc::value_too_large,
          "to_chars: buffer too small");
    check(snicholls::to_string(rome) == "+390669884444" && std::string(snicholls::to_text(rome).c_str()).size() == 13 &&
              snicholls::to_text(rome).size() == 13 && std::string_view(snicholls::to_text(rome)) == "+390669884444",
          "to_text / to_string agree");
    std::cout << "  printed: ";
    mobile.print(std::cout);
    std::cout << '\n';
    check(phone_number{}.empty() && !mobile.empty() && snicholls::to_text(phone_number{}).size() == 0,
          "the empty number");
    check(snicholls::is_e164_country_code(61) && snicholls::is_e164_country_code(800) &&
              !snicholls::is_e164_country_code(999),
          "is_e164_country_code");

    std::puts("Ordering and hashing: the order of the canonical text");
    const std::set<phone_number> sorted{mobile, "+16502530000"_e164, "+442079460000"_e164};
    check(*sorted.begin() == "+16502530000"_e164 && *sorted.rbegin() == mobile, "sorts as text");
    const std::unordered_set<phone_number> directory{mobile, rome};
    check(directory.count(*snicholls::parse_phone_number("0061 412-345-678")) == 1, "found by another spelling");
    check(std::hash<phone_number>{}(mobile) == snicholls::hash_value(mobile) &&
              snicholls::hash_value(mobile) != snicholls::hash_value(rome),
          "std::hash");

    std::printf("%s\n", failures == 0 ? "phone_number_demo: all checks passed" : "phone_number_demo: FAILED");
    return failures == 0 ? 0 : 1;
}
//! [demo]
