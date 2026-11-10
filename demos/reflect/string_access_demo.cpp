//
//  string_access_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/reflect/string_access.hpp>.
//

// Demo: reflect/string_access.hpp - zero-copy, allocation-free access to any string or byte sequence:
// snicholls::to_string_view, snicholls::length_of, snicholls::c_str_of, snicholls::as_bytes_view.

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <snicholls/core/reflect/string_access.hpp>

namespace demo {

bool ok = true;

void check(bool condition, std::string_view what)
{
    std::cout << (condition ? "  ok    " : "  WRONG ") << what << '\n';
    ok = ok && condition;
}

// A key accepted from any text carrier, viewed without copying (what a JSON writer or hash would do).
template <snicholls::string_like S>
std::size_t key_length(const S& key)
{
    return snicholls::to_string_view(key).size();
}

// Raw bytes from any byte-viewable carrier (what a checksum or an encoder would do).
template <snicholls::byte_viewable S>
unsigned checksum(const S& data)
{
    unsigned sum = 0;
    for (std::byte b : snicholls::as_bytes_view(data)) sum += std::to_integer<unsigned>(b);
    return sum;
}

} // namespace demo

//! [demo]
int main()
{
    using demo::check;
    using namespace std::string_view_literals;

    std::cout << "to_string_view / length_of over every text carrier:\n";
    const char* ptr = "pointer";
    const char* null = nullptr;
    char buffer[] = "buffer";
    const std::string str = "std::string";
    const std::vector<char> vec{'v', 'e', 'c'};
    const std::array<char, 5> arr{'a', 'r', 'r', 'a', 'y'};
    const std::span<const char, 5> fixed_span(arr);
    constexpr snicholls::fixed_string name("price");
    check(snicholls::to_string_view(ptr) == "pointer" && snicholls::length_of(ptr) == 7, "const char*: strlen, O(n)");
    check(snicholls::to_string_view(null).empty() && snicholls::length_of(null) == 0, "null const char*: empty, not UB");
    check(snicholls::to_string_view(buffer) == "buffer" && snicholls::length_of(buffer) == 6, "char[7]: N-1 code units");
    check(snicholls::to_string_view("a\0b") == "a\0b"sv, "string literal keeps embedded NULs (length 3)");
    check(snicholls::to_string_view(str).data() == str.data(), "std::string: same data(), no copy");
    check(snicholls::to_string_view(vec) == "vec" && snicholls::length_of(vec) == 3, "std::vector<char>");
    check(snicholls::to_string_view(arr) == "array" && snicholls::length_of(arr) == 5, "std::array<char, 5>: compile-time length");
    check(snicholls::to_string_view(fixed_span) == "array", "std::span<const char, 5>");
    check(snicholls::to_string_view(std::string_view("view")) == "view", "std::string_view");
    static_assert(snicholls::to_string_view(name) == "price" && snicholls::length_of(name) == 5);
    check(snicholls::to_string_view(name) == "price", "snicholls::fixed_string (constexpr)");
    check(snicholls::to_string_view(u"€") == u"€" && snicholls::length_of(u"€") == 1, "char16_t literal");
    check(snicholls::length_of(u8"€") == 3, "char8_t: lengths are code units (U+20AC is 3 in UTF-8)");
    check(demo::key_length("px") + demo::key_length(str) + demo::key_length(name) == 2 + 11 + 5,
          "one generic function accepts every carrier");

    std::cout << "\nc_str_of for C APIs (NUL-terminated carriers only):\n";
    char out[32];
    std::snprintf(out, sizeof out, "%s|%s|%s", snicholls::c_str_of(str), snicholls::c_str_of(name), snicholls::c_str_of(null));
    check(out == "std::string|price|"sv, "std::string, fixed_string, null pointer -> \"\"");
    check(snicholls::c_str_of(ptr) == ptr && snicholls::c_str_of(buffer) == buffer, "pointer and array pass straight through");

    std::cout << "\nas_bytes_view over byte sequences and one-byte text:\n";
    const std::vector<std::uint8_t> u8{1, 2, 3};
    const std::array<std::byte, 2> raw{std::byte{10}, std::byte{20}};
    const unsigned char packet[3] = {4, 0, 5};   // a byte array has no terminator: all 3 bytes
    check(demo::checksum(u8) == 6, "std::vector<std::uint8_t>");
    check(demo::checksum(raw) == 30, "std::array<std::byte, 2> (constexpr-capable: no cast)");
    check(demo::checksum(packet) == 9 && snicholls::length_of(packet) == 3, "unsigned char[3]: length N");
    check(demo::checksum("AB") == 65 + 66, "char text: its bytes, without the terminator");
    check(snicholls::as_bytes_view(null).empty(), "null const char*: empty span");
    static_assert(!snicholls::byte_viewable<std::u16string>, "wide text bytes are endian-dependent");
    std::cout << "  std::u16string is byte_viewable: " << std::boolalpha << snicholls::byte_viewable<std::u16string> << '\n';

    std::cout << (demo::ok ? "\nOK\n" : "\nFAILED\n");
    return demo::ok ? 0 : 1;
}
//! [demo]
