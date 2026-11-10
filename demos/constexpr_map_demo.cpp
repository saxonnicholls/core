//
//  constexpr_map_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/constexpr_map.hpp>: an enumeration's names as one
//  table that is looked up in both directions, at compile time and at run time.
//  Exits non-zero if any lookup is not the one documented.
//

#include <snicholls/core/constexpr_map.hpp>

#include <cstdio>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace {

enum class http_method { get, post, put, del };

//! [table]
// One table, written once, a constant.
constexpr auto method_names = snicholls::make_constexpr_map(
    std::pair{http_method::get, std::string_view{"GET"}},
    std::pair{http_method::post, std::string_view{"POST"}},
    std::pair{http_method::put, std::string_view{"PUT"}},
    std::pair{http_method::del, std::string_view{"DELETE"}});
static_assert(method_names.keys_unique());

// Enumeration to name, and name to enumeration, both at compile time ...
static_assert(method_names.at(http_method::put) == "PUT");
static_assert(*method_names.find_key(std::string_view{"DELETE"}) == http_method::del);

// ... and at run time, where an unknown name is a null pointer, not an exception.
const http_method* parse_method(std::string_view text) { return method_names.find_key(text); }
//! [table]

} // namespace

int main()
{
    std::printf("snicholls-core constexpr_map demo\n");
    bool ok = true;

    for (const auto& [method, name] : method_names)
        std::printf("  %-6.*s -> %d\n", static_cast<int>(name.size()), name.data(), static_cast<int>(method));

    const http_method* post = parse_method("POST");
    const http_method* patch = parse_method("PATCH");
    std::printf("  parse_method(\"POST\")  -> %s\n", post ? "found" : "not found");
    std::printf("  parse_method(\"PATCH\") -> %s\n", patch ? "found" : "not found");
    ok &= post != nullptr && *post == http_method::post;
    ok &= patch == nullptr;

    // at() throws for an absent key at run time
    bool threw = false;
    try {
        (void)method_names.at(static_cast<http_method>(99));
    } catch (const std::out_of_range&) {
        threw = true;
    }
    std::printf("  at(unknown) throws: %s\n", threw ? "yes" : "no");
    ok &= threw;

    ok &= method_names.size() == 4 && method_names.contains(http_method::get);

    std::printf(ok ? "constexpr_map demo: ok\n" : "constexpr_map demo: FAILED\n");
    return ok ? 0 : 1;
}
