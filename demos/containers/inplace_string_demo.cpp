//
//  inplace_string_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/containers/inplace_string.hpp>: contract ids held in place in a trivially copyable
//  record, a text too long refused, comparison, hashing and use as a template argument. Exits non-zero if a value
//  is not the one documented.
//

#include <snicholls/core/acout.hpp>
#include <snicholls/core/containers/inplace_string.hpp>

#include <cstdio>
#include <cstring>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace {
int failures = 0;
void check(bool ok, const char* what)
{
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++failures;
}

//! [inplace_string]
struct order {                                  // a record that can be copied as bytes
    snicholls::inplace_string<64> contract;
    double quantity;
};
static_assert(std::is_trivially_copyable_v<order> && sizeof(snicholls::inplace_string<64>) == 65);

template <snicholls::inplace_string<16> Venue>  // a string as a template argument
constexpr std::string_view venue() { return Venue.view(); }
static_assert(venue<"XNYS">() == "XNYS");
//! [inplace_string]
} // namespace

int main()
{
    std::printf("snicholls-core inplace_string demo\n");
    const std::string from_wire = "BTC-USD-PERP";
    const auto id = snicholls::inplace_string<64>::make(from_wire);
    const auto too_long = snicholls::inplace_string<8>::make(from_wire);   // refused, never cut to "BTC-USD-"
    if (!id)
        return 1;
    const order o{*id, 2.5};
    order copy{};
    std::memcpy(&copy, &o, sizeof o);

    std::unordered_map<snicholls::inplace_string<64>, double> positions;
    positions[o.contract] += o.quantity;
    std::printf("  %.*s: %zu of %zu characters, hash %016llx\n", static_cast<int>(id->size()), id->data(), id->size(),
                id->capacity(), static_cast<unsigned long long>(id->hash()));

    check(!too_long && too_long.unwrap_err_unchecked() == snicholls::inplace_string_errc::too_long, "too long is refused");
    check(copy.contract == o.contract && copy.contract == std::string_view{"BTC-USD-PERP"}, "copied as bytes, equal as text");
    check(positions.at(snicholls::inplace_string<64>{"BTC-USD-PERP"}) == 2.5, "keys a hash map");
    check(snicholls::inplace_string<8>{"BTC"} < *id && snicholls::inplace_string<8>{"BTC"} != *id, "orders across capacities");
    check(snicholls::inplace_string<4>{}.empty() && !id->empty(), "empty");
    //! [print]
    snicholls::acout{} << "  contract " << o.contract << ", in a list " << std::vector{o.contract} << '\n';   // no .view()
    //! [print]
    std::ostringstream text;
    snicholls::print_to(text, std::vector{o.contract});
    check(text.str() == "[\"BTC-USD-PERP\"]", "prints as text, quoted inside a container");
    const std::string_view as_view = *id;   // a std::string_view wherever one is taken
    check(as_view == "BTC-USD-PERP", "converts to std::string_view");
    return failures == 0 ? 0 : 1;
}
