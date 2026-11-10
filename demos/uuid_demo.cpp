//
//  uuid_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/uuid.hpp>.
//

// snicholls::uuid - the value type: text in and out, version/variant, nil/max, composers, v5, hashing, ordering.
#include <array>
#include <cstdio>
#include <functional>
#include <iostream>
#include <set>
#include <span>
#include <string>
#include <string_view>

#include <snicholls/core/uuid.hpp>

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
    using snicholls::uuid;

    std::puts("Text: canonical out, any hex case in; lenient forms on request");
    constexpr uuid a = "F81D4FAE-7DEC-11D0-A765-00A0C91E6BF6"_uuid;   // checked at compile time
    std::cout << "  " << a << "  version " << a.version() << '\n';
    char buf[36];
    const auto [end, ec] = snicholls::to_chars(buf, buf + 36, a);
    check(ec == std::errc{} && std::string_view(buf, end - buf) == "f81d4fae-7dec-11d0-a765-00a0c91e6bf6", "to_chars");
    char small[10];
    check(snicholls::to_chars(small, small + 10, a).ec == std::errc::value_too_large, "to_chars: buffer too small");
    check(snicholls::to_text(a).view() == snicholls::to_string(a) && std::string(snicholls::to_text(a).c_str()).size() == 36 &&
              snicholls::uuid_text::size() == 36 && std::string_view(snicholls::to_text(a)) == snicholls::to_string(a),
          "to_text / to_string agree");
    check(snicholls::parse_uuid(std::string("f81d4fae-7dec-11d0-a765-00a0c91e6bf6")) == a, "parse_uuid canonical");
    check(!snicholls::parse_uuid("{f81d4fae-7dec-11d0-a765-00a0c91e6bf6}"), "braces rejected by default");
    const auto lenient = snicholls::uuid_parse_options::lenient();
    check(snicholls::parse_uuid("{f81d4fae-7dec-11d0-a765-00a0c91e6bf6}", lenient) == a, "braces when enabled");
    check(snicholls::parse_uuid("urn:uuid:f81d4fae-7dec-11d0-a765-00a0c91e6bf6", lenient) == a, "urn:uuid: when enabled");
    check(snicholls::parse_uuid("f81d4fae7dec11d0a76500a0c91e6bf6", lenient) == a, "32 hex digits when enabled");
    for (const char* bad : {"", "f81d4fae-7dec-11d0-a765-00a0c91e6bfg", "{f81d4fae-7dec-11d0-a765-00a0c91e6bf6",
                            "urn:uid:f81d4fae-7dec-11d0-a765-00a0c91e6bf6", "uf81d4fae", "f81d4fae-7dec"})
        check(!snicholls::parse_uuid(bad, lenient), (std::string("rejects \"") + bad + "\"").c_str());
    uuid out = uuid::max();
    const std::string_view prefix = "f81d4fae-7dec-11d0-a765-00a0c91e6bf6,next";
    const auto r = snicholls::from_chars(prefix.data(), prefix.data() + prefix.size(), out);
    check(r.ec == std::errc{} && *r.ptr == ',' && out == a, "from_chars parses a prefix");

    std::puts("Special values, version, variant");
    check(uuid{}.is_nil() && uuid::nil().is_nil() && uuid::max().is_max(), "nil and max");
    check(uuid::nil().variant() == snicholls::uuid_variant::ncs && a.variant() == snicholls::uuid_variant::rfc9562 &&
              uuid::max().variant() == snicholls::uuid_variant::future,
          "variants");
    uuid ms = a;
    ms.bytes[8] = 0xC0;
    check(ms.variant() == snicholls::uuid_variant::microsoft, "microsoft variant");
    check(uuid::from_u64(a.high(), a.low()) == a && a.as_bytes()[0] == std::byte{0xf8}, "halves and bytes");

    std::puts("Composers (RFC 9562 appendix vectors)");
    std::array<std::uint8_t, 16> random{0x91, 0x91, 0x08, 0xf7, 0x52, 0xd1, 0x43, 0x20,
                                        0x9b, 0xac, 0xf8, 0x47, 0xdb, 0x41, 0x48, 0xa8};
    check(snicholls::make_uuid_v4(random) == "919108f7-52d1-4320-9bac-f847db4148a8"_uuid, "v4 from random bits");
    constexpr uuid v7 = snicholls::make_uuid_v7(0x017F22E279B0, 0xCC3, 0x18C4DC0C0C07398F);
    std::cout << "  v7 " << v7 << " created " << v7.timestamp()->time_since_epoch().count() << " ms after 1970\n";
    check(v7 == "017f22e2-79b0-7cc3-98c4-dc0c0c07398f"_uuid && !a.timestamp(), "v7 fields and timestamp");
    constexpr uuid v5 = snicholls::make_uuid_v5(snicholls::uuid_namespace::dns, "www.example.com");
    check(v5 == "2ed6657d-e927-568b-95e1-2665a8aea6a2"_uuid, "v5 (DNS, www.example.com)");
    const std::string_view name = "www.example.com";
    check(snicholls::make_uuid_v5(snicholls::uuid_namespace::dns, std::as_bytes(std::span(name.data(), name.size()))) == v5,
          "v5 from bytes");
    check(snicholls::uuid_namespace::url != snicholls::uuid_namespace::oid && snicholls::uuid_namespace::x500.version() == 1,
          "namespace IDs");

    std::puts("Ordering and hashing");
    std::set<uuid> sorted{uuid::max(), a, uuid::nil()};
    check(*sorted.begin() == uuid::nil() && *sorted.rbegin() == uuid::max(), "sorts bytewise");
    check(std::hash<uuid>{}(a) == snicholls::hash_value(a) && snicholls::hash_value(a) != snicholls::hash_value(v5), "std::hash");

    std::printf("%s\n", failures == 0 ? "uuid_demo: all checks passed" : "uuid_demo: FAILED");
    return failures == 0 ? 0 : 1;
}

//! [demo]