//
//  mapped_header_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/containers/mapped_header.hpp>.
//

// Demo: snicholls::containers::mapped_header - the checked 64-byte header in front of a persisted container: write it,
// read it back, and refuse a torn, foreign or mismatched one. Exits non-zero if a result is not as printed.

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>

#include <snicholls/core/containers/mapped_header.hpp>
#include <snicholls/core/hash/fnv.hpp>

namespace {
int failures = 0;
void expect(bool ok, const char* what)
{
    std::printf("  %-60s %s\n", what, ok ? "ok" : "FAILED");
    if (!ok) ++failures;
}
struct tick
{
    std::int64_t time;
    double price;
};
} // namespace

int main()
{
    using namespace snicholls::containers;
    std::printf("mapped_header: %zu bytes, version %u\n", mapped_header_size, mapped_header_version);
    //! [header]
    constexpr std::uint64_t tag = snicholls::fnv1a_64("tick v1");
    std::array<std::byte, mapped_header_size> file_start{};
    // Writer: 1000 ticks in a file with room for 4096.
    if (encode_header(mapped_header_for<tick>(tag, 1000, 4096), file_start).is_err()) return 1;
    // Reader: is this a file of ticks, and how many?
    if (auto ok = validate_header(file_start, mapped_header_for<tick>(tag)); ok.is_err()) return 1;
    auto h = decode_header(file_start);
    //! [header]
    expect(h.is_ok() && h->count == 1000 && h->capacity == 4096, "the header round-trips");

    auto torn = file_start;
    torn[33] = std::byte{0x7f};   // half-written count
    expect(decode_header(torn).err() == header_error::bad_checksum, "a torn header fails its checksum");
    expect(validate_header(file_start, mapped_header_for<double>(tag)).err() == header_error::element_mismatch,
           "a reader of another element size is refused");
    expect(validate_header(file_start, mapped_header_for<tick>(snicholls::fnv1a_64("tick v2"))).err()
               == header_error::type_mismatch,
           "a reader of another schema is refused");
    expect(decode_header(std::array<std::byte, 64>{}).err() == header_error::bad_magic, "zeros are not a header");
    expect(encode_header(mapped_header_for<tick>(tag, 5, 4), file_start).err() == header_error::count_over_capacity,
           "count over capacity is not written");
    return failures == 0 ? 0 : 1;
}
