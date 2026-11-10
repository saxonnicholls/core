//
//  endian_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/endian.hpp>: writing and reading a small binary
//  frame header with explicit byte order. Exits non-zero if anything reads
//  back differently from what was written.
//

#include <snicholls/core/endian.hpp>

#include <cstdint>
#include <cstdio>
#include <string>

int main()
{
    std::printf("snicholls-core endian demo\n");
    bool ok = true;

    //! [frame]
    // A frame header: a big-endian u16 length (network order), then a
    // little-endian u64 call id and a double, the way the remote-compute
    // frames carry them. The same bytes on every machine.
    unsigned char header[18];
    snicholls::store_be(header, std::uint16_t{512});
    snicholls::store_le(header + 2, std::uint64_t{42});
    snicholls::store_le(header + 10, 101.25);

    const auto length = snicholls::load_be<std::uint16_t>(header);
    const auto call_id = snicholls::load_le<std::uint64_t>(header + 2);
    const auto price = snicholls::load_le<double>(header + 10);   // unaligned is fine
    //! [frame]

    std::string hex;
    for (unsigned char b : header) {
        char two[4];
        std::snprintf(two, sizeof two, "%02x ", b);
        hex += two;
    }
    std::printf("  bytes    %s\n", hex.c_str());
    std::printf("  length   %u, call id %llu, price %.2f\n", unsigned(length),
                static_cast<unsigned long long>(call_id), price);
    ok &= length == 512 && call_id == 42 && price == 101.25;
    ok &= header[0] == 0x02 && header[1] == 0x00;   // 512 big-endian
    ok &= header[2] == 42 && header[3] == 0;        // 42 little-endian

    std::printf("  byteswap 0x%08x -> 0x%08x\n", 0x01020304u, snicholls::byteswap(0x01020304u));
    ok &= snicholls::byteswap(0x01020304u) == 0x04030201u;

    std::printf(ok ? "endian demo: ok\n" : "endian demo: FAILED\n");
    return ok ? 0 : 1;
}
