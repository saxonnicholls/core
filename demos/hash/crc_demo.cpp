//
//  crc_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/hash/crc.hpp>: the three CRCs' check values at compile time, a frame checked as it is
//  received, and zlib's running CRC. Exits non-zero if any value is not the one documented.
//

#include <snicholls/core/hash/crc.hpp>

#include <cstdint>
#include <cstdio>
#include <string>

namespace {

//! [crc]
// Each CRC's catalogue check value, computed by the compiler
static_assert(snicholls::crc32::hash("123456789") == 0xCBF43926u);
static_assert(snicholls::crc32c::hash("123456789") == 0xE3069283u);
static_assert(snicholls::crc64_xz::hash("123456789") == 0x995DC9BBDF1939FAull);

// zlib's crc32(crc, buf, len): continue from a finished value
std::uint32_t running_crc(std::uint32_t crc_so_far, const std::string& more)
{
    return snicholls::crc32::resume(crc_so_far).update(more).finish();
}
//! [crc]

} // namespace

int main()
{
    std::printf("snicholls-core crc demo\n");
    bool ok = true;

    const std::uint32_t whole = snicholls::crc32::hash("The quick brown fox jumps over the lazy dog");
    const std::uint32_t pieces = running_crc(snicholls::crc32::hash("The quick brown fox "), "jumps over the lazy dog");
    std::printf("  crc32(fox)  = %08x, in two pieces %08x\n", whole, pieces);
    ok &= whole == 0x414FA339u && pieces == whole;

    // A frame checked with CRC-32C as it arrives (iSCSI, ext4, LevelDB)
    snicholls::crc32c c;
    c.update("frame ").update("payload");
    std::printf("  crc32c      = %08x\n", c.finish());
    ok &= c.finish() == snicholls::crc32c::hash("frame payload");
    c.update(" and more");   // finish is const: absorbing continues
    ok &= c.finish() == snicholls::crc32c::hash("frame payload and more");

    std::printf("  crc64/xz    = %016llx\n", static_cast<unsigned long long>(snicholls::crc64_xz::hash("123456789")));
    ok &= snicholls::crc64_xz::resume(snicholls::crc64_xz::hash("1234")).update("56789").finish() == 0x995DC9BBDF1939FAull;

    std::printf(ok ? "crc demo: ok\n" : "crc demo: FAILED\n");
    return ok ? 0 : 1;
}
