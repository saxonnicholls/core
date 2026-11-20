//
//  sha2_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/hash/sha2.hpp>: a content address computed at compile time, a file checksummed in
//  pieces as it arrives, and every SHA-2 variant on the same input. Exits non-zero if any value is not the one
//  documented.
//

#include <snicholls/core/hash/sha2.hpp>

#include <cstdio>
#include <string>
#include <string_view>

namespace {

//! [sha256]
// A content address the compiler computes: a build can refuse a payload whose hash changed
constexpr auto schema_id = snicholls::sha256::hash("schema v3: id u64, price i64, qty i64");
static_assert(schema_id.first<4>().hex().view() == "13f63a67");

// Data that arrives in pieces is hashed as it arrives: the digest is the one-shot digest
std::string checksum(const std::string& header, const std::string& body)
{
    snicholls::sha256 h;
    h.update(header).update(body);
    return std::string(h.finish().hex().view());
}
//! [sha256]

} // namespace

int main()
{
    std::printf("snicholls-core sha2 demo\n");
    bool ok = true;

    std::printf("  schema id         = %s\n", schema_id.hex().c_str());
    const std::string pieces = checksum("GET / HTTP/1.1\r\n", "\r\n");
    std::printf("  checksum(pieces)  = %s\n", pieces.c_str());
    ok &= pieces == snicholls::sha256::hash(std::string_view("GET / HTTP/1.1\r\n\r\n")).hex().view();

    // Every variant on "abc": FIPS 180-4's examples
    std::printf("  sha224(abc)       = %s\n", snicholls::sha224::hash("abc").hex().c_str());
    std::printf("  sha384(abc)       = %s\n", snicholls::sha384::hash("abc").hex().c_str());
    std::printf("  sha512(abc)       = %s\n", snicholls::sha512::hash("abc").hex().c_str());
    std::printf("  sha512/224(abc)   = %s\n", snicholls::sha512_224::hash("abc").hex().c_str());
    std::printf("  sha512/256(abc)   = %s\n", snicholls::sha512_256::hash("abc").hex().c_str());
    ok &= snicholls::sha224::hash("abc").hex().view() == "23097d223405d8228642a477bda255b32aadbce4bda0b3f7e36c9da7";
    ok &= snicholls::sha384::hash("abc").hex().view().starts_with("cb00753f45a35e8b");
    ok &= snicholls::sha512::hash("abc").hex().view().starts_with("ddaf35a193617aba");
    ok &= snicholls::sha512_224::hash("abc").hex().view() == "4634270f707b6a54daae7530460842e20e37ed265ceee9a43e8924aa";
    ok &= snicholls::sha512_256::hash("abc").hex().view() == "53048e2681941ef99b2e29b76b4c7dabe4c2d0c634fc6d46e0e2f13107e7af23";
    ok &= snicholls::sha256::block_size == 64 && snicholls::sha512::block_size == 128 && snicholls::sha384::digest_size == 48;

    std::printf(ok ? "sha2 demo: ok\n" : "sha2 demo: FAILED\n");
    return ok ? 0 : 1;
}
