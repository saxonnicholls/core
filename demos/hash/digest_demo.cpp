//
//  digest_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/hash/digest.hpp>: a digest checked at compile time against its published hexadecimal,
//  parsed back from text, ordered, truncated, printed, and the inputs a hash accepts. Exits non-zero if any value
//  is not the one documented.
//

#include <snicholls/core/hash/digest.hpp>
#include <snicholls/core/hash/sha2.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <set>
#include <span>
#include <string>
#include <vector>

namespace {

//! [digest]
// The compiler computes the digest and checks it against the published value
constexpr snicholls::digest<32> empty_sha256 = snicholls::sha256::hash("");
static_assert(empty_sha256 == snicholls::digest<32>::from_hex("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"));
static_assert(empty_sha256.hex().view().starts_with("e3b0c442"));

// Any one-byte sequence is an input, and a digest is one too
const std::vector<std::uint8_t> bytes{'a', 'b', 'c'};
const auto once = snicholls::sha256::hash(bytes);
const auto twice = snicholls::sha256::hash(once);   // Bitcoin's double SHA-256
//! [digest]

} // namespace

int main()
{
    std::printf("snicholls-core digest demo\n");
    bool ok = true;

    std::printf("  sha256(\"\")        = %s\n", empty_sha256.hex().c_str());
    std::printf("  sha256(\"abc\")     = %s\n", once.hex().c_str());
    std::printf("  sha256(sha256(..)) = %s\n", twice.hex().c_str());
    ok &= once == snicholls::sha256::hash("abc");
    ok &= twice.hex().view() == "4f8b42c22dd3729b519ba6f68d2da7cc5b2d606d05daed5ad5128cc03e6c6358";

    // Text both ways: from_hex accepts either case and an optional 0x, and refuses anything else
    const auto parsed = snicholls::digest<32>::from_hex("0xBA7816BF8F01CFEA414140DE5DAE2223B00361A396177A9CB410FF61F20015AD");
    std::printf("  from_hex(0xBA78...) == sha256(\"abc\"): %s\n", parsed == once ? "yes" : "no");
    ok &= parsed == once && !snicholls::digest<32>::from_hex("ba78").has_value() && !snicholls::digest<4>::from_hex("ba7z16bf");

    // A prefix is a digest of its own: the first four bytes, as a selector or a short id
    const auto prefix = once.first<4>();
    std::printf("  first<4>() = %s\n", prefix.hex().c_str());
    ok &= prefix.hex().view() == "ba7816bf";

    // Ordered for sorted containers, printed as hexadecimal, bytes readable and writable
    std::set<snicholls::digest<32>> sorted{once, twice, empty_sha256};
    ok &= sorted.begin()->hex().view().starts_with("4f8b") && sorted.size() == 3;
    std::cout << "  print: ";
    prefix.print(std::cout);
    std::cout << "\n";
    snicholls::digest<4> d = prefix;
    d[0] = std::byte{0};
    ok &= d.hex().view() == "007816bf" && d.size() == 4 && d.span().size() == 4 && *d.data() == std::byte{0} && d[3] == std::byte{0xbf};
    std::size_t count = 0;
    for (const std::byte b : d)
        count += b != std::byte{0};
    ok &= count == 3;

    // hash_input: what the hashes accept
    static_assert(snicholls::hash_input<std::string> && snicholls::hash_input<std::span<const std::byte>>);
    static_assert(snicholls::hash_input<snicholls::digest<32>> && !snicholls::hash_input<std::u16string>);
    static_assert(snicholls::hash_function<snicholls::sha256>);
    static_assert(std::is_same_v<snicholls::digest_of_t<snicholls::sha512>, snicholls::digest<64>>);
    static_assert(snicholls::is_digest_v<snicholls::digest<1>> && !snicholls::is_digest_v<std::array<std::byte, 1>>);

    std::printf(ok ? "digest demo: ok\n" : "digest demo: FAILED\n");
    return ok ? 0 : 1;
}
