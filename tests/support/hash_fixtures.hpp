//
//  hash_fixtures.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Reads the fixtures in tests/fixtures/hash/: NIST's byte-oriented SHA-1, SHA-2, SHA-3 and SHAKE vectors
//  (nist/*.rsp, from CSRC's Cryptographic Algorithm Validation Program, unmodified), and the oracle files written
//  by scripts/generate_hash_fixtures.py from Python's hashlib, hmac and zlib, PyCryptodome, the blake3 package and
//  crcmod - never from core. Also the message and key generators that script uses, so a test can rebuild every
//  input from its length.
//

#ifndef snicholls_core_tests_support_hash_fixtures_hpp
#define snicholls_core_tests_support_hash_fixtures_hpp

#include <snicholls/core/hash/digest.hpp>
#include <snicholls/core/hash/wyhash.hpp>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <map>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#ifndef SNICHOLLS_CORE_FIXTURE_DIR
#error "SNICHOLLS_CORE_FIXTURE_DIR must name tests/fixtures (set by tests/CMakeLists.txt)"
#endif

namespace snicholls::test {

/// Bytes from a hexadecimal string (even length, either case).
inline std::vector<std::byte> unhex(std::string_view text)
{
    const auto nibble = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        throw std::runtime_error("not hexadecimal: " + std::string(1, c));
    };
    if (text.size() % 2 != 0) throw std::runtime_error("odd hexadecimal length");
    std::vector<std::byte> out(text.size() / 2);
    for (std::size_t i = 0; i < out.size(); ++i)
        out[i] = static_cast<std::byte>(nibble(text[2 * i]) * 16 + nibble(text[2 * i + 1]));
    return out;
}

/// Lowercase hexadecimal of bytes.
inline std::string hex(std::span<const std::byte> bytes)
{
    constexpr char digits[] = "0123456789abcdef";
    std::string out;
    out.reserve(2 * bytes.size());
    for (const std::byte b : bytes) {
        out += digits[static_cast<unsigned>(b) >> 4];
        out += digits[static_cast<unsigned>(b) & 15];
    }
    return out;
}

/// `n` bytes of splitmix64 output from `seed`: what generate_hash_fixtures.py's stream() writes.
inline std::vector<std::byte> splitmix_stream(std::uint64_t seed, std::size_t n)
{
    std::vector<std::byte> out(n);
    for (std::size_t i = 0; i < n; ++i)
        out[i] = static_cast<std::byte>(snicholls::splitmix64(seed + i / 8) >> (8 * (i % 8)));
    return out;
}

/// The oracle message of length n.
inline std::vector<std::byte> oracle_message(std::size_t n) { return splitmix_stream(std::uint64_t{n} << 32, n); }

/// The oracle key of length n.
inline std::vector<std::byte> oracle_key(std::size_t n) { return splitmix_stream((std::uint64_t{n} << 32) + 0x6b65790000000000ull, n); }

/// One oracle line: `name length params... hex`.
struct oracle_case
{
    std::size_t line = 0;
    std::size_t length = 0;
    std::vector<std::uint64_t> params;
    std::string hex;
};

/// Every case called `name` in tests/fixtures/hash/<file>. Throws if the file is missing or has no such case, so a
/// fixture that is not found can never pass by testing nothing.
inline std::vector<oracle_case> read_oracle(const std::string& file, const std::string& name)
{
    const std::string path = std::string(SNICHOLLS_CORE_FIXTURE_DIR) + "/hash/" + file;
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("hash fixture not found: " + path);
    std::vector<oracle_case> out;
    std::string text;
    std::size_t line = 0;
    while (std::getline(in, text)) {
        ++line;
        if (!text.empty() && text.back() == '\r') text.pop_back();
        if (text.empty() || text.front() == '#') continue;
        std::vector<std::string> fields;
        for (std::size_t start = 0;;) {
            const std::size_t space = text.find(' ', start);
            fields.push_back(text.substr(start, space == std::string::npos ? std::string::npos : space - start));
            if (space == std::string::npos) break;
            start = space + 1;
        }
        if (fields.size() < 3 || fields[0] != name) continue;
        oracle_case c;
        c.line = line;
        c.length = static_cast<std::size_t>(std::stoull(fields[1]));
        for (std::size_t i = 2; i + 1 < fields.size(); ++i)
            c.params.push_back(std::stoull(fields[i]));
        c.hex = fields.back();
        out.push_back(std::move(c));
    }
    if (out.empty()) throw std::runtime_error("no case " + name + " in " + path);
    return out;
}

/// The `key = value` records of a NIST .rsp file, in order, with the `[L = ...]` style headers as records too.
struct rsp_record
{
    std::map<std::string, std::string> fields;
};

inline std::vector<rsp_record> read_rsp(const std::string& file)
{
    const std::string path = std::string(SNICHOLLS_CORE_FIXTURE_DIR) + "/hash/nist/" + file;
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("NIST vector file not found: " + path);
    std::vector<rsp_record> out;
    rsp_record current;
    std::string text;
    const auto flush = [&] {
        if (!current.fields.empty()) out.push_back(std::move(current));
        current = {};
    };
    while (std::getline(in, text)) {
        if (!text.empty() && text.back() == '\r') text.pop_back();
        if (text.empty()) {
            flush();
            continue;
        }
        if (text.front() == '#') continue;
        if (text.front() == '[') {   // [L = 32], [Minimum Output Length (bits) = 128]
            flush();
            const std::size_t eq = text.find(" = ");
            if (eq != std::string::npos) current.fields[text.substr(1, eq - 1)] = text.substr(eq + 3, text.size() - eq - 4);
            flush();
            continue;
        }
        const std::size_t eq = text.find(" = ");
        if (eq != std::string::npos) current.fields[text.substr(0, eq)] = text.substr(eq + 3);
    }
    flush();
    if (out.empty()) throw std::runtime_error("NIST vector file is empty: " + path);
    return out;
}

/// The messages and digests of a ShortMsg file (`Len` in bits; `Len = 0` has the placeholder `Msg = 00`).
struct nist_message
{
    std::vector<std::byte> message;
    std::string digest;
};

inline std::vector<nist_message> read_nist_short(const std::string& file)
{
    std::vector<nist_message> out;
    for (const auto& r : read_rsp(file)) {
        const auto len = r.fields.find("Len");
        if (len == r.fields.end()) continue;
        nist_message m;
        m.message = unhex(r.fields.at("Msg"));
        m.message.resize(std::stoull(len->second) / 8);
        m.digest = r.fields.at("MD");
        out.push_back(std::move(m));
    }
    if (out.empty()) throw std::runtime_error("no messages in " + file);
    return out;
}

/// The digest of `message` hashed in three updates, cut at `a` and `b` (`a <= b <= size`).
template <typename H>
auto hash_in_parts(std::span<const std::byte> message, std::size_t a, std::size_t b)
{
    H h;
    h.update(message.first(a));
    h.update(message.subspan(a, b - a));
    h.update(message.subspan(b));
    return h.finish();
}

/// The digest of `message` hashed one byte at a time.
template <typename H>
auto hash_bytewise(std::span<const std::byte> message)
{
    H h;
    for (std::size_t i = 0; i < message.size(); ++i)
        h.update(message.subspan(i, 1));
    return h.finish();
}

/// SHAVS's Monte Carlo test for SHA-1 and SHA-2: from `seed`, 100 checkpoints of 1,000 hashes of the last three.
template <typename H>
std::vector<std::string> sha2_monte(std::vector<std::byte> seed)
{
    std::vector<std::string> out;
    for (int j = 0; j < 100; ++j) {
        std::vector<std::byte> md[3] = {seed, seed, seed};
        for (int i = 3; i < 1003; ++i) {
            std::vector<std::byte> m;
            for (const auto& part : md)
                m.insert(m.end(), part.begin(), part.end());
            const auto d = H::hash(m);
            md[0] = std::move(md[1]);
            md[1] = std::move(md[2]);
            md[2].assign(d.begin(), d.end());
        }
        seed = md[2];
        out.push_back(hex(seed));
    }
    return out;
}

/// SHA3VS's Monte Carlo test: from `seed`, 100 checkpoints of 1,000 chained hashes.
template <typename H>
std::vector<std::string> sha3_monte(std::vector<std::byte> md)
{
    std::vector<std::string> out;
    for (int j = 0; j < 100; ++j) {
        for (int i = 0; i < 1000; ++i) {
            const auto d = H::hash(md);
            md.assign(d.begin(), d.end());
        }
        out.push_back(hex(md));
    }
    return out;
}

/// The checkpoints a Monte file records (`MD` or `Output`), in order.
inline std::vector<std::string> read_monte_expected(const std::string& file, const char* field)
{
    std::vector<std::string> out;
    for (const auto& r : read_rsp(file))
        if (r.fields.count("COUNT") != 0) out.push_back(r.fields.at(field));
    if (out.size() != 100) throw std::runtime_error("expected 100 Monte checkpoints in " + file);
    return out;
}

/// The value of the first record holding `field` in a NIST file.
inline std::string read_rsp_field(const std::string& file, const std::string& field)
{
    for (const auto& r : read_rsp(file))
        if (const auto it = r.fields.find(field); it != r.fields.end()) return it->second;
    throw std::runtime_error("no " + field + " in " + file);
}

} // namespace snicholls::test

#endif /* snicholls_core_tests_support_hash_fixtures_hpp */
