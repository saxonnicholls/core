//
//  quant_rows.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Test-only: the rows the quantisation fixtures are made from, shared by the fixture generators
//  (scripts/generate_quant_*_fixture.cpp, which include this file and link ggml) and by the tests, so the inputs
//  are never stored, only a hash of each. Every number is made with integer operations and exact float scalings,
//  so the rows are the same bits on every compiler and machine. Also the FNV-1a hash the fixtures record and a
//  reader for their lines.
//

#ifndef snicholls_core_tests_ml_quant_rows_hpp
#define snicholls_core_tests_ml_quant_rows_hpp

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace quant_rows {

// The row kinds: what each case of a fixture quantises
enum kind : int {
    uniform,       // uniform in [-1, 1)
    weights,       // the sum of three uniforms times 0.02: a trained weight's shape
    outliers,      // uniform with two large outliers
    zero_half,     // the first half zero
    tiny,          // 1e-4 scale, then 1e-30: subnormal and vanishing float16 scales
    huge,          // 6e6 scale: float16 scales that overflow
    halves,        // multiples of one half: ties in rounding
    negative_max,  // uniform with -3 at two places: the extreme is negative
    constant,      // every number 0.3
    spike,         // zeros and one 1
    negative,      // every number -2.5
    ramp,          // (i - n/2) / n
    heavy,         // u^5 · 10: heavy tails
    positive,      // |uniform|
    subnormal,     // u · 2^-130: subnormal floats
    with_nan,      // uniform with a NaN and an infinity
    kinds
};

inline const char* name(int k)
{
    static const char* names[] = {"uniform", "weights", "outliers", "zero_half", "tiny", "huge", "halves", "negative_max",
                                  "constant", "spike", "negative", "ramp", "heavy", "positive", "subnormal", "with_nan"};
    return names[k];
}

struct stream {
    std::uint64_t state;
    explicit stream(std::uint64_t seed) : state(seed * 0x9e3779b97f4a7c15ULL + 0x243f6a8885a308d3ULL) {}
    std::uint64_t next()
    {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return state;
    }
    // uniform in [-1, 1): a 24-bit integer scaled by 2^-23, exact
    float u()
    {
        const auto k = static_cast<std::int32_t>(next() >> 40) - (1 << 23);
        return static_cast<float>(k) * 0x1p-23f;
    }
};

inline std::vector<float> row(int k, std::size_t n, std::uint64_t seed)
{
    stream s(seed * 31 + static_cast<std::uint64_t>(k));
    std::vector<float> r(n);
    for (std::size_t i = 0; i < n; ++i) {
        const float u = s.u();
        switch (k) {
        case uniform: r[i] = u; break;
        case weights: {
            const float a = u + s.u();
            const float b = a + s.u();
            r[i] = b * 0.02f;
            break;
        }
        case outliers: r[i] = i == 5 ? 100.0f : i == n / 2 + 40 ? -80.0f : u; break;
        case zero_half: r[i] = i < n / 2 ? 0.0f : u; break;
        case tiny: r[i] = u * (i < n / 2 ? 1e-4f : 1e-30f); break;
        case huge: r[i] = u * 6e6f; break;
        case halves: r[i] = static_cast<float>(static_cast<int>(s.next() % 255) - 127) * 0.5f; break;
        case negative_max: r[i] = i == 9 || i == n / 2 + 50 ? -3.0f : u; break;
        case constant: r[i] = 0.3f; break;
        case spike: r[i] = i == 77 ? 1.0f : 0.0f; break;
        case negative: r[i] = -2.5f; break;
        case ramp: r[i] = (static_cast<float>(i) - static_cast<float>(n / 2)) / static_cast<float>(n); break;
        case heavy: {
            const float u2 = u * u;
            const float u4 = u2 * u2;
            const float u5 = u4 * u;
            r[i] = u5 * 10.0f;
            break;
        }
        case positive: r[i] = u < 0 ? -u : u; break;
        case subnormal: r[i] = u * 0x1p-130f; break;
        case with_nan:
            r[i] = i == 7 ? std::numeric_limits<float>::quiet_NaN() : i == n / 2 + 3 ? std::numeric_limits<float>::infinity() : u;
            break;
        }
    }
    return r;
}

// An importance row: positive weights like an importance matrix's (mean squared activations), some zero
inline std::vector<float> importance(std::size_t n, std::uint64_t seed)
{
    stream s(seed * 977 + 5);
    std::vector<float> w(n);
    for (std::size_t i = 0; i < n; ++i) {
        const float u = s.u();
        const float u2 = u * u;
        w[i] = i % 61 == 3 ? 0.0f : u2 + 0.01f;
    }
    return w;
}

inline std::uint64_t fnv(const void* p, std::size_t bytes)
{
    std::uint64_t h = 0xcbf29ce484222325ULL;
    const auto* b = static_cast<const unsigned char*>(p);
    for (std::size_t i = 0; i < bytes; ++i) {
        h ^= b[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

template <typename T>
std::uint64_t fnv(const std::vector<T>& v)
{
    return fnv(v.data(), v.size() * sizeof(T));
}

// Floats hashed with every NaN as one pattern: the sign and payload of a NaN the hardware makes differ (x86, ARM)
inline std::uint64_t fnv_floats(const std::vector<float>& v)
{
    std::vector<std::uint32_t> bits(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) {
        std::uint32_t b;
        std::memcpy(&b, &v[i], 4);
        bits[i] = v[i] != v[i] ? 0x7fc00000u : b;
    }
    return fnv(bits);
}

inline std::uint32_t float_bits(float f)
{
    std::uint32_t b;
    std::memcpy(&b, &f, 4);
    return f != f ? 0x7fc00000u : b;
}

// One fixture line: "<format> <kind> key=hex key=hex ...", read into a map
struct line {
    std::string format;
    int kind = 0;
    std::map<std::string, std::uint64_t> values;
};

inline bool parse(const std::string& text, line& out)
{
    if (text.empty() || text[0] == '#')
        return false;
    std::istringstream ls(text);
    std::string kind;
    ls >> out.format >> kind;
    out.kind = std::stoi(kind);
    out.values.clear();
    std::string word;
    while (ls >> word) {
        const auto eq = word.find('=');
        out.values[word.substr(0, eq)] = std::stoull(word.substr(eq + 1), nullptr, 16);
    }
    return true;
}

} // namespace quant_rows

#endif // snicholls_core_tests_ml_quant_rows_hpp
