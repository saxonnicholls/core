//
//  phone_number_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  snicholls::parse_phone_number and snicholls::to_chars against the same work written by hand: one scan of the
//  text into a digit array, with the same separator rules, the same country code table (detail::e164_code_flags)
//  and the same checks in the same order, keeping the digits as characters instead of packing them into a word;
//  and writing the text by copying those stored characters. 4,096 numbers under 40 country codes, each written in
//  one of five forms. main() runs a fairness check first and refuses to benchmark if the two sides disagree on any
//  input (CLAUDE.md "Zero overhead").
//

#include <snicholls/core/phone_number.hpp>

#include <benchmark/benchmark.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

namespace {

std::vector<std::string> make_inputs()
{
    constexpr std::array<const char*, 40> codes{"1",   "7",   "20",  "27",  "31",  "33",  "34",  "39",
                                                "44",  "49",  "52",  "55",  "61",  "62",  "64",  "65",
                                                "81",  "82",  "86",  "90",  "91",  "212", "234", "254",
                                                "351", "353", "358", "380", "420", "502", "593", "673",
                                                "852", "880", "886", "966", "971", "972", "800", "998"};
    std::vector<std::string> out;
    std::uint64_t x = 0x243f6a8885a308d3ULL;   // a fixed xorshift stream: identical data for both
    auto next = [&] { x ^= x << 13; x ^= x >> 7; x ^= x << 17; return x; };
    for (std::size_t i = 0; i < 4096; ++i) {
        const std::string cc = codes[next() % codes.size()];
        std::string nsn = std::to_string(1 + next() % 9);
        const std::size_t length = 7 + next() % (14 - cc.size() - 6);
        while (nsn.size() < length)
            nsn += static_cast<char>('0' + next() % 10);
        const std::string a = nsn.substr(0, 3), b = nsn.substr(3, 3), c = nsn.substr(6);
        switch (i % 5) {
        case 0: out.push_back("+" + cc + nsn); break;
        case 1: out.push_back("+" + cc + " " + a + " " + b + " " + c); break;
        case 2: out.push_back("+" + cc + "-" + a + "-" + b + "-" + c); break;
        case 3: out.push_back("(+" + cc + ") " + a + b + c); break;
        default: out.push_back("00 " + cc + " " + a + " " + b + " " + c); break;
        }
    }
    return out;
}

const std::vector<std::string> inputs = make_inputs();

// By hand: the digits kept as characters
struct hand_number
{
    std::array<char, 15> digits{};
    std::uint8_t length = 0;
    std::uint8_t cc_length = 0;
};

bool parse_by_hand(std::string_view text, hand_number& out) noexcept
{
    hand_number h;
    std::size_t n = 0;
    bool international = false;
    bool open = false;
    for (std::size_t i = 0; i < text.size(); ++i) {
        const char c = text[i];
        if (c >= '0' && c <= '9') {
            if (!international) {
                if (c != '0' || i + 1 == text.size() || text[i + 1] != '0') return false;
                international = true;
                ++i;
                continue;
            }
            if (n < h.digits.size()) h.digits[n] = c;
            ++n;
        } else if (c == '+') {
            if (international) return false;
            international = true;
        } else if (c == '(') {
            if (open) return false;
            open = true;
        } else if (c == ')') {
            if (!open) return false;
            open = false;
        } else if (c != ' ' && c != '-' && c != '.' && c != '/') {
            return false;
        }
    }
    if (open || !international || n > 15 || n == 0 || h.digits[0] == '0') return false;
    unsigned code = 0;
    for (std::size_t k = 0; k < 3 && k < n && h.cc_length == 0; ++k) {
        code = code * 10 + static_cast<unsigned>(h.digits[k] - '0');
        if ((snicholls::detail::e164_code_flags[code] & 1U) != 0) h.cc_length = static_cast<std::uint8_t>(k + 1);
    }
    if (h.cc_length == 0 || n - h.cc_length < 2) return false;
    if (h.digits[h.cc_length] == '0' && (snicholls::detail::e164_code_flags[code] & 2U) != 0) return false;
    h.length = static_cast<std::uint8_t>(n);
    out = h;
    return true;
}

char* write_by_hand(char* first, const hand_number& h) noexcept
{
    *first = '+';
    std::memcpy(first + 1, h.digits.data(), h.length);
    return first + 1 + h.length;
}

void BM_PhoneNumber_Parse_Core(benchmark::State& state)
{
    for (auto _ : state)
        for (const auto& text : inputs)
            benchmark::DoNotOptimize(snicholls::parse_phone_number(text));
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(inputs.size()));
}
BENCHMARK(BM_PhoneNumber_Parse_Core);

void BM_PhoneNumber_Parse_HandWritten(benchmark::State& state)
{
    for (auto _ : state)
        for (const auto& text : inputs) {
            hand_number h;
            benchmark::DoNotOptimize(parse_by_hand(text, h));
            benchmark::DoNotOptimize(h);
        }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(inputs.size()));
}
BENCHMARK(BM_PhoneNumber_Parse_HandWritten);

std::vector<snicholls::phone_number> core_numbers()
{
    std::vector<snicholls::phone_number> out;
    for (const auto& text : inputs)
        out.push_back(*snicholls::parse_phone_number(text));
    return out;
}

std::vector<hand_number> hand_numbers()
{
    std::vector<hand_number> out(inputs.size());
    for (std::size_t i = 0; i < inputs.size(); ++i)
        (void)parse_by_hand(inputs[i], out[i]);
    return out;
}

void BM_PhoneNumber_ToChars_Core(benchmark::State& state)
{
    const auto numbers = core_numbers();
    std::array<char, 16> buf{};
    for (auto _ : state)
        for (const auto& p : numbers) {
            benchmark::DoNotOptimize(snicholls::to_chars(buf.data(), buf.data() + buf.size(), p).ptr);
            benchmark::ClobberMemory();
        }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(numbers.size()));
}
BENCHMARK(BM_PhoneNumber_ToChars_Core);

void BM_PhoneNumber_ToChars_HandWritten(benchmark::State& state)
{
    const auto numbers = hand_numbers();
    std::array<char, 16> buf{};
    for (auto _ : state)
        for (const auto& h : numbers) {
            benchmark::DoNotOptimize(write_by_hand(buf.data(), h));
            benchmark::ClobberMemory();
        }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(numbers.size()));
}
BENCHMARK(BM_PhoneNumber_ToChars_HandWritten);

// Every input parses on both sides to the same text; a refusal on either side, or any difference, stops the run
bool fair()
{
    for (const auto& text : inputs) {
        const auto r = snicholls::parse_phone_number(text);
        hand_number h;
        if (!r.is_ok() || !parse_by_hand(text, h)) {
            std::fprintf(stderr, "fairness: %s refused\n", text.c_str());
            return false;
        }
        std::array<char, 16> a{}, b{};
        const auto end_a = snicholls::to_chars(a.data(), a.data() + a.size(), *r).ptr;
        const auto end_b = write_by_hand(b.data(), h);
        if (std::string_view(a.data(), static_cast<std::size_t>(end_a - a.data())) !=
                std::string_view(b.data(), static_cast<std::size_t>(end_b - b.data())) ||
            r->country_code_length() != h.cc_length) {
            std::fprintf(stderr, "fairness: %s differs\n", text.c_str());
            return false;
        }
    }
    for (const char* bad : {"0412 345 678", "+999 1234", "+44 020 7946 0000", "+1-800-FLOWERS", "+1234567890123456"}) {
        hand_number h;
        if (snicholls::parse_phone_number(bad).is_ok() || parse_by_hand(bad, h)) {
            std::fprintf(stderr, "fairness: %s accepted\n", bad);
            return false;
        }
    }
    return true;
}

} // namespace

int main(int argc, char** argv)
{
    if (!fair()) return 1;
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
