//
//  string_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/metric/string.hpp>: the closest known name to a mistyped one by edit distance, edit
//  distances between token sequences of different types, and the sentinel for a sequence beyond MaxLength. Exits
//  non-zero if a result is not what it documents.
//

#include <snicholls/core/metric/string.hpp>

#include <array>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <span>
#include <string_view>
#include <vector>

namespace {

namespace me = snicholls::metric;

using namespace std::string_view_literals;

bool check(bool ok, const char* what)
{
    std::printf("  %-62s %s\n", what, ok ? "ok" : "FAILED");
    return ok;
}

} // namespace

int main()
{
    std::printf("snicholls-core string demo\n");
    bool ok = true;

    //! [string]
    // The closest known call sign to one received with a typo; the working row is sized at compile time
    constexpr me::levenshtein<16> edit{};
    const std::array known{"SPEEDBIRD"sv, "SHAMROCK"sv, "SPEEDWAY"sv, "SPRINGBOK"sv};
    const std::string_view received = "SPEEDBRID";
    std::size_t best = 0;
    for (std::size_t i = 1; i < known.size(); ++i)
        if (edit(received, known[i]) < edit(received, known[best]))
            best = i;

    // Any two sequences whose elements compare: token ids in a vector against a span, characters against an array
    const std::vector<int> sentence{7, 3, 9, 9, 2};
    const std::array<int, 4> other{7, 9, 9, 4};
    const std::size_t tokens = edit(sentence, std::span<const int>(other));
    const std::size_t mixed = edit("kitten"sv, std::array<char, 7>{'s', 'i', 't', 't', 'i', 'n', 'g'});
    //! [string]

    std::printf("  \"%.*s\" is nearest \"%.*s\" (%zu edits); tokens %zu edits; kitten-sitting %zu\n", static_cast<int>(received.size()), received.data(),
                static_cast<int>(known[best].size()), known[best].data(), edit(received, known[best]), tokens, mixed);
    ok &= check(best == 0 && edit(received, known[0]) == 2, "SPEEDBRID is two edits from SPEEDBIRD, the nearest");
    ok &= check(tokens == 2, "7 3 9 9 2 -> 7 9 9 4: delete 3, substitute 2 by 4");
    ok &= check(mixed == 3, "kitten -> sitting: 3");
    ok &= check(edit("flaw"sv, "lawn"sv) == 2 && edit(""sv, "abc"sv) == 3 && edit("same"sv, "same"sv) == 0, "flaw/lawn 2, empty/abc 3, identical 0");
    static_assert(me::levenshtein<8>{}("kitten"sv, "sitting"sv) == 3, "evaluated by the compiler");
    // The shorter sequence must fit the row; if it does not, the answer is the largest size_t, never a wrong number
    ok &= check(me::levenshtein<3>{}("abc"sv, "abcdefgh"sv) == 5, "MaxLength 3: the shorter fits, the longer may not");
    ok &= check(me::levenshtein<3>{}("abcd"sv, "abcd"sv) == std::numeric_limits<std::size_t>::max(), "both longer than MaxLength: the sentinel");

    std::printf(ok ? "string demo: ok\n" : "string demo: FAILED\n");
    return ok ? 0 : 1;
}
