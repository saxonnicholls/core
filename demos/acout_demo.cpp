//
//  acout_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/acout.hpp>: four threads printing through acout to one
//  stream, every line arriving whole; the format-string forms; a message to a
//  file. Exits non-zero if any line is broken or any text is not the one
//  documented.
//

#include <snicholls/core/acout.hpp>

#include <array>
#include <cstdio>
#include <iomanip>
#include <map>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

namespace {

//! [printable]
// A type that prints itself: one member, and that is the whole contract. The classic form
// takes a stream ...
struct fill {
    const char* symbol;
    int quantity;
    double price;

    void print(std::ostream& os) const { os << symbol << " x" << quantity << " @ " << std::fixed << std::setprecision(2) << price; }
};

// ... and the better form takes the message: format strings inside, and the whole object is
// written in one call wherever it is streamed.
struct quote {
    const char* symbol;
    double bid;
    double ask;

    void print(snicholls::acout& out) const { out.print("{} {:.2f}/{:.2f}", symbol, bid, ask); }
};

// Plain streams get the operator by opting in, as with magic_enum's iostream_operators.
using snicholls::print_operators::operator<<;
//! [printable]

} // namespace

int main()
{
    std::printf("snicholls-core acout demo\n");
    bool ok = true;

    //! [acout]
    // One message, assembled first, written once: another thread's message cannot land in the
    // middle of it. The temporary's destructor does the write at the end of the statement.
    snicholls::acout{} << "  balance " << std::fixed << std::setprecision(2) << 1234.5 << " in thread " << 0 << '\n';
    snicholls::acerr{} << "  (to stderr, the same way)\n";

    // Four threads, one stream, 100 lines each: counted afterwards, every line is intact.
    std::ostringstream shared;
    std::vector<std::thread> threads;
    for (int t = 1; t <= 4; ++t)
        threads.emplace_back([&shared, t] {
            for (int i = 0; i < 100; ++i)
                snicholls::acout(shared) << "thread " << t << " says hello number " << i << '\n';
        });
    for (auto& th : threads)
        th.join();
    //! [acout]

    std::istringstream lines(shared.str());
    std::string line;
    int intact = 0, total = 0;
    while (std::getline(lines, line)) {
        ++total;
        int t = 0, i = 0;
        intact += std::sscanf(line.c_str(), "thread %d says hello number %d", &t, &i) == 2 && t >= 1 && t <= 4 && i >= 0 && i < 100 ? 1 : 0;
    }
    std::printf("  %d lines from 4 threads, %d intact\n", total, intact);
    ok &= total == 400 && intact == 400;

    //! [print]
    // The one-line forms: a format string checked at compile time, written in one call.
    snicholls::println("  {} of {} orders filled ({:.1f}%)", 7, 8, 87.5);
    snicholls::print(stderr, "  {}: {}\n", "warning", "stderr through fwrite");

    // Into a message, mixed with streaming, emitted when it goes out of scope.
    {
        snicholls::acout message;
        message << "  order " << 42 << ": ";
        message.print("{} @ {:.2f}", "AAPL", 189.25).println(" {}", "filled");
    }
    //! [print]

    std::ostringstream check;
    snicholls::println(check, "{} of {} orders filled ({:.1f}%)", 7, 8, 87.5);
    ok &= check.str() == "7 of 8 orders filled (87.5%)\n";

    //! [printable_use]
    // Through acout, print() is found by the concept; through a plain stream, by the operator above;
    // println(value) is the one-liner.
    const fill f{"AAPL", 100, 189.25};
    const quote q{"AAPL", 189.20, 189.30};
    snicholls::acout{} << "  filled: " << f << " at " << q << '\n';
    snicholls::println(q);
    std::ostringstream plain;
    plain << f << "; " << q;
    //! [printable_use]
    ok &= plain.str() == "AAPL x100 @ 189.25; AAPL 189.20/189.30";

    //! [containers]
    // Containers, pairs, tuples, optionals and type lists print as they are, recursively, through
    // the same engine: into an acout always, onto a plain stream with the operator above in scope.
    const std::map<std::string, std::vector<int>> book{{"bids", {100, 99}}, {"asks", {101, 102}}};
    const std::vector<std::pair<std::string, std::optional<double>>> marks{{"AAPL", 189.25}, {"PRIV", std::nullopt}};
    snicholls::acout{} << "  book " << book << '\n';
    snicholls::println(marks);
    snicholls::acout{} << "  types " << snicholls::meta::type_list<int, double>{} << '\n';
    std::ostringstream shown;
    shown << book << " " << std::tuple{1, 'a', "two"};
    //! [containers]
    ok &= shown.str() == R"({"asks": [101, 102], "bids": [100, 99]} (1, 'a', "two"))";

    //! [colour]
    // A colour holds until colour_reset, or the end of the message, which always ends it.
    using namespace snicholls::colours;
    snicholls::acout{} << "  " << green << "filled" << snicholls::colour_reset << " 100 AAPL\n";

    // One value in a colour, in a container too; the message returns to the colour it was in.
    const int up = 3, down = -2;
    const std::array moves{snicholls::styled(green, up), snicholls::styled(red, down)};
    snicholls::acout{} << "  moves " << moves << ' ' << snicholls::styled(snicholls::rgb(0x8250df), "any rgb") << '\n';

    // A whole line, as a format string.
    snicholls::println(yellow, "  {} of {} orders late", 1, 8);

    // A string stream gets the plain text; set_colour_mode(always) is how to ask for the codes anyway.
    std::ostringstream plain_text;
    snicholls::acout(plain_text) << red << "no codes here";
    //! [colour]
    ok &= plain_text.str() == "no codes here";

    snicholls::set_colour_mode(snicholls::colour_mode::always);
    std::ostringstream with_codes;
    snicholls::acout(with_codes) << red << "x";
    snicholls::set_colour_mode(snicholls::colour_mode::automatic);
    ok &= with_codes.str() == "\x1b[31mx\x1b[0m";

    std::FILE* file = std::tmpfile();
    ok &= file != nullptr;
    if (file != nullptr) {
        snicholls::println(file, "line {}", 1);
        std::rewind(file);
        char read[32] = {};
        const auto n = std::fread(read, 1, sizeof read - 1, file);
        ok &= std::string(read, n) == "line 1\n";
        std::fclose(file);
    }

    std::printf(ok ? "acout demo: ok\n" : "acout demo: FAILED\n");
    return ok ? 0 : 1;
}
