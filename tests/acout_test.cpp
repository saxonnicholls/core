//
//  acout_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/acout.hpp>: the message is written once and whole,
//  the streaming and format forms, flush requests, the FILE* target, the
//  256-byte spill, and messages from many threads arriving intact.
//

#include <snicholls/core/acout.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <span>
#include <sstream>
#include <streambuf>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

namespace {

using snicholls::acerr;
using snicholls::acout;
using snicholls::print;
using snicholls::println;

// A target that records every write it receives and every flush, so "one write per message" is observable.
class recording_buffer : public std::streambuf {
public:
    std::vector<std::string> writes;
    int flushes = 0;

protected:
    std::streamsize xsputn(const char* s, std::streamsize n) override
    {
        writes.emplace_back(s, static_cast<std::size_t>(n));
        return n;
    }
    int_type overflow(int_type ch) override
    {
        if (!traits_type::eq_int_type(ch, traits_type::eof()))
            writes.emplace_back(1, traits_type::to_char_type(ch));
        return traits_type::not_eof(ch);
    }
    int sync() override
    {
        ++flushes;
        return 0;
    }
};

TEST(AcoutStreaming, WritesTheWholeMessageInOneCallWhenTheStatementEnds)
{
    recording_buffer buffer;
    std::ostream target(&buffer);
    acout(target) << "x = " << 42 << ", y = " << 2.5 << '\n';
    ASSERT_EQ(buffer.writes.size(), 1u);
    EXPECT_EQ(buffer.writes[0], "x = 42, y = 2.5\n");
    EXPECT_EQ(buffer.flushes, 0);
}

TEST(AcoutStreaming, ManipulatorsApplyToTheMessageNotTheTarget)
{
    std::ostringstream target;
    acout(target) << std::hex << 255 << ' ' << std::setw(4) << std::setfill('0') << 7;
    EXPECT_EQ(target.str(), "ff 0007");
    target << 255;   // the target's own state is untouched
    EXPECT_EQ(target.str(), "ff 0007255");
}

TEST(AcoutStreaming, EndlAndFlushRequestAFlushAfterTheWrite)
{
    recording_buffer buffer;
    std::ostream target(&buffer);
    acout(target) << "line" << std::endl;
    ASSERT_EQ(buffer.writes.size(), 1u);
    EXPECT_EQ(buffer.writes[0], "line\n");
    EXPECT_EQ(buffer.flushes, 1);

    acout(target) << "more" << std::flush;
    EXPECT_EQ(buffer.flushes, 2);

    acout(target).flush();   // nothing to write, the flush still happens
    EXPECT_EQ(buffer.flushes, 3);
    EXPECT_EQ(buffer.writes.size(), 2u);
}

TEST(AcoutFormat, PrintAndPrintlnFormatIntoTheMessage)
{
    std::ostringstream target;
    acout(target).print("{} + {} = {}", 1, 2, 3).println(" ({:.2f})", 3.0);
    EXPECT_EQ(target.str(), "1 + 2 = 3 (3.00)\n");

    print(target, "{:>5}|", "ab");
    println(target, "{:#x}", 255);
    EXPECT_EQ(target.str(), "1 + 2 = 3 (3.00)\n   ab|0xff\n");
}

TEST(AcoutFormat, StreamingAndFormattingMix)
{
    std::ostringstream target;
    acout message(target);
    message << "[" << 7 << "] ";
    message.print("{}%", 50).println(" done");
    EXPECT_EQ(message.message(), "[7] 50% done\n");
    EXPECT_EQ(target.str(), "");   // not yet written
    message.emit();
    EXPECT_EQ(target.str(), "[7] 50% done\n");
    EXPECT_EQ(message.message(), "");   // empty and reusable
    message << "again";
    message.emit();
    EXPECT_EQ(target.str(), "[7] 50% done\nagain");
}

TEST(AcoutBuffer, LongMessagesSpillAndArriveWholeInOneWrite)
{
    recording_buffer buffer;
    std::ostream target(&buffer);
    const std::string a(200, 'a'), b(300, 'b'), c(1000, 'c');
    acout(target) << a << b << c;
    ASSERT_EQ(buffer.writes.size(), 1u);
    EXPECT_EQ(buffer.writes[0], a + b + c);
    EXPECT_EQ(buffer.writes[0].size(), 1500u);
}

TEST(AcoutBuffer, EmptyMessageWritesNothing)
{
    recording_buffer buffer;
    std::ostream target(&buffer);
    { acout nothing(target); }
    EXPECT_TRUE(buffer.writes.empty());
    EXPECT_EQ(buffer.flushes, 0);
}

TEST(AcoutFile, WritesToACStreamWithOneFwrite)
{
    std::FILE* file = std::tmpfile();
    ASSERT_NE(file, nullptr);
    acout(file) << "to a file " << 1 << '\n';
    print(file, "{}-{}", 2, 3);
    println(file, "");
    std::rewind(file);
    char read[64] = {};
    const auto n = std::fread(read, 1, sizeof read - 1, file);
    EXPECT_EQ(std::string_view(read, n), "to a file 1\n2-3\n");
    std::fclose(file);
}

TEST(AcoutTargets, AcerrGoesToCerr)
{
    std::stringbuf captured;
    std::streambuf* const previous = std::cerr.rdbuf(&captured);
    acerr{} << "error " << 7 << '\n';
    std::cerr.rdbuf(previous);
    EXPECT_EQ(captured.str(), "error 7\n");
}

TEST(AcoutTargets, DefaultTargetIsCout)
{
    std::stringbuf captured;
    std::streambuf* const previous = std::cout.rdbuf(&captured);
    acout{} << "out " << 1 << '\n';
    println("{}", "fmt");
    std::cout.rdbuf(previous);
    EXPECT_EQ(captured.str(), "out 1\nfmt\n");
}

TEST(AcoutThreads, MessagesFromManyThreadsNeverInterleave)
{
    // Eight threads, each writing 500 lines of its own pattern and varying length, through one stream.
    std::ostringstream target;
    std::vector<std::thread> threads;
    for (int t = 0; t < 8; ++t)
        threads.emplace_back([&target, t] {
            for (int i = 0; i < 500; ++i)
                acout(target) << "thread " << t << " line " << i << ' ' << std::string(static_cast<std::size_t>(i % 300), static_cast<char>('a' + t))
                              << " end\n";
        });
    for (auto& th : threads)
        th.join();

    // Every line must be exactly one message, character for character: read its thread and index,
    // rebuild what that message was, and compare.
    std::istringstream lines(target.str());
    std::string line;
    int count = 0;
    while (std::getline(lines, line)) {
        ++count;
        int t = -1, i = -1;
        ASSERT_EQ(std::sscanf(line.c_str(), "thread %d line %d ", &t, &i), 2) << line;
        ASSERT_TRUE(t >= 0 && t < 8 && i >= 0 && i < 500) << line;
        const std::string expected = "thread " + std::to_string(t) + " line " + std::to_string(i) + ' ' +
                                     std::string(static_cast<std::size_t>(i % 300), static_cast<char>('a' + t)) + " end";
        ASSERT_EQ(line, expected);
    }
    EXPECT_EQ(count, 8 * 500);
}

// ---- printable: print(std::ostream&) const is the whole contract ----------------------------------

struct order {
    int id;
    double price;
    void print(std::ostream& os) const { os << "order " << id << " @ " << price; }
};

struct loud {   // both a print member and its own operator<<: acout takes print, a plain stream the operator
    void print(std::ostream& os) const { os << "print"; }
    friend std::ostream& operator<<(std::ostream& os, const loud&) { return os << "operator"; }
};

struct trade {   // the acout form: format strings inside print, and one write for the whole object
    std::string symbol;
    int quantity;
    double price;
    void print(acout& out) const { out.print("{} x{} @ {:.2f}", symbol, quantity, price); }
};

struct both_forms {
    void print(std::ostream& os) const { os << "stream form"; }
    void print(acout& out) const { out << "acout form"; }
};

TEST(AcoutPrintable, AcoutStreamsAPrintableThroughItsPrintMember)
{
    std::ostringstream target;
    acout(target) << "[" << order{7, 1.5} << "]";
    EXPECT_EQ(target.str(), "[order 7 @ 1.5]");
}

TEST(AcoutPrintable, TheAcoutFormFormatsIntoTheMessageAndWinsOverTheStreamForm)
{
    recording_buffer buffer;
    std::ostream target(&buffer);
    acout(target) << trade{"AAPL", 100, 189.25} << " | " << both_forms{};
    ASSERT_EQ(buffer.writes.size(), 1u);
    EXPECT_EQ(buffer.writes[0], "AAPL x100 @ 189.25 | acout form");
}

TEST(AcoutPrintable, PrintMemberWinsOverTheTypesOwnOperatorInAcoutOnly)
{
    std::ostringstream target;
    acout(target) << loud{};
    EXPECT_EQ(target.str(), "print");
    std::ostringstream plain;
    plain << loud{};   // the type's own operator, print_operators not in scope here
    EXPECT_EQ(plain.str(), "operator");
}

TEST(AcoutPrintable, PrintAndPrintlnTakeAPrintableValue)
{
    recording_buffer buffer;
    std::ostream target(&buffer);
    print(target, trade{"MSFT", 5, 400.0});
    println(target, order{1, 2.0});
    ASSERT_EQ(buffer.writes.size(), 2u);
    EXPECT_EQ(buffer.writes[0], "MSFT x5 @ 400.00");
    EXPECT_EQ(buffer.writes[1], "order 1 @ 2\n");

    std::stringbuf captured;
    std::streambuf* const previous = std::cout.rdbuf(&captured);
    println(trade{"X", 1, 1.0});
    std::cout.rdbuf(previous);
    EXPECT_EQ(captured.str(), "X x1 @ 1.00\n");

    std::FILE* file = std::tmpfile();
    ASSERT_NE(file, nullptr);
    println(file, order{3, 4.0});
    std::rewind(file);
    char read[32] = {};
    const auto n = std::fread(read, 1, sizeof read - 1, file);
    EXPECT_EQ(std::string_view(read, n), "order 3 @ 4\n");
    std::fclose(file);
}

TEST(AcoutPrintable, EndlInsidePrintIsAFlushRequest)
{
    struct ends_lines {
        void print(std::ostream& os) const { os << "line" << std::endl; }
    };
    recording_buffer buffer;
    std::ostream target(&buffer);
    acout(target) << ends_lines{};
    ASSERT_EQ(buffer.writes.size(), 1u);
    EXPECT_EQ(buffer.writes[0], "line\n");
    EXPECT_EQ(buffer.flushes, 1);
}

TEST(AcoutPrintable, PlainStreamsGetBothFormsByOptingIn)
{
    using snicholls::print_operators::operator<<;
    std::ostringstream os;
    os << order{1, 2.0} << " and " << trade{"GOOG", 2, 3.0};
    EXPECT_EQ(os.str(), "order 1 @ 2 and GOOG x2 @ 3.00");

    recording_buffer buffer;   // the acout form is one write even onto a plain stream
    std::ostream target(&buffer);
    target << trade{"A", 1, 1.0};
    ASSERT_EQ(buffer.writes.size(), 1u);
    EXPECT_EQ(buffer.writes[0], "A x1 @ 1.00");
}

// ---- the engine: containers, tuples, optionals, variants, results, type lists --------------------

std::string printed(const auto& value)
{
    std::ostringstream os;
    snicholls::print_to(os, value);
    return os.str();
}

enum class side { buy = 1, sell = 2 };
enum class small : std::uint8_t { seven = 7 };
enum class own_operator { a };
std::ostream& operator<<(std::ostream& os, own_operator) { return os << "custom"; }

TEST(AcoutEngine, SequencesSetsAndMapsHaveTheirBrackets)
{
    EXPECT_EQ(printed(std::vector<int>{1, 2, 3}), "[1, 2, 3]");
    EXPECT_EQ(printed(std::vector<int>{}), "[]");
    EXPECT_EQ(printed(std::array<int, 2>{4, 5}), "[4, 5]");
    EXPECT_EQ(printed(std::set<int>{3, 1, 2}), "{1, 2, 3}");
    EXPECT_EQ(printed(std::map<std::string, int>{{"a", 1}, {"b", 2}}), R"({"a": 1, "b": 2})");
    EXPECT_EQ(printed(std::deque<double>{1.5}), "[1.5]");
    const int raw[3] = {7, 8, 9};
    EXPECT_EQ(printed(raw), "[7, 8, 9]");
    EXPECT_EQ(printed(std::span<const int>{raw}), "[7, 8, 9]");
}

TEST(AcoutEngine, StringsAreTextAtTheTopAndQuotedInside)
{
    EXPECT_EQ(printed(std::string{"plain"}), "plain");
    EXPECT_EQ(printed("literal"), "literal");
    EXPECT_EQ(printed(std::string_view{"view"}), "view");
    EXPECT_EQ(printed('c'), "c");
    EXPECT_EQ(printed(std::vector<std::string>{"a", "b"}), R"(["a", "b"])");
    EXPECT_EQ(printed(std::vector<char>{'x', 'y'}), "['x', 'y']");
    EXPECT_EQ(printed(std::vector<const char*>{"p"}), R"(["p"])");
}

TEST(AcoutEngine, NestingRecurses)
{
    EXPECT_EQ(printed(std::vector<std::vector<int>>{{1}, {2, 3}}), "[[1], [2, 3]]");
    EXPECT_EQ(printed(std::map<int, std::vector<std::string>>{{1, {"x"}}}), R"({1: ["x"]})");
    EXPECT_EQ(printed(std::vector<std::optional<int>>{1, std::nullopt}), "[1, none]");
}

TEST(AcoutEngine, PairsAndTuples)
{
    EXPECT_EQ(printed(std::pair{1, std::string{"one"}}), R"((1, "one"))");
    EXPECT_EQ(printed(std::tuple{1, 2.5, 'c'}), "(1, 2.5, 'c')");
    EXPECT_EQ(printed(std::tuple<>{}), "()");
    EXPECT_EQ(printed(std::vector<std::pair<int, int>>{{1, 2}}), "[(1, 2)]");
}

TEST(AcoutEngine, OptionalVariantAndTheOddOnes)
{
    EXPECT_EQ(printed(std::optional<int>{7}), "7");
    EXPECT_EQ(printed(std::optional<int>{}), "none");
    EXPECT_EQ(printed(std::nullopt), "none");
    EXPECT_EQ(printed(std::optional<std::string>{"s"}), R"("s")");   // a value inside is nested
    EXPECT_EQ(printed(std::variant<int, std::string>{std::string{"v"}}), "v");
    EXPECT_EQ(printed(std::variant<int, std::string>{3}), "3");
    EXPECT_EQ(printed(std::variant<std::monostate, int>{}), "monostate");
    EXPECT_EQ(printed(nullptr), "nullptr");
    EXPECT_EQ(printed(std::byte{0x1f}), "0x1f");
    EXPECT_EQ(printed(side::sell), "sell");                 // a scoped enumeration prints by name
    EXPECT_EQ(printed(static_cast<side>(9)), "9");          // and as its integer when it is not an enumerator
    EXPECT_EQ(printed(std::vector{side::buy, side::sell}), "[buy, sell]");
    EXPECT_EQ(printed(small::seven), "seven");
    EXPECT_EQ(printed(static_cast<small>(200)), "200");     // a std::uint8_t is not a character
    EXPECT_EQ(printed(own_operator::a), "custom");          // its own operator<< wins
    EXPECT_EQ(printed(true), "1");         // bool is the stream's business
}

TEST(AcoutEngine, ResultsAreOkOrErr)
{
    using r = snicholls::result<int, std::string>;
    EXPECT_EQ(printed(r{snicholls::ok(5)}), "ok(5)");
    EXPECT_EQ(printed(r{snicholls::err(std::string{"bad"})}), R"(err("bad"))");
    using v = snicholls::result<void, int>;
    EXPECT_EQ(printed(v{snicholls::ok()}), "ok()");
    EXPECT_EQ(printed(v{snicholls::err(9)}), "err(9)");
}

TEST(AcoutEngine, TypeListsPrintTheirTypeNames)
{
    EXPECT_EQ(printed(snicholls::meta::type_list<int, double>{}), "type_list<int, double>");
    EXPECT_EQ(printed(snicholls::meta::type_list<>{}), "type_list<>");
}

TEST(AcoutEngine, PrintablesAndStreamablesInsideContainers)
{
    EXPECT_EQ(printed(std::vector<order>{{1, 2.0}, {3, 4.0}}), "[order 1 @ 2, order 3 @ 4]");
    EXPECT_EQ(printed(std::vector<trade>{{"A", 1, 1.0}}), "[A x1 @ 1.00]");   // the acout form, nested
    EXPECT_EQ(printed(std::optional<loud>{loud{}}), "print");
}

TEST(AcoutEngine, AcoutAndPrintlnUseTheEngine)
{
    std::ostringstream target;
    acout(target) << std::vector<int>{1, 2} << ' ' << std::pair{std::string{"k"}, 3} << ' ' << std::optional<int>{};
    EXPECT_EQ(target.str(), R"([1, 2] ("k", 3) none)");

    recording_buffer buffer;
    std::ostream one(&buffer);
    println(one, std::map<std::string, std::vector<int>>{{"a", {1, 2}}});
    ASSERT_EQ(buffer.writes.size(), 1u);
    EXPECT_EQ(buffer.writes[0], R"({"a": [1, 2]})" "\n");
}

TEST(AcoutEngine, PlainStreamsGetContainersByOptingIn)
{
    using snicholls::print_operators::operator<<;
    std::ostringstream os;
    os << std::vector<int>{1, 2} << " " << std::tuple{1, 'a'} << " " << std::optional<int>{4} << " " << std::string{"text"} << " "
       << snicholls::meta::type_list<int>{};
    EXPECT_EQ(os.str(), "[1, 2] (1, 'a') 4 text type_list<int>");
}

static_assert(snicholls::printable_any<std::vector<int>> && snicholls::printable_any<std::map<int, int>> &&
              snicholls::printable_any<std::tuple<int, int>> && snicholls::printable_any<std::optional<int>> &&
              snicholls::printable_any<int> && snicholls::printable_any<order> && snicholls::printable_any<trade>);
struct unprintable {};
static_assert(!snicholls::printable_any<unprintable>);

// ---- colour ---------------------------------------------------------------------------------------

using snicholls::colour_mode;
using snicholls::colour_reset;
using snicholls::styled;
using namespace snicholls::colours;

constexpr const char* start_red = "\x1b[31m";
constexpr const char* start_green = "\x1b[32m";
constexpr const char* reset = "\x1b[0m";

// The mode is program-wide: every test that sets it puts it back.
struct mode_guard {
    explicit mode_guard(colour_mode mode) : before(snicholls::current_colour_mode()) { snicholls::set_colour_mode(mode); }
    ~mode_guard() { snicholls::set_colour_mode(before); }
    mode_guard(const mode_guard&) = delete;
    mode_guard& operator=(const mode_guard&) = delete;
    colour_mode before;
};

// Whatever went to std::cout while the guard was alive
struct cout_capture {
    cout_capture() : before(std::cout.rdbuf(captured.rdbuf())) {}
    ~cout_capture() { std::cout.rdbuf(before); }
    cout_capture(const cout_capture&) = delete;
    cout_capture& operator=(const cout_capture&) = delete;
    std::ostringstream captured;
    std::streambuf* before;
};

// The rule for automatic, without touching the real environment
static_assert(snicholls::detail::environment_allows_colour(nullptr, nullptr, nullptr, false));
static_assert(snicholls::detail::environment_allows_colour(nullptr, "xterm-256color", nullptr, false));
static_assert(snicholls::detail::environment_allows_colour("", nullptr, nullptr, false));          // an empty NO_COLOR does not count
static_assert(!snicholls::detail::environment_allows_colour("1", "xterm", nullptr, false));
static_assert(!snicholls::detail::environment_allows_colour(nullptr, "dumb", nullptr, false));
static_assert(!snicholls::detail::environment_allows_colour(nullptr, nullptr, nullptr, true));     // a classic Windows console
static_assert(snicholls::detail::environment_allows_colour(nullptr, nullptr, "guid", true));       // Windows Terminal
static_assert(snicholls::detail::environment_allows_colour(nullptr, "xterm", nullptr, true));      // mintty and the like
static_assert(!snicholls::detail::environment_allows_colour("1", nullptr, "guid", true));

static_assert(snicholls::printable_any<snicholls::styled_value<int>>);
static_assert(snicholls::printable<snicholls::styled_value<std::string>>);
static_assert(noexcept(styled(red, 1)));
static_assert(noexcept(snicholls::set_colour_mode(colour_mode::never)) && noexcept(snicholls::current_colour_mode()));

TEST(AcoutColour, AStringStreamGetsPlainTextByDefault)
{
    const mode_guard mode(colour_mode::automatic);
    std::ostringstream os;
    acout(os) << red << "a" << colour_reset << styled(green, "b");
    snicholls::print_to(os, styled(green, "c"));
    EXPECT_EQ(os.str(), "abc");
}

TEST(AcoutColour, NeverWritesNoCodesEvenToTheTerminalStreams)
{
    const mode_guard mode(colour_mode::never);
    const cout_capture capture;
    acout{} << red << "a" << styled(green, "b");
    snicholls::println(red, "{}", "c");
    EXPECT_EQ(capture.captured.str(), "abc\n");
}

TEST(AcoutColour, AutomaticFollowsTheEnvironmentOnTheTerminalStreams)
{
    const mode_guard mode(colour_mode::automatic);
    const bool allowed = snicholls::detail::colour_allowed_by_environment();
    const int one = 1;
    {
        const cout_capture capture;
        acout{} << red << "a";
        EXPECT_EQ(capture.captured.str(), allowed ? std::string(start_red) + "a" + reset : "a");
    }
    {
        const cout_capture capture;   // nested in a container: the inner message asks the outer one
        acout{} << std::array{styled(green, one)};
        EXPECT_EQ(capture.captured.str(), allowed ? std::string("[") + start_green + "1" + reset + "]" : "[1]");
    }
}

TEST(AcoutColour, AColourHoldsUntilResetAndNeverOutlivesTheMessage)
{
    const mode_guard mode(colour_mode::always);
    std::ostringstream reset_by_hand;
    acout(reset_by_hand) << red << "a" << colour_reset << "b";
    EXPECT_EQ(reset_by_hand.str(), std::string(start_red) + "a" + reset + "b");

    std::ostringstream reset_by_the_end;
    acout(reset_by_the_end) << red << "a";
    EXPECT_EQ(reset_by_the_end.str(), std::string(start_red) + "a" + reset);

    std::ostringstream no_colour_no_codes;
    acout(no_colour_no_codes) << "a" << colour_reset;
    EXPECT_EQ(no_colour_no_codes.str(), "a");
}

TEST(AcoutColour, AColouredMessageIsStillOneWrite)
{
    const mode_guard mode(colour_mode::always);
    recording_buffer buffer;
    std::ostream target(&buffer);
    acout(target) << red << "a" << styled(green, "b") << "c";
    ASSERT_EQ(buffer.writes.size(), 1u);
    EXPECT_EQ(buffer.writes[0], std::string(start_red) + "a" + start_green + "b" + start_red + "c" + reset);
}

TEST(AcoutColour, AMessageIsReusableAfterEmittingAndStartsUncoloured)
{
    const mode_guard mode(colour_mode::always);
    std::ostringstream os;
    acout message(os);
    message << red << "a";
    message.emit();
    message << "b";
    message.emit();
    EXPECT_EQ(os.str(), std::string(start_red) + "a" + reset + "b");
}

TEST(AcoutColour, StyledReturnsToTheColourTheMessageWasIn)
{
    const mode_guard mode(colour_mode::always);
    std::ostringstream none;
    acout(none) << styled(green, "a") << "b";
    EXPECT_EQ(none.str(), std::string(start_green) + "a" + reset + "b");

    std::ostringstream nested;
    acout(nested) << styled(red, styled(green, "a"));
    EXPECT_EQ(nested.str(), std::string(start_red) + start_green + "a" + start_red + reset);
}

TEST(AcoutColour, ElementsOfAContainerKeepTheirColoursAndTheOuterOneReturns)
{
    const mode_guard mode(colour_mode::always);
    const int one = 1, two = 2;
    const std::array elements{styled(green, one), styled(green, two)};

    std::ostringstream alone;
    acout(alone) << elements;
    const std::string g = start_green;
    EXPECT_EQ(alone.str(), "[" + g + "1" + reset + ", " + g + "2" + reset + "]");

    std::ostringstream inside_red;
    acout(inside_red) << red << elements << "z";
    const std::string r = start_red;
    EXPECT_EQ(inside_red.str(), r + "[" + g + "1" + r + ", " + g + "2" + r + "]z" + reset);
}

TEST(AcoutColour, APrintableUsesColourThroughItsMessage)
{
    struct status {
        void print(acout& out) const { out << green << "ok" << colour_reset << " done"; }
    };
    const mode_guard mode(colour_mode::always);
    std::ostringstream direct;
    acout(direct) << status{};
    EXPECT_EQ(direct.str(), std::string(start_green) + "ok" + reset + " done");

    std::ostringstream in_a_list;
    acout(in_a_list) << std::vector<status>(1);
    EXPECT_EQ(in_a_list.str(), std::string("[") + start_green + "ok" + reset + " done]");
}

TEST(AcoutColour, AnEscapeCodeDoesNotUseUpAFieldWidth)
{
    const mode_guard mode(colour_mode::always);
    std::ostringstream os;
    acout(os) << std::setw(3) << red << 1;
    EXPECT_EQ(os.str(), std::string(start_red) + "  1" + reset);
}

TEST(AcoutColour, RgbIs24Bit)
{
    const mode_guard mode(colour_mode::always);
    std::ostringstream os;
    acout(os) << styled(snicholls::rgb(0x123456), "a");
    EXPECT_EQ(os.str(), std::string("\x1b[38;2;18;52;86ma") + reset);
}

TEST(AcoutColour, PrintAndPrintlnColourAWholeFormatString)
{
    const mode_guard mode(colour_mode::always);
    const cout_capture capture;
    snicholls::print(red, "{} {}", 1, "x");
    snicholls::println(green, "{}", 2);   // the newline after the colour ends
    EXPECT_EQ(capture.captured.str(), std::string(start_red) + "1 x" + reset + start_green + "2" + reset + "\n");
}

TEST(AcoutColour, CFilesFollowTheSameRule)
{
    const auto contents = [](std::FILE* file) {
        std::rewind(file);
        char read[64] = {};
        const auto n = std::fread(read, 1, sizeof read, file);
        std::fclose(file);
        return std::string(read, n);
    };
    {
        const mode_guard mode(colour_mode::automatic);
        std::FILE* file = std::tmpfile();
        ASSERT_NE(file, nullptr);
        acout(file) << red << "a";
        EXPECT_EQ(contents(file), "a");
    }
    {
        const mode_guard mode(colour_mode::always);
        std::FILE* file = std::tmpfile();
        ASSERT_NE(file, nullptr);
        acout(file) << red << "a";
        EXPECT_EQ(contents(file), std::string(start_red) + "a" + reset);
    }
}

static_assert(noexcept(std::declval<acout&>().flush()));
static_assert(!std::is_copy_constructible_v<acout> && !std::is_move_constructible_v<acout>);

} // namespace
