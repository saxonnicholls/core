//
//  traced_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/traced.hpp>.
//

// snicholls::traced<E> (include/snicholls/core/traced.hpp): location, context chain, fields and masking, the stack hook, SNICHOLLS_TRY
// frames, the formatter and the flat field list, error_sink (GoogleMock) and text_error_sink (contract suite).

#include <snicholls/core/traced.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <format>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>


#include "support/error_sink_contract.hpp"

namespace {

// A secret, as a credential type is one: it hides its value behind reveal() and has no std::formatter.
template <class T>
class test_secret {
public:
    explicit test_secret(T v) : value_(std::move(v)) {}
    const T& reveal() const noexcept { return value_; }

private:
    T value_;
};

enum class io_error
{
    not_found = 2,
    denied = 13,
};

using io_result = snicholls::result<std::string, snicholls::traced<io_error>>;

// Line numbers below are asserted: keep these functions where they are, or update the expectations.
io_result read_file(std::string_view path)
{
    if (path == "missing")
        return snicholls::traced_err(io_error::not_found);   // origin
    return snicholls::ok(std::string("contents of ") + std::string(path));
}
constexpr std::uint_least32_t read_file_origin_line = __LINE__ - 3;

snicholls::result<std::size_t, snicholls::traced<io_error>> load_book(std::string_view account)
{
    SNICHOLLS_TRY_ASSIGN(const std::string text, read_file(account).context("loading positions for {}", account));
    return snicholls::ok(text.size());
}
constexpr std::uint_least32_t load_book_line = __LINE__ - 3;

// A plain error entering a traced function through SNICHOLLS_TRY.
snicholls::result<int, io_error> plain_open(bool fail)
{
    if (fail)
        return snicholls::err(io_error::denied);
    return snicholls::ok(1);
}

snicholls::result<int, snicholls::traced<io_error>> traced_open(bool fail)
{
    SNICHOLLS_TRY_ASSIGN(const int fd, plain_open(fail));
    return snicholls::ok(fd);
}
constexpr std::uint_least32_t traced_open_line = __LINE__ - 3;

std::string fake_stack(int skip)
{
    return "frame skip=" + std::to_string(skip) + "\nmain";
}

struct stack_hook_guard
{
    stack_hook_guard() { snicholls::set_stack_trace_hook(&fake_stack); }
    ~stack_hook_guard() { snicholls::set_stack_trace_hook(nullptr); }
};

// ---- layout and cost -------------------------------------------------------------------------------------------

// The location is one pointer in libc++ and libstdc++ and three words in MSVC's STL (line, column, file, function).
static_assert(sizeof(snicholls::traced<io_error>)
              <= sizeof(io_error) + sizeof(snicholls::source_location) + sizeof(void*) + alignof(void*));
static_assert(std::is_nothrow_move_constructible_v<snicholls::traced<io_error>>);
static_assert(std::is_copy_constructible_v<snicholls::traced<io_error>>);
static_assert(std::is_same_v<snicholls::traced_t<io_error>, snicholls::traced<io_error>>);
static_assert(std::is_same_v<snicholls::traced_t<snicholls::traced<io_error>>, snicholls::traced<io_error>>);
static_assert(snicholls::is_traced_v<snicholls::traced<int>> && !snicholls::is_traced_v<int>);
// A plain error does not convert implicitly into a traced one (it would be located in result.hpp).
static_assert(!std::is_constructible_v<snicholls::result<int, snicholls::traced<int>>, const snicholls::result<int, int>&>);
static_assert(!std::is_convertible_v<int, snicholls::traced<int>>);

// ---- creation and context ----------------------------------------------------------------------------------------

TEST(TracedCreation, RecordsWhereTheErrorWasCreated)
{
    const auto r = read_file("missing");
    ASSERT_TRUE(r.is_err());
    const auto& e = r.unwrap_err();
    EXPECT_EQ(e.error(), io_error::not_found);
    EXPECT_EQ(e.where().line(), read_file_origin_line);
    EXPECT_EQ(snicholls::source_site(e.where()), "traced_test.cpp:" + std::to_string(read_file_origin_line));
    EXPECT_TRUE(e.frames().empty());
    EXPECT_TRUE(e.fields().empty());
    EXPECT_TRUE(e.stack().empty());
    EXPECT_EQ(e.message(), "error 2");
}

TEST(TracedCreation, ExplicitConstructionAndFail)
{
    snicholls::traced t{io_error::denied};
    static_assert(std::is_same_v<decltype(t), snicholls::traced<io_error>>);
    EXPECT_EQ(t.where().line(), static_cast<std::uint_least32_t>(__LINE__ - 2));
    snicholls::result<int, snicholls::traced<std::string>> f = snicholls::fail("bad price {} for {}", 1.5, "ES");
    EXPECT_EQ(f.unwrap_err().error(), "bad price 1.5 for ES");
    EXPECT_EQ(f.unwrap_err().where().line(), static_cast<std::uint_least32_t>(__LINE__ - 2));
}

TEST(TracedContext, ChainIsRecordedInnermostFirstAndReadOutermostFirst)
{
    auto r = load_book("missing").context("start of day for {}", "desk-7");
    ASSERT_TRUE(r.is_err());
    const auto& e = r.unwrap_err();
    ASSERT_EQ(e.frames().size(), 3u);
    EXPECT_EQ(e.frames()[0].message, "loading positions for missing");
    EXPECT_EQ(e.frames()[0].kind, snicholls::frame_kind::context);
    EXPECT_EQ(e.frames()[0].where.line(), load_book_line);
    EXPECT_EQ(e.frames()[1].kind, snicholls::frame_kind::propagated);
    EXPECT_EQ(e.frames()[1].message, "read_file(account).context(\"loading positions for {}\", account)");
    EXPECT_EQ(e.frames()[1].where.line(), load_book_line);
    EXPECT_EQ(e.frames()[2].message, "start of day for desk-7");
    EXPECT_EQ(e.message(), "start of day for desk-7: loading positions for missing: error 2");
    EXPECT_EQ(snicholls::error_text(e), e.message());
}

TEST(TracedContext, SuccessPathRecordsNothingAndDoesNotFormat)
{
    auto r = load_book("present").context("never {}", "formatted").with_field("k", "v").with_stack();
    EXPECT_TRUE(r.is_ok());
    EXPECT_EQ(*r, std::string("contents of present").size());
}

TEST(TracedContext, PlainErrorEnteringATracedFunctionIsWrappedAtTheTry)
{
    auto r = traced_open(true);
    ASSERT_TRUE(r.is_err());
    const auto& e = r.unwrap_err();
    EXPECT_EQ(e.error(), io_error::denied);
    EXPECT_EQ(e.where().line(), traced_open_line);
    ASSERT_EQ(e.frames().size(), 1u);
    EXPECT_EQ(e.frames()[0].kind, snicholls::frame_kind::propagated);
    EXPECT_EQ(e.frames()[0].message, "plain_open(fail)");
    EXPECT_EQ(traced_open(false), snicholls::ok(1));
}

TEST(TracedContext, ContextOnAPlainErrorStartsTracingAtTheCaller)
{
    snicholls::result<int, std::string> plain = snicholls::err(std::string("eof"));
    auto r = std::move(plain).context("reading header");
    static_assert(std::is_same_v<decltype(r), snicholls::result<int, snicholls::traced<std::string>>>);
    EXPECT_EQ(r.unwrap_err().where().line(), static_cast<std::uint_least32_t>(__LINE__ - 2));
    EXPECT_EQ(r.unwrap_err().message(), "reading header: eof");
    auto t = snicholls::result<int, int>(snicholls::err(4)).trace();
    EXPECT_EQ(t.unwrap_err().where().line(), static_cast<std::uint_least32_t>(__LINE__ - 1));
    const snicholls::result<int, int> lvalue = snicholls::err(5);
    EXPECT_EQ(lvalue.context("copied").unwrap_err().message(), "copied: 5");
    EXPECT_TRUE(lvalue.is_err());   // the lvalue is untouched
}

TEST(TracedContext, TracedErrorsDropToPlainOnesThroughTry)
{
    auto f = []() -> snicholls::result<std::size_t, io_error> {
        SNICHOLLS_TRY_ASSIGN(const std::size_t n, load_book("missing"));
        return snicholls::ok(n);
    };
    EXPECT_EQ(f(), snicholls::err(io_error::not_found));
}

// ---- fields, masking, stack --------------------------------------------------------------------------------------

TEST(TracedFields, ValuesAreTextAndSecretsAreMasked)
{
    auto r = read_file("missing")
                 .with_field("account", "ACC1")
                 .with_field("attempt", 2)
                 .with_field("px", 101.5)
                 .with_field("retry", true)
                 .with_field("password", test_secret<std::string>("hunter2"));
    const auto& e = r.unwrap_err();
    ASSERT_EQ(e.fields().size(), 5u);
    EXPECT_EQ(e.fields()[0], (snicholls::error_field{"account", "ACC1"}));
    EXPECT_EQ(e.fields()[1], (snicholls::error_field{"attempt", "2"}));
    EXPECT_EQ(e.fields()[2], (snicholls::error_field{"px", "101.5"}));
    EXPECT_EQ(e.fields()[3], (snicholls::error_field{"retry", "true"}));
    EXPECT_EQ(e.fields()[4], (snicholls::error_field{"password", "********"}));
    EXPECT_EQ(snicholls::error_field_text(test_secret<int>(4)), snicholls::masked_field_text);
}

TEST(TracedFields, StackIsCapturedOnlyOnRequestThroughTheHook)
{
    EXPECT_TRUE(read_file("missing").with_stack().unwrap_err().stack().empty());   // no hook: nothing
    stack_hook_guard guard;
    EXPECT_TRUE(read_file("missing").unwrap_err().stack().empty());   // not requested
    auto r = read_file("missing").with_stack();
    EXPECT_EQ(r.unwrap_err().stack(), "frame skip=2\nmain");
    EXPECT_TRUE(read_file("present").with_stack().is_ok());
}

TEST(TracedFields, CopiesAreDeepAndEqualityIgnoresTheDetails)
{
    auto r = read_file("missing").context("a").with_field("k", "v");
    snicholls::traced<io_error> copy = r.unwrap_err();
    copy.add_context("only in the copy");
    EXPECT_EQ(r.unwrap_err().frames().size(), 1u);
    EXPECT_EQ(copy.frames().size(), 2u);
    EXPECT_EQ(copy, r.unwrap_err());   // same inner error
    snicholls::traced<io_error> assigned{io_error::denied};
    assigned = copy;
    EXPECT_EQ(assigned.frames().size(), 2u);
    std::unordered_set<snicholls::traced<io_error>> set{copy, assigned, snicholls::traced{io_error::denied}};
    EXPECT_EQ(set.size(), 2u);
    snicholls::traced<long> widened(snicholls::traced<int>(3));
    EXPECT_EQ(widened.error(), 3L);
}

TEST(TracedFields, RethrowThrowsTheInnerError)
{
    snicholls::result<int, snicholls::traced<std::exception_ptr>> r =
        snicholls::traced_err(std::make_exception_ptr(std::out_of_range("deep")));
    EXPECT_THROW(r.rethrow(), std::out_of_range);
    EXPECT_THROW((void)read_file("missing").unwrap(), snicholls::bad_result_access<snicholls::traced<io_error>>);
    EXPECT_THROW(read_file("missing").rethrow(), snicholls::bad_result_access<io_error>);
}

// ---- formatter -------------------------------------------------------------------------------------------------------

TEST(TracedFormat, TextReport)
{
    stack_hook_guard guard;
    auto r = load_book("missing").context("start of day").with_field("desk", "7").with_stack(); const auto here = __LINE__;
    const std::string L = std::to_string(load_book_line);
    const std::string expected = "error: start of day: loading positions for missing: error 2\n"
                                 "  at traced_test.cpp:" + std::to_string(read_file_origin_line) + "\n"
                                 "  context (innermost first):\n"
                                 "    0: loading positions for missing at traced_test.cpp:" + L + "\n"
                                 "    1: propagated `read_file(account).context(\"loading positions for {}\", account)` "
                                 "at traced_test.cpp:" + L + "\n"
                                 "    2: start of day at traced_test.cpp:" + std::to_string(here) + "\n"
                                 "  fields:\n"
                                 "    desk = 7\n"
                                 "  stack:\n"
                                 "    frame skip=2\n"
                                 "    main";
    EXPECT_EQ(snicholls::to_string(r.unwrap_err()), expected);
}

TEST(TracedFormat, PlainErrorsReportTheLogSite)
{
    const auto report = snicholls::make_error_report(std::string("timeout"));
    EXPECT_FALSE(report.located);
    EXPECT_EQ(report.message, "timeout");
    EXPECT_EQ(snicholls::format_error_report(report),
              "error: timeout\n  logged at traced_test.cpp:" + std::to_string(__LINE__ - 4));
}

TEST(TracedFormat, FlatFieldList)
{
    auto r = load_book("missing").with_field("account", "A").with_field("error", "clash");
    const auto fields = snicholls::error_report_fields(snicholls::make_error_report(r.unwrap_err()));
    const std::string L = std::to_string(load_book_line);
    const std::vector<snicholls::error_field> expected{
        {"error", "error 2"},
        {"src", "traced_test.cpp:" + std::to_string(read_file_origin_line)},
        {"context.0", "loading positions for missing @ traced_test.cpp:" + L},
        {"context.1", "propagated read_file(account).context(\"loading positions for {}\", account) @ traced_test.cpp:" + L},
        {"account", "A"},
        {"field.error", "clash"},
    };
    EXPECT_EQ(fields, expected);
    EXPECT_TRUE(snicholls::is_reserved_error_field("context.3"));
    EXPECT_FALSE(snicholls::is_reserved_error_field("venue"));
}

TEST(TracedFormat, LevelNamesAreSuperLogs)
{
    EXPECT_EQ(snicholls::to_string(snicholls::error_level::trace), "TRACE");
    EXPECT_EQ(snicholls::to_string(snicholls::error_level::debug), "DEBUG");
    EXPECT_EQ(snicholls::to_string(snicholls::error_level::info), "INFO");
    EXPECT_EQ(snicholls::to_string(snicholls::error_level::warning), "WARN");
    EXPECT_EQ(snicholls::to_string(snicholls::error_level::error), "ERROR");
    EXPECT_EQ(snicholls::to_string(snicholls::error_level::critical), "CRITICAL");
    EXPECT_EQ(snicholls::to_string(static_cast<snicholls::error_level>(99)), "ERROR");
}

// ---- error_sink with GoogleMock --------------------------------------------------------------------------------------

class mock_error_sink : public snicholls::error_sink
{
public:
    MOCK_METHOD(void, write, (snicholls::error_level level, const snicholls::error_report& report), (override));
};

TEST(TracedSink, LogErrWritesTheReportOnceAtTheLevel)
{
    mock_error_sink sink;
    snicholls::error_report seen;
    EXPECT_CALL(sink, write(snicholls::error_level::critical, testing::_)).WillOnce(testing::SaveArg<1>(&seen));
    auto r = load_book("missing").with_field("k", "v").log_err(sink, snicholls::error_level::critical);
    EXPECT_TRUE(r.is_err());
    EXPECT_TRUE(seen.located);
    EXPECT_EQ(seen.message, "loading positions for missing: error 2");
    EXPECT_EQ(seen.error, "error 2");
    EXPECT_EQ(seen.frames.size(), 2u);
    EXPECT_EQ(seen.fields, (std::vector<snicholls::error_field>{{"k", "v"}}));
}

TEST(TracedSink, DefaultLevelIsErrorAndPlainErrorsGetTheLogSite)
{
    mock_error_sink sink;
    snicholls::error_report seen;
    EXPECT_CALL(sink, write(snicholls::error_level::error, testing::_)).WillOnce(testing::SaveArg<1>(&seen));
    snicholls::result<void, std::string> r = snicholls::err(std::string("closed"));
    r.log_err(sink);
    EXPECT_FALSE(seen.located);
    EXPECT_EQ(seen.where.line(), static_cast<std::uint_least32_t>(__LINE__ - 2));
}

TEST(TracedSink, SuccessesAreNotLogged)
{
    mock_error_sink sink;
    EXPECT_CALL(sink, write(testing::_, testing::_)).Times(0);
    (void)load_book("present").log_err(sink, snicholls::error_level::error);
    const io_result ok = snicholls::ok(std::string("x"));
    ok.log_err(sink);
}

TEST(TracedSink, AThrowingSinkNeverBreaksTheCaller)
{
    mock_error_sink sink;
    EXPECT_CALL(sink, write(testing::_, testing::_)).WillOnce(testing::Throw(std::runtime_error("sink down")));
    auto r = load_book("missing").log_err(sink);
    EXPECT_TRUE(r.is_err());
}

TEST(TracedSink, TextSinkFormatAndThreshold)
{
    std::ostringstream out;
    snicholls::text_error_sink sink(out, snicholls::error_level::warning);
    snicholls::result<int, std::string> r = snicholls::err(std::string("x"));
    (void)r.log_err(sink, snicholls::error_level::info);   // below the threshold
    EXPECT_TRUE(out.str().empty());
    (void)r.log_err(sink, snicholls::error_level::warning);
    EXPECT_EQ(out.str(), "[WARN] error: x\n  logged at traced_test.cpp:" + std::to_string(__LINE__ - 1) + "\n");
}

// ---- review regressions (fix/pre-0.1.0-review) -----------------------------------------------------------------------

TEST(TracedReview, NullCStringFieldIsWrittenAsNull)
{
    const char* nothing = nullptr;   // std::string_view(nullptr) was undefined behaviour (UBSan: null pointer)
    EXPECT_EQ(snicholls::error_field_text(nothing), "(null)");
    auto r = read_file("missing").with_field("path", nothing);
    ASSERT_EQ(r.unwrap_err().fields().size(), 1u);
    EXPECT_EQ(r.unwrap_err().fields()[0], (snicholls::error_field{"path", "(null)"}));
    char* mutable_nothing = nullptr;
    EXPECT_EQ(snicholls::error_field_text(mutable_nothing), "(null)");
}

TEST(TracedReview, ARunTimeInvalidFormatArgumentThrowsFormatError)
{
    // The format string is checked at compile time; a negative dynamic width is only known at run time.
    int width = -1;
    EXPECT_THROW((void)snicholls::fail("width {:{}}", "x", width), std::format_error);
    snicholls::result<int, int> r = snicholls::err(5);
    EXPECT_THROW((void)std::move(r).context("width {:{}}", 1, width), std::format_error);
    snicholls::result<int, int> fine = snicholls::ok(1);
    EXPECT_NO_THROW((void)std::move(fine).context("width {:{}}", 1, width));   // a success formats nothing
}

TEST(TracedReview, NewlinesInTextCannotForgeLines)
{
    std::ostringstream out;
    snicholls::text_error_sink sink(out);
    auto r = read_file("missing").context("bad\nline").with_field("note", "x\n[ERROR] forged\r\\");
    (void)r.log_err(sink);
    const std::string text = out.str();
    EXPECT_EQ(text.find("\n[ERROR] forged"), std::string::npos) << text;
    EXPECT_NE(text.find("note = x\\n[ERROR] forged\\r\\\\"), std::string::npos) << text;
    EXPECT_NE(text.find("bad\\nline"), std::string::npos) << text;
    // exactly the lines the report has: error, at, context header, one frame, fields header, one field
    EXPECT_EQ(std::count(text.begin(), text.end(), '\n'), 6) << text;
}

// ---- contract suite registration -------------------------------------------------------------------------------

struct text_sink_probe
{
    std::ostringstream out;
    snicholls::text_error_sink text{out};
    snicholls::error_sink& sink() { return text; }
    std::vector<std::string> written() const
    {
        // One report per "[LEVEL] error: " block.
        std::vector<std::string> blocks;
        const std::string all = out.str();
        std::size_t at = 0;
        while ((at = all.find("[", at)) != std::string::npos)
        {
            const std::size_t next = all.find("\n[", at + 1);
            blocks.push_back(all.substr(at, next == std::string::npos ? std::string::npos : next - at));
            if (next == std::string::npos)
                break;
            at = next + 1;
        }
        return blocks;
    }
};

} // namespace

namespace snicholls_test {
using text_probes = testing::Types<text_sink_probe>;
INSTANTIATE_TYPED_TEST_SUITE_P(TextErrorSink, ErrorSinkContract, text_probes);
} // namespace snicholls_test
