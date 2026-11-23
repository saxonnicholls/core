//
//  error_sink_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/interfaces/error_sink.hpp>. text_error_sink and error_sink_model also run the
//  conformance suite (tests/support/snicholls/core/conformance/error_sink.hpp).
//

#include <snicholls/core/conformance/error_sink.hpp>
#include <snicholls/core/interfaces/error_sink.hpp>

#include <gtest/gtest.h>

#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace {

using namespace snicholls;

static_assert(std::is_abstract_v<error_sink>);
static_assert(std::is_base_of_v<error_sink, text_error_sink>);

struct recording_sink {
    std::vector<std::string> lines;
    std::shared_ptr<std::mutex> mutex = std::make_shared<std::mutex>();
    void write(error_level, const error_report& report)
    {
        const std::lock_guard<std::mutex> lock(*mutex);
        lines.push_back(format_error_report(report));
    }
};

enum class load_error { disk_full };

TEST(ErrorSinkModel, ForwardsEveryReportToTheAdaptedSink)
{
    error_sink_model<recording_sink> model;
    error_sink& sink = model;
    sink.log_error(traced<load_error>(load_error::disk_full));
    sink.log_error(error_level::warning, traced<load_error>(load_error::disk_full));
    EXPECT_EQ(model.get().lines.size(), 2u);
}

TEST(ErrorSinkModel, AdaptsAGivenValue)
{
    recording_sink seed;
    seed.lines.push_back("before");
    error_sink_model<recording_sink> model(seed);
    model.log_error(traced<load_error>(load_error::disk_full));
    ASSERT_EQ(model.get().lines.size(), 2u);
    EXPECT_EQ(model.get().lines[0], "before");
}

TEST(TextErrorSink, DropsReportsBelowItsThreshold)
{
    std::ostringstream out;
    text_error_sink sink(out, error_level::error);
    sink.log_error(error_level::warning, traced<load_error>(load_error::disk_full));
    EXPECT_TRUE(out.str().empty());
    sink.log_error(error_level::error, traced<load_error>(load_error::disk_full));
    EXPECT_EQ(out.str().rfind("[ERROR] ", 0), 0u);
}

struct model_probe {
    error_sink_model<recording_sink> m;
    error_sink& sink() { return m; }
    std::vector<std::string> written() const { return m.get().lines; }
};

} // namespace

SNICHOLLS_ERROR_SINK_CONFORMANCE(ErrorSinkModelContract, model_probe);
