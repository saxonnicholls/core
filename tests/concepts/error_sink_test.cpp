//
//  error_sink_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/concepts/error_sink.hpp>.
//

#include <snicholls/core/concepts/error_sink.hpp>
#include <snicholls/core/traced.hpp>

#include <gtest/gtest.h>

namespace {

using namespace snicholls;

struct plain_sink {
    int count = 0;
    void write(error_level, const error_report&) { ++count; }
};
struct wrong_arguments {
    void write(int) {}
};
struct const_only {
    void write(error_level, error_report&) {}   // takes the report by non-const reference
};

static_assert(error_sink_like<plain_sink>);
static_assert(error_sink_like<error_sink> && error_sink_like<text_error_sink>);
static_assert(!error_sink_like<int> && !error_sink_like<wrong_arguments> && !error_sink_like<const_only>);

enum class load_error { disk_full };

TEST(ErrorSinkConcept, AModellingTypeReceivesReportsWithoutAVirtualCall)
{
    plain_sink sink;
    sink.write(error_level::error, make_error_report(traced<load_error>(load_error::disk_full)));
    EXPECT_EQ(sink.count, 1);
}

TEST(ErrorSinkConcept, ResultLogsThroughAModelledSinkOnlyOnFailure)
{
    error_sink_model<plain_sink> sink;
    result<int, traced<load_error>> failed = traced_err(load_error::disk_full);
    failed.log_err(sink, error_level::error);
    result<int, traced<load_error>> fine = ok(7);
    fine.log_err(sink, error_level::error);
    EXPECT_EQ(sink.get().count, 1);
}

} // namespace
