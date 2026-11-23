//
//  error_sink_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  The error_sink_like concept: a sink chosen at compile time, with no virtual call.
//

#include <snicholls/core/concepts/error_sink.hpp>
#include <snicholls/core/traced.hpp>

#include <cstdio>
#include <string>

namespace {
int failures = 0;
void check(bool ok, const char* what)
{
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++failures;
}

enum class load_error { disk_full };
} // namespace

int main()
{
    using namespace snicholls;

    //! [concept]
    struct counting_sink {   // no base class: a sink for one program, chosen at compile time
        int errors = 0;
        void write(error_level, const error_report&) { ++errors; }
    };
    static_assert(error_sink_like<counting_sink>);

    counting_sink sink;
    sink.write(error_level::error, make_error_report(traced<load_error>(load_error::disk_full)));
    // result::log_err takes the interface's log_error: put the sink behind it with error_sink_model
    // (interfaces/error_sink.hpp)
    //! [concept]
    std::puts("error_sink_like");
    check(sink.errors == 1, "one report written");
    check(!error_sink_like<int>, "an int is not a sink");
    check(error_sink_like<text_error_sink>, "text_error_sink models the concept");
    std::printf("%s\n", failures == 0 ? "all checks passed" : "FAILED");
    return failures == 0 ? 0 : 1;
}
