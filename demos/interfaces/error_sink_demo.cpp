//
//  error_sink_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  The error_sink interface: text_error_sink, and a concept-modelling sink put behind the interface.
//

#include <snicholls/core/interfaces/error_sink.hpp>

#include <cstdio>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {
int failures = 0;
void check(bool ok, const char* what)
{
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++failures;
}

enum class load_error { disk_full };

struct last_level_sink {
    snicholls::error_level last = snicholls::error_level::trace;
    void write(snicholls::error_level level, const snicholls::error_report&) { last = level; }
};
} // namespace

int main()
{
    using namespace snicholls;

    //! [interface]
    std::ostringstream log;
    std::vector<std::unique_ptr<error_sink>> sinks;   // chosen at run time
    sinks.push_back(std::make_unique<text_error_sink>(log, error_level::warning));
    sinks.push_back(std::make_unique<error_sink_model<last_level_sink>>());

    for (auto& sink : sinks)
        sink->log_error(error_level::error, traced<load_error>(load_error::disk_full));
    //! [interface]
    std::puts("error_sink");
    check(log.str().rfind("[ERROR] ", 0) == 0, "text_error_sink writes the level, then the report");
    check(static_cast<error_sink_model<last_level_sink>&>(*sinks[1]).get().last == error_level::error,
          "the model forwards to the adapted sink");
    std::ostringstream quiet;
    text_error_sink filtered(quiet, error_level::error);
    filtered.log_error(error_level::warning, traced<load_error>(load_error::disk_full));
    check(quiet.str().empty(), "a report below the threshold is dropped");
    std::printf("%s\n", failures == 0 ? "all checks passed" : "FAILED");
    return failures == 0 ? 0 : 1;
}
