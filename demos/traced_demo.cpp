//
//  traced_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/traced.hpp>.
//

// Demo: snicholls::traced<E> (snicholls/core/traced.hpp) - an error that carries where it happened, what the program was doing on
// the way up (a context chain, anyhow-style), fields with secrets masked and an optional stack trace, then is
// logged once, completely, through a snicholls::error_sink.
//
//   ./traced_demo        prints the report; exits 1 if any result is not what this file documents.

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>

#include <snicholls/core/traced.hpp>

namespace {

bool all_ok = true;

void check(bool cond, std::string_view what)
{
    std::cout << (cond ? "  ok   " : "  FAIL ") << what << '\n';
    all_ok = all_ok && cond;
}

// A stand-in for a credential type: anything with reveal() is masked in fields.
struct api_token
{
    std::string value;
    [[nodiscard]] const std::string& reveal() const noexcept { return value; }
};

//! [chain]
enum class io_error
{
    not_found = 2,
};

using io_result = snicholls::result<std::string, snicholls::traced<io_error>>;

io_result read_file(std::string_view path)
{
    if (path != "positions.csv")
        return snicholls::traced_err(io_error::not_found);   // records this file and line
    return snicholls::ok(std::string("ES,10\nNQ,-4\n"));
}

snicholls::result<std::size_t, snicholls::traced<io_error>> load_positions(std::string_view account, std::string_view path)
{
    // context() adds a frame at this line; SNICHOLLS_TRY adds a "propagated" frame with the expression text.
    SNICHOLLS_TRY_ASSIGN(const std::string text, read_file(path).context("loading positions for {}", account));
    return snicholls::ok(static_cast<std::size_t>(std::count(text.begin(), text.end(), '\n')));
}
//! [chain]

} // namespace

int main()
{
    std::cout << "snicholls::traced<E>: errors with a location, a context chain and fields\n";

    // Success: nothing is formatted, recorded or allocated.
    auto good = load_positions("ACC1", "positions.csv").context("start of day").with_field("desk", 7);
    check(good == snicholls::ok(std::size_t{2}), "success: 2 positions, no context recorded");

    //! [report]
    auto bad = load_positions("ACC1", "missing.csv")
                   .context("start of day for desk {}", 7)
                   .with_field("account", "ACC1")
                   .with_field("token", api_token{"s3cr3t"});
    const snicholls::traced<io_error>& e = bad.unwrap_err();
    std::cout << snicholls::to_string(e) << '\n';
    //! [report]
    check(e.error() == io_error::not_found, "the inner error is kept");
    check(e.message() == "start of day for desk 7: loading positions for ACC1: error 2", "one-line message");
    check(e.frames().size() == 3 && e.frames()[1].kind == snicholls::frame_kind::propagated, "three frames, one propagated");
    check(e.fields().size() == 2 && e.fields()[1].value == "********", "the token is masked");
    check(snicholls::to_string(e).find("s3cr3t") == std::string::npos, "the secret never reaches the text");

    // A flat field list, for structured logs.
    const auto fields = snicholls::error_report_fields(snicholls::make_error_report(e));
    check(fields.size() == 7 && fields[0].key == "error" && fields[2].key == "context.0", "flat field list");

    // Stack traces only on request, through a hook the composition root installs.
    snicholls::set_stack_trace_hook(+[](int) { return std::string("main\nload_positions"); });
    auto with_stack = read_file("nope").with_stack();
    snicholls::set_stack_trace_hook(nullptr);
    check(with_stack.unwrap_err().stack() == "main\nload_positions", "with_stack() captured through the hook");

    // anyhow's bail!: a formatted, located string error.
    snicholls::result<int, snicholls::traced<std::string>> invalid = snicholls::fail("bad quantity {} for {}", -3, "ES");
    check(invalid.unwrap_err().message() == "bad quantity -3 for ES", "snicholls::fail");

    //! [sink]
    // Log once, at the top, through any snicholls::error_sink (for example a super-log adapter).
    std::ostringstream log;
    snicholls::text_error_sink sink(log);
    auto logged = load_positions("ACC2", "gone.csv").log_err(sink, snicholls::error_level::warning);
    //! [sink]
    check(logged.is_err() && log.str().rfind("[WARN] error: loading positions for ACC2: error 2", 0) == 0,
          "log_err wrote one report and kept the result");
    std::cout << log.str();

    std::cout << (all_ok ? "traced_demo: all checks passed\n" : "traced_demo: FAILED\n");
    return all_ok ? 0 : 1;
}
