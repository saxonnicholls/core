//
//  error_sink.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::error_sink`, the virtual twin of `snicholls::error_sink_like`; `text_error_sink`; the adapter
///        `error_sink_model`.
/// \ingroup core
///
/// **Overview.** The run-time form of a log sink: `result::log_err(sink)` and `error_sink::log_error` build an
/// `error_report` and hand it to `write` once. `text_error_sink` writes text to a stream; `error_sink_model<S>` puts
/// any `error_sink_like` type behind the interface. Included by `traced.hpp`, so existing code that includes only
/// `traced.hpp` sees the same names.
///
/// \snippet interfaces/error_sink_demo.cpp interface
///
/// **Relationships.** `concepts/error_sink.hpp` (the concept), `traced.hpp`, `result.hpp`. Group: \ref core.
///
/// **Implementations.** `text_error_sink` here; `error_sink_model<S>` over any concept-modelling type; a super-log
/// adapter in the consuming library.
///
/// **Guarantees.** See each class. `log_error` never throws.
///
/// **Errors.** None escape `log_error`; `write` may throw.
///
/// **Performance.** Not benchmarked (error path only).
///
/// **Tests and demos.** `tests/interfaces/error_sink_test.cpp`, `demos/interfaces/error_sink_demo.cpp`; the
/// conformance suite `tests/support/snicholls/core/conformance/error_sink.hpp`.
///
/// **Dependencies.** `traced.hpp`, `<mutex>`, `<ostream>`, `<string>`.
///
/// **Known issues.** None.
///
/// **Status.** Experimental, pre-1.0. Moved out of `traced.hpp` in 0.23.0.

#ifndef snicholls_core_interfaces_error_sink_hpp
#define snicholls_core_interfaces_error_sink_hpp

#include <snicholls/core/concepts/error_sink.hpp>
#include <snicholls/core/traced.hpp>

#include <mutex>
#include <ostream>
#include <string>
#include <utility>

namespace snicholls {

/// \brief Where errors are logged: receives a structured `snicholls::error_report` at a level.
/// \ingroup core
///
/// \overview
/// The log hook of `snicholls::result`: `r.log_err(sink)` / `r.log_err(sink, level)` call `log_error`, which builds the
/// report and calls `write` once. Implement `write` to send it anywhere; the core stays free of logging libraries.
///
/// \relations
/// - `snicholls::error_sink_like` (`concepts/error_sink.hpp`), its concept twin; `snicholls::error_sink_model`, the adapter.
/// - `snicholls::result::log_err`, `snicholls::error_report`, `snicholls::traced`.
/// - Module: \ref core.
///
/// \impls
/// | Implementation | Target | Writes |
/// |---|---|---|
/// | `snicholls::text_error_sink` | `snicholls::core` | `format_error_report` text to a `std::ostream` |
/// | a super-log adapter | the consuming library | one super-log event (NDJSON) per error |
///
/// \contracts
/// - `log_error` never throws: an exception from building the report or from `write` is swallowed (logging is
///   best effort; the result is returned unchanged either way). `write` may throw.
/// - Thread safety is the implementation's: both shipped sinks may be called from any thread.
///
/// \errors
/// None escape `log_error`.
///
/// \examples
/// \snippet traced_demo.cpp sink
///
/// \perf
/// Not benchmarked (error path only).
///
/// \testing
/// - Tests: tests/interfaces/error_sink_test.cpp, tests/traced_test.cpp (GoogleMock sink, `text_error_sink`),
///   tests/support/snicholls/core/conformance/error_sink.hpp.
/// - Demo: demos/traced_demo.cpp.
///
/// \deps
/// CMake target `snicholls::core`; standard library only.
///
/// \gotchas
/// None known.
///
/// \status
/// Experimental, pre-1.0. Design: [RESULT_PLAN.md](RESULT_PLAN.md).
class error_sink
{
public:
    /// \brief Destroys it.
    virtual ~error_sink() = default;

    /// \brief Receives one error.
    /// \param level the level.
    /// \param report the report.
    virtual void write(error_level level, const error_report& report) = 0;

    /// \brief Builds the report of `e` and writes it at `level`. Never throws.
    /// \param level the level.
    /// \param e the error.
    /// \param where the log site.
    template <class E>
    void log_error(error_level level, const E& e, source_location where = source_location::current()) noexcept
    {
#if SNICHOLLS_RESULT_HAS_EXCEPTIONS
        try
        {
            write(level, make_error_report(e, where));
        }
        catch (...)
        {
        }
#else
        write(level, make_error_report(e, where));
#endif
    }

    /// \brief As `log_error(error_level::error, e, where)`.
    /// \param e the error.
    /// \param where the log site.
    template <class E>
    void log_error(const E& e, source_location where = source_location::current()) noexcept
    {
        log_error(error_level::error, e, where);
    }

protected:
    /// \brief Constructs it.
    error_sink() = default;
    /// \brief Copies it.
    error_sink(const error_sink&) = default;
    /// \brief Copy-assigns.
    /// \return `*this`.
    error_sink& operator=(const error_sink&) = default;
};

/// \brief A `snicholls::error_sink` writing `format_error_report` text, one error per block, to a `std::ostream`.
/// \ingroup core
///
/// Each report is written as `[LEVEL] <text>\n` in one `<<` sequence under a mutex, so concurrent writers never
/// interleave. Reports below `threshold` are dropped.
class text_error_sink final : public error_sink
{
public:
    /// \brief Constructs it.
    /// \param out the stream (outlives the sink).
    /// \param threshold the lowest level written.
    explicit text_error_sink(std::ostream& out, error_level threshold = error_level::trace) noexcept
        : out_(&out), threshold_(threshold)
    {
    }

    /// \brief Writes the report.
    /// \param level the level.
    /// \param report the report.
    /// \throws std::bad_alloc, or what the stream throws.
    void write(error_level level, const error_report& report) override
    {
        if (level < threshold_)
            return;
        std::string text = "[" + std::string(to_string(level)) + "] " + format_error_report(report) + "\n";
        const std::lock_guard<std::mutex> lock(mutex_);
        *out_ << text;
        out_->flush();
    }

private:
    std::ostream* out_;
    error_level threshold_;
    std::mutex mutex_;
};

/// \brief Any `error_sink_like` type behind `snicholls::error_sink`, for a sink chosen at run time.
/// \ingroup core
/// \tparam S the sink, held by value.
template <class S>
    requires error_sink_like<S>
class error_sink_model final : public error_sink
{
public:
    /// \brief Adapts a default-constructed `S`.
    error_sink_model() = default;
    /// \brief Adapts this sink.
    /// \param s the sink.
    explicit error_sink_model(S s) : s_(std::move(s)) {}

    /// \brief Forwards to the adapted sink's `write`.
    /// \param level the level.
    /// \param report the report.
    void write(error_level level, const error_report& report) override { s_.write(level, report); }

    /// \brief The adapted sink.
    /// \return A reference to it.
    [[nodiscard]] S& get() noexcept { return s_; }
    /// \brief The adapted sink.
    /// \return A const reference to it.
    [[nodiscard]] const S& get() const noexcept { return s_; }

private:
    S s_{};
};

static_assert(error_sink_like<error_sink>, "snicholls::error_sink must model snicholls::error_sink_like");
static_assert(error_sink_like<text_error_sink>, "snicholls::text_error_sink must model snicholls::error_sink_like");

} // namespace snicholls

#endif // snicholls_core_interfaces_error_sink_hpp
