//
//  traced.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::traced<E>`: an error with its source location, a context chain, fields and an optional stack
///        trace; `snicholls::error_sink` and the formatter to log it.
/// \ingroup core
///
/// \code
///   snicholls::result<book, snicholls::traced<io_error>> load_book(std::string_view desk)
///   {
///       auto positions = SNICHOLLS_TRY(load_positions(account).context("loading positions for {}", account));
///       ...
///   }
///   auto r = load_book("desk-7").with_field("desk", "desk-7").log_err(sink, snicholls::error_level::error);
/// \endcode
///
/// Opt-in: a plain `snicholls::result<V, E>` stays zero cost. Standard library only; a super-log adapter belongs in
/// the consuming library. Design: [RESULT_PLAN.md](RESULT_PLAN.md).

#ifndef snicholls_core_traced_hpp
#define snicholls_core_traced_hpp

#include <atomic>
#include <cstdint>
#include <format>
#include <functional>
#include <memory>
#include <mutex>
#include <ostream>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <snicholls/core/meta/concepts.hpp>
#include <snicholls/core/result.hpp>

namespace snicholls {

/// \brief The severity of a logged error; the names are super-log's (`to_string`).
/// \ingroup core
enum class error_level : std::uint8_t
{
    trace,      ///< `TRACE`
    debug,      ///< `DEBUG`
    info,       ///< `INFO`
    warning,    ///< `WARN`
    error,      ///< `ERROR` (the default of `log_err`)
    critical,   ///< `CRITICAL`
};

/// \brief The level's name as super-log spells it: TRACE, DEBUG, INFO, WARN, ERROR, CRITICAL.
/// \ingroup core
/// \param level the level.
/// \return the name (static storage).
[[nodiscard]] constexpr std::string_view to_string(error_level level) noexcept
{
    switch (level)
    {
    case error_level::trace: return "TRACE";
    case error_level::debug: return "DEBUG";
    case error_level::info: return "INFO";
    case error_level::warning: return "WARN";
    case error_level::error: return "ERROR";
    case error_level::critical: return "CRITICAL";
    }
    return "ERROR";
}

/// \brief How a context frame was added.
/// \ingroup core
enum class frame_kind : std::uint8_t
{
    context,      ///< `context("...")`: a sentence about what was being done.
    propagated,   ///< `SNICHOLLS_TRY(expr)`: the expression text, recorded automatically.
};

/// \brief One frame of a traced error's context chain.
/// \ingroup core
struct context_frame
{
    /// \brief The message (`context`) or the expression text (`propagated`).
    std::string message;
    /// \brief Where the frame was added.
    source_location where;
    /// \brief How it was added.
    frame_kind kind = frame_kind::context;
};

/// \brief One key-value field of a traced error; the value is already text, and masked if it was a secret.
/// \ingroup core
struct error_field
{
    /// \brief The key.
    std::string key;
    /// \brief The value as text.
    std::string value;

    /// \brief Equality.
    /// \return the result.
    friend bool operator==(const error_field&, const error_field&) = default;
};

/// \brief What a masked value is written as: the same text as `snicholls::masked_text`.
/// \ingroup core
inline constexpr std::string_view masked_field_text = "********";

/// \brief A captured stack trace provider: returns the trace of the calling thread, skipping `skip` frames.
/// \ingroup core
using stack_trace_hook = std::string (*)(int skip);

namespace traced_detail {

inline std::atomic<stack_trace_hook>& stack_hook() noexcept
{
    static std::atomic<stack_trace_hook> hook{nullptr};
    return hook;
}

/// A secret: `snicholls::secret<T>` or anything else that hides its value behind `reveal()`.
template <class T>
concept masked_value = requires(const T& t) { t.reveal(); };

struct details
{
    std::vector<context_frame> frames;
    std::vector<error_field> fields;
    std::string stack;
};

inline std::string_view basename(std::string_view path) noexcept
{
    const auto slash = path.find_last_of("/\\");
    return slash == std::string_view::npos ? path : path.substr(slash + 1);
}

inline std::string site(source_location where)
{
    std::string s(basename(where.file_name()));
    s += ':';
    s += std::to_string(where.line());
    return s;
}

} // namespace traced_detail

/// \brief Installs the stack trace provider used by `capture_stack()` / `with_stack()` (nullptr: none).
/// \ingroup core
///
/// The core has none (a portable stack trace needs C++23 `<stacktrace>` or a platform library); the composition
/// root installs one, for example from super-log's `superlog::stacktrace`.
/// Thread-safe (atomic).
/// \param hook the provider.
inline void set_stack_trace_hook(stack_trace_hook hook) noexcept
{
    traced_detail::stack_hook().store(hook, std::memory_order_release);
}

/// \brief The current stack trace from the installed hook, or an empty string if none is installed.
/// \ingroup core
/// \param skip frames to skip above the caller.
/// \return the trace, one frame per line.
/// \throws std::bad_alloc, or what the hook throws.
[[nodiscard]] inline std::string capture_stack_trace(int skip = 0)
{
    const stack_trace_hook hook = traced_detail::stack_hook().load(std::memory_order_acquire);
    return hook ? hook(skip + 1) : std::string();
}

/// \brief A field value as text, with secrets (anything with `reveal()`) masked.
/// \ingroup core
///
/// A string is itself, a null `const char*` is `(null)`, a `bool` is `true`/`false`, anything else formattable goes
/// through `std::format("{}")`.
/// \param value the value.
/// \return the text.
/// \throws std::bad_alloc.
template <class T>
[[nodiscard]] std::string error_field_text(const T& value)
{
    if constexpr (traced_detail::masked_value<T>)
        return std::string(masked_field_text);
    else if constexpr (std::is_same_v<T, bool>)
        return value ? "true" : "false";
    else if constexpr (std::is_convertible_v<const T&, std::string_view>)
    {
        if constexpr (std::is_pointer_v<T>)
            if (value == nullptr)
                return std::string("(null)");   // std::string_view(nullptr) is undefined behaviour
        return std::string(std::string_view(value));
    }
    else if constexpr (result_detail::std_formattable<T>)
        return std::format("{}", value);
    else
        static_assert(snicholls::always_false_v<T>,
                      "snicholls::traced: a field value must be a secret, string-like or std::format-able");
}

/// \brief An error E with the `source_location` where it was created, a chain of context frames, key-value
///        fields and an optional stack trace (anyhow / eyre for `snicholls::result`).
/// \ingroup core
///
/// \overview
/// Logging an error well needs more than the error: where it happened, what the program was doing at each level on
/// the way up, the identifiers involved, and sometimes a stack. `snicholls::traced<E>` carries all of it beside the
/// original E, so the function that finally handles the error can log a complete account in one place:
/// `snicholls::err(snicholls::traced{e})` records the location; `r.context("loading {}", id)` adds a frame at the caller;
/// `SNICHOLLS_TRY` adds a "propagated" frame with the expression text; `r.with_field("account", id)` adds a field;
/// `r.with_stack()` captures a stack through the installed hook. `format_error_report` renders text,
/// `error_report_fields` a flat field list, and a `snicholls::error_sink` (a super-log adapter, say)
/// receives the structured `snicholls::error_report`.
///
/// Use it where errors are logged or shown to people. Keep plain error types (an enum, `std::error_code`) on hot
/// paths whose errors are handled locally.
///
/// \relations
/// - `snicholls::result<V, snicholls::traced<E>>`: `context`, `with_field`, `with_stack`, `trace`, `log_err`, `SNICHOLLS_TRY`.
/// - `snicholls::error_report`, `snicholls::make_error_report`, `snicholls::format_error_report`, `snicholls::error_report_fields`.
/// - `snicholls::error_sink`: the logging interface; `snicholls::text_error_sink`, or a consumer's super-log adapter.
/// - Module: \ref core.
///
/// \impls
/// None: a single implementation.
///
/// \contracts
/// - **Cost:** `sizeof(traced<E>) == sizeof(E)` + a `source_location` (one pointer; three words on MSVC) + one
///   pointer, rounded to alignment. Creating one stores the location and does not allocate (beyond what E's own constructor does). The
///   first `add_context` / `add_field` / `capture_stack` allocates one block holding the frames, fields and stack;
///   each frame or field then allocates its text. A success never constructs a traced at all, so the success path
///   of a `result<V, traced<E>>` costs what a `result<V, E>` of the same size costs.
/// - Copying is a deep copy (allocates when there are details); moving is `noexcept` when E's move is.
/// - Equality and `std::hash` use E only: the context is diagnostics, not identity.
/// - Not thread-safe beyond the usual value-type rules.
///
/// \errors
/// `std::bad_alloc` from the adders and from copies. `std::format_error` from `context(fmt, args...)` and
/// `snicholls::fail(fmt, args...)` when an argument is invalid for its format at run time, e.g. a negative dynamic
/// width (`"{:{}}"` with `-1`): the format string itself is checked at compile time, its run-time arguments are not.
///
/// \examples
/// \snippet traced_demo.cpp chain
/// \snippet traced_demo.cpp report
///
/// \perf
/// Not benchmarked: the error path allocates by design and is not on a hot path. The success path is
/// benchmarked as part of `snicholls::result` (benchmarks/result_benchmark.cpp).
///
/// \testing
/// - Tests: tests/traced_test.cpp (chain, fields, masking, stack hook, formatter, sinks with GoogleMock),
///   tests/support/error_sink_contract.hpp (the sink contract suite).
/// - Demo: demos/traced_demo.cpp.
///
/// \deps
/// CMake target `snicholls::core`; standard library only (core).
///
/// \gotchas
/// - Never put a secret in the context text: format arguments are printed as they are (a `snicholls::secret` has no
///   `std::formatter`, so passing one is a compile error). Fields mask secrets.
/// - Core masks field *values* that are secrets (anything with `reveal()`) and nothing else: there is no masking by
///   key name here. Masking keys that look secret is a sink's own choice;
///   `snicholls::text_error_sink` sees every key and every non-secret value as given.
/// - `SNICHOLLS_TRY` records the text of the expression it propagates, verbatim, in a "propagated" frame: an
///   expression that spells out a secret (a literal, a variable name that matters) puts it in the log.
/// - `rethrow()` of a traced error throws the inner E; the chain is not in the exception.
/// - `source_location::function_name()` differs between compilers, so the text and the field list carry
///   `file:line` only (the report keeps the full location).
///
/// \status
/// Experimental, pre-1.0. Design: [RESULT_PLAN.md](RESULT_PLAN.md).
///
/// \tparam E the inner error type.
template <class E>
class traced
{
    static_assert(!is_traced_v<E>, "snicholls::traced<snicholls::traced<E>>: already traced");

    template <class>
    friend class traced;

public:
    /// \brief The inner error type.
    using error_type = E;

    /// \brief Wraps `error`, recording the caller's location.
    /// \param error the error.
    /// \param where where the error was created (the caller, by default).
    explicit traced(E error, source_location where = source_location::current()) noexcept(
        std::is_nothrow_move_constructible_v<E>)
        : error_(std::move(error)), where_(where)
    {
    }

    /// \brief Converts from a traced error of another inner type, keeping the location and the details.
    /// \param other the traced error.
    template <class Y>
        requires(!std::is_same_v<Y, E> && std::is_constructible_v<E, Y &&>)
    explicit traced(traced<Y>&& other) noexcept(std::is_nothrow_constructible_v<E, Y&&>)
        : error_(std::move(other.error_)), where_(other.where_), details_(std::move(other.details_))
    {
    }

    /// \brief Copies, deeply.
    /// \param o the other.
    traced(const traced& o)
        : error_(o.error_), where_(o.where_),
          details_(o.details_ ? std::make_unique<traced_detail::details>(*o.details_) : nullptr)
    {
    }

    /// \brief Moves.
    traced(traced&&) noexcept(std::is_nothrow_move_constructible_v<E>) = default;

    /// \brief Copy-assigns, deeply.
    /// \param o the other.
    /// \return `*this`.
    traced& operator=(const traced& o)
    {
        if (this != &o)
        {
            traced copy(o);
            *this = std::move(copy);
        }
        return *this;
    }

    /// \brief Move-assigns.
    /// \return `*this`.
    traced& operator=(traced&&) noexcept(std::is_nothrow_move_assignable_v<E>) = default;

    /// \brief Destroys it.
    ~traced() = default;

    /// \name The inner error
    ///@{
    /// \brief The inner error.
    /// \return the error.
    [[nodiscard]] const E& error() const& noexcept { return error_; }
    [[nodiscard]] E& error() & noexcept { return error_; }
    [[nodiscard]] E&& error() && noexcept { return std::move(error_); }
    ///@}

    /// \brief Where the error was created.
    /// \return the location.
    [[nodiscard]] source_location where() const noexcept { return where_; }

    /// \brief The context frames, in the order they were added (innermost first).
    /// \return the frames.
    [[nodiscard]] std::span<const context_frame> frames() const noexcept
    {
        return details_ ? std::span<const context_frame>(details_->frames) : std::span<const context_frame>();
    }

    /// \brief The fields, in the order they were added.
    /// \return the fields.
    [[nodiscard]] std::span<const error_field> fields() const noexcept
    {
        return details_ ? std::span<const error_field>(details_->fields) : std::span<const error_field>();
    }

    /// \brief The captured stack trace (empty if none was requested or no hook is installed).
    /// \return the trace.
    [[nodiscard]] std::string_view stack() const noexcept { return details_ ? details_->stack : std::string_view(); }

    /// \brief Adds a context frame.
    /// \param message the message.
    /// \param where where it was added.
    /// \param kind how it was added.
    /// \return `*this`.
    /// \throws std::bad_alloc (unchanged on failure).
    traced& add_context(std::string message, source_location where = source_location::current(),
                        frame_kind kind = frame_kind::context)
    {
        get_details().frames.push_back(context_frame{std::move(message), where, kind});
        return *this;
    }

    /// \brief Adds a field; the value goes through `snicholls::error_field_text` (secrets masked).
    /// \param key the key.
    /// \param value the value.
    /// \return `*this`.
    /// \throws std::bad_alloc (unchanged on failure).
    template <class T>
    traced& add_field(std::string_view key, const T& value)
    {
        error_field f{std::string(key), error_field_text(value)};
        get_details().fields.push_back(std::move(f));
        return *this;
    }

    /// \brief Captures a stack trace through the installed hook (`snicholls::set_stack_trace_hook`).
    /// \return `*this`.
    /// \throws std::bad_alloc, or what the hook throws.
    traced& capture_stack()
    {
        std::string s = capture_stack_trace(1);
        get_details().stack = std::move(s);
        return *this;
    }

    /// \brief One line: the context messages outermost first, then the error ("loading book: not found").
    ///
    /// Propagated frames are not part of it.
    /// \return the message.
    /// \throws std::bad_alloc.
    [[nodiscard]] std::string message() const
    {
        std::string out;
        for (auto it = frames().rbegin(); it != frames().rend(); ++it)
        {
            if (it->kind != frame_kind::context)
                continue;
            out += it->message;
            out += ": ";
        }
        out += error_text(error_);
        return out;
    }

    /// \brief `snicholls::error_text` of a traced error: its `message()`.
    /// \param t the traced error.
    /// \return the message.
    friend std::string snicholls_error_text(const traced& t) { return t.message(); }

    /// \brief Equal when the inner errors are equal.
    /// \param a a traced error.
    /// \param b a traced error.
    /// \return the result.
    [[nodiscard]] friend bool operator==(const traced& a, const traced& b) noexcept(noexcept(a.error_ == b.error_))
        requires std::equality_comparable<E>
    {
        return a.error_ == b.error_;
    }

private:
    traced_detail::details& get_details()
    {
        if (!details_)
            details_ = std::make_unique<traced_detail::details>();
        return *details_;
    }

    E error_;
    source_location where_;
    std::unique_ptr<traced_detail::details> details_;
};

/// \brief Deduces `snicholls::traced<E>` from `snicholls::traced{e}`.
/// \ingroup core
template <class E>
traced(E) -> traced<E>;

/// \brief A traced error located at the caller: `return snicholls::traced_err(io_error::not_found);`.
/// \ingroup core
/// \param e the error.
/// \param where the caller's location.
/// \return `snicholls::err(snicholls::traced<E>(e, where))`.
template <class E>
[[nodiscard]] err_t<traced<std::decay_t<E>>> traced_err(
    E&& e, source_location where = source_location::current()) noexcept(std::is_nothrow_constructible_v<std::decay_t<E>, E>)
{
    return err_t<traced<std::decay_t<E>>>{traced<std::decay_t<E>>(std::decay_t<E>(std::forward<E>(e)), where)};
}

/// \brief anyhow's `anyhow!` / `bail!`: a traced `std::string` error, formatted, located at the caller.
/// \ingroup core
/// \param fmt the format string (checked at compile time; its location is recorded).
/// \param args the arguments.
/// \return `snicholls::err(snicholls::traced<std::string>(...))`.
/// \throws std::bad_alloc; std::format_error when an argument is invalid for its format at run time (e.g. a
///         negative dynamic width).
template <class... A>
[[nodiscard]] err_t<traced<std::string>> fail(result_impl::located_format<std::type_identity_t<A>...> fmt, A&&... args)
{
    return err_t<traced<std::string>>{traced<std::string>(std::format(fmt.fmt, std::forward<A>(args)...), fmt.where)};
}

/// \brief `SNICHOLLS_TRY` from a traced error into a traced error: adds a "propagated" frame.
/// \ingroup core
///
/// The frame holds the expression text and the `SNICHOLLS_TRY` location; the inner error converts if the types differ.
/// \tparam X the enclosing function's inner error type.
/// \tparam Y the expression's inner error type.
template <class X, class Y>
struct error_conversion<traced<X>, traced<Y>>
{
    /// \brief Converts.
    /// \param e the error.
    /// \param expr the expression text.
    /// \param where the `SNICHOLLS_TRY` location.
    /// \return the traced error.
    static traced<X> convert(traced<Y>&& e, const char* expr, source_location where)
    {
        e.add_context(expr, where, frame_kind::propagated);
        if constexpr (std::is_same_v<X, Y>)
            return std::move(e);
        else
            return traced<X>(std::move(e));
    }
};

/// \brief `SNICHOLLS_TRY` from a plain error into a traced one: wraps it, located at the `SNICHOLLS_TRY`, with a
///        "propagated" frame.
/// \ingroup core
/// \tparam X the enclosing function's inner error type.
/// \tparam Y the expression's error type.
template <class X, class Y>
    requires(!is_traced_v<Y>)
struct error_conversion<traced<X>, Y>
{
    /// \brief Converts.
    /// \param e the error.
    /// \param expr the expression text.
    /// \param where the `SNICHOLLS_TRY` location.
    /// \return the traced error.
    static traced<X> convert(Y&& e, const char* expr, source_location where)
    {
        traced<X> t(X(std::move(e)), where);
        t.add_context(expr, where, frame_kind::propagated);
        return t;
    }
};

/// \brief `SNICHOLLS_TRY` from a traced error into a plain one: keeps the inner error, drops the context.
/// \ingroup core
///
/// Log it first if the context matters.
/// \tparam To the enclosing function's error type.
/// \tparam Y the expression's inner error type.
template <class To, class Y>
    requires(!is_traced_v<To>)
struct error_conversion<To, traced<Y>>
{
    /// \brief Converts.
    /// \param e the error.
    /// \return the inner error, converted.
    static To convert(traced<Y>&& e, const char* /*expr*/, source_location /*where*/)
    {
        return To(std::move(e).error());
    }
};

/// \brief `rethrow()` of a traced error throws its inner error (`snicholls::throw_error`).
/// \ingroup core
/// \tparam X the inner error type.
template <class X>
struct error_exception<traced<X>>
{
    /// \brief Throws the inner error.
    /// \param t the traced error.
    [[noreturn]] static void raise(const traced<X>& t) { throw_error(t.error()); }
};

// -------------------------------------------------------------------------------------------------------------------
// Reports, formatting, sinks
// -------------------------------------------------------------------------------------------------------------------

/// \brief Everything known about one error, ready to be logged: std types only.
/// \ingroup core
struct error_report
{
    /// \brief One line: context messages outermost first, then the error (`snicholls::error_text`).
    std::string message;
    /// \brief The error alone (`snicholls::error_text` of the inner error).
    std::string error;
    /// \brief Where the error was created (traced), else where it was logged.
    source_location where;
    /// \brief True when `where` is the error's own location (a traced error), false when it is the log site.
    bool located = false;
    /// \brief The context frames, innermost first.
    std::vector<context_frame> frames;
    /// \brief The fields (secrets already masked).
    std::vector<error_field> fields;
    /// \brief The stack trace, if one was captured.
    std::string stack;
};

/// \brief The report of any error: all of a traced one; the text and the log site of another.
/// \ingroup core
/// \param e the error.
/// \param log_site where it is being logged.
/// \return the report.
/// \throws std::bad_alloc.
template <class E>
[[nodiscard]] error_report make_error_report(const E& e, source_location log_site = source_location::current())
{
    error_report r;
    if constexpr (is_traced_v<E>)
    {
        r.message = e.message();
        r.error = error_text(e.error());
        r.where = e.where();
        r.located = true;
        r.frames.assign(e.frames().begin(), e.frames().end());
        r.fields.assign(e.fields().begin(), e.fields().end());
        r.stack = std::string(e.stack());
    }
    else
    {
        r.message = error_text(e);
        r.error = r.message;
        r.where = log_site;
    }
    return r;
}

/// \brief The report's location as `file:line` (the file's base name, so it is the same on every machine).
/// \ingroup core
/// \param where the location.
/// \return the text.
[[nodiscard]] inline std::string source_site(source_location where) { return traced_detail::site(where); }

/// \brief The report as human-readable text.
/// \ingroup core
///
/// \code
///   error: loading book desk-7: file not found
///     at book.cpp:12
///     context (innermost first):
///       0: propagated `read_file(path)` at book.cpp:20
///       1: loading book desk-7 at book.cpp:31
///     fields:
///       desk = desk-7
///     stack:
///       ...
/// \endcode
/// For an untraced error the second line is `logged at file:line`. Empty sections are omitted. Newlines,
/// carriage returns and backslashes in the message, context frames, field keys and values are escaped (`\n`, `\r`,
/// `\\`), so a value cannot forge a line; the stack is the only multi-line section.
/// \param r the report.
/// \return the text (no trailing newline).
/// \throws std::bad_alloc.
[[nodiscard]] inline std::string format_error_report(const error_report& r)
{
    // Every line of text is one line: a newline in a message, key or value is written as \n (and \r, \\ alike), so
    // a value cannot forge a line of its own ("  fields:", "[ERROR] ..."). Only the stack keeps its own lines.
    const auto one_line = [](std::string_view s) {
        std::string escaped;
        escaped.reserve(s.size());
        for (const char c : s)
        {
            if (c == '\n')
                escaped += "\\n";
            else if (c == '\r')
                escaped += "\\r";
            else if (c == '\\')
                escaped += "\\\\";
            else
                escaped += c;
        }
        return escaped;
    };
    std::string out = "error: " + one_line(r.message);
    out += r.located ? "\n  at " : "\n  logged at ";
    out += source_site(r.where);
    if (!r.frames.empty())
    {
        out += "\n  context (innermost first):";
        for (std::size_t i = 0; i < r.frames.size(); ++i)
        {
            const auto& f = r.frames[i];
            out += "\n    " + std::to_string(i) + ": ";
            if (f.kind == frame_kind::propagated)
                out += "propagated `" + one_line(f.message) + "`";
            else
                out += one_line(f.message);
            out += " at " + source_site(f.where);
        }
    }
    if (!r.fields.empty())
    {
        out += "\n  fields:";
        for (const auto& f : r.fields)
            out += "\n    " + one_line(f.key) + " = " + one_line(f.value);
    }
    if (!r.stack.empty())
    {
        out += "\n  stack:";
        std::string_view s = r.stack;
        while (!s.empty())
        {
            const auto nl = s.find('\n');
            out += "\n    ";
            out += s.substr(0, nl);
            s = nl == std::string_view::npos ? std::string_view() : s.substr(nl + 1);
        }
    }
    return out;
}

/// \brief The keys `snicholls::error_report_fields` reserves; a user field with one of these keys is written as
///        `field.<key>`.
/// \ingroup core
/// \param key the key.
/// \return true if reserved.
[[nodiscard]] constexpr bool is_reserved_error_field(std::string_view key) noexcept
{
    return key == "error" || key == "src" || key == "stack" || key == "where" || key.starts_with("context.") ||
           key.starts_with("field.");
}

/// \brief The report as a flat, ordered list of string fields (for structured logs).
/// \ingroup core
///
/// In order: `error` (the inner error's text), `src` (`file:line` of the origin or the log site), one
/// `context.<i>` per frame (innermost first, `"<message> @ file:line"`, a propagated frame as
/// `"propagated <expr> @ file:line"`), the user fields (a reserved key becomes `field.<key>`), and `stack` if
/// captured. The one-line message is not repeated: it is the log line's message.
/// \param r the report.
/// \return the fields.
/// \throws std::bad_alloc.
[[nodiscard]] inline std::vector<error_field> error_report_fields(const error_report& r)
{
    std::vector<error_field> out;
    out.reserve(3 + r.frames.size() + r.fields.size());
    out.push_back({"error", r.error});
    out.push_back({"src", source_site(r.where)});
    for (std::size_t i = 0; i < r.frames.size(); ++i)
    {
        const auto& f = r.frames[i];
        std::string v = f.kind == frame_kind::propagated ? "propagated " + f.message : f.message;
        v += " @ " + source_site(f.where);
        out.push_back({"context." + std::to_string(i), std::move(v)});
    }
    for (const auto& f : r.fields)
        out.push_back({is_reserved_error_field(f.key) ? "field." + f.key : f.key, f.value});
    if (!r.stack.empty())
        out.push_back({"stack", r.stack});
    return out;
}

/// \brief The text of a traced error: `format_error_report(make_error_report(t))`.
/// \ingroup core
/// \param t the traced error.
/// \return the text.
/// \throws std::bad_alloc.
template <class E>
[[nodiscard]] std::string to_string(const traced<E>& t)
{
    return format_error_report(make_error_report(t, t.where()));
}

/// \brief Where errors are logged: receives a structured `snicholls::error_report` at a level.
/// \ingroup core
///
/// \overview
/// The log hook of `snicholls::result`: `r.log_err(sink)` / `r.log_err(sink, level)` call `log_error`, which builds the
/// report and calls `write` once. Implement `write` to send it anywhere; the core stays free of logging libraries.
///
/// \relations
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
/// - Tests: tests/traced_test.cpp (GoogleMock sink, `text_error_sink`), tests/support/error_sink_contract.hpp.
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

} // namespace snicholls

/// \brief `std::hash` of a `snicholls::traced<E>`: the hash of the inner error (enabled when `std::hash<E>` is).
/// \ingroup core
/// \tparam E the inner error type.
template <class E>
    requires std::is_default_constructible_v<std::hash<E>>
struct std::hash<snicholls::traced<E>>
{
    /// \brief The hash.
    /// \param t the traced error.
    /// \return `std::hash<E>{}(t.error())`.
    [[nodiscard]] std::size_t operator()(const snicholls::traced<E>& t) const noexcept(noexcept(std::hash<E>{}(t.error())))
    {
        return std::hash<E>{}(t.error());
    }
};

#endif /* snicholls_core_traced_hpp */
