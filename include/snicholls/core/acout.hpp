//
//  acout.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  That mutex was `static` in a header, so every translation unit had its own and two files printing at once
//  still interleaved; and the lock was held while the arguments were formatted. This version assembles the
//  message off to the side and writes it in one call under one program-wide lock, so nothing is held while
//  formatting.
//

/// \file
/// \brief `snicholls::acout`, `acerr`, `print`, `println`: thread-safe printing, one write per message, format strings included.
/// \ingroup core
///
/// **Overview.** Several threads printing to one stream interleave their characters, because
/// `std::cout << a << b` is two calls and another thread's call can land between them. The
/// fix is to assemble the whole message first and hand it to the stream in a single call:
///
/// \snippet acout_demo.cpp acout
///
/// `acout` is a message: stream into it, or `print` a format string into it, and when it goes
/// out of scope (the end of the statement, for a temporary) the text is written once. Two
/// messages from two threads never interleave: for a `std::ostream` the one write is made
/// under one program-wide lock; for a `FILE*` it is one `fwrite`, which the C library locks.
/// The lock is held for that write alone, never while the message is formatted, so a message
/// may print another (`acout{} << f()` where `f` prints) and formatting a slow argument never
/// blocks anyone. Messages under 256 bytes allocate nothing.
///
/// `print` and `println` are the one-line forms, C++23's `std::print` shape on C++20:
///
/// \snippet acout_demo.cpp print
///
/// Formatting is `std::format`, with the format string checked at compile time. The `{fmt}`
/// library itself cannot enter core (standard library only); its API is this one.
///
/// A type that knows how to print itself gives it one member, and that is the whole contract
/// (`snicholls::printable`): the classic `print(std::ostream&) const`, or the better
/// `print(snicholls::acout&) const`, inside which format strings and streaming both work
/// and the whole object is written in one call. `acout` streams either through the member,
/// `print(value)` and `println(value)` write one in one line, and
/// `using snicholls::print_operators::operator<<;` lends both to any plain stream, the way
/// magic_enum's `iostream_operators` does for enumerations:
///
/// \snippet acout_demo.cpp printable
/// \snippet acout_demo.cpp printable_use
///
/// Everything else prints too, through one engine (`snicholls::print_to`) that `acout` uses for
/// every `<<` and that `print_operators` lends to plain streams: sequences as `[a, b]`, sets as
/// `{a, b}`, maps as `{k: v}`, pairs and tuples as `(a, b)`, a `std::optional` as its value or
/// `none`, a `std::variant` as what it holds, a `snicholls::result` as `ok(v)` or `err(e)`, a
/// `snicholls::meta::type_list` as `type_list<int, double>`, recursively, with strings and
/// characters quoted inside a container:
///
/// \snippet acout_demo.cpp containers
///
/// The engine is a `first_match` chain, so a `std::string` is text and not a range of chars, and
/// a type's own `operator<<` is taken before its shape is looked at. A type nothing in the chain
/// accepts is a compile error with a message, not a silent address.
///
/// Colour is part of the message, and ends with it, so one thread's colour cannot tint another
/// thread's line. `colours::red` in a message holds until `colour_reset`; `styled(colour, value)`
/// colours one value, in a container too; `print(colour, ...)` and `println(colour, ...)` colour a
/// whole line. Any `rgb(0xRRGGBB)` works where a named colour does:
///
/// \snippet acout_demo.cpp colour
///
/// A type that declares display attributes (`display.hpp`, `meta/attributes.hpp`) prints in its colour with its label
/// or number format, everywhere and in containers too; a number with `sign_colours` is green when up and red when
/// down, and `by_sign(x)` does the same for a bare number. `print_displayed` prints a value under attributes given.
///
/// Escape codes are written only where they mean something (`snicholls::colour_mode`). By
/// default that is `std::cout`, `std::cerr`, `std::clog`, `stdout` and `stderr`, unless `NO_COLOR`
/// is set or `TERM` is `dumb`; a string stream or a file gets the plain text.
///
/// **Relationships.** `colour.hpp` for the colours; `meta/concepts.hpp` for `printable`; `meta/type_name.hpp` for the names in a
/// `type_list`; `result.hpp`, whose `error_text` is what to print for an error;
/// ts-moveables' logger, for logging with levels, fields and sinks, which this is not;
/// C++20's `std::osyncstream`, the same idea in the standard, which libc++ ships only as
/// experimental and core therefore does not depend on. Group: \ref core.
///
/// **Guarantees.**
/// - Atomic per message: the characters of one `acout` reach the target in one call, under a
///   lock shared by every `acout` in the program, so two messages from two threads never
///   interleave, whichever thread finishes first. Text written to the same target *not*
///   through `acout` can still land between two of ours.
/// - The lock is one `std::mutex`, program-wide, held only for the write; never while
///   formatting, so no user code runs under it and no message can deadlock another. It is the
///   only state this header owns.
/// - No allocation for a message of up to 256 bytes; longer messages grow a `std::string`.
/// - The message's formatting state (`std::hex`, width, precision) is its own, fresh per
///   message, not the target's.
/// - `flush()`, `std::endl` and `std::flush` request a flush of the target after the write;
///   otherwise the target's own buffering applies, as for a plain stream.
/// - `emit()` writes now and empties the message, which is then reusable. The destructor
///   calls it and swallows exceptions (there is nowhere to report them); call `emit()`
///   yourself to see them.
/// - Not copyable or movable: a message is a scoped object.
///
/// **Errors.** `std::bad_alloc` from a message over 256 bytes; `std::format_error` from a
/// format argument invalid at run time; `std::system_error` if the lock cannot be taken;
/// whatever a streamed type's `operator<<` throws. A failed write sets the target's state as
/// the stream would.
///
/// **Performance.** Benchmarked in `benchmarks/acout_benchmark.cpp` against a plain
/// unsynchronised stream, and against `std::osyncstream` where the library has it, writing
/// the same 4,096 messages (a label, an integer, a double) to a string stream, after a
/// fairness check that the text is identical. Release, Apple clang 17, Xeon W-3245,
/// 2026-11-06: 432 ns per message through `acout`, 375 ns through the plain stream, 341 ns
/// through `println` with a format string. The 57 ns over the plain stream is the lock and
/// the one extra copy; the format form is faster than streaming because `std::format` of a
/// `double` is faster than `operator<<`. Colour, 2026-11-07, same
/// machine: with a coloured label, 554 ns per message against 421 ns for a plain stream given
/// the same codes by hand; the same run's message without colour took 456 ns against 381 ns.
/// The engine on a container, the same day and machine:
/// 4,096 rows of 16 `int`s through `print_to` 1.17 us a row against 1.16 us for the loop
/// written by hand, identical text; the branch choice is at compile time and the cost is
/// the stream's.
///
/// **Tests and demos.** `tests/acout_test.cpp`, `demos/acout_demo.cpp`.
///
/// **Dependencies.** `<atomic>`, `<concepts>`, `<cstdint>`, `<cstdio>`, `<cstdlib>`, `<format>`, `<iostream>`, `<iterator>`, `<mutex>`,
/// `<optional>`, `<ostream>`, `<ranges>`, `<streambuf>`, `<string>`, `<string_view>`, `<tuple>`,
/// `<type_traits>`, `<utility>`, `<variant>`, and `colour.hpp`, `meta/concepts.hpp`, `overloaded.hpp`,
/// `meta/type_list.hpp`, `meta/type_name.hpp`, `result.hpp`. Heavier than it looks: the engine
/// prints `result`s, so `result.hpp` comes along. The one header in
/// core that performs I/O and holds a mutex, admitted by name in CLAUDE.md: it does nothing
/// but hand a finished message to a stream the caller names.
///
/// **Known issues.** `std::cout` redirected to a file (`./app > log.txt`) still gets the escape
/// codes of whatever is printed in colour: telling a terminal from a file takes `isatty`, a
/// platform header, which core does not include. A program that may be redirected makes the check
/// itself and calls `set_colour_mode(colour_mode::never)`. On Windows, colour is on by default only
/// in Windows Terminal or where `TERM` is set, because a classic console must be switched into
/// escape-code mode by a platform call. Built without RTTI, in `automatic` mode, a coloured value
/// nested in a container prints without its colour. One lock for every stream target means two threads printing to two
/// different streams wait for each other for the length of a `write`; a per-stream lock would
/// need a registry, which `std::osyncstream` provides but libc++ does not ship. A type that is
/// both `printable` and has its own `operator<<` is streamed through `print` by `acout`; on a
/// plain stream the two are ambiguous only if the `print_operators` declaration is in scope.
/// A type with both `print` forms is streamed through the `acout` one. The engine does not
/// read `std::boolalpha` for its own words (`none`, `ok(...)`) but passes `bool` elements to
/// the stream as they are. A scoped enumeration prints by name (`meta/enums.hpp`), or as its integer
/// when its value is not an enumerator; one with its own `operator<<` uses that, and an unscoped one prints
/// as the stream prints it.
///
/// **Status.** New in 0.5.0.

#ifndef snicholls_core_acout_hpp
#define snicholls_core_acout_hpp

#include <snicholls/core/colour.hpp>
#include <snicholls/core/display.hpp>
#include <snicholls/core/meta/attributes.hpp>
#include <snicholls/core/meta/concepts.hpp>
#include <snicholls/core/meta/enums.hpp>
#include <snicholls/core/overloaded.hpp>
#include <snicholls/core/meta/type_list.hpp>
#include <snicholls/core/meta/type_name.hpp>
#include <snicholls/core/result.hpp>

#include <atomic>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <format>
#include <iostream>
#include <iterator>
#include <mutex>
#include <optional>
#include <ostream>
#include <ranges>
#include <streambuf>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>

namespace snicholls {

    /// \brief When a message writes the escape codes for the colours it is given.
    /// \ingroup core
    enum class colour_mode : std::uint8_t {
        automatic,   ///< To `std::cout`, `std::cerr`, `std::clog`, `stdout` and `stderr`, unless the environment says not to; never to anything else.
        always,      ///< To every target, a file or a string stream included.
        never,       ///< To no target: colours are ignored.
    };

    namespace detail {

        inline std::atomic<colour_mode> acout_colour_mode{colour_mode::automatic};

        // The rule for colour_mode::automatic, apart from reading the variables. NO_COLOR is the
        // no-color.org convention. A Windows console shows escape codes only when told to, which takes a
        // platform call, so there colour needs a terminal known to be on: Windows Terminal, or one that sets TERM.
        constexpr bool environment_allows_colour(const char* no_color, const char* term, const char* wt_session, bool on_windows) noexcept
        {
            if (no_color != nullptr && no_color[0] != '\0')
                return false;
            if (term != nullptr && std::string_view{term} == "dumb")
                return false;
            return !on_windows || wt_session != nullptr || term != nullptr;
        }

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)   // std::getenv is standard C++; only reading it, once, before any setenv could matter
#endif
        // Read once, on the first coloured message of the program.
        inline bool colour_allowed_by_environment() noexcept
        {
#if defined(_WIN32)
            constexpr bool on_windows = true;
#else
            constexpr bool on_windows = false;
#endif
            static const bool allowed =
                environment_allows_colour(std::getenv("NO_COLOR"), std::getenv("TERM"), std::getenv("WT_SESSION"), on_windows);
            return allowed;
        }
#if defined(_MSC_VER)
#pragma warning(pop)
#endif

        // The one lock, program-wide (an inline variable: one instance however many translation units),
        // held only while a finished message is handed to a std::ostream. Never held while formatting.
        inline std::mutex acout_write_lock;

        // The message's buffer: 256 bytes in place, a std::string beyond. A flush requested through the
        // stream (std::endl, std::flush) is recorded, not performed, until the message is emitted.
        class message_buffer : public std::streambuf {
        public:
            message_buffer(std::ostream* target, std::FILE* file) noexcept : target_(target), file_(file)
            {
                setp(inline_, inline_ + sizeof inline_);
            }

            std::ostream* target() const noexcept { return target_; }
            std::FILE* file() const noexcept { return file_; }

            // Whether this message writes escape codes. A message printed inside another (a type's
            // print(acout&) reached through a container) asks the message it will end up in.
            bool colour_wanted() const noexcept
            {
                const colour_mode mode = acout_colour_mode.load(std::memory_order_relaxed);
                if (mode != colour_mode::automatic)
                    return mode == colour_mode::always;
                if (file_ != nullptr)
                    return (file_ == stdout || file_ == stderr) && colour_allowed_by_environment();
                if (target_ == &std::cout || target_ == &std::cerr || target_ == &std::clog)
                    return colour_allowed_by_environment();
                const message_buffer* outer = enclosing();
                return outer != nullptr && outer->colour_wanted();
            }

            std::optional<colour> active_colour() const noexcept { return active_; }
            void set_active_colour(std::optional<colour> c) noexcept { active_ = c; }

            // The colour the enclosing message is in, which this one must hand back.
            std::optional<colour> inherited_colour() const noexcept
            {
                const message_buffer* outer = enclosing();
                return outer != nullptr ? outer->current_colour() : std::nullopt;
            }

            std::optional<colour> current_colour() const noexcept { return active_ ? active_ : inherited_colour(); }

            // The whole message, contiguous. Spills the in-place part when a std::string is in use.
            std::string_view text()
            {
                if (spilled_.empty())
                    return {pbase(), static_cast<std::size_t>(pptr() - pbase())};
                spilled_.append(pbase(), pptr());
                setp(inline_, inline_ + sizeof inline_);
                return spilled_;
            }

            // A copy of the message so far, without disturbing it.
            std::string copy() const
            {
                std::string out = spilled_;
                out.append(pbase(), pptr());
                return out;
            }

            bool flush_requested() const noexcept { return flush_; }
            void request_flush() noexcept { flush_ = true; }

            void reset() noexcept
            {
                spilled_.clear();
                setp(inline_, inline_ + sizeof inline_);
                flush_ = false;
            }

        protected:
            int_type overflow(int_type ch) override
            {
                spilled_.append(pbase(), pptr());
                if (!traits_type::eq_int_type(ch, traits_type::eof()))
                    spilled_.push_back(traits_type::to_char_type(ch));
                setp(inline_, inline_ + sizeof inline_);
                return traits_type::not_eof(ch);
            }

            int sync() override
            {
                flush_ = true;
                return 0;
            }

        private:
            // Only ever asked on the coloured path, so a message without colour pays nothing. Without
            // RTTI a nested message cannot see the enclosing one, and prints without colour.
            const message_buffer* enclosing() const noexcept
            {
#if defined(__cpp_rtti)
                return target_ != nullptr ? dynamic_cast<const message_buffer*>(target_->rdbuf()) : nullptr;
#else
                return nullptr;
#endif
            }

            char inline_[256];
            std::string spilled_;
            std::ostream* target_;
            std::FILE* file_;
            std::optional<colour> active_;
            bool flush_ = false;
        };

    } // namespace detail

    /// \brief The type of `snicholls::colour_reset`.
    /// \ingroup core
    struct colour_reset_t {};

    /// \brief Ends the colour a message is in: `acout{} << colours::red << "failed" << colour_reset << " at 3\n";`.
    /// \ingroup core
    inline constexpr colour_reset_t colour_reset{};

    /// \brief Set when messages write escape codes, for the whole program; `colour_mode::automatic` until set.
    ///
    /// A program that knows its output is not a terminal says so here: see the file description.
    /// \ingroup core
    /// \param mode The mode, read by every message as it next uses a colour.
    inline void set_colour_mode(colour_mode mode) noexcept { detail::acout_colour_mode.store(mode, std::memory_order_relaxed); }

    /// \brief The mode `set_colour_mode` last set.
    /// \ingroup core
    /// \return The mode.
    [[nodiscard]] inline colour_mode current_colour_mode() noexcept { return detail::acout_colour_mode.load(std::memory_order_relaxed); }

    // Declared here for acout::operator<<; documented and defined after the engine below.
    template <typename T>
    void print_to(std::ostream& os, const T& value);

    /// \brief A message assembled off to the side and written to its target in one call when it is emitted or destroyed.
    /// \ingroup core
    ///
    /// `acout{} << "x = " << x << '\n';` or `acout{}.println("x = {}", x);` to `std::cout`;
    /// `acout{std::cerr}`, `acout{stream}`, `acout{file}` for another target. See the file
    /// description for the guarantees.
    class acout {
    public:
        /// \brief A message to a stream; `std::cout` by default.
        /// \param target The stream written when the message is emitted; it must outlive the message.
        explicit acout(std::ostream& target = std::cout) noexcept : buffer_(&target, nullptr), stream_(&buffer_) {}

        /// \brief A message to a C stream, written with one `fwrite`.
        /// \param target The file; it must outlive the message.
        explicit acout(std::FILE* target) noexcept : buffer_(nullptr, target), stream_(&buffer_) {}

        acout(const acout&) = delete;
        acout& operator=(const acout&) = delete;

        /// \brief Emits whatever was not emitted. Exceptions are swallowed here; call `emit()` to see them.
        ~acout()
        {
            try {
                emit();
            } catch (...) {   // NOLINT(bugprone-empty-catch): a destructor has nowhere to report a failed write
            }
        }

        /// \brief Stream a value into the message: anything `snicholls::print_to` can write, with the message's own formatting state.
        ///
        /// A streamable value goes through its `operator<<`; a `printable` one through its `print`
        /// member (the `acout` form first); a container, pair, tuple, optional, variant, `result`
        /// or `type_list` through the engine described in the file description, recursively.
        /// \tparam T Anything `snicholls::print_to` accepts; manipulators have their own overloads.
        /// \param value The value.
        /// \return `*this`.
        template <typename T>
            requires(!printable_to_acout<T>)
        acout& operator<<(const T& value)
        {
            print_to(stream_, value);
            return *this;
        }

        /// \brief Stream a value with `print(snicholls::acout&) const`: it writes into this message, format strings and all.
        ///
        /// The member wins over any `operator<<` the type also has, and over a `print(std::ostream&)`.
        /// \tparam T A `snicholls::printable_to_acout` type.
        /// \param value The value.
        /// \return `*this`.
        template <printable_to_acout T>
        acout& operator<<(const T& value)
        {
            value.print(*this);
            return *this;
        }

        /// \brief A stream manipulator: `std::endl` and `std::flush` request a flush after the write.
        /// \param manipulator The manipulator.
        /// \return `*this`.
        acout& operator<<(std::ostream& (*manipulator)(std::ostream&))
        {
            manipulator(stream_);
            return *this;
        }

        /// \brief A format manipulator: `std::hex`, `std::fixed` and the like, applied to this message.
        /// \param manipulator The manipulator.
        /// \return `*this`.
        acout& operator<<(std::ios_base& (*manipulator)(std::ios_base&))
        {
            manipulator(stream_);
            return *this;
        }

        /// \brief Print in this colour from here on: until `colour_reset`, another colour, or the end of the message, which always ends it.
        /// \param c The colour; ignored when this message should not write escape codes (`snicholls::colour_mode`).
        /// \return `*this`.
        acout& operator<<(colour c)
        {
            if (buffer_.colour_wanted())
                start_colour(c);
            return *this;
        }

        /// \brief End the colour this message is in; nothing is written if it is in none.
        /// \return `*this`.
        acout& operator<<(colour_reset_t)
        {
            close_colour();
            return *this;
        }

        /// \brief Print one value in a colour, then return to the colour the message was in.
        /// \tparam T Anything this message can stream.
        /// \param c     The colour; ignored when this message should not write escape codes.
        /// \param value The value.
        /// \return `*this`.
        template <typename T>
        acout& coloured(colour c, const T& value)
        {
            if (!buffer_.colour_wanted())
                return *this << value;
            const std::optional<colour> before = buffer_.current_colour();
            start_colour(c);
            *this << value;
            return_to_colour(before);
            return *this;
        }

        /// \brief Format into the message: `std::format` with a compile-time-checked format string.
        /// \tparam Args The argument types.
        /// \param fmt  The format string.
        /// \param args The arguments.
        /// \return `*this`.
        /// \throws std::format_error for an argument invalid at run time; std::bad_alloc.
        template <typename... Args>
        acout& print(std::format_string<Args...> fmt, Args&&... args)
        {
            std::format_to(std::ostreambuf_iterator<char>(stream_), fmt, std::forward<Args>(args)...);
            return *this;
        }

        /// \brief `print` followed by a newline.
        /// \tparam Args The argument types.
        /// \param fmt  The format string.
        /// \param args The arguments.
        /// \return `*this`.
        /// \throws As `print`.
        template <typename... Args>
        acout& println(std::format_string<Args...> fmt, Args&&... args)
        {
            print(fmt, std::forward<Args>(args)...);
            stream_.put('\n');
            return *this;
        }

        /// \brief Request a flush of the target after the write.
        /// \return `*this`.
        acout& flush() noexcept
        {
            buffer_.request_flush();
            return *this;
        }

        /// \brief The message so far, as it would be written.
        /// \return A copy of the text.
        /// \throws std::bad_alloc.
        [[nodiscard]] std::string message() const { return buffer_.copy(); }

        /// \brief Copy the message so far into another stream, without the lock, and empty it.
        ///
        /// For a message nested inside another: the text joins the enclosing message and is
        /// written when that one is. A flush requested here is dropped.
        /// \param into The stream; a failed write sets its state.
        void copy_into(std::ostream& into)
        {
            close_colour();
            const std::string_view text = buffer_.text();
            if (!text.empty())
                into.write(text.data(), static_cast<std::streamsize>(text.size()));
            buffer_.reset();
        }

        /// \brief Write the message now, in one call, and empty it; nothing is written when it is empty and no flush is due.
        /// \throws std::system_error if the lock cannot be taken; a failed write sets the target's state instead.
        void emit()
        {
            close_colour();   // a colour never outlives its message, so it cannot tint another thread's
            const std::string_view text = buffer_.text();
            const bool flush = buffer_.flush_requested();
            if (text.empty() && !flush)
                return;
            if (std::FILE* const file = buffer_.file()) {   // the C library locks the FILE for the one call
                if (!text.empty())
                    std::fwrite(text.data(), 1, text.size(), file);
                if (flush)
                    std::fflush(file);
            } else {
                std::ostream* const target = buffer_.target();
                const std::lock_guard<std::mutex> lock(detail::acout_write_lock);
                if (!text.empty())
                    target->write(text.data(), static_cast<std::streamsize>(text.size()));
                if (flush)
                    target->flush();
            }
            buffer_.reset();
        }

    private:
        // Unformatted, so an escape code does not use up a std::setw meant for the value after it.
        void write_escape(std::string_view code) { stream_.write(code.data(), static_cast<std::streamsize>(code.size())); }

        void start_colour(colour c)
        {
            write_escape(ansi_start(c).view());
            buffer_.set_active_colour(c);
        }

        void return_to_colour(std::optional<colour> to)
        {
            if (to)
                write_escape(ansi_start(*to).view());
            else
                write_escape(ansi_reset);
            // What the enclosing message is in is not this message's to close
            buffer_.set_active_colour(to == buffer_.inherited_colour() ? std::nullopt : to);
        }

        void close_colour()
        {
            if (buffer_.active_colour())
                return_to_colour(buffer_.inherited_colour());
        }

        detail::message_buffer buffer_;
        std::ostream stream_;
    };

    /// \brief An `acout` to `std::cerr`: `acerr{} << "failed: " << why << '\n';`.
    /// \ingroup core
    struct acerr : acout {
        /// \brief A message to `std::cerr`.
        acerr() noexcept : acout(std::cerr) {}
    };

    /// \brief A value and a colour to print it in; what `snicholls::styled` returns.
    /// \ingroup core
    ///
    /// It refers to the value, so print it in the statement that made it, as with `std::format_args`.
    /// \tparam T The value's type.
    template <typename T>
    struct styled_value {
        colour tint;      ///< The colour.
        const T& value;   ///< The value, not owned.

        /// \brief The `snicholls::printable` member: the value in its colour, within the message.
        /// \param out The message.
        void print(acout& out) const { out.coloured(tint, value); }
    };

    /// \brief One value in a colour, anywhere a value prints: `acout{} << styled(colours::red, "failed") << " at " << n;`.
    ///
    /// The colour ends with the value and the message returns to the one it was in. Inside a container
    /// each element keeps its own colour; onto a plain stream it needs `print_operators`, like any printable.
    /// \ingroup core
    /// \tparam T Anything `snicholls::print_to` accepts.
    /// \param c     The colour.
    /// \param value The value; it must outlive the result.
    /// \return The pair, to print.
    template <typename T>
    [[nodiscard]] constexpr styled_value<T> styled(colour c, const T& value) noexcept
    {
        return {c, value};
    }

    /// \brief A number in its sign's colour: `acout{} << by_sign(pnl);` is green when up, red when down, grey at zero.
    ///
    /// For a number with no type of its own; a type that is always P&L declares `sign_colours` in its `attributes`
    /// instead (`display.hpp`) and prints so everywhere. NaN takes the zero colour.
    /// \ingroup core
    /// \tparam T A `number_like` type.
    /// \param x       The number; it must outlive the result.
    /// \param colours The colours; green, red and grey by default.
    /// \return The number and its colour, to print.
    template <number_like T>
    [[nodiscard]] constexpr styled_value<T> by_sign(const T& x, sign_colours colours = {}) noexcept
    {
        return {colours.of(static_cast<double>(numeric_of(x))).value_or(colours.zero), x};
    }

    // ---- the printing engine: how any value becomes text --------------------------------------------------------

    namespace detail {

        template <typename T>
        concept string_viewable = std::convertible_to<const std::remove_cvref_t<T>&, std::string_view>;

        // std::pair and std::tuple; std::array is a range and prints as one.
        template <typename T>
        concept pair_or_tuple = requires { std::tuple_size<std::remove_cvref_t<T>>::value; } && !std::ranges::range<std::remove_cvref_t<T>>;

        template <typename T>
        concept result_of_void = result_impl::is_result_v<T> && std::is_void_v<typename std::remove_cvref_t<T>::value_type>;

        template <typename T>
        concept result_of_value = result_impl::is_result_v<T> && !std::is_void_v<typename std::remove_cvref_t<T>::value_type>;

        template <typename T>
        concept type_list_value = meta::is_type_list_v<std::remove_cvref_t<T>>;

        // The engine is a first_match chain: the first branch that accepts the value prints it, so a
        // std::string is text and not a range of chars, a std::filesystem::path is its own operator<<
        // and not a range of paths, and a std::map is {k: v} and not a range of pairs. Every branch
        // states its result type, so only the chosen branch's body is ever instantiated. The branches
        // are named so that two chains can share them: the full one behind print_to, and the lent one
        // behind print_operators, which must never ask "is it streamable?" of its own argument, or a
        // type made streamable by that very operator would stream itself forever.
        void print_to_chain(std::ostream& os, const auto& value, bool nested);

        template <typename Range>
        void print_elements(std::ostream& os, const Range& range, char open, char close)
        {
            os << open;
            const char* separator = "";
            for (const auto& element : range) {
                os << separator;
                print_to_chain(os, element, true);
                separator = ", ";
            }
            os << close;
        }

        // The acout form, written into a nested message and copied out: no lock is taken here,
        // because os is either the enclosing message's buffer or a stream the caller owns.
        inline constexpr auto print_acout_form = [](std::ostream& os, const printable_to_acout auto& v, bool) -> void {
            acout message(os);
            v.print(message);
            message.copy_into(os);
        };
        inline constexpr auto print_stream_form = [](std::ostream& os, const printable_to_stream auto& v, bool) -> void {
            v.print(os);
        };
        inline constexpr auto print_null = [](std::ostream& os, std::nullptr_t, bool) -> void { os << "nullptr"; };
        inline constexpr auto print_nullopt = [](std::ostream& os, std::nullopt_t, bool) -> void { os << "none"; };
        inline constexpr auto print_monostate = [](std::ostream& os, std::monostate, bool) -> void { os << "monostate"; };
        inline constexpr auto print_string = [](std::ostream& os, const string_viewable auto& s, bool nested) -> void {
            const std::string_view text{s};
            if (nested)
                os << '"' << text << '"';
            else
                os << text;
        };
        inline constexpr auto print_char = [](std::ostream& os, std::same_as<char> auto c, bool nested) -> void {
            if (nested)
                os << '\'' << c << '\'';
            else
                os << c;
        };
        inline constexpr auto print_byte = [](std::ostream& os, std::same_as<std::byte> auto b, bool) -> void {
            constexpr const char* digits = "0123456789abcdef";
            os << "0x" << digits[std::to_integer<int>(b) >> 4] << digits[std::to_integer<int>(b) & 0xf];
        };
        // A scoped enumeration without its own operator<< prints by name (meta/enums.hpp), or as its integer
        // when the value is not an enumerator; unary + so that a std::uint8_t enumeration is not a character.
        inline constexpr auto print_enum = []<typename E>(std::ostream& os, const E& e, bool) -> void
            requires(scoped_enumeration<E> && !streamable<E>)
        {
            const std::string_view name = enum_name(e);
            if (!name.empty())
                os << name;
            else
                os << +static_cast<std::underlying_type_t<E>>(e);
        };
        // A C array is streamable only by decaying to a pointer; it is a range here.
        inline constexpr auto print_streamable = []<typename T>(std::ostream& os, const T& v, bool) -> void
            requires(streamable<T> && !std::is_array_v<T>)
        {
            os << v;
        };
        inline constexpr auto print_optional = [](std::ostream& os, const specialisation_of<std::optional> auto& o, bool) -> void {
            if (o)
                print_to_chain(os, *o, true);
            else
                os << "none";
        };
        inline constexpr auto print_variant = [](std::ostream& os, const specialisation_of<std::variant> auto& v, bool nested) -> void {
            if (v.valueless_by_exception())
                os << "valueless";
            else
                std::visit([&](const auto& held) { print_to_chain(os, held, nested); }, v);
        };
        inline constexpr auto print_result_void = [](std::ostream& os, const result_of_void auto& r, bool) -> void {
            if (r.is_ok()) {
                os << "ok()";
            } else {
                os << "err(";
                print_to_chain(os, r.unwrap_err(), true);
                os << ')';
            }
        };
        inline constexpr auto print_result_value = [](std::ostream& os, const result_of_value auto& r, bool) -> void {
            os << (r.is_ok() ? "ok(" : "err(");
            if (r.is_ok())
                print_to_chain(os, r.unwrap(), true);
            else
                print_to_chain(os, r.unwrap_err(), true);
            os << ')';
        };
        inline constexpr auto print_map = [](std::ostream& os, const map_like auto& m, bool) -> void {
            os << '{';
            const char* separator = "";
            for (const auto& [key, value] : m) {
                os << separator;
                print_to_chain(os, key, true);
                os << ": ";
                print_to_chain(os, value, true);
                separator = ", ";
            }
            os << '}';
        };
        inline constexpr auto print_set = [](std::ostream& os, const set_like auto& s, bool) -> void {
            print_elements(os, s, '{', '}');
        };
        inline constexpr auto print_range = [](std::ostream& os, const std::ranges::range auto& r, bool) -> void {
            print_elements(os, r, '[', ']');
        };
        inline constexpr auto print_tuple = [](std::ostream& os, const pair_or_tuple auto& t, bool) -> void {
            os << '(';
            [[maybe_unused]] const char* separator = "";   // unused when the tuple is empty: the fold is empty too
            std::apply([&](const auto&... element) { ((os << separator, print_to_chain(os, element, true), separator = ", "), ...); },
                       t);
            os << ')';
        };
        inline constexpr auto print_type_list = []<typename... Ts>(std::ostream& os, const meta::type_list<Ts...>&, bool) -> void {
            os << "type_list<";
            [[maybe_unused]] const char* separator = "";   // unused for type_list<>: the fold is empty
            ((os << separator << type_name<Ts>(), separator = ", "), ...);
            os << '>';
        };

        // Everything, in the order that makes text text and a type's own operator<< its own.
        inline constexpr auto plain_chain =
            first_match{print_acout_form, print_stream_form, print_null,  print_nullopt,     print_monostate,    print_string,
                        print_char,       print_byte,        print_enum,  print_streamable,  print_optional,     print_variant,
                        print_result_void, print_result_value, print_map, print_set,         print_range,        print_tuple,
                        print_type_list};

        // A value's text under its display attributes. A number with any of number_format, scale or units: scaled,
        // formatted (as a double), the scale's suffix, then the units. Else its label. Else as it prints.
        template <typename Set, typename T>
            requires(number_like<T> && (Set::template has<number_format> || Set::template has<scale> || Set::template has<units>))
        void write_displayed(std::ostream& os, const Set& s, const T& v, bool, display::rank<2>)
        {
            const scale sc = attribute_lookup::pick<scale>(s, scale{1.0, ""});
            const number_format nf = attribute_lookup::pick<number_format>(s, number_format{"{}"});
            const std::string_view unit = attribute_lookup::pick<units>(s, units{""}).text;
            const double x = static_cast<double>(numeric_of(v)) / sc.factor;   // number_format's check assumes a double
            std::vformat_to(std::ostreambuf_iterator<char>(os), nf.spec, std::make_format_args(x));
            os << sc.suffix;
            if (!unit.empty())
                os << ' ' << unit;
        }
        template <typename Set, typename T>
            requires(Set::template has<label>)
        void write_displayed(std::ostream& os, const Set& s, const T&, bool, display::rank<1>)
        {
            os << s.template get<label>().text;
        }
        template <typename Set, typename T>
        void write_displayed(std::ostream& os, const Set&, const T& v, bool nested, display::rank<0>)
        {
            plain_chain(os, v, nested);
        }

        // The text, as a printable, so that a message can put it in a colour.
        template <typename Set, typename T>
        struct displayed {
            const Set& set;
            const T& value;
            bool nested;
            void print(std::ostream& os) const { write_displayed(os, set, value, nested, display::rank<2>{}); }
        };
    } // namespace detail

    /// \brief Print a value under these attributes: in its colour (by sign for a number with `sign_colours`), with its number format or label.
    ///
    /// What `acout` does for a value whose type declares display attributes, and what `stats::pivot` does for a cell
    /// with its measure's attributes. Colour is written only where the stream is an `acout` message that wants it.
    /// \ingroup core
    /// \tparam Set An `attribute_set`.
    /// \tparam T   The value's type.
    /// \param os     The stream.
    /// \param s      The attributes.
    /// \param v      The value.
    /// \param nested Whether the value is inside a container (a string is then quoted).
    template <typename Set, typename T>
    void print_displayed(std::ostream& os, const Set& s, const T& v, bool nested = false)
    {
        const detail::displayed<Set, T> text{s, v, nested};
        const std::optional<colour> c = display_colour(s, v);
        if (!c) {
            text.print(os);
            return;
        }
        acout message(os);
        message.coloured(*c, text);
        message.copy_into(os);
    }

    namespace detail {
        inline constexpr auto print_attributed = []<typename T>(std::ostream& os, const T& v, bool nested) -> void
            requires has_display_attributes<T>
        {
            print_displayed(os, attributes_of(v), v, nested);
        };

        inline constexpr auto print_chain = first_match{print_attributed, plain_chain};

        // What a plain stream is lent: the shapes, never "is it streamable?".
        inline constexpr auto lent_chain =
            first_match{print_acout_form, print_stream_form, print_optional, print_variant, print_result_void, print_result_value,
                        print_map,        print_set,         print_range,    print_tuple,   print_type_list};

        template <typename T>
        concept chain_printable = std::is_invocable_v<const decltype(print_chain)&, std::ostream&, const T&, bool>;

        template <typename T>
        concept lent_printable = std::is_invocable_v<const decltype(lent_chain)&, std::ostream&, const T&, bool> && !string_viewable<T>;

        // A string literal: print("{{}}") is a format string, as in std::print, never a value.
        template <typename T>
        concept character_array = std::is_array_v<std::remove_cvref_t<T>> && character<std::remove_all_extents_t<std::remove_cvref_t<T>>>;

        template <typename T>
        concept print_value = chain_printable<std::remove_cvref_t<T>> && !character_array<T>;

        void print_to_chain(std::ostream& os, const auto& value, bool nested)
        {
            static_assert(chain_printable<std::remove_cvref_t<decltype(value)>>,
                          "snicholls: no way to print this type; give it print(snicholls::acout&) const, "
                          "print(std::ostream&) const, or an operator<<");
            print_chain(os, value, nested);
        }

    } // namespace detail

    /// \brief Anything `snicholls::print_to` can write: streamable, `printable`, a container, pair, tuple, optional, variant, `result`, `type_list`.
    /// \ingroup core
    /// \tparam T The type asked about.
    template <typename T>
    concept printable_any = detail::chain_printable<std::remove_cvref_t<T>>;

    /// \brief Write any value as text to a stream: the engine behind `acout`'s `operator<<`.
    /// \ingroup core
    ///
    /// Streamable values go through their `operator<<`; `printable` ones through their `print`;
    /// sequences print as `[a, b]`, sets as `{a, b}`, maps as `{k: v}`, pairs and tuples as
    /// `(a, b)`, a `std::optional` as its value or `none`, a `std::variant` as what it holds, a
    /// `snicholls::result` as `ok(v)` or `err(e)`, a `snicholls::meta::type_list` as
    /// `type_list<int, double>`; all of it recursively, with strings and characters quoted inside
    /// a container. A scoped enumeration prints by name; a `std::byte` as `0x..`.
    /// \tparam T A `snicholls::printable_any` type; anything else is a compile error with a message.
    /// \param os    The stream.
    /// \param value The value.
    /// \throws Whatever the elements' `operator<<` or `print` throw.
    template <typename T>
    void print_to(std::ostream& os, const T& value)
    {
        detail::print_to_chain(os, value, false);
    }

    /// \brief The stream operators core lends to a plain `std::ostream`, to bring into scope: `using snicholls::print_operators::operator<<;`.
    /// \ingroup core
    ///
    /// With the declaration in scope, `printable` types, containers, pairs, tuples, optionals,
    /// variants, `result`s and `type_list`s all stream with `os << value`, as they do into an
    /// `acout` without it, the way magic_enum's `iostream_operators` lends enumerations theirs.
    /// It is an opt-in namespace so that no stream in another library changes meaning by core
    /// being included. Values that have their own `operator<<` are not touched.
    namespace print_operators {

        /// \brief Print a `printable`, container, pair, tuple, optional, variant, `result` or `type_list` to a plain stream.
        ///
        /// Never a string (the standard's operator is the one wanted) and never by asking whether
        /// the value is streamable, which this very operator would make it.
        /// \tparam T One of those shapes; its elements may be anything `snicholls::print_to` accepts.
        /// \param os    The stream.
        /// \param value The value.
        /// \return `os`.
        template <typename T>
            requires detail::lent_printable<std::remove_cvref_t<T>>
        std::ostream& operator<<(std::ostream& os, const T& value)
        {
            detail::lent_chain(os, value, false);
            return os;
        }

    } // namespace print_operators

    /// \brief Print one value to a stream as one message: a `printable`, a container, a tuple, anything `print_to` accepts.
    ///
    /// A string literal is a format string (the overload below), as in `std::print`; a
    /// `std::string` or `std::string_view` is a value and is printed as it is.
    /// \ingroup core
    /// \tparam T A `snicholls::printable_any` type other than a character array.
    /// \param target The stream.
    /// \param value  The value.
    /// \throws Whatever printing the value throws; std::bad_alloc.
    template <detail::print_value T>
    void print(std::ostream& target, const T& value)
    {
        acout message(target);
        message << value;
        message.emit();
    }

    /// \brief Print one value to a C stream with one `fwrite`.
    /// \ingroup core
    /// \tparam T A `snicholls::printable_any` type other than a character array.
    /// \param target The file.
    /// \param value  The value.
    /// \throws Whatever printing the value throws; std::bad_alloc.
    template <detail::print_value T>
    void print(std::FILE* target, const T& value)
    {
        acout message(target);
        message << value;
        message.emit();
    }

    /// \brief Print one value to `std::cout` as one message.
    /// \ingroup core
    /// \tparam T A `snicholls::printable_any` type other than a character array.
    /// \param value The value.
    /// \throws Whatever printing the value throws; std::bad_alloc.
    template <detail::print_value T>
    void print(const T& value)
    {
        print(std::cout, value);
    }

    /// \brief Print one value and a newline to a stream as one message.
    /// \ingroup core
    /// \tparam T A `snicholls::printable_any` type other than a character array.
    /// \param target The stream.
    /// \param value  The value.
    /// \throws Whatever printing the value throws; std::bad_alloc.
    template <detail::print_value T>
    void println(std::ostream& target, const T& value)
    {
        acout message(target);
        message << value << '\n';
        message.emit();
    }

    /// \brief Print one value and a newline to a C stream with one `fwrite`.
    /// \ingroup core
    /// \tparam T A `snicholls::printable_any` type other than a character array.
    /// \param target The file.
    /// \param value  The value.
    /// \throws Whatever printing the value throws; std::bad_alloc.
    template <detail::print_value T>
    void println(std::FILE* target, const T& value)
    {
        acout message(target);
        message << value << '\n';
        message.emit();
    }

    /// \brief Print one value and a newline to `std::cout` as one message: `println(order)`, `println(prices)`.
    /// \ingroup core
    /// \tparam T A `snicholls::printable_any` type other than a character array.
    /// \param value The value.
    /// \throws Whatever printing the value throws; std::bad_alloc.
    template <detail::print_value T>
    void println(const T& value)
    {
        println(std::cout, value);
    }

    /// \brief Format one message in a colour to `std::cout` and write it in one call: `print(colours::red, "{} failed", n)`.
    /// \ingroup core
    /// \tparam Args The argument types.
    /// \param c    The colour; ignored when the message should not write escape codes.
    /// \param fmt  The format string.
    /// \param args The arguments.
    /// \throws As `acout::print`; std::system_error if the lock cannot be taken.
    template <typename... Args>
    void print(colour c, std::format_string<Args...> fmt, Args&&... args)
    {
        acout message;
        message << c;
        message.print(fmt, std::forward<Args>(args)...);
        message.emit();
    }

    /// \brief As `print(colour, ...)` with a newline, written after the colour ends: `println(colours::green, "{} filled", n)`.
    /// \ingroup core
    /// \tparam Args The argument types.
    /// \param c    The colour; ignored when the message should not write escape codes.
    /// \param fmt  The format string.
    /// \param args The arguments.
    /// \throws As `acout::print`; std::system_error if the lock cannot be taken.
    template <typename... Args>
    void println(colour c, std::format_string<Args...> fmt, Args&&... args)
    {
        acout message;
        message << c;
        message.print(fmt, std::forward<Args>(args)...);
        message << colour_reset << '\n';
        message.emit();
    }

    /// \brief Format one message to a stream and write it in one call: `print(std::cout, "{} of {}\n", i, n)`.
    /// \ingroup core
    /// \tparam Args The argument types.
    /// \param target The stream.
    /// \param fmt    The format string, checked at compile time.
    /// \param args   The arguments.
    /// \throws std::format_error for an argument invalid at run time; std::bad_alloc.
    template <typename... Args>
    void print(std::ostream& target, std::format_string<Args...> fmt, Args&&... args)
    {
        acout message(target);
        message.print(fmt, std::forward<Args>(args)...);
        message.emit();
    }

    /// \brief Format one message to a C stream and write it with one `fwrite`: `print(stderr, "{}\n", why)`.
    /// \ingroup core
    /// \tparam Args The argument types.
    /// \param target The file.
    /// \param fmt    The format string, checked at compile time.
    /// \param args   The arguments.
    /// \throws std::format_error for an argument invalid at run time; std::bad_alloc.
    template <typename... Args>
    void print(std::FILE* target, std::format_string<Args...> fmt, Args&&... args)
    {
        acout message(target);
        message.print(fmt, std::forward<Args>(args)...);
        message.emit();
    }

    /// \brief Format one message to `std::cout` and write it in one call.
    /// \ingroup core
    /// \tparam Args The argument types.
    /// \param fmt  The format string, checked at compile time.
    /// \param args The arguments.
    /// \throws As the stream overload.
    template <typename... Args>
    void print(std::format_string<Args...> fmt, Args&&... args)
    {
        print(std::cout, fmt, std::forward<Args>(args)...);
    }

    /// \brief `print` to a stream, with a newline.
    /// \ingroup core
    /// \tparam Args The argument types.
    /// \param target The stream.
    /// \param fmt    The format string.
    /// \param args   The arguments.
    /// \throws As `print`.
    template <typename... Args>
    void println(std::ostream& target, std::format_string<Args...> fmt, Args&&... args)
    {
        acout message(target);
        message.println(fmt, std::forward<Args>(args)...);
        message.emit();
    }

    /// \brief `print` to a C stream, with a newline.
    /// \ingroup core
    /// \tparam Args The argument types.
    /// \param target The file.
    /// \param fmt    The format string.
    /// \param args   The arguments.
    /// \throws As `print`.
    template <typename... Args>
    void println(std::FILE* target, std::format_string<Args...> fmt, Args&&... args)
    {
        acout message(target);
        message.println(fmt, std::forward<Args>(args)...);
        message.emit();
    }

    /// \brief `print` to `std::cout`, with a newline.
    /// \ingroup core
    /// \tparam Args The argument types.
    /// \param fmt  The format string.
    /// \param args The arguments.
    /// \throws As `print`.
    template <typename... Args>
    void println(std::format_string<Args...> fmt, Args&&... args)
    {
        println(std::cout, fmt, std::forward<Args>(args)...);
    }

} // namespace snicholls

#endif /* snicholls_core_acout_hpp */
