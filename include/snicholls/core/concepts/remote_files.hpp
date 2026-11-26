//
//  remote_files.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::remote_files`: anything that lists the files of a remote drop, streams one to the caller with
/// its SHA-256, and optionally moves one.
/// \ingroup core
///
/// **Overview.** A bank, a payment network or a broker leaves a file a day in a drop: Australia Post's settlement file
/// over HTTPS with a client certificate, a clearer's statements over SFTP. A collector lists the drop, fetches what is
/// new into its own raw store, verifies it, and may move the original to an archive folder. `remote_files` is that
/// seam:
/// - `list(prefix)`: every file whose name starts with `prefix`, as `remote_file{name, size, modified}`;
/// - `fetch(name, sink)`: the file's bytes, streamed in order to the caller's `remote_file_sink`, and its
///   `file_digest` (size and SHA-256 of exactly the bytes the sink received), so the collector verifies what it
///   stored against what was sent;
/// - `move(from, to)`: optional, gated by `supports(remote_files_op::move)`.
///
/// The raw store, the manifest of what was collected and the schedule are the caller's. This is the concept half of a
/// pair; the pure virtual half is `remote_files_interface` (`interfaces/remote_files.hpp`), with the adapter
/// `remote_files_adapter<S>` and the handle `remote_files_ref`.
///
/// **The contract** (every implementation; `interfaces/remote_files_conformance.hpp` checks it):
/// - A name is relative to the drop's root: 1 to 255 bytes, segments separated by `/`, no empty, `.` or `..` segment,
///   no leading or trailing `/`, no backslash, NUL or control character (`is_remote_name`). A prefix is empty or the
///   start of a name and may end in `/` (`is_remote_prefix`). Anything else is `capability_errc::invalid_argument`,
///   before any I/O.
/// - `list` returns the matching files sorted by name, each once; an empty drop or no match is an empty list, never
///   an error. `size` is in bytes; `modified` is the drop's own modification time, to the second.
/// - `fetch` of a name that is not there is `capability_errc::not_found`. The sink receives the bytes in order, in
///   chunks of any size, and every byte once. A sink's error stops the fetch and is returned as it is. A fetch that
///   fails may have written a prefix of the file to the sink: the caller discards it.
/// - The digest is of exactly the bytes delivered: `size` their count, `sha256` their SHA-256.
/// - `move` of a name that is not there is `not_found`; onto a name that exists, `conflict`; on an implementation
///   that does not support it, `unsupported`, and `supports(remote_files_op::move)` says so in advance. `list` and
///   `fetch` are always supported.
/// - Nothing logged or returned contains a file's name, its contents, a credential or the server's response: errors
///   follow the redaction rule of `capability_error.hpp`.
///
/// \snippet concepts/remote_files_demo.cpp concept
///
/// **Relationships.** `capability_error.hpp`, `capability_ids.hpp` (`short_text`), `function_ref.hpp`, `hash/sha2.hpp`,
/// `time/utc.hpp`; the twin `interfaces/remote_files.hpp`, the reference `interfaces/remote_files_reference.hpp`, the
/// kit `interfaces/remote_files_conformance.hpp`. Group: \ref core.
///
/// **Implementations.** `memory_remote_files` (`interfaces/remote_files_reference.hpp`, standard library only: name to
/// bytes and a modification time; for tests). Above core: a local folder and SFTP (system OpenSSH, a pinned host key)
/// in ts-moveables; HTTPS with a client certificate in the servers.
///
/// **Guarantees.** The validators and `file_digester` are `constexpr`, `noexcept` and allocation-free; `remote_file`
/// and `file_digest` are trivially copyable. Thread safety is the implementation's.
///
/// **Errors.** `capability_error`, as the contract says.
///
/// **Performance.** Not benchmarked: a drop is I/O bound. `file_digester` is SHA-256 (`hash/sha2.hpp`).
///
/// **Tests and demos.** `tests/concepts/remote_files_test.cpp`, `demos/concepts/remote_files_demo.cpp`.
///
/// **Dependencies.** `<concepts>`, `<cstddef>`, `<cstdint>`, `<span>`, `<string_view>`, `<vector>`,
/// `capability_error.hpp`, `capability_ids.hpp`, `function_ref.hpp`, `hash/digest.hpp`, `hash/sha2.hpp`,
/// `time/utc.hpp`.
///
/// **Known issues.** A name is bytes: two names that are the same text in different Unicode normalisations are two
/// files. Modification times finer than a second are truncated by the implementation.
///
/// **Status.** New in 0.27.0. Experimental until two real implementations pass the kit.

#ifndef snicholls_core_concepts_remote_files_hpp
#define snicholls_core_concepts_remote_files_hpp

#include <snicholls/core/capability_error.hpp>
#include <snicholls/core/capability_ids.hpp>
#include <snicholls/core/function_ref.hpp>
#include <snicholls/core/hash/digest.hpp>
#include <snicholls/core/hash/sha2.hpp>
#include <snicholls/core/time/utc.hpp>

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace snicholls {

    /// \brief An operation of a `remote_files`, for `supports`.
    /// \ingroup core
    enum class remote_files_op : std::uint8_t {
        list = 1,   ///< `list`: always supported
        fetch = 2,  ///< `fetch`: always supported
        move = 3,   ///< `move`: optional
    };

    /// \brief The longest name or prefix accepted, in bytes.
    /// \ingroup core
    inline constexpr std::size_t max_remote_name_length = 255;

    /// \brief One file in a drop, as `list` describes it.
    /// \ingroup core
    struct remote_file {
        snicholls::short_text name;  ///< relative to the drop's root, `/`-separated
        std::uint64_t size{0};       ///< in bytes
        utc_s modified{};            ///< the drop's modification time

        /// \brief Equality, member by member.
        friend constexpr bool operator==(const remote_file&, const remote_file&) noexcept = default;
    };

    /// \brief What a fetch delivered: the byte count and the SHA-256 of exactly those bytes.
    /// \ingroup core
    struct file_digest {
        std::uint64_t size{0};   ///< the bytes delivered
        digest<32> sha256{};     ///< their SHA-256

        /// \brief Equality, member by member.
        friend constexpr bool operator==(const file_digest&, const file_digest&) noexcept = default;
    };

    /// \brief Where `fetch` streams a file: called once per chunk, in order. An error stops the fetch and is returned.
    /// \ingroup core
    using remote_file_sink = function_ref<capability_result<void>(std::span<const std::byte>)>;

    /// \brief The `file_digest` of bytes seen in chunks: what an implementation runs beside its sink.
    /// \ingroup core
    class file_digester {
    public:
        /// \brief Adds the next chunk.
        /// \param chunk The bytes.
        constexpr void update(std::span<const std::byte> chunk) noexcept
        {
            hash_.update(chunk);
            size_ += chunk.size();
        }

        /// \brief The digest of every chunk added; the digester must not be used after.
        /// \return The size and SHA-256.
        [[nodiscard]] constexpr file_digest finish() noexcept { return {size_, hash_.finish()}; }

        /// \brief The digest of one buffer.
        /// \param bytes The bytes.
        /// \return Their size and SHA-256.
        [[nodiscard]] static constexpr file_digest of(std::span<const std::byte> bytes) noexcept
        {
            file_digester d;
            d.update(bytes);
            return d.finish();
        }

    private:
        snicholls::sha256 hash_{};
        std::uint64_t size_{0};
    };

    namespace detail {
        // the rules shared by a name and a prefix; a prefix may end in '/' or with an empty input
        constexpr bool remote_path_ok(std::string_view s, bool prefix) noexcept
        {
            if (s.size() > max_remote_name_length || (!prefix && s.empty()))
                return false;
            if (s.empty())
                return true;
            if (s.front() == '/' || (!prefix && s.back() == '/'))
                return false;
            for (const char c : s) {
                const auto u = static_cast<unsigned char>(c);
                if (u < 0x20 || u == 0x7f || c == '\\')
                    return false;
            }
            std::size_t start = 0;
            while (start <= s.size()) {
                const std::size_t slash = s.find('/', start);
                const std::size_t end = slash == std::string_view::npos ? s.size() : slash;
                const std::string_view segment = s.substr(start, end - start);
                const bool last = slash == std::string_view::npos;
                if (segment.empty() && !(prefix && last))
                    return false;   // "a//b"; a prefix's trailing '/' leaves an empty last segment
                if (segment == "..")
                    return false;
                if (segment == "." && !(prefix && last))
                    return false;   // a prefix's last segment is the start of a name: ".git" begins with "."
                if (last)
                    break;
                start = slash + 1;
            }
            return true;
        }
    } // namespace detail

    /// \brief Whether `s` is a name in a drop: 1 to 255 bytes, `/`-separated segments, none empty, `.` or `..`, no
    /// leading or trailing `/`, no backslash, NUL or control character.
    /// \ingroup core
    /// \param s The text.
    /// \return `true` when it is.
    [[nodiscard]] constexpr bool is_remote_name(std::string_view s) noexcept { return detail::remote_path_ok(s, false); }

    /// \brief Whether `s` is a prefix for `list`: empty, or the start of a name, which may end in `/`.
    /// \ingroup core
    /// \param s The text.
    /// \return `true` when it is.
    [[nodiscard]] constexpr bool is_remote_prefix(std::string_view s) noexcept { return detail::remote_path_ok(s, true); }

    /// \brief The check every implementation applies to a name before any I/O.
    /// \ingroup core
    /// \param name The name.
    /// \return Success, or `invalid_argument` with a fixed message (never the name).
    [[nodiscard]] constexpr capability_result<void> validate_remote_name(std::string_view name) noexcept
    {
        if (!is_remote_name(name))
            return fail(capability_errc::invalid_argument, {}, "not a file name in the drop");
        return ok();
    }

    /// \brief The check every implementation applies to a prefix before any I/O.
    /// \ingroup core
    /// \param prefix The prefix.
    /// \return Success, or `invalid_argument` with a fixed message (never the prefix).
    [[nodiscard]] constexpr capability_result<void> validate_remote_prefix(std::string_view prefix) noexcept
    {
        if (!is_remote_prefix(prefix))
            return fail(capability_errc::invalid_argument, {}, "not a prefix of a file name in the drop");
        return ok();
    }

    /// \brief A remote drop: `r.list(prefix)`, `r.fetch(name, sink)`, `r.move(from, to)` and `cr.supports(op)`.
    /// \ingroup core
    template <typename R>
    concept remote_files = requires(R& r, const R& cr, std::string_view s, remote_file_sink sink, remote_files_op op) {
        { r.list(s) } -> std::same_as<capability_result<std::vector<remote_file>>>;
        { r.fetch(s, sink) } -> std::same_as<capability_result<file_digest>>;
        { r.move(s, s) } -> std::same_as<capability_result<void>>;
        { cr.supports(op) } -> std::same_as<bool>;
    };

} // namespace snicholls

#endif // snicholls_core_concepts_remote_files_hpp
