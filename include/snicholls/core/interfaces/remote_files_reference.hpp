//
//  remote_files_reference.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::memory_remote_files`: the standard-library reference `remote_files`, a drop held in memory as
/// name to bytes and a modification time.
/// \ingroup core
///
/// **Overview.** What a collector's tests list and fetch from: it applies the contract exactly (name validation,
/// sorted listing, streaming in chunks with the digest, `move` when switched on) over files the test `put`s, so no
/// server, credential or file system is needed; core has no I/O. `fail_next(code, response)` makes the next
/// operation that would be attempted fail as though the server had answered `response`, to test a collector's retry
/// path; the error keeps the response's first word as the provider code when it is one (`provider_error`) and nothing
/// else of it. A small `chunk_size` makes every fetch arrive in many chunks, as it would from a network.
///
/// \snippet interfaces/remote_files_reference_demo.cpp reference
///
/// **Relationships.** Models `remote_files` (`concepts/remote_files.hpp`); passes
/// `interfaces/remote_files_conformance.hpp` directly and through `interfaces/remote_files.hpp`. Group: \ref core.
///
/// **Implementations / variants.** One; `move` can be switched off, to stand for a read-only drop.
///
/// **Guarantees.** Not thread-safe (one per test). Strong exception safety for `put` and `move`: a throwing
/// allocation leaves the drop unchanged. Errors follow the redaction rule: fixed messages only, never a name.
///
/// **Errors.** As the contract, and whatever `fail_next` armed. `put` refuses an invalid name with
/// `invalid_argument` and an existing one with `conflict`.
///
/// **Performance.** Not benchmarked: a test double.
///
/// **Tests and demos.** `tests/interfaces/remote_files_reference_test.cpp`,
/// `demos/interfaces/remote_files_reference_demo.cpp`.
///
/// **Dependencies.** `<algorithm>`, `<cstddef>`, `<functional>`, `<map>`, `<optional>`, `<span>`, `<string>`,
/// `<string_view>`, `<utility>`, `<vector>`, `interfaces/remote_files.hpp`.
///
/// **Known issues.** None.
///
/// **Status.** New in 0.25.0.

#ifndef snicholls_core_interfaces_remote_files_reference_hpp
#define snicholls_core_interfaces_remote_files_reference_hpp

#include <snicholls/core/interfaces/remote_files.hpp>

#include <algorithm>
#include <cstddef>
#include <functional>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace snicholls {

    /// \brief The reference drop, in memory.
    /// \ingroup core
    class memory_remote_files {
    public:
        /// \brief A drop that supports `move` and fetches in 4 KiB chunks.
        memory_remote_files() = default;

        /// \brief A drop with `move` on or off and a chunk size of its own.
        /// \param supports_move Whether `move` is supported.
        /// \param chunk_size    The largest chunk a fetch hands the sink; at least 1.
        explicit memory_remote_files(bool supports_move, std::size_t chunk_size = 4096) noexcept
            : movable_(supports_move), chunk_(std::max<std::size_t>(chunk_size, 1))
        {
        }

        /// \brief Adds a file to the drop, as the counterparty would.
        /// \param name     The name; see `is_remote_name`.
        /// \param bytes    The contents.
        /// \param modified The modification time.
        /// \return Success, `invalid_argument` for a bad name, or `conflict` when the name is taken.
        capability_result<void> put(std::string_view name, std::span<const std::byte> bytes, utc_s modified)
        {
            if (auto valid = validate_remote_name(name); !valid)
                return valid;
            if (files_.find(name) != files_.end())
                return fail(capability_errc::conflict, {}, "the name is taken");
            files_.emplace(std::string(name), entry{std::vector<std::byte>(bytes.begin(), bytes.end()), modified});
            return ok();
        }

        /// \brief The files whose names start with `prefix`, sorted by name.
        /// \param prefix The prefix; empty for every file.
        /// \return The files, or the error.
        capability_result<std::vector<remote_file>> list(std::string_view prefix)
        {
            if (auto valid = validate_remote_prefix(prefix); !valid)
                return err(valid.unwrap_err());
            if (auto injected = take_injected())
                return err(*injected);
            std::vector<remote_file> out;
            for (auto it = files_.lower_bound(prefix); it != files_.end() && it->first.starts_with(prefix); ++it)
                out.push_back({*short_text::make(it->first), it->second.bytes.size(), it->second.modified});
            return ok(std::move(out));
        }

        /// \brief Streams one file to `sink` in chunks of at most `chunk_size` bytes.
        /// \param name The file.
        /// \param sink Where the bytes go.
        /// \return The size and SHA-256 of the bytes delivered, `not_found`, or the sink's error.
        capability_result<file_digest> fetch(std::string_view name, remote_file_sink sink)
        {
            if (auto valid = validate_remote_name(name); !valid)
                return err(valid.unwrap_err());
            if (auto injected = take_injected())
                return err(*injected);
            const auto it = files_.find(name);
            if (it == files_.end())
                return fail(capability_errc::not_found, {}, "no such file in the drop");
            const std::span<const std::byte> all{it->second.bytes};
            file_digester digest;
            for (std::size_t at = 0; at < all.size(); at += chunk_) {
                const auto chunk = all.subspan(at, std::min(chunk_, all.size() - at));
                if (auto taken = sink(chunk); !taken)
                    return err(taken.unwrap_err());
                digest.update(chunk);
            }
            return ok(digest.finish());
        }

        /// \brief Renames one file within the drop.
        /// \param from The file.
        /// \param to   Its new name, which must not exist.
        /// \return Success, `unsupported`, `invalid_argument`, `not_found` or `conflict`.
        capability_result<void> move(std::string_view from, std::string_view to)
        {
            if (!movable_)
                return fail(capability_errc::unsupported);
            if (auto valid = validate_remote_name(from); !valid)
                return valid;
            if (auto valid = validate_remote_name(to); !valid)
                return valid;
            if (auto injected = take_injected())
                return err(*injected);
            const auto it = files_.find(from);
            if (it == files_.end())
                return fail(capability_errc::not_found, {}, "no such file in the drop");
            if (files_.find(to) != files_.end())
                return fail(capability_errc::conflict, {}, "the new name is taken");
            std::string new_name(to);   // allocate before anything changes
            auto node = files_.extract(it);
            node.key() = std::move(new_name);
            files_.insert(std::move(node));
            return ok();
        }

        /// \brief Whether an operation is supported.
        /// \param op The operation.
        /// \return `true` for `list` and `fetch`; for `move`, as constructed.
        [[nodiscard]] bool supports(remote_files_op op) const noexcept { return op != remote_files_op::move || movable_; }

        /// \brief A file's contents, for a test to inspect.
        /// \param name The file.
        /// \return The bytes, or null when there is no such file.
        [[nodiscard]] const std::vector<std::byte>* contents(std::string_view name) const noexcept
        {
            const auto it = files_.find(name);
            return it == files_.end() ? nullptr : &it->second.bytes;
        }

        /// \brief How many files the drop holds.
        /// \return The count.
        [[nodiscard]] std::size_t size() const noexcept { return files_.size(); }

        /// \brief Makes the next operation that passes validation fail (once) as though the server answered `response`.
        /// \param code     The error's code.
        /// \param response The server's response: its first word becomes the provider code when it is one, and nothing
        ///                 else of it is kept.
        void fail_next(capability_errc code, std::string_view response = {}) noexcept
        {
            injected_ = provider_error(code, response.substr(0, response.find(' ')));
        }

    private:
        struct entry {
            std::vector<std::byte> bytes;
            utc_s modified;
        };

        std::optional<capability_error> take_injected() noexcept
        {
            auto e = injected_;
            injected_.reset();
            return e;
        }

        bool movable_{true};
        std::size_t chunk_{4096};
        std::map<std::string, entry, std::less<>> files_;
        std::optional<capability_error> injected_;
    };

    static_assert(remote_files<memory_remote_files>, "snicholls::memory_remote_files must model snicholls::remote_files");

} // namespace snicholls

#endif // snicholls_core_interfaces_remote_files_reference_hpp
