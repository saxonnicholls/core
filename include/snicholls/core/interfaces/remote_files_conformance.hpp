//
//  remote_files_conformance.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::run_remote_files_conformance`: the conformance kit of the `remote_files` contract.
/// \ingroup core
///
/// **Overview.** Drives any `remote_files`, in either form, through the contract of `concepts/remote_files.hpp` and
/// reports which check failed (`interfaces/capability_conformance.hpp`). The probe is a default-constructible type
/// with
/// - `auto& files()`: the implementation, a model of the concept (a drop, a `remote_files_interface&`, a
///   `remote_files_ref`), over an empty drop;
/// - `void seed(std::string_view name, std::span<const std::byte> bytes, utc_s modified)`: puts a file in the drop,
///   as the counterparty would (for SFTP, an upload to the test server);
/// - `void provider_fails(capability_errc code, std::string_view response)`: the next operation fails with `code`,
///   as though the server had answered `response`.
///
/// A fresh probe is made for every check. The checks: `list returns the matching files sorted by name`, `fetch
/// streams every byte with its digest`, `a missing file is not_found without echoing its name`, `an invalid name or
/// prefix is refused`, `a sink's error stops the fetch and is returned`, `move renames a file, or is unsupported`, `a
/// provider failure never leaks its response`.
///
/// \snippet interfaces/remote_files_conformance_demo.cpp kit
///
/// **Relationships.** `concepts/remote_files.hpp`, `interfaces/remote_files.hpp`,
/// `interfaces/capability_conformance.hpp`; run on `memory_remote_files` in every form by
/// `tests/interfaces/remote_files_test.cpp`. Group: \ref core.
///
/// **Implementations / variants.** One kit for every form.
///
/// **Guarantees.** As `interfaces/capability_conformance.hpp`: no exception escapes a check. Modification times are
/// compared to the second, the contract's resolution.
///
/// **Errors.** Failures are data.
///
/// **Performance.** Not benchmarked: a test kit. The largest file it seeds is 10,000 bytes.
///
/// **Tests and demos.** `tests/interfaces/remote_files_conformance_test.cpp` (the kit catches broken drops),
/// `demos/interfaces/remote_files_conformance_demo.cpp`.
///
/// **Dependencies.** `<chrono>`, `<cstddef>`, `<cstdint>`, `<initializer_list>`, `<span>`, `<string>`, `<string_view>`, `<type_traits>`,
/// `<utility>`, `<vector>`, `concepts/remote_files.hpp`, `interfaces/capability_conformance.hpp`.
///
/// **Known issues.** The kit cannot see what an implementation logs; it checks only what is returned.
///
/// **Status.** New in 0.25.0.

#ifndef snicholls_core_interfaces_remote_files_conformance_hpp
#define snicholls_core_interfaces_remote_files_conformance_hpp

#include <snicholls/core/concepts/remote_files.hpp>
#include <snicholls/core/interfaces/capability_conformance.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace snicholls {

    namespace detail {
        template <typename Probe>
        concept remote_files_probe = std::is_default_constructible_v<Probe> &&
            remote_files<std::remove_reference_t<decltype(std::declval<Probe&>().files())>> &&
            requires(Probe& p, std::string_view n, std::span<const std::byte> b, utc_s t, capability_errc c) {
                p.seed(n, b, t);
                p.provider_fails(c, n);
            };

        // n bytes of a fixed pattern, different for each seed
        inline std::vector<std::byte> kit_bytes(std::size_t n, unsigned seed)
        {
            std::vector<std::byte> out(n);
            std::uint32_t x = 0x9e3779b9u ^ seed;
            for (auto& b : out) {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                b = static_cast<std::byte>(x);
            }
            return out;
        }

        inline const utc_s kit_mtime = utc_s::from_unix_seconds(1'800'000'000);

        // a name that carries the redaction marker, to see that errors never echo a name
        inline constexpr std::string_view kit_secret_name = "in/REDACTION-MARKER-c0re.csv";
    } // namespace detail

    /// \brief Runs every check of the `remote_files` contract, each on a fresh `Probe`.
    /// \ingroup core
    /// \tparam Probe The probe (see the file's overview).
    /// \return The report.
    template <typename Probe>
        requires detail::remote_files_probe<Probe>
    [[nodiscard]] capability_conformance_report run_remote_files_conformance()
    {
        capability_conformance_report report;
        using detail::kit_bytes;
        using detail::kit_mtime;

        // collects what a fetch delivers
        struct collected {
            std::vector<std::byte> bytes;
            std::size_t chunks = 0;
            capability_result<void> operator()(std::span<const std::byte> chunk)
            {
                bytes.insert(bytes.end(), chunk.begin(), chunk.end());
                ++chunks;
                return ok();
            }
        };

        detail::capability_check<Probe>(report, "list returns the matching files sorted by name", [](Probe& p, const auto& expect) {
            const auto b = kit_bytes(10, 1), a = kit_bytes(20, 2), c = kit_bytes(30, 3), top = kit_bytes(0, 4);
            p.seed("in/b.csv", b, kit_mtime);
            p.seed("in/a.csv", a, kit_mtime + std::chrono::seconds{60});
            p.seed("out/c.csv", c, kit_mtime);
            p.seed("inbox.txt", top, kit_mtime);
            auto& f = p.files();
            expect(f.supports(remote_files_op::list) && f.supports(remote_files_op::fetch), "list and fetch are supported");
            const auto in = f.list("in/");
            expect(in.is_ok(), "a prefix lists");
            if (in.is_ok()) {
                expect(in->size() == 2, "only the files under the prefix");
                if (in->size() == 2) {
                    expect((*in)[0].name == std::string_view{"in/a.csv"} && (*in)[1].name == std::string_view{"in/b.csv"},
                           "sorted by name, each named in full");
                    expect((*in)[0].size == 20 && (*in)[1].size == 10, "each with its size");
                    expect((*in)[0].modified == kit_mtime + std::chrono::seconds{60} && (*in)[1].modified == kit_mtime,
                           "each with its modification time");
                }
            }
            const auto all = f.list("");
            expect(all.is_ok() && all->size() == 4, "an empty prefix lists every file");
            if (all.is_ok() && all->size() == 4)
                expect((*all)[0].name == std::string_view{"in/a.csv"} && (*all)[2].name == std::string_view{"inbox.txt"} &&
                           (*all)[3].name == std::string_view{"out/c.csv"},
                       "every file sorted by name");
            const auto partial = f.list("in");
            expect(partial.is_ok() && partial->size() == 3, "a prefix need not end at a '/'");
            const auto none = f.list("archive/");
            expect(none.is_ok() && none->empty(), "no match is an empty list, not an error");
        });

        detail::capability_check<Probe>(report, "fetch streams every byte with its digest", [](Probe& p, const auto& expect) {
            const auto big = kit_bytes(10'000, 5);
            const std::vector<std::byte> empty;
            p.seed("in/settlement.csv", big, kit_mtime);
            p.seed("in/empty.csv", empty, kit_mtime);
            auto& f = p.files();
            collected got;
            const auto d = f.fetch("in/settlement.csv", got);
            expect(d.is_ok(), "an existing file is fetched");
            expect(got.bytes == big, "the sink receives every byte, in order, once");
            if (d.is_ok()) {
                expect(d->size == big.size(), "the digest's size is the bytes delivered");
                expect(d->sha256 == sha256::hash(std::span<const std::byte>{big}), "the digest is their SHA-256");
            }
            collected nothing;
            const auto e = f.fetch("in/empty.csv", nothing);
            expect(e.is_ok() && nothing.bytes.empty(), "an empty file is fetched");
            if (e.is_ok())
                expect(*e == file_digester::of({}), "with the digest of no bytes");
        });

        detail::capability_check<Probe>(report, "a missing file is not_found without echoing its name", [](Probe& p, const auto& expect) {
            auto& f = p.files();
            collected got;
            const auto r = f.fetch(detail::kit_secret_name, got);
            expect(r.is_err() && r.unwrap_err().code == capability_errc::not_found, "a missing file is not_found");
            expect(got.bytes.empty(), "nothing reaches the sink");
            if (r.is_err())
                expect(!capability_error_leaks(r.unwrap_err(), capability_redaction_marker), "the error does not echo the name");
        });

        detail::capability_check<Probe>(report, "an invalid name or prefix is refused", [](Probe& p, const auto& expect) {
            const auto bytes = kit_bytes(8, 6);
            p.seed("in/a.csv", bytes, kit_mtime);
            auto& f = p.files();
            const std::string too_long(max_remote_name_length + 1, 'a');
            const std::string with_nul("in/a\0.csv", 9);
            for (const std::string_view bad : {std::string_view{"../etc/passwd"}, std::string_view{"/in/a.csv"},
                                               std::string_view{"in//a.csv"}, std::string_view{"in/./a.csv"},
                                               std::string_view{"in\\a.csv"}, std::string_view{"in/"}, std::string_view{""},
                                               std::string_view{too_long}, std::string_view{with_nul},
                                               std::string_view{"in/REDACTION-MARKER-c0re/../a"}}) {
                collected got;
                const auto r = f.fetch(bad, got);
                expect(r.is_err() && r.unwrap_err().code == capability_errc::invalid_argument, "a bad name is invalid_argument");
                expect(got.bytes.empty(), "nothing reaches the sink");
                if (r.is_err())
                    expect(!capability_error_leaks(r.unwrap_err(), capability_redaction_marker), "the error does not echo the name");
                if (f.supports(remote_files_op::move)) {
                    const auto m = f.move("in/a.csv", bad);
                    expect(m.is_err() && m.unwrap_err().code == capability_errc::invalid_argument, "a bad new name is invalid_argument");
                }
            }
            for (const std::string_view bad : {std::string_view{"../"}, std::string_view{"/in"}, std::string_view{"in//"},
                                               std::string_view{too_long}}) {
                const auto r = f.list(bad);
                expect(r.is_err() && r.unwrap_err().code == capability_errc::invalid_argument, "a bad prefix is invalid_argument");
            }
            const auto still = f.list("");
            expect(still.is_ok() && still->size() == 1, "the drop is unchanged");
        });

        detail::capability_check<Probe>(report, "a sink's error stops the fetch and is returned", [](Probe& p, const auto& expect) {
            const auto big = kit_bytes(10'000, 7);
            p.seed("in/settlement.csv", big, kit_mtime);
            auto& f = p.files();
            std::size_t calls = 0;
            const auto refusing = [&calls](std::span<const std::byte>) -> capability_result<void> {
                ++calls;
                return fail(capability_errc::internal, {}, "the raw store is full");
            };
            const auto r = f.fetch("in/settlement.csv", refusing);
            expect(r.is_err(), "the fetch fails");
            if (r.is_err())
                expect(r.unwrap_err() == capability_error{capability_errc::internal, {}, "the raw store is full"},
                       "with the sink's error, unchanged");
            expect(calls == 1, "and stops at the chunk the sink refused");
            collected again;
            const auto retry = f.fetch("in/settlement.csv", again);
            expect(retry.is_ok() && again.bytes == big, "the file is still there to fetch again");
        });

        detail::capability_check<Probe>(report, "move renames a file, or is unsupported", [](Probe& p, const auto& expect) {
            const auto a = kit_bytes(64, 8), b = kit_bytes(32, 9);
            p.seed("in/a.csv", a, kit_mtime);
            p.seed("in/b.csv", b, kit_mtime);
            auto& f = p.files();
            if (!f.supports(remote_files_op::move)) {
                const auto r = f.move("in/a.csv", "archive/a.csv");
                expect(r.is_err() && r.unwrap_err().code == capability_errc::unsupported, "an unsupported move is unsupported");
                return;
            }
            const auto moved = f.move("in/a.csv", "archive/a.csv");
            expect(moved.is_ok(), "a file moves");
            const auto in = f.list("in/");
            const auto archive = f.list("archive/");
            expect(in.is_ok() && in->size() == 1, "it is gone from its old name");
            expect(archive.is_ok() && archive->size() == 1 && (*archive)[0].size == a.size(), "and listed under its new one");
            collected got;
            const auto d = f.fetch("archive/a.csv", got);
            expect(d.is_ok() && got.bytes == a, "with the same contents");
            const auto missing = f.move("in/a.csv", "archive/again.csv");
            expect(missing.is_err() && missing.unwrap_err().code == capability_errc::not_found, "moving a missing file is not_found");
            const auto onto = f.move("in/b.csv", "archive/a.csv");
            expect(onto.is_err() && onto.unwrap_err().code == capability_errc::conflict, "moving onto a file is a conflict");
            collected kept;
            expect(f.fetch("in/b.csv", kept).is_ok() && kept.bytes == b, "and leaves both files as they were");
        });

        detail::capability_check<Probe>(report, "a provider failure never leaks its response", [](Probe& p, const auto& expect) {
            p.provider_fails(capability_errc::unavailable, capability_provider_response);
            auto& f = p.files();
            const auto r = f.list("");
            expect(r.is_err() && r.unwrap_err().code == capability_errc::unavailable, "the server's failure is returned");
            if (r.is_err())
                expect(!capability_error_leaks(r.unwrap_err(), capability_redaction_marker),
                       "neither provider_code nor message carries the server's response");
        });

        return report;
    }

} // namespace snicholls

#endif // snicholls_core_interfaces_remote_files_conformance_hpp
