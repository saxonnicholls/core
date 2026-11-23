//
//  key_value_store_conformance.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  The cases of crtp's key_value_store_contract.hpp, as plain functions over the concept instead of a GoogleTest
//  typed suite, so any repository runs them under any framework. The thread-contention and reopen cases stay with
//  the implementations that have threads and persistence; added here: insert in batches and the append-once
//  ledger, chunked scans, sequence, snapshots, the digest's golden vector, and that no exception crosses.
//

/// \file
/// \brief The conformance kit of the key-value contract: plain functions that drive any store and report which check failed.
/// \ingroup core
///
/// **Overview.** One kit for every implementation of `key_value_store` (`concepts/key_value_store.hpp`), in
/// either form: a model of the concept, or a `key_value_store_interface` (`interfaces/key_value_store.hpp`).
/// `run_kv_conformance(make)` calls `make()` once per check for a fresh, empty store (a store by value, or a
/// pointer-like handle such as `std::unique_ptr<key_value_store_interface>`), runs the check, and records every
/// expectation that failed with the check's name. No test framework is included: a test asserts
/// `report.ok()` and prints `report.failures`.
///
/// \snippet interfaces/key_value_store_conformance_demo.cpp kit
///
/// The checks, by name: `starts_empty`, `put_then_read_round_trips`, `missing_key_is_not_an_error`,
/// `put_overwrites`, `empty_key_and_value_exist`, `embedded_nul_round_trips`, `arbitrary_bytes_round_trip`,
/// `insert_only_when_absent`, `ledger_entry_is_written_once`, `erase_reports_whether_the_key_existed`,
/// `scan_is_in_unsigned_bytewise_order`, `ranges_are_half_open`, `scan_prefix_is_a_true_prefix`,
/// `prefix_ending_in_ff_is_bounded`, `scan_stops_when_the_visitor_says_so`, `chunked_scan_equals_one_scan`,
/// `erase_range_removes_exactly_the_range`, `apply_runs_writes_in_order`, `batch_insert_conflict_changes_nothing`,
/// `visitor_exceptions_propagate_and_the_store_stays_usable`, `many_keys_scan_in_order`, `sequence_rules`,
/// `snapshot_is_isolated`, `digest_v1_matches_its_golden_vector`, `no_exception_crosses_the_interface`.
///
/// Two checks need a way to make the store fail, which only its owner has; they take the store and a callable
/// that arms the failure: `kv_check_failure_changes_nothing` (a write that fails, alone or mid-batch, is a typed
/// error and changes nothing) and `kv_check_no_exception_escapes` (every operation, run under whatever failure the
/// caller arranged, returns rather than throws). `run_kv_health_conformance` and `run_kv_backup_conformance` cover
/// the other capabilities.
///
/// **Relationships.** Drives `key_value_store`, `kv_health` and `kv_backup` implementations; run on
/// `reference_key_value_store` in both forms by `tests/interfaces/key_value_store_test.cpp`. Group: \ref core.
///
/// **Implementations / variants.** One kit for both forms; `kv_conformance_options::binary_safe = false` skips the
/// two checks with bytes that a text-only backend cannot store.
///
/// **Guarantees.** A check never lets an exception out: one that escapes a check is recorded as a failure of
/// `no exception crosses the interface`. The kit allocates (it builds expected values); the contract does not.
/// Not thread-safe: no check uses threads (core has none); contention and durability are checked by the
/// implementations' own suites.
///
/// **Errors.** Failures are data (`kv_conformance_failure`), never thrown.
///
/// **Examples.** See the overview.
///
/// **Performance.** Not benchmarked: a test kit.
///
/// **Tests and demos.** `tests/interfaces/key_value_store_conformance_test.cpp` (the kit catches broken stores),
/// `demos/interfaces/key_value_store_conformance_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<cstddef>`, `<cstdint>`, `<optional>`, `<stdexcept>`, `<string>`,
/// `<string_view>`, `<utility>`, `<vector>`, and the three capability concepts.
///
/// **Known issues.** The kit checks what one thread can observe; atomicity under contention is the
/// implementation's to prove.
///
/// **Status.** New in 0.22.0. Experimental.

#ifndef snicholls_core_interfaces_key_value_store_conformance_hpp
#define snicholls_core_interfaces_key_value_store_conformance_hpp

#include <snicholls/core/concepts/key_value_store.hpp>
#include <snicholls/core/concepts/kv_backup.hpp>
#include <snicholls/core/concepts/kv_health.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace snicholls {

    /// \brief One expectation that failed: the check's name and what was expected (both static strings).
    /// \ingroup core
    struct kv_conformance_failure {
        std::string_view check;        ///< the check, e.g. `"insert_only_when_absent"`
        std::string_view expectation;  ///< what did not hold
    };

    /// \brief What a run of the kit found.
    /// \ingroup core
    struct kv_conformance_report {
        std::size_t checks = 0;                        ///< how many checks ran
        std::vector<kv_conformance_failure> failures;  ///< every expectation that failed, in order

        /// \brief Whether every check passed.
        /// \return `failures.empty()`.
        [[nodiscard]] bool ok() const noexcept { return failures.empty(); }

        /// \brief Whether the check `name` recorded a failure.
        /// \param name the check's name.
        /// \return `true` if it failed.
        [[nodiscard]] bool failed(std::string_view name) const noexcept
        {
            for (const auto& f : failures)
                if (f.check == name)
                    return true;
            return false;
        }
    };

    /// \brief What the store under test cannot do, so the kit skips the checks that need it.
    /// \ingroup core
    struct kv_conformance_options {
        bool binary_safe = true;  ///< accepts any bytes; false for a text-only backend (skips two checks)
    };

    /// \brief The golden `kv_digest_v1` of the kit's four pairs ("" -> "e", "a" -> "", "ab" -> "x", "b" -> "y").
    /// \ingroup core
    inline constexpr std::string_view kv_digest_v1_kit_vector = "9b8b4b671ab3f21ef6c962d25731eecc82be4780249f523d5946c5c576f6fe23";

    namespace detail {
        // Records the failures of one check.
        struct kv_expect {
            std::string_view check;
            kv_conformance_report& report;

            void operator()(bool holds, std::string_view expectation) const
            {
                if (!holds)
                    report.failures.push_back({check, expectation});
            }
        };

        inline kv_bytes kv_b(std::string_view s) noexcept { return kv_bytes_of(s); }

        template <typename S>
        std::vector<std::string> kv_keys(const S& s, const kv_range& r = kv_range::all())
        {
            std::vector<std::string> out;
            static_cast<void>(kv_scan(s, r, [&](kv_bytes k, kv_bytes) { out.emplace_back(reinterpret_cast<const char*>(k.data()), k.size()); }));
            return out;
        }

        template <typename R>
        bool kv_is(const R& r, const typename R::value_type& v)
        {
            return r.is_ok() && *r == v;
        }

        template <typename S>
        std::uint64_t kv_seq(const S& s)
        {
            const auto q = s.sequence();
            return q.is_ok() ? *q : ~std::uint64_t{0};
        }

        template <typename S>
        std::optional<std::string> kv_value(const S& s, std::string_view key)
        {
            const auto r = kv_get(s, key);
            return r.is_ok() ? *r : std::nullopt;
        }

        // Runs one check on a fresh store from make(), recording an escaping exception as a failure.
        template <typename Make, typename Check>
        void kv_run(kv_conformance_report& report, std::string_view name, Make& make, Check&& check)
        {
            ++report.checks;
            const kv_expect expect{name, report};
#if SNICHOLLS_RESULT_HAS_EXCEPTIONS
            try {
                decltype(auto) store = make();
                check(kv_target(store), expect);
            } catch (...) {
                expect(false, "no exception crosses the interface");
            }
#else
            decltype(auto) store = make();
            check(kv_target(store), expect);
#endif
        }
    }

    /// \brief Runs every generic check of the store contract, each on a fresh store from `make()`.
    /// \ingroup core
    /// \tparam Make callable with no arguments, returning a `key_value_store` or a pointer-like handle to one.
    /// \param make makes a fresh, empty store.
    /// \param options what the store cannot do.
    /// \return the report.
    template <typename Make>
    kv_conformance_report run_kv_conformance(Make&& make, kv_conformance_options options = {})
    {
        using detail::kv_b;
        using detail::kv_is;
        using detail::kv_keys;
        using detail::kv_seq;
        using detail::kv_value;
        using strings = std::vector<std::string>;
        kv_conformance_report report;
        const auto run = [&](std::string_view name, auto&& check) { detail::kv_run(report, name, make, check); };

        run("starts_empty", [](auto& s, const auto& expect) {
            expect(kv_is(kv_count(s), 0), "a new store has no keys");
            expect(kv_is(kv_contains(s, "missing"), false), "a new store contains nothing");
            expect(kv_is(kv_get(s, "missing"), std::nullopt), "a missing key reads as std::nullopt");
            expect(s.sequence().is_ok(), "sequence() answers");
        });

        run("put_then_read_round_trips", [](auto& s, const auto& expect) {
            expect(kv_put(s, "k", "value").is_ok(), "put succeeds");
            expect(kv_value(s, "k") == "value", "kv_get returns the value");
            std::string buffer = "previous";
            expect(kv_is(kv_get(s, "k", buffer), true) && buffer == "value", "kv_get into a buffer replaces it");
            std::string seen;
            expect(kv_is(kv_read(s, "k", [&](kv_bytes v) { seen.assign(reinterpret_cast<const char*>(v.data()), v.size()); }), true) &&
                       seen == "value",
                   "read hands the visitor the value");
            expect(kv_is(kv_contains(s, "k"), true), "contains finds the key");
        });

        run("missing_key_is_not_an_error", [](auto& s, const auto& expect) {
            std::string buffer = "untouched";
            expect(kv_is(kv_get(s, "nope", buffer), false) && buffer == "untouched", "a missing key leaves the buffer alone");
            bool called = false;
            expect(kv_is(kv_read(s, "nope", [&](kv_bytes) { called = true; }), false) && !called,
                   "read of a missing key is false and never calls the visitor");
            expect(kv_is(kv_erase(s, "nope"), false), "erase of a missing key is false, not an error");
            expect(kv_is(kv_contains(s, "nope"), false), "contains of a missing key is false, not an error");
        });

        run("put_overwrites", [](auto& s, const auto& expect) {
            static_cast<void>(kv_put(s, "k", "a long first value that does not fit in a small string buffer"));
            static_cast<void>(kv_put(s, "k", "b"));
            expect(kv_value(s, "k") == "b", "the second put wins");
            expect(kv_is(kv_count(s), 1), "an overwrite adds no key");
        });

        run("empty_key_and_value_exist", [](auto& s, const auto& expect) {
            expect(kv_put(s, "", "").is_ok(), "an empty key and value can be stored");
            expect(kv_is(kv_contains(s, ""), true), "the empty key is present");
            expect(kv_value(s, "") == std::string(), "present and empty, not missing");
            static_cast<void>(kv_put(s, "e", ""));
            expect(kv_is(kv_contains(s, "e"), true), "an empty value is present");
            expect(kv_is(kv_count(s), 2), "both are counted");
        });

        run("embedded_nul_round_trips", [](auto& s, const auto& expect) {
            const std::string_view key("a\0b", 3);
            const std::string_view value("x\0y\0", 4);
            static_cast<void>(kv_put(s, key, value));
            expect(kv_value(s, key) == value, "a NUL inside a key or value is kept");
            expect(kv_is(kv_contains(s, "a"), false), "a key is not truncated at a NUL");
            expect(kv_keys(s) == strings{std::string(key)}, "the scan sees the whole key");
        });

        if (options.binary_safe)
            run("arbitrary_bytes_round_trip", [](auto& s, const auto& expect) {
                const std::array<std::byte, 3> key{std::byte{0xFF}, std::byte{0x00}, std::byte{0x80}};
                std::array<std::byte, 256> value{};
                for (std::size_t i = 0; i < value.size(); ++i)
                    value[i] = static_cast<std::byte>(i);
                static_cast<void>(s.put(key, value));
                std::vector<std::byte> out;
                expect(kv_is(kv_get(s, key, out), true) && kv_equal(out, value), "every byte value round-trips");
            });

        run("insert_only_when_absent", [](auto& s, const auto& expect) {
            expect(kv_is(kv_insert(s, "k", "first"), true), "insert of an absent key inserts");
            const auto before = kv_seq(s);
            expect(kv_is(kv_insert(s, "k", "second"), false), "insert of a present key reports that it existed");
            expect(kv_value(s, "k") == "first", "insert keeps the existing value");
            expect(kv_seq(s) == before, "an insert that found its key changes nothing, the sequence included");
        });

        run("ledger_entry_is_written_once", [](auto& s, const auto& expect) {
            // Two writers append entry n of a book: each reads the next n, then inserts e/<book>/<n>.
            const auto next_entry = [&] {
                const auto n = kv_count(s, kv_range::prefixed(kv_b("e/book-7/")));
                return std::string("e/book-7/") + std::to_string(n.is_ok() ? *n : 0);
            };
            static_cast<void>(kv_insert(s, "e/book-7/0", "opening balance"));
            const std::string key_a = next_entry();
            const std::string key_b = next_entry();   // read before writer A writes: stale
            expect(key_a == "e/book-7/1" && key_a == key_b, "both writers chose entry 1");
            expect(kv_is(kv_insert(s, key_a, "A: +100"), true), "the first writer's entry is written");
            const auto before = kv_seq(s);
            expect(kv_is(kv_insert(s, key_b, "B: +250"), false), "the stale writer is told the entry exists");
            expect(kv_value(s, key_a) == "A: +100", "the stale writer changed nothing");
            expect(kv_seq(s) == before, "the stale insert did not move the sequence");
            // The stale writer's whole batch (its entry plus its index) is refused, all or nothing.
            kv_write_batch batch;
            batch.put("idx/book-7/last", "B").insert(key_b, "B: +250");
            const auto refused = s.apply(batch);
            expect(refused.is_err() && refused.unwrap_err_unchecked() == kv_errc::conflict,
                   "a batch whose insert finds its key fails with kv_errc::conflict");
            expect(kv_is(kv_contains(s, "idx/book-7/last"), false), "nothing of the refused batch was applied");
            expect(kv_seq(s) == before, "the refused batch did not move the sequence");
            expect(kv_is(kv_count(s, kv_range::prefixed(kv_b("e/book-7/"))), 2), "the book holds exactly two entries");
        });

        run("erase_reports_whether_the_key_existed", [](auto& s, const auto& expect) {
            static_cast<void>(kv_put(s, "k", "v"));
            expect(kv_is(kv_erase(s, "k"), true), "erase of a present key is true");
            expect(kv_is(kv_erase(s, "k"), false), "a second erase is false");
            expect(kv_is(kv_contains(s, "k"), false), "the key is gone");
        });

        run("scan_is_in_unsigned_bytewise_order", [](auto& s, const auto& expect) {
            for (const char* k : {"b", "a", "\xC3\xA9", "A", "ab", "z"})
                static_cast<void>(kv_put(s, k, k));
            expect(kv_keys(s) == strings{"A", "a", "ab", "b", "z", "\xC3\xA9"}, "keys sort as unsigned bytes (0xC3 after 'z')");
            bool pairs_match = true;
            static_cast<void>(kv_scan(s, kv_range::all(), [&](kv_bytes k, kv_bytes v) { pairs_match = pairs_match && kv_equal(k, v); }));
            expect(pairs_match, "each key is scanned with its own value");
        });

        run("ranges_are_half_open", [](auto& s, const auto& expect) {
            for (const char* k : {"a", "b", "c", "d", "e"})
                static_cast<void>(kv_put(s, k, "v"));
            expect(kv_keys(s, kv_range::between(kv_b("b"), kv_b("d"))) == strings{"b", "c"}, "between includes first, excludes last");
            expect(kv_keys(s, kv_range::between(kv_b("bb"), kv_b("dd"))) == strings{"c", "d"}, "bounds need not be keys");
            expect(kv_keys(s, kv_range::from(kv_b("d"))) == strings{"d", "e"}, "from includes its bound");
            expect(kv_keys(s, kv_range::between(kv_b("d"), kv_b("b"))).empty(), "an inverted range is empty");
            expect(kv_keys(s, kv_range::between(kv_b("c"), kv_b("c"))).empty(), "an empty range is empty");
            expect(kv_is(kv_count(s, kv_range::between(kv_b("a"), kv_b("c"))), 2), "count counts the range");
        });

        run("scan_prefix_is_a_true_prefix", [](auto& s, const auto& expect) {
            for (const char* k : {"use", "user/1", "user/2", "users", "my/user/3", "user0"})
                static_cast<void>(kv_put(s, k, "v"));
            strings seen;
            const auto n = kv_scan_prefix(s, "user/", [&](kv_bytes k, kv_bytes) { seen.emplace_back(reinterpret_cast<const char*>(k.data()), k.size()); });
            expect(kv_is(n, 2) && seen == strings{"user/1", "user/2"}, "a prefix visits exactly its keys, never a substring");
            expect(kv_is(kv_count(s, kv_range::prefixed(kv_b(""))), 6), "the empty prefix is every key");
        });

        if (options.binary_safe)
            run("prefix_ending_in_ff_is_bounded", [](auto& s, const auto& expect) {
                using namespace std::string_literals;
                for (const std::string& k : {"a\xFF"s, "a\xFF\x01"s, "a\xFF\xFF"s, "b"s, "a"s})
                    static_cast<void>(kv_put(s, k, "v"));
                expect(kv_keys(s, kv_range::prefixed(kv_b("a\xFF"))) == strings{"a\xFF"s, "a\xFF\x01"s, "a\xFF\xFF"s},
                       "a prefix ending in 0xFF holds its keys and not \"b\"");
                expect(kv_is(kv_count(s, kv_range::prefixed(kv_b("\xFF"))), 0), "an all-0xFF prefix matches nothing here");
            });

        run("scan_stops_when_the_visitor_says_so", [](auto& s, const auto& expect) {
            for (const char* k : {"a", "b", "c", "d"})
                static_cast<void>(kv_put(s, k, "v"));
            strings seen;
            const auto n = kv_scan(s, kv_range::all(), [&](kv_bytes k, kv_bytes) {
                seen.emplace_back(reinterpret_cast<const char*>(k.data()), k.size());
                return seen.back() != "b";
            });
            expect(kv_is(n, 2) && seen == strings{"a", "b"}, "the count includes the entry that stopped the scan");
        });

        run("chunked_scan_equals_one_scan", [](auto& s, const auto& expect) {
            for (int i = 0; i < 10; ++i)
                static_cast<void>(kv_put(s, "p/" + std::to_string(i), "v"));
            static_cast<void>(kv_put(s, "q", "v"));
            for (const kv_range whole : {kv_range::all(), kv_range::prefixed(kv_b("p/"))}) {
                strings chunked;
                kv_range rest = whole;
                for (;;) {   // chunks of three, each resumed after the last key it saw
                    strings chunk;
                    static_cast<void>(kv_scan(s, rest, [&](kv_bytes k, kv_bytes) {
                        chunk.emplace_back(reinterpret_cast<const char*>(k.data()), k.size());
                        return chunk.size() < 3;
                    }));
                    if (chunk.empty())
                        break;
                    chunked.insert(chunked.end(), chunk.begin(), chunk.end());
                    rest = whole.after(kv_b(chunked.back()));
                }
                expect(chunked == kv_keys(s, whole), "scanning in chunks with after() visits the same keys once each");
            }
        });

        run("erase_range_removes_exactly_the_range", [](auto& s, const auto& expect) {
            for (const char* k : {"log/1", "log/2", "log/3", "logs", "m"})
                static_cast<void>(kv_put(s, k, "v"));
            expect(kv_is(s.erase_range(kv_range::between(kv_b("log/1"), kv_b("log/3"))), 2), "between erases its keys");
            expect(kv_keys(s) == strings{"log/3", "logs", "m"}, "and nothing else");
            expect(kv_is(s.erase_range(kv_range::prefixed(kv_b("log/"))), 1), "a prefix erases its keys");
            expect(kv_is(s.erase_range(kv_range::between(kv_b("z"), kv_b("a"))), 0), "an inverted range erases nothing");
            expect(kv_is(s.erase_range(kv_range::all()), 2), "all() erases everything");
            expect(kv_is(kv_count(s), 0), "the store is empty");
        });

        run("apply_runs_writes_in_order", [](auto& s, const auto& expect) {
            static_cast<void>(kv_put(s, "old", "x"));
            kv_write_batch batch;
            batch.put("a", "1").put("b", "2").erase("a").put("c", "3").put("c", "4").erase("old").erase("never");
            batch.erase("d").put("d", "5").insert("e", "6");
            expect(s.apply(batch).is_ok(), "a valid batch applies");
            expect(kv_keys(s) == strings{"b", "c", "d", "e"}, "writes apply in order");
            expect(kv_value(s, "c") == "4" && kv_value(s, "d") == "5" && kv_value(s, "e") == "6", "the last write of a key wins");
            const auto before = kv_seq(s);
            expect(s.apply(std::span<const kv_write>{}).is_ok(), "an empty batch succeeds");
            expect(kv_seq(s) == before && kv_is(kv_count(s), 4), "an empty batch changes nothing");
            // A batch of views needs no allocation at all.
            const std::array<kv_write, 2> views{kv_write::put(kv_b("f"), kv_b("7")), kv_write::erase(kv_b("b"))};
            expect(s.apply(views).is_ok() && kv_keys(s) == strings{"c", "d", "e", "f"}, "a std::array of kv_write applies");
        });

        run("batch_insert_conflict_changes_nothing", [](auto& s, const auto& expect) {
            static_cast<void>(kv_put(s, "taken", "kept"));
            const auto before = kv_seq(s);
            kv_write_batch batch;
            batch.put("a", "1").insert("taken", "lost").put("b", "2");
            const auto r = s.apply(batch);
            expect(r.is_err() && r.unwrap_err_unchecked() == kv_errc::conflict, "an insert of a present key fails the batch with conflict");
            expect(kv_keys(s) == strings{"taken"} && kv_value(s, "taken") == "kept", "no write of the failed batch was applied");
            expect(kv_seq(s) == before, "the failed batch did not move the sequence");
            kv_write_batch twice;
            twice.insert("n", "1").insert("n", "2");
            const auto t = s.apply(twice);
            expect(t.is_err() && t.unwrap_err_unchecked() == kv_errc::conflict, "a second insert of a key in one batch conflicts with the first");
            expect(kv_is(kv_contains(s, "n"), false), "and the first is not applied either");
        });

        run("visitor_exceptions_propagate_and_the_store_stays_usable", [](auto& s, const auto& expect) {
            static_cast<void>(kv_put(s, "a", "1"));
            static_cast<void>(kv_put(s, "b", "2"));
#if SNICHOLLS_RESULT_HAS_EXCEPTIONS
            bool read_threw = false;
            try {
                static_cast<void>(kv_read(s, "a", [](kv_bytes) { throw std::logic_error("visitor"); }));
            } catch (const std::logic_error&) {
                read_threw = true;
            }
            expect(read_threw, "a read visitor's exception reaches the caller unchanged");
            bool scan_threw = false;
            try {
                static_cast<void>(kv_scan(s, kv_range::all(), [](kv_bytes, kv_bytes) { throw std::logic_error("visitor"); }));
            } catch (const std::logic_error&) {
                scan_threw = true;
            }
            expect(scan_threw, "a scan visitor's exception reaches the caller unchanged");
#endif
            expect(kv_put(s, "c", "3").is_ok() && kv_value(s, "c") == "3", "the store still writes");
            expect(kv_is(kv_count(s), 3), "and still scans");
        });

        run("many_keys_scan_in_order", [](auto& s, const auto& expect) {
            kv_write_batch batch;
            for (int i = 999; i >= 0; --i) {
                std::string key = std::to_string(i);
                key.insert(0, 4 - key.size(), '0');
                batch.put("k" + key, std::to_string(i));
            }
            expect(s.apply(batch).is_ok(), "a batch of 1000 writes applies");
            const auto keys = kv_keys(s);
            bool sorted = keys.size() == 1000;
            for (std::size_t i = 1; sorted && i < keys.size(); ++i)
                sorted = keys[i - 1] < keys[i];
            expect(sorted, "1000 keys scan in ascending order");
            expect(kv_value(s, "k0500") == "500", "and read back");
        });

        run("sequence_rules", [](auto& s, const auto& expect) {
            const auto s0 = kv_seq(s);
            static_cast<void>(kv_put(s, "a", "1"));
            const auto s1 = kv_seq(s);
            expect(s1 > s0, "a put raises the sequence");
            static_cast<void>(kv_get(s, "a"));
            static_cast<void>(kv_count(s));
            static_cast<void>(kv_contains(s, "a"));
            expect(kv_seq(s) == s1, "reads never move it");
            static_cast<void>(kv_erase(s, "missing"));
            expect(kv_seq(s) == s1, "an erase that removed nothing does not move it");
            static_cast<void>(kv_erase(s, "a"));
            const auto s2 = kv_seq(s);
            expect(s2 > s1, "an erase that removed a key raises it");
            kv_write_batch batch;
            batch.put("x", "1").put("y", "2");
            static_cast<void>(s.apply(batch));
            expect(kv_seq(s) > s2, "a batch raises it");
        });

        run("snapshot_is_isolated", [](auto& s, const auto& expect) {
            static_cast<void>(kv_put(s, "a", "1"));
            static_cast<void>(kv_put(s, "gone", "x"));
            const auto at = kv_seq(s);
            auto snap = s.snapshot();
            expect(snap.is_ok(), "snapshot() succeeds");
            if (snap.is_err())
                return;
            const auto& view = detail::kv_target(*snap);
            static_cast<void>(kv_put(s, "a", "2"));
            static_cast<void>(kv_put(s, "b", "new"));
            static_cast<void>(kv_erase(s, "gone"));
            expect(kv_value(view, "a") == "1", "a snapshot keeps the value it saw");
            expect(kv_is(kv_contains(view, "b"), false), "a snapshot does not see later inserts");
            expect(kv_is(kv_contains(view, "gone"), true), "a snapshot does not see later erases");
            expect(kv_keys(view) == strings{"a", "gone"}, "a snapshot scans its own state");
            expect(kv_seq(view) == at, "a snapshot's sequence is the store's when it was taken");
            expect(kv_value(s, "a") == "2" && kv_seq(s) > at, "the store moved on");
        });

        run("digest_v1_matches_its_golden_vector", [](auto& s, const auto& expect) {
            for (const auto& [k, v] : {std::pair{"b", "y"}, std::pair{"", "e"}, std::pair{"ab", "x"}, std::pair{"a", ""}})
                static_cast<void>(kv_put(s, k, v));
            const auto d = kv_digest_v1(s);
            expect(d.is_ok() && d->hex().view() == kv_digest_v1_kit_vector, "kv_digest_v1 of the kit's pairs is the golden vector");
            auto snap = s.snapshot();
            if (snap.is_ok()) {
                static_cast<void>(kv_put(s, "c", "z"));
                const auto ds = kv_digest_v1(detail::kv_target(*snap));
                expect(ds.is_ok() && d.is_ok() && *ds == *d, "a snapshot's digest is the digest of its moment");
            }
        });

        run("no_exception_crosses_the_interface", [](auto& s, const auto& expect) {
            // Every operation, with non-throwing visitors, returns: kv_run records anything that escapes.
            static_cast<void>(kv_put(s, "k", "v"));
            static_cast<void>(kv_insert(s, "k", "w"));
            static_cast<void>(kv_get(s, "k"));
            static_cast<void>(kv_contains(s, "k"));
            static_cast<void>(kv_count(s));
            static_cast<void>(s.sequence());
            static_cast<void>(s.snapshot());
            kv_write_batch batch;
            batch.insert("k", "conflict");
            static_cast<void>(s.apply(batch));
            static_cast<void>(s.erase_range(kv_range::all()));
            static_cast<void>(kv_erase(s, "k"));
            expect(true, "");
        });

        return report;
    }

    /// \brief Checks that a write which fails, alone or in the middle of a batch, is a typed error and changes nothing.
    /// \ingroup core
    ///
    /// `arm(store, n)` must make the write `n` writes from now fail (as `reference_key_value_store::inject_failure`
    /// does); the check arms `n = 0` for a `put` and `n = 2` for a three-write batch, then checks the error, that
    /// nothing changed (keys, values, sequence) and that the store still works.
    /// \tparam S a `key_value_store`.
    /// \tparam Arm callable as `void(S&, std::size_t)`.
    /// \param store a store; its contents are replaced.
    /// \param arm arms the failure.
    /// \return the report.
    template <key_value_store S, typename Arm>
    kv_conformance_report kv_check_failure_changes_nothing(S& store, Arm&& arm)
    {
        kv_conformance_report report;
        auto make = [&store]() -> S& { return store; };
        detail::kv_run(report, "failure_changes_nothing", make, [&](auto& s, const auto& expect) {
            static_cast<void>(s.erase_range(kv_range::all()));
            static_cast<void>(kv_put(s, "a", "1"));
            const auto before = detail::kv_seq(s);
            arm(s, std::size_t{0});
            expect(kv_put(s, "a", "2").is_err(), "an armed put fails with an error, not an exception");
            expect(detail::kv_value(s, "a") == "1" && detail::kv_seq(s) == before, "the failed put changed nothing");
            arm(s, std::size_t{2});
            kv_write_batch batch;
            batch.put("b", "1").erase("a").put("c", "1");
            expect(s.apply(batch).is_err(), "a batch failing at its third write fails");
            expect(detail::kv_keys(s) == std::vector<std::string>{"a"} && detail::kv_seq(s) == before,
                   "the failed batch applied none of its writes");
            expect(kv_put(s, "d", "1").is_ok() && detail::kv_is(kv_count(s), 2), "the store works after the failures");
        });
        return report;
    }

    /// \brief Checks that every operation returns rather than throws, under whatever failure `arm(store)` sets up.
    /// \ingroup core
    ///
    /// For a store that can be made unreachable (a closed connection, `set_unavailable(true)`): every operation must
    /// then answer an error, and no exception may cross.
    /// \tparam S a `key_value_store`.
    /// \tparam Arm callable as `void(S&)`.
    /// \param store the store.
    /// \param arm puts the store into its failing state.
    /// \return the report.
    template <key_value_store S, typename Arm>
    kv_conformance_report kv_check_no_exception_escapes(S& store, Arm&& arm)
    {
        kv_conformance_report report;
        auto make = [&store]() -> S& { return store; };
        detail::kv_run(report, "no_exception_escapes", make, [&](auto& s, const auto& expect) {
            arm(s);
            const auto k = detail::kv_b("k");
            bool visited = false;
            expect(s.put(k, k).is_err(), "put answers an error");
            expect(s.insert(k, k).is_err(), "insert answers an error");
            expect(s.erase(k).is_err(), "erase answers an error");
            expect(s.erase_range(kv_range::all()).is_err(), "erase_range answers an error");
            expect(s.apply(std::array{kv_write::put(k, k)}).is_err(), "apply answers an error");
            expect(kv_read(s, k, [&](kv_bytes) { visited = true; }).is_err() && !visited, "read answers an error");
            expect(s.contains(k).is_err(), "contains answers an error");
            expect(kv_count(s).is_err(), "scan answers an error");
            expect(s.sequence().is_err(), "sequence answers an error");
            expect(s.snapshot().is_err(), "snapshot answers an error");
        });
        return report;
    }

    /// \brief Runs the checks of the health capability, each on a fresh store from `make()`.
    /// \ingroup core
    /// \tparam Make callable with no arguments, returning a store that models `kv_health` and `kv_writable`, or a
    /// handle to one.
    /// \param make makes a fresh, empty store.
    /// \return the report.
    template <typename Make>
    kv_conformance_report run_kv_health_conformance(Make&& make)
    {
        kv_conformance_report report;
        detail::kv_run(report, "ping_succeeds_on_a_working_store", make, [](auto& s, const auto& expect) {
            expect(s.ping().is_ok(), "ping() succeeds");
        });
        detail::kv_run(report, "diagnostics_describe_the_store", make, [](auto& s, const auto& expect) {
            static_cast<void>(s.put(detail::kv_b("a"), detail::kv_b("12")));
            const auto d = s.diagnostics();
            expect(d.is_ok(), "diagnostics() succeeds");
            if (d.is_err())
                return;
            expect(!d->backend.empty(), "the backend names itself");
            expect(d->sequence == detail::kv_seq(s), "the sequence is the store's");
            expect(!d->keys || *d->keys == 1, "a key count, when given, is exact");
            expect(!d->bytes || *d->bytes >= 3, "a byte count, when given, covers the data");
            expect(detail::kv_seq(s) == d->sequence, "diagnostics changed nothing");
        });
        return report;
    }

    /// \brief Runs the checks of the backup capability, each on a fresh store from `make()`.
    /// \ingroup core
    /// \tparam Make callable with no arguments, returning a store that models `kv_backup` and `kv_writable`, or a
    /// handle to one.
    /// \param make makes a fresh, empty store.
    /// \param destination a checkpoint destination that does not exist yet, used by each check in turn.
    /// \return the report.
    template <typename Make>
    kv_conformance_report run_kv_backup_conformance(Make&& make, std::string_view destination)
    {
        kv_conformance_report report;
        detail::kv_run(report, "checkpoint_captures_one_moment", make, [&](auto& s, const auto& expect) {
            static_cast<void>(s.put(detail::kv_b("a"), detail::kv_b("1")));
            const auto at = detail::kv_seq(s);
            const auto c = s.checkpoint(destination);
            expect(c.is_ok() && c->sequence == at, "a checkpoint reports the sequence it captured");
            expect(detail::kv_seq(s) == at && detail::kv_value(s, "a") == "1", "checkpointing changed nothing");
            const auto again = s.checkpoint(destination);
            expect(again.is_err() && again.unwrap_err_unchecked() == kv_errc::invalid_argument,
                   "an existing destination is refused, never overwritten");
            const auto empty = s.checkpoint(std::string_view{});
            expect(empty.is_err() && empty.unwrap_err_unchecked() == kv_errc::invalid_argument, "an empty destination is refused");
        });
        detail::kv_run(report, "verify_passes_a_healthy_store", make, [](auto& s, const auto& expect) {
            static_cast<void>(s.put(detail::kv_b("a"), detail::kv_b("1")));
            static_cast<void>(s.put(detail::kv_b("b"), detail::kv_b("2")));
            const auto v = s.verify();
            expect(v.is_ok() && v->ok(), "verify() finds no problem in a healthy store");
        });
        return report;
    }
}

#endif /* snicholls_core_interfaces_key_value_store_conformance_hpp */
