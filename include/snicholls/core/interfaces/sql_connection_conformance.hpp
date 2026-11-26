//
//  sql_connection_conformance.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::run_sql_connection_conformance`: the conformance kit of the `sql_connection` contract.
/// \ingroup core
///
/// **Overview.** Drives any `sql_connection`, in either form, through the contract of `concepts/sql_connection.hpp`
/// and reports which check failed (`interfaces/capability_conformance.hpp`); crtp's `sql_connection_contract` in core's
/// form. Every statement is generated through the connection's own `dialect()`, so the same kit runs on SQLite,
/// PostgreSQL and the in-memory reference. The probe is a default-constructible type with
/// - `auto& connection()`: the implementation (a connection, a `sql_connection_interface&`, a `sql_connection_ref`),
///   on an empty database or schema of its own;
/// - `void connection_breaks()`: the connection is lost (a closed socket, a terminated backend).
///
/// A fresh probe is made for every check. The checks: `values round-trip`, `a duplicate key is a unique_violation
/// and conflicts resolve`, `a rollback discards and a commit keeps`, `misuse and missing objects are errors`, `a
/// failed batch changes nothing`, `ranges, orders and limits select as written`, `a visitor can stop`, `errors carry
/// no value`, `a lost connection is connection_lost`.
///
/// \snippet interfaces/sql_connection_conformance_demo.cpp kit
///
/// **Relationships.** `concepts/sql_connection.hpp`, `concepts/sql_dialect.hpp`, `interfaces/sql_connection.hpp`,
/// `interfaces/capability_conformance.hpp`; run on `memory_sql_connection` in every form by
/// `tests/interfaces/sql_connection_test.cpp`. Group: \ref core.
///
/// **Implementations / variants.** One kit for every form and engine.
///
/// **Guarantees.** As `interfaces/capability_conformance.hpp`: no exception escapes a check. The kit creates and drops
/// tables named `kit_*`.
///
/// **Errors.** Failures are data.
///
/// **Performance.** Not benchmarked: a test kit (a few dozen statements per check).
///
/// **Tests and demos.** `tests/interfaces/sql_connection_conformance_test.cpp` (the kit catches broken connections),
/// `demos/interfaces/sql_connection_conformance_demo.cpp`.
///
/// **Dependencies.** `<algorithm>`, `<array>`, `<chrono>`, `<cstddef>`, `<cstdint>`, `<span>`, `<string>`, `<string_view>`, `<type_traits>`,
/// `<utility>`, `<vector>`, `concepts/sql_connection.hpp`, `interfaces/capability_conformance.hpp`.
///
/// **Known issues.** Concurrency (two connections, serialization failures) is not checked: one connection is.
///
/// **Status.** New in 0.27.0.

#ifndef snicholls_core_interfaces_sql_connection_conformance_hpp
#define snicholls_core_interfaces_sql_connection_conformance_hpp

#include <snicholls/core/concepts/sql_connection.hpp>
#include <snicholls/core/interfaces/capability_conformance.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace snicholls {

    namespace detail {
        template <typename Probe>
        concept sql_connection_probe = std::is_default_constructible_v<Probe> &&
            sql_connection<std::remove_reference_t<decltype(std::declval<Probe&>().connection())>> &&
            requires(Probe& p) { p.connection_breaks(); };

        inline sql_table kit_values_table()
        {
            return {"kit_values",
                    {{"id", sql_type::int64, false},
                     {"flag", sql_type::boolean, false},
                     {"price", sql_type::float64, false},
                     {"note", sql_type::text, true},
                     {"raw", sql_type::blob, false},
                     {"at", sql_type::timestamp, false},
                     {"ref", sql_type::uuid, false},
                     {"amount", sql_type::decimal, false}},
                    {0},
                    false};
        }

        inline sql_table kit_series_table()
        {
            return {"kit_series", {{"series", sql_type::text, false}, {"ts", sql_type::int64, false}, {"px", sql_type::float64, true}}, {0, 1}, true};
        }

        // Drops and creates `t`; false (and a recorded failure) when the engine refuses.
        template <typename C, typename E>
        bool kit_fresh_table(C& c, const sql_table& t, const E& expect)
        {
            const auto drop = sql_drop_table(c.dialect(), t.name);
            const auto create = sql_create_table(c.dialect(), t);
            const bool good = drop && create && c.execute(*drop, {}).is_ok() && c.execute(*create, {}).is_ok();
            expect(good, "the kit's table is created");
            return good;
        }

        template <typename C>
        std::int64_t kit_count(C& c, const sql_table& t)
        {
            std::int64_t n = -1;
            const auto q = sql_count(c.dialect(), t);
            if (q)
                static_cast<void>(sql_query(c, *q, std::span<const sql_param>{}, [&n](const sql_row& r) { n = r.get_int64(0).unwrap_or(-1); }));
            return n;
        }

        template <typename C>
        sql_result<std::size_t> kit_insert_series(C& c, std::string_view series, std::int64_t ts, sql_conflict conflict = sql_conflict::fail)
        {
            const auto sql = sql_insert(c.dialect(), kit_series_table(), conflict);
            if (!sql)
                return err(sql.unwrap_err());
            return sql_execute(c, *sql, {sql_text(series), ts, static_cast<double>(ts) / 2});
        }
    } // namespace detail

    /// \brief Runs every check of the `sql_connection` contract, each on a fresh `Probe`.
    /// \ingroup core
    /// \tparam Probe The probe (see the file's overview).
    /// \return The report.
    template <typename Probe>
        requires detail::sql_connection_probe<Probe>
    [[nodiscard]] capability_conformance_report run_sql_connection_conformance()
    {
        capability_conformance_report report;
        using detail::kit_series_table;

        detail::capability_check<Probe>(report, "values round-trip", [](Probe& p, const auto& expect) {
            auto& c = p.connection();
            const sql_table t = detail::kit_values_table();
            if (!detail::kit_fresh_table(c, t, expect))
                return;
            const std::array<std::byte, 4> raw{std::byte{0}, std::byte{0xff}, std::byte{'"'}, std::byte{7}};
            const utc_ns at = utc_ns::from_unix_nanos(-1'234'567'891);   // before 1970, to the nanosecond
            const uuid ref = *parse_uuid("0190a5c4-7e1b-7c3a-9f12-3b4c5d6e7f80");
            const decimal_text amount = decimal::parse("-1234.5678")->text();
            const auto insert = sql_insert(c.dialect(), t);
            const auto inserted = insert ? sql_execute(c, *insert, {std::int64_t{1}, true, 0.1, sql_null, sql_blob(raw), to_sql(at), to_sql(ref), to_sql(amount)})
                                         : sql_result<std::size_t>(err(insert.unwrap_err()));
            expect(inserted.is_ok() && *inserted == 1, "a row is inserted");
            sql_select by_key;
            by_key.equal = {0};
            const auto select = sql_select_rows(c.dialect(), t, by_key);
            bool seen = false;
            if (select)
                static_cast<void>(sql_query(c, *select, {std::int64_t{1}}, [&](const sql_row& r) {
                    seen = true;
                    expect(r.size() == 8, "every column comes back");
                    expect(r.get_int64(0).unwrap_or(0) == 1, "int64");
                    expect(r.get_bool(1).unwrap_or(false), "boolean");
                    expect(r.get_double(2).unwrap_or(0) == 0.1, "float64, exactly");
                    expect(r.is_null(3), "NULL");
                    const auto b = r.get_blob(4);
                    expect(b && b->size() == raw.size() && std::equal(b->begin(), b->end(), raw.begin()), "a blob, byte for byte");
                    const auto a = r.cell(5);
                    expect(a && utc_from_sql<std::chrono::nanoseconds>(*a).is_ok() && *utc_from_sql<std::chrono::nanoseconds>(*a) == at,
                           "a utc before 1970, to the nanosecond");
                    const auto u = r.cell(6);
                    expect(u && uuid_from_sql(*u).is_ok() && *uuid_from_sql(*u) == ref, "a uuid");
                    const auto d = r.cell(7);
                    expect(d && decimal_from_sql(*d).is_ok() && decimal_from_sql(*d)->text().view() == amount.view(), "a decimal, exactly");
                }));
            expect(seen, "the row is found by its key");
        });

        detail::capability_check<Probe>(report, "a duplicate key is a unique_violation and conflicts resolve", [](Probe& p, const auto& expect) {
            auto& c = p.connection();
            const sql_table t = kit_series_table();
            if (!detail::kit_fresh_table(c, t, expect))
                return;
            expect(detail::kit_insert_series(c, "a", 1).is_ok(), "the first insert");
            const auto dup = detail::kit_insert_series(c, "a", 1);
            expect(dup.is_err() && dup.unwrap_err().code == sql_errc::unique_violation, "the same key is a unique_violation");
            const auto ignored = detail::kit_insert_series(c, "a", 1, sql_conflict::ignore);
            expect(ignored.is_ok() && *ignored == 0, "DO NOTHING changes no row");
            const auto sql = sql_insert(c.dialect(), t, sql_conflict::update);
            const auto upserted = sql ? sql_execute(c, *sql, {sql_text("a"), std::int64_t{1}, 99.0}) : sql_result<std::size_t>(err(sql.unwrap_err()));
            expect(upserted.is_ok() && *upserted == 1, "an upsert changes one row");
            double px = 0;
            const auto all = sql_select_rows(c.dialect(), t);
            if (all)
                static_cast<void>(sql_query(c, *all, std::span<const sql_param>{}, [&px](const sql_row& r) { px = r.get_double(2).unwrap_or(0); }));
            expect(px == 99.0, "the upsert overwrote the other columns");
            expect(detail::kit_count(c, t) == 1, "still one row");
        });

        detail::capability_check<Probe>(report, "a rollback discards and a commit keeps", [](Probe& p, const auto& expect) {
            auto& c = p.connection();
            const sql_table t = kit_series_table();
            if (!detail::kit_fresh_table(c, t, expect))
                return;
            expect(c.begin().is_ok() && c.in_transaction(), "begin opens a transaction");
            const auto nested = c.begin();
            expect(nested.is_err() && nested.unwrap_err().code == sql_errc::misuse, "a nested begin is misuse");
            static_cast<void>(detail::kit_insert_series(c, "a", 1));
            expect(c.rollback().is_ok() && !c.in_transaction(), "rollback closes it");
            expect(detail::kit_count(c, t) == 0, "the rollback discarded the row");
            const auto stray = c.commit();
            expect(stray.is_err() && stray.unwrap_err().code == sql_errc::misuse, "commit without begin is misuse");
            {
                auto tx = sql_transaction<std::remove_reference_t<decltype(c)>>::begin(c);
                expect(tx.is_ok(), "sql_transaction begins");
                static_cast<void>(detail::kit_insert_series(c, "a", 2));
            }   // not committed: rolled back
            expect(detail::kit_count(c, t) == 0 && !c.in_transaction(), "an uncommitted sql_transaction rolls back");
            auto tx = sql_transaction<std::remove_reference_t<decltype(c)>>::begin(c);
            static_cast<void>(detail::kit_insert_series(c, "a", 3));
            expect(tx.is_ok() && tx->commit().is_ok(), "commit succeeds");
            expect(detail::kit_count(c, t) == 1, "the commit kept the row");
        });

        detail::capability_check<Probe>(report, "misuse and missing objects are errors", [](Probe& p, const auto& expect) {
            auto& c = p.connection();
            const sql_table t = kit_series_table();
            if (!detail::kit_fresh_table(c, t, expect))
                return;
            const auto sql = sql_insert(c.dialect(), t);
            if (sql) {
                const auto few = sql_execute(c, *sql, {sql_text("a")});
                expect(few.is_err() && few.unwrap_err().code == sql_errc::misuse, "the wrong number of parameters is misuse");
                const auto null_key = sql_execute(c, *sql, {sql_text("a"), sql_null, 1.0});
                expect(null_key.is_err() && (null_key.unwrap_err().code == sql_errc::constraint_violation ||
                                             null_key.unwrap_err().code == sql_errc::type_mismatch),
                       "NULL in a NOT NULL key is refused");
                const auto text_in_int = sql_execute(c, *sql, {sql_text("a"), sql_text("not a number"), 1.0});
                expect(text_in_int.is_err() && text_in_int.unwrap_err().code == sql_errc::type_mismatch, "text in an integer column is a type_mismatch");
            }
            sql_table missing = t;
            missing.name = "kit_no_such_table";
            const auto drop = sql_drop_table(c.dialect(), missing.name);
            if (drop)
                static_cast<void>(c.execute(*drop, {}));
            const auto absent = sql_count(c.dialect(), missing);
            const auto r = absent ? c.query(*absent, {}, sql_row_visitor([](const sql_row&) { return true; })) : sql_result<std::size_t>(err(absent.unwrap_err()));
            expect(r.is_err() && r.unwrap_err().code == sql_errc::undefined_object, "an unknown table is undefined_object");
            const auto garbage = c.prepare("SELEKT nothing FROMM nowhere");
            expect(garbage.is_err() && garbage.unwrap_err().code == sql_errc::syntax_error, "text that does not parse is a syntax_error");
            expect(c.prepare(*sql_select_rows(c.dialect(), t)).is_ok(), "a generated statement prepares");
        });

        detail::capability_check<Probe>(report, "a failed batch changes nothing", [](Probe& p, const auto& expect) {
            auto& c = p.connection();
            const sql_table t = kit_series_table();
            if (!detail::kit_fresh_table(c, t, expect))
                return;
            const auto sql = sql_insert(c.dialect(), t);
            if (!sql)
                return;
            const std::array<std::int64_t, 3> bad{1, 2, 1};   // the third repeats the first key
            const auto failed = c.execute_batch(*sql, 3, bad.size(), sql_row_filler([&bad](std::size_t row, std::span<sql_param> out) {
                out[0] = sql_text("a");
                out[1] = bad[row];
                out[2] = 1.0;
            }));
            expect(failed.is_err() && failed.unwrap_err().code == sql_errc::unique_violation, "the duplicate fails the batch");
            expect(detail::kit_count(c, t) == 0, "nothing of the batch was applied");
            const auto good = c.execute_batch(*sql, 3, 5, sql_row_filler([](std::size_t row, std::span<sql_param> out) {
                out[0] = sql_text("b");
                out[1] = static_cast<std::int64_t>(row);
                out[2] = sql_null;
            }));
            expect(good.is_ok() && *good == 5 && detail::kit_count(c, t) == 5, "a good batch inserts every row");
        });

        detail::capability_check<Probe>(report, "ranges, orders and limits select as written", [](Probe& p, const auto& expect) {
            auto& c = p.connection();
            const sql_table t = kit_series_table();
            if (!detail::kit_fresh_table(c, t, expect))
                return;
            for (std::int64_t ts = 1; ts <= 10; ++ts) {
                static_cast<void>(detail::kit_insert_series(c, "b", ts));
                static_cast<void>(detail::kit_insert_series(c, "a", ts));
            }
            sql_select w;
            w.equal = {0};
            w.range_column = 1;
            w.has_from = true;
            w.has_to = true;
            w.order_by = {1};
            w.descending = true;
            w.limit = 2;
            std::vector<std::int64_t> got;
            const auto q = sql_select_rows(c.dialect(), t, w);
            if (q)
                static_cast<void>(sql_query(c, *q, {sql_text("a"), std::int64_t{3}, std::int64_t{7}},
                                            [&got](const sql_row& r) { got.push_back(r.get_int64(1).unwrap_or(-1)); }));
            expect(got == std::vector<std::int64_t>{6, 5}, "series a, 3 <= ts < 7, newest first, two rows: 6 and 5");
            sql_select before;
            before.equal = {0};
            before.range_column = 1;
            before.has_to = true;
            const auto del = sql_delete(c.dialect(), t, before);
            const auto deleted = del ? sql_execute(c, *del, {sql_text("a"), std::int64_t{4}}) : sql_result<std::size_t>(err(del.unwrap_err()));
            expect(deleted.is_ok() && *deleted == 3, "deleting ts < 4 of series a removes three rows");
            expect(detail::kit_count(c, t) == 17, "the rest remain");
        });

        detail::capability_check<Probe>(report, "a visitor can stop", [](Probe& p, const auto& expect) {
            auto& c = p.connection();
            const sql_table t = kit_series_table();
            if (!detail::kit_fresh_table(c, t, expect))
                return;
            for (std::int64_t ts = 1; ts <= 5; ++ts)
                static_cast<void>(detail::kit_insert_series(c, "a", ts));
            int calls = 0;
            const auto q = sql_select_rows(c.dialect(), t);
            const auto seen = q ? sql_query(c, *q, std::span<const sql_param>{}, [&calls](const sql_row&) { return ++calls < 2; })
                                : sql_result<std::size_t>(err(q.unwrap_err()));
            expect(calls == 2, "the visitor is not called after it returns false");
            expect(seen.is_ok() && *seen == 2, "the result counts the rows seen");
        });

        detail::capability_check<Probe>(report, "errors carry no value", [](Probe& p, const auto& expect) {
            auto& c = p.connection();
            const sql_table t = kit_series_table();
            if (!detail::kit_fresh_table(c, t, expect))
                return;
            static_cast<void>(detail::kit_insert_series(c, capability_redaction_marker, 1));
            const auto dup = detail::kit_insert_series(c, capability_redaction_marker, 1);
            expect(dup.is_err(), "the duplicate fails");
            if (dup.is_err())
                expect(dup.unwrap_err().message.view().find(capability_redaction_marker) == std::string_view::npos,
                       "the error does not carry the key's value");
        });

        detail::capability_check<Probe>(report, "a lost connection is connection_lost", [](Probe& p, const auto& expect) {
            expect(p.connection().ping().is_ok() && p.connection().is_open(), "a fresh connection answers");
            p.connection_breaks();
            auto& c = p.connection();
            const auto r = c.ping();
            expect(r.is_err() && r.unwrap_err().code == sql_errc::connection_lost, "ping is connection_lost");
            expect(!c.is_open(), "is_open is false");
        });

        return report;
    }

} // namespace snicholls

#endif // snicholls_core_interfaces_sql_connection_conformance_hpp
