//
//  sql_dialect_conformance.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::run_sql_dialect_conformance`: the conformance kit of the `sql_dialect` contract.
/// \ingroup core
///
/// **Overview.** Drives any `sql_dialect`, in either form, through the contract of `concepts/sql_dialect.hpp` and
/// reports which check failed (`interfaces/capability_conformance.hpp`). The probe is a default-constructible type with
/// `auto& dialect()`: the implementation (a dialect, a `const sql_dialect_interface&`, a `sql_dialect_ref`). The
/// checks: `every type has a name`, `identifiers are quoted`, `unquotable identifiers are refused`, `placeholders are
/// distinct`, `a transaction begins with BEGIN`, `the generators use the dialect`.
///
/// \snippet interfaces/sql_dialect_conformance_demo.cpp kit
///
/// **Relationships.** `concepts/sql_dialect.hpp`, `interfaces/sql_dialect.hpp`,
/// `interfaces/capability_conformance.hpp`; run on both reference dialects in every form by
/// `tests/interfaces/sql_dialect_test.cpp`. Group: \ref core.
///
/// **Implementations / variants.** One kit for every form.
///
/// **Guarantees.** As `interfaces/capability_conformance.hpp`: no exception escapes a check.
///
/// **Errors.** Failures are data.
///
/// **Performance.** Not benchmarked: a test kit.
///
/// **Tests and demos.** `tests/interfaces/sql_dialect_conformance_test.cpp` (the kit catches broken dialects),
/// `demos/interfaces/sql_dialect_conformance_demo.cpp`.
///
/// **Dependencies.** `<cstddef>`, `<set>`, `<string>`, `<string_view>`, `<type_traits>`, `<utility>`,
/// `concepts/sql_dialect.hpp`, `interfaces/capability_conformance.hpp`.
///
/// **Known issues.** The text is checked for its shape, not run: the connection kit runs it.
///
/// **Status.** New in 0.27.0.

#ifndef snicholls_core_interfaces_sql_dialect_conformance_hpp
#define snicholls_core_interfaces_sql_dialect_conformance_hpp

#include <snicholls/core/concepts/sql_dialect.hpp>
#include <snicholls/core/interfaces/capability_conformance.hpp>

#include <cstddef>
#include <set>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace snicholls {

    namespace detail {
        template <typename Probe>
        concept sql_dialect_probe = std::is_default_constructible_v<Probe> &&
            sql_dialect<std::remove_cvref_t<decltype(std::declval<Probe&>().dialect())>>;

        inline sql_table kit_sql_table()
        {
            return {"kit \"quotes\"",
                    {{"id", sql_type::int64, false}, {"name", sql_type::text, true}, {"at", sql_type::timestamp, false}},
                    {0},
                    true};
        }
    } // namespace detail

    /// \brief Runs every check of the `sql_dialect` contract, each on a fresh `Probe`.
    /// \ingroup core
    /// \tparam Probe The probe (see the file's overview).
    /// \return The report.
    template <typename Probe>
        requires detail::sql_dialect_probe<Probe>
    [[nodiscard]] capability_conformance_report run_sql_dialect_conformance()
    {
        capability_conformance_report report;

        detail::capability_check<Probe>(report, "every type has a name", [](Probe& p, const auto& expect) {
            const auto& d = p.dialect();
            bool named = true;
            for (const sql_type t : sql_types) {
                const auto n = d.column_type(t);
                named = named && n && !n->empty();
            }
            expect(named, "column_type names every sql_type");
            expect(!d.name().empty(), "the dialect has a name");
        });

        detail::capability_check<Probe>(report, "identifiers are quoted", [](Probe& p, const auto& expect) {
            const auto& d = p.dialect();
            std::string out = "x";
            expect(d.append_identifier(out, "Price") && out == "x\"Price\"", "a name is quoted and keeps its case");
            out.clear();
            expect(d.append_identifier(out, "a\"b") && out == "\"a\"\"b\"", "an embedded quote is doubled");
            out.clear();
            expect(d.append_identifier(out, "select; drop table x") && out.front() == '"' && out.back() == '"',
                   "a keyword or punctuation stays inside the quotes");
        });

        detail::capability_check<Probe>(report, "unquotable identifiers are refused", [](Probe& p, const auto& expect) {
            const auto& d = p.dialect();
            std::string out = "kept";
            expect(!d.append_identifier(out, ""), "an empty identifier is refused");
            expect(!d.append_identifier(out, std::string_view("a\0b", 3)), "a NUL is refused");
            expect(out == "kept", "nothing is appended when refused");
        });

        detail::capability_check<Probe>(report, "placeholders are distinct", [](Probe& p, const auto& expect) {
            const auto& d = p.dialect();
            std::set<std::string> seen;
            bool numbered = true;
            for (std::size_t n = 1; n <= 100; ++n) {
                std::string out;
                d.append_placeholder(out, n);
                numbered = numbered && out.find(std::to_string(n)) != std::string::npos;
                seen.insert(out);
            }
            expect(seen.size() == 100, "100 numbers make 100 placeholders");
            expect(numbered, "each placeholder carries its number");
        });

        detail::capability_check<Probe>(report, "a transaction begins with BEGIN", [](Probe& p, const auto& expect) {
            expect(p.dialect().begin_transaction().substr(0, 5) == "BEGIN", "begin_transaction starts with BEGIN");
        });

        detail::capability_check<Probe>(report, "the generators use the dialect", [](Probe& p, const auto& expect) {
            const auto& d = p.dialect();
            const sql_table t = detail::kit_sql_table();
            const auto create = sql_create_table(d, t);
            expect(create.is_ok(), "create_table succeeds");
            if (create) {
                expect(create->find("\"kit \"\"quotes\"\"\"") != std::string::npos, "the table's name is quoted");
                expect(create->find(std::string(*d.column_type(sql_type::timestamp))) != std::string::npos, "the dialect's types are used");
                expect(create->find("PRIMARY KEY (\"id\")") != std::string::npos, "the primary key is declared");
            }
            const auto insert = sql_insert(d, t, sql_conflict::update);
            std::string third;
            d.append_placeholder(third, 3);
            expect(insert.is_ok() && insert->find(third) != std::string::npos, "an insert binds every column");
            sql_table bad = t;
            bad.columns[1].name = std::string("n\0", 2);
            expect(sql_create_table(d, bad).is_err() && sql_create_table(d, bad).unwrap_err().code == sql_errc::misuse,
                   "a column that cannot be quoted is misuse");
        });

        return report;
    }

} // namespace snicholls

#endif // snicholls_core_interfaces_sql_dialect_conformance_hpp
