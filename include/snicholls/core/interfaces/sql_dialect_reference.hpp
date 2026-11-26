//
//  sql_dialect_reference.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::sqlite_dialect` and `snicholls::postgres_dialect`: the standard-library reference `sql_dialect`s.
/// \ingroup core
///
/// **Overview.** The text SQLite and PostgreSQL need, from crtp's dialects (SQL_PLAN.md D3) without their engine
/// extras:
///
/// | | SQLite | PostgreSQL |
/// |---|---|---|
/// | placeholders | `?1` | `$1` |
/// | boolean, int64, timestamp | `INTEGER` | `BOOLEAN`, `BIGINT`, `BIGINT` |
/// | float64 | `REAL` | `DOUBLE PRECISION` |
/// | text, decimal | `TEXT` | `TEXT` |
/// | blob, uuid | `BLOB` | `BYTEA` |
/// | transaction | `BEGIN IMMEDIATE` (take the write lock now) | `BEGIN` |
/// | clustered table | ` WITHOUT ROWID` | (none) |
///
/// Identifiers are quoted the standard way in both (`append_quoted_identifier`). Everything but the two appends is
/// `constexpr`.
///
/// \snippet interfaces/sql_dialect_reference_demo.cpp reference
///
/// **Relationships.** Model `sql_dialect` (`concepts/sql_dialect.hpp`); pass `interfaces/sql_dialect_conformance.hpp`
/// directly and through `interfaces/sql_dialect.hpp`; `sqlite_dialect` is the dialect of `memory_sql_connection`
/// (`interfaces/sql_connection_reference.hpp`). Group: \ref core.
///
/// **Implementations / variants.** The two above.
///
/// **Guarantees.** Stateless, empty, trivially copyable; safe from any thread.
///
/// **Errors.** None.
///
/// **Performance.** Not benchmarked.
///
/// **Tests and demos.** `tests/interfaces/sql_dialect_reference_test.cpp`,
/// `demos/interfaces/sql_dialect_reference_demo.cpp`.
///
/// **Dependencies.** `<cstddef>`, `<optional>`, `<string>`, `<string_view>`, `interfaces/sql_dialect.hpp`.
///
/// **Known issues.** PostgreSQL has a native `UUID` type and `NUMERIC`; these dialects use `BYTEA` and `TEXT` so that
/// the stored bytes and order are the same on both engines (crtp's open item on a native UUID stays open).
///
/// **Status.** New in 0.27.0.

#ifndef snicholls_core_interfaces_sql_dialect_reference_hpp
#define snicholls_core_interfaces_sql_dialect_reference_hpp

#include <snicholls/core/interfaces/sql_dialect.hpp>

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>

namespace snicholls {

    /// \brief SQLite's dialect.
    /// \ingroup core
    struct sqlite_dialect {
        /// \brief The engine's name.
        /// \return `"sqlite"`.
        [[nodiscard]] constexpr std::string_view name() const noexcept { return "sqlite"; }
        /// \brief SQLite's name for a type (its storage classes).
        /// \param t The type.
        /// \return It.
        [[nodiscard]] constexpr std::optional<std::string_view> column_type(sql_type t) const noexcept
        {
            switch (t) {
            case sql_type::boolean:
            case sql_type::int64:
            case sql_type::timestamp: return "INTEGER";
            case sql_type::float64: return "REAL";
            case sql_type::text:
            case sql_type::decimal: return "TEXT";
            case sql_type::blob:
            case sql_type::uuid: return "BLOB";
            }
            return std::nullopt;
        }
        /// \brief Appends `id`, quoted.
        /// \param out Where to append.
        /// \param id  The identifier.
        /// \return `false` for an empty identifier or one with a NUL.
        bool append_identifier(std::string& out, std::string_view id) const { return append_quoted_identifier(out, id); }
        /// \brief Appends `?n`.
        /// \param out Where to append.
        /// \param n   The parameter's number.
        void append_placeholder(std::string& out, std::size_t n) const
        {
            out += '?';
            out += std::to_string(n);
        }
        /// \brief `BEGIN IMMEDIATE`: the write lock at once, so a transaction never fails half-way on `busy`.
        /// \return It.
        [[nodiscard]] constexpr std::string_view begin_transaction() const noexcept { return "BEGIN IMMEDIATE"; }
        /// \brief ` WITHOUT ROWID`: rows stored in key order.
        /// \return It.
        [[nodiscard]] constexpr std::string_view clustered_suffix() const noexcept { return " WITHOUT ROWID"; }
    };

    /// \brief PostgreSQL's dialect.
    /// \ingroup core
    struct postgres_dialect {
        /// \brief The engine's name.
        /// \return `"postgres"`.
        [[nodiscard]] constexpr std::string_view name() const noexcept { return "postgres"; }
        /// \brief PostgreSQL's name for a type.
        /// \param t The type.
        /// \return It.
        [[nodiscard]] constexpr std::optional<std::string_view> column_type(sql_type t) const noexcept
        {
            switch (t) {
            case sql_type::boolean: return "BOOLEAN";
            case sql_type::int64:
            case sql_type::timestamp: return "BIGINT";
            case sql_type::float64: return "DOUBLE PRECISION";
            case sql_type::text:
            case sql_type::decimal: return "TEXT";
            case sql_type::blob:
            case sql_type::uuid: return "BYTEA";
            }
            return std::nullopt;
        }
        /// \brief Appends `id`, quoted.
        /// \param out Where to append.
        /// \param id  The identifier.
        /// \return `false` for an empty identifier or one with a NUL.
        bool append_identifier(std::string& out, std::string_view id) const { return append_quoted_identifier(out, id); }
        /// \brief Appends `$n`.
        /// \param out Where to append.
        /// \param n   The parameter's number.
        void append_placeholder(std::string& out, std::size_t n) const
        {
            out += '$';
            out += std::to_string(n);
        }
        /// \brief `BEGIN`.
        /// \return It.
        [[nodiscard]] constexpr std::string_view begin_transaction() const noexcept { return "BEGIN"; }
        /// \brief None: PostgreSQL has no clustered storage clause.
        /// \return Empty.
        [[nodiscard]] constexpr std::string_view clustered_suffix() const noexcept { return {}; }
    };

    static_assert(sql_dialect<sqlite_dialect>, "snicholls::sqlite_dialect must model snicholls::sql_dialect");
    static_assert(sql_dialect<postgres_dialect>, "snicholls::postgres_dialect must model snicholls::sql_dialect");

} // namespace snicholls

#endif // snicholls_core_interfaces_sql_dialect_reference_hpp
