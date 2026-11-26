//
//  sql_dialect.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  The contract of crtp's sql_dialect (interfaces/sql/sql_dialect.hpp; SQL_PLAN.md D2, D3): what text an engine needs.
//  Core reduces it to the parts that differ between engines (type names, placeholders, the transaction statement, the
//  clustered-table suffix) and writes the statements once over them, in the SQL both SQLite and PostgreSQL accept.
//  crtp's engine extras (PostgreSQL domains, per-column CHECKs and COLLATE, BRIN indexes) stay crtp's.
//

/// \file
/// \brief `snicholls::sql_dialect`: the SQL text one engine needs, and the statements written once over any dialect.
/// \ingroup core
///
/// **Overview.** A dialect turns a database-neutral description of a table (`sql_table`) into the text one engine
/// understands, and executes nothing. The concept is what differs between engines:
/// - `d.name()`: `"sqlite"`, `"postgres"`;
/// - `d.column_type(t)`: the engine's name for a `sql_type` (`INTEGER`, `BIGINT`, `BYTEA`), or `std::nullopt`;
/// - `d.append_identifier(out, id)`: `id` quoted, so any name is valid and keeps its case (`"a""b"`); `false` for an
///   empty name or one with a NUL;
/// - `d.append_placeholder(out, n)`: the 1-based parameter `n` (`?1`, `$1`);
/// - `d.begin_transaction()`: the statement that starts a read-write transaction (`BEGIN`, SQLite's `BEGIN IMMEDIATE`);
/// - `d.clustered_suffix()`: what follows a table stored in primary-key order (SQLite's ` WITHOUT ROWID`).
///
/// Over it, the statements every engine runs: `sql_create_table`, `sql_drop_table`, `sql_insert` (with
/// `sql_conflict`: fail, ignore, upsert), `sql_select_rows`, `sql_count` and `sql_delete` (with `sql_select`: equality
/// on columns, a half-open range on one, an order and a limit). Parameters are numbered in the order each documents;
/// callers bind `sql_param`s in that order (`sql_value.hpp`).
///
/// \snippet concepts/sql_dialect_demo.cpp concept
///
/// **The contract** (`interfaces/sql_dialect_conformance.hpp` checks it): every `sql_type` has a name; an identifier is
/// always quoted, with an embedded quote doubled, and an empty one or one with a NUL is refused; placeholders are
/// distinct for distinct numbers; the transaction statement begins with `BEGIN`.
///
/// **Relationships.** `sql_value.hpp`; the twin `interfaces/sql_dialect.hpp`, the references
/// `interfaces/sql_dialect_reference.hpp` (`sqlite_dialect`, `postgres_dialect`), the kit
/// `interfaces/sql_dialect_conformance.hpp`; the engines' contract `concepts/sql_connection.hpp`. Group: \ref core.
///
/// **Implementations.** `sqlite_dialect` and `postgres_dialect`, standard library only and `constexpr` where the
/// language allows. Above core: crtp's dialects add their engines' extras.
///
/// **Guarantees.** Dialects are stateless and immutable, so one instance serves every connection and thread. The
/// generators allocate the text they return and nothing else.
///
/// **Errors.** The generators return `sql_errc::misuse` for a table with no name or no columns, a column index past
/// the table, an identifier that cannot be quoted, an upsert without a primary key, and a type the dialect cannot name.
///
/// **Performance.** Not benchmarked: generate a table's statements once and keep them (prepared statements are cached
/// by text).
///
/// **Tests and demos.** `tests/concepts/sql_dialect_test.cpp` (golden SQL for both dialects),
/// `demos/concepts/sql_dialect_demo.cpp`.
///
/// **Dependencies.** `<concepts>`, `<cstddef>`, `<optional>`, `<string>`, `<string_view>`, `<vector>`, `sql_value.hpp`.
///
/// **Known issues.** Identifiers are always quoted, so they are case-sensitive: `"Price"` and `"price"` are two
/// columns.
///
/// **Status.** New in 0.27.0. Experimental until two real implementations pass the kit.

#ifndef snicholls_core_concepts_sql_dialect_hpp
#define snicholls_core_concepts_sql_dialect_hpp

#include <snicholls/core/sql_value.hpp>

#include <concepts>
#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace snicholls {

    /// \brief A dialect: type names, identifier quoting, placeholders, the transaction statement, the clustered suffix.
    /// \ingroup core
    template <typename D>
    concept sql_dialect = requires(const D& d, std::string& out, std::string_view id, std::size_t n, sql_type t) {
        { d.name() } -> std::same_as<std::string_view>;
        { d.column_type(t) } -> std::same_as<std::optional<std::string_view>>;
        { d.append_identifier(out, id) } -> std::same_as<bool>;
        { d.append_placeholder(out, n) } -> std::same_as<void>;
        { d.begin_transaction() } -> std::same_as<std::string_view>;
        { d.clustered_suffix() } -> std::same_as<std::string_view>;
    };

    /// \brief One column of a table.
    /// \ingroup core
    struct sql_column {
        std::string name;                 ///< the name, any text but NUL (it is always quoted)
        sql_type type{sql_type::int64};   ///< the logical type
        bool nullable{false};             ///< whether NULL is allowed (`NOT NULL` otherwise)

        /// \brief Equality.
        friend bool operator==(const sql_column&, const sql_column&) = default;
    };

    /// \brief A database-neutral table: columns, primary key, storage order.
    /// \ingroup core
    struct sql_table {
        std::string name;                      ///< the table's name
        std::vector<sql_column> columns;       ///< the columns, in order
        std::vector<std::size_t> primary_key;  ///< indexes into `columns`, in key order; may be empty
        bool clustered{false};                 ///< rows stored in primary-key order (SQLite `WITHOUT ROWID`)

        /// \brief Equality.
        friend bool operator==(const sql_table&, const sql_table&) = default;
    };

    /// \brief What an insert does when the primary key exists.
    /// \ingroup core
    enum class sql_conflict : std::uint8_t {
        fail,    ///< a plain insert: `unique_violation`
        ignore,  ///< keep the existing row: `ON CONFLICT DO NOTHING`, 0 rows changed
        update,  ///< an upsert: overwrite the other columns (`ON CONFLICT (key) DO UPDATE`)
    };

    /// \brief A WHERE, ORDER BY and LIMIT over one table's columns (indexes into `sql_table::columns`).
    /// \ingroup core
    ///
    /// Parameters: one per `equal` column, in order; then the range's lower bound if `has_from`; then its upper bound
    /// if `has_to` (`from <= column < to`).
    struct sql_select {
        std::vector<std::size_t> equal;           ///< `column = ?` for each
        std::optional<std::size_t> range_column;  ///< the column of the range, if any
        bool has_from{false};                     ///< `range_column >= ?`
        bool has_to{false};                       ///< `range_column < ?`
        std::vector<std::size_t> order_by;        ///< `ORDER BY` these columns (none: unspecified order)
        bool descending{false};                   ///< every `order_by` column descending
        std::size_t limit{0};                     ///< `LIMIT`; 0 is none
    };

    namespace detail {
        template <sql_dialect D>
        sql_result<void> sql_append_id(const D& d, std::string& out, std::string_view id)
        {
            if (!d.append_identifier(out, id))
                return sql_fail(sql_errc::misuse, "an identifier that cannot be quoted");
            return ok();
        }

        inline sql_result<void> sql_check_table(const sql_table& t)
        {
            if (t.name.empty() || t.columns.empty())
                return sql_fail(sql_errc::misuse, "a table needs a name and columns");
            for (const std::size_t k : t.primary_key)
                if (k >= t.columns.size())
                    return sql_fail(sql_errc::misuse, "a key column past the table");
            return ok();
        }

        // " WHERE ... ORDER BY ... LIMIT n", numbering placeholders from `n`.
        template <sql_dialect D>
        sql_result<void> sql_append_where(const D& d, std::string& out, const sql_table& t, const sql_select& w, bool order)
        {
            const auto col = [&](std::size_t i) -> sql_result<void> {
                if (i >= t.columns.size())
                    return sql_fail(sql_errc::misuse, "a column index past the table");
                return sql_append_id(d, out, t.columns[i].name);
            };
            std::size_t n = 1;
            bool first = true;
            const auto conjunct = [&] {
                out += first ? " WHERE " : " AND ";
                first = false;
            };
            for (const std::size_t i : w.equal) {
                conjunct();
                SNICHOLLS_TRY_VOID(col(i));
                out += " = ";
                d.append_placeholder(out, n++);
            }
            if ((w.has_from || w.has_to) && !w.range_column)
                return sql_fail(sql_errc::misuse, "a range without a column");
            if (w.has_from) {
                conjunct();
                SNICHOLLS_TRY_VOID(col(*w.range_column));
                out += " >= ";
                d.append_placeholder(out, n++);
            }
            if (w.has_to) {
                conjunct();
                SNICHOLLS_TRY_VOID(col(*w.range_column));
                out += " < ";
                d.append_placeholder(out, n++);
            }
            if (!order)
                return ok();
            for (std::size_t k = 0; k < w.order_by.size(); ++k) {
                out += k == 0 ? " ORDER BY " : ", ";
                SNICHOLLS_TRY_VOID(col(w.order_by[k]));
                if (w.descending)
                    out += " DESC";
            }
            if (w.limit != 0)
                out += " LIMIT " + std::to_string(w.limit);
            return ok();
        }
    } // namespace detail

    /// \brief `CREATE TABLE IF NOT EXISTS` for `t`: idempotent.
    /// \ingroup core
    /// \param d The dialect.
    /// \param t The table.
    /// \return The statement, or `misuse`.
    template <sql_dialect D>
    [[nodiscard]] sql_result<std::string> sql_create_table(const D& d, const sql_table& t)
    {
        SNICHOLLS_TRY_VOID(detail::sql_check_table(t));
        std::string out = "CREATE TABLE IF NOT EXISTS ";
        SNICHOLLS_TRY_VOID(detail::sql_append_id(d, out, t.name));
        out += " (";
        for (std::size_t i = 0; i < t.columns.size(); ++i) {
            if (i != 0)
                out += ", ";
            SNICHOLLS_TRY_VOID(detail::sql_append_id(d, out, t.columns[i].name));
            const auto type = d.column_type(t.columns[i].type);
            if (!type)
                return sql_fail(sql_errc::misuse, "a type the dialect cannot name");
            out += ' ';
            out += *type;
            if (!t.columns[i].nullable)
                out += " NOT NULL";
        }
        if (!t.primary_key.empty()) {
            out += ", PRIMARY KEY (";
            for (std::size_t k = 0; k < t.primary_key.size(); ++k) {
                if (k != 0)
                    out += ", ";
                SNICHOLLS_TRY_VOID(detail::sql_append_id(d, out, t.columns[t.primary_key[k]].name));
            }
            out += ')';
        }
        out += ')';
        if (t.clustered && !t.primary_key.empty())
            out += d.clustered_suffix();
        return ok(std::move(out));
    }

    /// \brief `DROP TABLE`.
    /// \ingroup core
    /// \param d         The dialect.
    /// \param table     The table's name.
    /// \param if_exists Whether a missing table is not an error.
    /// \return The statement, or `misuse`.
    template <sql_dialect D>
    [[nodiscard]] sql_result<std::string> sql_drop_table(const D& d, std::string_view table, bool if_exists = true)
    {
        std::string out = if_exists ? "DROP TABLE IF EXISTS " : "DROP TABLE ";
        SNICHOLLS_TRY_VOID(detail::sql_append_id(d, out, table));
        return ok(std::move(out));
    }

    /// \brief `INSERT` of one row; the parameters are the columns in table order.
    /// \ingroup core
    /// \param d        The dialect.
    /// \param t        The table.
    /// \param conflict What an existing key does.
    /// \return The statement, or `misuse` (`ignore` and `update` need a primary key).
    template <sql_dialect D>
    [[nodiscard]] sql_result<std::string> sql_insert(const D& d, const sql_table& t, sql_conflict conflict = sql_conflict::fail)
    {
        SNICHOLLS_TRY_VOID(detail::sql_check_table(t));
        if (conflict != sql_conflict::fail && t.primary_key.empty())
            return sql_fail(sql_errc::misuse, "a conflict clause needs a primary key");
        std::string out = "INSERT INTO ";
        SNICHOLLS_TRY_VOID(detail::sql_append_id(d, out, t.name));
        out += " (";
        for (std::size_t i = 0; i < t.columns.size(); ++i) {
            if (i != 0)
                out += ", ";
            SNICHOLLS_TRY_VOID(detail::sql_append_id(d, out, t.columns[i].name));
        }
        out += ") VALUES (";
        for (std::size_t i = 0; i < t.columns.size(); ++i) {
            if (i != 0)
                out += ", ";
            d.append_placeholder(out, i + 1);
        }
        out += ')';
        if (conflict == sql_conflict::fail)
            return ok(std::move(out));
        out += " ON CONFLICT (";
        for (std::size_t k = 0; k < t.primary_key.size(); ++k) {
            if (k != 0)
                out += ", ";
            SNICHOLLS_TRY_VOID(detail::sql_append_id(d, out, t.columns[t.primary_key[k]].name));
        }
        out += ')';
        bool any = false;
        if (conflict == sql_conflict::update)
            for (std::size_t i = 0; i < t.columns.size(); ++i) {
                bool key = false;
                for (const std::size_t k : t.primary_key)
                    key = key || k == i;
                if (key)
                    continue;
                out += any ? ", " : " DO UPDATE SET ";
                any = true;
                SNICHOLLS_TRY_VOID(detail::sql_append_id(d, out, t.columns[i].name));
                out += " = excluded.";
                SNICHOLLS_TRY_VOID(detail::sql_append_id(d, out, t.columns[i].name));
            }
        if (!any)
            out += " DO NOTHING";
        return ok(std::move(out));
    }

    /// \brief `SELECT` every column, in table order, as `where` describes.
    /// \ingroup core
    /// \param d     The dialect.
    /// \param t     The table.
    /// \param where The selection.
    /// \return The statement, or `misuse`.
    template <sql_dialect D>
    [[nodiscard]] sql_result<std::string> sql_select_rows(const D& d, const sql_table& t, const sql_select& where = {})
    {
        SNICHOLLS_TRY_VOID(detail::sql_check_table(t));
        std::string out = "SELECT ";
        for (std::size_t i = 0; i < t.columns.size(); ++i) {
            if (i != 0)
                out += ", ";
            SNICHOLLS_TRY_VOID(detail::sql_append_id(d, out, t.columns[i].name));
        }
        out += " FROM ";
        SNICHOLLS_TRY_VOID(detail::sql_append_id(d, out, t.name));
        SNICHOLLS_TRY_VOID(detail::sql_append_where(d, out, t, where, true));
        return ok(std::move(out));
    }

    /// \brief `SELECT count(*)` with `where`'s conditions (its order and limit are ignored).
    /// \ingroup core
    /// \param d     The dialect.
    /// \param t     The table.
    /// \param where The selection.
    /// \return The statement, or `misuse`.
    template <sql_dialect D>
    [[nodiscard]] sql_result<std::string> sql_count(const D& d, const sql_table& t, const sql_select& where = {})
    {
        SNICHOLLS_TRY_VOID(detail::sql_check_table(t));
        std::string out = "SELECT count(*) FROM ";
        SNICHOLLS_TRY_VOID(detail::sql_append_id(d, out, t.name));
        SNICHOLLS_TRY_VOID(detail::sql_append_where(d, out, t, where, false));
        return ok(std::move(out));
    }

    /// \brief `DELETE` with `where`'s conditions (its order and limit are ignored).
    /// \ingroup core
    /// \param d     The dialect.
    /// \param t     The table.
    /// \param where The selection; empty deletes every row.
    /// \return The statement, or `misuse`.
    template <sql_dialect D>
    [[nodiscard]] sql_result<std::string> sql_delete(const D& d, const sql_table& t, const sql_select& where = {})
    {
        SNICHOLLS_TRY_VOID(detail::sql_check_table(t));
        std::string out = "DELETE FROM ";
        SNICHOLLS_TRY_VOID(detail::sql_append_id(d, out, t.name));
        SNICHOLLS_TRY_VOID(detail::sql_append_where(d, out, t, where, false));
        return ok(std::move(out));
    }

    /// \brief The standard quoting both engines share: `"` around `id`, each `"` inside doubled.
    /// \ingroup core
    /// \param out Where to append.
    /// \param id  The identifier.
    /// \return `false` (and nothing appended) for an empty identifier or one with a NUL.
    inline bool append_quoted_identifier(std::string& out, std::string_view id)
    {
        if (id.empty() || id.find('\0') != std::string_view::npos)
            return false;
        out += '"';
        for (const char c : id) {
            if (c == '"')
                out += '"';
            out += c;
        }
        out += '"';
        return true;
    }

} // namespace snicholls

#endif // snicholls_core_concepts_sql_dialect_hpp
