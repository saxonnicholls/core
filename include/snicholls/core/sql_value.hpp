//
//  sql_value.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  The values and errors of crtp's SQL interfaces (interfaces/sql/sql_value.hpp, sql_error.hpp; SQL_PLAN.md D4, D7):
//  six parameter kinds as views, rows visited with typed getters, database-neutral error codes. Getters return a
//  result instead of throwing; the mappings of core's own types (utc, uuid, decimal, money) are here, the mappings of
//  the libraries' types (crtp's column_traits) stay with those types.
//

/// \file
/// \brief SQL values: parameters, rows, column types, errors, and the mappings of core's types.
/// \ingroup core
///
/// **Overview.** What crosses the SQL contract (`concepts/sql_dialect.hpp`, `concepts/sql_connection.hpp`):
/// - `sql_param`: a parameter, one of the six storage classes every engine shares: `sql_null`, `bool`,
///   `std::int64_t`, `double`, text (`std::string_view`) and blob (`std::span<const std::byte>`). Text and blob are
///   **views** of the caller's bytes, valid for the call that receives them: values cross as typed parameters,
///   never as SQL text, so there is no quoting and no injection. `sql_text(s)` and `sql_blob(b)` make them explicitly;
///   `sql_value` is the owning twin, and `view_of` / `to_value` convert.
/// - `sql_row`: one result row, seen by a query's visitor (valid during the call), with typed getters that convert
///   exactly or return `type_mismatch` / `out_of_range`, never a silent truncation.
/// - `sql_type`: a column's logical type, which a dialect names (`INTEGER`, `BIGINT`, `BYTEA`, ...).
/// - `sql_errc` and `sql_error`: database-neutral codes, the SQLSTATE when the engine has one, a fixed message
///   (`capability_message`: a literal, so no parameter value, password or row can reach an error), and
///   `may_retry(code, policy)`.
///
/// Core's types map onto the storage classes exactly: `utc<D>` is `int64` ticks of `D` (8 bytes, sortable, before
/// 1970 too); `uuid` is a 16-byte blob (bytewise order is RFC order, so version 7 sorts by time); `decimal` is its
/// canonical text; `money` is its exact text at all of its asset's digits, the asset (and so the scale) being the
/// column's. `to_sql` encodes, and `utc_from_sql`, `uuid_from_sql`, `decimal_from_sql` and `money_from_sql` decode.
///
/// \snippet sql_value_demo.cpp values
///
/// **Relationships.** `concepts/sql_dialect.hpp`, `concepts/sql_connection.hpp`; `time/utc.hpp`, `uuid.hpp`,
/// `decimal.hpp`, `money/money.hpp`. Group: \ref core.
///
/// **Implementations / variants.** None: values.
///
/// **Guarantees.** `kind_of`, the error helpers and the integer and time mappings are `constexpr`; nothing here
/// allocates except `to_value` (owning copies). A `sql_param` is trivially copyable.
///
/// **Errors.** The decoders: `type_mismatch` for the wrong storage class or text that does not parse, `out_of_range`
/// for a value the target cannot hold. `sql_row`'s getters: the same, and `misuse` for a column index past the row.
///
/// **Performance.** Not benchmarked: views and integer copies.
///
/// **Tests and demos.** `tests/sql_value_test.cpp`, `demos/sql_value_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<cstddef>`, `<cstdint>`, `<limits>`, `<span>`, `<string>`, `<string_view>`,
/// `<variant>`, `<vector>`, `capability_error.hpp` (`capability_message`), `containers/inplace_string.hpp`,
/// `decimal.hpp`, `money/money.hpp`, `result.hpp`, `time/utc.hpp`, `uuid.hpp`.
///
/// **Known issues.** A `uuid`'s blob view points into the `uuid`: keep it alive for the call. C++ types of the
/// libraries above core (crtp's objects, enumerations, durations) map through their own column traits, not here
/// (type support, LIBRARIES section 5).
///
/// **Status.** New in 0.27.0. Experimental, with the SQL contract.

#ifndef snicholls_core_sql_value_hpp
#define snicholls_core_sql_value_hpp

#include <snicholls/core/capability_error.hpp>
#include <snicholls/core/containers/inplace_string.hpp>
#include <snicholls/core/decimal.hpp>
#include <snicholls/core/money/money.hpp>
#include <snicholls/core/result.hpp>
#include <snicholls/core/time/utc.hpp>
#include <snicholls/core/uuid.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace snicholls {

    /// \brief Why an SQL operation failed: database-neutral (SQLite maps its result codes, PostgreSQL its SQLSTATE).
    /// \ingroup core
    enum class sql_errc : std::uint8_t {
        connection_failed = 1,      ///< could not connect: host, authentication, TLS, a bad connection string
        connection_lost = 2,        ///< the connection broke during an operation; an unfinished commit's outcome is unknown
        timeout = 3,                ///< a statement, lock or pool wait timed out, or the statement was cancelled
        busy = 4,                   ///< locked by another connection; nothing was applied
        serialization_failure = 5,  ///< a serialization failure or deadlock; the transaction was rolled back
        unique_violation = 6,       ///< a duplicate primary or unique key
        constraint_violation = 7,   ///< a CHECK, NOT NULL, foreign key or domain constraint
        syntax_error = 8,           ///< the SQL text does not parse (or the engine does not understand it)
        undefined_object = 9,       ///< no such table, column, function or type
        type_mismatch = 10,         ///< a value does not convert to or from the column's type
        out_of_range = 11,          ///< a number does not fit the target type
        permission_denied = 12,     ///< privileges, a read-only database or file
        storage_full = 13,          ///< disk or quota full
        corrupt = 14,               ///< the database or a stored value is damaged
        misuse = 15,                ///< the wrong parameter count, a nested transaction, a closed connection, a bad index
        configuration = 16,         ///< missing or invalid configuration (a database name, a path, an option)
        unsupported = 17,           ///< the engine cannot do this
        internal = 18,              ///< anything else the engine reports
    };

    /// \brief When a failed SQL operation may be run again.
    /// \ingroup core
    enum class sql_retry : std::uint8_t {
        never,       ///< never
        safe,        ///< only when nothing was applied: `connection_failed`, `busy`, `serialization_failure`
        idempotent,  ///< also after `connection_lost` and `timeout`: reads, upserts, DDL
    };

    /// \brief Whether an operation that failed with `code` may be run again under `policy`.
    /// \ingroup core
    /// \param code   The failure.
    /// \param policy The caller's policy for this operation.
    /// \return `true` when a retry is allowed.
    [[nodiscard]] constexpr bool may_retry(sql_errc code, sql_retry policy) noexcept
    {
        if (policy == sql_retry::never)
            return false;
        if (code == sql_errc::connection_failed || code == sql_errc::busy || code == sql_errc::serialization_failure)
            return true;
        return policy == sql_retry::idempotent && (code == sql_errc::connection_lost || code == sql_errc::timeout);
    }

    /// \brief An SQL failure: the code, the engine's SQLSTATE when it has one, a fixed message.
    /// \ingroup core
    struct sql_error {
        sql_errc code{sql_errc::internal};  ///< what a caller branches on
        inplace_string<5> sqlstate{};       ///< `"23505"`, or empty
        capability_message message{};       ///< a literal: never a value, a row or a password

        /// \brief Equality, member by member.
        friend constexpr bool operator==(const sql_error&, const sql_error&) noexcept = default;
    };

    /// \brief The result of every SQL operation.
    /// \ingroup core
    /// \tparam V The value.
    template <typename V>
    using sql_result = result<V, sql_error>;

    /// \brief An error result: `return sql_fail(sql_errc::misuse, "a nested transaction");`.
    /// \ingroup core
    /// \param code     The code.
    /// \param message  A fixed diagnostic.
    /// \param sqlstate The engine's SQLSTATE, if any (five characters, else dropped).
    /// \return The error, ready to convert into any `sql_result<V>`.
    [[nodiscard]] constexpr auto sql_fail(sql_errc code, capability_message message = {}, std::string_view sqlstate = {}) noexcept
    {
        sql_error e{code, {}, message};
        if (sqlstate.size() == 5)
            if (auto s = inplace_string<5>::make(sqlstate); s)
                e.sqlstate = *s;
        return err(e);
    }

    /// \brief The type of SQL NULL.
    /// \ingroup core
    struct sql_null_t {
        /// \brief Every NULL is the same parameter.
        friend constexpr bool operator==(sql_null_t, sql_null_t) noexcept = default;
    };

    /// \brief SQL NULL.
    /// \ingroup core
    inline constexpr sql_null_t sql_null{};

    /// \brief A parameter: NULL, `bool`, `std::int64_t`, `double`, text or blob (views, valid for the call).
    /// \ingroup core
    using sql_param = std::variant<sql_null_t, bool, std::int64_t, double, std::string_view, std::span<const std::byte>>;

    /// \brief An owned value of the same six kinds: what a store keeps, or a row copied out of a visitor.
    /// \ingroup core
    using sql_value = std::variant<sql_null_t, bool, std::int64_t, double, std::string, std::vector<std::byte>>;

    /// \brief The storage class of a parameter or value, in variant order.
    /// \ingroup core
    enum class sql_kind : std::uint8_t {
        null,     ///< SQL NULL
        boolean,  ///< `bool`
        int64,    ///< `std::int64_t`
        float64,  ///< `double`
        text,     ///< UTF-8 text
        blob,     ///< bytes
    };

    /// \brief The storage class of `p`.
    /// \ingroup core
    /// \param p The parameter.
    /// \return Its kind.
    [[nodiscard]] constexpr sql_kind kind_of(const sql_param& p) noexcept { return static_cast<sql_kind>(p.index()); }

    /// \brief The storage class of `v`.
    /// \ingroup core
    /// \param v The value.
    /// \return Its kind.
    [[nodiscard]] constexpr sql_kind kind_of(const sql_value& v) noexcept { return static_cast<sql_kind>(v.index()); }

    /// \brief A text parameter (a view of `s`).
    /// \ingroup core
    /// \param s The text.
    /// \return The parameter.
    [[nodiscard]] constexpr sql_param sql_text(std::string_view s) noexcept { return sql_param(std::in_place_index<4>, s); }

    /// \brief A blob parameter (a view of `b`).
    /// \ingroup core
    /// \param b The bytes.
    /// \return The parameter.
    [[nodiscard]] constexpr sql_param sql_blob(std::span<const std::byte> b) noexcept { return sql_param(std::in_place_index<5>, b); }

    /// \brief A view of an owned value, valid while `v` is unchanged.
    /// \ingroup core
    /// \param v The value.
    /// \return The parameter.
    [[nodiscard]] inline sql_param view_of(const sql_value& v) noexcept
    {
        switch (kind_of(v)) {
        case sql_kind::boolean: return std::get<1>(v);
        case sql_kind::int64: return std::get<2>(v);
        case sql_kind::float64: return std::get<3>(v);
        case sql_kind::text: return sql_text(std::get<4>(v));
        case sql_kind::blob: return sql_blob(std::get<5>(v));
        case sql_kind::null: break;
        }
        return sql_null;
    }

    /// \brief An owned copy of a parameter.
    /// \ingroup core
    /// \param p The parameter.
    /// \return The value.
    [[nodiscard]] inline sql_value to_value(const sql_param& p)
    {
        switch (kind_of(p)) {
        case sql_kind::boolean: return std::get<1>(p);
        case sql_kind::int64: return std::get<2>(p);
        case sql_kind::float64: return std::get<3>(p);
        case sql_kind::text: return std::string(std::get<4>(p));
        case sql_kind::blob: {
            const auto b = std::get<5>(p);
            return std::vector<std::byte>(b.begin(), b.end());
        }
        case sql_kind::null: break;
        }
        return sql_null;
    }

    /// \brief A column's logical type, which a dialect names.
    /// \ingroup core
    enum class sql_type : std::uint8_t {
        boolean,    ///< true or false
        int64,      ///< a signed 64-bit integer
        float64,    ///< an IEEE 754 double
        text,       ///< UTF-8 text
        blob,       ///< bytes
        timestamp,  ///< a `utc<D>` as int64 ticks
        uuid,       ///< a `uuid` as 16 bytes
        decimal,    ///< a `decimal` or a `money` as exact text
    };

    /// \brief Every `sql_type`, in declaration order.
    /// \ingroup core
    inline constexpr std::array<sql_type, 8> sql_types{sql_type::boolean, sql_type::int64,     sql_type::float64, sql_type::text,
                                                       sql_type::blob,    sql_type::timestamp, sql_type::uuid,    sql_type::decimal};

    /// \brief The storage class a column of `t` holds.
    /// \ingroup core
    /// \param t The type.
    /// \return Its kind.
    [[nodiscard]] constexpr sql_kind storage_of(sql_type t) noexcept
    {
        switch (t) {
        case sql_type::boolean: return sql_kind::boolean;
        case sql_type::int64:
        case sql_type::timestamp: return sql_kind::int64;
        case sql_type::float64: return sql_kind::float64;
        case sql_type::text:
        case sql_type::decimal: return sql_kind::text;
        case sql_type::blob:
        case sql_type::uuid: return sql_kind::blob;
        }
        return sql_kind::null;
    }

    /// \brief One result row, seen by a query's visitor: column names and cells, valid during the call.
    /// \ingroup core
    class sql_row {
    public:
        /// \brief A row of `cells` named `names` (the same length).
        /// \param names The column names.
        /// \param cells The cells.
        constexpr sql_row(std::span<const std::string_view> names, std::span<const sql_param> cells) noexcept
            : names_(names), cells_(cells)
        {
        }

        /// \brief The number of columns.
        /// \return It.
        [[nodiscard]] constexpr std::size_t size() const noexcept { return cells_.size(); }
        /// \brief A column's name.
        /// \param i The column.
        /// \return Its name, or empty past the row.
        [[nodiscard]] constexpr std::string_view column_name(std::size_t i) const noexcept
        {
            return i < names_.size() ? names_[i] : std::string_view{};
        }
        /// \brief A cell, as it is.
        /// \param i The column.
        /// \return The cell, or `misuse` past the row.
        [[nodiscard]] constexpr sql_result<sql_param> cell(std::size_t i) const noexcept
        {
            if (i >= cells_.size())
                return sql_fail(sql_errc::misuse, "a column index past the row");
            return ok(cells_[i]);
        }
        /// \brief Whether a cell is NULL.
        /// \param i The column.
        /// \return `true` when NULL (or past the row).
        [[nodiscard]] constexpr bool is_null(std::size_t i) const noexcept
        {
            return i >= cells_.size() || kind_of(cells_[i]) == sql_kind::null;
        }
        /// \brief A boolean cell (an integer 0 or 1 too, as SQLite stores one).
        /// \param i The column.
        /// \return It, `type_mismatch` or `misuse`.
        [[nodiscard]] constexpr sql_result<bool> get_bool(std::size_t i) const noexcept
        {
            const auto c = cell(i);
            if (!c)
                return err(c.unwrap_err());
            if (kind_of(*c) == sql_kind::boolean)
                return ok(std::get<1>(*c));
            if (kind_of(*c) == sql_kind::int64 && (std::get<2>(*c) == 0 || std::get<2>(*c) == 1))
                return ok(std::get<2>(*c) == 1);
            return sql_fail(sql_errc::type_mismatch, "not a boolean");
        }
        /// \brief An integer cell.
        /// \param i The column.
        /// \return It, `type_mismatch` or `misuse`.
        [[nodiscard]] constexpr sql_result<std::int64_t> get_int64(std::size_t i) const noexcept
        {
            const auto c = cell(i);
            if (!c)
                return err(c.unwrap_err());
            if (kind_of(*c) == sql_kind::int64)
                return ok(std::get<2>(*c));
            if (kind_of(*c) == sql_kind::boolean)
                return ok(std::int64_t{std::get<1>(*c) ? 1 : 0});
            return sql_fail(sql_errc::type_mismatch, "not an integer");
        }
        /// \brief A floating-point cell (an integer converts when exact).
        /// \param i The column.
        /// \return It, `type_mismatch`, `out_of_range` or `misuse`.
        [[nodiscard]] constexpr sql_result<double> get_double(std::size_t i) const noexcept
        {
            const auto c = cell(i);
            if (!c)
                return err(c.unwrap_err());
            if (kind_of(*c) == sql_kind::float64)
                return ok(std::get<3>(*c));
            if (kind_of(*c) == sql_kind::int64) {
                const std::int64_t v = std::get<2>(*c);
                if (v > (std::int64_t{1} << 53) || v < -(std::int64_t{1} << 53))
                    return sql_fail(sql_errc::out_of_range, "an integer a double cannot hold exactly");
                return ok(static_cast<double>(v));
            }
            return sql_fail(sql_errc::type_mismatch, "not a number");
        }
        /// \brief A text cell, valid during the visitor.
        /// \param i The column.
        /// \return It, `type_mismatch` or `misuse`.
        [[nodiscard]] constexpr sql_result<std::string_view> get_text(std::size_t i) const noexcept
        {
            const auto c = cell(i);
            if (!c)
                return err(c.unwrap_err());
            if (kind_of(*c) == sql_kind::text)
                return ok(std::get<4>(*c));
            return sql_fail(sql_errc::type_mismatch, "not text");
        }
        /// \brief A blob cell, valid during the visitor.
        /// \param i The column.
        /// \return It, `type_mismatch` or `misuse`.
        [[nodiscard]] constexpr sql_result<std::span<const std::byte>> get_blob(std::size_t i) const noexcept
        {
            const auto c = cell(i);
            if (!c)
                return err(c.unwrap_err());
            if (kind_of(*c) == sql_kind::blob)
                return ok(std::get<5>(*c));
            return sql_fail(sql_errc::type_mismatch, "not a blob");
        }

    private:
        std::span<const std::string_view> names_;
        std::span<const sql_param> cells_;
    };

    // ---- core's types ------------------------------------------------------------------------------------------------

    /// \brief An instant as a parameter: `int64` ticks of `D` since 1970.
    /// \ingroup core
    /// \tparam D The resolution.
    /// \param t The instant.
    /// \return The parameter.
    template <utc_duration D>
    [[nodiscard]] constexpr sql_param to_sql(const utc<D>& t) noexcept
    {
        return static_cast<std::int64_t>(t.time_since_epoch().count());
    }

    /// \brief An instant from a cell of `int64` ticks of `D`.
    /// \ingroup core
    /// \tparam D The resolution the column was written in.
    /// \param p The cell.
    /// \return The instant, or `type_mismatch`.
    template <utc_duration D>
    [[nodiscard]] constexpr sql_result<utc<D>> utc_from_sql(const sql_param& p) noexcept
    {
        if (kind_of(p) != sql_kind::int64)
            return sql_fail(sql_errc::type_mismatch, "a timestamp is an integer");
        return ok(utc<D>(typename utc<D>::time_point(D(static_cast<typename D::rep>(std::get<2>(p))))));
    }

    /// \brief A UUID as a parameter: a view of its 16 bytes (keep `u` alive for the call).
    /// \ingroup core
    /// \param u The UUID.
    /// \return The parameter.
    [[nodiscard]] inline sql_param to_sql(const uuid& u) noexcept { return sql_blob(u.as_bytes()); }

    /// \brief A UUID from a 16-byte cell.
    /// \ingroup core
    /// \param p The cell.
    /// \return The UUID, `type_mismatch` or `out_of_range` (not 16 bytes).
    [[nodiscard]] constexpr sql_result<uuid> uuid_from_sql(const sql_param& p) noexcept
    {
        if (kind_of(p) != sql_kind::blob)
            return sql_fail(sql_errc::type_mismatch, "a uuid is a blob");
        const auto b = std::get<5>(p);
        if (b.size() != 16)
            return sql_fail(sql_errc::out_of_range, "a uuid is 16 bytes");
        uuid u;
        for (std::size_t i = 0; i < 16; ++i)
            u.bytes[i] = static_cast<std::uint8_t>(b[i]);
        return ok(u);
    }

    /// \brief A decimal's canonical text, as a parameter (a view of `t`: keep it alive for the call).
    /// \ingroup core
    /// \param t The text, from `decimal::text()`.
    /// \return The parameter.
    [[nodiscard]] constexpr sql_param to_sql(const decimal_text& t) noexcept { return sql_text(t.view()); }

    /// \brief A decimal from a text cell.
    /// \ingroup core
    /// \param p The cell.
    /// \return The decimal, or `type_mismatch` (not text, or not a decimal).
    [[nodiscard]] constexpr sql_result<decimal> decimal_from_sql(const sql_param& p) noexcept
    {
        if (kind_of(p) != sql_kind::text)
            return sql_fail(sql_errc::type_mismatch, "a decimal is text");
        const auto d = decimal::parse(std::get<4>(p));
        if (!d)
            return sql_fail(sql_errc::type_mismatch, "the text is not a decimal");
        return ok(*d);
    }

    /// \brief A money amount's exact text, at all of its asset's digits: `"-1000.50"` for AUD.
    /// \ingroup core
    struct sql_money_text {
        std::array<char, 82> data{};  ///< the characters
        std::size_t size = 0;         ///< how many are used

        /// \brief The text.
        /// \return A view of it.
        [[nodiscard]] constexpr std::string_view view() const noexcept { return {data.data(), size}; }
    };

    /// \brief The text of `m` that a `sql_type::decimal` column holds; the asset is the column's.
    /// \ingroup core
    /// \param m The amount.
    /// \return The text.
    [[nodiscard]] constexpr sql_money_text money_text(const money& m) noexcept
    {
        sql_money_text t;
        const auto r = to_chars(t.data.data(), t.data.data() + t.data.size(), m, money_digits::all);
        t.size = r.ec == std::errc{} ? static_cast<std::size_t>(r.ptr - t.data.data()) : 0;
        return t;
    }

    /// \brief A money amount's text as a parameter (a view of `t`: keep it alive for the call).
    /// \ingroup core
    /// \param t The text, from `money_text`.
    /// \return The parameter.
    [[nodiscard]] constexpr sql_param to_sql(const sql_money_text& t) noexcept { return sql_text(t.view()); }

    /// \brief An amount of `a` from a text cell.
    /// \ingroup core
    /// \param a The column's asset.
    /// \param p The cell.
    /// \return The amount, `type_mismatch` (not text or not a number) or `out_of_range` (more digits than the scale,
    /// or too large).
    [[nodiscard]] constexpr sql_result<money> money_from_sql(const asset& a, const sql_param& p) noexcept
    {
        if (kind_of(p) != sql_kind::text)
            return sql_fail(sql_errc::type_mismatch, "an amount is text");
        const auto m = parse_money(a, std::get<4>(p));
        if (m)
            return ok(*m);
        if (m.unwrap_err() == money_errc::too_many_digits || m.unwrap_err() == money_errc::overflow)
            return sql_fail(sql_errc::out_of_range, "the amount does not fit the column's asset");
        return sql_fail(sql_errc::type_mismatch, "the text is not an amount");
    }

} // namespace snicholls

#endif // snicholls_core_sql_value_hpp
