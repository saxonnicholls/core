//
//  sql_connection.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  The contract of crtp's sql_connection (interfaces/sql/sql_connection.hpp; SQL_PLAN.md D2, D5, D6, D7): prepared
//  statements, typed parameters, visited rows, transactions and batches, with every failure a result (crtp throws
//  sql_error). crtp's pool, retries, statistics, health, bulk COPY, drivers and configuration stay crtp's.
//

/// \file
/// \brief `snicholls::sql_connection`: one connection to an SQL engine; statements, queries, transactions, all returning
/// `sql_result`.
/// \ingroup core
///
/// **Overview.** The engine half of the SQL contract (the text half is `concepts/sql_dialect.hpp`):
/// - `c.dialect()`: the engine's dialect, so engine-neutral code generates its SQL through it and runs it here;
/// - `c.prepare(sql)`: parse and cache a statement (statements are cached by text, so this is optional: it surfaces
///   a `syntax_error` or `undefined_object` before any data is bound);
/// - `c.execute(sql, params)`: run one statement, returning the rows it changed;
/// - `c.query(sql, params, visitor)`: run a query and visit its rows in order (`sql_row`, valid during the call); the
///   visitor returns `false` to stop; the result is how many rows it saw;
/// - `c.execute_batch(sql, width, rows, fill)`: one statement for `rows` rows of `width` parameters, which `fill(row,
///   params)` writes; all or nothing;
/// - `c.begin()`, `c.commit()`, `c.rollback()`, `c.in_transaction()`: one transaction at a time;
/// - `c.ping()` and `c.is_open()`.
///
/// Over it: `sql_execute(c, sql, {params...})` and `sql_query(c, sql, {params...}, f)` (a visitor returning `void` or
/// `bool`), and `sql_transaction<C>`, which rolls back unless committed.
///
/// \snippet concepts/sql_connection_demo.cpp concept
///
/// **The contract** (`interfaces/sql_connection_conformance.hpp` checks it): values round-trip exactly through the
/// six kinds and core's types; a duplicate key is `unique_violation` and a conflict clause resolves it; a rollback
/// discards and a commit keeps; a nested `begin` and a `commit` without one are `misuse`; the wrong number of
/// parameters is `misuse`; an unknown table is `undefined_object`; text the engine does not understand is
/// `syntax_error`; a batch that fails changes nothing; ranges, orders and limits select what the dialect's SQL says;
/// and no error carries a parameter's value (`sql_error::message` is a literal).
///
/// **Relationships.** `sql_value.hpp`, `concepts/sql_dialect.hpp`; the twin `interfaces/sql_connection.hpp`, the
/// reference `interfaces/sql_connection_reference.hpp` (`memory_sql_connection`), the kit
/// `interfaces/sql_connection_conformance.hpp`. Group: \ref core.
///
/// **Implementations.** `memory_sql_connection`: an in-memory table store that runs exactly the statements core's
/// generators write (a test double, not an SQL engine). Above core, in crtp: SQLite and PostgreSQL (libpq).
///
/// **Guarantees.** A connection is used by one thread at a time (as `sqlite3*` and `PGconn`); a pool above core
/// shares them. A visitor must not use the connection it is called from.
///
/// **Errors.** `sql_error`, as the contract says; nothing is thrown but `std::bad_alloc`.
///
/// **Performance.** Not benchmarked here: crtp benchmarks its engines against hand-written sqlite3 and libpq.
///
/// **Tests and demos.** `tests/concepts/sql_connection_test.cpp`, `demos/concepts/sql_connection_demo.cpp`.
///
/// **Dependencies.** `<concepts>`, `<cstddef>`, `<initializer_list>`, `<span>`, `<string_view>`, `<type_traits>`,
/// `<utility>`, `concepts/sql_dialect.hpp`, `function_ref.hpp`, `sql_value.hpp`.
///
/// **Known issues.** No bulk insert (COPY) in the contract: `execute_batch` is the portable form, and an engine's
/// faster path stays its own.
///
/// **Status.** New in 0.27.0. Experimental until two real implementations pass the kit.

#ifndef snicholls_core_concepts_sql_connection_hpp
#define snicholls_core_concepts_sql_connection_hpp

#include <snicholls/core/concepts/sql_dialect.hpp>
#include <snicholls/core/function_ref.hpp>
#include <snicholls/core/sql_value.hpp>

#include <concepts>
#include <cstddef>
#include <initializer_list>
#include <span>
#include <string_view>
#include <type_traits>
#include <utility>

namespace snicholls {

    /// \brief A query's visitor: called with each row in order; `false` stops.
    /// \ingroup core
    using sql_row_visitor = function_ref<bool(const sql_row&)>;

    /// \brief A batch's filler: writes the parameters (`size()` of them) of row `row`. Views written must stay valid
    /// until the next call.
    /// \ingroup core
    using sql_row_filler = function_ref<void(std::size_t, std::span<sql_param>)>;

    /// \brief A connection: `dialect`, `prepare`, `execute`, `query`, `execute_batch`, the transaction and the
    /// lifecycle.
    /// \ingroup core
    template <typename C>
    concept sql_connection = requires(C& c, const C& cc, std::string_view sql, std::span<const sql_param> params,
                                      sql_row_visitor visit, sql_row_filler fill, std::size_t n) {
        requires sql_dialect<std::remove_cvref_t<decltype(cc.dialect())>>;
        { c.prepare(sql) } -> std::same_as<sql_result<void>>;
        { c.execute(sql, params) } -> std::same_as<sql_result<std::size_t>>;
        { c.query(sql, params, visit) } -> std::same_as<sql_result<std::size_t>>;
        { c.execute_batch(sql, n, n, fill) } -> std::same_as<sql_result<std::size_t>>;
        { c.begin() } -> std::same_as<sql_result<void>>;
        { c.commit() } -> std::same_as<sql_result<void>>;
        { c.rollback() } -> std::same_as<sql_result<void>>;
        { cc.in_transaction() } noexcept -> std::same_as<bool>;
        { c.ping() } -> std::same_as<sql_result<void>>;
        { cc.is_open() } noexcept -> std::same_as<bool>;
    };

    /// \brief Runs one statement with the parameters written in place: `sql_execute(c, sql, {sql_text("a"), 1})`.
    /// \ingroup core
    /// \param c      The connection.
    /// \param sql    The statement.
    /// \param params The parameters.
    /// \return The rows changed, or the error.
    template <sql_connection C>
    sql_result<std::size_t> sql_execute(C& c, std::string_view sql, std::initializer_list<sql_param> params = {})
    {
        return c.execute(sql, std::span<const sql_param>(params.begin(), params.size()));
    }

    /// \brief Runs a query whose visitor returns `void` (every row is seen).
    /// \ingroup core
    /// \param c      The connection.
    /// \param sql    The query.
    /// \param params The parameters.
    /// \param f      Called with each row.
    /// \return The rows seen, or the error.
    template <sql_connection C, typename F>
        requires std::is_invocable_r_v<void, F&, const sql_row&> && std::is_void_v<std::invoke_result_t<F&, const sql_row&>>
    sql_result<std::size_t> sql_query(C& c, std::string_view sql, std::span<const sql_param> params, F&& f)
    {
        auto visit = [&f](const sql_row& r) {
            f(r);
            return true;
        };
        return c.query(sql, params, sql_row_visitor(visit));
    }

    /// \brief Runs a query whose visitor returns `bool` (`false` stops).
    /// \ingroup core
    /// \param c      The connection.
    /// \param sql    The query.
    /// \param params The parameters.
    /// \param f      Called with each row.
    /// \return The rows seen, or the error.
    template <sql_connection C, typename F>
        requires std::is_same_v<std::invoke_result_t<F&, const sql_row&>, bool>
    sql_result<std::size_t> sql_query(C& c, std::string_view sql, std::span<const sql_param> params, F&& f)
    {
        auto visit = [&f](const sql_row& r) { return f(r); };
        return c.query(sql, params, sql_row_visitor(visit));
    }

    /// \brief `sql_query` with the parameters written in place.
    /// \ingroup core
    /// \param c      The connection.
    /// \param sql    The query.
    /// \param params The parameters.
    /// \param f      Called with each row.
    /// \return The rows seen, or the error.
    template <sql_connection C, typename F>
    sql_result<std::size_t> sql_query(C& c, std::string_view sql, std::initializer_list<sql_param> params, F&& f)
    {
        return sql_query(c, sql, std::span<const sql_param>(params.begin(), params.size()), std::forward<F>(f));
    }

    /// \brief A transaction that rolls back unless committed: `auto tx = sql_transaction<C>::begin(c);`.
    /// \ingroup core
    /// \tparam C The connection's type (a model, or `sql_connection_interface`).
    template <sql_connection C>
    class sql_transaction {
    public:
        /// \brief Begins a transaction on `c`.
        /// \param c The connection, which must outlive the transaction.
        /// \return The transaction, or `begin`'s error.
        [[nodiscard]] static sql_result<sql_transaction> begin(C& c)
        {
            SNICHOLLS_TRY_VOID(c.begin());
            return ok(sql_transaction(c));
        }

        sql_transaction(const sql_transaction&) = delete;
        sql_transaction& operator=(const sql_transaction&) = delete;
        /// \brief Takes over `other`'s transaction.
        /// \param other The transaction moved from; it no longer rolls back.
        sql_transaction(sql_transaction&& other) noexcept : c_(std::exchange(other.c_, nullptr)) {}
        sql_transaction& operator=(sql_transaction&&) = delete;

        /// \brief Rolls back, unless committed or rolled back already.
        ~sql_transaction()
        {
            if (c_ != nullptr && c_->in_transaction())
                static_cast<void>(c_->rollback());   // a failed rollback leaves the connection to report itself broken
        }

        /// \brief Commits.
        /// \return Success, or `commit`'s error (the transaction is then over either way).
        sql_result<void> commit()
        {
            C* c = std::exchange(c_, nullptr);
            return c->commit();
        }

        /// \brief Rolls back now.
        /// \return Success, or `rollback`'s error.
        sql_result<void> rollback()
        {
            C* c = std::exchange(c_, nullptr);
            return c->rollback();
        }

    private:
        explicit sql_transaction(C& c) noexcept : c_(&c) {}
        C* c_;
    };

} // namespace snicholls

#endif // snicholls_core_concepts_sql_connection_hpp
