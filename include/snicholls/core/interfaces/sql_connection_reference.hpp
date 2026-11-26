//
//  sql_connection_reference.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::memory_sql_connection`: the standard-library reference `sql_connection`, an in-memory table store
/// that runs the statements core's generators write.
/// \ingroup core
///
/// **Overview.** A test double for code written against `sql_connection`: its tables live in memory, its dialect is
/// `sqlite_dialect` (`$N` placeholders are read too), and it runs exactly the shapes of SQL that
/// `concepts/sql_dialect.hpp` generates, which is what the kit and engine-neutral code send:
/// - `CREATE TABLE [IF NOT EXISTS] "t" ("c" TYPE [NOT NULL], ..., PRIMARY KEY (...)) [WITHOUT ROWID]` and
///   `DROP TABLE [IF EXISTS] "t"`;
/// - `INSERT INTO "t" (...) VALUES (?1, ...) [ON CONFLICT (...) DO NOTHING | DO UPDATE SET "c" = excluded."c", ...]`;
/// - `SELECT "c", ... | count(*) FROM "t" [WHERE "c" = ?n AND "c" >= ?n AND "c" < ?n] [ORDER BY "c" [DESC], ...]
///   [LIMIT n]` and `DELETE FROM "t" [WHERE ...]`;
/// - `BEGIN [IMMEDIATE]`, `COMMIT`, `ROLLBACK`.
///
/// Anything else is `syntax_error`: it is not an SQL engine, and a test that needs one uses SQLite. Values are typed
/// as PostgreSQL types them (text into an integer column is `type_mismatch`), with SQLite's integer booleans; `NOT
/// NULL` and the primary key are enforced. Statements are parsed once and cached by text (`prepare`). A transaction
/// is a snapshot of every table, restored on rollback; a batch is all or nothing the same way. `break_connection()`
/// makes every later call `connection_lost`, to test a caller's recovery.
///
/// \snippet interfaces/sql_connection_reference_demo.cpp reference
///
/// **Relationships.** Models `sql_connection` (`concepts/sql_connection.hpp`); passes
/// `interfaces/sql_connection_conformance.hpp` directly and through `interfaces/sql_connection.hpp`;
/// `interfaces/sql_dialect_reference.hpp`. Group: \ref core.
///
/// **Implementations / variants.** One.
///
/// **Guarantees.** Not thread-safe (one per test). Strong exception safety for a failed statement: the tables are as
/// they were. Error messages are literals: no value, table or column name reaches one.
///
/// **Errors.** As the contract: `syntax_error`, `undefined_object`, `type_mismatch`, `constraint_violation`,
/// `unique_violation`, `misuse` (parameter count, nested or missing transaction), `connection_lost`.
///
/// **Performance.** Not benchmarked: a test double; every lookup is a scan.
///
/// **Tests and demos.** `tests/interfaces/sql_connection_reference_test.cpp`,
/// `demos/interfaces/sql_connection_reference_demo.cpp`.
///
/// **Dependencies.** `<algorithm>`, `<cctype>`, `<cstddef>`, `<cstdint>`, `<functional>`, `<map>`, `<optional>`,
/// `<span>`, `<string>`, `<string_view>`, `<utility>`, `<vector>`, `interfaces/sql_connection.hpp`,
/// `interfaces/sql_dialect_reference.hpp`.
///
/// **Known issues.** Only the statements above; no joins, expressions, indexes or functions but `count(*)`.
///
/// **Status.** New in 0.27.0. A test double.

#ifndef snicholls_core_interfaces_sql_connection_reference_hpp
#define snicholls_core_interfaces_sql_connection_reference_hpp

#include <snicholls/core/interfaces/sql_connection.hpp>
#include <snicholls/core/interfaces/sql_dialect_reference.hpp>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace snicholls {

    /// \brief The reference connection: an in-memory table store for the statements core generates.
    /// \ingroup core
    class memory_sql_connection {
    public:
        /// \brief Its dialect.
        /// \return `sqlite_dialect`.
        [[nodiscard]] const sqlite_dialect& dialect() const noexcept { return dialect_; }

        /// \brief Parses and caches `sql`.
        /// \param sql The statement.
        /// \return Success, `syntax_error` or `connection_lost`.
        sql_result<void> prepare(std::string_view sql)
        {
            if (!open_)
                return sql_fail(sql_errc::connection_lost, "the connection is broken");
            const auto s = statement_for(sql);
            if (!s)
                return err(s.unwrap_err());
            return ok();
        }

        /// \brief Runs one statement.
        /// \param sql    The statement.
        /// \param params The parameters.
        /// \return The rows changed, or the error.
        sql_result<std::size_t> execute(std::string_view sql, std::span<const sql_param> params)
        {
            return run(sql, params, nullptr);
        }

        /// \brief Runs a query and visits its rows.
        /// \param sql    The query.
        /// \param params The parameters.
        /// \param visit  The visitor.
        /// \return The rows seen, or the error.
        sql_result<std::size_t> query(std::string_view sql, std::span<const sql_param> params, sql_row_visitor visit)
        {
            return run(sql, params, &visit);
        }

        /// \brief Runs one statement for many rows, all or nothing.
        /// \param sql   The statement.
        /// \param width Parameters per row.
        /// \param rows  Rows.
        /// \param fill  The filler.
        /// \return The rows changed in all, or the first error (and nothing changed).
        sql_result<std::size_t> execute_batch(std::string_view sql, std::size_t width, std::size_t rows, sql_row_filler fill)
        {
            if (!open_)
                return sql_fail(sql_errc::connection_lost, "the connection is broken");
            const tables before = tables_;
            std::vector<sql_param> params(width);
            std::size_t changed = 0;
            for (std::size_t r = 0; r < rows; ++r) {
                std::fill(params.begin(), params.end(), sql_param{sql_null});
                fill(r, params);
                const auto n = run(sql, params, nullptr);
                if (!n) {
                    tables_ = before;
                    return err(n.unwrap_err());
                }
                changed += *n;
            }
            return ok(changed);
        }

        /// \brief Begins a transaction.
        /// \return Success, `misuse` when one is open, or `connection_lost`.
        sql_result<void> begin()
        {
            if (!open_)
                return sql_fail(sql_errc::connection_lost, "the connection is broken");
            if (saved_)
                return sql_fail(sql_errc::misuse, "a transaction is already open");
            saved_ = tables_;
            return ok();
        }
        /// \brief Commits.
        /// \return Success, `misuse` when none is open, or `connection_lost`.
        sql_result<void> commit()
        {
            if (!open_)
                return sql_fail(sql_errc::connection_lost, "the connection is broken");
            if (!saved_)
                return sql_fail(sql_errc::misuse, "no transaction is open");
            saved_.reset();
            return ok();
        }
        /// \brief Rolls back.
        /// \return Success, `misuse` when none is open, or `connection_lost`.
        sql_result<void> rollback()
        {
            if (!open_)
                return sql_fail(sql_errc::connection_lost, "the connection is broken");
            if (!saved_)
                return sql_fail(sql_errc::misuse, "no transaction is open");
            tables_ = std::move(*saved_);
            saved_.reset();
            return ok();
        }
        /// \brief Whether a transaction is open.
        /// \return It.
        [[nodiscard]] bool in_transaction() const noexcept { return saved_.has_value(); }
        /// \brief A round trip.
        /// \return Success, or `connection_lost`.
        sql_result<void> ping()
        {
            if (!open_)
                return sql_fail(sql_errc::connection_lost, "the connection is broken");
            return ok();
        }
        /// \brief Whether the connection is usable.
        /// \return `false` after `break_connection()`.
        [[nodiscard]] bool is_open() const noexcept { return open_; }

        /// \brief Every later call fails with `connection_lost`, as a dropped connection would.
        void break_connection() noexcept { open_ = false; }
        /// \brief The rows of a table.
        /// \param table The table's name.
        /// \return How many, or 0 for no such table.
        [[nodiscard]] std::size_t row_count(std::string_view table) const
        {
            const auto it = tables_.find(table);
            return it == tables_.end() ? 0 : it->second.rows.size();
        }
        /// \brief How many statements are cached.
        /// \return The count.
        [[nodiscard]] std::size_t prepared() const noexcept { return cache_.size(); }

    private:
        struct column {
            std::string name;
            sql_kind kind{sql_kind::int64};
            bool nullable{false};
        };
        struct table {
            std::vector<column> columns;
            std::vector<std::size_t> key;
            std::vector<std::vector<sql_value>> rows;
        };
        using tables = std::map<std::string, table, std::less<>>;

        enum class verb : std::uint8_t { create, drop, insert, select, count, erase, begin, commit, rollback };
        struct condition {
            std::string column;
            char rel = '=';   // '=', 'g' (>=) or '<'
            std::size_t param = 0;
        };
        struct order {
            std::string column;
            bool descending = false;
        };
        struct statement {
            verb kind{verb::select};
            std::string table;
            bool flag = false;                  // IF NOT EXISTS / IF EXISTS
            std::vector<column> columns;        // create
            std::vector<std::string> key;       // create: PRIMARY KEY
            std::vector<std::string> names;     // insert: the columns; select: the columns
            std::vector<std::size_t> values;    // insert: the parameter of each column
            sql_conflict conflict{sql_conflict::fail};
            std::vector<std::string> updates;   // DO UPDATE SET
            std::vector<condition> where;
            std::vector<order> order_by;
            std::size_t limit = 0;
            std::size_t params = 0;             // the highest placeholder
        };

        // ---- the tokenizer and the parser of the generated subset ----------------------------------------------------

        enum class tk : std::uint8_t { word, ident, param, number, punct, end };
        struct token {
            tk kind{tk::end};
            std::string text;
            std::size_t n = 0;
        };

        static bool tokenize(std::string_view s, std::vector<token>& out)
        {
            std::size_t i = 0;
            while (i < s.size()) {
                const char c = s[i];
                if (c == ' ' || c == '\n' || c == '\t' || c == '\r') {
                    ++i;
                } else if (c == '"') {
                    std::string id;
                    ++i;
                    for (;;) {
                        if (i >= s.size())
                            return false;
                        if (s[i] == '"') {
                            if (i + 1 < s.size() && s[i + 1] == '"') {
                                id += '"';
                                i += 2;
                                continue;
                            }
                            ++i;
                            break;
                        }
                        id += s[i++];
                    }
                    out.push_back({tk::ident, std::move(id), 0});
                } else if (c == '?' || c == '$') {
                    std::size_t n = 0, j = i + 1;
                    while (j < s.size() && s[j] >= '0' && s[j] <= '9')
                        n = n * 10 + static_cast<std::size_t>(s[j++] - '0');
                    if (j == i + 1 || n == 0)
                        return false;
                    out.push_back({tk::param, {}, n});
                    i = j;
                } else if (c >= '0' && c <= '9') {
                    std::size_t n = 0;
                    while (i < s.size() && s[i] >= '0' && s[i] <= '9')
                        n = n * 10 + static_cast<std::size_t>(s[i++] - '0');
                    out.push_back({tk::number, {}, n});
                } else if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
                    std::string w;
                    while (i < s.size() && (std::isalnum(static_cast<unsigned char>(s[i])) || s[i] == '_'))
                        w += static_cast<char>(std::toupper(static_cast<unsigned char>(s[i++])));
                    out.push_back({tk::word, std::move(w), 0});
                } else if ((c == '>' || c == '<') && i + 1 < s.size() && s[i + 1] == '=') {
                    out.push_back({tk::punct, std::string{c, '='}, 0});
                    i += 2;
                } else if (std::string_view("(),=<>*;.").find(c) != std::string_view::npos) {
                    out.push_back({tk::punct, std::string(1, c), 0});
                    ++i;
                } else {
                    return false;
                }
            }
            while (!out.empty() && out.back().kind == tk::punct && out.back().text == ";")
                out.pop_back();
            out.push_back({});
            return true;
        }

        struct parser {
            const std::vector<token>& t;
            std::size_t i = 0;
            statement s{};
            bool good = true;

            const token& peek() const { return t[i]; }
            bool word(std::string_view w)
            {
                if (t[i].kind == tk::word && t[i].text == w) {
                    ++i;
                    return true;
                }
                return false;
            }
            bool punct(std::string_view p)
            {
                if (t[i].kind == tk::punct && t[i].text == p) {
                    ++i;
                    return true;
                }
                return false;
            }
            void need_word(std::string_view w) { good = good && word(w); }
            void need_punct(std::string_view p) { good = good && punct(p); }
            std::string ident()
            {
                if (t[i].kind != tk::ident) {
                    good = false;
                    return {};
                }
                return t[i++].text;
            }
            std::size_t param()
            {
                if (t[i].kind != tk::param) {
                    good = false;
                    return 0;
                }
                s.params = std::max(s.params, t[i].n);
                return t[i++].n;
            }
            std::vector<std::string> ident_list()
            {
                std::vector<std::string> out;
                need_punct("(");
                do
                    out.push_back(ident());
                while (good && punct(","));
                need_punct(")");
                return out;
            }
            void where_clause()
            {
                if (!word("WHERE"))
                    return;
                do {
                    condition c;
                    c.column = ident();
                    if (punct("="))
                        c.rel = '=';
                    else if (punct(">="))
                        c.rel = 'g';
                    else if (punct("<"))
                        c.rel = '<';
                    else
                        good = false;
                    c.param = param();
                    s.where.push_back(std::move(c));
                } while (good && word("AND"));
            }
            sql_kind type_kind()
            {
                std::string name;
                while (t[i].kind == tk::word && t[i].text != "NOT")
                    name += (name.empty() ? "" : " ") + t[i++].text;
                if (name == "INTEGER" || name == "BIGINT")
                    return sql_kind::int64;
                if (name == "BOOLEAN")
                    return sql_kind::boolean;
                if (name == "REAL" || name == "DOUBLE PRECISION")
                    return sql_kind::float64;
                if (name == "TEXT")
                    return sql_kind::text;
                if (name == "BLOB" || name == "BYTEA")
                    return sql_kind::blob;
                good = false;
                return sql_kind::null;
            }

            void parse()
            {
                if (word("CREATE")) {
                    s.kind = verb::create;
                    need_word("TABLE");
                    if (word("IF")) {
                        need_word("NOT");
                        need_word("EXISTS");
                        s.flag = true;
                    }
                    s.table = ident();
                    need_punct("(");
                    do {
                        if (word("PRIMARY")) {
                            need_word("KEY");
                            s.key = ident_list();
                            continue;
                        }
                        column c;
                        c.name = ident();
                        c.kind = type_kind();
                        c.nullable = true;
                        if (word("NOT")) {
                            need_word("NULL");
                            c.nullable = false;
                        }
                        s.columns.push_back(std::move(c));
                    } while (good && punct(","));
                    need_punct(")");
                    if (word("WITHOUT"))
                        need_word("ROWID");
                } else if (word("DROP")) {
                    s.kind = verb::drop;
                    need_word("TABLE");
                    if (word("IF")) {
                        need_word("EXISTS");
                        s.flag = true;
                    }
                    s.table = ident();
                } else if (word("INSERT")) {
                    s.kind = verb::insert;
                    need_word("INTO");
                    s.table = ident();
                    s.names = ident_list();
                    need_word("VALUES");
                    need_punct("(");
                    do
                        s.values.push_back(param());
                    while (good && punct(","));
                    need_punct(")");
                    good = good && s.values.size() == s.names.size();
                    if (word("ON")) {
                        need_word("CONFLICT");
                        static_cast<void>(ident_list());
                        need_word("DO");
                        if (word("NOTHING")) {
                            s.conflict = sql_conflict::ignore;
                        } else {
                            need_word("UPDATE");
                            need_word("SET");
                            s.conflict = sql_conflict::update;
                            do {
                                std::string target = ident();
                                need_punct("=");
                                need_word("EXCLUDED");
                                need_punct(".");
                                good = good && ident() == target;
                                s.updates.push_back(std::move(target));
                            } while (good && punct(","));
                        }
                    }
                } else if (word("SELECT")) {
                    s.kind = verb::select;
                    if (word("COUNT")) {
                        s.kind = verb::count;
                        need_punct("(");
                        need_punct("*");
                        need_punct(")");
                    } else {
                        do
                            s.names.push_back(ident());
                        while (good && punct(","));
                    }
                    need_word("FROM");
                    s.table = ident();
                    where_clause();
                    if (word("ORDER")) {
                        need_word("BY");
                        do {
                            order o{ident(), false};
                            if (word("DESC"))
                                o.descending = true;
                            else
                                static_cast<void>(word("ASC"));
                            s.order_by.push_back(std::move(o));
                        } while (good && punct(","));
                    }
                    if (word("LIMIT")) {
                        if (peek().kind == tk::number)
                            s.limit = t[i++].n;
                        else
                            good = false;
                    }
                } else if (word("DELETE")) {
                    s.kind = verb::erase;
                    need_word("FROM");
                    s.table = ident();
                    where_clause();
                } else if (word("BEGIN")) {
                    s.kind = verb::begin;
                    static_cast<void>(word("IMMEDIATE"));
                } else if (word("COMMIT")) {
                    s.kind = verb::commit;
                } else if (word("ROLLBACK")) {
                    s.kind = verb::rollback;
                } else {
                    good = false;
                }
                good = good && peek().kind == tk::end;
            }
        };

        sql_result<const statement*> statement_for(std::string_view sql)
        {
            if (const auto it = cache_.find(sql); it != cache_.end())
                return ok(&it->second);
            std::vector<token> tokens;
            if (!tokenize(sql, tokens))
                return sql_fail(sql_errc::syntax_error, "the memory engine does not understand this", "42601");
            parser p{tokens};
            p.parse();
            if (!p.good)
                return sql_fail(sql_errc::syntax_error, "the memory engine does not understand this", "42601");
            const auto [it, inserted] = cache_.emplace(std::string(sql), std::move(p.s));
            static_cast<void>(inserted);
            return ok(&it->second);
        }

        // ---- execution --------------------------------------------------------------------------------------------------

        // -1, 0, 1: NULL first, then numbers, text, blobs.
        static int compare(const sql_value& a, const sql_value& b)
        {
            const auto rank = [](const sql_value& v) {
                switch (kind_of(v)) {
                case sql_kind::null: return 0;
                case sql_kind::boolean:
                case sql_kind::int64:
                case sql_kind::float64: return 1;
                case sql_kind::text: return 2;
                case sql_kind::blob: return 3;
                }
                return 4;
            };
            const int ra = rank(a), rb = rank(b);
            if (ra != rb)
                return ra < rb ? -1 : 1;
            if (ra == 1) {
                const auto as_int = [](const sql_value& v) { return kind_of(v) == sql_kind::boolean ? std::int64_t{std::get<1>(v)} : std::get<2>(v); };
                if (kind_of(a) != sql_kind::float64 && kind_of(b) != sql_kind::float64) {
                    const std::int64_t x = as_int(a), y = as_int(b);
                    return x < y ? -1 : (x > y ? 1 : 0);
                }
                const double x = kind_of(a) == sql_kind::float64 ? std::get<3>(a) : static_cast<double>(as_int(a));
                const double y = kind_of(b) == sql_kind::float64 ? std::get<3>(b) : static_cast<double>(as_int(b));
                return x < y ? -1 : (x > y ? 1 : 0);
            }
            if (ra == 2)
                return std::get<4>(a).compare(std::get<4>(b)) < 0 ? -1 : (std::get<4>(a) == std::get<4>(b) ? 0 : 1);
            if (ra == 3) {
                const auto& x = std::get<5>(a);
                const auto& y = std::get<5>(b);
                if (std::lexicographical_compare(x.begin(), x.end(), y.begin(), y.end()))
                    return -1;
                return x == y ? 0 : 1;
            }
            return 0;
        }

        // The parameter as the column stores it, or the error.
        static sql_result<sql_value> coerce(const column& c, const sql_param& p)
        {
            const sql_kind k = kind_of(p);
            if (k == sql_kind::null) {
                if (!c.nullable)
                    return sql_fail(sql_errc::constraint_violation, "NULL in a NOT NULL column", "23502");
                return ok(sql_value{sql_null});
            }
            if (k == c.kind)
                return ok(to_value(p));
            if (c.kind == sql_kind::int64 && k == sql_kind::boolean)
                return ok(sql_value{std::int64_t{std::get<1>(p) ? 1 : 0}});
            if (c.kind == sql_kind::float64 && k == sql_kind::int64)
                return ok(sql_value{static_cast<double>(std::get<2>(p))});
            if (c.kind == sql_kind::boolean && k == sql_kind::int64 && (std::get<2>(p) == 0 || std::get<2>(p) == 1))
                return ok(sql_value{std::get<2>(p) == 1});
            return sql_fail(sql_errc::type_mismatch, "a value the column's type cannot hold", "42804");
        }

        static std::optional<std::size_t> index_of(const table& t, std::string_view name)
        {
            for (std::size_t i = 0; i < t.columns.size(); ++i)
                if (t.columns[i].name == name)
                    return i;
            return std::nullopt;
        }

        sql_result<std::size_t> run(std::string_view sql, std::span<const sql_param> params, sql_row_visitor* visit)
        {
            if (!open_)
                return sql_fail(sql_errc::connection_lost, "the connection is broken");
            const auto found = statement_for(sql);
            if (!found)
                return err(found.unwrap_err());
            const statement& s = **found;
            if (params.size() != s.params)
                return sql_fail(sql_errc::misuse, "the wrong number of parameters");
            switch (s.kind) {
            case verb::begin: SNICHOLLS_TRY_VOID(begin()); return ok(std::size_t{0});
            case verb::commit: SNICHOLLS_TRY_VOID(commit()); return ok(std::size_t{0});
            case verb::rollback: SNICHOLLS_TRY_VOID(rollback()); return ok(std::size_t{0});
            case verb::create: return create(s);
            case verb::drop: {
                const auto it = tables_.find(s.table);
                if (it == tables_.end()) {
                    if (s.flag)
                        return ok(std::size_t{0});
                    return sql_fail(sql_errc::undefined_object, "no such table", "42P01");
                }
                tables_.erase(it);
                return ok(std::size_t{0});
            }
            default: break;
            }
            const auto it = tables_.find(s.table);
            if (it == tables_.end())
                return sql_fail(sql_errc::undefined_object, "no such table", "42P01");
            table& t = it->second;
            if (s.kind == verb::insert)
                return insert(t, s, params);
            // the rows the WHERE selects
            std::vector<std::size_t> picked;
            std::vector<std::pair<std::size_t, char>> conds;
            for (const condition& c : s.where) {
                const auto col = index_of(t, c.column);
                if (!col)
                    return sql_fail(sql_errc::undefined_object, "no such column", "42703");
                conds.emplace_back(*col, c.rel);
            }
            std::vector<sql_value> bound;
            for (const condition& c : s.where)
                bound.push_back(to_value(params[c.param - 1]));
            for (std::size_t r = 0; r < t.rows.size(); ++r) {
                bool keep = true;
                for (std::size_t k = 0; k < conds.size() && keep; ++k) {
                    const sql_value& cell = t.rows[r][conds[k].first];
                    if (kind_of(cell) == sql_kind::null || kind_of(bound[k]) == sql_kind::null) {
                        keep = false;
                        break;
                    }
                    const int c = compare(cell, bound[k]);
                    keep = conds[k].second == '=' ? c == 0 : (conds[k].second == 'g' ? c >= 0 : c < 0);
                }
                if (keep)
                    picked.push_back(r);
            }
            if (s.kind == verb::erase) {
                for (auto r = picked.rbegin(); r != picked.rend(); ++r)
                    t.rows.erase(t.rows.begin() + static_cast<std::ptrdiff_t>(*r));
                return ok(picked.size());
            }
            if (s.kind == verb::count) {
                const sql_param cell = static_cast<std::int64_t>(picked.size());
                const std::string_view name = "count(*)";
                if (visit != nullptr)
                    static_cast<void>((*visit)(sql_row(std::span<const std::string_view>(&name, 1), std::span<const sql_param>(&cell, 1))));
                return ok(std::size_t{1});
            }
            // select
            std::vector<std::pair<std::size_t, bool>> order_cols;
            for (const order& o : s.order_by) {
                const auto col = index_of(t, o.column);
                if (!col)
                    return sql_fail(sql_errc::undefined_object, "no such column", "42703");
                order_cols.emplace_back(*col, o.descending);
            }
            std::stable_sort(picked.begin(), picked.end(), [&](std::size_t a, std::size_t b) {
                for (const auto& [col, desc] : order_cols) {
                    const int c = compare(t.rows[a][col], t.rows[b][col]);
                    if (c != 0)
                        return desc ? c > 0 : c < 0;
                }
                return false;
            });
            if (s.limit != 0 && picked.size() > s.limit)
                picked.resize(s.limit);
            std::vector<std::size_t> cols;
            std::vector<std::string_view> names;
            for (const std::string& n : s.names) {
                const auto col = index_of(t, n);
                if (!col)
                    return sql_fail(sql_errc::undefined_object, "no such column", "42703");
                cols.push_back(*col);
                names.push_back(t.columns[*col].name);
            }
            std::size_t seen = 0;
            std::vector<sql_param> cells(cols.size());
            for (const std::size_t r : picked) {
                for (std::size_t k = 0; k < cols.size(); ++k)
                    cells[k] = view_of(t.rows[r][cols[k]]);
                ++seen;
                if (visit != nullptr && !(*visit)(sql_row(names, cells)))
                    break;
            }
            return ok(seen);
        }

        sql_result<std::size_t> create(const statement& s)
        {
            if (tables_.find(s.table) != tables_.end()) {
                if (s.flag)
                    return ok(std::size_t{0});
                return sql_fail(sql_errc::misuse, "the table exists", "42P07");
            }
            table t;
            t.columns = s.columns;
            for (const std::string& k : s.key) {
                const auto col = index_of(t, k);
                if (!col)
                    return sql_fail(sql_errc::undefined_object, "no such column", "42703");
                t.key.push_back(*col);
                t.columns[*col].nullable = false;
            }
            tables_.emplace(s.table, std::move(t));
            return ok(std::size_t{0});
        }

        static sql_result<std::size_t> insert(table& t, const statement& s, std::span<const sql_param> params)
        {
            std::vector<sql_value> row(t.columns.size(), sql_value{sql_null});
            std::vector<bool> given(t.columns.size(), false);
            for (std::size_t k = 0; k < s.names.size(); ++k) {
                const auto col = index_of(t, s.names[k]);
                if (!col)
                    return sql_fail(sql_errc::undefined_object, "no such column", "42703");
                auto v = coerce(t.columns[*col], params[s.values[k] - 1]);
                if (!v)
                    return err(v.unwrap_err());
                row[*col] = std::move(*v);
                given[*col] = true;
            }
            for (std::size_t c = 0; c < t.columns.size(); ++c)
                if (!given[c] && !t.columns[c].nullable)
                    return sql_fail(sql_errc::constraint_violation, "NULL in a NOT NULL column", "23502");
            if (!t.key.empty())
                for (auto& existing : t.rows) {
                    bool same = true;
                    for (const std::size_t k : t.key)
                        same = same && compare(existing[k], row[k]) == 0;
                    if (!same)
                        continue;
                    if (s.conflict == sql_conflict::fail)
                        return sql_fail(sql_errc::unique_violation, "a duplicate key", "23505");
                    if (s.conflict == sql_conflict::ignore)
                        return ok(std::size_t{0});
                    for (const std::string& u : s.updates)
                        if (const auto col = index_of(t, u))
                            existing[*col] = row[*col];
                    return ok(std::size_t{1});
                }
            t.rows.push_back(std::move(row));
            return ok(std::size_t{1});
        }

        sqlite_dialect dialect_{};
        tables tables_;
        std::optional<tables> saved_;
        std::map<std::string, statement, std::less<>> cache_;
        bool open_ = true;
    };

    static_assert(sql_connection<memory_sql_connection>, "snicholls::memory_sql_connection must model snicholls::sql_connection");

} // namespace snicholls

#endif // snicholls_core_interfaces_sql_connection_reference_hpp
