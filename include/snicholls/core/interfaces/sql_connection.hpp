//
//  sql_connection.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::sql_connection_interface`: the pure virtual twin of `sql_connection`, and the adapters both ways.
/// \ingroup core
///
/// **Overview.** When the engine is chosen at run time (configuration names SQLite or PostgreSQL), code holds a
/// `sql_connection_interface&` (crtp's `sql_connection`); its `dialect()` is a `const sql_dialect_interface&`. Its
/// pure virtual functions are the operations of the concept `sql_connection` (`concepts/sql_connection.hpp`, where the
/// contract is documented), so the interface models the concept and `sql_execute`, `sql_query` and `sql_transaction`
/// take it. `sql_connection_adapter<C>` puts any model behind the interface, owning it (`C`) or referring to it
/// (`C&`), with its dialect behind `sql_dialect_interface`; `sql_connection_ref` is a copyable, non-owning handle on
/// the interface that models the concept.
///
/// \snippet interfaces/sql_connection_demo.cpp virtual
///
/// **Relationships.** `concepts/sql_connection.hpp`, `interfaces/sql_dialect.hpp`,
/// `interfaces/sql_connection_reference.hpp`, `interfaces/sql_connection_conformance.hpp`. Group: \ref core.
///
/// **Implementations / variants.** `sql_connection_adapter<C>` over any model; crtp's engines may derive from the
/// interface directly.
///
/// **Guarantees.** The adapters add one virtual call per operation. An adapter refers to its own model's dialect, so
/// it is neither copied nor moved.
///
/// **Errors.** As the contract.
///
/// **Performance.** One virtual call; not benchmarked.
///
/// **Tests and demos.** `tests/interfaces/sql_connection_test.cpp` (the kit through every form),
/// `demos/interfaces/sql_connection_demo.cpp`.
///
/// **Dependencies.** `<cstddef>`, `<span>`, `<string_view>`, `<type_traits>`, `<utility>`,
/// `concepts/sql_connection.hpp`, `interfaces/sql_dialect.hpp`.
///
/// **Known issues.** None.
///
/// **Status.** New in 0.27.0. Experimental until two real implementations pass the kit.

#ifndef snicholls_core_interfaces_sql_connection_hpp
#define snicholls_core_interfaces_sql_connection_hpp

#include <snicholls/core/concepts/sql_connection.hpp>
#include <snicholls/core/interfaces/sql_dialect.hpp>

#include <cstddef>
#include <span>
#include <string_view>
#include <type_traits>
#include <utility>

namespace snicholls {

    /// \brief The pure virtual twin of `sql_connection`.
    /// \ingroup core
    class sql_connection_interface {
    public:
        /// \brief Destroys (closes) the connection.
        virtual ~sql_connection_interface() = default;
        /// \brief The engine's dialect.
        /// \return It, valid while the connection lives.
        [[nodiscard]] virtual const sql_dialect_interface& dialect() const noexcept = 0;
        /// \brief Parses and caches a statement.
        /// \param sql The statement.
        /// \return Success, or the error.
        virtual sql_result<void> prepare(std::string_view sql) = 0;
        /// \brief Runs one statement.
        /// \param sql    The statement.
        /// \param params Its parameters.
        /// \return The rows changed, or the error.
        virtual sql_result<std::size_t> execute(std::string_view sql, std::span<const sql_param> params) = 0;
        /// \brief Runs a query and visits its rows.
        /// \param sql    The query.
        /// \param params Its parameters.
        /// \param visit  The visitor; `false` stops.
        /// \return The rows seen, or the error.
        virtual sql_result<std::size_t> query(std::string_view sql, std::span<const sql_param> params, sql_row_visitor visit) = 0;
        /// \brief Runs one statement for many rows, all or nothing.
        /// \param sql   The statement.
        /// \param width Parameters per row.
        /// \param rows  Rows.
        /// \param fill  Writes each row's parameters.
        /// \return The rows changed in all, or the error.
        virtual sql_result<std::size_t> execute_batch(std::string_view sql, std::size_t width, std::size_t rows, sql_row_filler fill) = 0;
        /// \brief Begins a transaction.
        /// \return Success, or the error.
        virtual sql_result<void> begin() = 0;
        /// \brief Commits it.
        /// \return Success, or the error.
        virtual sql_result<void> commit() = 0;
        /// \brief Rolls it back.
        /// \return Success, or the error.
        virtual sql_result<void> rollback() = 0;
        /// \brief Whether a transaction is open.
        /// \return It.
        [[nodiscard]] virtual bool in_transaction() const noexcept = 0;
        /// \brief A round trip to the engine.
        /// \return Success, or the error.
        virtual sql_result<void> ping() = 0;
        /// \brief Whether the connection is usable.
        /// \return `false` once it is known to be broken.
        [[nodiscard]] virtual bool is_open() const noexcept = 0;

    protected:
        /// \brief Constructs the base.
        sql_connection_interface() = default;
        /// \brief Copies the base (there is nothing to copy).
        sql_connection_interface(const sql_connection_interface&) = default;
        /// \brief Assigns the base (there is nothing to assign).
        /// \return `*this`.
        sql_connection_interface& operator=(const sql_connection_interface&) = default;
    };

    static_assert(sql_connection<sql_connection_interface>, "snicholls::sql_connection_interface must model snicholls::sql_connection");

    /// \brief Any model of `sql_connection` behind the interface: owned (`C`) or referred to (`C&`).
    /// \ingroup core
    /// \tparam C The model, or a reference to one.
    template <typename C>
        requires sql_connection<std::remove_reference_t<C>>
    class sql_connection_adapter final : public sql_connection_interface {
        using dialect_type = std::remove_cvref_t<decltype(std::declval<const std::remove_reference_t<C>&>().dialect())>;

    public:
        /// \brief Takes (or, for a reference `C`, refers to) the connection.
        /// \param impl The connection.
        explicit sql_connection_adapter(C impl) : impl_(std::forward<C>(impl)), dialect_(impl_.dialect()) {}
        sql_connection_adapter(const sql_connection_adapter&) = delete;
        sql_connection_adapter& operator=(const sql_connection_adapter&) = delete;

        /// \brief The model's dialect, behind the interface.
        /// \return It.
        [[nodiscard]] const sql_dialect_interface& dialect() const noexcept override { return dialect_; }
        /// \brief Forwards to the model.
        /// \param sql The statement.
        /// \return As the model.
        sql_result<void> prepare(std::string_view sql) override { return impl_.prepare(sql); }
        /// \brief Forwards to the model.
        /// \param sql    The statement.
        /// \param params The parameters.
        /// \return As the model.
        sql_result<std::size_t> execute(std::string_view sql, std::span<const sql_param> params) override { return impl_.execute(sql, params); }
        /// \brief Forwards to the model.
        /// \param sql    The query.
        /// \param params The parameters.
        /// \param visit  The visitor.
        /// \return As the model.
        sql_result<std::size_t> query(std::string_view sql, std::span<const sql_param> params, sql_row_visitor visit) override
        {
            return impl_.query(sql, params, visit);
        }
        /// \brief Forwards to the model.
        /// \param sql   The statement.
        /// \param width Parameters per row.
        /// \param rows  Rows.
        /// \param fill  The filler.
        /// \return As the model.
        sql_result<std::size_t> execute_batch(std::string_view sql, std::size_t width, std::size_t rows, sql_row_filler fill) override
        {
            return impl_.execute_batch(sql, width, rows, fill);
        }
        /// \brief Forwards to the model.
        /// \return As the model.
        sql_result<void> begin() override { return impl_.begin(); }
        /// \brief Forwards to the model.
        /// \return As the model.
        sql_result<void> commit() override { return impl_.commit(); }
        /// \brief Forwards to the model.
        /// \return As the model.
        sql_result<void> rollback() override { return impl_.rollback(); }
        /// \brief Forwards to the model.
        /// \return As the model.
        [[nodiscard]] bool in_transaction() const noexcept override { return impl_.in_transaction(); }
        /// \brief Forwards to the model.
        /// \return As the model.
        sql_result<void> ping() override { return impl_.ping(); }
        /// \brief Forwards to the model.
        /// \return As the model.
        [[nodiscard]] bool is_open() const noexcept override { return impl_.is_open(); }
        /// \brief The adapted connection.
        /// \return It.
        [[nodiscard]] std::remove_reference_t<C>& get() noexcept { return impl_; }

    private:
        C impl_;
        sql_dialect_adapter<const dialect_type&> dialect_;
    };

    /// \brief A copyable, non-owning handle on a `sql_connection_interface`; it models the concept.
    /// \ingroup core
    class sql_connection_ref {
    public:
        /// \brief Refers to `c`, which must outlive the handle.
        /// \param c The connection.
        explicit sql_connection_ref(sql_connection_interface& c) noexcept : p_(&c) {}
        /// \brief Asks the interface.
        /// \return As the interface.
        [[nodiscard]] const sql_dialect_interface& dialect() const noexcept { return p_->dialect(); }
        /// \brief Forwards to the interface.
        /// \param sql The statement.
        /// \return As the interface.
        sql_result<void> prepare(std::string_view sql) { return p_->prepare(sql); }
        /// \brief Forwards to the interface.
        /// \param sql    The statement.
        /// \param params The parameters.
        /// \return As the interface.
        sql_result<std::size_t> execute(std::string_view sql, std::span<const sql_param> params) { return p_->execute(sql, params); }
        /// \brief Forwards to the interface.
        /// \param sql    The query.
        /// \param params The parameters.
        /// \param visit  The visitor.
        /// \return As the interface.
        sql_result<std::size_t> query(std::string_view sql, std::span<const sql_param> params, sql_row_visitor visit)
        {
            return p_->query(sql, params, visit);
        }
        /// \brief Forwards to the interface.
        /// \param sql   The statement.
        /// \param width Parameters per row.
        /// \param rows  Rows.
        /// \param fill  The filler.
        /// \return As the interface.
        sql_result<std::size_t> execute_batch(std::string_view sql, std::size_t width, std::size_t rows, sql_row_filler fill)
        {
            return p_->execute_batch(sql, width, rows, fill);
        }
        /// \brief Forwards to the interface.
        /// \return As the interface.
        sql_result<void> begin() { return p_->begin(); }
        /// \brief Forwards to the interface.
        /// \return As the interface.
        sql_result<void> commit() { return p_->commit(); }
        /// \brief Forwards to the interface.
        /// \return As the interface.
        sql_result<void> rollback() { return p_->rollback(); }
        /// \brief Forwards to the interface.
        /// \return As the interface.
        [[nodiscard]] bool in_transaction() const noexcept { return p_->in_transaction(); }
        /// \brief Forwards to the interface.
        /// \return As the interface.
        sql_result<void> ping() { return p_->ping(); }
        /// \brief Forwards to the interface.
        /// \return As the interface.
        [[nodiscard]] bool is_open() const noexcept { return p_->is_open(); }

    private:
        sql_connection_interface* p_;
    };

    static_assert(sql_connection<sql_connection_ref>, "snicholls::sql_connection_ref must model snicholls::sql_connection");

} // namespace snicholls

#endif // snicholls_core_interfaces_sql_connection_hpp
