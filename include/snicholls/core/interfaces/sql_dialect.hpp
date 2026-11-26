//
//  sql_dialect.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::sql_dialect_interface`: the pure virtual twin of `sql_dialect`, and the adapters both ways.
/// \ingroup core
///
/// **Overview.** When the engine is chosen at run time (configuration names SQLite or PostgreSQL), code holds a
/// `const sql_dialect_interface&`, which a connection returns from `dialect()`. Its pure virtual functions are the
/// operations of the concept `sql_dialect` (`concepts/sql_dialect.hpp`, where the contract and the generators are
/// documented), so the interface models the concept and the generators take it as they are.
/// `sql_dialect_adapter<D>` puts any model behind the interface, owning it (`D`) or referring to it (`const D&`);
/// `sql_dialect_ref` is a copyable, non-owning handle on the interface that models the concept.
///
/// \snippet interfaces/sql_dialect_demo.cpp virtual
///
/// **Relationships.** `concepts/sql_dialect.hpp`, `interfaces/sql_dialect_reference.hpp`,
/// `interfaces/sql_dialect_conformance.hpp`, `interfaces/sql_connection.hpp`. Group: \ref core.
///
/// **Implementations / variants.** `sql_dialect_adapter<D>` over any model; crtp's dialects may derive from the
/// interface directly.
///
/// **Guarantees.** The adapters add one virtual call; dialects are immutable, so a shared one is safe from any thread.
///
/// **Errors.** None from the members.
///
/// **Performance.** One virtual call; not benchmarked.
///
/// **Tests and demos.** `tests/interfaces/sql_dialect_test.cpp` (the kit through every form),
/// `demos/interfaces/sql_dialect_demo.cpp`.
///
/// **Dependencies.** `<cstddef>`, `<optional>`, `<string>`, `<string_view>`, `<type_traits>`, `<utility>`,
/// `concepts/sql_dialect.hpp`.
///
/// **Known issues.** None.
///
/// **Status.** New in 0.27.0. Experimental until two real implementations pass the kit.

#ifndef snicholls_core_interfaces_sql_dialect_hpp
#define snicholls_core_interfaces_sql_dialect_hpp

#include <snicholls/core/concepts/sql_dialect.hpp>

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace snicholls {

    /// \brief The pure virtual twin of `sql_dialect`.
    /// \ingroup core
    class sql_dialect_interface {
    public:
        /// \brief Destroys the dialect.
        virtual ~sql_dialect_interface() = default;
        /// \brief The engine's name.
        /// \return `"sqlite"`, `"postgres"`.
        [[nodiscard]] virtual std::string_view name() const noexcept = 0;
        /// \brief The engine's name for a type.
        /// \param t The type.
        /// \return It, or `std::nullopt`.
        [[nodiscard]] virtual std::optional<std::string_view> column_type(sql_type t) const noexcept = 0;
        /// \brief Appends `id`, quoted.
        /// \param out Where to append.
        /// \param id  The identifier.
        /// \return `false` for an empty identifier or one with a NUL.
        virtual bool append_identifier(std::string& out, std::string_view id) const = 0;
        /// \brief Appends the 1-based placeholder `n`.
        /// \param out Where to append.
        /// \param n   The parameter's number.
        virtual void append_placeholder(std::string& out, std::size_t n) const = 0;
        /// \brief The statement that begins a read-write transaction.
        /// \return It.
        [[nodiscard]] virtual std::string_view begin_transaction() const noexcept = 0;
        /// \brief What follows a clustered table's definition.
        /// \return It, or empty.
        [[nodiscard]] virtual std::string_view clustered_suffix() const noexcept = 0;

    protected:
        /// \brief Constructs the base.
        sql_dialect_interface() = default;
        /// \brief Copies the base (there is nothing to copy).
        sql_dialect_interface(const sql_dialect_interface&) = default;
        /// \brief Assigns the base (there is nothing to assign).
        /// \return `*this`.
        sql_dialect_interface& operator=(const sql_dialect_interface&) = default;
    };

    static_assert(sql_dialect<sql_dialect_interface>, "snicholls::sql_dialect_interface must model snicholls::sql_dialect");

    /// \brief Any model of `sql_dialect` behind the interface: owned (`D`) or referred to (`const D&`).
    /// \ingroup core
    /// \tparam D The model, or a reference to one.
    template <typename D>
        requires sql_dialect<std::remove_cvref_t<D>>
    class sql_dialect_adapter final : public sql_dialect_interface {
    public:
        /// \brief Takes (or, for a reference `D`, refers to) the dialect.
        /// \param impl The dialect.
        explicit sql_dialect_adapter(D impl) noexcept(std::is_nothrow_constructible_v<D, D&&>) : impl_(std::forward<D>(impl)) {}
        /// \brief Asks the model.
        /// \return As the model.
        [[nodiscard]] std::string_view name() const noexcept override { return impl_.name(); }
        /// \brief Asks the model.
        /// \param t The type.
        /// \return As the model.
        [[nodiscard]] std::optional<std::string_view> column_type(sql_type t) const noexcept override { return impl_.column_type(t); }
        /// \brief Asks the model.
        /// \param out Where to append.
        /// \param id  The identifier.
        /// \return As the model.
        bool append_identifier(std::string& out, std::string_view id) const override { return impl_.append_identifier(out, id); }
        /// \brief Asks the model.
        /// \param out Where to append.
        /// \param n   The parameter's number.
        void append_placeholder(std::string& out, std::size_t n) const override { impl_.append_placeholder(out, n); }
        /// \brief Asks the model.
        /// \return As the model.
        [[nodiscard]] std::string_view begin_transaction() const noexcept override { return impl_.begin_transaction(); }
        /// \brief Asks the model.
        /// \return As the model.
        [[nodiscard]] std::string_view clustered_suffix() const noexcept override { return impl_.clustered_suffix(); }
        /// \brief The adapted dialect.
        /// \return It.
        [[nodiscard]] const std::remove_cvref_t<D>& get() const noexcept { return impl_; }

    private:
        D impl_;
    };

    /// \brief Deduces an owning adapter: `sql_dialect_adapter(sqlite_dialect{})`.
    /// \ingroup core
    template <typename D>
    sql_dialect_adapter(D) -> sql_dialect_adapter<D>;

    /// \brief A copyable, non-owning handle on a `sql_dialect_interface`; it models the concept.
    /// \ingroup core
    class sql_dialect_ref {
    public:
        /// \brief Refers to `d`, which must outlive the handle.
        /// \param d The dialect.
        explicit sql_dialect_ref(const sql_dialect_interface& d) noexcept : p_(&d) {}
        /// \brief Asks the interface.
        /// \return As the interface.
        [[nodiscard]] std::string_view name() const noexcept { return p_->name(); }
        /// \brief Asks the interface.
        /// \param t The type.
        /// \return As the interface.
        [[nodiscard]] std::optional<std::string_view> column_type(sql_type t) const noexcept { return p_->column_type(t); }
        /// \brief Asks the interface.
        /// \param out Where to append.
        /// \param id  The identifier.
        /// \return As the interface.
        bool append_identifier(std::string& out, std::string_view id) const { return p_->append_identifier(out, id); }
        /// \brief Asks the interface.
        /// \param out Where to append.
        /// \param n   The parameter's number.
        void append_placeholder(std::string& out, std::size_t n) const { p_->append_placeholder(out, n); }
        /// \brief Asks the interface.
        /// \return As the interface.
        [[nodiscard]] std::string_view begin_transaction() const noexcept { return p_->begin_transaction(); }
        /// \brief Asks the interface.
        /// \return As the interface.
        [[nodiscard]] std::string_view clustered_suffix() const noexcept { return p_->clustered_suffix(); }

    private:
        const sql_dialect_interface* p_;
    };

    static_assert(sql_dialect<sql_dialect_ref>, "snicholls::sql_dialect_ref must model snicholls::sql_dialect");

} // namespace snicholls

#endif // snicholls_core_interfaces_sql_dialect_hpp
