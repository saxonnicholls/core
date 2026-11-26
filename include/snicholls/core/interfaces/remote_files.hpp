//
//  remote_files.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::remote_files_interface`: the pure virtual twin of `remote_files`, and the adapters both ways.
/// \ingroup core
///
/// **Overview.** When the drop is chosen at run time (configuration names SFTP for one counterparty and HTTPS for
/// another), the collector holds a `remote_files_interface&`. Its pure virtual functions are the operations of the
/// concept `remote_files` (`concepts/remote_files.hpp`, where the contract and its value types are documented), so
/// the interface models the concept itself. `remote_files_adapter<S>` puts any model behind the interface, owning it
/// (`S`) or referring to it (`S&`); `remote_files_ref` is a copyable, non-owning handle on the interface that models
/// the concept.
///
/// \snippet interfaces/remote_files_demo.cpp virtual
///
/// **Relationships.** `concepts/remote_files.hpp`, `interfaces/remote_files_reference.hpp`,
/// `interfaces/remote_files_conformance.hpp`. Group: \ref core.
///
/// **Implementations / variants.** `remote_files_adapter<S>` over any model; an implementation above core may derive
/// from the interface directly.
///
/// **Guarantees.** The adapters add one virtual call and nothing else; thread safety is the wrapped implementation's.
///
/// **Errors.** As the contract.
///
/// **Performance.** One virtual call; not benchmarked (a drop is I/O bound).
///
/// **Tests and demos.** `tests/interfaces/remote_files_test.cpp` (the kit through every form),
/// `demos/interfaces/remote_files_demo.cpp`.
///
/// **Dependencies.** `<string_view>`, `<type_traits>`, `<utility>`, `<vector>`, `concepts/remote_files.hpp`.
///
/// **Known issues.** None.
///
/// **Status.** New in 0.27.0. Experimental until two real implementations pass the kit.

#ifndef snicholls_core_interfaces_remote_files_hpp
#define snicholls_core_interfaces_remote_files_hpp

#include <snicholls/core/concepts/remote_files.hpp>

#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace snicholls {

    /// \brief The pure virtual twin of `remote_files`.
    /// \ingroup core
    class remote_files_interface {
    public:
        /// \brief Destroys the drop.
        virtual ~remote_files_interface() = default;
        /// \brief The files whose names start with `prefix`, sorted by name.
        /// \param prefix The prefix; empty for every file.
        /// \return The files, or the error.
        virtual capability_result<std::vector<remote_file>> list(std::string_view prefix) = 0;
        /// \brief Streams one file to `sink`.
        /// \param name The file.
        /// \param sink Where the bytes go, in order.
        /// \return The size and SHA-256 of the bytes delivered, or the error.
        virtual capability_result<file_digest> fetch(std::string_view name, remote_file_sink sink) = 0;
        /// \brief Renames one file within the drop.
        /// \param from The file.
        /// \param to   Its new name, which must not exist.
        /// \return Success, or the error (`unsupported` when `supports(remote_files_op::move)` is false).
        virtual capability_result<void> move(std::string_view from, std::string_view to) = 0;
        /// \brief Whether an operation is supported.
        /// \param op The operation.
        /// \return `true` for `list` and `fetch`; for `move`, whether it may succeed.
        [[nodiscard]] virtual bool supports(remote_files_op op) const = 0;

    protected:
        /// \brief Constructs the base.
        remote_files_interface() = default;
        /// \brief Copies the base (there is nothing to copy).
        remote_files_interface(const remote_files_interface&) = default;
        /// \brief Assigns the base (there is nothing to assign).
        /// \return `*this`.
        remote_files_interface& operator=(const remote_files_interface&) = default;
    };

    static_assert(remote_files<remote_files_interface>, "snicholls::remote_files_interface must model snicholls::remote_files");

    /// \brief Any model of `remote_files` behind the interface: owned (`S`) or referred to (`S&`).
    /// \ingroup core
    /// \tparam S The model, or a reference to one.
    template <typename S>
        requires remote_files<std::remove_reference_t<S>>
    class remote_files_adapter final : public remote_files_interface {
    public:
        /// \brief Takes (or, for a reference `S`, refers to) the drop.
        /// \param impl The drop.
        explicit remote_files_adapter(S impl) noexcept(std::is_nothrow_constructible_v<S, S&&>) : impl_(std::forward<S>(impl)) {}
        /// \brief Lists through the model.
        /// \param prefix The prefix.
        /// \return As the model.
        capability_result<std::vector<remote_file>> list(std::string_view prefix) override { return impl_.list(prefix); }
        /// \brief Fetches through the model.
        /// \param name The file.
        /// \param sink The sink.
        /// \return As the model.
        capability_result<file_digest> fetch(std::string_view name, remote_file_sink sink) override { return impl_.fetch(name, sink); }
        /// \brief Moves through the model.
        /// \param from The file.
        /// \param to   Its new name.
        /// \return As the model.
        capability_result<void> move(std::string_view from, std::string_view to) override { return impl_.move(from, to); }
        /// \brief Asks the model.
        /// \param op The operation.
        /// \return As the model.
        [[nodiscard]] bool supports(remote_files_op op) const override { return impl_.supports(op); }
        /// \brief The adapted drop.
        /// \return It.
        [[nodiscard]] std::remove_reference_t<S>& get() noexcept { return impl_; }

    private:
        S impl_;
    };

    /// \brief Deduces an owning adapter: `remote_files_adapter(std::move(drop))`.
    /// \ingroup core
    template <typename S>
    remote_files_adapter(S) -> remote_files_adapter<S>;

    /// \brief A copyable, non-owning handle on a `remote_files_interface`; it models the concept.
    /// \ingroup core
    class remote_files_ref {
    public:
        /// \brief Refers to `files`, which must outlive the handle.
        /// \param files The drop.
        explicit remote_files_ref(remote_files_interface& files) noexcept : p_(&files) {}
        /// \brief Lists through the interface.
        /// \param prefix The prefix.
        /// \return As the interface.
        capability_result<std::vector<remote_file>> list(std::string_view prefix) { return p_->list(prefix); }
        /// \brief Fetches through the interface.
        /// \param name The file.
        /// \param sink The sink.
        /// \return As the interface.
        capability_result<file_digest> fetch(std::string_view name, remote_file_sink sink) { return p_->fetch(name, sink); }
        /// \brief Moves through the interface.
        /// \param from The file.
        /// \param to   Its new name.
        /// \return As the interface.
        capability_result<void> move(std::string_view from, std::string_view to) { return p_->move(from, to); }
        /// \brief Asks the interface.
        /// \param op The operation.
        /// \return As the interface.
        [[nodiscard]] bool supports(remote_files_op op) const { return p_->supports(op); }

    private:
        remote_files_interface* p_;
    };

    static_assert(remote_files<remote_files_ref>, "snicholls::remote_files_ref must model snicholls::remote_files");

} // namespace snicholls

#endif // snicholls_core_interfaces_remote_files_hpp
