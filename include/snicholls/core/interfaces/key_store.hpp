//
//  key_store.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::key_store_interface`: the pure virtual twin of `key_store`, and the adapters both ways.
/// \ingroup core
///
/// **Overview.** When the store is a KMS chosen by configuration, code holds a `key_store_interface&`. Its pure virtual functions (`create`, `rotate`, `retire`, `destroy`, `info`, `list`) are the
/// operations of the concept `key_store` (`concepts/key_store.hpp`, where the contract and its value types are documented),
/// so the interface models the concept itself. `key_store_adapter<K>` puts any model behind the interface, owning it (`K`) or
/// referring to it (`K&`); `key_store_ref` is a copyable, non-owning handle on the interface that models the concept.
///
/// \snippet interfaces/key_store_demo.cpp virtual
///
/// **Relationships.** `concepts/key_store.hpp`, `interfaces/key_store_reference.hpp`, `interfaces/key_store_conformance.hpp`.
/// Group: \ref core.
///
/// **Implementations / variants.** `key_store_adapter<K>` over any model; an implementation above core may derive from the
/// interface directly.
///
/// **Guarantees.** The adapters add one virtual call and nothing else; thread safety and exception safety are the
/// wrapped implementation's.
///
/// **Errors.** As the contract.
///
/// **Performance.** One virtual call; not benchmarked (key management is not a hot path).
///
/// **Tests and demos.** `tests/interfaces/key_store_test.cpp` (the kit through every form, the adapters forward),
/// `demos/interfaces/key_store_demo.cpp`.
///
/// **Dependencies.** `<type_traits>`, `<utility>`, `concepts/key_store.hpp`.
///
/// **Known issues.** None.
///
/// **Status.** New in 0.23.0. Experimental until two real implementations pass the kit.

#ifndef snicholls_core_interfaces_key_store_hpp
#define snicholls_core_interfaces_key_store_hpp

#include <snicholls/core/concepts/key_store.hpp>

#include <type_traits>
#include <utility>

namespace snicholls {

    /// \brief The pure virtual twin of `key_store`.
    /// \ingroup core
    class key_store_interface {
    public:
        /// \brief Destroys a key store.
        virtual ~key_store_interface() = default;
        /// \brief Creates a key; version 1 is active.
        /// \param spec What to create.
        /// \return The new version, or the error.
        virtual capability_result<key_info> create(const key_spec& spec) = 0;
        /// \brief Makes the next version active and retires the previous one.
        /// \param id The key.
        /// \return The new version, or the error.
        virtual capability_result<key_info> rotate(std::string_view id) = 0;
        /// \brief Retires every version of a key.
        /// \param id The key.
        /// \return Success, or the error.
        virtual capability_result<void> retire(std::string_view id) = 0;
        /// \brief Destroys every version of a key.
        /// \param id The key.
        /// \return Success, or the error.
        virtual capability_result<void> destroy(std::string_view id) = 0;
        /// \brief The current version of a key.
        /// \param id The key.
        /// \return Its description, or the error.
        [[nodiscard]] virtual capability_result<key_info> info(std::string_view id) const = 0;
        /// \brief Every version not destroyed, by id then version.
        /// \return The versions, or the error.
        [[nodiscard]] virtual capability_result<std::vector<key_info>> list() const = 0;

    protected:
        /// \brief Constructs the base.
        key_store_interface() = default;
        /// \brief Copies the base (there is nothing to copy).
        key_store_interface(const key_store_interface&) = default;
        /// \brief Assigns the base (there is nothing to assign).
        /// \return `*this`.
        key_store_interface& operator=(const key_store_interface&) = default;
    };

    static_assert(key_store<key_store_interface>, "snicholls::key_store_interface must model snicholls::key_store");

    /// \brief Any model of `key_store` behind the interface: owned (`K`) or referred to (`K&`).
    /// \ingroup core
    /// \tparam K The model, or a reference to one.
    template <typename K>
        requires key_store<std::remove_reference_t<K>>
    class key_store_adapter final : public key_store_interface {
    public:
        /// \brief Takes (or, for a reference `K`, refers to) the implementation.
        /// \param impl The implementation.
        explicit key_store_adapter(K impl) noexcept(std::is_nothrow_constructible_v<K, K&&>) : impl_(std::forward<K>(impl)) {}
        /// \brief Creates a key; version 1 is active. Forwards to the model.
        /// \param spec What to create.
        /// \return The new version, or the error.
        capability_result<key_info> create(const key_spec& spec) override { return impl_.create(spec); }
        /// \brief Makes the next version active and retires the previous one. Forwards to the model.
        /// \param id The key.
        /// \return The new version, or the error.
        capability_result<key_info> rotate(std::string_view id) override { return impl_.rotate(id); }
        /// \brief Retires every version of a key. Forwards to the model.
        /// \param id The key.
        /// \return Success, or the error.
        capability_result<void> retire(std::string_view id) override { return impl_.retire(id); }
        /// \brief Destroys every version of a key. Forwards to the model.
        /// \param id The key.
        /// \return Success, or the error.
        capability_result<void> destroy(std::string_view id) override { return impl_.destroy(id); }
        /// \brief The current version of a key. Forwards to the model.
        /// \param id The key.
        /// \return Its description, or the error.
        [[nodiscard]] capability_result<key_info> info(std::string_view id) const override { return impl_.info(id); }
        /// \brief Every version not destroyed, by id then version. Forwards to the model.
        /// \return The versions, or the error.
        [[nodiscard]] capability_result<std::vector<key_info>> list() const override { return impl_.list(); }
        /// \brief The adapted implementation.
        /// \return It.
        [[nodiscard]] std::remove_reference_t<K>& get() noexcept { return impl_; }

    private:
        K impl_;
    };

    /// \brief Deduces an owning adapter: `key_store_adapter(std::move(impl))`.
    /// \ingroup core
    template <typename K>
    key_store_adapter(K) -> key_store_adapter<K>;

    /// \brief A copyable, non-owning handle on a `key_store_interface`; it models the concept.
    /// \ingroup core
    class key_store_ref {
    public:
        /// \brief Refers to `impl`, which must outlive the handle.
        /// \param impl The implementation.
        explicit key_store_ref(key_store_interface& impl) noexcept : p_(&impl) {}
        /// \brief Creates a key; version 1 is active. Forwards to the interface.
        /// \param spec What to create.
        /// \return The new version, or the error.
        capability_result<key_info> create(const key_spec& spec) { return p_->create(spec); }
        /// \brief Makes the next version active and retires the previous one. Forwards to the interface.
        /// \param id The key.
        /// \return The new version, or the error.
        capability_result<key_info> rotate(std::string_view id) { return p_->rotate(id); }
        /// \brief Retires every version of a key. Forwards to the interface.
        /// \param id The key.
        /// \return Success, or the error.
        capability_result<void> retire(std::string_view id) { return p_->retire(id); }
        /// \brief Destroys every version of a key. Forwards to the interface.
        /// \param id The key.
        /// \return Success, or the error.
        capability_result<void> destroy(std::string_view id) { return p_->destroy(id); }
        /// \brief The current version of a key. Forwards to the interface.
        /// \param id The key.
        /// \return Its description, or the error.
        [[nodiscard]] capability_result<key_info> info(std::string_view id) const { return p_->info(id); }
        /// \brief Every version not destroyed, by id then version. Forwards to the interface.
        /// \return The versions, or the error.
        [[nodiscard]] capability_result<std::vector<key_info>> list() const { return p_->list(); }

    private:
        key_store_interface* p_;
    };

    static_assert(key_store<key_store_ref>, "snicholls::key_store_ref must model snicholls::key_store");

} // namespace snicholls

#endif // snicholls_core_interfaces_key_store_hpp
