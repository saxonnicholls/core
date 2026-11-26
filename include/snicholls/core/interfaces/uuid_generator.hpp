//
//  uuid_generator.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::uuid_generator_interface`: the pure virtual twin of `uuid_generator`, and the adapters both ways.
/// \ingroup core
///
/// **Overview.** When the generator is chosen at run time (the operating system's in production, a seeded one in a
/// replay), code holds a `uuid_generator_interface&`. Its pure virtual functions are the operations of the concept
/// `uuid_generator` (`concepts/uuid_generator.hpp`, where the contract is documented), so the interface models the
/// concept. `uuid_generator_adapter<G>` puts any model behind the interface, owning it (`G`) or referring to it
/// (`G&`); `uuid_generator_ref` is a copyable, non-owning handle on the interface that models the concept.
///
/// \snippet interfaces/uuid_generator_demo.cpp virtual
///
/// **Relationships.** `concepts/uuid_generator.hpp`, `interfaces/uuid_generator_reference.hpp`,
/// `interfaces/uuid_generator_conformance.hpp`. Group: \ref core.
///
/// **Implementations / variants.** `uuid_generator_adapter<G>` over any model; a generator above core may derive from
/// the interface directly.
///
/// **Guarantees.** The adapters add one virtual call and nothing else; thread safety is the wrapped generator's.
///
/// **Errors.** As the contract.
///
/// **Performance.** One virtual call; not benchmarked.
///
/// **Tests and demos.** `tests/interfaces/uuid_generator_test.cpp` (the kit through every form),
/// `demos/interfaces/uuid_generator_demo.cpp`.
///
/// **Dependencies.** `<type_traits>`, `<utility>`, `concepts/uuid_generator.hpp`.
///
/// **Known issues.** None.
///
/// **Status.** New in 0.27.0. Experimental until two real implementations pass the kit.

#ifndef snicholls_core_interfaces_uuid_generator_hpp
#define snicholls_core_interfaces_uuid_generator_hpp

#include <snicholls/core/concepts/uuid_generator.hpp>

#include <type_traits>
#include <utility>

namespace snicholls {

    /// \brief The pure virtual twin of `uuid_generator`.
    /// \ingroup core
    class uuid_generator_interface {
    public:
        /// \brief Destroys the generator.
        virtual ~uuid_generator_interface() = default;
        /// \brief A version 4 (random) UUID.
        /// \return It, or the error.
        virtual capability_result<uuid> v4() = 0;
        /// \brief A version 7 (time-ordered) UUID.
        /// \return It, or the error.
        virtual capability_result<uuid> v7() = 0;

    protected:
        /// \brief Constructs the base.
        uuid_generator_interface() = default;
        /// \brief Copies the base (there is nothing to copy).
        uuid_generator_interface(const uuid_generator_interface&) = default;
        /// \brief Assigns the base (there is nothing to assign).
        /// \return `*this`.
        uuid_generator_interface& operator=(const uuid_generator_interface&) = default;
    };

    static_assert(uuid_generator<uuid_generator_interface>, "snicholls::uuid_generator_interface must model snicholls::uuid_generator");

    /// \brief Any model of `uuid_generator` behind the interface: owned (`G`) or referred to (`G&`).
    /// \ingroup core
    /// \tparam G The model, or a reference to one.
    template <typename G>
        requires uuid_generator<std::remove_reference_t<G>>
    class uuid_generator_adapter final : public uuid_generator_interface {
    public:
        /// \brief Takes (or, for a reference `G`, refers to) the generator.
        /// \param impl The generator.
        explicit uuid_generator_adapter(G impl) noexcept(std::is_nothrow_constructible_v<G, G&&>) : impl_(std::forward<G>(impl)) {}
        /// \brief Makes one through the model.
        /// \return As the model.
        capability_result<uuid> v4() override { return impl_.v4(); }
        /// \brief Makes one through the model.
        /// \return As the model.
        capability_result<uuid> v7() override { return impl_.v7(); }
        /// \brief The adapted generator.
        /// \return It.
        [[nodiscard]] std::remove_reference_t<G>& get() noexcept { return impl_; }

    private:
        G impl_;
    };

    /// \brief Deduces an owning adapter: `uuid_generator_adapter(std::move(generator))`.
    /// \ingroup core
    template <typename G>
    uuid_generator_adapter(G) -> uuid_generator_adapter<G>;

    /// \brief A copyable, non-owning handle on a `uuid_generator_interface`; it models the concept.
    /// \ingroup core
    class uuid_generator_ref {
    public:
        /// \brief Refers to `generator`, which must outlive the handle.
        /// \param generator The generator.
        explicit uuid_generator_ref(uuid_generator_interface& generator) noexcept : p_(&generator) {}
        /// \brief Makes one through the interface.
        /// \return As the interface.
        capability_result<uuid> v4() { return p_->v4(); }
        /// \brief Makes one through the interface.
        /// \return As the interface.
        capability_result<uuid> v7() { return p_->v7(); }

    private:
        uuid_generator_interface* p_;
    };

    static_assert(uuid_generator<uuid_generator_ref>, "snicholls::uuid_generator_ref must model snicholls::uuid_generator");

} // namespace snicholls

#endif // snicholls_core_interfaces_uuid_generator_hpp
