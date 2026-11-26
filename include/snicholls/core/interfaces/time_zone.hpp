//
//  time_zone.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::time_zone_interface`: the pure virtual twin of `time_zone`, and the adapters both ways.
/// \ingroup core
///
/// **Overview.** When the zone is chosen at run time (by an exchange, from a user's settings, from the IANA database),
/// code holds a `time_zone_interface&` (crtp's interface of the same name). Its pure virtual functions are the
/// operations of the concept `time_zone` (`concepts/time_zone.hpp`, where the contract and the conversions are
/// documented), so the interface models the concept and `from_local`, `to_local` and `offset_at` take it as they are.
/// `time_zone_adapter<Z>` puts any model behind the interface, owning it (`Z`) or referring to it (`Z&`);
/// `time_zone_ref` is a copyable, non-owning handle on the interface that models the concept.
///
/// \snippet interfaces/time_zone_demo.cpp virtual
///
/// **Relationships.** `concepts/time_zone.hpp`, `interfaces/time_zone_reference.hpp`,
/// `interfaces/time_zone_conformance.hpp`. Group: \ref core.
///
/// **Implementations / variants.** `time_zone_adapter<Z>` over any model; a database's zones above core derive from
/// the interface directly.
///
/// **Guarantees.** The adapters add one virtual call and nothing else. A zone is immutable once made, so a shared one
/// may be read from any thread when the implementation is.
///
/// **Errors.** None from the members.
///
/// **Performance.** One virtual call; not benchmarked.
///
/// **Tests and demos.** `tests/interfaces/time_zone_test.cpp` (the kit through every form),
/// `demos/interfaces/time_zone_demo.cpp`.
///
/// **Dependencies.** `<chrono>`, `<string_view>`, `<type_traits>`, `<utility>`, `concepts/time_zone.hpp`.
///
/// **Known issues.** None.
///
/// **Status.** New in 0.27.0. Experimental until two real implementations pass the kit.

#ifndef snicholls_core_interfaces_time_zone_hpp
#define snicholls_core_interfaces_time_zone_hpp

#include <snicholls/core/concepts/time_zone.hpp>

#include <chrono>
#include <string_view>
#include <type_traits>
#include <utility>

namespace snicholls {

    /// \brief The pure virtual twin of `time_zone`.
    /// \ingroup core
    class time_zone_interface {
    public:
        /// \brief Destroys the zone.
        virtual ~time_zone_interface() = default;
        /// \brief The zone's name.
        /// \return It, valid while the zone lives.
        [[nodiscard]] virtual std::string_view name() const noexcept = 0;
        /// \brief The rules in force at an instant.
        /// \param t The instant.
        /// \return The rules and their interval.
        [[nodiscard]] virtual zone_info info(std::chrono::sys_seconds t) const = 0;
        /// \brief Classifies a wall-clock time.
        /// \param l The wall-clock time.
        /// \return Unique, in a gap or in a fold, with the rules.
        [[nodiscard]] virtual local_info info(std::chrono::local_seconds l) const = 0;

    protected:
        /// \brief Constructs the base.
        time_zone_interface() = default;
        /// \brief Copies the base (there is nothing to copy).
        time_zone_interface(const time_zone_interface&) = default;
        /// \brief Assigns the base (there is nothing to assign).
        /// \return `*this`.
        time_zone_interface& operator=(const time_zone_interface&) = default;
    };

    static_assert(time_zone<time_zone_interface>, "snicholls::time_zone_interface must model snicholls::time_zone");

    /// \brief Any model of `time_zone` behind the interface: owned (`Z`) or referred to (`Z&`).
    /// \ingroup core
    /// \tparam Z The model, or a reference to one.
    template <typename Z>
        requires time_zone<std::remove_reference_t<Z>>
    class time_zone_adapter final : public time_zone_interface {
    public:
        /// \brief Takes (or, for a reference `Z`, refers to) the zone.
        /// \param impl The zone.
        explicit time_zone_adapter(Z impl) noexcept(std::is_nothrow_constructible_v<Z, Z&&>) : impl_(std::forward<Z>(impl)) {}
        /// \brief Asks the model.
        /// \return As the model.
        [[nodiscard]] std::string_view name() const noexcept override { return impl_.name(); }
        /// \brief Asks the model.
        /// \param t The instant.
        /// \return As the model.
        [[nodiscard]] zone_info info(std::chrono::sys_seconds t) const override { return impl_.info(t); }
        /// \brief Asks the model.
        /// \param l The wall-clock time.
        /// \return As the model.
        [[nodiscard]] local_info info(std::chrono::local_seconds l) const override { return impl_.info(l); }
        /// \brief The adapted zone.
        /// \return It.
        [[nodiscard]] std::remove_reference_t<Z>& get() noexcept { return impl_; }

    private:
        Z impl_;
    };

    /// \brief Deduces an owning adapter: `time_zone_adapter(std::move(zone))`.
    /// \ingroup core
    template <typename Z>
    time_zone_adapter(Z) -> time_zone_adapter<Z>;

    /// \brief A copyable, non-owning handle on a `time_zone_interface`; it models the concept.
    /// \ingroup core
    class time_zone_ref {
    public:
        /// \brief Refers to `zone`, which must outlive the handle.
        /// \param zone The zone.
        explicit time_zone_ref(const time_zone_interface& zone) noexcept : p_(&zone) {}
        /// \brief Asks the interface.
        /// \return As the interface.
        [[nodiscard]] std::string_view name() const noexcept { return p_->name(); }
        /// \brief Asks the interface.
        /// \param t The instant.
        /// \return As the interface.
        [[nodiscard]] zone_info info(std::chrono::sys_seconds t) const { return p_->info(t); }
        /// \brief Asks the interface.
        /// \param l The wall-clock time.
        /// \return As the interface.
        [[nodiscard]] local_info info(std::chrono::local_seconds l) const { return p_->info(l); }

    private:
        const time_zone_interface* p_;
    };

    static_assert(time_zone<time_zone_ref>, "snicholls::time_zone_ref must model snicholls::time_zone");

} // namespace snicholls

#endif // snicholls_core_interfaces_time_zone_hpp
