//
//  random_source.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::random_source_interface`, the virtual twin of `random_source_like`, and the adapters both ways.
/// \ingroup core
///
/// **Overview.** The run-time form of a source of cryptographically secure random bytes, for code that chooses its
/// source at run time (the operating system's in production, a seeded test source in tests). `random_source_adapter`
/// puts any `random_source_like` type behind the interface, held by value or, for a reference type, borrowed;
/// `any_random_source` owns an interface and is itself `random_source_like`, so a function written against the concept
/// accepts either form.
///
/// \snippet interfaces/random_source_demo.cpp interface
///
/// **Relationships.** `concepts/random_source.hpp` (the concept), `interfaces/random_source_reference.hpp`,
/// `interfaces/random_source_conformance.hpp`. Group: \ref core.
///
/// **Implementations.** `random_source_adapter<Impl>`; ts-moveables' operating-system source derives from the interface
/// or is adapted.
///
/// **Guarantees.** `fill` is `noexcept` in every form and fails closed: an error leaves no claim that any byte is
/// random. **C++17-clean.** An `any_random_source` that holds nothing reports `std::errc::operation_not_permitted`.
///
/// **Errors.** Whatever the source reports, passed through unchanged.
///
/// **Performance.** One virtual call per `fill`; not benchmarked, the source's system call dominates.
///
/// **Tests and demos.** `tests/interfaces/random_source_test.cpp`, `demos/interfaces/random_source_demo.cpp`.
///
/// **Dependencies.** `<cstddef>`, `<memory>`, `<system_error>`, `<type_traits>`, `<utility>`,
/// `concepts/random_source.hpp`.
///
/// **Known issues.** None.
///
/// **Status.** New in 0.25.0.

#ifndef snicholls_core_interfaces_random_source_hpp
#define snicholls_core_interfaces_random_source_hpp

#include <snicholls/core/concepts/random_source.hpp>

#include <cstddef>
#include <memory>
#include <system_error>
#include <type_traits>
#include <utility>

namespace snicholls {

    /// \brief A source of cryptographically secure random bytes, chosen at run time.
    /// \ingroup core
    class random_source_interface {
    public:
        /// \brief Virtual, so an implementation can be deleted through the interface.
        virtual ~random_source_interface() = default;
        /// \brief Writes exactly `size` random bytes to `data`, or reports why it cannot.
        /// \param data Where to write; may be null when `size` is 0.
        /// \param size How many bytes.
        /// \return Empty on success; an error (the source unavailable or unseeded) otherwise, with no claim on `data`.
        virtual std::errc fill(std::byte* data, std::size_t size) noexcept = 0;

    protected:
        /// \brief Constructs it.
        random_source_interface() = default;
        /// \brief Copies it.
        random_source_interface(const random_source_interface&) = default;
        /// \brief Copy-assigns.
        /// \return `*this`.
        random_source_interface& operator=(const random_source_interface&) = default;
    };

    /// \brief Any `random_source_like` type behind `random_source_interface`: held by value, or borrowed when `Impl`
    /// is a reference type (`random_source_adapter<my_source&>`).
    /// \ingroup core
    /// \tparam Impl The source, or a reference to one.
    template <typename Impl>
    class random_source_adapter final : public random_source_interface {
        static_assert(is_random_source_like_v<std::remove_reference_t<Impl>>,
                      "snicholls::random_source_adapter: Impl needs std::errc fill(std::byte*, std::size_t) noexcept");

    public:
        /// \brief Adapts this source.
        /// \param impl The source (moved in, or bound when `Impl` is a reference).
        explicit random_source_adapter(Impl impl) noexcept(std::is_nothrow_move_constructible_v<Impl> ||
                                                            std::is_reference_v<Impl>)
            : impl_(std::forward<Impl>(impl))
        {
        }
        /// \brief Forwards to the adapted source.
        /// \param data Where to write.
        /// \param size How many bytes.
        /// \return The source's answer.
        std::errc fill(std::byte* data, std::size_t size) noexcept override { return impl_.fill(data, size); }
        /// \brief The adapted source.
        /// \return A reference to it.
        [[nodiscard]] std::remove_reference_t<Impl>& get() noexcept { return impl_; }

    private:
        Impl impl_;
    };

    /// \brief An owned `random_source_interface` as a value that is itself `random_source_like`.
    /// \ingroup core
    class any_random_source {
    public:
        /// \brief Holds nothing: every `fill` fails with `std::errc::operation_not_permitted`.
        any_random_source() noexcept = default;
        /// \brief Takes ownership of an implementation.
        /// \param source The implementation (may be null, as the default).
        explicit any_random_source(std::unique_ptr<random_source_interface> source) noexcept : source_(std::move(source)) {}
        /// \brief Adapts and owns any `random_source_like` value.
        /// \tparam Impl The source type.
        /// \param impl The source.
        /// \return The owning value.
        template <typename Impl>
        [[nodiscard]] static any_random_source of(Impl impl)
        {
            return any_random_source(std::make_unique<random_source_adapter<Impl>>(std::move(impl)));
        }
        /// \brief Forwards to the held source; fails closed when there is none.
        /// \param data Where to write.
        /// \param size How many bytes.
        /// \return The source's answer, or `operation_not_permitted` when empty.
        std::errc fill(std::byte* data, std::size_t size) noexcept
        {
            if (!source_)
                return std::errc::operation_not_permitted;
            return source_->fill(data, size);
        }
        /// \brief Whether a source is held.
        /// \return `true` when one is.
        [[nodiscard]] explicit operator bool() const noexcept { return source_ != nullptr; }

    private:
        std::unique_ptr<random_source_interface> source_;
    };

    static_assert(is_random_source_like_v<random_source_interface>, "the interface must model its own concept");
    static_assert(is_random_source_like_v<any_random_source>, "any_random_source must model random_source_like");

} // namespace snicholls

#endif // snicholls_core_interfaces_random_source_hpp
