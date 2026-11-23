//
//  key_store.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::key_store`: anything that creates, rotates, retires and destroys signing keys, and never exports
/// one.
/// \ingroup core
///
/// **Overview.** A key store manages the life of signing keys; a `signer` (`concepts/signer.hpp`) uses them. The two
/// are separate capabilities because a product may sign through a KMS it does not administer. Neither exports private
/// key material: there is no operation that could. This is the concept half of a pair; the pure virtual half is
/// `key_store_interface` (`interfaces/key_store.hpp`), with `key_store_adapter<K>` and `key_store_ref`.
///
/// The life of a key: `create` makes version 1, active. `rotate` makes the next version active and retires the one
/// before it (retired versions still verify, so a JWKS keeps publishing them until their tokens expire). `retire`
/// stops the whole key signing. `destroy` removes every version; nothing verifies afterwards.
///
/// **The contract** (`interfaces/key_store_conformance.hpp` checks it): an unknown id is `not_found`; an algorithm the
/// store cannot hold is `unsupported`; a label longer than 255 characters (`short_text`) is `invalid_argument`;
/// `rotate` of a retired or destroyed key is `conflict`; `info` returns the current version; `list` returns every
/// version not destroyed, by key id, then version; ids are unique and never reused, even after `destroy`. Errors
/// follow the redaction rule of `capability_error.hpp`.
///
/// \snippet concepts/key_store_demo.cpp concept
///
/// **Relationships.** `concepts/signer.hpp` (the algorithms), `capability_error.hpp`, `capability_ids.hpp`,
/// `time/utc.hpp`; the twin `interfaces/key_store.hpp`, the reference `interfaces/key_store_reference.hpp`, the kit
/// `interfaces/key_store_conformance.hpp`. Group: \ref core.
///
/// **Implementations.** `memory_key_store` (`interfaces/key_store_reference.hpp`; HS256 only, for tests). Above core:
/// a KMS server, AWS KMS.
///
/// **Guarantees.** A concept: no code, no cost. Thread safety is the implementation's.
///
/// **Errors.** `capability_error`, as the contract says.
///
/// **Performance.** Not benchmarked: key management is not a hot path.
///
/// **Tests and demos.** `tests/concepts/key_store_test.cpp`, `demos/concepts/key_store_demo.cpp`.
///
/// **Dependencies.** `<concepts>`, `<cstdint>`, `<string_view>`, `<vector>`, `capability_error.hpp`,
/// `capability_ids.hpp`, `concepts/signer.hpp`, `time/utc.hpp`.
///
/// **Known issues.** Envelope encryption (data keys under a master key) needs AES-GCM and is not part of the contract.
///
/// **Status.** New in 0.23.0. Experimental until two real implementations pass the kit.

#ifndef snicholls_core_concepts_key_store_hpp
#define snicholls_core_concepts_key_store_hpp

#include <snicholls/core/capability_error.hpp>
#include <snicholls/core/capability_ids.hpp>
#include <snicholls/core/concepts/signer.hpp>
#include <snicholls/core/time/utc.hpp>

#include <concepts>
#include <cstdint>
#include <string_view>
#include <vector>

namespace snicholls {

    /// \brief Where a key version is in its life.
    /// \ingroup core
    enum class key_state : std::uint8_t {
        active = 1,     ///< signs and verifies
        retired = 2,    ///< verifies only
        destroyed = 3,  ///< gone
    };

    /// \brief What to create.
    /// \ingroup core
    struct key_spec {
        signature_algorithm algorithm{signature_algorithm::eddsa};  ///< the algorithm
        std::string_view label;  ///< what the key is for (`"jwt-access"`), for people; never secret; at most 255 characters

        /// \brief Equality.
        friend constexpr bool operator==(const key_spec&, const key_spec&) noexcept = default;
    };

    /// \brief One key version, as the store describes it.
    /// \ingroup core
    struct key_info {
        snicholls::provider_id key_id;                              ///< the stable id
        std::uint32_t version{0};                                   ///< 1, 2, ...
        signature_algorithm algorithm{signature_algorithm::eddsa};  ///< the algorithm
        key_state state{key_state::active};                         ///< its state
        snicholls::short_text label;                                ///< its label
        utc_s created_at{};                                         ///< when this version was made

        /// \brief Equality.
        friend constexpr bool operator==(const key_info&, const key_info&) noexcept = default;
    };

    /// \brief A key store: create, rotate, retire, destroy, info, list.
    /// \ingroup core
    template <typename K>
    concept key_store = requires(K& k, const K& ck, const key_spec& spec, std::string_view id) {
        { k.create(spec) } -> std::same_as<capability_result<key_info>>;
        { k.rotate(id) } -> std::same_as<capability_result<key_info>>;
        { k.retire(id) } -> std::same_as<capability_result<void>>;
        { k.destroy(id) } -> std::same_as<capability_result<void>>;
        { ck.info(id) } -> std::same_as<capability_result<key_info>>;
        { ck.list() } -> std::same_as<capability_result<std::vector<key_info>>>;
    };

} // namespace snicholls

#endif // snicholls_core_concepts_key_store_hpp
