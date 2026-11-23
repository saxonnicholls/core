//
//  signer_reference.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::memory_signer`: the standard-library reference `signer`, which is `memory_key_store`.
/// \ingroup core
///
/// **Overview.** A signer signs with the keys a key store keeps, so the reference of both capabilities is one class,
/// `memory_key_store` (`interfaces/key_store_reference.hpp`): HS256 keys in memory, for tests only. This header names
/// it as the signer's reference, so each capability has its `<capability>_reference.hpp`.
///
/// \snippet interfaces/signer_reference_demo.cpp reference
///
/// **Relationships.** `concepts/signer.hpp`, `interfaces/signer.hpp`, `interfaces/signer_conformance.hpp`,
/// `interfaces/key_store_reference.hpp`. Group: \ref core.
///
/// **Implementations / variants.** An alias.
///
/// **Guarantees.** Those of `memory_key_store`.
///
/// **Errors.** Those of `memory_key_store`.
///
/// **Performance.** `benchmarks/interfaces/key_store_reference_benchmark.cpp`.
///
/// **Tests and demos.** `tests/interfaces/signer_reference_test.cpp`, `demos/interfaces/signer_reference_demo.cpp`.
///
/// **Dependencies.** `interfaces/key_store_reference.hpp`, `interfaces/signer.hpp`.
///
/// **Known issues.** None.
///
/// **Status.** New in 0.23.0.

#ifndef snicholls_core_interfaces_signer_reference_hpp
#define snicholls_core_interfaces_signer_reference_hpp

#include <snicholls/core/interfaces/key_store_reference.hpp>
#include <snicholls/core/interfaces/signer.hpp>

namespace snicholls {

    /// \brief The reference signer: the reference key store, which signs with the keys it holds.
    /// \ingroup core
    using memory_signer = memory_key_store;

    static_assert(signer<memory_signer>, "snicholls::memory_signer must model snicholls::signer");

} // namespace snicholls

#endif // snicholls_core_interfaces_signer_reference_hpp
