//
//  core.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief Umbrella header: all of snicholls-core in one include.
/// \ingroup core
///
/// Every header here stands alone and includes what it uses, so include the
/// one you need. This is for the cases where the one you need is all of it.

/// \defgroup core snicholls-core
/// \brief The vocabulary types and compile-time reflection shared by the snicholls libraries.
///
/// Everything in this group is header-only, standard-library-only and C++20.
/// There is no I/O, no threading, no networking and no codec here, by rule
/// (CLAUDE.md "What belongs here"); the libraries that pin core supply those.

/// \namespace snicholls
/// \brief Everything the snicholls libraries share, at the bottom of the stack.
///
/// Public names live directly in `snicholls`, implementation details in
/// `snicholls::detail`, and the library's own metadata in `snicholls::core`.

#ifndef snicholls_core_core_hpp
#define snicholls_core_core_hpp

#include <snicholls/core/acout.hpp>
#include <snicholls/core/backoff.hpp>
#include <snicholls/core/basic_type_registry.hpp>
#include <snicholls/core/checked.hpp>
#include <snicholls/core/colour.hpp>
#include <snicholls/core/constexpr_for.hpp>
#include <snicholls/core/constexpr_map.hpp>
#include <snicholls/core/decimal.hpp>
#include <snicholls/core/display.hpp>
#include <snicholls/core/endian.hpp>
#include <snicholls/core/fixed_string.hpp>
#include <snicholls/core/function_ref.hpp>
#include <snicholls/core/invoke_with.hpp>
#include <snicholls/core/iterate.hpp>
#include <snicholls/core/overloaded.hpp>
#include <snicholls/core/result.hpp>
#include <snicholls/core/secret.hpp>
#include <snicholls/core/strong.hpp>
#include <snicholls/core/thread_apply.hpp>
#include <snicholls/core/traced.hpp>
#include <snicholls/core/uint256.hpp>
#include <snicholls/core/uuid.hpp>
#include <snicholls/core/version.hpp>
#include <snicholls/core/zip.hpp>
#include <snicholls/core/combinatorics/partition.hpp>
#include <snicholls/core/combinatorics/permutations.hpp>
#include <snicholls/core/combinatorics/subsets.hpp>
#include <snicholls/core/containers/bloom_filter.hpp>
#include <snicholls/core/containers/byte_map.hpp>
#include <snicholls/core/containers/dense_id_map.hpp>
#include <snicholls/core/containers/flat_id_map.hpp>
#include <snicholls/core/containers/hierarchical_bitset.hpp>
#include <snicholls/core/containers/index_pool.hpp>
#include <snicholls/core/containers/inplace_vector.hpp>
#include <snicholls/core/containers/intrusive_avl_tree.hpp>
#include <snicholls/core/containers/mapped_header.hpp>
#include <snicholls/core/containers/md_array.hpp>
#include <snicholls/core/containers/object_pool.hpp>
#include <snicholls/core/containers/offset_ptr.hpp>
#include <snicholls/core/containers/perfect_map.hpp>
#include <snicholls/core/containers/static_map.hpp>
#include <snicholls/core/finance/attribution.hpp>
#include <snicholls/core/game/change.hpp>
#include <snicholls/core/game/coalition.hpp>
#include <snicholls/core/game/harsanyi.hpp>
#include <snicholls/core/game/owen.hpp>
#include <snicholls/core/game/properties.hpp>
#include <snicholls/core/game/shapley.hpp>
#include <snicholls/core/game/tabular.hpp>
#include <snicholls/core/hash/fnv.hpp>
#include <snicholls/core/hash/hash_combine.hpp>
#include <snicholls/core/hash/wyhash.hpp>
#include <snicholls/core/linalg/backend.hpp>
#include <snicholls/core/linalg/batched.hpp>
#include <snicholls/core/linalg/reference.hpp>
#include <snicholls/core/linalg/types.hpp>
#include <snicholls/core/math/constexpr_math.hpp>
#include <snicholls/core/meta/assignable.hpp>
#include <snicholls/core/meta/attributes.hpp>
#include <snicholls/core/meta/callable_traits.hpp>
#include <snicholls/core/meta/compare.hpp>
#include <snicholls/core/meta/concepts.hpp>
#include <snicholls/core/meta/enums.hpp>
#include <snicholls/core/meta/type_list.hpp>
#include <snicholls/core/meta/type_name.hpp>
#include <snicholls/core/meta/values.hpp>
#include <snicholls/core/ml/activations.hpp>
#include <snicholls/core/ml/attention.hpp>
#include <snicholls/core/ml/attention_layer.hpp>
#include <snicholls/core/ml/bpe.hpp>
#include <snicholls/core/ml/embedding.hpp>
#include <snicholls/core/ml/kv_cache.hpp>
#include <snicholls/core/ml/linear.hpp>
#include <snicholls/core/ml/minifloat.hpp>
#include <snicholls/core/ml/mlp.hpp>
#include <snicholls/core/ml/norm.hpp>
#include <snicholls/core/ml/optim.hpp>
#include <snicholls/core/ml/quantize.hpp>
#include <snicholls/core/ml/rope.hpp>
#include <snicholls/core/ml/row.hpp>
#include <snicholls/core/ml/sampling.hpp>
#include <snicholls/core/ml/softmax.hpp>
#include <snicholls/core/ml/transformer_block.hpp>
#include <snicholls/core/ml/transformer_config.hpp>
#include <snicholls/core/pipeline/stages.hpp>
#include <snicholls/core/reflect/member_count.hpp>
#include <snicholls/core/reflect/string_access.hpp>
#include <snicholls/core/reflect/string_traits.hpp>
#include <snicholls/core/reflect/struct_tuple.hpp>
#include <snicholls/core/stats/accumulators.hpp>
#include <snicholls/core/stats/bootstrap.hpp>
#include <snicholls/core/stats/covariance.hpp>
#include <snicholls/core/stats/low_discrepancy.hpp>
#include <snicholls/core/stats/pivot.hpp>
#include <snicholls/core/stats/random.hpp>
#include <snicholls/core/stochastic/martingale.hpp>
#include <snicholls/core/stochastic/snell.hpp>
#include <snicholls/core/time/business_days.hpp>
#include <snicholls/core/time/civil.hpp>
#include <snicholls/core/time/rfc3339.hpp>
#include <snicholls/core/time/solar.hpp>
#include <snicholls/core/time/utc.hpp>
#include <snicholls/core/time/utc_format.hpp>

#endif /* snicholls_core_core_hpp */
