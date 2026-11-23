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
#include <snicholls/core/combinatorics/assignment.hpp>
#include <snicholls/core/combinatorics/partition.hpp>
#include <snicholls/core/combinatorics/permutations.hpp>
#include <snicholls/core/combinatorics/subsets.hpp>
#include <snicholls/core/containers/ann_forest.hpp>
#include <snicholls/core/containers/bloom_filter.hpp>
#include <snicholls/core/containers/byte_map.hpp>
#include <snicholls/core/containers/dense_id_map.hpp>
#include <snicholls/core/containers/flat_id_map.hpp>
#include <snicholls/core/containers/hierarchical_bitset.hpp>
#include <snicholls/core/containers/index_pool.hpp>
#include <snicholls/core/containers/inplace_vector.hpp>
#include <snicholls/core/containers/intrusive_avl_tree.hpp>
#include <snicholls/core/containers/kd_tree.hpp>
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
#include <snicholls/core/geometry/alignment.hpp>
#include <snicholls/core/geometry/angle.hpp>
#include <snicholls/core/geometry/box.hpp>
#include <snicholls/core/geometry/camera.hpp>
#include <snicholls/core/geometry/coordinates.hpp>
#include <snicholls/core/geometry/geodetic.hpp>
#include <snicholls/core/hash/blake2.hpp>
#include <snicholls/core/hash/blake3.hpp>
#include <snicholls/core/hash/crc.hpp>
#include <snicholls/core/hash/digest.hpp>
#include <snicholls/core/hash/fnv.hpp>
#include <snicholls/core/hash/hash_combine.hpp>
#include <snicholls/core/hash/hmac.hpp>
#include <snicholls/core/hash/kdf.hpp>
#include <snicholls/core/hash/ripemd160.hpp>
#include <snicholls/core/hash/scrypt.hpp>
#include <snicholls/core/hash/secure.hpp>
#include <snicholls/core/hash/sha1.hpp>
#include <snicholls/core/hash/sha2.hpp>
#include <snicholls/core/hash/sha3.hpp>
#include <snicholls/core/hash/wyhash.hpp>
#include <snicholls/core/linalg/backend.hpp>
#include <snicholls/core/linalg/batched.hpp>
#include <snicholls/core/linalg/fixed.hpp>
#include <snicholls/core/linalg/krylov.hpp>
#include <snicholls/core/linalg/randomized_svd.hpp>
#include <snicholls/core/linalg/reference.hpp>
#include <snicholls/core/linalg/types.hpp>
#include <snicholls/core/math/constexpr_math.hpp>
#include <snicholls/core/math/roots.hpp>
#include <snicholls/core/math/special.hpp>
#include <snicholls/core/meta/assignable.hpp>
#include <snicholls/core/meta/attributes.hpp>
#include <snicholls/core/meta/callable_traits.hpp>
#include <snicholls/core/meta/compare.hpp>
#include <snicholls/core/meta/concepts.hpp>
#include <snicholls/core/meta/enums.hpp>
#include <snicholls/core/meta/type_list.hpp>
#include <snicholls/core/meta/type_name.hpp>
#include <snicholls/core/meta/values.hpp>
#include <snicholls/core/metric/angular.hpp>
#include <snicholls/core/metric/binary.hpp>
#include <snicholls/core/metric/concepts.hpp>
#include <snicholls/core/metric/geodesic.hpp>
#include <snicholls/core/metric/minkowski.hpp>
#include <snicholls/core/metric/sets.hpp>
#include <snicholls/core/metric/statistical.hpp>
#include <snicholls/core/metric/string.hpp>
#include <snicholls/core/ml/activations.hpp>
#include <snicholls/core/ml/alibi.hpp>
#include <snicholls/core/ml/attention.hpp>
#include <snicholls/core/ml/attention_layer.hpp>
#include <snicholls/core/ml/attention_sink.hpp>
#include <snicholls/core/ml/bitnet.hpp>
#include <snicholls/core/ml/bitnet_kernels.hpp>
#include <snicholls/core/ml/bpe.hpp>
#include <snicholls/core/ml/bpe_merges.hpp>
#include <snicholls/core/ml/collapse.hpp>
#include <snicholls/core/ml/composed_block.hpp>
#include <snicholls/core/ml/control_vector.hpp>
#include <snicholls/core/ml/conv.hpp>
#include <snicholls/core/ml/delta_rule.hpp>
#include <snicholls/core/ml/diloco.hpp>
#include <snicholls/core/ml/distill.hpp>
#include <snicholls/core/ml/embedding.hpp>
#include <snicholls/core/ml/expert_parallel.hpp>
#include <snicholls/core/ml/gather.hpp>
#include <snicholls/core/ml/gla.hpp>
#include <snicholls/core/ml/glu.hpp>
#include <snicholls/core/ml/grammar.hpp>
#include <snicholls/core/ml/hadamard.hpp>
#include <snicholls/core/ml/jepa.hpp>
#include <snicholls/core/ml/json_schema.hpp>
#include <snicholls/core/ml/kv_cache.hpp>
#include <snicholls/core/ml/kv_cache_quant.hpp>
#include <snicholls/core/ml/kv_cells.hpp>
#include <snicholls/core/ml/latent_attention.hpp>
#include <snicholls/core/ml/latent_block.hpp>
#include <snicholls/core/ml/linear.hpp>
#include <snicholls/core/ml/lm_head.hpp>
#include <snicholls/core/ml/logprobs.hpp>
#include <snicholls/core/ml/lora.hpp>
#include <snicholls/core/ml/masking.hpp>
#include <snicholls/core/ml/minifloat.hpp>
#include <snicholls/core/ml/mixer_layer.hpp>
#include <snicholls/core/ml/mlp.hpp>
#include <snicholls/core/ml/moe.hpp>
#include <snicholls/core/ml/moe_block.hpp>
#include <snicholls/core/ml/multi_token_prediction.hpp>
#include <snicholls/core/ml/muon.hpp>
#include <snicholls/core/ml/nf4.hpp>
#include <snicholls/core/ml/norm.hpp>
#include <snicholls/core/ml/optim.hpp>
#include <snicholls/core/ml/patch.hpp>
#include <snicholls/core/ml/pooling.hpp>
#include <snicholls/core/ml/preference.hpp>
#include <snicholls/core/ml/pretokenize.hpp>
#include <snicholls/core/ml/probe.hpp>
#include <snicholls/core/ml/quant_iq.hpp>
#include <snicholls/core/ml/quant_iq_grids.hpp>
#include <snicholls/core/ml/quant_k.hpp>
#include <snicholls/core/ml/quant_ternary.hpp>
#include <snicholls/core/ml/quantize.hpp>
#include <snicholls/core/ml/recurrence.hpp>
#include <snicholls/core/ml/reward_model.hpp>
#include <snicholls/core/ml/rl.hpp>
#include <snicholls/core/ml/rope.hpp>
#include <snicholls/core/ml/rope_modes.hpp>
#include <snicholls/core/ml/row.hpp>
#include <snicholls/core/ml/rwkv.hpp>
#include <snicholls/core/ml/rwkv_tokenizer.hpp>
#include <snicholls/core/ml/samplers.hpp>
#include <snicholls/core/ml/sampling.hpp>
#include <snicholls/core/ml/self_attention.hpp>
#include <snicholls/core/ml/sentencepiece.hpp>
#include <snicholls/core/ml/short_conv.hpp>
#include <snicholls/core/ml/softmax.hpp>
#include <snicholls/core/ml/sparse_attention.hpp>
#include <snicholls/core/ml/spatial.hpp>
#include <snicholls/core/ml/speculative.hpp>
#include <snicholls/core/ml/ssd.hpp>
#include <snicholls/core/ml/tensor_ops.hpp>
#include <snicholls/core/ml/token_trie.hpp>
#include <snicholls/core/ml/tokenizer.hpp>
#include <snicholls/core/ml/transformer_block.hpp>
#include <snicholls/core/ml/transformer_config.hpp>
#include <snicholls/core/ml/unicode.hpp>
#include <snicholls/core/ml/unigram.hpp>
#include <snicholls/core/ml/wordpiece.hpp>
#include <snicholls/core/optim/cones.hpp>
#include <snicholls/core/optim/lbfgs.hpp>
#include <snicholls/core/optim/line_search.hpp>
#include <snicholls/core/optim/prox.hpp>
#include <snicholls/core/optim/scaling.hpp>
#include <snicholls/core/pipeline/stages.hpp>
#include <snicholls/core/reflect/member_count.hpp>
#include <snicholls/core/reflect/string_access.hpp>
#include <snicholls/core/reflect/string_traits.hpp>
#include <snicholls/core/reflect/struct_tuple.hpp>
#include <snicholls/core/signal/complex.hpp>
#include <snicholls/core/signal/convolve.hpp>
#include <snicholls/core/signal/dct.hpp>
#include <snicholls/core/signal/fft.hpp>
#include <snicholls/core/signal/fir.hpp>
#include <snicholls/core/signal/hilbert.hpp>
#include <snicholls/core/signal/iir.hpp>
#include <snicholls/core/signal/mel.hpp>
#include <snicholls/core/signal/resample.hpp>
#include <snicholls/core/signal/stft.hpp>
#include <snicholls/core/signal/window.hpp>
#include <snicholls/core/sparse/csc.hpp>
#include <snicholls/core/sparse/ldl.hpp>
#include <snicholls/core/sparse/ordering.hpp>
#include <snicholls/core/stats/accumulators.hpp>
#include <snicholls/core/stats/bootstrap.hpp>
#include <snicholls/core/stats/covariance.hpp>
#include <snicholls/core/stats/low_discrepancy.hpp>
#include <snicholls/core/stats/normal.hpp>
#include <snicholls/core/stats/pivot.hpp>
#include <snicholls/core/stats/random.hpp>
#include <snicholls/core/stats/ransac.hpp>
#include <snicholls/core/stats/resample.hpp>
#include <snicholls/core/stats/uniform_int.hpp>
#include <snicholls/core/stochastic/martingale.hpp>
#include <snicholls/core/stochastic/snell.hpp>
#include <snicholls/core/time/business_days.hpp>
#include <snicholls/core/time/civil.hpp>
#include <snicholls/core/time/rfc3339.hpp>
#include <snicholls/core/time/solar.hpp>
#include <snicholls/core/time/utc.hpp>
#include <snicholls/core/time/utc_format.hpp>
#include <snicholls/core/tracking/association.hpp>
#include <snicholls/core/tracking/discrete.hpp>
#include <snicholls/core/tracking/fusion.hpp>
#include <snicholls/core/tracking/gaussian.hpp>
#include <snicholls/core/tracking/kalman.hpp>
#include <snicholls/core/tracking/measurement.hpp>
#include <snicholls/core/tracking/mixture.hpp>
#include <snicholls/core/tracking/transition.hpp>
#include <snicholls/core/tracking/unscented.hpp>

#endif /* snicholls_core_core_hpp */
