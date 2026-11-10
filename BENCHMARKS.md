# Benchmarks

Every benchmark compares core against the hand-written equivalent on identical data, after a fairness
check that refuses to run if the two disagree (CLAUDE.md, "Zero overhead"). Only Release numbers mean
anything; a header quotes its own numbers in its **Performance** section.

**This file is generated.** Do not edit it: run

    make benchmarks                      # every benchmark, about a minute
    scripts/benchmarks_md.py --filter X  # one file's sections; the rest are kept

and commit the result. The "relative" column is each row's time over the fastest row in its file.

| | |
| --- | --- |
| date | 2026-11-09 |
| machine | Intel(R) Xeon(R) W-3245 CPU @ 3.20GHz, 32 CPUs |
| OS | Darwin 25.6.0 |
| compiler | AppleClang 17.0.0.17000309 |
| build | Release, `build-bench` |
| min time per benchmark | 1.0 s |
| this run | only `attributes`; other sections are from earlier runs |

## accumulators_benchmark

`benchmarks/stats/accumulators_benchmark.cpp`

| benchmark | time | relative | throughput |
| --- | ---: | ---: | ---: |
| `BM_Accumulators_Core` | 6.42 ms | 1.04x | 155.90 M/s |
| `BM_Accumulators_WelfordByHand` | 6.15 ms | 1.00x | 162.54 M/s |

## acout_benchmark

`benchmarks/acout_benchmark.cpp`

| benchmark | time | relative | throughput |
| --- | ---: | ---: | ---: |
| `BM_Acout_Core` | 1.81 ms | 1.31x | 2.26 M/s |
| `BM_Acout_PlainStreamUnsafe` | 1.49 ms | 1.07x | 2.76 M/s |
| `BM_Acout_Println` | 1.39 ms | 1.00x | 2.95 M/s |
| `BM_Acout_EngineVector` | 4.59 ms | 3.31x | 892.98 k/s |
| `BM_Acout_HandWrittenVector` | 4.52 ms | 3.26x | 906.09 k/s |
| `BM_Acout_Coloured` | 2.19 ms | 1.58x | 1.87 M/s |
| `BM_Acout_ColouredByHand` | 1.67 ms | 1.21x | 2.45 M/s |

## attributes_benchmark

`benchmarks/meta/attributes_benchmark.cpp`

| benchmark | time | relative | throughput |
| --- | ---: | ---: | ---: |
| `BM_Attributes_Colour` | 764.31 us | 2.72x | 1.47 G/s |
| `BM_Attributes_TableByHand` | 809.42 us | 2.89x | 1.30 G/s |
| `BM_Attributes_BareColoursNoGuard` | 280.53 us | 1.00x | 3.74 G/s |

## bloom_filter_benchmark

`benchmarks/containers/bloom_filter_benchmark.cpp`

| benchmark | time | relative | throughput |
| --- | ---: | ---: | ---: |
| `BM_BloomFilter_Core` | 160.30 us | 1.00x | 25.55 M/s |
| `BM_BloomFilter_HandWritten` | 160.85 us | 1.00x | 25.47 M/s |
| `BM_BloomFilter_UnorderedSetForScale` | 219.84 us | 1.37x | 18.63 M/s |

## bootstrap_benchmark

`benchmarks/stats/bootstrap_benchmark.cpp`

| benchmark | time | relative | throughput |
| --- | ---: | ---: | ---: |
| `BM_Bootstrap_StandardMt19937Gather` | 10.37 ms | 4.90x | 194.57 k/s |
| `BM_Bootstrap_ResampleTableReused` | 2.12 ms | 1.00x | 945.29 k/s |
| `BM_Bootstrap_CoreIndices` | 3.74 ms | 1.77x | 537.53 k/s |
| `BM_Bootstrap_HandWrittenIndices` | 4.35 ms | 2.06x | 463.27 k/s |
| `BM_Bootstrap_CoreMultinomialCounts` | 3.73 ms | 1.76x | 538.39 k/s |
| `BM_Bootstrap_CorePoissonCounts` | 5.38 ms | 2.54x | 371.91 k/s |

## dense_id_map_benchmark

`benchmarks/containers/dense_id_map_benchmark.cpp`

| benchmark | time | relative | throughput |
| --- | ---: | ---: | ---: |
| `dense_id_map` | 615.81 us | 1.00x | 106.43 M/s |
| `std_unordered_map` | 4.85 ms | 7.88x | 13.50 M/s |

## flat_id_map_benchmark

`benchmarks/containers/flat_id_map_benchmark.cpp`

| benchmark | time | relative | throughput |
| --- | ---: | ---: | ---: |
| `flat_id_map` | 103.92 us | 1.00x | 39.42 M/s |
| `std_unordered_map` | 597.84 us | 5.75x | 6.85 M/s |

## hash_combine_benchmark

`benchmarks/hash/hash_combine_benchmark.cpp`

| benchmark | time | relative | throughput |
| --- | ---: | ---: | ---: |
| `BM_HashCombine_Core` | 14.99 us | 1.01x | 273.20 M/s |
| `BM_HashCombine_HandWritten` | 14.91 us | 1.00x | 274.73 M/s |

## hierarchical_bitset_benchmark

`benchmarks/containers/hierarchical_bitset_benchmark.cpp`

| benchmark | time | relative | throughput |
| --- | ---: | ---: | ---: |
| `hierarchical_bitset` | 16.20 us | 1.00x | 63.29 M/s |
| `std_vector_bool` | 613.36 us | 37.85x | 1.67 M/s |

## intrusive_avl_tree_benchmark

`benchmarks/containers/intrusive_avl_tree_benchmark.cpp`

| benchmark | time | relative | throughput |
| --- | ---: | ---: | ---: |
| `intrusive_avl_tree` | 714.14 us | 1.00x | 5.74 M/s |
| `std_set` | 1.21 ms | 1.70x | 3.38 M/s |

## invoke_with_benchmark

`benchmarks/invoke_with_benchmark.cpp`

| benchmark | time | relative | throughput |
| --- | ---: | ---: | ---: |
| `BM_InvokeWith_Core` | 133.89 us | 5.66x | 30.59 M/s |
| `BM_InvokeWith_HandWritten` | 91.89 us | 3.88x | 44.58 M/s |
| `BM_InvokeWith_DirectCall` | 23.65 us | 1.00x | 173.17 M/s |

## object_pool_benchmark

`benchmarks/containers/object_pool_benchmark.cpp`

| benchmark | time | relative | throughput |
| --- | ---: | ---: | ---: |
| `object_pool` | 32.46 us | 1.00x | 126.19 M/s |
| `new_delete` | 593.52 us | 18.28x | 6.90 M/s |

## perfect_map_benchmark

`benchmarks/containers/perfect_map_benchmark.cpp`

| benchmark | time | relative | throughput |
| --- | ---: | ---: | ---: |
| `BM_Lookup_PerfectMap` | 205.81 us | 1.00x | 159.50 M/s |
| `BM_Lookup_StaticMap` | 2.75 ms | 13.37x | 11.91 M/s |
| `BM_Lookup_BinarySearchByHand` | 2.81 ms | 13.67x | 11.66 M/s |
| `BM_Lookup_StdUnorderedMap` | 740.82 us | 3.60x | 47.13 M/s |

## result_benchmark

`benchmarks/result_benchmark.cpp`

| benchmark | time | relative | throughput |
| --- | ---: | ---: | ---: |
| `result_core/0` | 9.06 us | 1.79x | 451.86 M/s |
| `result_core/1` | 9.36 us | 1.85x | 437.80 M/s |
| `result_core/10` | 11.12 us | 2.20x | 368.27 M/s |
| `result_core/50` | 21.95 us | 4.33x | 187.56 M/s |
| `result_core/100` | 11.08 us | 2.19x | 369.80 M/s |
| `result_hand_tagged/0` | 5.06 us | 1.00x | 808.95 M/s |
| `result_hand_tagged/1` | 5.39 us | 1.06x | 760.48 M/s |
| `result_hand_tagged/10` | 7.69 us | 1.52x | 532.66 M/s |
| `result_hand_tagged/50` | 16.98 us | 3.35x | 241.23 M/s |
| `result_hand_tagged/100` | 10.95 us | 2.16x | 374.11 M/s |
| `result_error_code/0` | 6.31 us | 1.25x | 649.57 M/s |
| `result_error_code/1` | 6.57 us | 1.30x | 623.01 M/s |
| `result_error_code/10` | 8.97 us | 1.77x | 456.65 M/s |
| `result_error_code/50` | 19.87 us | 3.92x | 206.16 M/s |
| `result_error_code/100` | 10.71 us | 2.12x | 382.42 M/s |
| `result_exceptions/0` | 5.07 us | 1.00x | 807.74 M/s |
| `result_exceptions/1` | 200.88 us | 39.67x | 20.51 M/s |
| `result_exceptions/10` | 1.95 ms | 384.99x | 2.10 M/s |
| `result_exceptions/50` | 9.67 ms | 1909.52x | 423.63 k/s |
| `result_exceptions/100` | 19.42 ms | 3835.39x | 210.91 k/s |
| `result_traced_success/0` | 5.08 us | 1.00x | 805.74 M/s |
| `result_core_propagate/0` | 14.19 us | 2.80x | 289.09 M/s |
| `result_core_propagate/1` | 14.40 us | 2.84x | 284.48 M/s |
| `result_core_propagate/10` | 16.85 us | 3.33x | 243.13 M/s |
| `result_core_propagate/50` | 26.58 us | 5.25x | 154.09 M/s |
| `result_core_propagate/100` | 15.29 us | 3.02x | 267.85 M/s |
| `result_hand_propagate/0` | 9.06 us | 1.79x | 452.34 M/s |
| `result_hand_propagate/1` | 9.53 us | 1.88x | 430.00 M/s |
| `result_hand_propagate/10` | 11.64 us | 2.30x | 351.78 M/s |
| `result_hand_propagate/50` | 20.23 us | 3.99x | 202.54 M/s |
| `result_hand_propagate/100` | 12.07 us | 2.38x | 339.37 M/s |

## shapley_benchmark

`benchmarks/game/shapley_benchmark.cpp`

| benchmark | time | relative | throughput |
| --- | ---: | ---: | ---: |
| `BM_Shapley_Core` | 466.62 us | 1.00x |  |
| `BM_Shapley_Formula3ByHand` | 1.58 ms | 3.39x |  |

## struct_tuple_benchmark

`benchmarks/reflect/struct_tuple_benchmark.cpp`

| benchmark | time | relative | throughput |
| --- | ---: | ---: | ---: |
| `BM_ToTuple_Core` | 4.08 us | 1.18x | 1.00 G/s |
| `BM_ToTuple_HandWritten` | 4.04 us | 1.17x | 1.02 G/s |
| `BM_Tie_Core` | 3.46 us | 1.00x | 1.18 G/s |
| `BM_Tie_HandWritten` | 3.55 us | 1.03x | 1.16 G/s |
| `BM_FromTuple_Core` | 3.90 us | 1.13x | 1.05 G/s |
| `BM_FromTuple_HandWritten` | 3.88 us | 1.12x | 1.06 G/s |

## uint256_benchmark

`benchmarks/uint256_benchmark.cpp`

| benchmark | time | relative | throughput |
| --- | ---: | ---: | ---: |
| `BM_Uint256_Add_Core` | 8.54 us | 1.00x | 479.92 M/s |
| `BM_Uint256_Add_HandWritten` | 8.53 us | 1.00x | 479.96 M/s |
| `BM_Uint256_Mul_Core` | 18.09 us | 2.12x | 226.47 M/s |
| `BM_Uint256_Mul_HandWritten` | 18.07 us | 2.12x | 226.74 M/s |
| `BM_Uint256_Divmod_Core` | 185.28 us | 21.71x | 22.11 M/s |
| `BM_Uint256_ToString_Core` | 2.29 ms | 268.27x | 1.79 M/s |

## wyhash_benchmark

`benchmarks/hash/wyhash_benchmark.cpp`

| benchmark | time | relative | throughput |
| --- | ---: | ---: | ---: |
| `BM_Wyhash_Core` | 36.11 us | 1.13x | 113.45 M/s |
| `BM_Wyhash_HandWritten` | 31.96 us | 1.00x | 128.15 M/s |
| `BM_Wyhash_Fnv1a64ForScale` | 158.40 us | 4.96x | 25.86 M/s |

## zip_benchmark

`benchmarks/zip_benchmark.cpp`

| benchmark | time | relative | throughput |
| --- | ---: | ---: | ---: |
| `BM_Zip_Core` | 4.06 us | 1.01x | 1.01 G/s |
| `BM_Zip_HandWritten` | 4.02 us | 1.00x | 1.02 G/s |
