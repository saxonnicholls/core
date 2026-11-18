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
| date | 2026-11-19 |
| machine | Intel(R) Xeon(R) W-3245 CPU @ 3.20GHz, 32 CPUs |
| OS | Darwin 25.6.0 |
| compiler | AppleClang 17.0.0.17000309 |
| build | Release, `build-werror` |
| min time per benchmark | 0.5 s |
| this run | only `diloco`; other sections are from earlier runs |

## accumulators_benchmark

`benchmarks/stats/accumulators_benchmark.cpp`

| benchmark | time | relative | throughput |
| --- | ---: | ---: | ---: |
| `BM_Accumulators_Core` | 6.72 ms | 1.09x | 149.36 M/s |
| `BM_Accumulators_WelfordByHand` | 6.19 ms | 1.00x | 161.72 M/s |

## acout_benchmark

`benchmarks/acout_benchmark.cpp`

| benchmark | time | relative | throughput |
| --- | ---: | ---: | ---: |
| `BM_Acout_Core` | 1.92 ms | 1.32x | 2.20 M/s |
| `BM_Acout_PlainStreamUnsafe` | 1.58 ms | 1.08x | 2.61 M/s |
| `BM_Acout_Println` | 1.46 ms | 1.00x | 2.81 M/s |
| `BM_Acout_EngineVector` | 4.78 ms | 3.28x | 856.35 k/s |
| `BM_Acout_HandWrittenVector` | 4.69 ms | 3.22x | 873.74 k/s |
| `BM_Acout_Coloured` | 2.46 ms | 1.69x | 1.67 M/s |
| `BM_Acout_ColouredByHand` | 1.77 ms | 1.21x | 2.33 M/s |

## activations_benchmark

`benchmarks/ml/activations_benchmark.cpp`

| benchmark | time | relative | throughput |
| --- | ---: | ---: | ---: |
| `BM_Sigmoid_Core` | 580.08 us | 1.02x | 113.14 M/s |
| `BM_Sigmoid_HandWritten` | 566.94 us | 1.00x | 115.71 M/s |
| `BM_GeluTanh_Core` | 633.00 us | 1.12x | 103.72 M/s |
| `BM_GeluTanh_HandWritten` | 623.60 us | 1.10x | 105.24 M/s |

## attention_benchmark

`benchmarks/ml/attention_benchmark.cpp`

| benchmark | time | relative | throughput |
| --- | ---: | ---: | ---: |
| `BM_Attend_Core/256` | 24.31 us | 1.00x | 10.62 M/s |
| `BM_Attend_Core/4096` | 356.86 us | 14.68x | 11.48 M/s |
| `BM_Attend_HandWritten/256` | 24.96 us | 1.03x | 10.26 M/s |
| `BM_Attend_HandWritten/4096` | 359.98 us | 14.81x | 11.40 M/s |

## attributes_benchmark

`benchmarks/meta/attributes_benchmark.cpp`

| benchmark | time | relative | throughput |
| --- | ---: | ---: | ---: |
| `BM_Attributes_Colour` | 657.39 us | 2.46x | 1.60 G/s |
| `BM_Attributes_TableByHand` | 792.53 us | 2.97x | 1.32 G/s |
| `BM_Attributes_BareColoursNoGuard` | 267.01 us | 1.00x | 3.93 G/s |

## bloom_filter_benchmark

`benchmarks/containers/bloom_filter_benchmark.cpp`

| benchmark | time | relative | throughput |
| --- | ---: | ---: | ---: |
| `BM_BloomFilter_Core` | 167.29 us | 1.00x | 24.51 M/s |
| `BM_BloomFilter_HandWritten` | 171.49 us | 1.03x | 24.06 M/s |
| `BM_BloomFilter_UnorderedSetForScale` | 282.88 us | 1.69x | 14.88 M/s |

## bootstrap_benchmark

`benchmarks/stats/bootstrap_benchmark.cpp`

| benchmark | time | relative | throughput |
| --- | ---: | ---: | ---: |
| `BM_Bootstrap_StandardMt19937Gather` | 10.09 ms | 4.91x | 200.04 k/s |
| `BM_Bootstrap_ResampleTableReused` | 2.06 ms | 1.00x | 972.45 k/s |
| `BM_Bootstrap_CoreIndices` | 3.66 ms | 1.78x | 553.64 k/s |
| `BM_Bootstrap_HandWrittenIndices` | 4.34 ms | 2.11x | 468.04 k/s |
| `BM_Bootstrap_CoreMultinomialCounts` | 4.22 ms | 2.05x | 474.80 k/s |
| `BM_Bootstrap_CorePoissonCounts` | 5.29 ms | 2.57x | 377.86 k/s |

## bpe_benchmark

`benchmarks/ml/bpe_benchmark.cpp`

| benchmark | time | relative | throughput |
| --- | ---: | ---: | ---: |
| `BM_BpeEncode_Core` | 96.77 us | 26.34x |  |
| `BM_BpeEncode_HandWritten` | 112.42 us | 30.60x |  |
| `BM_BpeLongPiece_Rescan/64` | 3.69 us | 1.00x |  |
| `BM_BpeLongPiece_Rescan/256` | 15.86 us | 4.32x |  |
| `BM_BpeLongPiece_Rescan/1024` | 89.73 us | 24.43x |  |
| `BM_BpeLongPiece_Rescan/4096` | 678.43 us | 184.69x |  |
| `BM_BpeLongPiece_Heap/64` | 3.67 us | 1.00x |  |
| `BM_BpeLongPiece_Heap/256` | 15.01 us | 4.09x |  |
| `BM_BpeLongPiece_Heap/1024` | 65.76 us | 17.90x |  |
| `BM_BpeLongPiece_Heap/4096` | 289.35 us | 78.77x |  |

## byte_map_benchmark

`benchmarks/containers/byte_map_benchmark.cpp`

| benchmark | time | relative | throughput |
| --- | ---: | ---: | ---: |
| `BM_Lookup512_ByteMap` | 956.66 us | 1.00x | 68.56 M/s |
| `BM_Lookup512_ByteMapBatched` | 1.02 ms | 1.06x | 65.09 M/s |
| `BM_Lookup512_OpenAddressingByHand` | 1.30 ms | 1.36x | 50.38 M/s |
| `BM_Lookup512_StdUnorderedMap` | 1.97 ms | 2.06x | 33.21 M/s |
| `BM_Lookup100K_ByteMap` | 5.25 ms | 5.49x | 12.62 M/s |
| `BM_Lookup100K_ByteMapBatched` | 1.90 ms | 1.98x | 34.63 M/s |
| `BM_Lookup100K_OpenAddressingByHand` | 2.19 ms | 2.29x | 29.99 M/s |
| `BM_Lookup100K_StdUnorderedMap` | 10.80 ms | 11.29x | 6.91 M/s |

## dense_id_map_benchmark

`benchmarks/containers/dense_id_map_benchmark.cpp`

| benchmark | time | relative | throughput |
| --- | ---: | ---: | ---: |
| `dense_id_map` | 636.77 us | 1.00x | 103.10 M/s |
| `std_unordered_map` | 5.97 ms | 9.37x | 11.51 M/s |

## diloco_benchmark

`benchmarks/ml/diloco_benchmark.cpp`

| benchmark | time | relative | throughput |
| --- | ---: | ---: | ---: |
| `BM_OrderedSum_Core` | 2.22 ms | 2.66x | 474.98 M/s |
| `BM_OrderedSum_HandWritten` | 2.07 ms | 2.48x | 511.93 M/s |
| `BM_OuterStep_Core` | 929.41 us | 1.11x | 1.14 G/s |
| `BM_OuterStep_HandWritten` | 834.46 us | 1.00x | 1.27 G/s |

## flat_id_map_benchmark

`benchmarks/containers/flat_id_map_benchmark.cpp`

| benchmark | time | relative | throughput |
| --- | ---: | ---: | ---: |
| `flat_id_map` | 111.26 us | 1.00x | 36.95 M/s |
| `std_unordered_map` | 643.04 us | 5.78x | 6.43 M/s |

## hash_combine_benchmark

`benchmarks/hash/hash_combine_benchmark.cpp`

| benchmark | time | relative | throughput |
| --- | ---: | ---: | ---: |
| `BM_HashCombine_Core` | 19.70 us | 1.27x | 211.13 M/s |
| `BM_HashCombine_HandWritten` | 15.55 us | 1.00x | 263.59 M/s |

## hierarchical_bitset_benchmark

`benchmarks/containers/hierarchical_bitset_benchmark.cpp`

| benchmark | time | relative | throughput |
| --- | ---: | ---: | ---: |
| `hierarchical_bitset` | 16.76 us | 1.00x | 61.11 M/s |
| `std_vector_bool` | 642.18 us | 38.31x | 1.60 M/s |

## index_pool_benchmark

`benchmarks/containers/index_pool_benchmark.cpp`

| benchmark | time | relative | throughput |
| --- | ---: | ---: | ---: |
| `index_pool` | 22.56 us | 1.00x | 183.38 M/s |
| `vector_free_list` | 31.43 us | 1.39x | 130.86 M/s |
| `object_pool` | 41.35 us | 1.83x | 99.77 M/s |

## inplace_vector_benchmark

`benchmarks/containers/inplace_vector_benchmark.cpp`

| benchmark | time | relative | throughput |
| --- | ---: | ---: | ---: |
| `BM_InplaceVector_Core` | 68.32 us | 1.01x |  |
| `BM_InplaceVector_StdVectorReused` | 77.81 us | 1.15x |  |
| `BM_InplaceVector_ArrayByHand` | 67.70 us | 1.00x |  |

## intrusive_avl_tree_benchmark

`benchmarks/containers/intrusive_avl_tree_benchmark.cpp`

| benchmark | time | relative | throughput |
| --- | ---: | ---: | ---: |
| `intrusive_avl_tree` | 744.35 us | 1.00x | 5.51 M/s |
| `std_set` | 1.27 ms | 1.71x | 3.23 M/s |

## invoke_with_benchmark

`benchmarks/invoke_with_benchmark.cpp`

| benchmark | time | relative | throughput |
| --- | ---: | ---: | ---: |
| `BM_InvokeWith_Core` | 135.35 us | 5.52x | 30.51 M/s |
| `BM_InvokeWith_HandWritten` | 101.69 us | 4.15x | 40.33 M/s |
| `BM_InvokeWith_DirectCall` | 24.52 us | 1.00x | 167.11 M/s |

## linalg_batched_benchmark

`benchmarks/linalg/batched_benchmark.cpp`

| benchmark | time | relative | throughput |
| --- | ---: | ---: | ---: |
| `BM_LinalgBatched_LuCore/25` | 4.82 s | 1.09x | 207.68 k/s |
| `BM_LinalgBatched_LuCore/65` | 49.12 s | 11.14x | 20.58 k/s |
| `BM_LinalgBatched_LuByHand/25` | 4.41 s | 1.00x | 226.87 k/s |
| `BM_LinalgBatched_LuByHand/65` | 77.52 s | 17.58x | 16.66 k/s |
| `BM_LinalgBatched_CholeskyCore/25` | 6.51 s | 1.48x | 223.80 k/s |
| `BM_LinalgBatched_CholeskyCore/65` | 86.82 s | 19.69x | 17.53 k/s |
| `BM_LinalgBatched_CholeskyByHand/25` | 6.47 s | 1.47x | 222.40 k/s |
| `BM_LinalgBatched_CholeskyByHand/65` | 89.94 s | 20.40x | 17.56 k/s |
| `BM_LinalgBatched_LdltCore/25` | 8.39 s | 1.90x | 174.82 k/s |
| `BM_LinalgBatched_LdltCore/65` | 109.07 s | 24.74x | 13.92 k/s |
| `BM_LinalgBatched_LdltByHand/25` | 8.77 s | 1.99x | 178.60 k/s |
| `BM_LinalgBatched_LdltByHand/65` | 56.98 s | 12.92x | 18.34 k/s |

## linalg_randomized_svd_benchmark

`benchmarks/linalg/randomized_svd_benchmark.cpp`

| benchmark | time | relative | throughput |
| --- | ---: | ---: | ---: |
| `BM_TruncatedSvd_K10/500` | 7.69 ms | 1.00x |  |
| `BM_TruncatedSvd_K10/2000` | 56.53 ms | 7.35x |  |
| `BM_ExactSvd/500` | 167.40 ms | 21.77x |  |
| `BM_ExactSvd/2000` | 3.68 s | 479.17x |  |

## linalg_reference_benchmark

`benchmarks/linalg/reference_benchmark.cpp`

| benchmark | time | relative | throughput |
| --- | ---: | ---: | ---: |
| `BM_LinalgDot_Core` | 4.22 us | 2.94x | 973.70 M/s |
| `BM_LinalgDot_ByHand` | 4.19 us | 2.92x | 978.84 M/s |
| `BM_LinalgGemm_Core` | 197.85 us | 137.92x | 1.33 G/s |
| `BM_LinalgGemm_ByHand` | 168.78 us | 117.66x | 1.55 G/s |
| `BM_LinalgLuSolve_Core` | 37.10 us | 25.86x | 26.97 k/s |
| `BM_LinalgLuSolve_ByHand` | 33.61 us | 23.43x | 29.77 k/s |
| `BM_LinalgAxpy_Core` | 1.89 us | 1.32x | 530.65 k/s |
| `BM_LinalgAxpy_ByHand` | 2.37 us | 1.65x | 442.08 k/s |
| `BM_LinalgGemv_Core` | 2.89 us | 2.01x | 348.04 k/s |
| `BM_LinalgGemv_ByHand` | 3.14 us | 2.19x | 319.16 k/s |
| `BM_LinalgSyrk_Core` | 94.19 us | 65.66x | 10.73 k/s |
| `BM_LinalgSyrk_ByHand` | 83.97 us | 58.53x | 11.92 k/s |
| `BM_LinalgTriangularSolve_Core` | 1.51 us | 1.05x | 670.58 k/s |
| `BM_LinalgTriangularSolve_ByHand` | 1.43 us | 1.00x | 699.23 k/s |
| `BM_LinalgLdltSolve_Core` | 47.10 us | 32.84x | 21.29 k/s |
| `BM_LinalgLdltSolve_ByHand` | 44.08 us | 30.73x | 22.69 k/s |
| `BM_LinalgQrQ_Core` | 139.53 us | 97.27x | 7.18 k/s |
| `BM_LinalgQrQ_ByHand` | 158.30 us | 110.35x | 6.34 k/s |
| `BM_LinalgCholeskySolve_Core` | 55.81 us | 38.91x | 21.57 k/s |
| `BM_LinalgCholeskySolve_ByHand` | 46.15 us | 32.17x | 24.48 k/s |
| `BM_LinalgLeastSquares_Core` | 92.49 us | 64.47x | 10.86 k/s |
| `BM_LinalgLeastSquares_ByHand` | 87.94 us | 61.30x | 11.39 k/s |
| `BM_LinalgSymmetricEigen_Core` | 67.54 us | 47.08x | 14.85 k/s |
| `BM_LinalgSymmetricEigen_ByHand` | 61.60 us | 42.94x | 16.26 k/s |
| `BM_LinalgSvd_Core` | 81.52 us | 56.83x | 12.28 k/s |
| `BM_LinalgSvd_ByHand` | 100.02 us | 69.73x | 10.00 k/s |

## linear_benchmark

`benchmarks/ml/linear_benchmark.cpp`

| benchmark | time | relative | throughput |
| --- | ---: | ---: | ---: |
| `BM_LinearSequence_Core/256` | 4.07 ms | 1.00x | 15.83 k/s |
| `BM_LinearSequence_Core/1024` | 16.17 ms | 3.97x | 3.99 k/s |
| `BM_LinearSequence_HandWritten/256` | 4.23 ms | 1.04x | 15.74 k/s |
| `BM_LinearSequence_HandWritten/1024` | 17.91 ms | 4.40x | 3.64 k/s |
| `BM_LinearSequenceBackward_Core/256` | 8.63 ms | 2.12x | 7.45 k/s |
| `BM_LinearSequenceBackward_Core/1024` | 42.99 ms | 10.56x | 1.53 k/s |
| `BM_LinearSequenceBackward_HandWritten/256` | 9.23 ms | 2.27x | 7.09 k/s |
| `BM_LinearSequenceBackward_HandWritten/1024` | 40.39 ms | 9.92x | 1.59 k/s |
| `BM_LinearPerToken_Context/256` | 4.09 ms | 1.01x | 15.82 k/s |
| `BM_LinearPerToken_Context/1024` | 15.63 ms | 3.84x | 4.10 k/s |

## mapped_header_benchmark

`benchmarks/containers/mapped_header_benchmark.cpp`

| benchmark | time | relative | throughput |
| --- | ---: | ---: | ---: |
| `mapped_header_round_trip` | 18.64 us | 1.06x | 55.28 M/s |
| `memcpy_wyhash_round_trip` | 17.60 us | 1.00x | 58.31 M/s |

## md_array_benchmark

`benchmarks/containers/md_array_benchmark.cpp`

| benchmark | time | relative | throughput |
| --- | ---: | ---: | ---: |
| `BM_MdArray_Core` | 68.23 us | 1.02x |  |
| `BM_MdArray_OffsetsByHand` | 68.00 us | 1.02x |  |
| `BM_MdSpan_RowMajor` | 67.97 us | 1.02x | 965.34 M/s |
| `BM_MdSpan_RowMajorByHand` | 67.89 us | 1.02x | 966.24 M/s |
| `BM_MdSpan_ColumnMajor` | 68.77 us | 1.03x | 956.93 M/s |
| `BM_MdSpan_ColumnMajorByHand` | 67.96 us | 1.02x | 966.40 M/s |
| `BM_MdSpan_Strided` | 66.89 us | 1.00x | 980.09 M/s |
| `BM_MdSpan_StridedByHand` | 67.17 us | 1.00x | 976.33 M/s |

## minifloat_benchmark

`benchmarks/ml/minifloat_benchmark.cpp`

| benchmark | time | relative | throughput |
| --- | ---: | ---: | ---: |
| `BM_ToBfloat16_Core` | 22.20 us | 3.16x | 2.95 G/s |
| `BM_ToBfloat16_HandWritten` | 22.68 us | 3.22x | 2.89 G/s |
| `BM_Bfloat16ToFloat_Core` | 7.03 us | 1.00x | 9.33 G/s |
| `BM_Bfloat16ToFloat_HandWritten` | 7.08 us | 1.01x | 9.26 G/s |
| `BM_ToFloat8E4M3_Generic` | 1.21 ms | 172.31x | 54.19 M/s |

## norm_benchmark

`benchmarks/ml/norm_benchmark.cpp`

| benchmark | time | relative | throughput |
| --- | ---: | ---: | ---: |
| `BM_LayerNorm_Core` | 900.48 us | 1.99x | 436.97 M/s |
| `BM_LayerNorm_HandWritten` | 969.84 us | 2.15x | 422.31 M/s |
| `BM_RmsNorm_Core` | 471.31 us | 1.04x | 836.45 M/s |
| `BM_RmsNorm_HandWritten` | 451.48 us | 1.00x | 871.34 M/s |

## normal_benchmark

`benchmarks/stats/normal_benchmark.cpp`

| benchmark | time | relative | throughput |
| --- | ---: | ---: | ---: |
| `BM_NormalInverseCdf_Core` | 72.35 ms | 5.69x | 18.21 M/s |
| `BM_NormalInverseCdf_HandWritten` | 48.22 ms | 3.80x | 21.75 M/s |
| `BM_NormalBoxMuller_Core` | 12.70 ms | 1.00x | 82.56 M/s |
| `BM_NormalBoxMuller_HandWritten` | 13.32 ms | 1.05x | 78.78 M/s |

## object_pool_benchmark

`benchmarks/containers/object_pool_benchmark.cpp`

| benchmark | time | relative | throughput |
| --- | ---: | ---: | ---: |
| `object_pool` | 35.69 us | 1.03x | 114.88 M/s |
| `object_pool_pool_resource` | 34.50 us | 1.00x | 118.74 M/s |
| `new_delete` | 617.82 us | 17.91x | 6.63 M/s |

## offset_ptr_benchmark

`benchmarks/containers/offset_ptr_benchmark.cpp`

| benchmark | time | relative | throughput |
| --- | ---: | ---: | ---: |
| `offset_ptr_walk` | 16.69 us | 1.25x | 245.99 M/s |
| `raw_pointer_walk` | 13.36 us | 1.00x | 306.67 M/s |

## optim_benchmark

`benchmarks/ml/optim_benchmark.cpp`

| benchmark | time | relative | throughput |
| --- | ---: | ---: | ---: |
| `BM_AdamW_Core` | 1.30 ms | 1.02x | 808.98 M/s |
| `BM_AdamW_HandWritten` | 1.27 ms | 1.00x | 824.98 M/s |

## perfect_map_benchmark

`benchmarks/containers/perfect_map_benchmark.cpp`

| benchmark | time | relative | throughput |
| --- | ---: | ---: | ---: |
| `BM_Lookup_PerfectMap` | 209.31 us | 1.00x | 156.63 M/s |
| `BM_Lookup_StaticMap` | 2.74 ms | 13.11x | 11.94 M/s |
| `BM_Lookup_BinarySearchByHand` | 2.77 ms | 13.25x | 11.82 M/s |
| `BM_Lookup_StdUnorderedMap` | 680.30 us | 3.25x | 48.18 M/s |

## pivot_benchmark

`benchmarks/stats/pivot_benchmark.cpp`

| benchmark | time | relative | throughput |
| --- | ---: | ---: | ---: |
| `BM_Pivot_EnumKeysCore` | 5.54 ms | 1.07x | 182.78 M/s |
| `BM_Pivot_EnumKeysByHand` | 5.18 ms | 1.00x | 193.21 M/s |
| `BM_Pivot_StringKeysCore` | 35.81 ms | 6.91x | 27.95 M/s |
| `BM_Pivot_StringKeysByHand` | 31.73 ms | 6.13x | 31.53 M/s |

## quantize_benchmark

`benchmarks/ml/quantize_benchmark.cpp`

| benchmark | time | relative | throughput |
| --- | ---: | ---: | ---: |
| `BM_DotQ8_0_Core` | 1.77 us | 1.13x | 2.35 G/s |
| `BM_DotQ8_0_HandWritten` | 1.56 us | 1.00x | 2.62 G/s |
| `BM_DotQ4_0_Core` | 2.27 us | 1.45x | 1.81 G/s |
| `BM_DotQ4_0_HandWritten` | 2.16 us | 1.38x | 1.90 G/s |
| `BM_DotMxfp4_Core` | 1.61 us | 1.03x | 2.54 G/s |
| `BM_DotMxfp4_HandWritten` | 1.64 us | 1.05x | 2.51 G/s |
| `BM_QuantizeQ8_0_Core` | 6.86 us | 4.39x | 597.96 M/s |
| `BM_QuantizeQ8_0_HandWritten` | 14.83 us | 9.48x | 280.26 M/s |

## result_benchmark

`benchmarks/result_benchmark.cpp`

| benchmark | time | relative | throughput |
| --- | ---: | ---: | ---: |
| `result_core/0` | 9.40 us | 1.82x | 435.78 M/s |
| `result_core/1` | 9.80 us | 1.90x | 418.10 M/s |
| `result_core/10` | 11.46 us | 2.22x | 357.37 M/s |
| `result_core/50` | 22.02 us | 4.26x | 186.03 M/s |
| `result_core/100` | 11.55 us | 2.24x | 354.67 M/s |
| `result_hand_tagged/0` | 5.27 us | 1.02x | 776.85 M/s |
| `result_hand_tagged/1` | 5.54 us | 1.07x | 739.57 M/s |
| `result_hand_tagged/10` | 8.75 us | 1.69x | 472.26 M/s |
| `result_hand_tagged/50` | 18.58 us | 3.59x | 220.64 M/s |
| `result_hand_tagged/100` | 11.36 us | 2.20x | 360.95 M/s |
| `result_error_code/0` | 6.82 us | 1.32x | 611.37 M/s |
| `result_error_code/1` | 6.81 us | 1.32x | 601.36 M/s |
| `result_error_code/10` | 9.59 us | 1.86x | 441.81 M/s |
| `result_error_code/50` | 20.53 us | 3.97x | 199.57 M/s |
| `result_error_code/100` | 11.21 us | 2.17x | 365.80 M/s |
| `result_exceptions/0` | 5.17 us | 1.00x | 792.66 M/s |
| `result_exceptions/1` | 203.94 us | 39.46x | 20.09 M/s |
| `result_exceptions/10` | 2.04 ms | 394.96x | 2.01 M/s |
| `result_exceptions/50` | 9.93 ms | 1922.17x | 412.44 k/s |
| `result_exceptions/100` | 19.70 ms | 3811.25x | 208.01 k/s |
| `result_traced_success/0` | 5.27 us | 1.02x | 778.19 M/s |
| `result_core_propagate/0` | 14.70 us | 2.84x | 278.81 M/s |
| `result_core_propagate/1` | 15.52 us | 3.00x | 264.19 M/s |
| `result_core_propagate/10` | 18.57 us | 3.59x | 221.85 M/s |
| `result_core_propagate/50` | 27.80 us | 5.38x | 147.49 M/s |
| `result_core_propagate/100` | 15.70 us | 3.04x | 260.99 M/s |
| `result_hand_propagate/0` | 9.75 us | 1.89x | 421.96 M/s |
| `result_hand_propagate/1` | 9.65 us | 1.87x | 424.77 M/s |
| `result_hand_propagate/10` | 11.95 us | 2.31x | 342.90 M/s |
| `result_hand_propagate/50` | 20.22 us | 3.91x | 202.56 M/s |
| `result_hand_propagate/100` | 12.41 us | 2.40x | 330.13 M/s |

## self_attention_benchmark

`benchmarks/ml/self_attention_benchmark.cpp`

| benchmark | time | relative | throughput |
| --- | ---: | ---: | ---: |
| `BM_SelfAttentionHead_Core/128` | 3.28 ms | 1.85x | 39.24 k/s |
| `BM_SelfAttentionHead_Core/512` | 22.69 ms | 12.81x | 22.59 k/s |
| `BM_SelfAttentionHead_HandWritten/128` | 3.18 ms | 1.80x | 40.33 k/s |
| `BM_SelfAttentionHead_HandWritten/512` | 21.71 ms | 12.26x | 23.62 k/s |
| `BM_SelfAttentionHeadBackward_Core/128` | 2.08 ms | 1.17x | 62.63 k/s |
| `BM_SelfAttentionHeadBackward_Core/512` | 21.87 ms | 12.35x | 23.54 k/s |
| `BM_SelfAttentionHeadBackward_HandWritten/128` | 1.77 ms | 1.00x | 72.30 k/s |
| `BM_SelfAttentionHeadBackward_HandWritten/512` | 21.50 ms | 12.14x | 23.84 k/s |

## shapley_benchmark

`benchmarks/game/shapley_benchmark.cpp`

| benchmark | time | relative | throughput |
| --- | ---: | ---: | ---: |
| `BM_Shapley_Core` | 486.77 us | 1.00x |  |
| `BM_Shapley_Formula3ByHand` | 1.50 ms | 3.08x |  |

## softmax_benchmark

`benchmarks/ml/softmax_benchmark.cpp`

| benchmark | time | relative | throughput |
| --- | ---: | ---: | ---: |
| `BM_Softmax_Core` | 310.66 us | 1.00x | 163.95 M/s |
| `BM_Softmax_HandWritten` | 323.35 us | 1.04x | 159.11 M/s |
| `BM_CrossEntropyWithGradient_Core` | 524.51 us | 1.69x | 95.98 M/s |
| `BM_CrossEntropyWithGradient_HandWritten` | 511.22 us | 1.65x | 98.42 M/s |

## stages_benchmark

`benchmarks/pipeline/stages_benchmark.cpp`

| benchmark | time | relative | throughput |
| --- | ---: | ---: | ---: |
| `stage/chain` | 9.46 us | 1.32x | 113.40 M/s |
| `stage/hand_written` | 7.14 us | 1.00x | 143.42 M/s |

## struct_tuple_benchmark

`benchmarks/reflect/struct_tuple_benchmark.cpp`

| benchmark | time | relative | throughput |
| --- | ---: | ---: | ---: |
| `BM_ToTuple_Core` | 4.51 us | 1.28x | 975.31 M/s |
| `BM_ToTuple_HandWritten` | 3.99 us | 1.13x | 1.03 G/s |
| `BM_Tie_Core` | 3.52 us | 1.00x | 1.16 G/s |
| `BM_Tie_HandWritten` | 3.57 us | 1.01x | 1.15 G/s |
| `BM_FromTuple_Core` | 4.16 us | 1.18x | 993.66 M/s |
| `BM_FromTuple_HandWritten` | 4.00 us | 1.13x | 1.03 G/s |

## transformer_block_benchmark

`benchmarks/ml/transformer_block_benchmark.cpp`

| benchmark | time | relative | throughput |
| --- | ---: | ---: | ---: |
| `BM_DecodeGpt2Layer_Float_Core` | 7.68 ms | 2.50x | 130.4 /s |
| `BM_DecodeGpt2Layer_Float_HandWritten` | 7.66 ms | 2.50x | 130.5 /s |
| `BM_DecodeGpt2Layer_Q8_0_Core` | 3.25 ms | 1.06x | 310.9 /s |
| `BM_DecodeGpt2Layer_Q8_0_HandWritten` | 3.07 ms | 1.00x | 325.7 /s |
| `BM_TrainGpt2ShapedLayerForward_Core` | 49.80 ms | 16.22x | 1.29 k/s |
| `BM_TrainGpt2ShapedLayerForward_HandWritten` | 48.93 ms | 15.94x | 1.31 k/s |
| `BM_TrainGpt2ShapedLayerBackward_Core` | 20.27 ms | 6.60x | 3.16 k/s |
| `BM_TrainGpt2ShapedLayerBackward_HandWritten` | 19.15 ms | 6.24x | 3.34 k/s |

## uint256_benchmark

`benchmarks/uint256_benchmark.cpp`

| benchmark | time | relative | throughput |
| --- | ---: | ---: | ---: |
| `BM_Uint256_Add_Core` | 9.23 us | 1.06x | 457.13 M/s |
| `BM_Uint256_Add_HandWritten` | 8.72 us | 1.00x | 469.98 M/s |
| `BM_Uint256_Mul_Core` | 18.64 us | 2.14x | 219.95 M/s |
| `BM_Uint256_Mul_HandWritten` | 18.73 us | 2.15x | 218.90 M/s |
| `BM_Uint256_Divmod_Core` | 191.36 us | 21.95x | 21.41 M/s |
| `BM_Uint256_ToString_Core` | 2.25 ms | 257.84x | 1.82 M/s |

## utc_benchmark

`benchmarks/time/utc_benchmark.cpp`

| benchmark | time | relative | throughput |
| --- | ---: | ---: | ---: |
| `format_core_to_chars` | 120.49 us | 9.02x | 34.00 M/s |
| `format_std_format_to_n` | 6.20 ms | 463.85x | 663.03 k/s |
| `format_gmtime_snprintf` | 2.48 ms | 185.89x | 1.65 M/s |
| `format_gmtime_strftime` | 5.00 ms | 374.37x | 834.25 k/s |
| `format_string_core` | 451.97 us | 33.82x | 9.12 M/s |
| `format_string_std` | 9.07 ms | 678.65x | 466.36 k/s |
| `format_string_original_ostream` | 8.41 ms | 629.60x | 488.18 k/s |
| `parse_core_from_chars` | 158.20 us | 11.84x | 25.97 M/s |
| `parse_original_exact` | 188.98 us | 14.14x | 21.74 M/s |
| `parse_sscanf_timegm` | 37.46 ms | 2803.60x | 109.92 k/s |
| `parse_istream_get_time` | 34.28 ms | 2565.52x | 119.80 k/s |
| `calendar_core_civil_from_days` | 41.77 us | 3.13x | 98.52 M/s |
| `calendar_std_year_month_day` | 44.20 us | 3.31x | 93.76 M/s |
| `unix_ms_core` | 13.36 us | 1.00x | 308.27 M/s |
| `unix_ms_std_chrono` | 13.71 us | 1.03x | 300.37 M/s |

## wyhash_benchmark

`benchmarks/hash/wyhash_benchmark.cpp`

| benchmark | time | relative | throughput |
| --- | ---: | ---: | ---: |
| `BM_Wyhash_Core` | 42.60 us | 1.13x | 96.34 M/s |
| `BM_Wyhash_HandWritten` | 37.54 us | 1.00x | 109.29 M/s |
| `BM_Wyhash_Fnv1a64ForScale` | 183.48 us | 4.89x | 22.47 M/s |

## zip_benchmark

`benchmarks/zip_benchmark.cpp`

| benchmark | time | relative | throughput |
| --- | ---: | ---: | ---: |
| `BM_Zip_Core` | 4.42 us | 1.03x | 927.77 M/s |
| `BM_Zip_HandWritten` | 4.31 us | 1.00x | 951.30 M/s |
