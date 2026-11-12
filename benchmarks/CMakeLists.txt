# Benchmarks (Google Benchmark). The layout mirrors include/snicholls/core/:
# <sub>/<name>.hpp has <sub>/<name>_benchmark.cpp here. Only Release numbers
# mean anything; a short smoke run of each is a CTest (label benchmark) so a
# benchmark that stops compiling is noticed before a release quotes it.
#
# Every group is like-for-like against hand-written code: identical data,
# identical arithmetic in the same order, and a fairness check that refuses to
# benchmark if any pair disagrees (CLAUDE.md "Zero overhead").
#
# Each benchmark has its own main(), which runs the fairness check before
# handing over to Google Benchmark, so it links benchmark::benchmark rather
# than benchmark_main.
function(snicholls_core_add_benchmark name source)
    add_executable(snicholls_core_${name} ${source})
    target_link_libraries(snicholls_core_${name} PRIVATE snicholls::core benchmark::benchmark)
    target_compile_definitions(snicholls_core_${name} PRIVATE NDEBUG)
    snicholls_core_test_options(snicholls_core_${name})
    add_test(NAME benchmark.${name} COMMAND snicholls_core_${name} --benchmark_min_time=0.01s)
    # A smoke run takes seconds; a benchmark that hangs fails here instead of holding a CI job until it is killed
    set_tests_properties(benchmark.${name} PROPERTIES LABELS benchmark TIMEOUT 300)
endfunction()

snicholls_core_add_benchmark(acout_benchmark acout_benchmark.cpp)
snicholls_core_add_benchmark(invoke_with_benchmark invoke_with_benchmark.cpp)
snicholls_core_add_benchmark(result_benchmark result_benchmark.cpp)
snicholls_core_add_benchmark(uint256_benchmark uint256_benchmark.cpp)
snicholls_core_add_benchmark(zip_benchmark zip_benchmark.cpp)
snicholls_core_add_benchmark(bloom_filter_benchmark containers/bloom_filter_benchmark.cpp)
snicholls_core_add_benchmark(byte_map_benchmark containers/byte_map_benchmark.cpp)
snicholls_core_add_benchmark(dense_id_map_benchmark containers/dense_id_map_benchmark.cpp)
snicholls_core_add_benchmark(flat_id_map_benchmark containers/flat_id_map_benchmark.cpp)
snicholls_core_add_benchmark(hierarchical_bitset_benchmark containers/hierarchical_bitset_benchmark.cpp)
snicholls_core_add_benchmark(inplace_vector_benchmark containers/inplace_vector_benchmark.cpp)
snicholls_core_add_benchmark(intrusive_avl_tree_benchmark containers/intrusive_avl_tree_benchmark.cpp)
snicholls_core_add_benchmark(md_array_benchmark containers/md_array_benchmark.cpp)
snicholls_core_add_benchmark(object_pool_benchmark containers/object_pool_benchmark.cpp)
snicholls_core_add_benchmark(perfect_map_benchmark containers/perfect_map_benchmark.cpp)
snicholls_core_add_benchmark(shapley_benchmark game/shapley_benchmark.cpp)
snicholls_core_add_benchmark(hash_combine_benchmark hash/hash_combine_benchmark.cpp)
snicholls_core_add_benchmark(wyhash_benchmark hash/wyhash_benchmark.cpp)
snicholls_core_add_benchmark(attributes_benchmark meta/attributes_benchmark.cpp)
snicholls_core_add_benchmark(linalg_batched_benchmark linalg/batched_benchmark.cpp)
# 10^6 systems take minutes; the smoke run solves 4096 (the benchmark reads this variable)
set_tests_properties(benchmark.linalg_batched_benchmark PROPERTIES ENVIRONMENT SNICHOLLS_CORE_BENCHMARK_SMOKE=1)
snicholls_core_add_benchmark(linalg_reference_benchmark linalg/reference_benchmark.cpp)
snicholls_core_add_benchmark(activations_benchmark ml/activations_benchmark.cpp)
snicholls_core_add_benchmark(attention_benchmark ml/attention_benchmark.cpp)
snicholls_core_add_benchmark(bpe_benchmark ml/bpe_benchmark.cpp)
snicholls_core_add_benchmark(minifloat_benchmark ml/minifloat_benchmark.cpp)
snicholls_core_add_benchmark(norm_benchmark ml/norm_benchmark.cpp)
snicholls_core_add_benchmark(optim_benchmark ml/optim_benchmark.cpp)
snicholls_core_add_benchmark(quantize_benchmark ml/quantize_benchmark.cpp)
snicholls_core_add_benchmark(softmax_benchmark ml/softmax_benchmark.cpp)
snicholls_core_add_benchmark(transformer_block_benchmark ml/transformer_block_benchmark.cpp)
snicholls_core_add_benchmark(stages_benchmark pipeline/stages_benchmark.cpp)
snicholls_core_add_benchmark(struct_tuple_benchmark reflect/struct_tuple_benchmark.cpp)
snicholls_core_add_benchmark(accumulators_benchmark stats/accumulators_benchmark.cpp)
snicholls_core_add_benchmark(bootstrap_benchmark stats/bootstrap_benchmark.cpp)
snicholls_core_add_benchmark(pivot_benchmark stats/pivot_benchmark.cpp)
# The baselines it compares against are the C library's POSIX calls (gmtime_r,
# timegm), which MSVC does not have.
if(NOT WIN32)
    snicholls_core_add_benchmark(utc_benchmark time/utc_benchmark.cpp)
endif()

# The BPE benchmark reads the committed tokenizer fixture
target_compile_definitions(snicholls_core_bpe_benchmark PRIVATE SNICHOLLS_CORE_FIXTURE_DIR="${PROJECT_SOURCE_DIR}/tests/fixtures")
target_compile_definitions(snicholls_core_byte_map_benchmark PRIVATE SNICHOLLS_CORE_FIXTURE_DIR="${PROJECT_SOURCE_DIR}/tests/fixtures")
