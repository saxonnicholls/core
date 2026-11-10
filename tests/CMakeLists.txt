# Unit tests. The layout mirrors include/snicholls/core/: <sub>/<name>.hpp has
# <sub>/<name>_test.cpp here, and scripts/check_mirror.py fails CI if one is
# missing or not listed below.
add_executable(snicholls_core_tests
    acout_test.cpp
    backoff_test.cpp
    basic_type_registry_test.cpp
    checked_test.cpp
    colour_test.cpp
    constexpr_for_test.cpp
    constexpr_map_test.cpp
    decimal_test.cpp
    display_test.cpp
    endian_test.cpp
    fixed_string_test.cpp
    function_ref_test.cpp
    invoke_with_test.cpp
    iterate_test.cpp
    overloaded_test.cpp
    result_terminate_test.cpp
    result_test.cpp
    secret_test.cpp
    strong_test.cpp
    thread_apply_test.cpp
    traced_test.cpp
    uint256_test.cpp
    uuid_test.cpp
    version_test.cpp
    zip_test.cpp
    combinatorics/partition_test.cpp
    combinatorics/permutations_test.cpp
    combinatorics/subsets_test.cpp
    containers/bloom_filter_test.cpp
    containers/dense_id_map_test.cpp
    containers/flat_id_map_test.cpp
    containers/hierarchical_bitset_test.cpp
    containers/inplace_vector_test.cpp
    containers/intrusive_avl_tree_test.cpp
    containers/md_array_test.cpp
    containers/object_pool_test.cpp
    containers/perfect_map_test.cpp
    containers/static_map_test.cpp
    detail/forward_like_test.cpp
    detail/pp_for_each_test.cpp
    detail/sha1_test.cpp
    detail/sobol_directions_test.cpp
    detail/struct_binders_test.cpp
    detail/wide_mul_test.cpp
    finance/attribution_test.cpp
    game/change_test.cpp
    game/coalition_test.cpp
    game/harsanyi_test.cpp
    game/owen_test.cpp
    game/properties_test.cpp
    game/shapley_test.cpp
    game/tabular_test.cpp
    hash/fnv_test.cpp
    hash/hash_combine_test.cpp
    hash/wyhash_test.cpp
    math/constexpr_math_test.cpp
    meta/assignable_test.cpp
    meta/attributes_test.cpp
    meta/callable_traits_test.cpp
    meta/compare_test.cpp
    meta/concepts_test.cpp
    meta/enums_test.cpp
    meta/type_list_test.cpp
    meta/type_name_test.cpp
    meta/values_test.cpp
    pipeline/stages_test.cpp
    reflect/member_count_test.cpp
    reflect/string_access_test.cpp
    reflect/string_traits_test.cpp
    reflect/struct_tuple_test.cpp
    stats/accumulators_test.cpp
    stats/bootstrap_test.cpp
    stats/covariance_test.cpp
    stats/low_discrepancy_test.cpp
    stats/pivot_test.cpp
    stats/random_test.cpp
    stochastic/martingale_test.cpp
    stochastic/snell_test.cpp
    time/business_days_test.cpp
    time/civil_test.cpp
    time/rfc3339_test.cpp
    time/solar_test.cpp
    time/utc_format_test.cpp
    time/utc_test.cpp)
target_link_libraries(snicholls_core_tests PRIVATE snicholls::core GTest::gtest_main GTest::gmock)
# The portability fixtures (tests/fixtures/), read at run time by the same tests on every platform
target_compile_definitions(snicholls_core_tests PRIVATE SNICHOLLS_CORE_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures")
snicholls_core_compile_options(snicholls_core_tests)

include(GoogleTest)

# Allocation-failure injection needs a replaced global operator new
# (support/failing_new.cpp), which must not reach the main test binary: its
# own executable. Companion tests of the container headers they prove.
add_executable(snicholls_core_allocation_tests
    containers/dense_id_map_allocation_test.cpp
    containers/object_pool_allocation_test.cpp
    support/failing_new.cpp)
target_link_libraries(snicholls_core_allocation_tests PRIVATE snicholls::core GTest::gtest_main)
snicholls_core_compile_options(snicholls_core_allocation_tests)
gtest_discover_tests(snicholls_core_allocation_tests PROPERTIES LABELS unit DISCOVERY_MODE PRE_TEST)
gtest_discover_tests(snicholls_core_tests PROPERTIES LABELS unit DISCOVERY_MODE PRE_TEST)

# The diamond rules are CMake, so they are tested with CMake: a fixture project
# reaches core twice and the script checks that the right copy wins, that an
# unmet minimum fails configure, and that two versions are refused.
add_test(NAME cmake.diamond
    COMMAND ${CMAKE_COMMAND}
        -DCORE_SOURCE_DIR=${PROJECT_SOURCE_DIR}
        -DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/diamond
        -DCXX=${CMAKE_CXX_COMPILER}
        -DGENERATOR=${CMAKE_GENERATOR}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/diamond_test.cmake)
set_tests_properties(cmake.diamond PROPERTIES LABELS boundary TIMEOUT 900)

# Generated code must match its generator: detail/struct_binders.hpp is written
# by scripts/generate_struct_binders.py and never edited by hand.
find_package(Python3 COMPONENTS Interpreter REQUIRED)
add_test(NAME generated.struct_binders_up_to_date
    COMMAND Python3::Interpreter ${PROJECT_SOURCE_DIR}/scripts/generate_struct_binders.py --check)
set_tests_properties(generated.struct_binders_up_to_date PROPERTIES LABELS boundary)

# The time portability fixtures (tests/fixtures/time/) must be exactly what
# scripts/generate_time_fixtures.py writes from Python's datetime and exact integer arithmetic.
add_test(NAME generated.time_fixtures_up_to_date
    COMMAND Python3::Interpreter ${PROJECT_SOURCE_DIR}/scripts/generate_time_fixtures.py --check)
set_tests_properties(generated.time_fixtures_up_to_date PROPERTIES LABELS boundary)

# Compile-fail tests. Each case in compile_fail/ is one source built twice, as
# an object library outside ALL: with SNICHOLLS_COMPILE_FAIL defined it must
# fail, and the test passes only if the build output matches the expected
# message (PASS_REGULAR_EXPRESSION, so a build that fails for another reason
# does not pass). Without it the same source is the positive control and must
# build, so a broken harness, or a case that fails for a reason unrelated to
# the rule, cannot pass silently. The build runs through the generator
# (cmake --build), never by calling the compiler directly, so it gets the
# compiler's environment on every platform, MSVC included, and the same flags
# as every other target.
set(_compile_fail_cases
    "struct_tuple_size_mismatch|snicholls::from_tuple: tuple size must equal the struct's number of members"
    "struct_tuple_narrowing_int_to_double|snicholls::from_tuple: every element must convert to its member without narrowing"
    "struct_tuple_narrowing_int_to_bool|snicholls::from_tuple: every element must convert to its member without narrowing"
    "member_count_reference_member|snicholls::member_count: cannot count members - reference members are not supported"
    "member_count_over_limit|snicholls::member_count: struct has more than SNICHOLLS_MAX_STRUCT_MEMBERS members"
    "struct_binders_over_generated|SNICHOLLS_MAX_STRUCT_MEMBERS exceeds the generated binders"
    "string_access_unterminated_array|snicholls_character_array_is_not_nul_terminated"
    "invoke_with_generic_lambda|snicholls::invoke_with: every callable needs exactly one non-template operator"
    "invoke_with_mutable_reference|snicholls::invoke_with: a parameter cannot be a non-const lvalue reference"
    "acout_unprintable|snicholls: no way to print this type"
    "enums_unfixed|snicholls::enum reflection: give the enumeration a fixed underlying type"
    "permutation_count_overflow|snicholls::permutation_count: the count does not fit"
    "thread_t_lengths|snicholls::meta::thread_t: the threaded arguments must all be the same length"
    "thread_t_position|snicholls::meta::thread_t: the position specification does not fit"
    "thread_apply_lengths|snicholls::thread_apply: the threaded tuples must all be the same length"
    "thread_apply_mixed_void|snicholls::thread_apply: the calls must all return void or all return a value")
foreach(_case_and_regex IN LISTS _compile_fail_cases)
    string(REPLACE "|" ";" _parts "${_case_and_regex}")
    list(GET _parts 0 _case)
    list(GET _parts 1 _regex)
    foreach(_variant IN ITEMS fail control)
        set(_target snicholls_core_compile_fail_${_case}_${_variant})
        add_library(${_target} OBJECT EXCLUDE_FROM_ALL compile_fail/${_case}.cpp)
        target_link_libraries(${_target} PRIVATE snicholls::core)
        snicholls_core_compile_options(${_target})
        if(_variant STREQUAL "fail")
            target_compile_definitions(${_target} PRIVATE SNICHOLLS_COMPILE_FAIL)
            set(_test compile_fail.${_case})
        else()
            set(_test compile_fail.${_case}.control)
        endif()
        add_test(NAME ${_test}
            COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target ${_target} --config $<CONFIG>)
        # One build at a time in this tree, whatever ctest -j says
        set_tests_properties(${_test} PROPERTIES LABELS boundary RESOURCE_LOCK snicholls_core_build_tree)
        if(_variant STREQUAL "fail")
            set_tests_properties(${_test} PROPERTIES PASS_REGULAR_EXPRESSION "${_regex}")
        endif()
    endforeach()
endforeach()
