# snicholls-core - tests for the diamond rules in CMakeLists.txt.
#
# Run as the CTest `cmake.diamond` by tests/CMakeLists.txt, with
#   CORE_SOURCE_DIR   this checkout
#   WORK_DIR          a scratch directory, wiped first
#   CXX, GENERATOR    the compiler and generator of the enclosing build
#
# A fixture project adds core from two directories - this checkout, and a copy
# whose project() version is patched to 0.0.1 - and the configure output is
# checked against each rule. One case also builds a consumer executable, so the
# target is known to be usable from a project that reached core twice.

cmake_minimum_required(VERSION 3.20)

foreach(_var CORE_SOURCE_DIR WORK_DIR CXX GENERATOR)
    if(NOT DEFINED ${_var})
        message(FATAL_ERROR "diamond_test: ${_var} is not set")
    endif()
endforeach()

file(READ "${CORE_SOURCE_DIR}/CMakeLists.txt" _real_cmakelists)
if(NOT _real_cmakelists MATCHES "project\\(snicholls_core[ \t\r\n]+VERSION ([0-9]+\\.[0-9]+\\.[0-9]+)")
    message(FATAL_ERROR "diamond_test: could not read the version out of ${CORE_SOURCE_DIR}/CMakeLists.txt")
endif()
set(CORE_VERSION "${CMAKE_MATCH_1}")

file(REMOVE_RECURSE "${WORK_DIR}")
file(MAKE_DIRECTORY "${WORK_DIR}")

# Two more copies of core: one identical, one at version 0.0.1. Only the build
# files and headers are needed.
set(SAME_COPY "${WORK_DIR}/core-same")
file(COPY "${CORE_SOURCE_DIR}/cmake" "${CORE_SOURCE_DIR}/include" DESTINATION "${SAME_COPY}")
file(WRITE "${SAME_COPY}/CMakeLists.txt" "${_real_cmakelists}")

set(OLD_COPY "${WORK_DIR}/core-0.0.1")
file(COPY "${CORE_SOURCE_DIR}/cmake" "${CORE_SOURCE_DIR}/include" DESTINATION "${OLD_COPY}")
string(REGEX REPLACE "(project\\(snicholls_core[ \t\r\n]+VERSION )[0-9]+\\.[0-9]+\\.[0-9]+" "\\10.0.1"
    _old_cmakelists "${_real_cmakelists}")
file(WRITE "${OLD_COPY}/CMakeLists.txt" "${_old_cmakelists}")

set(REAL "${CORE_SOURCE_DIR}")

# Configure a fixture whose CMakeLists.txt body is `body`, expect success or
# failure, and leave the combined output in LAST_LOG for the expect_* checks.
function(run_case name expect_success body)
    set(_src "${WORK_DIR}/${name}/src")
    set(_bin "${WORK_DIR}/${name}/build")
    file(MAKE_DIRECTORY "${_src}")
    file(WRITE "${_src}/CMakeLists.txt"
        "cmake_minimum_required(VERSION 3.20)\nproject(fixture_${name} LANGUAGES CXX)\n${body}\n")
    file(WRITE "${_src}/main.cpp"
        "#include <snicholls/core/version.hpp>\n"
        "SNICHOLLS_CORE_REQUIRE_VERSION(0, 1, 0);\n"
        "int main() { return snicholls::core::version_major() >= 0 ? 0 : 1; }\n")
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -S "${_src}" -B "${_bin}" -G "${GENERATOR}" "-DCMAKE_CXX_COMPILER=${CXX}"
        RESULT_VARIABLE _rc
        OUTPUT_VARIABLE _out
        ERROR_VARIABLE _err)
    set(_log "${_out}${_err}")
    file(WRITE "${WORK_DIR}/${name}/configure.log" "${_log}")
    if(expect_success AND NOT _rc EQUAL 0)
        message(FATAL_ERROR "diamond.${name}: expected configure to succeed, it failed (${_rc}):\n${_log}")
    elseif(NOT expect_success AND _rc EQUAL 0)
        message(FATAL_ERROR "diamond.${name}: expected configure to fail, it succeeded:\n${_log}")
    endif()
    set(LAST_LOG "${_log}" PARENT_SCOPE)
    set(LAST_BIN "${_bin}" PARENT_SCOPE)
    message(STATUS "diamond.${name}: configure exit ${_rc}, as expected")
endfunction()

# CMake wraps the text of message(WARNING) and message(FATAL_ERROR) at about 80
# columns, so both sides are compared with runs of whitespace collapsed to one space.
function(flatten out text)
    string(REGEX REPLACE "[ \t\r\n]+" " " _flat "${text}")
    set(${out} "${_flat}" PARENT_SCOPE)
endfunction()

function(expect_in_log name needle)
    flatten(_log "${LAST_LOG}")
    flatten(needle "${needle}")
    string(FIND "${_log}" "${needle}" _pos)
    if(_pos EQUAL -1)
        message(FATAL_ERROR "diamond.${name}: expected the configure output to contain\n  '${needle}'\nbut it was:\n${LAST_LOG}")
    endif()
endfunction()

function(expect_not_in_log name needle)
    flatten(_log "${LAST_LOG}")
    string(FIND "${_log}" "${needle}" _pos)
    if(NOT _pos EQUAL -1)
        message(FATAL_ERROR "diamond.${name}: expected the configure output NOT to contain\n  '${needle}'\nbut it was:\n${LAST_LOG}")
    endif()
endfunction()

# 1. The same version reached twice: the second copy is skipped quietly, one
#    version is in the build, the minimum is met, and a consumer builds against
#    the winning target.
run_case(same_twice TRUE "
add_subdirectory(\"${REAL}\" core_a EXCLUDE_FROM_ALL)
add_subdirectory(\"${SAME_COPY}\" core_b EXCLUDE_FROM_ALL)
snicholls_core_require(${CORE_VERSION})
snicholls_core_assert_single_version()
get_property(seen GLOBAL PROPERTY SNICHOLLS_CORE_VERSIONS_SEEN)
message(STATUS \"versions seen: \${seen}\")
add_executable(use main.cpp)
target_link_libraries(use PRIVATE snicholls::core)
")
expect_in_log(same_twice "is skipped")
expect_in_log(same_twice "versions seen: ${CORE_VERSION}")
expect_in_log(same_twice "satisfies the required minimum ${CORE_VERSION}")
expect_in_log(same_twice "one snicholls::core version in the build")
expect_not_in_log(same_twice "CMake Warning")

execute_process(
    COMMAND "${CMAKE_COMMAND}" --build "${LAST_BIN}" --target use --config Debug
    RESULT_VARIABLE _rc OUTPUT_VARIABLE _out ERROR_VARIABLE _err)
if(NOT _rc EQUAL 0)
    message(FATAL_ERROR "diamond.same_twice: the consumer did not build against the winning copy (${_rc}):\n${_out}${_err}")
endif()
message(STATUS "diamond.same_twice: consumer built against the winning copy")

# 2. Two versions reached, newer first: a warning names both, and the
#    single-version assertion fails configure.
run_case(two_versions FALSE "
add_subdirectory(\"${REAL}\" core_a EXCLUDE_FROM_ALL)
add_subdirectory(\"${OLD_COPY}\" core_b EXCLUDE_FROM_ALL)
snicholls_core_assert_single_version()
")
expect_in_log(two_versions "was added first and wins")
expect_in_log(two_versions "More than one snicholls::core version was reached")
expect_in_log(two_versions "0.0.1 at ")
expect_in_log(two_versions "${CORE_VERSION} at ")

# 3. A minimum the winning copy does not meet fails configure and names the number.
run_case(minimum_unmet FALSE "
add_subdirectory(\"${REAL}\" core_a EXCLUDE_FROM_ALL)
snicholls_core_require(99.0.0)
")
expect_in_log(minimum_unmet "needs snicholls::core 99.0.0 or newer")
expect_in_log(minimum_unmet "the copy that won is ${CORE_VERSION}")

# 4. The old copy is added first, so it wins, and a consumer that needs the
#    current version is refused rather than silently given the old one.
run_case(old_copy_wins FALSE "
add_subdirectory(\"${OLD_COPY}\" core_old EXCLUDE_FROM_ALL)
add_subdirectory(\"${REAL}\" core_new EXCLUDE_FROM_ALL)
snicholls_core_require(${CORE_VERSION})
")
expect_in_log(old_copy_wins "the copy that won is 0.0.1")

# 5. Asking for a minimum before core was added is a clear error, not a
#    confusing one about an undefined property.
run_case(require_before_add FALSE "
include(\"${REAL}/cmake/SnichollsCoreRequire.cmake\")
snicholls_core_require(0.1.0)
")
expect_in_log(require_before_add "before snicholls::core was added")

message(STATUS "diamond_test: all five cases behaved as documented")
