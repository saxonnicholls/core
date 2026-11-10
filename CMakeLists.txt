# snicholls-core - the vocabulary types and compile-time reflection shared by
# the snicholls libraries: ts-moveables, base-encode-decode and their consumers.
#
# Header-only. This produces the interface target snicholls::core, builds the
# unit tests, demos and documentation when configured as the top-level project,
# and installs a find_package(snicholls_core) config.
#
# It is also where the diamond is defused. A consumer may pin core and also pin
# ts-moveables, which pins core, so one build can reach this file twice from two
# checkouts. Two copies of a header-only library at different
# commits link silently and disagree quietly, so three rules apply here and in
# every consumer (CLAUDE.md "Versioning and the diamond"):
#
#   1. the target is defined only if nobody defined it first - the top-level
#      project's copy wins, because its pin is the one a human chose;
#   2. every consumer states the minimum it needs with
#      snicholls_core_require(x.y.z), and configure fails if the winning copy
#      is older;
#   3. every copy that is reached records itself, and
#      snicholls_core_assert_single_version() fails configure if more than one
#      distinct version was reached.
#
# tests/cmake/diamond_test.cmake exercises all three with a fixture project.

cmake_minimum_required(VERSION 3.20)

project(snicholls_core
    VERSION 0.5.1
    DESCRIPTION "snicholls-core - vocabulary types and compile-time reflection shared by the snicholls libraries"
    LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/SnichollsCoreRequire.cmake)

# Rule 3: every copy that is reached is recorded, winner or not
snicholls_core_record_version(${PROJECT_VERSION} "${CMAKE_CURRENT_SOURCE_DIR}")

# Rule 1: the first copy wins
if(TARGET snicholls::core)
    get_property(_core_winner GLOBAL PROPERTY SNICHOLLS_CORE_VERSION)
    get_property(_core_winner_dir GLOBAL PROPERTY SNICHOLLS_CORE_SOURCE_DIR)
    if(_core_winner VERSION_EQUAL PROJECT_VERSION)
        message(STATUS "snicholls::core ${PROJECT_VERSION}: already defined from ${_core_winner_dir}; "
                       "this copy (${CMAKE_CURRENT_SOURCE_DIR}) is skipped")
    else()
        message(WARNING "snicholls::core ${PROJECT_VERSION} at ${CMAKE_CURRENT_SOURCE_DIR} is skipped: "
                        "${_core_winner} at ${_core_winner_dir} was added first and wins. "
                        "Pin every dependency that carries core to one release.")
    endif()
    return()
endif()

# The library
add_library(snicholls_core INTERFACE)
add_library(snicholls::core ALIAS snicholls_core)
set_target_properties(snicholls_core PROPERTIES EXPORT_NAME core)

target_compile_features(snicholls_core INTERFACE cxx_std_20)

# Headers are included as <snicholls/core/version.hpp>, both from the source
# tree and once installed
target_include_directories(snicholls_core INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

# The preprocessor-driven headers (pp_for_each and what is built on it) are
# written to the standard preprocessor. MSVC's traditional one expands
# __VA_ARGS__ as a single token and breaks them, so consumers on MSVC need the
# conforming preprocessor. This is a correctness requirement of the headers,
# not a style preference, which is why it is the one flag the INTERFACE target
# propagates.
target_compile_options(snicholls_core INTERFACE $<$<CXX_COMPILER_ID:MSVC>:/Zc:preprocessor>)

set_property(GLOBAL PROPERTY SNICHOLLS_CORE_VERSION ${PROJECT_VERSION})
set_property(GLOBAL PROPERTY SNICHOLLS_CORE_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# ---------------------------------------------------------------------------
# Tests, demos, benchmarks and docs: on by default only as the top-level project

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(SNICHOLLS_CORE_TOP_LEVEL ON)
else()
    set(SNICHOLLS_CORE_TOP_LEVEL OFF)
endif()

option(SNICHOLLS_CORE_BUILD_TESTS "Build the snicholls-core unit tests and demos" ${SNICHOLLS_CORE_TOP_LEVEL})
option(SNICHOLLS_CORE_BUILD_BENCHMARKS "Build the snicholls-core benchmarks (Google Benchmark)" OFF)
option(SNICHOLLS_CORE_BUILD_DOCS "Build the Doxygen documentation; warnings are errors" OFF)
option(SNICHOLLS_CORE_WERROR "Treat warnings as errors in everything this project compiles (CI turns this on)" OFF)
set(SNICHOLLS_CORE_SANITIZE "" CACHE STRING "Sanitizers for the tests and demos, e.g. address,undefined")

# Warning flags go on the things this project compiles, never on the INTERFACE
# target, which would push them into every consumer's build.
function(snicholls_core_compile_options target)
    if(MSVC)
        # /utf-8: the sources are UTF-8. Without it MSVC reads a file with no BOM
        # in the system code page, so a literal such as u8"€" is mis-encoded and
        # the string tests fail. Our own targets only, never the INTERFACE one.
        target_compile_options(${target} PRIVATE /W4 /permissive- /utf-8)
        if(SNICHOLLS_CORE_WERROR)
            target_compile_options(${target} PRIVATE /WX)
        endif()
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
        if(SNICHOLLS_CORE_WERROR)
            target_compile_options(${target} PRIVATE -Werror)
        endif()
        if(SNICHOLLS_CORE_SANITIZE)
            target_compile_options(${target} PRIVATE -fsanitize=${SNICHOLLS_CORE_SANITIZE} -fno-omit-frame-pointer)
            target_link_options(${target} PRIVATE -fsanitize=${SNICHOLLS_CORE_SANITIZE})
        endif()
    endif()
endfunction()

# Our own tests, demos and benchmarks: the same options, and on MSVC two notes on test code, not on the library. C4459
# flags a test's global hiding a name inside a header (test hygiene; the header check compiles every header alone at
# /W4 /WX), and _CRT_SECURE_NO_WARNINGS quiets the CRT's deprecation of tmpfile, sscanf and friends.
function(snicholls_core_test_options target)
    snicholls_core_compile_options(${target})
    if(MSVC)
        target_compile_options(${target} PRIVATE /wd4459)
        target_compile_definitions(${target} PRIVATE _CRT_SECURE_NO_WARNINGS)
    endif()
endfunction()

# As the top-level project, everything here - including GoogleTest and Google
# Benchmark - is compiled as C++20, the standard the library requires. Left
# unset, CMake compiles the vendored test frameworks at their own minimum,
# which mixes standards in one link and has already failed to build once.
if(SNICHOLLS_CORE_TOP_LEVEL AND NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
endif()

if(SNICHOLLS_CORE_BUILD_TESTS)
    enable_testing()

    # Every header compiles on its own, and twice in one translation unit so a
    # broken include guard is caught too. A header that only compiles because
    # another was included first has a hidden dependency, and the consumer who
    # finds it will not be us.
    file(GLOB_RECURSE _core_headers RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/include
        CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/include/snicholls/*.hpp)
    set(_header_check_sources)
    foreach(_header IN LISTS _core_headers)
        string(MAKE_C_IDENTIFIER "${_header}" _id)
        set(HEADER "${_header}")
        configure_file(cmake/header_check.cpp.in ${CMAKE_CURRENT_BINARY_DIR}/header_check/${_id}.cpp @ONLY)
        list(APPEND _header_check_sources ${CMAKE_CURRENT_BINARY_DIR}/header_check/${_id}.cpp)
    endforeach()
    add_library(snicholls_core_header_check OBJECT ${_header_check_sources})
    target_link_libraries(snicholls_core_header_check PRIVATE snicholls::core)
    snicholls_core_compile_options(snicholls_core_header_check)

    # constexpr_for.hpp is also ts-moveables' constexpr_for, and ts-moveables supports C++17: it stays C++17-clean.
    set(HEADER "snicholls/core/constexpr_for.hpp")
    configure_file(cmake/header_check.cpp.in ${CMAKE_CURRENT_BINARY_DIR}/header_check_cpp17/constexpr_for.cpp @ONLY)
    add_library(snicholls_core_cpp17_check OBJECT ${CMAKE_CURRENT_BINARY_DIR}/header_check_cpp17/constexpr_for.cpp)
    target_include_directories(snicholls_core_cpp17_check PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    set_target_properties(snicholls_core_cpp17_check PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
    snicholls_core_compile_options(snicholls_core_cpp17_check)

    # GoogleTest, vendored as a shallow submodule pinned to a release tag
    # (CLAUDE.md "Dependencies"). For the tests only: the library has none.
    if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/googletest/CMakeLists.txt)
        message(FATAL_ERROR "third_party/googletest is empty - run: git submodule update --init --recursive")
    endif()
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    add_subdirectory(third_party/googletest EXCLUDE_FROM_ALL)
    # GoogleTest marks its includes SYSTEM, and AppleClang searches
    # /usr/local/include before any -isystem path, so a Homebrew googletest
    # silently shadows the submodule and gmock fails to compile against the
    # wrong gtest. Plain -I it is (CLAUDE.md "Dependencies").
    set_property(TARGET gtest gtest_main gmock gmock_main PROPERTY INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "")

    add_subdirectory(tests)
    add_subdirectory(demos)
endif()

if(SNICHOLLS_CORE_BUILD_BENCHMARKS)
    # Google Benchmark, pinned like GoogleTest. Arrives with the first runtime
    # type (result<V, E>); until then the directory holds only its CMake.
    if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/benchmark/CMakeLists.txt)
        message(FATAL_ERROR "third_party/benchmark is empty - run: git submodule update --init --recursive")
    endif()
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
    add_subdirectory(third_party/benchmark EXCLUDE_FROM_ALL)
    add_subdirectory(benchmarks)
endif()

if(SNICHOLLS_CORE_BUILD_DOCS)
    find_package(Doxygen REQUIRED)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)   # docs/doxygen_filter.py
    set(DOXYGEN_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/docs)
    configure_file(docs/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
    add_custom_target(docs
        COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Doxygen: every public entity documented, warnings are errors"
        VERBATIM)
    if(SNICHOLLS_CORE_BUILD_TESTS)
        add_test(NAME docs.build
            COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
        set_tests_properties(docs.build PROPERTIES LABELS docs)
    endif()
endif()

# ---------------------------------------------------------------------------
# Install and export: the include tree is installed as a tree, so consumers
# include <snicholls/core/...> exactly as they do from the source tree

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp")

install(TARGETS snicholls_core EXPORT snicholls_coreTargets)

install(EXPORT snicholls_coreTargets
    NAMESPACE snicholls::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/snicholls_core)

configure_package_config_file(
    cmake/snicholls_coreConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/snicholls_coreConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/snicholls_core)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/snicholls_coreConfigVersion.cmake
    COMPATIBILITY SameMajorVersion
    ARCH_INDEPENDENT)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/snicholls_coreConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/snicholls_coreConfigVersion.cmake
    cmake/SnichollsCoreRequire.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/snicholls_core)
