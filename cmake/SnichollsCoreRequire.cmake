# snicholls-core - the diamond rules, as CMake functions.
#
# Included by the in-tree CMakeLists.txt and by the installed
# snicholls_coreConfig.cmake, so a consumer gets the same functions whichever
# way it reached core. See CLAUDE.md "Versioning and the diamond".
#
#   snicholls_core_require(<minimum>)          rule 2: fail configure if the
#                                              winning copy is older
#   snicholls_core_assert_single_version()     rule 3: fail configure if more
#                                              than one version was reached
#   snicholls_core_record_version(<v> <where>) called by core itself, once per
#                                              copy reached; not for consumers
#
# Global properties, readable by any project in the build:
#
#   SNICHOLLS_CORE_VERSION         the version of the copy that won (rule 1)
#   SNICHOLLS_CORE_SOURCE_DIR      where that copy lives
#   SNICHOLLS_CORE_VERSIONS_SEEN   every distinct version reached, winner or not
#   SNICHOLLS_CORE_COPIES_SEEN     every copy reached, as "<version> at <path>"

include_guard(GLOBAL)

function(snicholls_core_record_version version where)
    get_property(_seen GLOBAL PROPERTY SNICHOLLS_CORE_VERSIONS_SEEN)
    list(APPEND _seen "${version}")
    list(REMOVE_DUPLICATES _seen)
    set_property(GLOBAL PROPERTY SNICHOLLS_CORE_VERSIONS_SEEN "${_seen}")

    get_property(_copies GLOBAL PROPERTY SNICHOLLS_CORE_COPIES_SEEN)
    list(APPEND _copies "${version} at ${where}")
    set_property(GLOBAL PROPERTY SNICHOLLS_CORE_COPIES_SEEN "${_copies}")
endfunction()

function(snicholls_core_require minimum)
    get_property(_winner GLOBAL PROPERTY SNICHOLLS_CORE_VERSION)
    if(NOT _winner)
        message(FATAL_ERROR
            "snicholls_core_require(${minimum}) was called before snicholls::core was added. "
            "add_subdirectory() the core checkout, or find_package(snicholls_core), first.")
    endif()
    if(_winner VERSION_LESS minimum)
        get_property(_where GLOBAL PROPERTY SNICHOLLS_CORE_SOURCE_DIR)
        message(FATAL_ERROR
            "${PROJECT_NAME} needs snicholls::core ${minimum} or newer, but the copy that won is ${_winner} "
            "(${_where}). The first copy added to the build wins, so bump the top-level project's pin.")
    endif()
    message(STATUS "${PROJECT_NAME}: snicholls::core ${_winner} satisfies the required minimum ${minimum}")
endfunction()

function(snicholls_core_assert_single_version)
    get_property(_seen GLOBAL PROPERTY SNICHOLLS_CORE_VERSIONS_SEEN)
    list(LENGTH _seen _n)
    if(_n GREATER 1)
        get_property(_copies GLOBAL PROPERTY SNICHOLLS_CORE_COPIES_SEEN)
        string(REPLACE ";" "\n  " _copies "${_copies}")
        message(FATAL_ERROR
            "More than one snicholls::core version was reached in this build:\n  ${_copies}\n"
            "Pin every dependency that carries core to the same release tag.")
    endif()
    if(_n EQUAL 1)
        message(STATUS "${PROJECT_NAME}: one snicholls::core version in the build (${_seen})")
    endif()
endfunction()
