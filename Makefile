# snicholls-core
#
# The library is header-only (include/snicholls/core/*.hpp). CMake is the
# build; this is the short way to drive it, with the same gates CI runs.
#
#   make test            configure, build, run every CTest (unit, demo, boundary)
#   make test-werror     the same with warnings as errors, as CI does
#   make asan            Address + UB sanitizers
#   make docs            Doxygen, warnings as errors; opens nothing, writes build/docs/html
#   make check           the static checks CI runs first: version, mirror
#   make ubuntu          the full suite on Ubuntu 24.04 in Docker (scripts/test_ubuntu.sh)
#   make benchmarks      every benchmark in Release, results written to BENCHMARKS.md
#                        (scripts/benchmarks_md.py; --filter for one file)
#   make demos           build and run every demo, output shown (scripts/run_demos.sh --all;
#                        scripts/run_demos.sh acout for one)
#   make clean
#
#   make test CXX=clang++      any target with a different compiler
#   make test BUILD=build-x    any target in a different build directory

CXX      ?= c++
BUILD    ?= build
CONFIG   ?= Release
CMAKE_FLAGS ?=

.PHONY: all test test-werror asan docs check ubuntu demos benchmarks clean

all: test

check:
	python3 scripts/check_version.py
	python3 scripts/check_mirror.py

$(BUILD)/CMakeCache.txt:
	cmake -B $(BUILD) -DCMAKE_BUILD_TYPE=$(CONFIG) -DCMAKE_CXX_COMPILER=$(CXX) $(CMAKE_FLAGS)

test: check $(BUILD)/CMakeCache.txt
	cmake --build $(BUILD) --parallel
	ctest --test-dir $(BUILD) --output-on-failure

test-werror:
	$(MAKE) test BUILD=build-werror CMAKE_FLAGS="-DSNICHOLLS_CORE_WERROR=ON"

asan:
	$(MAKE) test BUILD=build-asan CONFIG=RelWithDebInfo \
	    CMAKE_FLAGS="-DSNICHOLLS_CORE_WERROR=ON -DSNICHOLLS_CORE_SANITIZE=address,undefined"

docs:
	cmake -B build-docs -DSNICHOLLS_CORE_BUILD_TESTS=OFF -DSNICHOLLS_CORE_BUILD_DOCS=ON
	cmake --build build-docs --target docs

ubuntu:
	scripts/test_ubuntu.sh

benchmarks:
	cmake -B build-bench -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=$(CXX) -DSNICHOLLS_CORE_BUILD_BENCHMARKS=ON -DSNICHOLLS_CORE_BUILD_TESTS=OFF
	cmake --build build-bench --parallel
	python3 scripts/benchmarks_md.py --build build-bench

demos:
	BUILD=$(BUILD) CONFIG=$(CONFIG) CXX=$(CXX) scripts/run_demos.sh --all

clean:
	rm -rf build build-*
