#!/usr/bin/env bash
#
#  run_demos.sh
#  snicholls-core
#
#  Copyright 2010-2026 Saxon Herschel Nicholls
#
#  Build and run core's demos and show what they print. Every demo is also a
#  CTest (label demo), but ctest keeps the output; this is for reading it.
#
#      scripts/run_demos.sh --all                 every demo in demos/CMakeLists.txt
#      scripts/run_demos.sh acout uint256_demo    one or more, by name
#      scripts/run_demos.sh --list                the names, one per line
#
#  A name is as registered in demos/CMakeLists.txt (acout_demo), with or
#  without its _demo suffix, or the path of its source (demos/acout_demo.cpp).
#  The build directory is BUILD (default build, as the Makefile), configured
#  as CONFIG (default Release) if it does not exist yet; only the demos asked
#  for are built. Exits non-zero if any demo fails, after running them all.
#

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${BUILD:-build}"
CONFIG="${CONFIG:-Release}"
CXX="${CXX:-c++}"

# name<tab>source, from the registrations in demos/CMakeLists.txt
registered() {
    sed -n 's/^snicholls_core_add_demo(\([A-Za-z0-9_]*\)[[:space:]]*\([^)]*\))[[:space:]]*$/\1	\2/p' "$ROOT/demos/CMakeLists.txt"
}

ALL=0
LIST=0
WANTED=()
for arg in "$@"; do
    case "$arg" in
        --all) ALL=1 ;;
        --list) LIST=1 ;;
        -h|--help) sed -n '2,20p' "$0" | sed 's/^#\s\{0,1\}//'; exit 0 ;;
        -*) echo "run_demos: unknown option $arg" >&2; exit 2 ;;
        *) WANTED+=("$arg") ;;
    esac
done

if [ "$LIST" = 1 ]; then
    registered | cut -f1
    exit 0
fi
if [ "$ALL" = 0 ] && [ "${#WANTED[@]}" = 0 ]; then
    echo "run_demos: name a demo, or --all; --list shows the names" >&2
    exit 2
fi

# Resolve what was asked for to registered names
NAMES=()
if [ "$ALL" = 1 ]; then
    while IFS=$'\t' read -r name _; do NAMES+=("$name"); done < <(registered)
else
    for want in "${WANTED[@]}"; do
        base="$(basename "$want" .cpp)"
        base="${base#snicholls_core_}"
        case "$base" in *_demo) ;; *) base="${base}_demo" ;; esac
        if registered | cut -f1 | grep -qx "$base"; then
            NAMES+=("$base")
        else
            echo "run_demos: no demo named $want; the names are:" >&2
            registered | cut -f1 | sed 's/^/    /' >&2
            exit 2
        fi
    done
fi

# Configure: with the settings above the first time, and again with the cached ones every
# time after, so a demo registered since the tree was configured has a target to build
cd "$ROOT"
LOG="$(mktemp)"
trap 'rm -f "$LOG"' EXIT
if [ ! -f "$BUILD/CMakeCache.txt" ]; then
    echo "run_demos: configuring $BUILD ($CONFIG)"
    cmake -B "$BUILD" -DCMAKE_BUILD_TYPE="$CONFIG" -DCMAKE_CXX_COMPILER="$CXX" >"$LOG" 2>&1 || { cat "$LOG"; exit 1; }
else
    cmake -B "$BUILD" >"$LOG" 2>&1 || { cat "$LOG"; exit 1; }
fi

# Build only what is asked for; the compiler's output is shown only if it fails
TARGETS=()
for name in "${NAMES[@]}"; do TARGETS+=("snicholls_core_$name"); done
echo "run_demos: building ${#TARGETS[@]} demo(s) in $BUILD"
cmake --build "$BUILD" --parallel --target "${TARGETS[@]}" >"$LOG" 2>&1 || { cat "$LOG"; exit 1; }

# Run each, whole output shown, and keep score
FAILED=()
for name in "${NAMES[@]}"; do
    exe="$BUILD/demos/snicholls_core_$name"
    [ -x "$exe" ] || exe="$BUILD/demos/$CONFIG/snicholls_core_$name"   # a multi-config generator
    echo
    echo "==== $name ($exe)"
    if "$exe"; then
        echo "---- $name: ok"
    else
        echo "---- $name: FAILED (exit $?)"
        FAILED+=("$name")
    fi
done

echo
if [ "${#FAILED[@]}" = 0 ]; then
    echo "run_demos: ${#NAMES[@]} demo(s) ok"
    exit 0
fi
echo "run_demos: ${#FAILED[@]} of ${#NAMES[@]} demo(s) FAILED: ${FAILED[*]}" >&2
exit 1
