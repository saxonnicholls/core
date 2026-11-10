#!/usr/bin/env bash
#
#  test_ubuntu.sh
#  snicholls-core
#
#  Copyright 2010-2026 Saxon Herschel Nicholls
#
#  Run core's full test suite on Ubuntu 24.04, in Docker, from any machine with
#  Docker - the local half of the gate that CI runs (CLAUDE.md "Build and CI").
#
#      scripts/test_ubuntu.sh            every configuration below
#      scripts/test_ubuntu.sh --quick    GCC Release only
#
#  Configurations, each a fresh CMake build of everything with warnings as
#  errors, then every CTest:
#
#      gcc     Release   + benchmarks smoke run + documentation (Doxygen)
#      gcc     Debug     -O0, asserts live
#      clang   Release
#      clang   Debug
#      clang   ASan+UBSan
#      gcc     ASan+UBSan
#
#  The checkout is mounted read-only and copied into the container from
#  `git ls-files` plus the submodules, so nothing untracked is tested and
#  nothing is written back. Exits non-zero if any configuration fails, after
#  running them all.
#

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QUICK=0
for arg in "$@"; do
    case "$arg" in
        --quick) QUICK=1 ;;
        -h|--help) sed -n '2,30p' "$0" | sed 's/^#\s\{0,1\}//'; exit 0 ;;
        *) echo "test_ubuntu: unknown argument $arg" >&2; exit 2 ;;
    esac
done

command -v docker >/dev/null || { echo "test_ubuntu: docker is not installed" >&2; exit 2; }
docker info >/dev/null 2>&1 || { echo "test_ubuntu: docker is not running" >&2; exit 2; }

# The image is tagged by a hash of its Dockerfile, so editing the Dockerfile
# builds a new image and an unchanged one is reused
HASH="$(shasum -a 256 "$ROOT/docker/ubuntu/Dockerfile" 2>/dev/null || sha256sum "$ROOT/docker/ubuntu/Dockerfile")"
IMAGE="snicholls-core-ubuntu:${HASH:0:12}"
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "test_ubuntu: building $IMAGE"
    DOCKER_BUILDKIT=1 docker build -q -t "$IMAGE" "$ROOT/docker/ubuntu" >/dev/null
fi

# Tracked files only, including the submodules' tracked files
LIST="$(mktemp)"
trap 'rm -f "$LIST"' EXIT
( cd "$ROOT" && git ls-files && git submodule foreach --quiet --recursive \
    'git ls-files | sed "s|^|$sm_path/|"' ) > "$LIST"

docker run --rm -i \
    -v "$ROOT:/src:ro" -v "$LIST:/filelist:ro" \
    -e QUICK="$QUICK" \
    "$IMAGE" bash -s <<'IN_CONTAINER'
set -uo pipefail
mkdir -p /work/core && cd /src && tar -cf - -T /filelist | tar -C /work/core -xf -
cd /work/core

failed=()
passed=()
run() {   # run <label> <cmake args...>
    local label="$1"; shift
    local dir="/work/build-${label// /-}"
    local log="$dir.log"
    printf '  %-28s ' "$label"
    if cmake -S . -B "$dir" -G Ninja -DSNICHOLLS_CORE_WERROR=ON "$@" >"$log" 2>&1 &&
       cmake --build "$dir" >>"$log" 2>&1 &&
       ctest --test-dir "$dir" --output-on-failure >>"$log" 2>&1; then
        echo "ok   $(grep -E '^[0-9]+% tests passed' "$log" | tail -1)"
        passed+=("$label")
    else
        echo "FAILED"
        tail -40 "$log" | sed 's/^/      /'
        failed+=("$label")
    fi
}

echo "snicholls-core on $(. /etc/os-release && echo "$PRETTY_NAME"), $(uname -m)"
echo "  $(g++ --version | head -1)"
echo "  $(clang++ --version | head -1)"
echo "  doxygen $(doxygen --version)"

docs=-DSNICHOLLS_CORE_BUILD_DOCS=ON
case "$(doxygen --version)" in
    1.9.*|1.8.*) docs=-DSNICHOLLS_CORE_BUILD_DOCS=OFF
                 echo "  (docs check skipped: Doxygen $(doxygen --version) is older than the pinned 1.18 - an ARM64 host)";;
esac

run "gcc Release"        -DCMAKE_CXX_COMPILER=g++ -DCMAKE_BUILD_TYPE=Release -DSNICHOLLS_CORE_BUILD_BENCHMARKS=ON "$docs"
if [ "$QUICK" != 1 ]; then
    run "gcc Debug"          -DCMAKE_CXX_COMPILER=g++ -DCMAKE_BUILD_TYPE=Debug
    run "clang Release"      -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release
    run "clang Debug"        -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Debug
    run "clang ASan+UBSan"   -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=RelWithDebInfo -DSNICHOLLS_CORE_SANITIZE=address,undefined
    run "gcc ASan+UBSan"     -DCMAKE_CXX_COMPILER=g++ -DCMAKE_BUILD_TYPE=RelWithDebInfo -DSNICHOLLS_CORE_SANITIZE=address,undefined
fi

echo
if [ ${#failed[@]} -eq 0 ]; then
    echo "test_ubuntu: all ${#passed[@]} configurations passed"
    exit 0
fi
echo "test_ubuntu: ${#failed[@]} of $(( ${#passed[@]} + ${#failed[@]} )) configurations FAILED: ${failed[*]}"
exit 1
IN_CONTAINER
