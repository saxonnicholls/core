#!/usr/bin/env python3
#
#  check_mirror.py
#  snicholls-core
#
#  Copyright 2010-2026 Saxon Herschel Nicholls
#
#  The mirror rule (CLAUDE.md "Tests" and "Demos"): every header under
#  include/snicholls/core/<sub>/<name>.hpp has tests/<sub>/<name>_test.cpp and
#  demos/<sub>/<name>_demo.cpp, and both are listed in their CMakeLists.txt.
#  A rule that lives only in a document is a rule that is followed until the
#  first busy afternoon, so this runs on every CI job.
#
#  Companion tests (<name>_<aspect>_test.cpp) belong to <name>.hpp; see companion().
#
#  Exemptions, deliberately few and written down here rather than in a
#  side file:
#    - core.hpp, the umbrella header: it has no behaviour of its own. Every
#      demo exercises it indirectly and the header-check target compiles it.
#    - headers under a detail/ directory: tested, but not demonstrated, since
#      they are not public API.
#
#  Also checks the other direction: a test or demo file that exists but is not
#  listed in CMake is one that never runs, which is worse than one that is
#  missing, because it looks finished.
#
#  Exit status is 0 when the tree mirrors and 1 when it does not.
#

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INCLUDE = os.path.join(ROOT, "include", "snicholls", "core")
TESTS = os.path.join(ROOT, "tests")
DEMOS = os.path.join(ROOT, "demos")

UMBRELLA = "core.hpp"


def read(path):
    return open(path, encoding="utf-8").read()


def headers():
    for dirpath, _, files in os.walk(INCLUDE):
        for f in sorted(files):
            if f.endswith(".hpp"):
                yield os.path.relpath(os.path.join(dirpath, f), INCLUDE)


def sources(root, suffix):
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d != "cmake"]
        for f in sorted(files):
            if f.endswith(suffix):
                yield os.path.relpath(os.path.join(dirpath, f), root)


def companion(test, expected_tests):
    """A companion test: <name>_<aspect>_test.cpp next to <name>_test.cpp, for a part of a header's tests that
    needs its own translation unit (result_terminate_test.cpp builds result.hpp under another policy macro).
    It must still be listed in tests/CMakeLists.txt."""
    stem = test[: -len("_test.cpp")]
    while "_" in os.path.basename(stem):
        stem = stem[: stem.rfind("_")]
        if stem + "_test.cpp" in expected_tests:
            return True
    return False


def main():
    errors = []
    tests_cmake = read(os.path.join(TESTS, "CMakeLists.txt"))
    demos_cmake = read(os.path.join(DEMOS, "CMakeLists.txt"))

    expected_tests = set()
    expected_demos = set()

    for rel in headers():
        if rel == UMBRELLA:
            continue
        stem = rel[: -len(".hpp")]
        parts = stem.split(os.sep)

        test = stem + "_test.cpp"
        expected_tests.add(test)
        if not os.path.exists(os.path.join(TESTS, test)):
            errors.append("header %s has no test: expected tests/%s" % (rel, test))
        elif test.replace(os.sep, "/") not in tests_cmake:
            errors.append("tests/%s exists but is not listed in tests/CMakeLists.txt" % test)

        if "detail" in parts:
            continue
        demo = stem + "_demo.cpp"
        expected_demos.add(demo)
        if not os.path.exists(os.path.join(DEMOS, demo)):
            errors.append("header %s has no demo: expected demos/%s" % (rel, demo))
        elif demo.replace(os.sep, "/") not in demos_cmake:
            errors.append("demos/%s exists but is not listed in demos/CMakeLists.txt" % demo)

    # The other direction: no orphan tests or demos. A companion test (see companion()) still has to be listed.
    tests_listed = [line.strip() for line in tests_cmake.splitlines()]
    for test in sources(TESTS, "_test.cpp"):
        if test not in expected_tests and companion(test, expected_tests) and test.replace(os.sep, "/") not in tests_listed:
            errors.append("tests/%s exists but is not listed in tests/CMakeLists.txt" % test)
    for test in sources(TESTS, "_test.cpp"):
        if test not in expected_tests and not companion(test, expected_tests):
            errors.append("tests/%s has no header include/snicholls/core/%s" % (test, test[: -len("_test.cpp")] + ".hpp"))
    for demo in sources(DEMOS, "_demo.cpp"):
        if demo not in expected_demos:
            errors.append("demos/%s has no header include/snicholls/core/%s" % (demo, demo[: -len("_demo.cpp")] + ".hpp"))

    if errors:
        for e in errors:
            print("check_mirror: " + e, file=sys.stderr)
        print("check_mirror: FAILED - every header has a test and a demo, and both are registered", file=sys.stderr)
        return 1

    n = len(expected_tests)
    print("mirror check: %d header%s, each with a registered test%s" % (
        n, "" if n == 1 else "s", " and demo" if expected_demos else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
