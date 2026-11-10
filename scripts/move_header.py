#!/usr/bin/env python3
#
#  move_header.py
#  snicholls-core
#
#  Copyright 2010-2026 Saxon Herschel Nicholls
#
#  Move a public header to another area, with everything that mirrors it, in one step:
#
#      scripts/move_header.py permutations.hpp combinatorics/permutations.hpp
#      scripts/move_header.py reflect/type_list.hpp meta/type_list.hpp
#
#  Paths are relative to include/snicholls/core/. It git-moves the header, its test, its demo
#  and its benchmark (whichever exist), and rewrites every include path, header guard, CMake
#  registration and documented path in the tracked files. The namespace is not touched. Run
#  `make check` and a build afterwards; nothing is committed.
#

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
CORE = "include/snicholls/core/"


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout


def stem(path: str) -> str:
    return path[: -len(".hpp")]


def guard(path: str) -> str:
    return "snicholls_core_" + stem(path).replace("/", "_") + "_hpp"


def main() -> None:
    if len(sys.argv) != 3 or not all(a.endswith(".hpp") for a in sys.argv[1:]):
        sys.exit(__doc__ if False else "usage: scripts/move_header.py <old>.hpp <new>.hpp   (relative to include/snicholls/core/)")
    old, new = sys.argv[1], sys.argv[2]
    if not (ROOT / CORE / old).exists():
        sys.exit(f"move_header: {CORE}{old} does not exist")
    if (ROOT / CORE / new).exists():
        sys.exit(f"move_header: {CORE}{new} already exists")

    old_stem, new_stem = stem(old), stem(new)
    old_name, new_name = old_stem.rsplit("/", 1)[-1], new_stem.rsplit("/", 1)[-1]
    moves = [
        (CORE + old, CORE + new),
        (f"tests/{old_stem}_test.cpp", f"tests/{new_stem}_test.cpp"),
        (f"demos/{old_stem}_demo.cpp", f"demos/{new_stem}_demo.cpp"),
        (f"benchmarks/{old_stem}_benchmark.cpp", f"benchmarks/{new_stem}_benchmark.cpp"),
    ]
    for src, dst in moves:
        if (ROOT / src).exists():
            (ROOT / dst).parent.mkdir(parents=True, exist_ok=True)
            git("mv", src, dst)
            print(f"move_header: {src} -> {dst}")

    # Longest first, so that a path is rewritten before any shorter text inside it
    replacements = sorted(
        [
            (f"snicholls/core/{old}", f"snicholls/core/{new}"),
            (guard(old), guard(new)),
            (f"tests/{old_stem}_test.cpp", f"tests/{new_stem}_test.cpp"),
            (f"demos/{old_stem}_demo.cpp", f"demos/{new_stem}_demo.cpp"),
            (f"benchmarks/{old_stem}_benchmark.cpp", f"benchmarks/{new_stem}_benchmark.cpp"),
            (f"`{old}`", f"`{new}`"),
            (f" {old_stem}_test.cpp\n", f" {new_stem}_test.cpp\n"),
            (f" {old_stem}_test.cpp)", f" {new_stem}_test.cpp)"),
            (f"({old_name}_demo {old_stem}_demo.cpp)", f"({new_name}_demo {new_stem}_demo.cpp)"),
            (f"({old_name}_benchmark {old_stem}_benchmark.cpp)", f"({new_name}_benchmark {new_stem}_benchmark.cpp)"),
            (f"\\snippet {old_stem}_demo.cpp", f"\\snippet {new_stem}_demo.cpp"),
        ],
        key=lambda r: -len(r[0]),
    )
    changed = 0
    for f in git("ls-files").split():
        if f.startswith("third_party/"):
            continue
        p = ROOT / f
        if not p.is_file() or p.suffix not in {".hpp", ".cpp", ".txt", ".md", ".cmake", ".py", ".in", ".sh", ""}:
            continue
        try:
            text = p.read_text()
        except UnicodeDecodeError:
            continue
        updated = text
        for a, b in replacements:
            updated = updated.replace(a, b)
        if updated != text:
            p.write_text(updated)
            changed += 1
    print(f"move_header: rewrote {changed} files; now run `make check` and build")


if __name__ == "__main__":
    main()
