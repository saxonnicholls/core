#!/usr/bin/env python3
#
#  benchmarks_md.py
#  snicholls-core
#
#  Copyright 2010-2026 Saxon Herschel Nicholls
#
#  Run every benchmark in a Release build and write BENCHMARKS.md from the results, so the
#  file is regenerated, never edited by hand (CLAUDE.md "Zero overhead").
#
#      scripts/benchmarks_md.py                    build-bench, 0.5 s per benchmark
#      scripts/benchmarks_md.py --build build-x    another Release build with benchmarks on
#      scripts/benchmarks_md.py --min-time 2       longer runs, steadier numbers
#      scripts/benchmarks_md.py --filter shapley   only benchmarks whose file name contains this (a,b for several)
#      scripts/benchmarks_md.py --output - --title "Throughput - Linux GCC" --json bench-json
#                                                  CI: fresh tables to stdout (the job summary), raw JSON kept;
#                                                  BENCHMARKS.md is not touched
#      scripts/benchmarks_md.py --config Release   a multi-configuration build (Visual Studio)
#
#  `make benchmarks` configures build-bench, builds it and runs this. Each benchmark checks
#  its own fairness before it runs; a benchmark that refuses fails the script. A filtered
#  run rewrites only its own sections and keeps the others as they were.
#

import argparse
import datetime
import json
import pathlib
import platform
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "BENCHMARKS.md"

INTRO = """# Benchmarks

Every benchmark compares core against the hand-written equivalent on identical data, after a fairness
check that refuses to run if the two disagree (CLAUDE.md, "Zero overhead"). Only Release numbers mean
anything; a header quotes its own numbers in its **Performance** section.

**This file is generated.** Do not edit it: run

    make benchmarks                      # every benchmark, about a minute
    scripts/benchmarks_md.py --filter X  # one file's sections; the rest are kept

and commit the result. The "relative" column is each row's time over the fastest row in its file.
"""


def cmake_cache(build: pathlib.Path) -> dict:
    cache = {}
    for line in (build / "CMakeCache.txt").read_text().splitlines():
        m = re.match(r"^([A-Za-z0-9_]+):[A-Z]+=(.*)$", line)
        if m:
            cache[m.group(1)] = m.group(2)
    return cache


def compiler(build: pathlib.Path) -> str:
    for f in (build / "CMakeFiles").glob("*/CMakeCXXCompiler.cmake"):
        text = f.read_text()
        cid = re.search(r'set\(CMAKE_CXX_COMPILER_ID "([^"]*)"\)', text)
        ver = re.search(r'set\(CMAKE_CXX_COMPILER_VERSION "([^"]*)"\)', text)
        if cid and ver:
            return f"{cid.group(1)} {ver.group(1)}"
    return "unknown compiler"


def cpu() -> str:
    try:
        if platform.system() == "Darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        for line in pathlib.Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def human_time(ns: float) -> str:
    for unit, scale in (("s", 1e9), ("ms", 1e6), ("us", 1e3)):
        if ns >= scale:
            return f"{ns / scale:,.2f} {unit}"
    return f"{ns:,.1f} ns"


def human_rate(per_second: float) -> str:
    for unit, scale in (("G", 1e9), ("M", 1e6), ("k", 1e3)):
        if per_second >= scale:
            return f"{per_second / scale:,.2f} {unit}/s"
    return f"{per_second:,.1f} /s"


def run(binary: pathlib.Path, min_time: float) -> dict:
    result = subprocess.run([str(binary), "--benchmark_format=json", f"--benchmark_min_time={min_time}s"], capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"benchmarks_md: {binary.name} failed (exit {result.returncode}):\n{result.stderr.strip()}")
    return json.loads(result.stdout)


def section(name: str, source: str, data: dict) -> str:
    rows = [b for b in data["benchmarks"] if b.get("run_type", "iteration") == "iteration"]
    to_ns = {"ns": 1, "us": 1e3, "ms": 1e6, "s": 1e9}
    times = [b["real_time"] * to_ns[b.get("time_unit", "ns")] for b in rows]
    fastest = min(times) if times else 1
    lines = [f"## {name}", "", f"`{source}`", "", "| benchmark | time | relative | throughput |", "| --- | ---: | ---: | ---: |"]
    for b, t in zip(rows, times):
        rate = human_rate(b["items_per_second"]) if "items_per_second" in b else ""
        lines.append(f"| `{b['name']}` | {human_time(t)} | {t / fastest:.2f}x | {rate} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", default="build-bench")
    ap.add_argument("--min-time", type=float, default=0.5)
    ap.add_argument("--filter", default="")
    ap.add_argument("--output", default=str(OUT), help="the file to write, or - for stdout (default: BENCHMARKS.md)")
    ap.add_argument("--title", default="", help="a heading for the tables instead of BENCHMARKS.md's introduction")
    ap.add_argument("--json", default="", help="a folder to keep each benchmark's raw Google Benchmark JSON in")
    ap.add_argument("--config", default="", help="the configuration of a multi-configuration build (Release)")
    args = ap.parse_args()
    out = None if args.output == "-" else pathlib.Path(args.output).resolve()
    into_benchmarks_md = out == OUT.resolve()

    build = (ROOT / args.build).resolve()
    if not (build / "CMakeCache.txt").exists():
        sys.exit(f"benchmarks_md: {build} is not configured; run `make benchmarks`")
    cache = cmake_cache(build)
    if (args.config or cache.get("CMAKE_BUILD_TYPE", "")) != "Release":
        sys.exit(f"benchmarks_md: {build} is a {cache.get('CMAKE_BUILD_TYPE') or 'default'} build; only Release numbers mean anything")

    sources = {}   # target name -> source, from benchmarks/CMakeLists.txt
    for m in re.finditer(r"snicholls_core_add_benchmark\((\w+)\s+([^)\s]+)\)", (ROOT / "benchmarks/CMakeLists.txt").read_text()):
        sources[m.group(1)] = "benchmarks/" + m.group(2)

    previous = {}   # kept sections: only when rewriting BENCHMARKS.md itself
    if into_benchmarks_md and OUT.exists():
        for block in re.split(r"(?m)^(?=## )", OUT.read_text()):
            if block.startswith("## "):
                previous[block.split("\n", 1)[0][3:].strip()] = block.rstrip() + "\n"

    sections = dict(previous)
    context = None
    for target, source in sorted(sources.items()):
        if args.filter and not any(f in target for f in args.filter.split(",")):
            continue
        folder = build / "benchmarks" / args.config if args.config else build / "benchmarks"
        binary = folder / f"snicholls_core_{target}"
        if not binary.exists() and binary.with_suffix(".exe").exists():
            binary = binary.with_suffix(".exe")
        if not binary.exists():
            sys.exit(f"benchmarks_md: {binary} is missing; build the benchmarks first")
        print(f"benchmarks_md: running {target}", file=sys.stderr)
        data = run(binary, args.min_time)
        if args.json:
            (pathlib.Path(args.json)).mkdir(parents=True, exist_ok=True)
            (pathlib.Path(args.json) / f"{target}.json").write_text(json.dumps(data, indent=1))
        context = context or data["context"]
        sections[target] = section(target, source, data)

    sections = {k: v for k, v in sections.items() if k in sources}   # drop benchmarks that no longer exist
    today = datetime.date.today().isoformat()
    machine = f"{cpu()}, {context['num_cpus'] if context else '?'} CPUs" if context else cpu()
    env = (
        "| | |\n| --- | --- |\n"
        f"| date | {today} |\n| machine | {machine} |\n| OS | {platform.system()} {platform.release()} |\n"
        f"| compiler | {compiler(build)} |\n| build | Release, `{args.build}` |\n| min time per benchmark | {args.min_time} s |\n"
    )
    if args.filter:
        env += "| this run | only " + ", ".join(f"`{f}`" for f in args.filter.split(",")) + "; other sections are from earlier runs |\n"
    head = f"# {args.title}\n" if args.title else INTRO
    text = head + "\n" + env + "\n" + "\n".join(sections[k] for k in sorted(sections))
    if out is None:
        sys.stdout.write(text)
        return
    out.write_text(text)
    print(f"benchmarks_md: wrote {out} ({len(sections)} benchmark files)", file=sys.stderr)


if __name__ == "__main__":
    main()
