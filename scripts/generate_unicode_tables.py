#!/usr/bin/env python3
#
#  generate_unicode_tables.py
#  snicholls-core
#
#  Copyright 2010-2026 Saxon Herschel Nicholls
#
#  Writes include/snicholls/core/detail/unicode_tables.hpp, the Unicode tables of ml/unicode.hpp: every code
#  point's general category, the simple lowercase mapping, the canonical decompositions and the canonical
#  combining classes, as sorted constexpr arrays.
#
#  Two steps, so that the header can be checked on any machine without the Unicode Character Database:
#
#    1. Extract, once per Unicode version, from UnicodeData.txt into the committed scripts/data/ucd-<version>.txt
#       (the four tables above, range-compressed, about 70 KB):
#
#         curl -O https://www.unicode.org/Public/16.0.0/ucd/UnicodeData.txt
#         scripts/generate_unicode_tables.py --extract UnicodeData.txt [--clone <llama.cpp>/src/unicode-data.cpp]
#
#       With --clone the extract is cross-checked against llama.cpp's generated tables (its category flags,
#       whitespace set, lowercase map and canonical decompositions), which are the Unicode data the tokenizer
#       fixtures were produced with; any disagreement is an error.
#
#    2. Render the header from the extract:
#
#         scripts/generate_unicode_tables.py            # writes the header
#         scripts/generate_unicode_tables.py --check    # exits 1 if the committed header differs (a CTest)
#
#  Never edit the generated header by hand.
#

import hashlib
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
VERSION = "16.0.0"
DATA = ROOT / "scripts" / "data" / f"ucd-{VERSION}.txt"
OUT = ROOT / "include" / "snicholls" / "core" / "detail" / "unicode_tables.hpp"

# The order of snicholls::ml::unicode_category; the header stores these indices.
CATEGORIES = ["Cn", "Lu", "Ll", "Lt", "Lm", "Lo", "Mn", "Mc", "Me", "Nd", "Nl", "No", "Pc", "Pd", "Ps", "Pe", "Pi",
              "Pf", "Po", "Sm", "Sc", "Sk", "So", "Zs", "Zl", "Zp", "Cc", "Cf", "Cs", "Co"]

# White_Space (PropList.txt), unchanged since Unicode 6.3
WHITESPACE = list(range(0x09, 0x0E)) + [0x20, 0x85, 0xA0, 0x1680] + list(range(0x2000, 0x200B)) + \
             [0x2028, 0x2029, 0x202F, 0x205F, 0x3000]


def extract(unicode_data, clone):
    raw = pathlib.Path(unicode_data).read_bytes()
    gc, lower, dec, ccc = {}, {}, {}, {}
    first = None
    for line in raw.decode("ascii").splitlines():
        f = line.split(";")
        if len(f) != 15:
            sys.exit(f"generate_unicode_tables: not UnicodeData.txt: {line!r}")
        cp, name = int(f[0], 16), f[1]
        if name.endswith(", First>"):
            first = cp
            continue
        span = range(first, cp + 1) if name.endswith(", Last>") else [cp]
        for c in span:
            gc[c] = f[2]
            if f[3] != "0":
                ccc[c] = int(f[3])
        if f[5] and not f[5].startswith("<"):
            dec[cp] = [int(x, 16) for x in f[5].split()]
        if f[13]:
            lower[cp] = int(f[13], 16)
    for g in set(gc.values()):
        if g not in CATEGORIES:
            sys.exit(f"generate_unicode_tables: unknown general category {g}")
    if clone:
        cross_check(pathlib.Path(clone).read_text(encoding="utf-8"), gc, lower, dec)

    lines = [f"# Extracted from UnicodeData.txt {VERSION} (sha256 {hashlib.sha256(raw).hexdigest()})",
             "# by scripts/generate_unicode_tables.py --extract. Never edit by hand.",
             "# gc <first code point> <general category>, a run up to the next gc line (Cn where unassigned)",
             "# lower <first code point> <count> <stride> <delta>: a run of the simple lowercase mapping",
             "# dec <code point> <canonical decomposition> (Hangul syllables are algorithmic and not listed)",
             "# ccc <first> <last> <canonical combining class>"]
    last = None
    for c in range(0x110000):
        g = gc.get(c, "Cn")
        if g != last:
            lines.append(f"gc {c:X} {g}")
            last = g
    # The simple lowercase mapping as runs: first, count, stride, delta
    runs = []
    for c, lc in sorted(lower.items()):
        d = lc - c
        if runs:
            s, n, st, dd = runs[-1]
            if dd == d and ((n == 1 and c - s in (1, 2)) or (n > 1 and c == s + n * st)):
                runs[-1] = [s, n + 1, c - s if n == 1 else st, d]
                continue
        runs.append([c, 1, 1, d])
    lines += [f"lower {s:X} {n} {st} {d}" for s, n, st, d in runs]
    lines += ["dec " + " ".join(f"{x:X}" for x in [c] + d) for c, d in sorted(dec.items())]
    runs = []
    for c in sorted(ccc):
        if runs and runs[-1][1] == c - 1 and runs[-1][2] == ccc[c]:
            runs[-1][1] = c
        else:
            runs.append([c, c, ccc[c]])
    lines += [f"ccc {a:X} {b:X} {k}" for a, b, k in runs]
    DATA.parent.mkdir(parents=True, exist_ok=True)
    DATA.write_text("\n".join(lines) + "\n", encoding="ascii")
    print(f"wrote {DATA.relative_to(ROOT)}")


def cross_check(src, gc, lower, dec):
    # llama.cpp's flags: 1 undefined, 2 number, 4 letter, 8 separator, 16 mark, 32 punctuation, 64 symbol, 128 control
    coarse = {"C": 0x80, "L": 0x04, "M": 0x10, "N": 0x02, "P": 0x20, "S": 0x40, "Z": 0x08}
    body = src[src.index("unicode_ranges_flags"):]
    ranges = [(int(a, 16), int(b, 16)) for a, b in re.findall(r"\{0x([0-9A-F]+), 0x([0-9A-F]+)\}", body[:body.index("};")])]
    # The clone's category flags predate Unicode 16.0, its normalisation tables do not: a code point it has as
    # unassigned may be assigned here (new in 16.0), and nothing else may differ.
    newer = set()
    for (start, flags), (end, _) in zip(ranges, ranges[1:]):
        for c in range(start, end):
            g = gc.get(c, "Cn")
            want = 0x01 if g == "Cn" else coarse[g[0]]
            if flags == 0x01 and g != "Cn":
                newer.add(c)
            elif flags != want:
                sys.exit(f"generate_unicode_tables: U+{c:04X} is {g} here and flags {flags:#x} in the clone")
    ws = src[src.index("unicode_set_whitespace"):]
    ws = sorted(int(x, 16) for x in re.findall(r"0x([0-9A-F]+)", ws[:ws.index("};")]))
    if ws != sorted(WHITESPACE):
        sys.exit("generate_unicode_tables: the clone's whitespace set differs")
    lo = src[src.index("unicode_map_lowercase"):]
    lo = {int(a, 16): int(b, 16) for a, b in re.findall(r"\{0x([0-9A-F]+), 0x([0-9A-F]+)\}", lo[:lo.index("};")])}
    if {c: x for c, x in lo.items() if c not in newer} != {c: x for c, x in lower.items() if c not in newer}:
        sys.exit("generate_unicode_tables: the clone's lowercase map differs")
    de = src[src.index("unicode_decomposition ="):]
    de = {int(a, 16): [int(x, 16) for x in (b, c) if int(x, 16)]
          for a, b, c in re.findall(r"\{0x([0-9A-F]+), \{0x([0-9A-F]+), 0x([0-9A-F]+)\}\}", de[:de.index("};")])}
    if de != dec:
        sys.exit("generate_unicode_tables: the clone's canonical decompositions differ")
    print(f"cross-checked against the clone: categories, whitespace, lowercase and decompositions agree, except "
          f"{len(newer)} code points the clone has as unassigned (new in Unicode {VERSION})")


def render():
    lines = DATA.read_text(encoding="ascii").splitlines()
    source = lines[0]
    gc, runs, dec, ccc = [], [], [], []
    for line in lines:
        if line.startswith("#"):
            continue
        f = line.split()
        if f[0] == "gc":
            gc.append((int(f[1], 16), CATEGORIES.index(f[2])))
        elif f[0] == "lower":
            runs.append((int(f[1], 16), int(f[2]), int(f[3]), int(f[4])))
        elif f[0] == "dec":
            d = [int(x, 16) for x in f[1:]]
            dec.append((d[0], d[1], d[2] if len(d) == 3 else 0))
        elif f[0] == "ccc":
            ccc.append((int(f[1], 16), int(f[2], 16), int(f[3])))
        else:
            sys.exit(f"generate_unicode_tables: bad line in {DATA.name}: {line!r}")

    def table(values, per_line):
        rows = [", ".join(values[i:i + per_line]) for i in range(0, len(values), per_line)]
        return ",\n    ".join(rows)

    out = f"""//
//  unicode_tables.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \\file
/// \\brief Generated Unicode {VERSION} tables for ml/unicode.hpp. Never edit by hand.
/// \\ingroup core
///
/// Written by scripts/generate_unicode_tables.py from scripts/data/ucd-{VERSION}.txt, itself extracted from
/// UnicodeData.txt and cross-checked against llama.cpp's tables. The CTest generated.unicode_tables_up_to_date
/// fails if this file differs from what the generator writes. Not public API.
///
/// {source.lstrip("# ")}

#ifndef snicholls_core_detail_unicode_tables_hpp
#define snicholls_core_detail_unicode_tables_hpp

#include <cstdint>

namespace snicholls::detail::unicode_tables
{{

/// The Unicode version of these tables.
inline constexpr char version[] = "{VERSION}";

/// General categories as runs: (first code point << 5) | category, sorted; each run lasts to the next. The
/// category is an index into ml::unicode_category.
inline constexpr std::uint32_t categories[] = {{
    {table([f"0x{(c << 5) | g:X}" for c, g in gc], 10)}}};

/// The simple lowercase mapping as runs of four: first code point, count, stride, delta.
inline constexpr std::int32_t lowercase_runs[] = {{
    {table([f"{s}, {n}, {st}, {d}" for s, n, st, d in runs], 6)}}};

/// Canonical decompositions: (code point << 42) | (first << 21) | second (0 when there is one), sorted.
inline constexpr std::uint64_t decompositions[] = {{
    {table([f"0x{(c << 42) | (a << 21) | b:X}" for c, a, b in dec], 5)}}};

/// Canonical combining classes other than 0: (first << 29) | (last << 8) | class, sorted.
inline constexpr std::uint64_t combining_classes[] = {{
    {table([f"0x{(a << 29) | (b << 8) | k:X}" for a, b, k in ccc], 6)}}};

}} // namespace snicholls::detail::unicode_tables

#endif // snicholls_core_detail_unicode_tables_hpp
"""
    return out


def main():
    args = sys.argv[1:]
    if args and args[0] == "--extract":
        if len(args) not in (2, 4) or (len(args) == 4 and args[2] != "--clone"):
            sys.exit("usage: generate_unicode_tables.py --extract UnicodeData.txt [--clone unicode-data.cpp]")
        extract(args[1], args[3] if len(args) == 4 else None)
        return
    text = render()
    if args == ["--check"]:
        if not OUT.exists() or OUT.read_text(encoding="utf-8") != text:
            sys.exit(f"{OUT.relative_to(ROOT)} is out of date: run scripts/generate_unicode_tables.py")
        print(f"{OUT.relative_to(ROOT)} is up to date")
        return
    if args:
        sys.exit("usage: generate_unicode_tables.py [--check | --extract UnicodeData.txt [--clone unicode-data.cpp]]")
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
