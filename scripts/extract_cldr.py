#!/usr/bin/env python3
#
#  extract_cldr.py
#  snicholls-core
#
#  Copyright 2010-2026 Saxon Herschel Nicholls
#
#  Writes the subset of Unicode CLDR that core compiles in, under scripts/data/cldr/, from a pinned CLDR release:
#
#      currencies.tsv    per ISO 4217 code: CLDR's fractions (digits, rounding, cash digits, cash rounding) from
#                        common/supplemental/supplementalData.xml, and the root locale's symbol from
#                        common/main/root.xml
#      territories.tsv   every ISO 3166-1 country: alpha-2, alpha-3 and numeric from supplementalData's codeMappings,
#                        and the English name from common/main/en.xml (a country's currencies come from SIX's list
#                        one instead, which is current where CLDR lags: Bulgaria's euro, 2026)
#      LICENSE           the Unicode licence, which must travel with the data
#
#  ISO 3166-1 is CLDR's regular regions (common/validity/region.xml) that have an alpha-3 and a numeric code below
#  900; this leaves out CLDR's exceptional reservations (AC, CP, DG, EA, IC, TA, which have no numeric code) and
#  the user-assigned XK (Kosovo, 983).
#
#      scripts/extract_cldr.py                 # downloads CLDR (VERSION below) to a temporary folder and rewrites
#                                              # scripts/data/cldr/; generators then read only the committed files
#      scripts/extract_cldr.py --check         # the same download, and exit 1 if the committed files differ
#      scripts/extract_cldr.py --from <dir>    # read an unpacked core.zip instead of downloading
#      scripts/extract_cldr.py --latest <N>    # download CLDR N and report, line by line, how its extract differs
#                                              # from the committed one (informational, never a gate); to move to
#                                              # N, change VERSION, rerun, regenerate, and check babel carries N
#
#  Network, and never a CTest: the committed extract is the input of scripts/generate_iso4217.py and
#  scripts/generate_iso3166.py. The standard library only.
#

import io
import pathlib
import sys
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

VERSION = "47"   # the CLDR release; babel 2.18.0, the fixtures' oracle, carries the same one
URL = f"https://unicode.org/Public/cldr/{VERSION}/core.zip"
ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "scripts" / "data" / "cldr"
MEMBERS = ["LICENSE", "common/supplemental/supplementalData.xml", "common/main/root.xml", "common/main/en.xml",
           "common/validity/region.xml"]


def regular_regions(validity):
    """The regular region codes of common/validity/region.xml, ranges (AC~G) expanded."""
    root = ET.fromstring(validity)
    for e in root.iter("id"):
        if e.get("type") == "region" and e.get("idStatus") == "regular":
            out = set()
            for token in e.text.split():
                if "~" in token:
                    first, last = token.split("~")
                    for c in range(ord(first[-1]), ord(last) + 1):
                        out.add(first[:-1] + chr(c))
                else:
                    out.add(token)
            return out
    raise SystemExit("extract_cldr: no regular regions in region.xml")


def extract(files):
    supplemental = ET.fromstring(files["common/supplemental/supplementalData.xml"])
    root = ET.fromstring(files["common/main/root.xml"])
    en = ET.fromstring(files["common/main/en.xml"])
    regular = regular_regions(files["common/validity/region.xml"])

    header = f"# Unicode CLDR {VERSION} ({URL}), extracted by scripts/extract_cldr.py; never edit by hand.\n"

    fractions = {}
    for e in supplemental.iter("info"):
        if e.get("iso4217"):
            fractions[e.get("iso4217")] = (e.get("digits"), e.get("rounding"), e.get("cashDigits") or "",
                                           e.get("cashRounding") or "")
    symbols = {}
    for e in root.iter("currency"):
        for s in e.findall("symbol"):
            if s.get("alt") is None:
                symbols[e.get("type")] = s.text
    lines = [header, "# code\tdigits\trounding\tcash_digits\tcash_rounding\troot_symbol (an empty field: CLDR gives none)\n"]
    for code in sorted(set(fractions) | set(symbols)):
        f = fractions.get(code, ("", "", "", ""))
        lines.append("\t".join([code, *f, symbols.get(code, "")]) + "\n")
    currencies = "".join(lines)

    names = {}
    for e in en.iter("territory"):
        if e.get("alt") is None:
            names[e.get("type")] = e.text
    lines = [header, "# alpha2\talpha3\tnumeric\tname (en)\n"]
    count = 0
    for e in supplemental.iter("territoryCodes"):
        a2, a3, num = e.get("type"), e.get("alpha3"), e.get("numeric")
        if a2 in regular and a3 and num and int(num) < 900:
            lines.append("\t".join([a2, a3, num, names[a2]]) + "\n")
            count += 1
    territories = "".join(lines)
    return {"currencies.tsv": currencies, "territories.tsv": territories,
            "LICENSE": files["LICENSE"].decode("utf-8-sig")}, count


def main(argv):
    if "--from" in argv:
        folder = pathlib.Path(argv[argv.index("--from") + 1])
        files = {m: (folder / m).read_bytes() for m in MEMBERS}
    else:
        req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
        data = urllib.request.urlopen(req, timeout=120).read()
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            files = {m: z.read(m) for m in MEMBERS}
    out, count = extract(files)
    if "--latest" in argv:
        newer = argv[argv.index("--latest") + 1]
        url = URL.replace(f"/{VERSION}/", f"/{newer}/")
        data = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}),
                                      timeout=120).read()
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            latest, _ = extract({m: z.read(m) for m in MEMBERS})
        same = True
        for n, text in out.items():
            if n == "LICENSE":
                continue
            old = set(text.splitlines()[1:])
            new = set(latest[n].splitlines()[1:])
            for line in sorted(old - new):
                print(f"{n}: CLDR {VERSION} only: {line}")
            for line in sorted(new - old):
                print(f"{n}: CLDR {newer} only:  {line}")
            same = same and old == new
        print(f"extract_cldr: CLDR {newer} {'agrees with' if same else 'differs from'} the committed CLDR {VERSION}")
        return 0 if same else 1
    if "--check" in argv:
        stale = [n for n, text in out.items() if not (OUT / n).exists() or (OUT / n).read_text(encoding="utf-8") != text]
        if stale:
            print(f"extract_cldr: {', '.join(stale)} differ from CLDR {VERSION}; run scripts/extract_cldr.py", file=sys.stderr)
            return 1
        print(f"extract_cldr: scripts/data/cldr/ is CLDR {VERSION} ({count} countries)")
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    for n, text in out.items():
        (OUT / n).write_text(text, encoding="utf-8")
    print(f"extract_cldr: wrote scripts/data/cldr/ from CLDR {VERSION} ({count} countries)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
