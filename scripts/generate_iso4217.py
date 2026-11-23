#!/usr/bin/env python3
#
#  generate_iso4217.py
#  snicholls-core
#
#  Copyright 2010-2026 Saxon Herschel Nicholls
#
#  Writes include/snicholls/core/detail/iso4217_table.hpp: every ISO 4217 currency as a type (struct AUD, struct EUR,
#  ...) and as a row of a sorted constexpr table, from the lists the ISO 4217 maintenance agency (SIX) publishes:
#
#      scripts/data/iso4217/list-one.xml     current currencies and funds   (Pblshd attribute: the table's version)
#      scripts/data/iso4217/list-three.xml   historic denominations         (withdrawal dates)
#      scripts/data/iso4217/entities.tsv     list one's entities (CtryNm) to ISO 3166-1 alpha-2: the countries using
#                                            each currency; curated, and checked against pycountry by
#                                            scripts/generate_country_fixtures.py
#
#  and from the subset of Unicode CLDR that scripts/extract_cldr.py commits:
#
#      scripts/data/cldr/currencies.tsv      cash digits and cash rounding, and the root locale's symbol
#
#  The SIX lists are committed as downloaded, unmodified:
#
#      curl -o scripts/data/iso4217/list-one.xml \
#          https://www.six-group.com/dam/download/financial-information/data-center/iso-currrency/lists/list-one.xml
#      curl -o scripts/data/iso4217/list-three.xml \
#          https://www.six-group.com/dam/download/financial-information/data-center/iso-currrency/lists/list-three.xml
#
#      scripts/generate_iso4217.py            # writes the header
#      scripts/generate_iso4217.py --check    # exits 1 if the committed header differs (CTest generated.iso4217_up_to_date)
#      scripts/generate_iso4217.py --latest   # downloads SIX's current lists to a temporary folder and reports any
#                                             # difference from the committed ones (network; informational, never a gate)
#
#  The standard library only. Never edit the generated header by hand.
#

import pathlib
import sys
import tempfile
import urllib.request
import xml.etree.ElementTree as ET

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "scripts" / "data" / "iso4217"
CLDR = ROOT / "scripts" / "data" / "cldr"
OUT = ROOT / "include" / "snicholls" / "core" / "detail" / "iso4217_table.hpp"
URL = "https://www.six-group.com/dam/download/financial-information/data-center/iso-currrency/lists/list-{}.xml"


def tsv(path):
    """The rows of a committed tab-separated file, '#' lines skipped."""
    for line in path.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            yield line.split("\t")


def cldr_version():
    """The CLDR release of the committed extract, from its first line."""
    first = (CLDR / "currencies.tsv").read_text(encoding="utf-8").splitlines()[0]
    return first.split()[3]   # "# Unicode CLDR 47 (...)"


def entities():
    """{SIX entity name: alpha-2 or None}."""
    return {name: (None if a2 == "-" else a2) for name, a2 in tsv(DATA / "entities.tsv")}


def cldr_currencies():
    """{code: (cash scale or None, cash rounding or None, root symbol or None)}. CLDR's cash digits default to its
    digits, which default to DEFAULT's; a cash rounding of 0 means 1 (no rounding beyond the cash digits)."""
    rows = {r[0]: r[1:] for r in tsv(CLDR / "currencies.tsv")}
    default_digits = int(rows["DEFAULT"][0])
    out = {}
    for code, (digits, _rounding, cash_digits, cash_rounding, symbol) in rows.items():
        if code == "DEFAULT":
            continue
        cash = None
        if cash_digits or cash_rounding:
            scale = int(cash_digits) if cash_digits else (int(digits) if digits else default_digits)
            cash = (scale, max(1, int(cash_rounding or "0")))
        out[code] = (cash[0] if cash else None, cash[1] if cash else None, symbol or None)
    return out


def country_currencies():
    """{alpha-2: sorted ISO 4217 codes} from SIX's list one, funds (IsFund) left out: the inverse of the currencies'
    countries() without the funds. scripts/generate_iso3166.py reads it, so the two directions agree by construction."""
    names = entities()
    out = {}
    for e in ET.parse(DATA / "list-one.xml").getroot().iter("CcyNtry"):
        code = (e.findtext("Ccy") or "").strip()
        name = e.find("CcyNm")
        a2 = names.get((e.findtext("CtryNm") or "").strip())
        if code and a2 and (name is None or name.get("IsFund") != "true"):
            out.setdefault(a2, set()).add(code)
    return {a2: sorted(codes) for a2, codes in out.items()}


def read(folder):
    """(published dates, {code: row}) from list-one and list-three in `folder`. A current code wins over a historic one."""
    one = ET.parse(folder / "list-one.xml").getroot()
    three = ET.parse(folder / "list-three.xml").getroot()
    rows = {}
    for e in three.iter("HstrcCcyNtry"):
        code = (e.findtext("Ccy") or "").strip()
        if not code:
            continue
        name = (e.findtext("CcyNm") or "").strip()
        number = int((e.findtext("CcyNbr") or "0").strip() or 0)
        withdrawn = (e.findtext("WthdrwlDt") or "").strip()
        old = rows.get(code)
        # a code withdrawn more than once (a revaluation) keeps its latest withdrawal
        if old is None or withdrawn > old["withdrawn"]:
            rows[code] = {"code": code, "name": name, "numeric": number, "scale": None, "withdrawn": withdrawn}
    for e in one.iter("CcyNtry"):
        code = (e.findtext("Ccy") or "").strip()
        if not code:
            continue   # a country with no universal currency (Antarctica)
        units = (e.findtext("CcyMnrUnts") or "").strip()
        entity = (e.findtext("CtryNm") or "").strip()
        countries = rows[code]["countries"] if code in rows and not rows[code]["withdrawn"] else set()
        if entity:
            countries.add(entity)
        rows[code] = {
            "code": code,
            "name": (e.findtext("CcyNm") or "").strip(),
            "numeric": int((e.findtext("CcyNbr") or "0").strip()),
            "scale": None if units in ("", "N.A.") else int(units),
            "withdrawn": "",
            "countries": countries,
        }
    for r in rows.values():
        r.setdefault("countries", set())
    return (one.get("Pblshd"), three.get("Pblshd")), dict(sorted(rows.items()))


def with_metadata(rows):
    """The rows, each with the alpha-2 codes of its countries (from entities.tsv) and CLDR's cash and symbol."""
    names, cldr = entities(), cldr_currencies()
    for r in rows.values():
        unknown = sorted(n for n in r["countries"] if n not in names)
        if unknown:
            raise SystemExit(f"iso4217: list one names {unknown} for {r['code']}, missing from "
                             "scripts/data/iso4217/entities.tsv; add each with its alpha-2 (or '-')")
        r["alpha2"] = sorted({names[n] for n in r["countries"] if names[n]})
        r["cash_scale"], r["cash_rounding"], r["symbol"] = cldr.get(r["code"], (None, None, None))
    return rows


def cstr(s):
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def render(published, rows):
    one, three = published
    out = []
    w = out.append
    w("//")
    w("//  iso4217_table.hpp")
    w("//  snicholls-core")
    w("//")
    w("//  Copyright 2010-2026 Saxon Herschel Nicholls")
    w("//")
    w("//  GENERATED by scripts/generate_iso4217.py from the ISO 4217 lists published by SIX")
    w(f"//  (list one {one}, list three {three}) and Unicode CLDR {cldr_version()} (scripts/data/cldr/, the Unicode")
    w("//  licence). Never edit by hand: CTest generated.iso4217_up_to_date.")
    w("//")
    w("")
    w("#ifndef snicholls_core_detail_iso4217_table_hpp")
    w("#define snicholls_core_detail_iso4217_table_hpp")
    w("")
    w("#include <snicholls/core/display.hpp>")
    w("#include <snicholls/core/meta/attributes.hpp>")
    w("")
    w("#include <array>")
    w("#include <cstdint>")
    w("#include <span>")
    w("#include <string_view>")
    w("")
    w("namespace snicholls::detail {")
    w("")
    flat, offsets = [], {}
    for r in rows.values():
        offsets[r["code"]] = len(flat)
        flat.extend(r["alpha2"])
    w("    // The countries of every current code, by alpha-2, each code's run sorted; rows point into it.")
    w(f"    inline constexpr std::array<std::string_view, {len(flat)}> iso4217_countries{{{{")
    for i in range(0, len(flat), 16):
        w("        " + " ".join(f'"{a}",' for a in flat[i:i + 16]))
    w("    }};")
    w("")
    w("    // One row of the ISO 4217 table.")
    w("    struct iso4217_row {")
    w("        std::string_view code;        // the alphabetic code, three capitals")
    w("        std::string_view name;        // the currency's name as SIX publishes it")
    w("        std::uint16_t numeric;        // the numeric code")
    w("        std::int8_t scale;            // the minor units, or -1 where ISO gives none (N.A., and historic codes)")
    w("        std::string_view withdrawn;   // the withdrawal date as SIX gives it (\"2025-03\", \"1989 to 1990\"), empty if current")
    w("        std::span<const std::string_view> countries;   // alpha-2 of the countries using it (SIX list one), sorted")
    w("        std::string_view symbol;      // CLDR's root-locale symbol (\"A$\", \"€\"), or the code where CLDR gives none")
    w("        std::int8_t cash_scale;       // CLDR's cash digits, or -1 where CLDR gives no cash rule")
    w("        std::uint8_t cash_rounding;   // the cash increment in units of 10^-cash_scale (5 for CHF), 0 if none")
    w("    };")
    w("")
    w(f'    inline constexpr std::string_view iso4217_published = "{one}";            // list one')
    w(f'    inline constexpr std::string_view iso4217_historic_published = "{three}";   // list three')
    w(f'    inline constexpr std::string_view iso4217_cldr_version = "{cldr_version()}";               // Unicode CLDR')
    w("")
    w(f"    /// Every code, sorted by alphabetic code: {len(rows)} rows.")
    w(f"    inline constexpr std::array<iso4217_row, {len(rows)}> iso4217_rows{{{{")
    for r in rows.values():
        scale = -1 if r["scale"] is None else r["scale"]
        span = (f'{{iso4217_countries.data() + {offsets[r["code"]]}, {len(r["alpha2"])}}}' if r["alpha2"] else "{}")
        cash_scale = -1 if r["cash_scale"] is None else r["cash_scale"]
        cash_rounding = 0 if r["cash_rounding"] is None else r["cash_rounding"]
        w(f'        {{{cstr(r["code"])}, {cstr(r["name"])}, {r["numeric"]}, {scale}, {cstr(r["withdrawn"])}, {span}, '
          f'{cstr(r["symbol"] or r["code"])}, {cash_scale}, {cash_rounding}}},')
    w("    }};")
    w("")
    w("} // namespace snicholls::detail")
    w("")
    w("namespace snicholls::currency {")
    w("")
    w("    // One type per code. A type with scale() models snicholls::currency_like (and asset_like); the N.A. codes")
    w("    // (XAU, XAG, XPT, XPD, XDR, the funds) and historic ones have none: give one with snicholls::with_scale.")
    w("    // Each declares its presentation, found by ADL: a label, CLDR's root-locale symbol, and minor_units.")
    for index, r in enumerate(rows.values()):
        c = r["code"]
        units = "no ISO minor units" if r["scale"] is None else f"{r['scale']} minor units"
        status = f", withdrawn {r['withdrawn']}" if r["withdrawn"] else ""
        w(f"    /// \\brief ISO 4217 {c}: {r['name']}, numeric {r['numeric']:03d}, {units}{status}.")
        w(f"    struct {c} {{")
        w(f"        /// \\brief The alphabetic code.")
        w(f"        /// \\return `\"{c}\"`.")
        w(f'        static constexpr std::string_view code() noexcept {{ return "{c}"; }}')
        w(f"        /// \\brief The currency's name, as SIX publishes it.")
        w(f"        /// \\return The name.")
        w(f"        static constexpr std::string_view name() noexcept {{ return {cstr(r['name'])}; }}")
        w(f"        /// \\brief The numeric code.")
        w(f"        /// \\return {r['numeric']}.")
        w(f"        static constexpr std::uint16_t numeric() noexcept {{ return {r['numeric']}; }}")
        if r["scale"] is not None:
            w(f"        /// \\brief The minor units: the scale of an amount.")
            w(f"        /// \\return {r['scale']}.")
            w(f"        static constexpr std::uint8_t scale() noexcept {{ return {r['scale']}; }}")
        if r["withdrawn"]:
            w(f"        /// \\brief When the code was withdrawn, as SIX's historic list gives it.")
            w(f"        /// \\return The date.")
            w(f'        static constexpr std::string_view withdrawn() noexcept {{ return "{r["withdrawn"]}"; }}')
        else:
            n = len(r["alpha2"])
            w(f"        /// \\brief The countries using it, by ISO 3166-1 alpha-2, from SIX's list one.")
            w(f"        /// \\return {n} code{'' if n == 1 else 's'}, sorted.")
            w(f"        static constexpr std::span<const std::string_view> countries() noexcept {{ return detail::iso4217_rows[{index}].countries; }}")
        if r["cash_scale"] is not None:
            w(f"        /// \\brief CLDR's cash digits: the scale of a cash amount.")
            w(f"        /// \\return {r['cash_scale']}.")
            w(f"        static constexpr std::uint8_t cash_scale() noexcept {{ return {r['cash_scale']}; }}")
            w(f"        /// \\brief CLDR's cash rounding: a cash amount is a multiple of cash_rounding() x 10^-cash_scale().")
            w(f"        /// \\return {r['cash_rounding']}.")
            w(f"        static constexpr std::uint8_t cash_rounding() noexcept {{ return {r['cash_rounding']}; }}")
        symbol = r["symbol"] or c
        minor = f", ::snicholls::minor_units{{{r['scale']}}}" if r["scale"] is not None else ""
        w(f"        /// \\brief Presentation, found by ADL: the label {cstr(symbol)} (CLDR's root-locale symbol"
          + (", or the code where CLDR gives none" if not r["symbol"] else "") + ")"
          + (f" and {r['scale']} minor units" if minor else "") + ".")
        w(f"        /// \\return The attribute set.")
        w(f"        friend constexpr auto attributes(const {c}&) noexcept {{ return ::snicholls::attrs(::snicholls::label{{{cstr(symbol)}}}{minor}); }}")
        w("    };")
    w("")
    w("} // namespace snicholls::currency")
    w("")
    w("#endif // snicholls_core_detail_iso4217_table_hpp")
    w("")
    return "\n".join(out)


def main(argv):
    if "--latest" in argv:
        with tempfile.TemporaryDirectory() as tmp:
            folder = pathlib.Path(tmp)
            for n in ("one", "three"):
                req = urllib.request.Request(URL.format(n), headers={"User-Agent": "Mozilla/5.0"})
                (folder / f"list-{n}.xml").write_bytes(urllib.request.urlopen(req, timeout=30).read())
            (pa, a), (pb, b) = read(DATA), read(folder)
            if a == b:
                print(f"iso4217: the committed lists ({pa[0]}) agree with SIX's latest ({pb[0]})")
                return 0
            for code in sorted(set(a) | set(b)):
                if a.get(code) != b.get(code):
                    print(f"iso4217: {code}: committed {a.get(code)} latest {b.get(code)}")
            print(f"iso4217: SIX has published {pb[0]}; download it to scripts/data/iso4217/ and regenerate")
            return 1
    published, rows = read(DATA)
    text = render(published, with_metadata(rows))
    if "--check" in argv:
        if not OUT.exists() or OUT.read_text(encoding="utf-8") != text:
            print(f"iso4217: {OUT.relative_to(ROOT)} is not what scripts/generate_iso4217.py writes; regenerate it",
                  file=sys.stderr)
            return 1
        print(f"iso4217: up to date ({len(rows)} codes, list one {published[0]})")
        return 0
    OUT.write_text(text, encoding="utf-8")
    print(f"iso4217: wrote {OUT.relative_to(ROOT)} ({len(rows)} codes, list one {published[0]})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
