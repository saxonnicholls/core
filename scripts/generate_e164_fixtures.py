#!/usr/bin/env python3
#
#  generate_e164_fixtures.py
#  snicholls-core
#
#  Copyright 2010-2026 Saxon Herschel Nicholls
#
"""Generate the oracle fixtures for include/snicholls/core/phone_number.hpp (tests/fixtures/e164/parse.tsv).

Every expected value comes from the Python package `phonenumbers` (libphonenumber's metadata and parser), pinned to
PHONENUMBERS_VERSION below, never from core: `phonenumbers.parse(text, None)` and `format_number(..., E164)`. A text
written with the international prefix 00 is parsed with region "GB", whose international prefix is 00, since with
no region phonenumbers reads only "+".

The inputs are every region's example fixed-line and mobile numbers and every non-geographic entity's example
number, written in the forms people use (E.164, phonenumbers' international format, hyphens, dots, parentheses,
00), with a trunk 0 put in after the country code, without the "+", with extra digits, with letters; a sweep of
every country code from 0 to 999; and the edge cases below.

    ./scripts/generate_e164_fixtures.py           rewrite the fixture (needs phonenumbers==9.0.44)
    ./scripts/generate_e164_fixtures.py --check   exit 1 if the committed fixture differs from this script's
                                                  output; skipped (exit 0) without phonenumbers 9.0.44

Format: tab-separated, '#' starts a comment: `kind input oracle country_code national_number trunk_zero`, where
oracle is the E.164 text phonenumbers formats, or the name of the NumberParseException it raised (then the last
three fields are empty), and trunk_zero is 1 when the national prefix for parsing of the code's main region matches
a leading 0 (the same rule as scripts/generate_e164_country_codes.py, computed here from phonenumbers' metadata). tests/phone_number_test.cpp states how core's result follows from each row; core is stricter than
phonenumbers' parser in documented ways (no letters, no extensions, at most 15 digits, a trunk 0 refused rather
than removed), and those rules are decided from the input and the oracle, not from core.
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
PHONENUMBERS_VERSION = "9.0.44"
OUT = ROOT / "tests" / "fixtures" / "e164" / "parse.tsv"

EDGE_CASES = [
    "+61 412 345 678", "+61-412-345-678", "(+61) 412345678", "00 61 412 345 678", "0061412345678",
    "+61412345678", " +61 412 345 678 ", "+61.412.345.678", "+61/412/345/678", "+ 61 412 345 678",
    "0412 345 678", "61412345678", "+61 0412 345 678", "+44 (0)20 7946 0000", "+44 20 7946 0000",
    "+39 06 6988 4444", "+39 347 123 4567", "+1 (650) 253-0000", "+1 650 253 0000", "+7 495 123-45-67",
    "+7 701 123 4567", "+800 1234 5678", "+882 1234 5678", "+979 1234 5678", "+999 1234 5678", "+0 123 4567",
    "+00 61 412", "+28 1234 5678", "+42 1234 5678", "+1", "+12", "+123", "+61 4", "+61 41", "+1 23",
    "+123456789012345", "+1234567890123456", "+12345678901234567", "+1 2345678901234", "+1 23456789012345",
    "+61 412 345 678 ext. 9", "+61 412 345 678 x9", "+1-800-FLOWERS", "+61 412 ABC 678", "tel:+61-412-345-678",
    "+61 412 345 678;ext=9", "+61++412345678", "00+61412345678", "011 61 412 345 678", "0 0 61 412 345 678",
    "+353 1 234 5678", "+353 01 234 5678", "+225 01 23 45 67 89", "+229 01 20 21 12 34", "+241 06 03 12 34",
    "+36 1 234 5678", "+36 06 1 234 5678", "+86 10 1234 5678", "+86 010 1234 5678", "+81 3-1234-5678",
    "+81 03-1234-5678", "+49 30 123456", "+49 030 123456", "+33 1 23 45 67 89", "+33 01 23 45 67 89",
    "+1 1 650 253 0000", "+1 1 416 555 0123", "+7 8 495 123 4567", "+7 8 701 123 4567", "+36 06 20 123 4567",
    "+375 80 29 123 4567", "+375 029 123 4567", "+370 8 612 34567",
]


def oracle(pn, text):
    region = "GB" if text.lstrip(" ()-./").startswith("00") else None
    try:
        n = pn.parse(text, region)
    except pn.NumberParseException as e:
        names = {pn.NumberParseException.INVALID_COUNTRY_CODE: "INVALID_COUNTRY_CODE",
                 pn.NumberParseException.NOT_A_NUMBER: "NOT_A_NUMBER",
                 pn.NumberParseException.TOO_SHORT_AFTER_IDD: "TOO_SHORT_AFTER_IDD",
                 pn.NumberParseException.TOO_SHORT_NSN: "TOO_SHORT_NSN",
                 pn.NumberParseException.TOO_LONG: "TOO_LONG"}
        return names[e.error_type], "", "", ""
    return (pn.format_number(n, pn.PhoneNumberFormat.E164), str(n.country_code),
            pn.national_significant_number(n), trunk_zero(pn, n.country_code))


def trunk_zero(pn, code):
    md = pn.PhoneMetadata.metadata_for_region_or_calling_code(code, pn.region_code_for_country_code(code))
    npfp = md.national_prefix_for_parsing
    return "1" if npfp and re.match(npfp, "0123456789012") else "0"


def examples(pn):
    """(country code, national significant number, international format) of every example number."""
    out = []
    types = (pn.PhoneNumberType.FIXED_LINE, pn.PhoneNumberType.MOBILE)
    for region in sorted(pn.SUPPORTED_REGIONS):
        seen = set()
        for t in types:
            n = pn.example_number_for_type(region, t)
            if n is None:
                continue
            nsn = pn.national_significant_number(n)
            if nsn in seen:
                continue
            seen.add(nsn)
            out.append((n.country_code, nsn, pn.format_number(n, pn.PhoneNumberFormat.INTERNATIONAL)))
    for cc in sorted(pn.COUNTRY_CODES_FOR_NON_GEO_REGIONS):
        n = pn.example_number_for_non_geo_entity(cc)
        if n is not None:
            out.append((n.country_code, pn.national_significant_number(n),
                        pn.format_number(n, pn.PhoneNumberFormat.INTERNATIONAL)))
    return out


def inputs(pn):
    rows = [("edge", t) for t in EDGE_CASES]
    for k, (cc, nsn, intl) in enumerate(examples(pn)):
        rest = intl[len(f"+{cc}"):].strip()
        rows.append(("e164", f"+{cc}{nsn}"))
        rows.append(("international", intl))
        variant = k % 5
        if variant == 0:
            rows.append(("hyphens", intl.replace(" ", "-")))
        elif variant == 1:
            rows.append(("parentheses", f"(+{cc}) {rest}"))
        elif variant == 2:
            rows.append(("idd00", f"00 {cc} {rest}"))
        elif variant == 3:
            rows.append(("dots", f"+{cc}.{rest.replace(' ', '.')}"))
        else:
            rows.append(("idd00", f"00{cc}{nsn}"))
        rows.append(("trunk0", f"+{cc} 0{rest}"))
        if k % 4 == 0:
            rows.append(("no_plus", f"{cc} {rest}"))
        if k % 6 == 0:
            extra = "1234567890"[: max(1, 16 - len(f"{cc}{nsn}") + k % 3)]
            rows.append(("too_long", f"+{cc} {rest} {extra}"))
        if k % 9 == 0:
            rows.append(("letters", f"+{cc} {rest} ext. 12"))
    for code in range(0, 1000):
        rows.append(("cc_sweep", f"+{code} 2345 6789"))
    return rows


def render(pn):
    lines = ["# GENERATED by scripts/generate_e164_fixtures.py - do not edit by hand.",
             f"# phonenumbers {PHONENUMBERS_VERSION}: parse(input, None) (region GB for a leading 00), format E164",
             "# kind<TAB>input<TAB>oracle<TAB>country_code<TAB>national_number<TAB>trunk_zero"]
    seen = set()
    for kind, text in inputs(pn):
        if text in seen:
            continue
        seen.add(text)
        if "\t" in text:
            sys.exit(f"generate_e164_fixtures: a tab in {text!r}")
        lines.append("\t".join((kind, text) + oracle(pn, text)))
    return "\n".join(lines) + "\n"


def main():
    check = sys.argv[1:] == ["--check"]
    if sys.argv[1:] and not check:
        sys.exit("usage: generate_e164_fixtures.py [--check]")
    try:
        import phonenumbers as pn
    except ImportError:
        pn = None
    if pn is None or pn.__version__ != PHONENUMBERS_VERSION:
        if check:
            print(f"e164 fixtures: skipped, phonenumbers=={PHONENUMBERS_VERSION} is not installed")
            return 0
        sys.exit(f"generate_e164_fixtures: needs phonenumbers=={PHONENUMBERS_VERSION}")
    text = render(pn)
    if check:
        same = OUT.read_text(encoding="utf-8") == text
        print("e164 fixtures: " + ("up to date" if same else "DIFFER"))
        return 0 if same else 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}: {text.count(chr(10)) - 3} cases")
    return 0


if __name__ == "__main__":
    sys.exit(main())
