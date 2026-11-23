#!/usr/bin/env python3
#
#  generate_check_digit_fixtures.py
#  snicholls-core
#
#  Copyright 2010-2026 Saxon Herschel Nicholls
#
"""Generate the oracle fixture for include/snicholls/core/ids/check_digits.hpp (tests/fixtures/ids/check_digits.txt).

Every expected value comes from the Python package `python-stdnum`, pinned to STDNUM_VERSION below, never from
core: `stdnum.luhn`, `stdnum.isin`, `stdnum.iban` (with `stdnum.iso7064.mod_97_10` for the checksum alone),
`stdnum.lei`, `stdnum.cusip` and `stdnum.gb.sedol`. The inputs are well-known identifiers, pseudo-random bodies
(fixed seed) completed with stdnum's check digits, and each of those with one character changed, two adjacent
characters swapped, a character dropped and a character added.

    ./scripts/generate_check_digit_fixtures.py           rewrite the fixture (needs python-stdnum==2.2)
    ./scripts/generate_check_digit_fixtures.py --check   exit 1 if the committed fixture differs from this script's
                                                         output; skipped (exit 0) without python-stdnum 2.2

Format: tab-separated, '#' starts a comment: `scheme input verdict check`, where verdict is `valid` or the name of
the stdnum exception that validate raised, and check is stdnum's check digit(s) for the input's body (the input
without its check digits; for an IBAN, the input with them), or `-` when stdnum cannot compute one. For an IBAN the
verdict is the ISO 7064 MOD 97-10 check alone (stdnum.iso7064.mod_97_10 on the rearranged number), since core
does not check the per-country registry, and a fifth field gives stdnum.iban's own verdict for information.
tests/ids/check_digits_test.cpp states how core's answer follows from each row.
"""
import pathlib
import random
import sys
import types

ROOT = pathlib.Path(__file__).resolve().parent.parent
STDNUM_VERSION = "2.2"
OUT = ROOT / "tests" / "fixtures" / "ids" / "check_digits.txt"

DIGITS = "0123456789"
ALNUM = DIGITS + "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CONSONANTS = "BCDFGHJKLMNPQRSTVWXYZ"
KNOWN = {
    "luhn": ["79927398713", "4111111111111111", "4012888888881881", "378282246310005", "490154203237518", "0", "18"],
    "isin": ["US0378331005", "AU0000XVGZA3", "GB0002634946", "DE000BAY0017", "US5949181045", "JP3633400001",
             "XS2021832634", "US0378331006", "us0378331005"],
    "iban": ["GB82WEST12345698765432", "DE89370400440532013000", "FR1420041010050500013M02606", "NO9386011117947",
             "SA0380000000608010167519", "MT84MALT011000012345MTLCAST001S", "CH9300762011623852957",
             "GB00WEST12345698765432", "GB01WEST12345698765432", "GB99WEST12345698765432"],
    "lei": ["5493001KJTIIGC8Y1R12", "529900T8BM49AURSDO55", "HWUPKR0MPOU8FGXBT394", "7LTWFZYICNSX8D621K86",
            "5493001KJTIIGC8Y1R17"],
    "cusip": ["037833100", "594918104", "38259P508", "68389X105", "037833101", "0378331OO"],
    "sedol": ["0263494", "B0YBKJ7", "B0YBLH2", "B0WNLY7", "0263495", "B0YBKJA", "1A00000", "A000000"],
}


def outcome(fn, s):
    try:
        fn(s)
        return "valid"
    except Exception as e:   # noqa: BLE001: the oracle's exception class is the answer
        return type(e).__name__


def calc(fn, s):
    try:
        return fn(s)
    except Exception:   # noqa: BLE001
        return "-"


def oracle():
    """The stdnum modules, and per scheme: validate, the check digits of the input's body, the input's alphabet."""
    from stdnum import cusip, iban, isin, lei, luhn
    from stdnum.gb import sedol
    from stdnum.iso7064 import mod_97_10
    m = types.SimpleNamespace(cusip=cusip, iban=iban, isin=isin, lei=lei, luhn=luhn, sedol=sedol, mod_97_10=mod_97_10)
    m.table = {
        "luhn": (luhn.validate, lambda s: luhn.calc_check_digit(s[:-1]), DIGITS),
        "isin": (isin.validate, lambda s: isin.calc_check_digit(s[:11]), ALNUM),
        "iban": (lambda s: mod_97_10.validate(s[4:] + s[:4]), iban.calc_check_digits, ALNUM),
        "lei": (lei.validate, lambda s: mod_97_10.calc_check_digits(s[:18]), ALNUM),
        "cusip": (cusip.validate, lambda s: cusip.calc_check_digit(s[:8]), ALNUM + "*@#"),
        "sedol": (sedol.validate, lambda s: sedol.calc_check_digit(s[:6]), DIGITS + CONSONANTS),
    }
    return m


def bodies(name, rng, m):
    """Forty identifiers of the scheme: pseudo-random bodies completed with stdnum's check digits."""
    out = []
    for _ in range(40):
        if name == "luhn":
            body = "".join(rng.choice(DIGITS) for _ in range(rng.randint(5, 18)))
            out.append(body + m.luhn.calc_check_digit(body))
        elif name == "isin":
            body = rng.choice(["US", "GB", "AU", "DE", "FR", "JP", "CH", "XS", "CA", "NL"]) + \
                "".join(rng.choice(ALNUM) for _ in range(9))
            out.append(body + m.isin.calc_check_digit(body))
        elif name == "iban":
            country = rng.choice(["GB", "DE", "FR", "NO", "CH", "NL", "ES", "IT", "AU", "ZZ"])
            bban = "".join(rng.choice(ALNUM if rng.random() < 0.3 else DIGITS) for _ in range(rng.randint(8, 30)))
            out.append(country + m.iban.calc_check_digits(country + "00" + bban) + bban)
        elif name == "lei":
            body = "".join(rng.choice(ALNUM) for _ in range(18))
            out.append(body + m.mod_97_10.calc_check_digits(body))
        elif name == "cusip":
            body = "".join(rng.choice(ALNUM + "*@#") for _ in range(8))
            out.append(body + m.cusip.calc_check_digit(body))
        else:
            first = rng.choice(DIGITS + CONSONANTS)
            rest = DIGITS if first in DIGITS else DIGITS + CONSONANTS
            body = first + "".join(rng.choice(rest) for _ in range(5))
            out.append(body + m.sedol.calc_check_digit(body))
    return out


def corrupt(s, rng, alphabet):
    i = rng.randrange(len(s))
    changed = s[:i] + rng.choice([c for c in alphabet if c != s[i]]) + s[i + 1:]
    j = rng.randrange(len(s) - 1) if len(s) > 1 else 0
    swapped = s[:j] + s[j + 1:j + 2] + s[j:j + 1] + s[j + 2:]
    return [changed, swapped, s[:i] + s[i + 1:], s[:i] + rng.choice(alphabet) + s[i:]]


def render():
    m = oracle()
    rng = random.Random(20261123)
    lines = [f"# generated by scripts/generate_check_digit_fixtures.py from python-stdnum {STDNUM_VERSION}; do not edit\n",
             "# scheme\tinput\tverdict\tcheck\t[iban: stdnum.iban verdict]\n"]
    for name, (validate, check, alphabet) in m.table.items():
        inputs = list(KNOWN[name])
        for s in bodies(name, rng, m):
            inputs.append(s)
            inputs += corrupt(s, rng, alphabet)
        seen = set()
        for s in inputs:
            if s in seen:
                continue
            seen.add(s)
            fields = [name, s, outcome(validate, s), calc(check, s)]
            if name == "iban":
                fields.append(outcome(m.iban.validate, s))
            lines.append("\t".join(fields) + "\n")
    return "".join(lines)


def main():
    check = sys.argv[1:] == ["--check"]
    if sys.argv[1:] and not check:
        sys.exit("usage: generate_check_digit_fixtures.py [--check]")
    try:
        import stdnum
    except ImportError:
        stdnum = None
    if stdnum is None or stdnum.__version__ != STDNUM_VERSION:
        if check:
            print(f"check digit fixtures: skipped, python-stdnum=={STDNUM_VERSION} is not installed")
            return 0
        sys.exit(f"generate_check_digit_fixtures: needs python-stdnum=={STDNUM_VERSION}")
    text = render()
    if check:
        same = OUT.read_text(encoding="utf-8") == text
        print("check digit fixtures: " + ("up to date" if same else "DIFFER"))
        return 0 if same else 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}: {text.count(chr(10)) - 2} cases")
    return 0


if __name__ == "__main__":
    sys.exit(main())
