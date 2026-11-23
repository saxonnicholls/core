#!/usr/bin/env python3
#
#  generate_money_fixtures.py
#  snicholls-core
#
#  Copyright 2010-2026 Saxon Herschel Nicholls
#
#  Writes tests/fixtures/money/convert.txt and tests/fixtures/money/parse.txt: the expected results of money/rate.hpp's
#  convert and money/money.hpp's parse_money and to_chars, computed by Python's decimal module, an independent
#  implementation, never by core.
#
#  Rounding by decimal is exact here: the working precision is 200 digits, and a value of the form n / d with
#  d < 2^256 that is not a tie lies at least 1 / (2d), about 10^-78, from one, far above 10^-200.
#
#      scripts/generate_money_fixtures.py            # writes the fixtures
#      scripts/generate_money_fixtures.py --check    # exits 1 if the committed fixtures differ (CTest)
#
#  The standard library only. Never edit the fixtures by hand.
#

import decimal
import pathlib
import random
import sys
from decimal import Decimal

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "tests" / "fixtures" / "money"
MAX = 2**256 - 1

MODES = {
    "half_even": decimal.ROUND_HALF_EVEN,
    "half_up": decimal.ROUND_HALF_UP,
    "half_down": decimal.ROUND_HALF_DOWN,
    "toward_zero": decimal.ROUND_DOWN,
    "away_from_zero": decimal.ROUND_UP,
    "floor": decimal.ROUND_FLOOR,
    "ceiling": decimal.ROUND_CEILING,
}

CTX = decimal.Context(prec=200, Emax=decimal.MAX_EMAX, Emin=decimal.MIN_EMIN)


def rounded(numerator, denominator, mode):
    """numerator / denominator (signed) rounded to an integer by `mode`, with decimal."""
    value = CTX.divide(Decimal(numerator), Decimal(denominator))
    return int(value.quantize(Decimal(1), rounding=MODES[mode], context=CTX))


def convert_cases(rng):
    """Lines: from_scale to_scale numerator denominator sign units mode -> sign units residual_sign residual_numerator."""
    lines = ["# from_scale to_scale rate_numerator rate_denominator sign units mode | sign units residual_sign residual_numerator"]
    scales = [0, 2, 3, 6, 8, 9, 18, 24, 36]
    special_units = [0, 1, 5, 15, 25, 35, 10**18, 10**18 + 1, 1000500000000000000000, MAX // 10**40, 2**128 - 1]
    special_rates = [(1, 1), (3, 2), (1, 3), (2, 3), (3000123456789012345678, 10**18), (1, 10**30), (10**12, 7)]
    for _ in range(600):
        fs, ts = rng.choice(scales), rng.choice(scales)
        if rng.random() < 0.4:
            num, den = rng.choice(special_rates)
        else:
            num = rng.randrange(1, 10 ** rng.randint(1, 30))
            den = rng.randrange(1, 10 ** rng.randint(1, 30))
        units = rng.choice(special_units) if rng.random() < 0.3 else rng.randrange(0, 10 ** rng.randint(1, 40))
        sign = rng.choice(["+", "-"])
        mode = rng.choice(list(MODES))
        n = num * 10**ts
        d = den * 10**fs
        if n > MAX or d > MAX:
            continue
        signed = units if sign == "+" else -units
        q = rounded(signed * n, d, mode)
        if abs(q) > MAX:
            continue
        rest = signed * n - q * d   # exact - rounded, times d
        lines.append(f"{fs} {ts} {num} {den} {sign} {units} {mode} | {'-' if q < 0 else '+'} {abs(q)} "
                     f"{'-' if rest < 0 else '+'} {abs(rest)}")
    # ties in every mode, both signs: 0.5, 1.5, 2.5 of a minor unit
    for units in (1, 3, 5):
        for sign in "+-":
            for mode in MODES:
                signed = units if sign == "+" else -units
                q = rounded(signed * 1, 2, mode)
                rest = signed - q * 2
                lines.append(f"0 0 1 2 {sign} {units} {mode} | {'-' if q < 0 else '+'} {abs(q)} "
                             f"{'-' if rest < 0 else '+'} {abs(rest)}")
    return lines


def text_of(units, negative, scale, minimal):
    """The text of a signed amount of minor units at `scale`, with decimal."""
    d = Decimal(units).scaleb(-scale, context=CTX)
    t = format(d, "f") if scale else str(units)
    if scale and "." not in t:
        t += "." + "0" * scale
    if minimal and "." in t:
        t = t.rstrip("0").rstrip(".")
    return ("-" if negative and units else "") + t


def parse_cases(rng):
    """Lines: scale text mode -> outcome; and scale sign units -> minimal all."""
    lines = ["# parse: P scale text mode | sign units   or   P scale text mode | error",
             "# format: F scale sign units | minimal all"]
    scales = [0, 2, 6, 8, 18, 36]
    for _ in range(300):
        scale = rng.choice(scales)
        units = rng.randrange(0, 10 ** rng.randint(1, 50))
        negative = rng.random() < 0.5
        lines.append(f"F {scale} {'-' if negative else '+'} {units} | {text_of(units, negative, scale, True)} "
                     f"{text_of(units, negative, scale, False)}")
    for _ in range(300):
        scale = rng.choice(scales)
        whole = str(rng.randrange(0, 10 ** rng.randint(1, 30)))
        frac = "".join(rng.choice("0123456789") for _ in range(rng.randint(0, scale + 4)))
        text = ("-" if rng.random() < 0.3 else "") + whole + ("." + frac if frac or rng.random() < 0.2 else "")
        mode = rng.choice(list(MODES) + ["none"])
        value = CTX.create_decimal(text)
        scaled = value.scaleb(scale, context=CTX)
        exact = scaled == scaled.to_integral_value()
        if mode == "none":
            if not exact:
                lines.append(f"P {scale} {text} none | too_many_digits")
                continue
            q = int(scaled)
        else:
            q = int(scaled.quantize(Decimal(1), rounding=MODES[mode], context=CTX))
        lines.append(f"P {scale} {text} {mode} | {'-' if q < 0 or (q == 0 and False) else '+'} {abs(q)}")
    for text in ["", ".", "-", "1.2.3", "abc", "1e5", "--1", "1-", "0x10", "+.5"]:
        if text == "+.5":
            lines.append("P 2 +.5 none | + 50")
        else:
            lines.append(f"P 2 {text if text else '<empty>'} none | malformed")
    return lines


def render():
    rng = random.Random(20261123)
    return {"convert.txt": "\n".join(convert_cases(rng)) + "\n", "parse.txt": "\n".join(parse_cases(rng)) + "\n"}


def main(argv):
    files = render()
    if "--check" in argv:
        stale = [n for n, t in files.items() if not (OUT / n).exists() or (OUT / n).read_text() != t]
        if stale:
            print(f"money fixtures: {', '.join(stale)} differ from what scripts/generate_money_fixtures.py writes",
                  file=sys.stderr)
            return 1
        print("money fixtures: up to date")
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    for n, t in files.items():
        (OUT / n).write_text(t)
    print(f"money fixtures: wrote {', '.join(files)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
