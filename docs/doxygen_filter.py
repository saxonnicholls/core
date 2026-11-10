#!/usr/bin/env python3
#
#  doxygen_filter.py
#  snicholls-core
#
#  Copyright 2010-2026 Saxon Herschel Nicholls
#
"""Doxygen INPUT_FILTER for headers: works around Doxygen parser gaps without touching the code.

Doxygen sees the filtered text only for documentation, so line numbers must not change.
  - `explicit(<condition>)` (C++20 conditional explicit) becomes `explicit`: Doxygen 1.18 otherwise parses the
    constructor as a member function named `explicit`.
  - C++14 digit separators (`30'000`) lose their quote: Doxygen reads the `'` as a character literal and misparses
    the rest of the file.
"""
import re
import sys

EXPLICIT = re.compile(r"\bexplicit\s*\((?:[^()]|\((?:[^()]|\([^()]*\))*\))*\)")

with open(sys.argv[1], encoding="utf-8") as f:
    text = re.sub(r"(?<=[0-9A-Fa-f])'(?=[0-9A-Fa-f])", "", f.read())
    sys.stdout.write(EXPLICIT.sub("explicit", text))
