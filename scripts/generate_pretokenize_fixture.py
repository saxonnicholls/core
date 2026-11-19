#!/usr/bin/env python3
#
#  generate_pretokenize_fixture.py
#  snicholls-core
#
#  Copyright 2010-2026 Saxon Herschel Nicholls
#
#  Writes tests/fixtures/ml/pretokenize_fixture.txt: texts split by each pre-tokenizer of ml/pretokenize.hpp, with
#  the pieces Python's regex module gives for the original expressions (Hugging Face's tokenizer.json and
#  tiktoken's), each family's sequence applied as Hugging Face's Sequence of isolated Splits applies it. The texts
#  are llama.cpp's tokenizer test strings (models/ggml-vocab-llama-bpe.gguf.inp in a llama.cpp checkout) and the
#  edge cases below.
#
#  Needs the regex module (pip install regex) and a llama.cpp checkout, so CI does not rerun it; run it by hand
#  after changing a pattern or the texts, and commit the result:
#
#      python scripts/generate_pretokenize_fixture.py <llama.cpp>            # writes the fixture
#      python scripts/generate_pretokenize_fixture.py <llama.cpp> --check    # exits 1 if the committed fixture differs

import pathlib
import sys

import regex

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "tests" / "fixtures" / "ml" / "pretokenize_fixture.txt"

CONTRACTIONS = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
GPT2 = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
LLAMA3 = CONTRACTIONS + r"""|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
QWEN2 = CONTRACTIONS + r"""|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
QWEN35 = CONTRACTIONS + r"""|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
GPT4O = "|".join([
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+""" + CONTRACTIONS + "?",
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*""" + CONTRACTIONS + "?",
    r"""\p{N}{1,3}""", r""" ?[^\s\p{L}\p{N}]+[\r\n/]*""", r"""\s*[\r\n]+""", r"""\s+(?!\S)""", r"""\s+"""])
DEEPSEEK_V3 = [r"\p{N}{1,3}", r"[一-龥぀-ゟ゠-ヿ]+",
               r"""[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""]
# DeepSeek LLM's letter class, [A-Za-zµÀ-ÖØ-öø-ƺ...𞤀-𞥃], as code point ranges (written out, its Greek letters
# would not survive canonical normalisation of this file)
DEEPSEEK_LETTERS = [(0x41, 0x5A), (0x61, 0x7A), (0xB5, 0xB5), (0xC0, 0xD6), (0xD8, 0xF6), (0xF8, 0x1BA), (0x1BC, 0x1BF), (0x1C4, 0x293), (0x295, 0x2AF), (0x370, 0x373), (0x376, 0x377), (0x37B, 0x37D), (0x37F, 0x37F), (0x386, 0x386), (0x388, 0x38A), (0x38C, 0x38C), (0x38E, 0x3A1), (0x3A3, 0x3F5), (0x3F7, 0x481), (0x48A, 0x52F), (0x531, 0x556), (0x10A0, 0x10C5), (0x13A0, 0x13F5), (0x13F8, 0x13FD), (0x1C90, 0x1CBA), (0x1CBD, 0x1CBF), (0x1D00, 0x1D2B), (0x1D6B, 0x1D77), (0x1D79, 0x1D9A), (0x1E00, 0x1F15), (0x1F18, 0x1F1D), (0x1F20, 0x1F45), (0x1F48, 0x1F4D), (0x1F50, 0x1F57), (0x1F59, 0x1F59), (0x1F5B, 0x1F5B), (0x1F5D, 0x1F5D), (0x1F5F, 0x1F7D), (0x1F80, 0x1FB4), (0x1FB6, 0x1FBC), (0x1FBE, 0x1FBE), (0x1FC2, 0x1FC4), (0x1FC6, 0x1FCC), (0x1FD0, 0x1FD3), (0x1FD6, 0x1FDB), (0x1FE0, 0x1FEC), (0x1FF2, 0x1FF4), (0x1FF6, 0x1FFC), (0x2102, 0x2102), (0x2107, 0x2107), (0x210A, 0x2113), (0x2115, 0x2115), (0x2119, 0x211D), (0x2124, 0x2124), (0x2126, 0x2126), (0x2128, 0x2128), (0x212A, 0x212D), (0x212F, 0x2134), (0x2139, 0x2139), (0x213C, 0x213F), (0x2145, 0x2149), (0x214E, 0x214E), (0x2183, 0x2184), (0x2C00, 0x2C7B), (0x2C7E, 0x2CE4), (0x2CEB, 0x2CEE), (0x2CF2, 0x2CF3), (0xA640, 0xA66D), (0xA680, 0xA69B), (0xA722, 0xA76F), (0xA771, 0xA787), (0xA78B, 0xA78E), (0xAB70, 0xABBF), (0xFB00, 0xFB06), (0xFB13, 0xFB17), (0xFF21, 0xFF3A), (0xFF41, 0xFF5A), (0x10400, 0x1044F), (0x104B0, 0x104D3), (0x104D8, 0x104FB), (0x10C80, 0x10CB2), (0x10CC0, 0x10CF2), (0x118A0, 0x118DF), (0x1E900, 0x1E943)]
DEEPSEEK_LLM = [r"[\r\n]",
                r"\s?[" + "".join(f"\\U{a:08x}-\\U{b:08x}" for a, b in DEEPSEEK_LETTERS) + "]+",
                r"\s?[!-/:-~！-／：-～‘-‟　-。]+", r"\s+\Z", r"[一-龥ࠀ-一가-퟿]+", r"\p{N}+"]
DEEPSEEK_CODER = [r"[\r\n]", r"\s?\p{L}+", r"\s?\p{P}+", r"[一-龥ࠀ-一가-퟿]+", r"\p{N}"]
FALCON = [r"[\p{P}\$\+<=>\^~\|`]+", GPT2, r"[0-9][0-9][0-9]"]
STARCODER = [r"\p{N}", GPT2]
NEWLINES = [r"[^\n]+|[\n]+"]

FAMILIES = [("gpt2", [GPT2]), ("llama3", [LLAMA3]), ("qwen2", [QWEN2]), ("qwen35", [QWEN35]), ("gpt4o", [GPT4O]),
            ("deepseek_v3", DEEPSEEK_V3), ("deepseek_llm", DEEPSEEK_LLM), ("deepseek_coder", DEEPSEEK_CODER),
            ("falcon", FALCON), ("starcoder", STARCODER), ("newlines", NEWLINES)]

EDGE_CASES = [
    "", " ", "\t", "\n", "\r\n", "a", "'", "''s", "it's IT'S It'Ll we'VE they'D i'M",
    "it'ſ and ſ", "camelCaseWord HTTPServer XMLHttpRequest iPhone McDonald's ÉCOLE éCOLE École aÉb",
    "Ǆungla ǅungla ǆungla ʰello ᾈtest", "123 1234 12345 1234567890 ١٢٣٤٥ ½¾ Ⅻ", "x²+y³=z⁴",
    "  leading and trailing  ", "a  b   c    d", "tab\tand\t\ttabs", "line\nbreak\n\nand\r\n\r\nCRLF \n x",
    " \n \n  \n", "nbsp and　ideographic em\u0085nel", "trailing\n", "trailing \n ",
    "emoji 🙂🚀🔥 👩‍👩‍👧‍👦 🇦🇺 ❤️ x🙂y", "combining é ñ ǟ ́alone",
    "हिन्दी नमस्ते", "ภาษาไทย", "العربية مرحبا", "日本語のテキストとカタカナ", "中文分词测试", "한국어 텍스트",
    "Ελληνικά ΚΕΦΑΛΑΙΑ", "русский ТЕКСТ", "Ǉ ǈ ǉ", "punct!!! ??? ... --- /// \\\\ ''' \"\"\"",
    "path/to/file.txt http://example.com/a?b=c#d", "a/b//c\r\n/d", "!hello ?world .com", "$100 +5 <a> =b ^c ~d |e `f`",
    "（全角）、。「括弧」…", "def f(x):\n    return x**2\n\n\n", "    indented", "x​y‍z",
    "\U0001F600\U0001F601", "µÀ×÷ƻƼ", "MIXEDcase mixedCASE", "ﬁ ﬀ ﬃ",
]


def split_isolated(text, pattern):
    pieces, gap = [], 0
    for m in regex.finditer(pattern, text):
        if m.end() == m.start():
            continue
        if gap < m.start():
            pieces.append(text[gap:m.start()])
        pieces.append(m.group())
        gap = m.end()
    if gap < len(text):
        pieces.append(text[gap:])
    return pieces


def split_sequence(text, patterns):
    pieces = [text] if text else []
    for p in patterns:
        pieces = [q for piece in pieces for q in split_isolated(piece, p)]
    return pieces


def hexs(s):
    return s.encode("utf-8").hex() or "-"


def main():
    args = sys.argv[1:]
    check = "--check" in args
    args = [a for a in args if a != "--check"]
    if len(args) != 1:
        sys.exit("usage: generate_pretokenize_fixture.py <llama.cpp checkout> [--check]")
    inp = pathlib.Path(args[0]) / "models" / "ggml-vocab-llama-bpe.gguf.inp"
    texts = inp.read_text(encoding="utf-8").split("\n__ggml_vocab_test__\n")
    texts[-1] = texts[-1].removesuffix("\n__ggml_vocab_test__\n")
    texts = [t for t in texts] + EDGE_CASES
    lines = ["# Pre-tokenizer splits: Python's regex module on the original expressions, through",
             "# scripts/generate_pretokenize_fixture.py. Never edit by hand.",
             f"# regex {regex.__version__}. text <utf-8 hex, '-' if empty>, one line per text; then family <name>, and",
             "# per text: pieces <byte length of each piece>."]
    lines += [f"text {hexs(t)}" for t in texts]
    for name, patterns in FAMILIES:
        lines.append(f"family {name}")
        for t in texts:
            pieces = split_sequence(t, patterns)
            assert "".join(pieces) == t
            lines.append(" ".join(["pieces"] + [str(len(p.encode("utf-8"))) for p in pieces]))
    content = "\n".join(lines) + "\n"
    if check:
        if not OUT.exists() or OUT.read_text(encoding="utf-8") != content:
            sys.exit(f"{OUT.relative_to(ROOT)} is out of date")
        print(f"{OUT.relative_to(ROOT)} is up to date")
        return
    OUT.write_text(content, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}: {len(texts)} texts, {len(FAMILIES)} families")


if __name__ == "__main__":
    main()
