#!/usr/bin/env python3
#
#  generate_tokens.py
#  snicholls-core
#
#  Copyright 2010-2026 Saxon Herschel Nicholls
#
#  Writes include/snicholls/core/detail/token_table.hpp (the major tokens as types, snicholls::token::<chain>::<SYMBOL>,
#  and the list of them, snicholls::token::all) and tests/fixtures/tokens/tokens.txt (the same rows, read by the
#  tests), from scripts/data/tokens/issuers.json alone: every token from a first-party page (its issuer's, or the
#  bridge operator's, the pegging party's or the canonical contract's), with that page's URL, the issuer and the kind
#  (native, bridged, pegged or wrapped).
#
#      scripts/generate_tokens.py                    # writes the header and the fixture
#      scripts/generate_tokens.py --check            # exits 1 if either differs (CTest generated.tokens_up_to_date)
#      scripts/generate_tokens.py --cross-check      # downloads Uniswap Labs' default token list, the pinned npm
#                                                    # release, to a temporary directory and compares every curated
#                                                    # address and decimals value with it: 0 if all agree, 1 if any
#                                                    # differs, 77 if the download fails (CTest
#                                                    # external.tokens_cross_check, which passes --ctest)
#      scripts/generate_tokens.py --latest           # the same against npm's latest release (informational)
#
#  The cross-check writes nothing to the repository: Uniswap's list is GPL-3.0, so none of it is committed. With
#  --ctest it runs only if --enabled is given (the CMake option SNICHOLLS_CORE_NETWORK_TESTS) or the environment
#  variable SNICHOLLS_CORE_NETWORK_TESTS is 1, and otherwise exits 77 (skipped).
#
#  The curated set is CURATED below: core holds only the major tokens; the long tail is servers' run-time registry.
#  Every address must already be in its chain's canonical form (EIP-55 for EVM chains); the compiler checks that with
#  core's Keccak-256. The standard library only. Never edit the generated files by hand.
#

import hashlib
import io
import json
import os
import pathlib
import sys
import tarfile
import tempfile
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "scripts" / "data" / "tokens" / "issuers.json"
HEADER = ROOT / "include" / "snicholls" / "core" / "detail" / "token_table.hpp"
FIXTURE = ROOT / "tests" / "fixtures" / "tokens" / "tokens.txt"

# The cross-check's reference: Uniswap Labs' default token list, a pinned npm release, never committed.
NPM = "https://registry.npmjs.org/@uniswap/default-token-list"
PINNED = ("22.34.0", NPM + "/-/default-token-list-22.34.0.tgz",
          "66796d5f4b99ca05c1d49af7b7ac78c9c169a20f14567afbf84e21719f44513c")
LIST_PATH = "package/build/uniswap-default.tokenlist.json"
SKIP = 77

# chain id -> (namespace in snicholls::token, the chain type, its display name)
CHAINS = {
    1: ("ethereum", "ethereum", "Ethereum"),
    10: ("optimism", "optimism", "OP Mainnet"),
    56: ("bsc", "bsc", "BNB Smart Chain"),
    137: ("polygon", "polygon", "Polygon"),
    8453: ("base", "base", "Base"),
    42161: ("arbitrum", "arbitrum", "Arbitrum One"),
    43114: ("avalanche", "avalanche", "Avalanche C-Chain"),
    728126428: ("tron", "tron", "TRON"),
}

# (chain id, symbol as issuers.json writes it): exactly the tokens of issuers.json
CURATED = [
    (1, "USDT"), (1, "USDC"), (1, "DAI"), (1, "WETH"), (1, "WBTC"), (1, "EURC"),
    (10, "USDT"), (10, "USDC"),
    (56, "USDT"), (56, "USDC"),
    (137, "USDT0"), (137, "USDC"),
    (8453, "USDT"), (8453, "USDC"),
    (42161, "USDT0"), (42161, "USDC"),
    (43114, "USDT"), (43114, "USDC"),
    (728126428, "USDT"),
]

FIELDS = ("chainId", "address", "symbol", "name", "decimals", "kind", "issuer", "source")
KINDS = ("native", "bridged", "pegged", "wrapped")


def load():
    doc = json.loads(DATA.read_text())
    rows = doc["tokens"]
    for r in rows:
        missing = [f for f in FIELDS if f not in r]
        if missing:
            raise SystemExit(f"tokens: {r.get('symbol')} on {r.get('chainId')} lacks {', '.join(missing)}")
        if r["kind"] not in KINDS:
            raise SystemExit(f"tokens: {r['symbol']} on {r['chainId']}: kind {r['kind']!r} is not one of {KINDS}")
        if r["chainId"] not in CHAINS:
            raise SystemExit(f"tokens: chain {r['chainId']} has no chain type")
        if "page_address" in r and r["page_address"].lower() != r["address"].lower():
            raise SystemExit(f"tokens: {r['symbol']} on {r['chainId']}: address and page_address differ")
    keys = [(r["chainId"], r["symbol"]) for r in rows]
    if len(set(keys)) != len(keys):
        raise SystemExit("tokens: a (chain, symbol) appears twice")
    if sorted(keys) != sorted(CURATED):
        raise SystemExit(f"tokens: issuers.json and CURATED differ: {sorted(set(keys) ^ set(CURATED))}")
    for c in doc["corroborations"]:
        match = [r for r in rows if (r["chainId"], r["symbol"]) == (c["chainId"], c["symbol"])]
        if not match or match[0]["address"].lower() != c["address"].lower():
            raise SystemExit(f"tokens: {c['symbol']} on {c['chainId']} disagrees with {c['source']}")
    return doc["accessed"], sorted(rows, key=lambda r: (r["chainId"], r["symbol"]))


def cstr(s):
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def page(r):
    """The first-party page: the first word of `source`."""
    return r["source"].split(" ")[0]


def render_header(accessed, rows):
    out = []
    w = out.append
    w("//")
    w("//  token_table.hpp")
    w("//  snicholls-core")
    w("//")
    w("//  Copyright 2010-2026 Saxon Herschel Nicholls")
    w("//")
    w("//  GENERATED by scripts/generate_tokens.py from scripts/data/tokens/issuers.json (the issuers' and bridge")
    w(f"//  operators' own pages, accessed {accessed}). Never edit by hand: CTest generated.tokens_up_to_date.")
    w("//")
    w("")
    w("#ifndef snicholls_core_detail_token_table_hpp")
    w("#define snicholls_core_detail_token_table_hpp")
    w("")
    w("#include <snicholls/core/chain/chains.hpp>")
    w("#include <snicholls/core/meta/type_list.hpp>")
    w("#include <snicholls/core/token/basic_token.hpp>")
    w("")
    w("#include <string_view>")
    w("")
    w("namespace snicholls::detail {")
    w("")
    w(f"    inline constexpr std::string_view token_data_accessed = {cstr(accessed)};   // issuers.json")
    w("")
    w("} // namespace snicholls::detail")
    w("")
    w("namespace snicholls::token {")
    w("")
    w("    /// \\brief Who stands behind a token on its chain.")
    w("    enum class issuance {")
    w("        native,  ///< the issuer's own deployment (Circle's USDC, Tether's USDT on Ethereum)")
    w("        bridged, ///< minted by a bridge against the token locked on another chain (USDT0, OP's bridged USDT)")
    w("        pegged,  ///< a third party's custodial token, backed by reserves it holds (Binance-Peg)")
    w("        wrapped  ///< a contract that wraps another asset (WETH, WBTC)")
    w("    };")
    w("")
    w("} // namespace snicholls::token")
    names = []
    for cid in sorted({r["chainId"] for r in rows}):
        ns, chain, display = CHAINS[cid]
        w("")
        w(f"/// \\brief The major tokens on {display} (chain id {cid}).")
        w(f"namespace snicholls::token::{ns} {{")
        for r in [r for r in rows if r["chainId"] == cid]:
            sym = r["symbol"]
            names.append(f"{ns}::{sym}")
            w("")
            w(f"    /// \\brief {sym} on {display}: `{r['address']}`, {r['decimals']} decimals, code `{sym}:{cid}`;")
            w(f"    /// {r['kind']}, {r['issuer']}.")
            w("    ///")
            w(f"    /// Source: <{page(r)}>.")
            w(f"    struct {sym} : basic_token<::snicholls::chain::{chain}, {cstr(r['address'])}, {r['decimals']}, {cstr(sym)}> {{")
            w("        /// \\brief The name, as the contract's `name()` returns it.")
            w(f"        /// \\return `{cstr(r['name'])}`.")
            w(f"        static constexpr std::string_view name() noexcept {{ return {cstr(r['name'])}; }}")
            w("        /// \\brief Who issues it on this chain.")
            w(f"        /// \\return `{cstr(r['issuer'])}`.")
            w(f"        static constexpr std::string_view issuer() noexcept {{ return {cstr(r['issuer'])}; }}")
            w("        /// \\brief Whether it is the issuer's own deployment, bridged, pegged or wrapped.")
            w(f"        /// \\return `issuance::{r['kind']}`.")
            w(f"        static constexpr issuance kind() noexcept {{ return issuance::{r['kind']}; }}")
            w("        /// \\brief The first-party page that lists this address.")
            w(f"        /// \\return `{cstr(page(r))}`.")
            w(f"        static constexpr std::string_view source() noexcept {{ return {cstr(page(r))}; }}")
            w("    };")
        w("")
        w(f"}} // namespace snicholls::token::{ns}")
    w("")
    w("namespace snicholls::token {")
    w("")
    w(f"    /// \\brief Every token type, in order of chain id and then symbol: {len(names)} types.")
    w("    using all = meta::type_list<")
    for i, n in enumerate(names):
        w(f"        {n}{',' if i + 1 < len(names) else ''}")
    w("    >;")
    w("")
    w("} // namespace snicholls::token")
    w("")
    w("#endif // snicholls_core_detail_token_table_hpp")
    return "\n".join(out) + "\n"


def render_fixture(accessed, rows):
    out = [
        "# GENERATED by scripts/generate_tokens.py from scripts/data/tokens/issuers.json: never edit by hand.",
        f"# The issuers' and bridge operators' own pages, accessed {accessed}.",
        "# chain_id<TAB>symbol<TAB>decimals<TAB>address<TAB>kind<TAB>name",
    ]
    for r in rows:
        out.append("\t".join([str(r["chainId"]), r["symbol"], str(r["decimals"]), r["address"], r["kind"], r["name"]]))
    return "\n".join(out) + "\n"


def outputs():
    accessed, rows = load()
    return {HEADER: render_header(accessed, rows), FIXTURE: render_fixture(accessed, rows)}


def fetch(url):
    with urllib.request.urlopen(url, timeout=60) as f:
        return f.read()


def cross_check(latest):
    """Compares every curated token with Uniswap's list, downloaded into a temporary directory and deleted."""
    _, rows = load()
    try:
        if latest:
            meta = json.loads(fetch(NPM + "/latest"))
            version, url, sha = meta["dist"]["tarball"].rsplit("-", 1)[1][:-4], meta["dist"]["tarball"], None
        else:
            version, url, sha = PINNED
        with tempfile.TemporaryDirectory() as tmp:
            tgz = pathlib.Path(tmp) / "list.tgz"
            tgz.write_bytes(fetch(url))
            got = hashlib.sha256(tgz.read_bytes()).hexdigest()
            if sha is not None and got != sha:
                print(f"tokens: {url} has SHA-256 {got}, not the pinned {sha}", file=sys.stderr)
                return 1
            with tarfile.open(tgz) as tar:
                full = json.load(io.TextIOWrapper(tar.extractfile(LIST_PATH), encoding="utf-8"))
    except (OSError, ValueError, KeyError) as e:
        print(f"tokens: cannot download Uniswap's token list ({e}): skipped")
        return SKIP
    by_address = {(t["chainId"], t["address"].lower()): t for t in full["tokens"]}
    by_symbol = {(t["chainId"], t["symbol"]): t for t in full["tokens"]}
    differ = 0
    for r in rows:
        u = by_address.get((r["chainId"], r["address"].lower()))
        if u is None:
            other = by_symbol.get((r["chainId"], r["symbol"]))
            if other is not None:
                print(f"tokens: DIFFERS {r['symbol']}:{r['chainId']}: issuer {r['address']}, Uniswap {other['address']}")
                differ += 1
            else:
                print(f"tokens: absent  {r['symbol']}:{r['chainId']} is not in Uniswap's list")
            continue
        bad = [f"address case {u['address']}"] * (u["address"] != r["address"])
        bad += [f"decimals {u['decimals']}, issuer {r['decimals']}"] * (u["decimals"] != r["decimals"])
        notes = bad + [f"Uniswap's symbol {u['symbol']}"] * (u["symbol"] != r["symbol"])
        differ += bool(bad)
        print(f"tokens: {'DIFFERS' if bad else 'agrees '} {r['symbol']}:{r['chainId']} {r['address']}"
              + (f" ({'; '.join(notes)})" if notes else ""))
    print(f"tokens: Uniswap default token list {version}: {differ} of {len(rows)} curated tokens differ")
    return 1 if differ else 0


def main(argv):
    if "--cross-check" in argv or "--latest" in argv:
        if "--ctest" in argv and "--enabled" not in argv and os.environ.get("SNICHOLLS_CORE_NETWORK_TESTS") != "1":
            print("tokens: the cross-check needs the network; set SNICHOLLS_CORE_NETWORK_TESTS=1 or configure with "
                  "-DSNICHOLLS_CORE_NETWORK_TESTS=ON: skipped")
            return SKIP
        return cross_check("--latest" in argv)
    files = outputs()
    if "--check" in argv:
        stale = [p for p, text in files.items() if not p.exists() or p.read_text(encoding="utf-8") != text]
        for p in stale:
            print(f"tokens: {p.relative_to(ROOT)} is not what scripts/generate_tokens.py writes", file=sys.stderr)
        return 1 if stale else 0
    for p, text in files.items():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
