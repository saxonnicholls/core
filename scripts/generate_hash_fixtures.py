#!/usr/bin/env python3
#
#  generate_hash_fixtures.py
#  snicholls-core
#
#  Copyright 2010-2026 Saxon Herschel Nicholls
#
"""Generate the oracle fixtures for include/snicholls/core/hash/ (tests/fixtures/hash/oracle_*.txt).

Every expected value comes from an implementation other than core's: Python's hashlib (OpenSSL underneath), hmac
and zlib; PyCryptodome for Keccak-256 (hashlib has only SHA3's padding); the `blake3` package (the official Rust
implementation) for BLAKE3; crcmod for CRC-32C and CRC-64/XZ. The inputs are pseudo-random bytes from splitmix64,
which the C++ tests regenerate with snicholls::splitmix64, at lengths around every block size the hashes use and at
random lengths up to 10,000. The versions that produced the committed files are in their headers.

    ./scripts/generate_hash_fixtures.py           rewrite the fixtures (needs pycryptodome, blake3 and crcmod)
    ./scripts/generate_hash_fixtures.py --check   recompute every case this Python can and exit 1 on a difference

--check skips the cases whose oracle is not installed (and says so), so it runs on a stock Python: hashlib, hmac
and zlib cover the SHA-1/2/3, SHAKE, BLAKE2, HMAC, HKDF, PBKDF2, scrypt and CRC-32 cases.

Format: one case per line, space-separated: `name length params... hex`, '#' starts a comment. The message is
message(length); the key, salt and other inputs of each name are defined below and in tests/support/hash_fixtures.hpp.

    message(n)       n bytes: byte i is byte (i % 8) (little-endian) of splitmix64(n * 2^32 + i / 8)
    key(n)           message(n) with 0x6b65790000000000 added to the seed (n * 2^32 + i / 8)
"""
import hashlib
import hmac
import importlib
import importlib.metadata
import pathlib
import platform
import sys
import zlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "tests" / "fixtures" / "hash"
MASK = 2**64 - 1
KEY_TWEAK = 0x6B65790000000000
HKDF_INFO = b"snicholls-core hkdf"
DERIVE_CONTEXT = b"snicholls-core 2026-11-20 test vectors derive_key"


def splitmix64(x):
    x = (x + 0x9E3779B97F4A7C15) & MASK
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK
    return x ^ (x >> 31)


def stream(seed, n):
    out = bytearray()
    for i in range((n + 7) // 8):
        out += splitmix64((seed + i) & MASK).to_bytes(8, "little")
    return bytes(out[:n])


def message(n):
    return stream(n << 32, n)


def key(n):
    return stream(((n << 32) + KEY_TWEAK) & MASK, n)


def lengths():
    """Every length around the block sizes (55/56/57 for the 8-byte length field, rate-1 for Keccak's padding,
    1024 for a BLAKE3 chunk), and random lengths up to 10,000."""
    out = {0, 1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33}
    for b in (64, 72, 104, 128, 136, 144, 168, 1024):
        for k in (1, 2, 3):
            out |= {k * b - 17, k * b - 16, k * b - 9, k * b - 8, k * b - 1, k * b, k * b + 1}
    out |= {2048, 3072, 3073, 4096, 4097, 5120, 8192, 9999, 10000}
    x = 0x5EED
    while len(out) < 140:
        x = splitmix64(x)
        out.add(x % 10001)
    return sorted(n for n in out if n >= 0)


def few(ls, step):
    return ls[::step]


def try_import(name):
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


def has_hash(name):
    try:
        hashlib.new(name)
        return True
    except ValueError:
        return False


def keccak256(data):
    from Crypto.Hash import keccak

    return keccak.new(digest_bits=256, data=data).hexdigest()


def crcmod_fn(kind):
    import crcmod

    if kind == "crc32c":
        return crcmod.mkCrcFun(0x11EDC6F41, initCrc=0, rev=True, xorOut=0xFFFFFFFF)
    return crcmod.mkCrcFun(0x142F0E1EBA9EA3693, initCrc=0, rev=True, xorOut=0xFFFFFFFFFFFFFFFF)


def plain(name):
    return lambda n: [(n, hashlib.new(name, message(n)).hexdigest())]


# Each family: file name -> list of (case name, oracle module needed or None, generator(length) -> [(params..., hex)])
def families():
    ls = lengths()
    plains = {
        "sha1": "sha1",
        "sha224": "sha224",
        "sha256": "sha256",
        "sha384": "sha384",
        "sha512": "sha512",
        "sha512_224": "sha512_224",
        "sha512_256": "sha512_256",
        "sha3_224": "sha3_224",
        "sha3_256": "sha3_256",
        "sha3_384": "sha3_384",
        "sha3_512": "sha3_512",
        "ripemd160": "ripemd160",
    }

    def cases(name, need, fn, which):
        return (name, need, fn, which)

    sha = [cases(k, "hashlib:" + v, plain(v), ls) for k, v in plains.items() if not k.startswith("sha3_")]
    sha3 = [cases(k, "hashlib:" + v, plain(v), ls) for k, v in plains.items() if k.startswith("sha3_")]
    shake = [
        cases("shake128", None, lambda n: [(n, 1 + n % 300, hashlib.shake_128(message(n)).hexdigest(1 + n % 300))], ls),
        cases("shake256", None, lambda n: [(n, 1 + (n * 7) % 300, hashlib.shake_256(message(n)).hexdigest(1 + (n * 7) % 300))], ls),
    ]
    sha3.append(cases("keccak256", "Crypto", lambda n: [(n, keccak256(message(n)))], ls))

    def b3(n):
        import blake3

        return [(n, blake3.blake3(message(n)).hexdigest())]

    def b3_xof(n):
        import blake3

        return [(n, 1 + n % 200, blake3.blake3(message(n)).hexdigest(length=1 + n % 200))]

    def b3_keyed(n):
        import blake3

        return [(n, blake3.blake3(message(n), key=key(32)).hexdigest())]

    def b3_derive(n):
        import blake3

        return [(n, blake3.blake3(message(n), derive_key_context=DERIVE_CONTEXT.decode()).hexdigest(length=48))]

    blake2 = [
        cases("blake2b", None, lambda n: [(n, hashlib.blake2b(message(n)).hexdigest())], ls),
        cases("blake2s", None, lambda n: [(n, hashlib.blake2s(message(n)).hexdigest())], ls),
        cases("blake2b_out", None, lambda n: [(n, 1 + n % 64, hashlib.blake2b(message(n), digest_size=1 + n % 64).hexdigest())], ls),
        cases("blake2s_out", None, lambda n: [(n, 1 + n % 32, hashlib.blake2s(message(n), digest_size=1 + n % 32).hexdigest())], ls),
        cases("blake2b_key", None, lambda n: [(n, n % 65, hashlib.blake2b(message(n), key=key(n % 65)).hexdigest())], ls),
        cases("blake2s_key", None, lambda n: [(n, n % 33, hashlib.blake2s(message(n), key=key(n % 33)).hexdigest())], ls),
    ]
    blake3 = [
        cases("blake3", "blake3", b3, ls),
        cases("blake3_xof", "blake3", b3_xof, ls),
        cases("blake3_keyed", "blake3", b3_keyed, ls),
        cases("blake3_derive", "blake3", b3_derive, ls),
    ]

    crc = [
        cases("crc32", None, lambda n: [(n, "%08x" % zlib.crc32(message(n)))], ls),
        cases("crc32c", "crcmod", lambda n: [(n, "%08x" % crcmod_fn("crc32c")(message(n)))], ls),
        cases("crc64_xz", "crcmod", lambda n: [(n, "%016x" % crcmod_fn("crc64")(message(n)))], ls),
    ]

    def hm(algo):
        # key lengths cycle through empty, short, exactly the block and longer than the block (hashed first)
        def f(n):
            k = (0, 1, 20, 64, 65, 128, 129, 200)[n % 8]
            return [(n, k, hmac.new(key(k), message(n), algo).hexdigest())]

        return f

    def hkdf_sha256(n):
        length = 1 + (n * 13) % 300
        prk = hmac.new(key(16), message(n), hashlib.sha256).digest()
        okm, t, i = b"", b"", 1
        while len(okm) < length:
            t = hmac.new(prk, t + HKDF_INFO + bytes([i]), hashlib.sha256).digest()
            okm += t
            i += 1
        return [(n, length, okm[:length].hex())]

    def pbkdf2(algo):
        def f(n):
            iterations = 1 + n % 37
            return [(n, iterations, hashlib.pbkdf2_hmac(algo, message(n), key(16), iterations, 100).hex())]

        return f

    def scrypt(n):
        cost, r, p = (16, 1, 1) if n % 3 == 0 else (64, 2, 2) if n % 3 == 1 else (256, 8, 1)
        return [(n, cost, r, p, hashlib.scrypt(message(n), salt=key(16), n=cost, r=r, p=p, dklen=48).hex())]

    mac_lengths = few(ls, 4)
    kdf_lengths = few(ls, 10)
    mac = [
        cases("hmac_sha1", None, hm("sha1"), mac_lengths),
        cases("hmac_sha256", None, hm("sha256"), mac_lengths),
        cases("hmac_sha512", None, hm("sha512"), mac_lengths),
        cases("hmac_sha3_256", None, hm("sha3_256"), mac_lengths),
        cases("hkdf_sha256", None, hkdf_sha256, kdf_lengths),
        cases("pbkdf2_sha1", None, pbkdf2("sha1"), kdf_lengths),
        cases("pbkdf2_sha256", None, pbkdf2("sha256"), kdf_lengths),
        cases("pbkdf2_sha512", None, pbkdf2("sha512"), kdf_lengths),
        cases("scrypt", "hashlib:scrypt", scrypt, kdf_lengths),
    ]
    return {"oracle_sha.txt": sha, "oracle_sha3.txt": sha3, "oracle_shake.txt": shake, "oracle_blake2.txt": blake2,
            "oracle_blake3.txt": blake3, "oracle_crc.txt": crc, "oracle_mac_kdf.txt": mac}


def available(need):
    if need is None:
        return True
    if need.startswith("hashlib:"):
        what = need.split(":", 1)[1]
        return hasattr(hashlib, "scrypt") if what == "scrypt" else has_hash(what)
    return try_import(need) is not None


def versions():
    def version(dist):
        try:
            return importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            return "absent"

    import ssl

    return (f"# Python {platform.python_version()}, {ssl.OPENSSL_VERSION} (hashlib), zlib {zlib.ZLIB_VERSION}, "
            f"pycryptodome {version('pycryptodome')} (Keccak-256), blake3 {version('blake3')} (BLAKE3), "
            f"crcmod {version('crcmod')} (CRC-32C, CRC-64/XZ)")


def render(cases_):
    lines = []
    for name, _, fn, which in cases_:
        for n in which:
            for row in fn(n):
                lines.append(" ".join([name] + [str(v) for v in row]))
    return lines


def main():
    check = "--check" in sys.argv[1:]
    failed = False
    for file, cases_ in families().items():
        path = OUT / file
        if check:
            committed = {}
            for line in path.read_text().splitlines():
                if line and not line.startswith("#"):
                    parts = line.split(" ")
                    committed.setdefault(parts[0], []).append(line)
            for case in cases_:
                if not available(case[1]):
                    print(f"{file}: {case[0]}: skipped, {case[1]} is not installed")
                    continue
                if render([case]) != committed.get(case[0], []):
                    print(f"{file}: {case[0]}: differs from this Python's output")
                    failed = True
        else:
            missing = [c[1] for c in cases_ if not available(c[1])]
            if missing:
                sys.exit(f"generate_hash_fixtures.py: install {sorted(set(missing))} to write {file}")
            header = ["# GENERATED by scripts/generate_hash_fixtures.py - do not edit by hand.", versions(),
                      "# name length params... hex"]
            OUT.mkdir(parents=True, exist_ok=True)
            path.write_text("\n".join(header + render(cases_)) + "\n")
    if check:
        print("hash fixtures: " + ("DIFFER" if failed else "up to date"))
        return 1 if failed else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
