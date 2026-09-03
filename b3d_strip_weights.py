#!/usr/bin/env python3
"""
b3d_strip_weights.py -- drop zero-weight bone influences from a .b3d model.

Blitz3D BONE chunks store a (vertex_index, weight) pair for every vertex a bone
touches. Some export pipelines emit a pair for *every* bone/vertex combination,
almost all with weight 0.0 (e.g. 84 influences/vertex when only 1 is real). A
skinning term with weight 0 contributes nothing to the weighted vertex blend and
nothing to the per-vertex weight sum, so removing those pairs is an exact
identity -- the mesh deforms bit-for-bit the same in every animation frame.

Only BONE chunk bodies shrink; the size fields of the enclosing BB3D / NODE /
MESH chunks are recomputed. VRTS, TRIS, KEYS, ANIM, TEXS, BRUS and every node
name/transform are copied byte-for-byte, and the result is verified before it is
written.

Pure standard library (no numpy needed).

Usage:
    python b3d_strip_weights.py model.b3d               # -> model.stripped.b3d
    python b3d_strip_weights.py model.b3d out.b3d
    python b3d_strip_weights.py model.b3d --in-place
    python b3d_strip_weights.py model.b3d --dry-run
    python b3d_strip_weights.py model.b3d --epsilon 1e-4
"""
import argparse
import os
import struct
import sys

CONTAINER_PREFIX = {b"BB3D": 4, b"MESH": 4}   # fixed bytes before sub-chunks


def _node_prefix_len(buf, body):
    s = body
    while buf[s] != 0:
        s += 1
    return (s + 1 - body) + 40   # NUL-terminated name + position(3) + scale(3) + rotation(4) floats


def _rewrite(buf, p, eps, stats):
    tag = buf[p:p + 4]
    (size,) = struct.unpack_from("<i", buf, p + 4)
    b0, b1 = p + 8, p + 8 + size

    if tag == b"BONE":
        body = bytearray()
        for i in range(size // 8):
            off = b0 + i * 8
            (w,) = struct.unpack_from("<f", buf, off + 4)
            if abs(w) > eps:
                body += buf[off:off + 8]
            else:
                stats["dropped"] += 1
        stats["kept"] += len(body) // 8
        new_body = bytes(body)
    elif tag in CONTAINER_PREFIX or tag == b"NODE":
        prefix = CONTAINER_PREFIX.get(tag) or _node_prefix_len(buf, b0)
        body = bytearray(buf[b0:b0 + prefix])
        q = b0 + prefix
        while q + 8 <= b1:
            (csz,) = struct.unpack_from("<i", buf, q + 4)
            body += _rewrite(buf, q, eps, stats)
            q += 8 + csz
        if q != b1:
            raise ValueError(f"{tag!r}: sub-chunk walk ended at {q}, expected {b1}")
        new_body = bytes(body)
    else:
        new_body = buf[b0:b1]

    return tag + struct.pack("<i", len(new_body)) + new_body


def _collect(buf):
    """Depth-first snapshot of the chunks that must not change, for verification."""
    frozen = {}          # tag -> [raw bodies in document order]
    bones = {}           # bone path -> [(vid, w)]

    def walk(p, limit, ctx):
        while p + 8 <= limit:
            tag = buf[p:p + 4]
            (sz,) = struct.unpack_from("<i", buf, p + 4)
            b0, b1 = p + 8, p + 8 + sz
            if tag == b"NODE":
                s = b0
                while buf[s] != 0:
                    s += 1
                name = buf[b0:s].decode("latin1")
                frozen.setdefault(b"NODE", []).append(bytes(buf[b0:s + 1 + 40]))
                walk(s + 1 + 40, b1, name)
            elif tag == b"BONE":
                bones.setdefault(ctx, []).extend(
                    struct.unpack_from("<if", buf, b0 + i * 8) for i in range(sz // 8))
            elif tag == b"MESH":
                frozen.setdefault(b"MESH", []).append(bytes(buf[b0:b0 + 4]))
                walk(b0 + 4, b1, ctx)
            elif tag in (b"VRTS", b"TRIS", b"KEYS", b"ANIM", b"TEXS", b"BRUS"):
                frozen.setdefault(tag, []).append(bytes(buf[b0:b1]))
            p = b1

    walk(12, len(buf), None)
    return frozen, bones


def strip(data, eps):
    if data[:4] != b"BB3D":
        raise ValueError("not a B3D file")
    (rsz,) = struct.unpack_from("<i", data, 4)
    if 8 + rsz != len(data):
        raise ValueError("truncated or padded B3D (root size mismatch)")
    stats = {"dropped": 0, "kept": 0}
    out = _rewrite(data, 0, eps, stats)
    _verify(data, out, eps)
    return out, stats


def _verify(before, after, eps):
    fb, bb = _collect(before)
    fa, ba = _collect(after)
    if fb != fa:
        changed = [t for t in set(fb) | set(fa) if fb.get(t) != fa.get(t)]
        raise AssertionError("non-weight data changed: " + ", ".join(t.decode() for t in changed))
    if sorted(bb) != sorted(ba):
        raise AssertionError("set of BONE chunks changed")
    for bn in bb:
        want = sorted((v, round(w, 7)) for v, w in bb[bn] if abs(w) > eps)
        got = sorted((v, round(w, 7)) for v, w in ba[bn])
        if want != got:
            raise AssertionError(f"bone {bn!r}: non-zero weights not preserved")
        if any(abs(w) <= eps for _, w in ba[bn]):
            raise AssertionError(f"bone {bn!r}: zero weights remain")

    def sums(m):
        s = {}
        for lst in m.values():
            for v, w in lst:
                s[v] = s.get(v, 0.0) + w
        return s
    sb, sa = sums(bb), sums(ba)
    if set(sb) != set(sa) or any(abs(sb[v] - sa[v]) > 1e-4 for v in sb):
        raise AssertionError("per-vertex weight sum changed")


def _default_out(path):
    base, ext = os.path.splitext(path)
    return base + ".stripped" + ext


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("output", nargs="?")
    ap.add_argument("--in-place", action="store_true", help="overwrite the input file")
    ap.add_argument("--epsilon", type=float, default=1e-5,
                    help="drop pairs with |weight| <= this (default 1e-5)")
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = ap.parse_args(argv)

    data = open(args.input, "rb").read()
    try:
        out, stats = strip(data, args.epsilon)
    except (ValueError, AssertionError) as e:
        print("ERROR:", e, file=sys.stderr)
        return 1

    total = stats["dropped"] + stats["kept"]
    pct = 100 * stats["dropped"] / total if total else 0.0
    print(args.input)
    print(f"  weight pairs : {total:,} -> {stats['kept']:,}   (dropped {stats['dropped']:,}, -{pct:.1f}%)")
    print(f"  file size    : {len(data):,} -> {len(out):,} bytes   (-{100 * (1 - len(out) / len(data)):.1f}%)")
    print("  verified     : mesh / keyframes / materials byte-identical, "
          "every real weight kept, sums unchanged")

    if args.dry_run:
        print("  dry run -- nothing written")
        return 0
    dst = args.input if args.in_place else (args.output or _default_out(args.input))
    with open(dst, "wb") as fh:
        fh.write(out)
    print("  written      :", dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
