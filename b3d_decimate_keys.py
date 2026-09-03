#!/usr/bin/env python3
"""
b3d_decimate_keys.py -- thin out animation keyframes in a .b3d model.

Many exporters bake a keyframe on every single frame for every bone. Where the
motion is smooth, most of those keys are redundant: the engine would reconstruct
them by interpolation anyway. This tool runs Ramer-Douglas-Peucker per track and
drops any keyframe that interpolation from its kept neighbours reproduces within
a position and a rotation tolerance.

Properties:
  * Kept keyframes retain their EXACT original values (no resampling / phase drift).
  * ANIM (frame count + fps), the mesh (VRTS/TRIS), bone weights and node
    transforms are copied byte-for-byte.
  * Each --ranges segment is decimated independently and its two endpoints are
    always kept, so per-clip playback is unchanged at the range boundaries.
  * The worst-case vertex deviation over every played frame is measured (full
    forward-kinematics + rigid skinning) and printed, so you can see the cost.

This is LOSSY (unlike b3d_strip_weights.py). Always eyeball the result in-engine.

Needs numpy and b3dlib.py (same directory).

Usage:
    python b3d_decimate_keys.py model.b3d
    python b3d_decimate_keys.py model.b3d out.b3d --pos-tol 0.03 --rot-tol 0.15
    python b3d_decimate_keys.py model.b3d --ranges 1-40,45-90,95-130
    python b3d_decimate_keys.py model.b3d --dry-run
    python b3d_decimate_keys.py model.b3d --in-place --world-height 2.4

--pos-tol       max position error, model units          (default 0.03)
--rot-tol       max rotation error, degrees              (default 0.15)
--ranges        played frame ranges "a-b,c-d,..."        (default: whole clip, one segment)
--world-height  in-world height of the mesh, metres, for the cm-equivalent readout
"""
import argparse
import math
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from b3dlib import (parse, mat4, slerp, quat_angle_deg, world_at, bind_world,
                    vertex_owner, skinned_vertices)


# ---------------------------------------------------------------- decimation

def _interp(k0, k1, f):
    f0, p0, s0, q0 = k0
    f1, p1, s1, q1 = k1
    t = (f - f0) / (f1 - f0)
    return p0 + t * (p1 - p0), s0 + t * (s1 - s0), slerp(q0, q1, t)


def _rdp(keys, i0, i1, keep, pos_tol, rot_tol):
    if i1 - i0 < 2:
        return
    k0, k1 = keys[i0], keys[i1]
    worst_i, worst_e = -1, 0.0
    for i in range(i0 + 1, i1):
        pi, si, qi = _interp(k0, k1, keys[i][0])
        _, p, s, q = keys[i]
        e = max(
            np.linalg.norm(p - pi) / pos_tol,
            np.max(np.abs(s - si)) / pos_tol,
            quat_angle_deg(q, qi) / rot_tol,
        )
        if e > worst_e:
            worst_e, worst_i = e, i
    if worst_e > 1.0:
        keep[worst_i] = True
        _rdp(keys, i0, worst_i, keep, pos_tol, rot_tol)
        _rdp(keys, worst_i, i1, keep, pos_tol, rot_tol)


def decimate_track(keys, protected, pos_tol, rot_tol):
    n = len(keys)
    fr2i = {k[0]: i for i, k in enumerate(keys)}
    keep = [False] * n
    keep[0] = keep[-1] = True
    for pf in protected:
        if pf in fr2i:
            keep[fr2i[pf]] = True
    anchors = [i for i, v in enumerate(keep) if v]
    for a, b in zip(anchors, anchors[1:]):
        _rdp(keys, a, b, keep, pos_tol, rot_tol)
    return [keys[i] for i in range(n) if keep[i]]


# ---------------------------------------------------------------- rewrite

def _pack_keys(flags, keys):
    out = struct.pack("<i", flags)
    for fr, p, s, q in keys:
        out += struct.pack("<i", fr)
        if flags & 1:
            out += struct.pack("<3f", *p)
        if flags & 2:
            out += struct.pack("<3f", *s)
        if flags & 4:
            out += struct.pack("<4f", *q)
    return out


CONTAINER_PREFIX = {b"BB3D": 4, b"MESH": 4}


def _node_prefix_len(buf, body):
    s = body
    while buf[s] != 0:
        s += 1
    return (s + 1 - body) + 40


def _rewrite(buf, p, new_keys_by_span):
    tag = buf[p:p + 4]
    (size,) = struct.unpack_from("<i", buf, p + 4)
    b0, b1 = p + 8, p + 8 + size
    if tag == b"KEYS" and (b0, b1) in new_keys_by_span:
        body = new_keys_by_span[(b0, b1)]
    elif tag in CONTAINER_PREFIX or tag == b"NODE":
        prefix = CONTAINER_PREFIX.get(tag) or _node_prefix_len(buf, b0)
        body = bytearray(buf[b0:b0 + prefix])
        q = b0 + prefix
        while q + 8 <= b1:
            (csz,) = struct.unpack_from("<i", buf, q + 4)
            body += _rewrite(buf, q, new_keys_by_span)
            q += 8 + csz
        if q != b1:
            raise ValueError(f"{tag!r}: sub-chunk walk ended at {q}, expected {b1}")
        body = bytes(body)
    else:
        body = buf[b0:b1]
    return tag + struct.pack("<i", len(body)) + body


# ---------------------------------------------------------------- driver

def parse_ranges(spec, lo, hi):
    if not spec:
        return [(lo, hi)]
    out = []
    for part in spec.split(","):
        a, b = part.split("-")
        out.append((int(a), int(b)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("output", nargs="?")
    ap.add_argument("--in-place", action="store_true")
    ap.add_argument("--pos-tol", type=float, default=0.03)
    ap.add_argument("--rot-tol", type=float, default=0.15)
    ap.add_argument("--ranges", default="")
    ap.add_argument("--world-height", type=float, default=None,
                    help="in-world mesh height (m) for a cm-equivalent readout")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    data, nodes, verts, bones, tris = parse(args.input)
    all_frames = sorted({k[0] for nd in nodes if nd.keys for k in nd.keys})
    lo, hi = all_frames[0], all_frames[-1]
    ranges = parse_ranges(args.ranges, lo, hi)
    played = sorted(set().union(*(range(a, b + 1) for a, b in ranges)))
    protected = set()
    for a, b in ranges:
        protected |= {a, b}
    protected |= (set(range(lo, hi + 1)) - set(played))   # keep un-played gap frames verbatim
    protected |= {lo, hi}

    # decimate
    orig_total = sum(len(nd.keys) for nd in nodes if nd.keys)
    new_spans = {}
    new_keys = {}   # node index -> decimated key list (for measurement)
    kept_total = 0
    for i, nd in enumerate(nodes):
        if not nd.keys:
            continue
        dk = decimate_track(nd.keys, protected, args.pos_tol, args.rot_tol)
        new_keys[i] = dk
        kept_total += len(dk)
        new_spans[nd.keys_span] = _pack_keys(nd.keys_flags, dk)

    out = _rewrite(data, 0, new_spans)

    # ---- measure vertex deviation: original keys vs decimated keys ----
    owner = vertex_owner(nodes, bones)
    BW = bind_world(nodes)
    BWi = [np.linalg.inv(m) for m in BW]

    class _N:  # lightweight node view with swapped keys
        __slots__ = ("name", "parent", "rest_pos", "rest_scale", "rest_quat",
                     "keys", "keys_flags", "keys_span")
    dec_nodes = []
    for i, nd in enumerate(nodes):
        m = _N()
        for s in _N.__slots__:
            setattr(m, s, getattr(nd, s))
        if i in new_keys:
            m.keys = new_keys[i]
        dec_nodes.append(m)

    max_dev = mean_acc = cnt = 0.0
    worst_f = None
    per_range = {}
    for a, b in ranges:
        rmax = 0.0
        for f in range(a, b + 1):
            Po = skinned_vertices(nodes, verts, owner, f, BWi)
            Pd = skinned_vertices(dec_nodes, verts, owner, f, BWi)
            d = np.linalg.norm(Po - Pd, axis=1)
            m = float(d.max())
            if m > max_dev:
                max_dev, worst_f = m, f
            rmax = max(rmax, m)
            mean_acc += float(d.sum())
            cnt += len(d)
        per_range[(a, b)] = rmax

    diag = float(np.linalg.norm(verts.max(0) - verts.min(0)))
    unit_cm = (args.world_height * 100.0 / diag) if args.world_height else None

    print(args.input)
    print(f"  ranges       : {', '.join(f'{a}-{b}' for a, b in ranges)}   "
          f"({len(played)} played frames of {hi - lo + 1})")
    print(f"  tolerances   : pos {args.pos_tol} units, rot {args.rot_tol} deg")
    print(f"  keyframes    : {orig_total:,} -> {kept_total:,}   "
          f"(-{100 * (1 - kept_total / orig_total):.1f}%)")
    print(f"  file size    : {len(data):,} -> {len(out):,} bytes   "
          f"(-{100 * (1 - len(out) / len(data)):.1f}%)")
    dev_cm = f"  ({max_dev * unit_cm:.2f} cm-equiv)" if unit_cm else ""
    print(f"  max vtx dev  : {max_dev:.4f} model units{dev_cm}   "
          f"= {100 * max_dev / diag:.3f}% of model, at frame {worst_f}")
    print(f"  mean vtx dev : {mean_acc / cnt:.4f} model units")
    for (a, b), rm in per_range.items():
        print(f"    range {a}-{b}: max {rm:.4f}")

    _verify(data, out)

    if args.dry_run:
        print("  dry run -- nothing written")
        return 0
    dst = args.input if args.in_place else (args.output or _default_out(args.input))
    with open(dst, "wb") as fh:
        fh.write(out)
    print("  written      :", dst)
    return 0


def _default_out(path):
    base, ext = os.path.splitext(path)
    return base + ".decimated" + ext


def _verify(before, after):
    """Confirm only KEYS changed and every kept key is an unmodified original."""
    def collect(buf):
        frozen, keys = {}, {}

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
                elif tag == b"KEYS":
                    (kf,) = struct.unpack_from("<i", buf, b0)
                    per = (3 if kf & 1 else 0) + (3 if kf & 2 else 0) + (4 if kf & 4 else 0)
                    rec = 4 + per * 4
                    d = {}
                    for i in range((sz - 4) // rec):
                        off = b0 + 4 + i * rec
                        fr = struct.unpack_from("<i", buf, off)[0]
                        d[fr] = bytes(buf[off + 4:off + rec])
                    keys[ctx] = (kf, d)
                elif tag == b"MESH":
                    frozen.setdefault(b"MESH", []).append(bytes(buf[b0:b0 + 4]))
                    walk(b0 + 4, b1, ctx)
                elif tag in (b"VRTS", b"TRIS", b"BONE", b"ANIM", b"TEXS", b"BRUS"):
                    frozen.setdefault(tag, []).append(bytes(buf[b0:b1]))
                p = b1
        walk(12, len(buf), None)
        return frozen, keys

    fb, kb = collect(before)
    fa, ka = collect(after)
    if fb != fa:
        bad = [t.decode() for t in set(fb) | set(fa) if fb.get(t) != fa.get(t)]
        raise AssertionError("non-keyframe data changed: " + ", ".join(bad))
    if sorted(kb) != sorted(ka):
        raise AssertionError("set of KEYS tracks changed")
    for t, (kf_b, db) in kb.items():
        kf_a, da = ka[t]
        if kf_a != kf_b:
            raise AssertionError(f"track {t!r}: KEYS flags changed")
        if not set(da).issubset(set(db)):
            raise AssertionError(f"track {t!r}: decimated frames are not a subset of the original")
        for fr, blob in da.items():
            if blob != db[fr]:
                raise AssertionError(f"track {t!r} frame {fr}: value was modified (should be verbatim)")
    print("  verified     : mesh / weights / ANIM byte-identical; kept keys are unmodified originals")


if __name__ == "__main__":
    raise SystemExit(main())
