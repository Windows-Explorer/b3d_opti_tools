#!/usr/bin/env python3
"""
b3d_rotate_normals.py -- rotate the per-vertex NORMALS of a .b3d model, leaving
geometry (positions, triangles, UVs, colors, bones, keyframes, node transforms)
untouched.

Why you'd want this: MultiCraft's object shader shades mesh entities purely
from each vertex's *local-space* normal (see client/shaders/object_shader/
opengl_vertex.glsl, `directional_ambient()` / `vIDiff` -- it is never
multiplied by the node's world matrix). +Y-ish normals render brightest,
-Y-ish darkest, and horizontal normals fall in between. If a model's stored
normals don't actually match its surface geometry -- a very common exporter
bug when converting Blender's Z-up scene to Blitz3D/Minetest's Y-up, where
vertex *positions* get rotated -90 deg about X but the *normal* array is
copied through unrotated -- every face is shaded as if it were tilted, which
looks exactly like "the lighting is rotated" relative to the model.

Since rotating every normal by a fixed matrix commutes with however many
VRTS/MESH/NODE chunks the file has, this tool doesn't need to understand the
node hierarchy: it finds every VRTS chunk with the "has normals" flag set and
rotates each vertex's (nx,ny,nz) by the same matrix, in place. Normal fields
are fixed-size floats, so no chunk ever changes size -- the rest of the file
(including the root BB3D size field) is copied byte-for-byte.

Diagnosing the export bug above (or checking whether a model has it at all):
    python b3d_rotate_normals.py model.b3d --report

Fixing it (the common case: normals want -90 deg about X):
    python b3d_rotate_normals.py model.b3d --axis x --deg -90

General usage:
    python b3d_rotate_normals.py model.b3d --axis y --deg 180        # -> model.rotated.b3d
    python b3d_rotate_normals.py model.b3d out.b3d --axis x --deg -90
    python b3d_rotate_normals.py model.b3d --axis x --deg -90 --in-place
    python b3d_rotate_normals.py model.b3d --axis x --deg -90 --dry-run

Pure standard library (no numpy needed). See b3d_shading_viewer.py in this
same directory to eyeball a rotation candidate in a window before committing.
"""
import argparse
import math
import os
import struct
import sys


# --------------------------------------------------------------- rotation

def _rotate_exact(v, axis, quarter_turns):
    """Exact (no trig) rotation by a multiple of 90 deg -- permute + sign-flip only."""
    x, y, z = v
    q = quarter_turns % 4
    for _ in range(q):
        if axis == "x":
            x, y, z = x, -z, y      # +90 about X
        elif axis == "y":
            x, y, z = z, y, -x      # +90 about Y
        else:
            x, y, z = -y, x, z      # +90 about Z
    return (x, y, z)


def make_rotator(axis, deg):
    """Return f(v3)->v3 rotating a vector by `deg` about `axis` ('x'/'y'/'z')."""
    if axis not in ("x", "y", "z"):
        raise ValueError("axis must be x, y or z")
    if deg % 90 == 0:
        q = int(round(deg / 90))
        return lambda v: _rotate_exact(v, axis, q)
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    if axis == "x":
        return lambda v: (v[0], v[1] * c - v[2] * s, v[1] * s + v[2] * c)
    if axis == "y":
        return lambda v: (v[0] * c + v[2] * s, v[1], -v[0] * s + v[2] * c)
    return lambda v: (v[0] * c - v[1] * s, v[0] * s + v[1] * c, v[2])


def directional_ambient(n):
    """MultiCraft object_shader's vIDiff for a *unit* local-space normal n=(x,y,z).
    1.0 = brightest (+Y), 0.447213 = darkest (-Y), 0.670820 on +-X, 0.836660 on +-Z.
    Mirrors client/shaders/object_shader/opengl_vertex.glsl::directional_ambient()."""
    x2, y2, z2 = n[0] * n[0], n[1] * n[1], n[2] * n[2]
    if n[1] < 0.0:
        return 0.670820 * x2 + 0.447213 * y2 + 0.836660 * z2
    return 0.670820 * x2 + 1.000000 * y2 + 0.836660 * z2


# --------------------------------------------------------------- chunk walk

def _node_prefix_len(buf, body):
    s = body
    while buf[s] != 0:
        s += 1
    return (s + 1 - body) + 40   # NUL-terminated name + pos(3) + scale(3) + quat(4) floats


def _for_each_vrts(buf, p, limit, fn):
    """Call fn(normal0_offset, vertex_count, stride_floats) for every VRTS chunk
    that stores normals, found anywhere under [p, limit). normal0_offset already
    points at vertex 0's NORMAL field (past its 3 position floats) -- callers
    just do `off + i*stride*4` to reach vertex i's normal."""
    while p + 8 <= limit:
        tag = buf[p:p + 4]
        (sz,) = struct.unpack_from("<i", buf, p + 4)
        b0, b1 = p + 8, p + 8 + sz
        if tag == b"NODE":
            _for_each_vrts(buf, b0 + _node_prefix_len(buf, b0), b1, fn)
        elif tag in (b"BB3D", b"MESH"):
            _for_each_vrts(buf, b0 + 4, b1, fn)
        elif tag == b"VRTS":
            flags, tcs, tcz = struct.unpack_from("<3i", buf, b0)
            if flags & 1:
                stride = 3 + 3 + (4 if flags & 2 else 0) + tcs * tcz
                nv = (sz - 12) // stride // 4
                fn(b0 + 12 + 12, nv, stride)   # +12 VRTS header, +12 skip position
        p = b1


def _mesh_geometry(buf, p, limit):
    """Depth-first list of (positions[Nx3], normals[Nx3] or None, tris[Mx3]) --
    one entry per MESH, for the fit-quality report. Read-only, stdlib only."""
    out = []

    def walk(p, limit):
        while p + 8 <= limit:
            tag = buf[p:p + 4]
            (sz,) = struct.unpack_from("<i", buf, p + 4)
            b0, b1 = p + 8, p + 8 + sz
            if tag == b"NODE":
                walk(b0 + _node_prefix_len(buf, b0), b1)
            elif tag in (b"BB3D", b"MESH"):
                if tag == b"MESH":
                    out.append({"P": None, "N": None, "T": []})
                walk(b0 + 4, b1)
            elif tag == b"VRTS":
                flags, tcs, tcz = struct.unpack_from("<3i", buf, b0)
                has_n = bool(flags & 1)
                stride = 3 + (3 if has_n else 0) + (4 if flags & 2 else 0) + tcs * tcz
                nv = (sz - 12) // stride // 4
                P, N = [], [] if has_n else None
                for i in range(nv):
                    o = b0 + 12 + i * stride * 4
                    P.append(struct.unpack_from("<3f", buf, o))
                    if has_n:
                        N.append(struct.unpack_from("<3f", buf, o + 12))
                out[-1]["P"], out[-1]["N"] = P, N
            elif tag == b"TRIS":
                nt = (sz - 4) // 12
                out[-1]["T"].extend(struct.unpack_from("<3i", buf, b0 + 4 + i * 12) for i in range(nt))
            p = b1

    walk(p, limit)
    return out


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm(v):
    l = math.sqrt(_dot(v, v))
    return (v[0] / l, v[1] / l, v[2] / l) if l > 1e-12 else (0.0, 0.0, 0.0)


def _fit_score(meshes, rotator):
    """Mean cosine similarity between (rotated) stored normals and the
    surface's actual geometric normal (area-weighted, from positions+tris).
    1.0 = normals perfectly match the surface; ~0.3-0.4 is typical for the
    unrotated Z-up/Y-up mixup; -1.0 = normals exactly inverted."""
    num, den = 0.0, 0
    for m in meshes:
        P, N, T = m["P"], m["N"], m["T"]
        if not N or not T:
            continue
        gv = [(0.0, 0.0, 0.0)] * len(P)

        def add(i, g):
            gv[i] = (gv[i][0] + g[0], gv[i][1] + g[1], gv[i][2] + g[2])

        for a, b, c in T:
            ax, ay, az = P[a]; bx, by, bz = P[b]; cx, cy, cz = P[c]
            e1 = (bx - ax, by - ay, bz - az)
            e2 = (cx - ax, cy - ay, cz - az)
            g = (e1[1] * e2[2] - e1[2] * e2[1],
                 e1[2] * e2[0] - e1[0] * e2[2],
                 e1[0] * e2[1] - e1[1] * e2[0])
            add(a, g); add(b, g); add(c, g)
        for i in range(len(P)):
            gl = math.sqrt(_dot(gv[i], gv[i]))
            if gl < 1e-9:
                continue
            gN = (gv[i][0] / gl, gv[i][1] / gl, gv[i][2] / gl)
            n = rotator(_norm(N[i])) if rotator else _norm(N[i])
            num += _dot(n, gN)
            den += 1
    return (num / den) if den else None


# --------------------------------------------------------------- top level

def rotate(data, axis, deg):
    if data[:4] != b"BB3D":
        raise ValueError("not a B3D file")
    (rsz,) = struct.unpack_from("<i", data, 4)
    if 8 + rsz != len(data):
        raise ValueError("truncated or padded B3D (root size mismatch)")

    meshes_before = _mesh_geometry(data, 12, len(data))
    fit_before = _fit_score(meshes_before, None)

    rotator = make_rotator(axis, deg)
    out = bytearray(data)
    stats = {"chunks": 0, "verts": 0}

    def patch(off, nv, stride):
        stats["chunks"] += 1
        stats["verts"] += nv
        for i in range(nv):
            o = off + i * stride * 4
            v = struct.unpack_from("<3f", out, o)
            struct.pack_into("<3f", out, o, *rotator(v))

    _for_each_vrts(out, 12, len(out), patch)
    out = bytes(out)

    _verify(data, out, axis, deg)
    fit_after = _fit_score(_mesh_geometry(out, 12, len(out)), None)
    return out, stats, fit_before, fit_after


def _verify(before, after, axis, deg):
    if len(before) != len(after):
        raise AssertionError("file length changed (it never should)")
    rotator = make_rotator(axis, deg)
    # Every byte outside a normal-float triple must be identical.
    touched = bytearray(len(before))

    def mark(off, nv, stride):
        for i in range(nv):
            o = off + i * stride * 4
            for k in range(12):
                touched[o + k] = 1

    buf = bytearray(before)
    _for_each_vrts(buf, 12, len(buf), mark)
    for i in range(len(before)):
        if not touched[i] and before[i] != after[i]:
            raise AssertionError(f"byte {i} changed outside any normal field")

    # Rotated values must match re-applying the rotator to the originals,
    # and (if the input normal was unit length) stay unit length.
    def check(off, nv, stride):
        for i in range(nv):
            o = off + i * stride * 4
            v0 = struct.unpack_from("<3f", before, o)
            v1 = struct.unpack_from("<3f", after, o)
            want = rotator(v0)
            if any(abs(a - b) > 1e-5 for a, b in zip(v1, want)):
                raise AssertionError("rotated normal does not match expected value")
            l0 = math.sqrt(_dot(v0, v0))
            if l0 > 1e-6:
                l1 = math.sqrt(_dot(v1, v1))
                if abs(l0 - l1) > 1e-4:
                    raise AssertionError("normal length not preserved by rotation")

    _for_each_vrts(bytearray(before), 12, len(before), check)


def _default_out(path):
    base, ext = os.path.splitext(path)
    return base + ".rotated" + ext


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("output", nargs="?")
    ap.add_argument("--axis", choices=["x", "y", "z"], help="rotation axis")
    ap.add_argument("--deg", type=float, help="rotation angle in degrees")
    ap.add_argument("--in-place", action="store_true", help="overwrite the input file")
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    ap.add_argument("--report", action="store_true",
                     help="only print how well stored normals fit the geometry, "
                          "for several candidate fixes -- no --axis/--deg needed")
    args = ap.parse_args(argv)

    data = open(args.input, "rb").read()

    if args.report or not args.axis:
        meshes = _mesh_geometry(data, 12, len(data))
        has_normals = any(m["N"] for m in meshes)
        print(args.input)
        if not has_normals:
            print("  no per-vertex normals found (shader will use flat vIDiff=1.0)")
            return 0
        base = _fit_score(meshes, None)
        print(f"  stored normals vs. surface geometry, no rotation: mean cos = {base:+.3f}"
              f"   (1.0 = perfect, 0 = unrelated, -1.0 = inverted)")
        best = None
        for axis in "xyz":
            for deg in (90, -90, 180):
                s = _fit_score(meshes, make_rotator(axis, deg))
                if best is None or s > best[0]:
                    best = (s, axis, deg)
                print(f"  after --axis {axis} --deg {deg:<4d}: mean cos = {s:+.3f}")
        print(f"  best candidate: --axis {best[1]} --deg {best[2]}  (mean cos {best[0]:+.3f})")
        if not args.axis:
            return 0

    if args.axis and args.deg is None:
        ap.error("--axis requires --deg")

    try:
        out, stats, fit_before, fit_after = rotate(data, args.axis, args.deg)
    except (ValueError, AssertionError) as e:
        print("ERROR:", e, file=sys.stderr)
        return 1

    print(f"  rotated {stats['verts']:,} normals in {stats['chunks']} VRTS chunk(s) "
          f"by {args.deg:g} deg about {args.axis.upper()}")
    if fit_before is not None:
        print(f"  fit vs. surface geometry: {fit_before:+.3f} -> {fit_after:+.3f}"
              f"   (1.0 = normal exactly matches the surface it's on)")
    print("  verified     : positions / triangles / UVs / colors / bones / keys / "
          "node transforms byte-identical; normals re-derived and length-preserved")

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
