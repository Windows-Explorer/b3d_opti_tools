"""
b3dlib.py -- minimal Blitz3D (.b3d) reader + forward kinematics / skinning.

Shared helper for the b3d_* tools in this directory. Only the chunks needed for
animation work are decoded: node hierarchy + rest transforms, KEYS tracks, the
first VRTS/TRIS mesh, and BONE vertex weights.

B3D chunk = 4-byte tag + int32 little-endian size + body.
KEYS body  = int32 flags (1=position, 2=scale, 4=rotation), then per key:
             int32 frame, [3f position], [3f scale], [4f quaternion as w,x,y,z].

Requires numpy.
"""
import struct
import math
import numpy as np


class Node:
    __slots__ = ("name", "parent", "rest_pos", "rest_scale", "rest_quat",
                 "keys", "keys_flags", "keys_span")

    def __init__(self):
        self.keys = None          # list[(frame:int, pos:np(3), scale:np(3), quat_wxyz:np(4))] or None
        self.keys_flags = 0
        self.keys_span = None     # (body_start, body_end) byte range of the KEYS chunk, for rewriting


def parse(path):
    """Return (raw_bytes, nodes, verts Nx3, bone_pairs {node_name: [(vid, weight)]}, tris Mx3)."""
    data = open(path, "rb").read()
    if data[:4] != b"BB3D":
        raise ValueError("not a B3D file: " + path)
    (rsz,) = struct.unpack_from("<i", data, 4)
    if 8 + rsz != len(data):
        raise ValueError("truncated or padded B3D (root size mismatch)")

    r = {"nodes": [], "verts": None, "tris": [], "bones": {}}

    def walk(p, limit, parent_idx):
        while p + 8 <= limit:
            tag = data[p:p + 4].decode("latin1")
            (sz,) = struct.unpack_from("<i", data, p + 4)
            b0, b1 = p + 8, p + 8 + sz
            if tag == "NODE":
                s = b0
                while data[s] != 0:
                    s += 1
                name = data[b0:s].decode("latin1")
                v = struct.unpack_from("<10f", data, s + 1)
                nd = Node()
                nd.name = name
                nd.parent = parent_idx
                nd.rest_pos = np.array(v[0:3])
                nd.rest_scale = np.array(v[3:6])
                nd.rest_quat = np.array(v[6:10])
                idx = len(r["nodes"])
                r["nodes"].append(nd)
                walk(s + 1 + 40, b1, idx)
            elif tag == "BONE":
                r["bones"][r["nodes"][parent_idx].name] = [
                    struct.unpack_from("<if", data, b0 + i * 8) for i in range(sz // 8)]
                walk(b0 + (sz // 8) * 8, b1, parent_idx)
            elif tag == "KEYS":
                (kf,) = struct.unpack_from("<i", data, b0)
                per = (3 if kf & 1 else 0) + (3 if kf & 2 else 0) + (4 if kf & 4 else 0)
                rec = 4 + per * 4
                keys = []
                for i in range((sz - 4) // rec):
                    off = b0 + 4 + i * rec
                    fr = struct.unpack_from("<i", data, off)[0]
                    vv = struct.unpack_from("<%df" % per, data, off + 4)
                    j = 0
                    pos = np.array(vv[j:j + 3]) if kf & 1 else np.zeros(3)
                    j += 3 if kf & 1 else 0
                    scl = np.array(vv[j:j + 3]) if kf & 2 else np.ones(3)
                    j += 3 if kf & 2 else 0
                    qt = np.array(vv[j:j + 4]) if kf & 4 else np.array([1.0, 0, 0, 0])
                    keys.append((fr, pos, scl, qt))
                nd = r["nodes"][parent_idx]
                nd.keys, nd.keys_flags, nd.keys_span = keys, kf, (b0, b1)
            elif tag == "MESH":
                walk(b0 + 4, b1, parent_idx)
            elif tag == "VRTS":
                flags, tcs, tcz = struct.unpack_from("<3i", data, b0)
                stride = 3 + (3 if flags & 1 else 0) + (4 if flags & 2 else 0) + tcs * tcz
                nv = (sz - 12) // (stride * 4)
                arr = np.empty((nv, 3))
                for i in range(nv):
                    arr[i] = struct.unpack_from("<3f", data, b0 + 12 + i * stride * 4)
                if r["verts"] is None:
                    r["verts"] = arr
            elif tag == "TRIS":
                nt = (sz - 4) // 12
                ta = np.empty((nt, 3), dtype=int)
                for i in range(nt):
                    ta[i] = struct.unpack_from("<3i", data, b0 + 4 + i * 12)
                r["tris"].append(ta)
            p = b1

    walk(12, len(data), -1)
    tris = np.concatenate(r["tris"]) if r["tris"] else np.zeros((0, 3), int)
    return data, r["nodes"], r["verts"], r["bones"], tris


# ---------------------------------------------------------------- math / FK

def _qnorm(q):
    n = np.linalg.norm(q)
    return q / n if n > 1e-12 else np.array([1.0, 0, 0, 0])


def quat_to_R(q):
    w, x, y, z = _qnorm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def mat4(pos, scale, quat):
    M = np.eye(4)
    M[:3, :3] = quat_to_R(quat) * scale
    M[:3, 3] = pos
    return M


def slerp(q0, q1, t):
    q0, q1 = _qnorm(q0), _qnorm(q1)
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        return _qnorm(q0 + t * (q1 - q0))
    th = math.acos(max(-1.0, min(1.0, d)))
    return (q0 * math.sin((1 - t) * th) + q1 * math.sin(t * th)) / math.sin(th)


def quat_angle_deg(q0, q1):
    d = min(1.0, abs(float(np.dot(_qnorm(q0), _qnorm(q1)))))
    return math.degrees(2 * math.acos(d))


def local_at(nd, f):
    """Local transform matrix of a node at (fractional) frame f."""
    ks = nd.keys
    if not ks:
        return mat4(nd.rest_pos, nd.rest_scale, nd.rest_quat)
    if f <= ks[0][0]:
        _, p, s, q = ks[0]
        return mat4(p, s, q)
    if f >= ks[-1][0]:
        _, p, s, q = ks[-1]
        return mat4(p, s, q)
    lo, hi = 0, len(ks) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if ks[mid][0] <= f:
            lo = mid
        else:
            hi = mid
    f0, p0, s0, q0 = ks[lo]
    f1, p1, s1, q1 = ks[hi]
    t = (f - f0) / (f1 - f0)
    return mat4(p0 + t * (p1 - p0), s0 + t * (s1 - s0), slerp(q0, q1, t))


def world_at(nodes, f):
    W = [None] * len(nodes)
    for i, nd in enumerate(nodes):
        L = local_at(nd, f)
        W[i] = L if nd.parent < 0 else W[nd.parent] @ L
    return W


def bind_world(nodes):
    W = [None] * len(nodes)
    for i, nd in enumerate(nodes):
        L = mat4(nd.rest_pos, nd.rest_scale, nd.rest_quat)
        W[i] = L if nd.parent < 0 else W[nd.parent] @ L
    return W


def vertex_owner(nodes, bone_pairs, eps=1e-5):
    """vid -> node index of its highest-weight bone (ignores blend for metric purposes)."""
    name2idx = {nd.name: i for i, nd in enumerate(nodes)}
    best = {}
    for bn, pairs in bone_pairs.items():
        bi = name2idx[bn]
        for vid, w in pairs:
            if abs(w) > eps and (vid not in best or w > best[vid][1]):
                best[vid] = (bi, w)
    return {v: bi for v, (bi, _) in best.items()}


def skinned_vertices(nodes, verts, owner, f, BWi):
    """Vertex positions at frame f (rigid: each vertex follows its owner bone)."""
    W = world_at(nodes, f)
    out = verts.copy()
    by_bone = {}
    for vid, bi in owner.items():
        by_bone.setdefault(bi, []).append(vid)
    for bi, vids in by_bone.items():
        S = W[bi] @ BWi[bi]
        vh = np.c_[verts[vids], np.ones(len(vids))]
        out[vids] = (vh @ S.T)[:, :3]
    return out


# ------------------------------------------------------- full render mesh

def load_render_mesh(path):
    """Read every MESH in a .b3d as flat numpy arrays for rendering/inspection:
    positions Nx3, normals Nx3 (or None if the file has none), first-channel
    UVs Nx2 (zero-filled where absent), triangle indices Mx3 (re-based to be
    global across all meshes concatenated). Local space, no node transforms
    applied -- matches what MultiCraft's shader shades from (see
    b3d_rotate_normals.directional_ambient). Unlike parse()/walk() above this
    keeps every mesh, not just the first, and reads UVs; used by
    b3d_shading_viewer.py.
    """
    data = open(path, "rb").read()
    if data[:4] != b"BB3D":
        raise ValueError("not a B3D file: " + path)
    Ps, Ns, UVs, Ts = [], [], [], []
    base = [0]
    any_normals = [False]

    def node_prefix_len(body):
        s = body
        while data[s] != 0:
            s += 1
        return (s + 1 - body) + 40

    def walk(p, limit):
        while p + 8 <= limit:
            tag = data[p:p + 4]
            (sz,) = struct.unpack_from("<i", data, p + 4)
            b0, b1 = p + 8, p + 8 + sz
            if tag == b"NODE":
                walk(b0 + node_prefix_len(b0), b1)
            elif tag in (b"BB3D", b"MESH"):
                walk(b0 + 4, b1)
            elif tag == b"VRTS":
                flags, tcs, tcz = struct.unpack_from("<3i", data, b0)
                has_n = bool(flags & 1)
                stride = 3 + (3 if has_n else 0) + (4 if flags & 2 else 0) + tcs * tcz
                nv = (sz - 12) // stride // 4
                P = np.empty((nv, 3)); N = np.zeros((nv, 3)); UV = np.zeros((nv, 2))
                uv_off = 3 + (3 if has_n else 0) + (4 if flags & 2 else 0)
                for i in range(nv):
                    o = b0 + 12 + i * stride * 4
                    P[i] = struct.unpack_from("<3f", data, o)
                    if has_n:
                        N[i] = struct.unpack_from("<3f", data, o + 12)
                    if tcs * tcz >= 2:
                        UV[i] = struct.unpack_from("<2f", data, o + uv_off * 4)
                Ps.append(P); Ns.append(N if has_n else None); UVs.append(UV)
                any_normals[0] = any_normals[0] or has_n
                base.append(base[-1] + nv)
            elif tag == b"TRIS":
                nt = (sz - 4) // 12
                T = np.empty((nt, 3), dtype=int)
                for i in range(nt):
                    T[i] = struct.unpack_from("<3i", data, b0 + 4 + i * 12)
                Ts.append(T + base[-2] if len(base) >= 2 else T)
            p = b1

    walk(12, len(data))
    if not Ps:
        raise ValueError("no VRTS mesh found in " + path)
    P = np.concatenate(Ps)
    UV = np.concatenate(UVs)
    N = np.concatenate([n if n is not None else np.zeros((len(Ps[i]), 3))
                         for i, n in enumerate(Ns)]) if any_normals[0] else None
    T = np.concatenate(Ts) if Ts else np.zeros((0, 3), dtype=int)
    return P, N, UV, T
