#!/usr/bin/env python3
"""
b3d_shading_viewer.py -- pop up a window comparing a model's MultiCraft
shading BEFORE / AFTER a normal fix, so you can eyeball a candidate rotation
in seconds instead of a full game launch.

Two panels side by side, same orbit. LEFT = a model as it is on disk. RIGHT
= either a second .b3d file, or the same model with its normals rotated
in-memory (nothing is written to disk -- this never touches your files).

    python b3d_shading_viewer.py model.b3d --axis x --deg -90
    python b3d_shading_viewer.py before.b3d after.b3d
    python b3d_shading_viewer.py model.b3d --axis x --deg -90 --texture model_texture.png
    python b3d_shading_viewer.py model.b3d                       # single-model view

Controls: drag (left mouse button) = orbit both panels together | mouse
wheel = zoom | T = toggle texture (needs --texture) | R = reset view |
Esc / close window = quit.

--export out.png   render one frame straight to a PNG and exit -- no window,
                    so this also works headless / over a remote session.

Requires numpy, Pillow, tkinter (all stdlib/ships with the python.org
Windows build; tkinter is skipped automatically for --export).

This is a flat-shaded, painter's-algorithm software rasterizer -- no
z-buffer, no Gouraud smoothing, crude nearest-neighbour texture sampling.
It exists to answer ONE question fast ("does this rotation make the
front/back/top/bottom brightness look right?"), using MultiCraft's own
shading formula (directional_ambient() in b3d_rotate_normals.py, mirroring
client/shaders/object_shader/opengl_vertex.glsl) on each triangle's local
normal. It is not a substitute for looking at the real model in-game.
"""
import argparse
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

import b3dlib
from b3d_rotate_normals import make_rotator, directional_ambient

# Sanity-check the vectorized formula below against the authoritative
# per-vector one in b3d_rotate_normals.py, once, at import time.
for _v in ((0, 1, 0), (0, -1, 0), (1, 0, 0), (0, 0, 1), (0.6, -0.6, 0.529)):
    _l = math.sqrt(sum(c * c for c in _v))
    _u = tuple(c / _l for c in _v)
    assert abs(directional_ambient(_u) - (
        (0.670820 * _u[0] ** 2 + (0.447213 if _u[1] < 0 else 1.0) * _u[1] ** 2 + 0.836660 * _u[2] ** 2)
    )) < 1e-9, "shading formula drifted from b3d_rotate_normals.directional_ambient"


def vidiff_np(N):
    """Vectorized directional_ambient(), including the shader's zero-normal
    special case (length 0 -> fully lit, vIDiff=1.0)."""
    l = np.linalg.norm(N, axis=-1, keepdims=True)
    unit = np.divide(N, l, out=np.zeros_like(N), where=l > 1e-9)
    x2, y2, z2 = unit[..., 0] ** 2, unit[..., 1] ** 2, unit[..., 2] ** 2
    dark = 0.670820 * x2 + 0.447213 * y2 + 0.836660 * z2
    bright = 0.670820 * x2 + 1.000000 * y2 + 0.836660 * z2
    out = np.where(unit[..., 1] < 0, dark, bright)
    return np.where(l[..., 0] > 1e-9, out, 1.0)


def geom_fit(P, N, T):
    """Mean cosine similarity of stored normals to the true surface normal --
    same metric as b3d_rotate_normals.py --report, computed straight from
    already-loaded arrays (no second file parse)."""
    if N is None or len(T) == 0:
        return None
    e1 = P[T[:, 1]] - P[T[:, 0]]
    e2 = P[T[:, 2]] - P[T[:, 0]]
    g = np.cross(e1, e2)
    gv = np.zeros_like(P)
    for k in range(3):
        np.add.at(gv, T[:, k], g)
    gl = np.linalg.norm(gv, axis=1)
    m = gl > 1e-9
    if not m.any():
        return None
    gvn = gv[m] / gl[m, None]
    nl = np.linalg.norm(N[m], axis=1)
    nn = N[m] / np.where(nl[:, None] < 1e-9, 1, nl[:, None])
    return float((nn * gvn).sum(1).mean())


def sample_tex(img, uv):
    h, w, _ = img.shape
    x = np.clip((uv[..., 0] * w).astype(int), 0, w - 1)
    y = np.clip((uv[..., 1] * h).astype(int), 0, h - 1)
    return img[y, x, :3]


def rotate_pts(P, yaw, pitch):
    ry = math.radians(yaw); cy, sy = math.cos(ry), math.sin(ry)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rp = math.radians(pitch); cp, sp = math.cos(rp), math.sin(rp)
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    return P @ Ry.T @ Rx.T


def render_panel(draw, ox, oy, P, N, UV, T, yaw, pitch, scale, tex):
    """P already centered on the model's bbox midpoint. Positions are rotated
    for the on-screen view; normals are NOT (MultiCraft shades from the
    untouched local-space normal -- see directional_ambient's docstring)."""
    Pc = rotate_pts(P, yaw, pitch)
    tri_n = N[T].mean(1)
    vid = vidiff_np(tri_n)
    if tex is not None:
        corner_cols = sample_tex(tex, UV[T]) / 255.0   # sample per corner, average
        base_col = corner_cols.mean(1)                  # colors (not UVs) to dodge seam bleed
    else:
        base_col = np.ones((len(T), 3))
    colors = np.clip(base_col * vid[:, None], 0, 1)

    e1 = Pc[T[:, 1]] - Pc[T[:, 0]]
    e2 = Pc[T[:, 2]] - Pc[T[:, 0]]
    visible = np.cross(e1, e2)[:, 2] < 0            # backface cull (view-space only)
    depth = Pc[T].mean(1)[:, 2]
    order = np.argsort(depth)[::-1]                  # far first (painter's algorithm)

    sx = Pc[:, 0] * scale + ox
    sy = -Pc[:, 1] * scale + oy
    px, py = sx[T], sy[T]
    for ti in order:
        if not visible[ti]:
            continue
        c = colors[ti]
        draw.polygon(
            [(px[ti, 0], py[ti, 0]), (px[ti, 1], py[ti, 1]), (px[ti, 2], py[ti, 2])],
            fill=(int(c[0] * 255), int(c[1] * 255), int(c[2] * 255)))


class ViewState:
    def __init__(self, before, after, label_before, label_after, texture):
        self.before, self.after = before, after
        self.label_before, self.label_after = label_before, label_after
        self.texture = texture
        self.yaw, self.pitch, self.zoom = 35.0, -25.0, 1.0
        self.show_tex = False
        self.panel_w, self.panel_h = 480, 440

    def compose(self):
        W, H = self.panel_w, self.panel_h
        img = Image.new("RGB", (W * 2 + 24, H + 76), (16, 16, 18))
        draw = ImageDraw.Draw(img)
        Pb, Nb, UVb, Tb = self.before
        Pa, Na, UVa, Ta = self.after
        mn = np.minimum(Pb.min(0), Pa.min(0))
        mx = np.maximum(Pb.max(0), Pa.max(0))
        ext = np.linalg.norm(mx - mn) or 1.0
        c = (mn + mx) / 2
        scale = min(W, H) * 0.72 / ext * self.zoom
        tex = self.texture if (self.show_tex and self.texture is not None) else None

        render_panel(draw, W / 2, H / 2 + 36, Pb - c, Nb, UVb, Tb, self.yaw, self.pitch, scale, tex)
        render_panel(draw, W + 24 + W / 2, H / 2 + 36, Pa - c, Na, UVa, Ta, self.yaw, self.pitch, scale, tex)
        draw.line([(W + 12, 0), (W + 12, H + 36)], fill=(50, 50, 54))

        draw.text((10, 8), self.label_before, fill=(255, 210, 80))
        draw.text((W + 34, 8), self.label_after, fill=(120, 255, 140))
        tex_state = "on" if self.show_tex else ("n/a" if self.texture is None else "off")
        draw.text((10, H + 54),
                  f"drag = orbit   wheel = zoom   T = texture ({tex_state})   R = reset   Esc = quit",
                  fill=(150, 150, 150))
        return img


def load_after(args, before):
    Pb, Nb, UVb, Tb = before
    if args.after:
        Pa, Na, UVa, Ta = b3dlib.load_render_mesh(args.after)
        if Na is None:
            sys.exit(f"ERROR: {args.after} has no per-vertex normals")
        return (Pa, Na, UVa, Ta), os.path.basename(args.after)
    if args.axis:
        rotator = make_rotator(args.axis, args.deg)
        Na = np.array([rotator(tuple(n)) for n in Nb])
        return (Pb, Na, UVb, Tb), f"AFTER   --axis {args.axis} --deg {args.deg:g}"
    return (Pb, Nb, UVb, Tb), "(same model)"


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("before", help="the .b3d to view (or the 'before' side of a comparison)")
    ap.add_argument("after", nargs="?", help="a second .b3d to compare against 'before'")
    ap.add_argument("--axis", choices=["x", "y", "z"],
                     help="instead of a second file, rotate 'before'-s normals in memory")
    ap.add_argument("--deg", type=float, help="rotation angle in degrees (with --axis)")
    ap.add_argument("--texture", help="texture PNG to modulate shading with (press T to toggle)")
    ap.add_argument("--export", metavar="PNG",
                     help="render one static frame to this PNG and exit -- no window")
    ap.add_argument("--yaw", type=float, default=35.0, help="initial/export camera yaw, degrees")
    ap.add_argument("--pitch", type=float, default=-25.0, help="initial/export camera pitch, degrees")
    args = ap.parse_args(argv)
    if args.after and args.axis:
        ap.error("pass either a second file or --axis/--deg, not both")
    if args.axis and args.deg is None:
        ap.error("--axis requires --deg")
    return args


def main(argv=None):
    args = parse_args(argv)
    before = b3dlib.load_render_mesh(args.before)
    if before[1] is None:
        sys.exit(f"ERROR: {args.before} has no per-vertex normals")
    after, after_label = load_after(args, before)
    before_label = "BEFORE  " + os.path.basename(args.before) if (args.after or args.axis) \
        else os.path.basename(args.before)

    tex = None
    if args.texture:
        tex = np.array(Image.open(args.texture).convert("RGB"))

    fb = geom_fit(before[0], before[1], before[3])
    fa = geom_fit(after[0], after[1], after[3])
    print(f"{args.before}")
    print(f"  BEFORE fit vs. surface geometry: {fb:+.3f}" if fb is not None else "  BEFORE: n/a")
    print(f"  AFTER  fit vs. surface geometry: {fa:+.3f}" if fa is not None else "  AFTER: n/a")
    print("  (1.0 = normal exactly matches the surface it's on, 0 = unrelated, -1.0 = inverted)")

    state = ViewState(before, after, before_label, after_label, tex)
    state.yaw, state.pitch = args.yaw, args.pitch

    if args.export:
        state.compose().save(args.export)
        print("  exported:", args.export)
        return 0

    import tkinter as tk
    from PIL import ImageTk

    root = tk.Tk()
    root.title("b3d shading viewer -- " + os.path.basename(args.before))
    label = tk.Label(root, bd=0)
    label.pack()

    photo_ref = {}

    def redraw():
        img = state.compose()
        photo_ref["img"] = ImageTk.PhotoImage(img)   # keep a reference or Tk garbage-collects it
        label.configure(image=photo_ref["img"])

    drag = {}

    def on_press(ev):
        drag["xy"] = (ev.x, ev.y)

    def on_release(ev):
        drag.pop("xy", None)

    def on_motion(ev):
        if "xy" not in drag:
            return
        dx, dy = ev.x - drag["xy"][0], ev.y - drag["xy"][1]
        drag["xy"] = (ev.x, ev.y)
        state.yaw += dx * 0.4
        state.pitch = max(-89.0, min(89.0, state.pitch - dy * 0.4))
        redraw()

    def on_wheel(ev):
        factor = 1.1 if ev.delta > 0 else (1 / 1.1)
        state.zoom = max(0.2, min(6.0, state.zoom * factor))
        redraw()

    def on_key(ev):
        k = (ev.char or "").lower()
        if k == "t" and state.texture is not None:
            state.show_tex = not state.show_tex
            redraw()
        elif k == "r":
            state.yaw, state.pitch, state.zoom = args.yaw, args.pitch, 1.0
            redraw()
        elif ev.keysym == "Escape":
            root.destroy()

    label.bind("<ButtonPress-1>", on_press)
    label.bind("<ButtonRelease-1>", on_release)
    label.bind("<B1-Motion>", on_motion)
    root.bind("<MouseWheel>", on_wheel)
    root.bind("<Key>", on_key)

    redraw()
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
