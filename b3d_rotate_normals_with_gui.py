#!/usr/bin/env python3
"""
b3d_rotate_normals_with_gui.py -- b3d_rotate_normals.py's rotate-and-save PLUS
b3d_shading_viewer.py's before/after preview, in one window with fields and
buttons instead of re-running a CLI for every candidate.

    python b3d_rotate_normals_with_gui.py model.b3d
    python b3d_rotate_normals_with_gui.py model.b3d --axis x --deg -90
    python b3d_rotate_normals_with_gui.py model.b3d --texture cars_wooden_car.png

Left panel = the file on disk right now ("BEFORE"). Right panel = the same
model with its normals rotated by the Axis/Deg fields, recomputed in memory
as you change them ("AFTER") -- nothing touches disk until you click Save.

Controls:
  Axis / Deg fields + [-90][90][180] presets -> Preview (or just press Enter
    in the Deg field, or click an axis radio button) recomputes AFTER.
  Report   -> prints the same 9-candidate geometric-fit table as
              `b3d_rotate_normals.py --report` for the currently loaded file.
  Use best -> fills Axis/Deg from the best candidate Report just found.
  Save As...    -> write BEFORE rotated by Axis/Deg to a new file you pick.
  Save In-Place -> overwrite the loaded file (confirmation dialog first).
                   BEFORE is then reloaded from disk and Deg resets to 0, so
                   you don't accidentally rotate an already-fixed file again.
  Open Model...  -> load a different .b3d without restarting.

  drag (on the image) = orbit both panels together | wheel (over the image)
  = zoom | T = toggle texture (pass --texture at startup to load one) |
  R = reset view | Esc = quit.

Saving reuses b3d_rotate_normals.rotate() byte-for-byte (same verified
in-place-editing code as the CLI tool); this window only adds the controls
around it. Requires numpy, Pillow, tkinter.
"""
import argparse
import os
import sys

import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

import b3dlib
import b3d_rotate_normals as rn
from b3d_shading_viewer import ViewState, geom_fit


class App:
    def __init__(self, root, model_path, axis, deg, texture_path):
        self.root = root
        self.model_path = model_path
        self.raw_bytes = open(model_path, "rb").read()
        self.before_mesh = b3dlib.load_render_mesh(model_path)
        if self.before_mesh[1] is None:
            raise SystemExit(f"{model_path}: no per-vertex normals to rotate")

        self.texture = None
        if texture_path:
            self.texture = np.array(Image.open(texture_path).convert("RGB"))

        self.axis_var = tk.StringVar(value=axis)
        self.deg_var = tk.StringVar(value=self._fmt_deg(deg))
        self._best = None   # (axis, deg) from the last Report, for "Use best"
        self._drag = None
        self.photo = None   # keep a reference or Tk garbage-collects the image

        after_mesh = self._rotated_mesh(axis, deg)
        self.state = ViewState(self.before_mesh, after_mesh,
                                self._before_label(), self._after_label(axis, deg),
                                self.texture)

        self._build_ui()
        self.log(f"loaded {model_path}")
        fit = geom_fit(*self._pnt(self.before_mesh))
        self.log(f"  fit vs. surface geometry: {fit:+.3f}" if fit is not None else "  fit: n/a")
        self.redraw()

    # ------------------------------------------------------------ helpers

    @staticmethod
    def _fmt_deg(d):
        return f"{d:g}"

    @staticmethod
    def _pnt(mesh):
        P, N, UV, T = mesh
        return P, N, T

    def _before_label(self):
        return "BEFORE  " + os.path.basename(self.model_path)

    def _after_label(self, axis, deg):
        return f"AFTER   axis={axis} deg={deg:g}"

    def _rotated_mesh(self, axis, deg):
        P, N, UV, T = self.before_mesh
        rotator = rn.make_rotator(axis, deg)
        Nr = np.array([rotator(tuple(n)) for n in N])
        return (P, Nr, UV, T)

    def _read_params(self):
        axis = self.axis_var.get()
        try:
            deg = float(self.deg_var.get())
        except ValueError:
            self.log(f"ERROR: '{self.deg_var.get()}' is not a valid number of degrees")
            return None
        return axis, deg

    def log(self, msg):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", msg + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    # ------------------------------------------------------------- UI

    def _build_ui(self):
        self.image_label = tk.Label(self.root, bd=0, cursor="fleur")
        self.image_label.pack()
        self.image_label.bind("<ButtonPress-1>", self.on_press)
        self.image_label.bind("<ButtonRelease-1>", self.on_release)
        self.image_label.bind("<B1-Motion>", self.on_motion)
        self.root.bind("<MouseWheel>", self.on_wheel)
        self.root.bind("<Key>", self.on_key)

        params = tk.Frame(self.root)
        params.pack(fill="x", padx=8, pady=(6, 2))
        tk.Label(params, text="Axis:").pack(side="left")
        for a in "xyz":
            tk.Radiobutton(params, text=a.upper(), variable=self.axis_var, value=a,
                           command=self.on_params_changed).pack(side="left")
        tk.Label(params, text="   Deg:").pack(side="left")
        deg_entry = tk.Entry(params, textvariable=self.deg_var, width=8)
        deg_entry.pack(side="left")
        deg_entry.bind("<Return>", lambda e: self.on_params_changed())
        for d in (-90, 90, 180):
            tk.Button(params, text=f"{d:+d}°", width=4,
                      command=lambda d=d: self.set_deg(d)).pack(side="left", padx=1)
        tk.Button(params, text="Preview", command=self.on_params_changed).pack(side="left", padx=(8, 0))

        actions = tk.Frame(self.root)
        actions.pack(fill="x", padx=8, pady=2)
        tk.Button(actions, text="Report", command=self.on_report).pack(side="left")
        tk.Button(actions, text="Use best", command=self.on_use_best).pack(side="left", padx=4)
        tk.Button(actions, text="Save As...", command=self.on_save_as).pack(side="left", padx=(16, 4))
        tk.Button(actions, text="Save In-Place", fg="#b03030",
                  command=self.on_save_inplace).pack(side="left")
        tk.Button(actions, text="Open Model...", command=self.on_open_model).pack(side="left", padx=(16, 4))

        logf = tk.Frame(self.root)
        logf.pack(fill="both", expand=True, padx=8, pady=(2, 6))
        self.log_text = tk.Text(logf, height=7, bg="#111318", fg="#cfd8dc",
                                 font=("Consolas", 9), wrap="none")
        self.log_text.pack(fill="both", expand=True, side="left")
        sb = tk.Scrollbar(logf, command=self.log_text.yview)
        sb.pack(side="right", fill="y")
        self.log_text.configure(yscrollcommand=sb.set, state="disabled")

    def redraw(self):
        img = self.state.compose()
        self.photo = ImageTk.PhotoImage(img)
        self.image_label.configure(image=self.photo)

    # -------------------------------------------------------- param/view

    def set_deg(self, d):
        self.deg_var.set(self._fmt_deg(d))
        self.on_params_changed()

    def on_params_changed(self):
        p = self._read_params()
        if p is None:
            return
        axis, deg = p
        self.state.after = self._rotated_mesh(axis, deg)
        self.state.label_after = self._after_label(axis, deg)
        self.redraw()

    def _focus_is_entry(self):
        return isinstance(self.root.focus_get(), tk.Entry)

    def on_press(self, ev):
        self._drag = (ev.x, ev.y)

    def on_release(self, ev):
        self._drag = None

    def on_motion(self, ev):
        if self._drag is None:
            return
        dx, dy = ev.x - self._drag[0], ev.y - self._drag[1]
        self._drag = (ev.x, ev.y)
        self.state.yaw += dx * 0.4
        self.state.pitch = max(-89.0, min(89.0, self.state.pitch - dy * 0.4))
        self.redraw()

    def on_wheel(self, ev):
        if ev.widget is not self.image_label:
            return
        self.state.zoom = max(0.2, min(6.0, self.state.zoom * (1.1 if ev.delta > 0 else 1 / 1.1)))
        self.redraw()

    def on_key(self, ev):
        if ev.keysym == "Escape":
            self.root.destroy()
            return
        if self._focus_is_entry():
            return   # let the Deg field take normal keystrokes
        k = (ev.char or "").lower()
        if k == "t" and self.state.texture is not None:
            self.state.show_tex = not self.state.show_tex
            self.redraw()
        elif k == "r":
            self.state.yaw, self.state.pitch, self.state.zoom = 35.0, -25.0, 1.0
            self.redraw()

    # ------------------------------------------------------------ report

    def on_report(self):
        P, N, T = self._pnt(self.before_mesh)
        base = geom_fit(P, N, T)
        self.log(f"--- report: {os.path.basename(self.model_path)} ---")
        if base is None:
            self.log("  no triangles/normals to compare")
            return
        self.log(f"  no rotation      : {base:+.3f}")
        best = (None, None, base)
        for axis in "xyz":
            for deg in (90, -90, 180):
                rotator = rn.make_rotator(axis, deg)
                Nr = np.array([rotator(tuple(n)) for n in N])
                s = geom_fit(P, Nr, T)
                self.log(f"  axis={axis} deg={deg:<4d} : {s:+.3f}")
                if s > best[2]:
                    best = (axis, deg, s)
        self._best = (best[0], best[1])
        if best[0] is None:
            self.log("  (no rotation improves on the current normals)")
        else:
            self.log(f"  best: axis={best[0]} deg={best[1]}  ({best[2]:+.3f})")

    def on_use_best(self):
        if self._best is None:
            self.on_report()
        if self._best is None or self._best[0] is None:
            self.log("  nothing to use -- run Report first")
            return
        axis, deg = self._best
        self.axis_var.set(axis)
        self.deg_var.set(self._fmt_deg(deg))
        self.on_params_changed()

    # -------------------------------------------------------------- save

    def on_save_as(self):
        p = self._read_params()
        if p is None:
            return
        axis, deg = p
        base, ext = os.path.splitext(os.path.basename(self.model_path))
        path = filedialog.asksaveasfilename(
            initialdir=os.path.dirname(os.path.abspath(self.model_path)),
            initialfile=base + ".rotated" + ext,
            defaultextension=".b3d",
            filetypes=[("B3D model", "*.b3d"), ("All files", "*.*")])
        if not path:
            return
        try:
            out, stats, fb, fa = rn.rotate(self.raw_bytes, axis, deg)
        except (ValueError, AssertionError) as e:
            self.log(f"ERROR: {e}")
            return
        with open(path, "wb") as fh:
            fh.write(out)
        self.log(f"saved {path}")
        self.log(f"  {stats['verts']} normals rotated, fit {fb:+.3f} -> {fa:+.3f}")

    def on_save_inplace(self):
        p = self._read_params()
        if p is None:
            return
        axis, deg = p
        if not messagebox.askyesno(
                "Overwrite in place?",
                f"Overwrite\n{self.model_path}\n\nwith normals rotated "
                f"axis={axis} deg={deg:g}?\n\nOnly the normal fields change "
                f"(geometry/UVs/bones/keys are untouched), but this cannot be "
                f"undone from inside this tool."):
            return
        try:
            out, stats, fb, fa = rn.rotate(self.raw_bytes, axis, deg)
        except (ValueError, AssertionError) as e:
            self.log(f"ERROR: {e}")
            return
        with open(self.model_path, "wb") as fh:
            fh.write(out)
        self.log(f"OVERWROTE {self.model_path}")
        self.log(f"  {stats['verts']} normals rotated, fit {fb:+.3f} -> {fa:+.3f}")

        # Reload from disk and drop back to identity so the two panels agree
        # and a second click doesn't re-rotate an already-fixed file.
        self.raw_bytes = out
        self.before_mesh = b3dlib.load_render_mesh(self.model_path)
        self.deg_var.set("0")
        self.state.before = self.before_mesh
        self.state.label_before = self._before_label() + "  (just saved)"
        self.on_params_changed()

    # ------------------------------------------------------------- open

    def on_open_model(self):
        path = filedialog.askopenfilename(filetypes=[("B3D model", "*.b3d"), ("All files", "*.*")])
        if not path:
            return
        try:
            mesh = b3dlib.load_render_mesh(path)
        except ValueError as e:
            self.log(f"ERROR loading {path}: {e}")
            return
        if mesh[1] is None:
            self.log(f"ERROR: {path} has no per-vertex normals")
            return
        self.model_path = path
        self.raw_bytes = open(path, "rb").read()
        self.before_mesh = mesh
        self._best = None
        self.axis_var.set("x")
        self.deg_var.set("-90")
        self.state.before = self.before_mesh
        self.state.label_before = self._before_label()
        self.on_params_changed()
        self.log(f"opened {path}")
        fit = geom_fit(*self._pnt(self.before_mesh))
        self.log(f"  fit vs. surface geometry: {fit:+.3f}" if fit is not None else "  fit: n/a")


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="the .b3d to load")
    ap.add_argument("--axis", choices=["x", "y", "z"], default="x", help="initial rotation axis")
    ap.add_argument("--deg", type=float, default=-90.0, help="initial rotation angle in degrees")
    ap.add_argument("--texture", help="optional texture PNG to modulate shading with (press T to toggle)")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    root = tk.Tk()
    root.title("b3d rotate normals (GUI) -- " + os.path.basename(args.model))
    App(root, args.model, args.axis, args.deg, args.texture)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
