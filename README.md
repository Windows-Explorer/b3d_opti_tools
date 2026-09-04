# b3d_opti_tools

Offline utilities for `.b3d` model assets. Python 3; `pip install numpy Pillow`
covers everything (the strip/rotate CLIs are standard-library only, the
decimator and viewer need numpy, the viewer also needs Pillow + tkinter).

| file | what |
|---|---|
| `b3d_strip_weights.py` | remove zero-weight bone influences — **lossless** |
| `b3d_decimate_keys.py`  | thin out redundant animation keyframes — **lossy but measured** |
| `b3d_rotate_normals.py` | rotate per-vertex normals to fix "shading looks rotated" — **lossless for 90°-multiples** |
| `b3d_shading_viewer.py` | pop up a window (or export a PNG) comparing a model's shading before/after a fix |
| `b3dlib.py`             | shared reader + forward-kinematics/skinning/render-mesh helper (imported by the others) |

The editing tools (strip/decimate/rotate) rewrite only what they need to and
copy everything else byte-for-byte, then run a verification pass before
writing. None of them touch the input unless you pass `--in-place`.
`--dry-run` reports without writing.

---

## b3d_strip_weights.py

Some export pipelines write a `(vertex, weight)` pair for *every* bone/vertex
combination — e.g. 84 influences per vertex when only one is real, the other 83
being weight `0.0`. A zero-weight term adds nothing to the skinning blend and
nothing to the per-vertex weight sum, so dropping it is an exact identity: the
mesh deforms bit-for-bit the same. This is pure file-size / load-time / memory
savings at zero visual cost.

```
python b3d_strip_weights.py model.b3d                 # -> model.stripped.b3d
python b3d_strip_weights.py model.b3d out.b3d
python b3d_strip_weights.py model.b3d --in-place
python b3d_strip_weights.py model.b3d --dry-run
```

Example (giant jellyfish): 73,248 -> 874 weight pairs, 1.12 MB -> 538 KB.

---

## b3d_decimate_keys.py

Exporters often bake a keyframe on every frame for every bone. Where motion is
smooth most of those are redundant — the engine would interpolate the same pose
anyway. This runs Ramer–Douglas–Peucker per track and drops any keyframe that
interpolation from its kept neighbours reproduces within tolerance.

* Kept keyframes keep their **exact original values** (no resampling, no phase drift).
* `ANIM`, the mesh, bone weights and node transforms are byte-identical.
* Runs full FK + rigid skinning and prints the **worst-case vertex deviation**
  over every played frame, so the cost is visible before you commit.
* **Lossy.** Always look at the result in-engine.

```
python b3d_decimate_keys.py model.b3d --world-height 2.4
python b3d_decimate_keys.py model.b3d out.b3d --pos-tol 0.03 --rot-tol 0.15
python b3d_decimate_keys.py model.b3d --ranges 1-40,45-90,95-130 --dry-run
```

`--ranges` — the frame ranges the game actually plays (e.g. the `range` values in
a mob's `animations = {...}`). Each range is decimated on its own and its two
endpoints are always kept, so per-clip playback is untouched at the seams. Frames
outside every range are kept verbatim. Omit it to treat the whole clip as one
segment (more reduction, but a `set_animation` range that starts mid-clip may
lose its exact start pose).

`--pos-tol` (model units) / `--rot-tol` (degrees) — raise for smaller files,
lower for tighter fidelity. Defaults 0.03 / 0.15 are conservative.

`--world-height` — the mob's in-world height in metres (from its `collisionbox` /
`hitbox`), only used to print a "cm-equivalent" deviation figure.

Example (giant jellyfish, `--ranges 1-40,45-90,95-130`): 11,135 -> 4,919 keys,
538 KB -> 264 KB, worst-case vertex deviation 0.09 model units (~0.5 cm on a
2.4 m mob, 0.2% of model size).

---

## b3d_rotate_normals.py

MultiCraft's mesh-entity shader (`client/shaders/object_shader/opengl_vertex.glsl`,
`directional_ambient()`) shades every vertex from its **local-space normal
only** — never transformed by the node/world matrix. +Y-ish normals render
brightest, -Y-ish darkest, ±X dimmer than ±Z. So if a model's stored normals
don't actually match its surface (a very common Blender-Z-up -> Blitz3D/
Minetest-Y-up export bug, where *positions* get rotated -90° about X but the
*normal* array is copied through unrotated), every face looks shaded as if
tilted — "the lighting looks rotated" — no matter how the entity itself is
turned in world. Fixing it means rotating the stored normals to match the
geometry; this tool does that in place, without touching
positions/UVs/colors/bones/keyframes.

```
python b3d_rotate_normals.py model.b3d --report                  # diagnose only
python b3d_rotate_normals.py model.b3d --axis x --deg -90         # -> model.rotated.b3d
python b3d_rotate_normals.py model.b3d --axis x --deg -90 --in-place
python b3d_rotate_normals.py model.b3d --axis x --deg -90 --dry-run
```

`--report` prints, for every 90°-multiple rotation about each axis, the mean
cosine similarity between the (rotated) stored normal and the true geometric
normal computed from the triangle's own vertices — 1.0 = perfect match, ~0
= unrelated, -1.0 = inverted — and names the best candidate. Run it with no
`--axis`/`--deg` first to find the fix; `cars_wooden_car.b3d` and its siblings
all diagnose to `--axis x --deg -90` (mean cos +0.31 -> +1.00), i.e. the same
export bug on every vehicle model.

90°-multiple rotations are exact (component permute + sign flip, no trig, no
float error) and reversible — rotating back reproduces the original file
byte-for-byte. Arbitrary angles use sin/cos and renormalize. Either way the
result is verified before writing: every byte outside a normal field must be
byte-identical to the input, and every rotated normal must equal the rotation
applied to the original.

Example (`cars_wooden_car.b3d`): 494 triangles, 34% had a stored normal in the
wrong hemisphere entirely (dot product < 0) before the fix; mean fit +0.308 ->
+1.000 (988/988 vertices exact) after `--axis x --deg -90`.

Use `b3d_shading_viewer.py` (below) to eyeball a candidate rotation before
committing to it.

---

## b3d_shading_viewer.py

A quick look at "did that rotation actually fix it" without launching the
game. Opens a window with two panels — **before** and **after** — rendered
with MultiCraft's own shading formula (it imports `directional_ambient()`
straight from `b3d_rotate_normals.py`, so the two tools can never disagree).
Drag to orbit both panels together, scroll to zoom.

```
python b3d_shading_viewer.py model.b3d --axis x --deg -90        # try a rotation, nothing written to disk
python b3d_shading_viewer.py before.b3d after.b3d                # compare two actual files
python b3d_shading_viewer.py model.b3d --axis x --deg -90 --texture cars_wooden_car.png
python b3d_shading_viewer.py model.b3d                           # just look at one model
```

Controls: **drag** (left button) = orbit · **wheel** = zoom · **T** = toggle
texture (needs `--texture`) · **R** = reset view · **Esc** / close window =
quit. It also prints the same before/after geometric-fit numbers as
`b3d_rotate_normals.py --report` to the console on startup.

`--export out.png` renders one static frame and exits instead of opening a
window — useful over a remote/headless session, or for batch-generating
comparison images.

This is a flat-shaded, painter's-algorithm software rasterizer (no z-buffer,
no Gouraud smoothing, nearest-neighbour texture sampling averaged per
triangle-corner to dodge UV-seam bleed) — it exists to answer one question in
seconds, not to replace looking at the real model in-game.

---

## Notes on the .b3d format

```
chunk        = tag[4 bytes] + size[int32 LE] + body[size]
BB3D body    = version[int32] + { TEXS | BRUS | NODE }
NODE body    = name[NUL-terminated] + position[3f] + scale[3f] + rotation[4f w,x,y,z]
               + { MESH | BONE | KEYS | NODE(child) | ANIM }
MESH body    = brush_id[int32] + VRTS + TRIS...
BONE body    = { vertex_id[int32] + weight[f] }...
KEYS body    = flags[int32]  (1=pos, 2=scale, 4=rot)
               + { frame[int32] + [pos 3f] + [scale 3f] + [rot 4f] }...
ANIM body    = flags[int32] + frames[int32] + fps[f]
```

`BB3D`, `NODE` and `MESH` are containers whose `size` must be recomputed when a
child shrinks; `BONE` and `KEYS` are the leaf chunks these tools edit.
