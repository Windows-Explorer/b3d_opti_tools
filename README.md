# tools

Offline utilities for `.b3d` model assets. Python 3, `pip install numpy` for the
decimator (the stripper is standard-library only).

| file | what |
|---|---|
| `b3d_strip_weights.py` | remove zero-weight bone influences — **lossless** |
| `b3d_decimate_keys.py`  | thin out redundant animation keyframes — **lossy but measured** |
| `b3dlib.py`             | shared reader + forward-kinematics/skinning helper (imported by the decimator) |

Both tools rewrite only what they need to and copy everything else byte-for-byte,
then run a verification pass before writing. Neither touches the input unless you
pass `--in-place`. `--dry-run` reports without writing.

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
