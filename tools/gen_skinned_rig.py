#!/usr/bin/env python3
"""Generate a minimal self-contained rigged + animated glTF for the skeletal
animation import path. Three rings of four vertices form a thin tube skinned to
two joints (smooth blend at the middle ring); one animation rotates the upper
joint about Z. All buffer data is embedded as a base64 data URI so the asset is a
single .gltf with no external .bin. Run from the repo root:

    python3 tools/gen_skinned_rig.py

Output: assets/models/skinned_rig.gltf
"""

import base64
import json
import math
import struct
from pathlib import Path

HALF_WIDTH = 0.25
RING_Y = [0.0, 1.0, 2.0]  # three rings stacked along +Y


def ring_corners(y):
    return [
        (-HALF_WIDTH, y, -HALF_WIDTH),
        (HALF_WIDTH, y, -HALF_WIDTH),
        (HALF_WIDTH, y, HALF_WIDTH),
        (-HALF_WIDTH, y, HALF_WIDTH),
    ]


def build_geometry():
    positions, normals, joints, weights = [], [], [], []
    # Per-ring joint weighting: ring 0 -> joint 0, ring 1 -> 50/50, ring 2 -> joint 1.
    ring_weights = [
        ((0, 0, 0, 0), (1.0, 0.0, 0.0, 0.0)),
        ((0, 1, 0, 0), (0.5, 0.5, 0.0, 0.0)),
        ((1, 0, 0, 0), (1.0, 0.0, 0.0, 0.0)),
    ]
    for ring_index, y in enumerate(RING_Y):
        for (x, vy, z) in ring_corners(y):
            positions.append((x, vy, z))
            length = math.hypot(x, z) or 1.0
            normals.append((x / length, 0.0, z / length))
            j, w = ring_weights[ring_index]
            joints.append(j)
            weights.append(w)

    indices = []
    # Side quads between consecutive rings (CCW outward).
    for ring in range(len(RING_Y) - 1):
        base = ring * 4
        top = base + 4
        for c in range(4):
            n = (c + 1) % 4
            a0, a1 = base + c, base + n
            b0, b1 = top + c, top + n
            indices += [a0, b0, b1, a0, b1, a1]
    # End caps.
    indices += [0, 2, 1, 0, 3, 2]            # bottom ring
    t = (len(RING_Y) - 1) * 4
    indices += [t + 0, t + 1, t + 2, t + 0, t + 2, t + 3]  # top ring
    return positions, normals, joints, weights, indices


def quat_z(angle_deg):
    half = math.radians(angle_deg) * 0.5
    return (0.0, 0.0, math.sin(half), math.cos(half))  # (x, y, z, w)


def main():
    positions, normals, joints, weights, indices = build_geometry()
    vertex_count = len(positions)

    # Two joints: root at origin, child translated +1 on Y. Inverse-bind is the
    # inverse of each joint's bind-space global transform.
    inverse_bind = [
        # joint 0: inverse(identity)
        (1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1),
        # joint 1: inverse(translate(0,1,0)) = translate(0,-1,0), column-major
        (1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, -1, 0, 1),
    ]

    times = [0.0, 0.5, 1.0, 1.5, 2.0]
    rotations = [quat_z(a) for a in (0.0, 35.0, 0.0, -35.0, 0.0)]

    # Pack everything into one buffer, tracking (offset, length) per section.
    blob = bytearray()
    views = []

    def add(data_bytes, target=None):
        offset = len(blob)
        blob.extend(data_bytes)
        # 4-byte align the next section.
        while len(blob) % 4 != 0:
            blob.append(0)
        views.append({"offset": offset, "length": len(data_bytes), "target": target})
        return len(views) - 1

    ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER = 34962, 34963
    pos_view = add(b"".join(struct.pack("<3f", *p) for p in positions), ARRAY_BUFFER)
    nrm_view = add(b"".join(struct.pack("<3f", *n) for n in normals), ARRAY_BUFFER)
    jnt_view = add(b"".join(struct.pack("<4B", *j) for j in joints), ARRAY_BUFFER)
    wgt_view = add(b"".join(struct.pack("<4f", *w) for w in weights), ARRAY_BUFFER)
    idx_view = add(b"".join(struct.pack("<H", i) for i in indices), ELEMENT_ARRAY_BUFFER)
    ibm_view = add(b"".join(struct.pack("<16f", *m) for m in inverse_bind))
    time_view = add(b"".join(struct.pack("<f", t) for t in times))
    rot_view = add(b"".join(struct.pack("<4f", *r) for r in rotations))

    def fmin_max(values, dim):
        cols = list(zip(*values)) if dim > 1 else [values]
        mn = [min(c) for c in cols]
        mx = [max(c) for c in cols]
        return mn, mx

    pos_min, pos_max = fmin_max(positions, 3)

    FLOAT, UBYTE, USHORT = 5126, 5121, 5123
    accessors = [
        {"bufferView": pos_view, "componentType": FLOAT, "count": vertex_count, "type": "VEC3",
         "min": list(pos_min), "max": list(pos_max)},
        {"bufferView": nrm_view, "componentType": FLOAT, "count": vertex_count, "type": "VEC3"},
        {"bufferView": jnt_view, "componentType": UBYTE, "count": vertex_count, "type": "VEC4"},
        {"bufferView": wgt_view, "componentType": FLOAT, "count": vertex_count, "type": "VEC4"},
        {"bufferView": idx_view, "componentType": USHORT, "count": len(indices), "type": "SCALAR"},
        {"bufferView": ibm_view, "componentType": FLOAT, "count": 2, "type": "MAT4"},
        {"bufferView": time_view, "componentType": FLOAT, "count": len(times), "type": "SCALAR",
         "min": [min(times)], "max": [max(times)]},
        {"bufferView": rot_view, "componentType": FLOAT, "count": len(rotations), "type": "VEC4"},
    ]

    buffer_views = [
        {"buffer": 0, "byteOffset": v["offset"], "byteLength": v["length"],
         **({"target": v["target"]} if v["target"] else {})}
        for v in views
    ]

    gltf = {
        "asset": {"version": "2.0", "generator": "gen_skinned_rig.py"},
        "scene": 0,
        "scenes": [{"nodes": [0, 1]}],
        "nodes": [
            {"name": "SkinnedTube", "mesh": 0, "skin": 0},
            {"name": "Joint0", "children": [2], "translation": [0.0, 0.0, 0.0]},
            {"name": "Joint1", "translation": [0.0, 1.0, 0.0]},
        ],
        "meshes": [{
            "name": "SkinnedTube",
            "primitives": [{
                "attributes": {"POSITION": 0, "NORMAL": 1, "JOINTS_0": 2, "WEIGHTS_0": 3},
                "indices": 4,
                "mode": 4,
            }],
        }],
        "skins": [{"joints": [1, 2], "skeleton": 1, "inverseBindMatrices": 5}],
        "animations": [{
            "name": "Wave",
            "samplers": [{"input": 6, "output": 7, "interpolation": "LINEAR"}],
            "channels": [{"sampler": 0, "target": {"node": 2, "path": "rotation"}}],
        }],
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{
            "byteLength": len(blob),
            "uri": "data:application/octet-stream;base64," + base64.b64encode(bytes(blob)).decode("ascii"),
        }],
    }

    out = Path(__file__).resolve().parent.parent / "assets" / "models" / "skinned_rig.gltf"
    out.write_text(json.dumps(gltf, indent=1))
    print(f"wrote {out} ({len(blob)} bytes of buffer data, {vertex_count} verts, {len(indices)} indices)")


if __name__ == "__main__":
    main()
