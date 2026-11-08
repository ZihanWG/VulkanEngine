"""Convert Amazon Lumberyard Bistro (FBX + DDS) into the glTF --scene bistro loads.

Runs inside Blender, which is the only free tool that reads both the FBX and the
block-compressed DDS textures it references:

    blender -b --factory-startup --python tools/convert_bistro.py -- \\
        <Bistro_v5_2/BistroExterior.fbx> <build/fetched-assets/bistro/Bistro.gltf>

The scene is CC BY 4.0 (NVIDIA ORCA). Nothing it produces is tracked; the output
sits beside the fetched Sponza under build/fetched-assets.

What the conversion has to decide, rather than copy:

* Materials. Blender's FBX importer wires the packed "Specular" texture into the
  Principled BSDF's Specular IOR Level, which the glTF exporter cannot express. The
  scene's own README defines that texture as occlusion (R), roughness (G) and
  metalness (B) -- glTF's metallicRoughness layout exactly -- so every material is
  rebuilt from the four textures its name carries.
* Normal maps are DirectX-convention BC5, two channels. Green is flipped to the
  OpenGL convention glTF specifies and Z is reconstructed, so the PNG is a complete
  normal map for any consumer rather than only for one that rebuilds Z.
* Alpha. Foliage, curtains and lettering carry cutout alpha (MASK); glass blends.
  Everything else is single-sided -- the FBX importer marks every material
  double-sided, which would put the whole scene on the un-culled pipeline.
* Lights. The FBX holds one sun. The lamps are emissive geometry meant for a path
  tracer, so a raster renderer needs analytic lights placed at them: one per
  emissive island (merged within a radius for string lights), coloured by the
  emissive texture, with a range written into KHR_lights_punctual so the clustered
  cull stays bounded. The table below is the authored part of this scene.
"""

import json
import math
import os
import sys
import time

import bpy
import mathutils
import numpy as np

# Emissive material -> how its lamps become lights. Intensity is written verbatim as
# glTF candela (export lighting mode RAW); range is metres.
DERIVED_LIGHTS = {
    "Emissive_StreetLight": {"intensity": 60.0, "range": 14.0, "merge": 0.75},
    "Spotlight_Glass_Emissive": {"intensity": 20.0, "range": 8.0, "merge": 0.5},
    "Shopsign_Book_Store_Emissive": {"intensity": 10.0, "range": 6.0, "merge": 2.0},
    "Shopsign_Pharmacy_Emissive": {"intensity": 10.0, "range": 6.0, "merge": 2.0},
    "Paris_StringLights_01_Blue_Color_Emissive": {"intensity": 3.0, "range": 4.0, "merge": 2.5},
    "Paris_StringLights_01_Green_Color_Emissive": {"intensity": 3.0, "range": 4.0, "merge": 2.5},
    "Paris_StringLights_01_Orange_Color_Emissive": {"intensity": 3.0, "range": 4.0, "merge": 2.5},
    "Paris_StringLights_01_Pink_Color_Emissive": {"intensity": 3.0, "range": 4.0, "merge": 2.5},
    "Paris_StringLights_01_Red_Color_Emissive": {"intensity": 3.0, "range": 4.0, "merge": 2.5},
    "Paris_StringLights_01_White_Color_Emissive": {"intensity": 3.0, "range": 4.0, "merge": 2.5},
}

TEXTURE_KINDS = ("BaseColor", "Specular", "Normal", "Emissive")


def log(message):
    print("[convert_bistro] " + message, flush=True)


def texture_kind(image):
    stem = os.path.splitext(os.path.basename(image.filepath))[0]
    for kind in TEXTURE_KINDS:
        if stem.endswith("_" + kind):
            return kind
    return None


def read_pixels(image):
    pixels = np.empty(len(image.pixels), dtype=np.float32)
    image.pixels.foreach_get(pixels)
    return pixels.reshape(-1, 4)


def write_png(path, pixels, width, height, colorspace):
    image = bpy.data.images.new(os.path.basename(path), width, height, alpha=True)
    image.colorspace_settings.name = colorspace
    image.pixels.foreach_set(pixels.reshape(-1))
    image.filepath_raw = path
    image.file_format = "PNG"
    image.save()
    return image


class TextureConverter:
    """DDS -> PNG, once per source file, with the per-kind fixes applied."""

    def __init__(self, texture_dir):
        self.texture_dir = texture_dir
        self.converted = {}  # source path -> (png image, has cutout alpha)
        os.makedirs(texture_dir, exist_ok=True)

    def convert(self, image, kind):
        source = bpy.path.abspath(image.filepath)
        if source in self.converted:
            return self.converted[source]

        width, height = image.size
        pixels = read_pixels(image)
        has_alpha = False
        colorspace = "sRGB"
        if kind == "Normal":
            colorspace = "Non-Color"
            x = pixels[:, 0] * 2.0 - 1.0
            y = -(pixels[:, 1] * 2.0 - 1.0)  # DirectX -> OpenGL
            z = np.sqrt(np.clip(1.0 - x * x - y * y, 0.0, 1.0))
            pixels[:, 0] = x * 0.5 + 0.5
            pixels[:, 1] = y * 0.5 + 0.5
            pixels[:, 2] = z * 0.5 + 0.5
            pixels[:, 3] = 1.0
        elif kind == "Specular":
            colorspace = "Non-Color"
            pixels[:, 3] = 1.0
        elif kind == "Emissive":
            pixels[:, 3] = 1.0
        else:
            has_alpha = bool((pixels[:, 3] < 0.5).any())
            if not has_alpha:
                pixels[:, 3] = 1.0

        name = os.path.splitext(os.path.basename(source))[0] + ".png"
        png = write_png(os.path.join(self.texture_dir, name), pixels, width, height, colorspace)
        self.converted[source] = (png, has_alpha)
        return self.converted[source]


def gltf_output_group():
    """The node group the glTF exporter reads occlusion from."""
    group = bpy.data.node_groups.get("glTF Material Output")
    if group is None:
        group = bpy.data.node_groups.new("glTF Material Output", "ShaderNodeTree")
        group.interface.new_socket(name="Occlusion", in_out="INPUT", socket_type="NodeSocketFloat")
    return group


def rebuild_material(material, textures, converter):
    found = {}
    for node in material.node_tree.nodes:
        if node.type == "TEX_IMAGE" and node.image is not None:
            kind = texture_kind(node.image)
            if kind and kind not in found:
                found[kind] = node.image

    name = material.name
    is_glass = "Glass" in name and "MASKED" not in name
    double_sided = name.endswith(".DoubleSided") or name.startswith("Foliage")

    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    def texture_node(kind):
        png, has_alpha = converter.convert(found[kind], kind)
        node = nodes.new("ShaderNodeTexImage")
        node.image = png
        return node, has_alpha

    alpha_mode = "OPAQUE"
    if "BaseColor" in found:
        base, has_alpha = texture_node("BaseColor")
        links.new(base.outputs["Color"], bsdf.inputs["Base Color"])
        if is_glass:
            alpha_mode = "BLEND"
            if has_alpha:
                links.new(base.outputs["Alpha"], bsdf.inputs["Alpha"])
            else:
                bsdf.inputs["Alpha"].default_value = 0.25
        elif has_alpha:
            # Round is the exporter's MASK pattern, with a cutoff of 0.5.
            alpha_mode = "MASK"
            round_node = nodes.new("ShaderNodeMath")
            round_node.operation = "ROUND"
            links.new(base.outputs["Alpha"], round_node.inputs[0])
            links.new(round_node.outputs[0], bsdf.inputs["Alpha"])
    elif is_glass:
        alpha_mode = "BLEND"
        bsdf.inputs["Alpha"].default_value = 0.25

    if "Specular" in found:
        spec, _ = texture_node("Specular")
        split = nodes.new("ShaderNodeSeparateColor")
        links.new(spec.outputs["Color"], split.inputs["Color"])
        links.new(split.outputs["Green"], bsdf.inputs["Roughness"])
        links.new(split.outputs["Blue"], bsdf.inputs["Metallic"])
        occlusion = nodes.new("ShaderNodeGroup")
        occlusion.node_tree = gltf_output_group()
        links.new(split.outputs["Red"], occlusion.inputs["Occlusion"])
    else:
        bsdf.inputs["Metallic"].default_value = 0.0
        bsdf.inputs["Roughness"].default_value = 0.5

    if "Normal" in found:
        normal, _ = texture_node("Normal")
        normal_map = nodes.new("ShaderNodeNormalMap")
        links.new(normal.outputs["Color"], normal_map.inputs["Color"])
        links.new(normal_map.outputs["Normal"], bsdf.inputs["Normal"])

    if "Emissive" in found:
        emissive, _ = texture_node("Emissive")
        links.new(emissive.outputs["Color"], bsdf.inputs["Emission Color"])
        bsdf.inputs["Emission Strength"].default_value = 1.0

    material.use_backface_culling = not double_sided
    textures[name] = (alpha_mode, sorted(found))
    return alpha_mode


def emissive_color(png):
    pixels = read_pixels(png)[:, :3]
    luminance = pixels @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    bright = pixels[luminance > 0.5 * luminance.max()]
    color = bright.mean(0) if len(bright) else np.ones(3, dtype=np.float32)
    return tuple(float(c) for c in color / max(float(color.max()), 1e-6))


def emissive_islands(obj, slot_index):
    """World-space centroids of the connected pieces drawn with one material slot."""
    mesh = obj.data
    parent = list(range(len(mesh.vertices)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    used = set()
    for polygon in mesh.polygons:
        if polygon.material_index != slot_index:
            continue
        vertices = list(polygon.vertices)
        used.update(vertices)
        root = find(vertices[0])
        for v in vertices[1:]:
            other = find(v)
            if other != root:
                parent[other] = root

    sums = {}
    for v in used:
        root = find(v)
        position = obj.matrix_world @ mesh.vertices[v].co
        total, count = sums.get(root, (mathutils.Vector((0.0, 0.0, 0.0)), 0))
        sums[root] = (total + position, count + 1)
    return [total / count for total, count in sums.values()]


def merge_points(points, radius):
    merged = []
    for point in points:
        for cluster in merged:
            if (cluster[0] / cluster[1] - point).length <= radius:
                cluster[0] += point
                cluster[1] += 1
                break
        else:
            merged.append([point.copy(), 1])
    return [total / count for total, count in merged]


def derive_lights(converter):
    lights = []
    for material_name, rule in DERIVED_LIGHTS.items():
        material = bpy.data.materials.get(material_name)
        if material is None:
            log("no material " + material_name + "; skipping its lights")
            continue
        emissive = next((n.image for n in material.node_tree.nodes
                         if n.type == "TEX_IMAGE" and n.image is not None), None)
        color = emissive_color(emissive) if emissive is not None else (1.0, 1.0, 1.0)

        points = []
        for obj in bpy.data.objects:
            if obj.type != "MESH":
                continue
            for slot_index, slot in enumerate(obj.material_slots):
                if slot.material == material:
                    points.extend(emissive_islands(obj, slot_index))
        points = merge_points(points, rule["merge"])

        for index, point in enumerate(points):
            name = "Derived_%s_%03d" % (material_name, index)
            data = bpy.data.lights.new(name, "POINT")
            data.energy = rule["intensity"]
            data.color = color
            data.shadow_soft_size = 0.0
            light = bpy.data.objects.new(name, data)
            light.location = point
            bpy.context.scene.collection.objects.link(light)
            lights.append((name, rule["range"]))
        log("%s: %d light(s), colour (%.2f, %.2f, %.2f)" % (material_name, len(points), *color))
    return lights


def write_light_ranges(gltf_path, ranges):
    with open(gltf_path, encoding="utf-8") as f:
        document = json.load(f)
    lights = document.get("extensions", {}).get("KHR_lights_punctual", {}).get("lights", [])
    written = 0
    for light in lights:
        if light.get("name") in ranges:
            light["range"] = ranges[light["name"]]
            written += 1
    with open(gltf_path, "w", encoding="utf-8") as f:
        json.dump(document, f, indent=1)
    log("wrote range on %d of %d glTF light(s)" % (written, len(lights)))


def main():
    args = sys.argv[sys.argv.index("--") + 1:]
    if len(args) != 2:
        raise SystemExit("usage: blender -b --python convert_bistro.py -- <scene.fbx> <out.gltf>")
    source, output = (os.path.abspath(a) for a in args)
    output_dir = os.path.dirname(output)
    os.makedirs(output_dir, exist_ok=True)

    start = time.time()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.render.image_settings.compression = 15
    bpy.ops.import_scene.fbx(filepath=source)
    log("imported %s in %.1f s" % (os.path.basename(source), time.time() - start))

    converter = TextureConverter(os.path.join(output_dir, "textures"))
    report = {}
    modes = {"OPAQUE": 0, "MASK": 0, "BLEND": 0}
    for material in list(bpy.data.materials):
        if material.node_tree is not None:
            modes[rebuild_material(material, report, converter)] += 1
    log("rebuilt %d material(s) %s from %d texture(s) in %.1f s"
        % (len(report), modes, len(converter.converted), time.time() - start))

    derived = derive_lights(converter)

    bpy.ops.export_scene.gltf(
        filepath=output,
        export_format="GLTF_SEPARATE",
        export_keep_originals=True,
        export_yup=True,
        export_apply=False,
        export_texcoords=True,
        export_normals=True,
        export_tangents=True,
        export_materials="EXPORT",
        export_cameras=True,
        export_lights=True,
        export_import_convert_lighting_mode="RAW",
        export_extras=False,
        use_selection=False,
    )
    write_light_ranges(output, dict(derived))
    log("exported %s in %.1f s total" % (output, time.time() - start))


if __name__ == "__main__":
    main()
