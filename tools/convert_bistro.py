"""Convert Amazon Lumberyard Bistro (FBX + DDS) into the glTF --scene bistro loads.

Runs inside Blender, which is the only free tool that reads both the FBX and the
block-compressed DDS textures it references:

    blender -b --factory-startup --python tools/convert_bistro.py -- \\
        <Bistro_v5_2/BistroExterior.fbx> <build/fetched-assets/bistro/Bistro.gltf>
    blender -b --factory-startup --python tools/convert_bistro.py -- \\
        <Bistro_v5_2/BistroInterior_Wine.fbx> \\
        <build/fetched-assets/bistro-interior/BistroInterior.gltf>

Which scene it is comes from the FBX's file name (SCENE_PROFILES): each has its
own derived-light table, and the interior keeps the factors its drinks are made of.

The scene is CC BY 4.0 (NVIDIA ORCA). Nothing it produces is tracked; the output
sits beside the fetched Sponza under build/fetched-assets.

What the conversion has to decide, rather than copy:

* Materials. Blender's FBX importer wires the packed "Specular" texture into the
  Principled BSDF's Specular IOR Level, which the glTF exporter cannot express. The
  scene's own README defines that texture as occlusion (R), roughness (G) and
  metalness (B) -- glTF's metallicRoughness layout exactly -- so every material is
  rebuilt from the four textures its name carries. Occlusion is not exported: red
  is zero in every texel of every one of them, and a viewer that reads it as
  glTF occlusion renders the whole scene without ambient light.
* Normal maps are DirectX-convention BC5, two channels. Green is flipped to the
  OpenGL convention glTF specifies and Z is reconstructed, so the PNG is a complete
  normal map for any consumer rather than only for one that rebuilds Z.
* Alpha. Foliage, curtains and lettering carry cutout alpha (MASK); glass blends.
  Everything else is single-sided -- the FBX importer marks every material
  double-sided, which would put the whole scene on the un-culled pipeline.
* Sky. The bundled HDR is copied beside the glTF as environment.hdr.
* Lights. The FBX holds one sun. The lamps are emissive geometry meant for a path
  tracer, so a raster renderer needs analytic lights placed at them: one per
  emissive island (merged within a radius for string lights), coloured by the
  emissive texture, with a range written into KHR_lights_punctual so the clustered
  cull stays bounded. The table below is the authored part of this scene.
"""

import glob
import json
import os
import shutil
import sys
import time

import bpy
import mathutils
import numpy as np

# Emissive material -> how its lamps become lights. Intensity is written verbatim as
# glTF candela (export lighting mode RAW); range is metres. Tuned by eye against the
# FBX camera's shot under the bundled sky: most street lamps hang off a facade, so
# a light at the lantern sits a few centimetres from the wall, and inverse-square
# falloff blew it out above 20. A "color" overrides the one read from the emissive
# texture: the lamps' brightest texels are clipped to white, which lit the street
# cold under amber lanterns -- obvious once --scene bistro-night removed the sun.
WARM_INCANDESCENT = (1.0, 0.72, 0.42)
EXTERIOR_LIGHTS = {
    "Emissive_StreetLight": {"intensity": 20.0, "range": 14.0, "merge": 0.75, "color": WARM_INCANDESCENT},
    "Spotlight_Glass_Emissive": {"intensity": 6.0, "range": 6.0, "merge": 0.5, "color": WARM_INCANDESCENT},
    "Shopsign_Book_Store_Emissive": {"intensity": 10.0, "range": 6.0, "merge": 2.0},
    "Shopsign_Pharmacy_Emissive": {"intensity": 10.0, "range": 6.0, "merge": 2.0},
    "Paris_StringLights_01_Blue_Color_Emissive": {"intensity": 3.0, "range": 4.0, "merge": 2.5},
    "Paris_StringLights_01_Green_Color_Emissive": {"intensity": 3.0, "range": 4.0, "merge": 2.5},
    "Paris_StringLights_01_Orange_Color_Emissive": {"intensity": 3.0, "range": 4.0, "merge": 2.5},
    "Paris_StringLights_01_Pink_Color_Emissive": {"intensity": 3.0, "range": 4.0, "merge": 2.5},
    "Paris_StringLights_01_Red_Color_Emissive": {"intensity": 3.0, "range": 4.0, "merge": 2.5},
    "Paris_StringLights_01_White_Color_Emissive": {"intensity": 3.0, "range": 4.0, "merge": 2.5},
}

# The interior's lamps, by the same rules. Its FBX holds no lights either. The
# ceiling fixtures are clusters of small emissive pieces, merged into one light
# per fixture; the wall sconces likewise. Tuned by eye against the FBX camera's
# shot, as above.
INTERIOR_LIGHTS = {
    "Paris_Ceiling_Lamp": {"intensity": 15.0, "range": 10.0, "merge": 0.75, "color": WARM_INCANDESCENT},
    "Paris_Wall_Light_Interior": {"intensity": 6.0, "range": 6.0, "merge": 0.75, "color": WARM_INCANDESCENT},
    "Paris_Wall_Bulb_Light": {"intensity": 4.0, "range": 5.0, "merge": 0.3, "color": WARM_INCANDESCENT},
    "Lightbulb": {"intensity": 4.0, "range": 5.0, "merge": 0.3, "color": WARM_INCANDESCENT},
    "Wall_Lamp": {"intensity": 4.0, "range": 5.0, "merge": 0.75, "color": WARM_INCANDESCENT},
}

# Per scene, by the FBX's file name. keep_untextured_factors keeps a material's own
# base colour, opacity and emission where no texture replaces them: the interior's
# drinks (wine, beer, water, ice) and its bare bulb are colour and opacity factors
# only, and rebuilding them from textures alone made them opaque grey. The exterior
# does not set it, so its conversion is unchanged.
SCENE_PROFILES = {
    "BistroExterior": {"lights": EXTERIOR_LIGHTS, "keep_untextured_factors": False},
    "BistroInterior_Wine": {"lights": INTERIOR_LIGHTS, "keep_untextured_factors": True},
    "BistroInterior": {"lights": INTERIOR_LIGHTS, "keep_untextured_factors": True},
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


def rebuild_material(material, textures, converter, keep_untextured_factors=False):
    found = {}
    for node in material.node_tree.nodes:
        if node.type == "TEX_IMAGE" and node.image is not None:
            kind = texture_kind(node.image)
            if kind and kind not in found:
                found[kind] = node.image

    name = material.name
    is_glass = "Glass" in name and "MASKED" not in name
    double_sided = name.endswith(".DoubleSided") or name.startswith("Foliage")

    # The importer's own factors, read before the node tree is cleared.
    original = next((n for n in material.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
    original_base = tuple(original.inputs["Base Color"].default_value) if original else None
    original_alpha = float(original.inputs["Alpha"].default_value) if original else 1.0
    original_emission = tuple(original.inputs["Emission Color"].default_value) if original else None

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
    elif keep_untextured_factors and original_base is not None:
        bsdf.inputs["Base Color"].default_value = original_base
        if original_alpha < 1.0:
            alpha_mode = "BLEND"
            bsdf.inputs["Alpha"].default_value = original_alpha

    if "Specular" in found:
        spec, _ = texture_node("Specular")
        split = nodes.new("ShaderNodeSeparateColor")
        links.new(spec.outputs["Color"], split.inputs["Color"])
        links.new(split.outputs["Green"], bsdf.inputs["Roughness"])
        links.new(split.outputs["Blue"], bsdf.inputs["Metallic"])
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
    elif keep_untextured_factors and original_emission is not None and max(original_emission[:3]) > 0.0:
        bsdf.inputs["Emission Color"].default_value = original_emission
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


def derive_lights(converter, table):
    lights = []
    for material_name, rule in table.items():
        material = bpy.data.materials.get(material_name)
        if material is None:
            log("no material " + material_name + "; skipping its lights")
            continue
        emissive = next((n.image for n in material.node_tree.nodes
                         if n.type == "TEX_IMAGE" and n.image is not None), None)
        if "color" in rule:
            color = rule["color"]
        else:
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
    stem = os.path.splitext(os.path.basename(source))[0]
    if stem not in SCENE_PROFILES:
        raise SystemExit("no profile for %s; known: %s" % (stem, ", ".join(SCENE_PROFILES)))
    profile = SCENE_PROFILES[stem]
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
            modes[rebuild_material(material, report, converter, profile["keep_untextured_factors"])] += 1
    log("rebuilt %d material(s) %s from %d texture(s) in %.1f s"
        % (len(report), modes, len(converter.converted), time.time() - start))

    derived = derive_lights(converter, profile["lights"])

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

    # The sky the scene was lit under ships beside the FBX. The renderer picks up
    # an imported scene's environment.hdr in place of its default environment.
    skies = sorted(glob.glob(os.path.join(os.path.dirname(source), "*.hdr")))
    if skies:
        shutil.copyfile(skies[0], os.path.join(output_dir, "environment.hdr"))
        log("environment: " + os.path.basename(skies[0]))
    log("exported %s in %.1f s total" % (output, time.time() - start))


if __name__ == "__main__":
    main()
