#ifndef VE_PROBE_HIT_SHADING_GLSL
#define VE_PROBE_HIT_SHADING_GLSL

// One ray of the probe gather, from anywhere: the closest hit, shaded the way a
// probe sees it. probe_trace.comp fires these from a probe; the ray-traced
// reference (probe_reference.comp) fires them from a pixel's surface, so the
// reference measures the same light the probes are meant to hold -- the same
// surfaces, the same shading, the same sky -- and differs only in where the rays
// start and how many there are. See docs/irradiance_probes.md.
//
// The includer declares uSceneTopLevel and uBaseColorTextures and includes
// irradiance_probes.glsl, probe_capture_shading.glsl and
// ray_traced_shadow_trace.glsl first.

layout(buffer_reference, std430) readonly buffer RtPositionFloats {
    float values[];
};

vec3 rtVertexPosition(RtVertexFloats vertices, uint vertexIndex)
{
    uint base = vertexIndex * kRtVertexStrideFloats;
    return vec3(vertices.values[base], vertices.values[base + 1u], vertices.values[base + 2u]);
}

vec3 rtVertexNormal(RtVertexFloats vertices, uint vertexIndex)
{
    uint base = vertexIndex * kRtVertexStrideFloats + kRtVertexNormalOffsetFloats;
    return vec3(vertices.values[base], vertices.values[base + 1u], vertices.values[base + 2u]);
}

vec3 rtSkinnedPosition(RtPositionFloats positions, uint vertexIndex)
{
    uint base = vertexIndex * kRtSkinnedPositionStrideFloats;
    return vec3(positions.values[base], positions.values[base + 1u], positions.values[base + 2u]);
}

// Mirrors ve::renderer::probeRayConeTextureLod, with the cone's spread a
// parameter: the probe gather's rays each stand for a capture texel
// (kProbeTraceConeSpread), the reference's for an infinitesimal direction (0,
// which reads level 0 and lets the many samples average the texture instead).
float probeRayConeTextureLod(float hitDistance,
                             float cosIncidence,
                             float worldArea,
                             float uvArea,
                             ivec2 textureSize,
                             float spread)
{
    float texelArea = uvArea * float(textureSize.x) * float(textureSize.y);
    if (!(worldArea > 0.0) || !(texelArea > 0.0) || !(hitDistance > 0.0)) {
        return 0.0;
    }
    float cosine = max(abs(cosIncidence), kProbeTraceMinCosine);
    if (!(spread > 0.0)) {
        return 0.0;
    }
    float coneWidth = hitDistance * spread / cosine;
    float texelWorldSize = sqrt(worldArea / texelArea);
    return max(log2(coneWidth / texelWorldSize), 0.0);
}

// What a ray needs besides where it goes.
struct ProbeRayContext {
    FrameConstantsBuffer frame;
    ObjectFrameDataBuffer objects;
    RtGeometryRecordBuffer geometry;
    ProbeCaptureLighting lighting;
    // How far a hit's sun shadow ray may go: the scene's extent.
    float sunRayLength;
    // The angle the ray stands for, which picks its texture level
    // (probeRayConeTextureLod).
    float coneSpread;
};

// The radiance arriving at `origin` from `direction`, in rgb, and how far the
// surface it came from is, in a: negated for the back of a single-sided surface,
// which is what classification counts. Nothing within tMax is the sky term at
// kProbeMaxDistance, as the raster capture's clear.
vec4 probeTraceRadiance(ProbeRayContext context, vec3 origin, vec3 direction, float tMin, float tMax)
{
    vec3 ambient = context.lighting.ambient;

    // Closest hit, so cutout candidates are confirmed where they are solid and
    // the traversal carries on past them where they are not -- the raster
    // capture's discard, answered per ray.
    rayQueryEXT query;
    rayQueryInitializeEXT(query, uSceneTopLevel, gl_RayFlagsNoneEXT, 0xFFu, origin, tMin, direction, tMax);
    while (rayQueryProceedEXT(query)) {
        if (rayQueryGetIntersectionTypeEXT(query, false) == gl_RayQueryCandidateIntersectionTriangleEXT) {
            uint candidateIndex = uint(rayQueryGetIntersectionInstanceCustomIndexEXT(query, false));
            uint candidatePrimitive = uint(rayQueryGetIntersectionPrimitiveIndexEXT(query, false));
            vec2 candidateBarycentrics = rayQueryGetIntersectionBarycentricsEXT(query, false);
            if (rtCandidateIsSolid(context.geometry, context.objects, candidateIndex, candidatePrimitive,
                                   candidateBarycentrics)) {
                rayQueryConfirmIntersectionEXT(query);
            }
        }
    }

    // Nothing within reach: the sky term and the far bound, which is exactly
    // what the raster capture's clear leaves in a texel nothing drew into.
    if (rayQueryGetIntersectionTypeEXT(query, true) == gl_RayQueryCommittedIntersectionNoneEXT) {
        return vec4(ambient, kProbeMaxDistance);
    }

    float hitDistance = rayQueryGetIntersectionTEXT(query, true);
    uint frameDataIndex = uint(rayQueryGetIntersectionInstanceCustomIndexEXT(query, true));
    uint primitive = uint(rayQueryGetIntersectionPrimitiveIndexEXT(query, true));
    vec2 barycentrics = rayQueryGetIntersectionBarycentricsEXT(query, true);
    mat4x3 objectToWorld = rayQueryGetIntersectionObjectToWorldEXT(query, true);
    uint instanceFlags = rayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetEXT(query, true);
    bool skinned = (instanceFlags & kRtInstanceFlagSkinned) != 0u;
    vec3 weights = vec3(1.0 - barycentrics.x - barycentrics.y, barycentrics.x, barycentrics.y);

    RtGeometryRecord record = context.geometry.records[frameDataIndex];
    RtIndices indices = RtIndices(record.indexAddress);
    uint first = primitive * 3u;
    uvec3 corner = uvec3(indices.values[first], indices.values[first + 1u], indices.values[first + 2u]);

    vec3 p0;
    vec3 p1;
    vec3 p2;
    vec3 interpolatedNormal;
    vec2 uv = vec2(0.5);
    float uvArea = 0.0;
    if (skinned) {
        // The skinned caster's record names its pre-skinned positions, not a
        // renderer::Vertex stream: no normals and no UVs to interpolate. The
        // face normal stands in, and nothing below samples a texture for it --
        // its material is untextured.
        RtPositionFloats positions = RtPositionFloats(record.vertexAddress);
        p0 = rtSkinnedPosition(positions, corner.x);
        p1 = rtSkinnedPosition(positions, corner.y);
        p2 = rtSkinnedPosition(positions, corner.z);
        interpolatedNormal = cross(p1 - p0, p2 - p0);
    } else {
        RtVertexFloats vertices = RtVertexFloats(record.vertexAddress);
        p0 = rtVertexPosition(vertices, corner.x);
        p1 = rtVertexPosition(vertices, corner.y);
        p2 = rtVertexPosition(vertices, corner.z);
        interpolatedNormal = rtVertexNormal(vertices, corner.x) * weights.x +
                             rtVertexNormal(vertices, corner.y) * weights.y +
                             rtVertexNormal(vertices, corner.z) * weights.z;
        vec2 uv0 = rtVertexUv(vertices, corner.x);
        vec2 uv1 = rtVertexUv(vertices, corner.y);
        vec2 uv2 = rtVertexUv(vertices, corner.z);
        uv = uv0 * weights.x + uv1 * weights.y + uv2 * weights.z;
        vec2 du = uv1 - uv0;
        vec2 dv = uv2 - uv0;
        uvArea = abs(du.x * dv.y - du.y * dv.x);
    }

    // Object to world the way the raster capture's vertex stage does it:
    // mat3(model) on the normal, without the inverse transpose.
    mat3 linear = mat3(objectToWorld);
    vec3 worldEdge1 = linear * (p1 - p0);
    vec3 worldEdge2 = linear * (p2 - p0);
    vec3 faceNormal = cross(worldEdge1, worldEdge2);
    float worldArea = length(faceNormal);
    faceNormal = worldArea > 0.0 ? faceNormal / worldArea : -direction;

    vec3 normal = linear * interpolatedNormal;
    float normalLength = length(normal);
    normal = normalLength > 0.0 ? normal / normalLength : faceNormal;
    // Two-sided, as the raster capture: a probe inside a room sees the inward
    // face of walls authored facing out.
    //
    // The same test says the ray hit the back of the surface -- judged against
    // the authored normal, so it does not depend on the winding convention. A
    // probe whose rays mostly do that is inside a closed mesh; the convolution
    // counts them from the sign of the distance written below. Not on a
    // double-sided material, whose back is a real surface, nor on the skinned
    // caster, whose face normal does follow the winding.
    bool hitBackFace = dot(normal, direction) > 0.0;
    if (hitBackFace) {
        normal = -normal;
    }
    bool countsAsBackFace = hitBackFace && !skinned && (instanceFlags & kRtInstanceFlagDoubleSided) == 0u;

    vec3 worldPosition = origin + direction * hitDistance;

    ObjectFrameData object = context.objects.objects[frameDataIndex];
    uvec4 textureIndices = object.textureIndices;
    vec3 albedo = object.baseColorFactor.rgb;
    vec3 emissive = object.emissiveFactor.rgb;
    // The skinned caster has no UVs and an untextured material, so its factors
    // are its colour. Fetching anyway read the fallback texture at a made-up
    // (0.5, 0.5), which tinted the hit by whatever texel sits there and
    // described nothing -- and on lavapipe, from the ray-query fragment shader
    // of hybrid reflections, that fetch faulted (isolated by elimination: the
    // crash went with the fetch and stayed with every other part of the
    // skinned hit, including forcing the indices to a known-good slot).
    if (!skinned) {
        float baseLod = probeRayConeTextureLod(hitDistance, dot(faceNormal, direction), worldArea, uvArea,
                                               textureSize(uBaseColorTextures[nonuniformEXT(textureIndices.x)], 0), context.coneSpread);
        float emissiveLod = probeRayConeTextureLod(hitDistance, dot(faceNormal, direction), worldArea, uvArea,
                                                   textureSize(uBaseColorTextures[nonuniformEXT(textureIndices.w)], 0), context.coneSpread);
        albedo *= textureLod(uBaseColorTextures[nonuniformEXT(textureIndices.x)], uv, baseLod).rgb;
        emissive *= textureLod(uBaseColorTextures[nonuniformEXT(textureIndices.w)], uv, emissiveLod).rgb;
    }

    ProbeCaptureLighting lighting = context.lighting;

    // Traced only where the sun could light the surface at all: a surface facing
    // away has no lambert to shadow, and a scene with the sun off (the Cornell
    // box) has nothing to trace towards.
    float sunVisibility = 0.0;
    vec3 towardSun = normalize(-lighting.sunDirection);
    if (dot(normal, towardSun) > 0.0 && dot(lighting.sunColor, vec3(1.0)) > 0.0) {
        float sunHitDistance;
        vec3 sunOrigin = worldPosition + normal * kProbeTraceSurfaceOffset;
        sunVisibility = rtOccluded(context.geometry, context.objects, sunOrigin, towardSun, context.sunRayLength,
                                   sunHitDistance) ? 0.0 : 1.0;
    }

#ifdef VE_PROBE_HIT_AMBIENT
    // The includer answers the hit's indirect light itself. A reflection ray
    // (ssr_trace_rq.frag) shows the surface it hits to the camera, next to the
    // same surface seen directly, so it reads the main pass's diffuse irradiance
    // cube rather than the probes' constant ambient -- otherwise the two would
    // disagree across every screen-space/traced seam. The probe gathers leave
    // this undefined, so what they record is unchanged.
    lighting.ambient = probeHitAmbient(normal);
#endif
    vec3 radiance = probeCaptureRadiance(lighting, albedo, emissive, worldPosition, normal, origin, sunVisibility);
    // Negative distance marks a back-face hit. The convolution reads the
    // magnitude for visibility, so the depth a probe stores is unchanged.
    float storedDistance = min(hitDistance, kProbeMaxDistance);
    return vec4(radiance, countsAsBackFace ? -storedDistance : storedDistance);
}

#endif // VE_PROBE_HIT_SHADING_GLSL
