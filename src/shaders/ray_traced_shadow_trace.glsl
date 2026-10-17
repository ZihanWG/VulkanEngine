// The sun shadow ray, shared by the two places that trace it: the ray-query
// variant of simple_bindless.frag, inline per fragment, and
// ray_traced_shadow_mask.comp, per pixel ahead of the main pass. One copy, so the
// mask and the inline fallback cannot drift apart -- the main pass mixes the two
// pixel by pixel, and any difference between them would show as a seam. See
// docs/ray_traced_shadows.md.
//
// The includer declares, before including this file:
//   uSceneTopLevel      accelerationStructureEXT, the scene's TLAS;
//   uBaseColorTextures  sampler2D[], the bindless base colour heap;
// and enables GL_EXT_ray_query, GL_EXT_buffer_reference,
// GL_EXT_buffer_reference_uvec2 and GL_EXT_nonuniform_qualifier. Everything else arrives through the two buffer
// references every function takes, rather than through either shader's push
// block, whose layouts differ.

#ifndef VE_RAY_TRACED_SHADOW_TRACE_GLSL
#define VE_RAY_TRACED_SHADOW_TRACE_GLSL

#include "object_frame_data.glsl"
// renderer::Vertex as floats (kRtVertexStrideFloats, kRtVertexUvOffsetFloats).
#include "ray_traced_shadows.glsl"

// Where each draw item's geometry lives, indexed by the same frameDataIndex the
// instance carries as its custom index. Mirrors
// ve::renderer::RayTracingGeometryRecord.
struct RtGeometryRecord {
    uvec2 vertexAddress;
    // Already advanced to the draw item's firstIndex.
    uvec2 indexAddress;
};
layout(buffer_reference, std430) readonly buffer RtGeometryRecordBuffer {
    RtGeometryRecord records[];
};
layout(buffer_reference, std430) readonly buffer RtVertexFloats {
    float values[];
};
layout(buffer_reference, std430) readonly buffer RtIndices {
    uint values[];
};

vec2 rtVertexUv(RtVertexFloats vertices, uint vertexIndex)
{
    uint base = vertexIndex * kRtVertexStrideFloats + kRtVertexUvOffsetFloats;
    return vec2(vertices.values[base], vertices.values[base + 1u]);
}

// Whether a candidate triangle of a cutout caster blocks the sun at the point the
// ray crossed it. The cascades' masked shadow pass, answered for one hit: the
// same three ObjectFrameData fields and the same comparison as
// shadow_masked.frag, so a cutout throws the same shape either way. Level 0 of
// the base colour where the raster pass takes its derivative-selected mip, since
// a ray has no footprint to select one with.
bool rtCandidateBlocksLight(FrameConstantsBuffer frame,
                            ObjectFrameDataBuffer objects,
                            uint frameDataIndex,
                            uint primitive,
                            vec2 barycentrics)
{
    RtGeometryRecord record = RtGeometryRecordBuffer(frame.values.rayTracedShadow.xy).records[frameDataIndex];
    RtIndices indices = RtIndices(record.indexAddress);
    RtVertexFloats vertices = RtVertexFloats(record.vertexAddress);
    uint first = primitive * 3u;
    vec2 uv = rtVertexUv(vertices, indices.values[first]) * (1.0 - barycentrics.x - barycentrics.y) +
              rtVertexUv(vertices, indices.values[first + 1u]) * barycentrics.x +
              rtVertexUv(vertices, indices.values[first + 2u]) * barycentrics.y;

    uint textureIndex = objects.objects[frameDataIndex].textureIndices.x;
    float alpha = textureLod(uBaseColorTextures[nonuniformEXT(textureIndex)], uv, 0.0).a *
                  objects.objects[frameDataIndex].baseColorFactor.a;
    return alpha >= objects.objects[frameDataIndex].materialParams.w;
}

// 1.0 when nothing in the scene lies between this surface and the sun along
// rayDirection, 0.0 when something does, and how far the ray went before it hit
// (negative when it hit nothing). One ray: a hard shadow when rayDirection is the
// sun's centre, one sample of a soft one when it is a point on the sun's disc
// (rtSunConeDirection).
//
// The origin moves off the surface by a constant plus a term that grows with
// distance from the camera (FrameConstants::rayTracedShadowParams). The second
// is what covers LOD: rays see every mesh at level 0, while the raster pass drew
// whichever level the cull chose, so a distant simplified surface would
// otherwise find its own detailed version just above it. The offset goes to the
// side of the surface facing the sun's centre, so a two-sided caster seen from
// behind does not start its ray inside itself.
float rayTracedSunVisibilityAlong(FrameConstantsBuffer frame,
                                  ObjectFrameDataBuffer objects,
                                  vec3 worldPosition,
                                  vec3 surfaceNormal,
                                  vec3 towardLight,
                                  vec3 rayDirection,
                                  out float hitDistance)
{
    vec4 params = frame.values.rayTracedShadowParams;
    float offset = params.x + params.y * length(frame.values.cameraPosition.xyz - worldPosition);
    vec3 normal = dot(surfaceNormal, towardLight) >= 0.0 ? surfaceNormal : -surfaceNormal;
    vec3 origin = worldPosition + normal * offset;

    rayQueryEXT query;
    rayQueryInitializeEXT(query, uSceneTopLevel, gl_RayFlagsTerminateOnFirstHitEXT, 0xFFu, origin, 0.0,
                          rayDirection, params.z);
    // Opaque instances never stop here -- they are FORCE_OPAQUE and commit on
    // their own. Only cutout instances present candidates.
    while (rayQueryProceedEXT(query)) {
        if (rayQueryGetIntersectionTypeEXT(query, false) == gl_RayQueryCandidateIntersectionTriangleEXT) {
            uint frameDataIndex = uint(rayQueryGetIntersectionInstanceCustomIndexEXT(query, false));
            uint primitive = uint(rayQueryGetIntersectionPrimitiveIndexEXT(query, false));
            vec2 barycentrics = rayQueryGetIntersectionBarycentricsEXT(query, false);
            if (rtCandidateBlocksLight(frame, objects, frameDataIndex, primitive, barycentrics)) {
                rayQueryConfirmIntersectionEXT(query);
            }
        }
    }
    if (rayQueryGetIntersectionTypeEXT(query, true) == gl_RayQueryCommittedIntersectionNoneEXT) {
        hitDistance = -1.0;
        return 1.0;
    }
    hitDistance = rayQueryGetIntersectionTEXT(query, true);
    return 0.0;
}

// The hard shadow: one ray towards the sun's centre.
float rayTracedSunVisibility(FrameConstantsBuffer frame,
                             ObjectFrameDataBuffer objects,
                             vec3 worldPosition,
                             vec3 surfaceNormal,
                             vec3 towardLight)
{
    float hitDistance;
    return rayTracedSunVisibilityAlong(
        frame, objects, worldPosition, surfaceNormal, towardLight, towardLight, hitDistance);
}

// --- Sampling the sun's disc --------------------------------------------------
//
// Mirrored in C++ by renderer::concentricDiskSample, renderer::orthonormalBasis
// and renderer::sunConeDirection, where the tests that pin them live.

// A point on the unit disc from a point in the unit square, area-preserving and
// continuous (Shirley and Chiu's concentric map), so stratified or low-
// discrepancy inputs stay stratified on the disc.
vec2 rtConcentricDisk(vec2 u)
{
    vec2 offset = 2.0 * u - 1.0;
    if (offset.x == 0.0 && offset.y == 0.0) {
        return vec2(0.0);
    }
    float radius;
    float angle;
    if (abs(offset.x) > abs(offset.y)) {
        radius = offset.x;
        angle = kRtQuarterPi * (offset.y / offset.x);
    } else {
        radius = offset.y;
        angle = kRtHalfPi - kRtQuarterPi * (offset.x / offset.y);
    }
    return radius * vec2(cos(angle), sin(angle));
}

// Two unit vectors that complete n to an orthonormal basis, without a branch on
// which axis n is closest to (Duff et al., "Building an Orthonormal Basis,
// Revisited").
void rtOrthonormalBasis(vec3 n, out vec3 tangent, out vec3 bitangent)
{
    float sign = n.z >= 0.0 ? 1.0 : -1.0;
    float a = -1.0 / (sign + n.z);
    float b = n.x * n.y * a;
    tangent = vec3(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x);
    bitangent = vec3(b, sign + n.y * n.y * a, -n.y);
}

// A direction towards a point on the sun's disc, whose half-angle has tangent
// tanRadius, picked by u in the unit square. u = (0.5, 0.5) is the centre.
vec3 rtSunConeDirection(vec3 towardLight, float tanRadius, vec2 u)
{
    vec3 tangent;
    vec3 bitangent;
    rtOrthonormalBasis(towardLight, tangent, bitangent);
    vec2 disk = rtConcentricDisk(u) * tanRadius;
    return normalize(towardLight + disk.x * tangent + disk.y * bitangent);
}

// Interleaved gradient noise (Jimenez): a per-pixel value in [0, 1) whose
// neighbours differ as much as possible, which is what a spatial filter over
// one sample per pixel wants to average.
float rtInterleavedGradientNoise(vec2 pixel)
{
    return fract(52.9829189 * fract(dot(floor(pixel), vec2(0.06711056, 0.00583715))));
}

#endif // VE_RAY_TRACED_SHADOW_TRACE_GLSL
