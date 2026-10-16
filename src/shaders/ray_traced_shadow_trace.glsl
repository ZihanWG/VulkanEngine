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

// 1.0 when nothing in the scene lies between this surface and the sun, 0.0 when
// something does. Hard by construction: one ray, toward a light with no extent.
//
// The origin moves off the surface by a constant plus a term that grows with
// distance from the camera (FrameConstants::rayTracedShadowParams). The second
// is what covers LOD: rays see every mesh at level 0, while the raster pass drew
// whichever level the cull chose, so a distant simplified surface would
// otherwise find its own detailed version just above it. The offset goes to the
// side of the surface facing the sun, so a two-sided caster seen from behind
// does not start its ray inside itself.
float rayTracedSunVisibility(FrameConstantsBuffer frame,
                             ObjectFrameDataBuffer objects,
                             vec3 worldPosition,
                             vec3 surfaceNormal,
                             vec3 towardLight)
{
    vec4 params = frame.values.rayTracedShadowParams;
    float offset = params.x + params.y * length(frame.values.cameraPosition.xyz - worldPosition);
    vec3 normal = dot(surfaceNormal, towardLight) >= 0.0 ? surfaceNormal : -surfaceNormal;
    vec3 origin = worldPosition + normal * offset;

    rayQueryEXT query;
    rayQueryInitializeEXT(query, uSceneTopLevel, gl_RayFlagsTerminateOnFirstHitEXT, 0xFFu, origin, 0.0,
                          towardLight, params.z);
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
    return rayQueryGetIntersectionTypeEXT(query, true) == gl_RayQueryCommittedIntersectionNoneEXT ? 1.0 : 0.0;
}

#endif // VE_RAY_TRACED_SHADOW_TRACE_GLSL
