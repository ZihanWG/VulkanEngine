// Rebuilding a surface from the depth prepass, shared by the ray-traced shadow
// mask's trace and resolve passes. The prepass has no fragment shader and no
// colour target by design, so position and normal both come from depth: the
// position by unprojecting a pixel centre through the inverse of the jittered
// view-projection the prepass rasterised with, the normal from the position
// differences towards whichever neighbour on each axis is closer in depth. See
// docs/ray_traced_shadows.md#the-screen-space-mask.
//
// The includer declares uSceneDepth (sampler2D, main depth through a nearest
// sampler) before including this file.

#ifndef VE_RAY_TRACED_SHADOW_RECONSTRUCT_GLSL
#define VE_RAY_TRACED_SHADOW_RECONSTRUCT_GLSL

// The full-resolution sub-rect the prepass drew, and how to unproject it.
struct RtDepthGrid {
    mat4 inverseViewProjection;
    vec2 inverseExtent;
    ivec2 extent;
};

vec3 rtWorldPositionAt(RtDepthGrid grid, ivec2 pixel, float depth)
{
    vec2 ndc = (vec2(pixel) + 0.5) * grid.inverseExtent * 2.0 - 1.0;
    vec4 world = grid.inverseViewProjection * vec4(ndc, depth, 1.0);
    return world.xyz / world.w;
}

// Depth at a neighbour, or 2.0 -- further than any surface -- when it is outside
// the sub-rect or on the cleared far plane, so it is never the closer choice.
float rtNeighbourDepth(RtDepthGrid grid, ivec2 pixel)
{
    if (any(lessThan(pixel, ivec2(0))) || any(greaterThanEqual(pixel, grid.extent))) {
        return 2.0;
    }
    float depth = texelFetch(uSceneDepth, pixel, 0).r;
    return depth >= 1.0 ? 2.0 : depth;
}

// One axis of the reconstructed surface: towards whichever neighbour is closer
// in depth, so a silhouette takes its tangent from the surface it belongs to and
// not across the gap to whatever lies behind. False when neither side is usable.
bool rtSurfaceTangent(RtDepthGrid grid, ivec2 pixel, ivec2 step, float depth, vec3 position, out vec3 tangent)
{
    float after = rtNeighbourDepth(grid, pixel + step);
    float before = rtNeighbourDepth(grid, pixel - step);
    if (after > 1.0 && before > 1.0) {
        return false;
    }
    if (abs(after - depth) <= abs(before - depth)) {
        tangent = rtWorldPositionAt(grid, pixel + step, after) - position;
    } else {
        tangent = position - rtWorldPositionAt(grid, pixel - step, before);
    }
    return true;
}

// The face normal at a pixel, turned towards the camera. False when a neighbour
// on either axis gave nothing to take a tangent from.
bool rtSurfaceNormal(RtDepthGrid grid, ivec2 pixel, float depth, vec3 position, vec3 cameraPosition, out vec3 normal)
{
    vec3 tangentX;
    vec3 tangentY;
    if (!rtSurfaceTangent(grid, pixel, ivec2(1, 0), depth, position, tangentX) ||
        !rtSurfaceTangent(grid, pixel, ivec2(0, 1), depth, position, tangentY)) {
        return false;
    }
    normal = normalize(cross(tangentX, tangentY));
    if (dot(normal, cameraPosition - position) < 0.0) {
        normal = -normal;
    }
    return true;
}

#endif // VE_RAY_TRACED_SHADOW_RECONSTRUCT_GLSL
