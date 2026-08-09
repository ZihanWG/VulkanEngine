// Per-draw-item data, mirrored from ve::ObjectFrameData in
// src/renderer/RendererInternal.h. That struct's offsets are pinned by
// static_asserts; this block is the other half of the contract and nothing
// checks the two agree, so they have to be edited together.
//
// This lives in a shared header because the array stride depends on the whole
// struct: a shader that reads only lightMvp still has to declare every member,
// so six shaders used to carry identical copies and any edit had to land in all
// six. Missing one is silent -- the layout error shows up as garbage in whatever
// member came after the divergence, with no validation message.
//
// std430, reached through buffer_reference rather than a descriptor, which is
// why the vertex stage binds no descriptor set at all.
//
// Every member is a multiple of 16 bytes on purpose. That is what makes the C++
// and std430 layouts agree: glm's alignment is only 4, so the agreement comes
// from each member's *size* landing on a 16-byte boundary, not from matching
// alignment. Adding a bare float or a vec3 here breaks that silently.

#ifndef VE_OBJECT_FRAME_DATA_GLSL
#define VE_OBJECT_FRAME_DATA_GLSL

struct ObjectFrameData {
    mat4 mvp;
    mat4 model;
    vec4 baseColorFactor;
    // x = metallic, y = roughness, z = multiScatterStrength,
    // w = alpha-test cutoff (>= 0 clips, < 0 disables)
    vec4 materialParams;
    // x = base color, y = normal, z = metallic-roughness, w = emissive
    uvec4 textureIndices;
    // rgb = emissive factor, w = 1.0 when an emissive texture is bound
    vec4 emissiveFactor;
    // Unjittered current/previous MVP, for the velocity buffer. Rasterization
    // keeps using the jittered mvp so TAA jitter never leaks into velocity.
    mat4 currMvpNoJitter;
    mat4 prevMvpNoJitter;
};

layout(buffer_reference, std430) readonly buffer ObjectFrameDataBuffer {
    ObjectFrameData objects[];
};

// Values identical for every draw item in the frame. Stored once and reached
// through its own address rather than copied into all N object records; see
// ve::FrameConstants in src/renderer/RendererInternal.h.
struct FrameConstants {
    // One per cascade. Shaders build the light-space position as
    // cascadeViewProjection[i] * worldPosition rather than reading a
    // per-object cascadeViewProjection[i] * model product, which used to be
    // 256 of every object record's bytes.
    mat4 cascadeViewProjection[4];
    vec4 lightDirection;
    vec4 lightColor;
    vec4 ambientColor;
    // Positive camera-view depths for cascades 0..3.
    vec4 cascadeSplits;
    // x = constant depth bias, y = slope bias, z/w reserved
    vec4 shadowSettings;
    // xyz = world-space camera position, w = cascade debug-colour toggle
    vec4 cameraPosition;
    // xyz = camera forward, w = cascade count
    vec4 cameraForward;
};

layout(buffer_reference, std430) readonly buffer FrameConstantsBuffer {
    FrameConstants values;
};

#endif // VE_OBJECT_FRAME_DATA_GLSL
