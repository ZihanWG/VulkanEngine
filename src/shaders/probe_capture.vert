#version 460

#extension GL_EXT_buffer_reference : require

// Vertex stage for irradiance-probe capture: rasterises the scene from a probe's
// position along one cube face.
//
// Shaped like shadow_punctual.vert rather than simple.vert: the face's
// view-projection arrives as a push constant, one push per (probe, face) rather
// than per object, so ObjectFrameData does not have to grow a matrix per probe.
// Unlike the shadow path this one also carries the shading inputs, because the
// capture records radiance and not just depth.

#include "object_frame_data.glsl"
// The mat4 sits at offset 16, not 8: push-constant layout rounds the 8-byte
// buffer reference up to the matrix's 16-byte alignment. The C++
// ProbeCapturePushConstants mirror carries explicit padding to match.
layout(push_constant) uniform PushConstants {
    ObjectFrameDataBuffer objectFrameData;
    // Occupies what used to be two padding words in ProbeCapturePushConstants.
    FrameConstantsBuffer frameConstants;
    mat4 faceViewProjection;
    // xyz = the capturing probe's world position. The fragment stage needs it to
    // record how far away each surface is, which is what the depth atlas stores.
    vec4 probePosition;
} pc;

const uint kObjectIndexMask = 0xFFFFu;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec2 inUV;
layout(location = 3) in vec3 inNormal;
layout(location = 4) in vec4 inTangent;

layout(location = 0) out vec2 vUV;
layout(location = 1) out vec3 vNormal;
layout(location = 2) out vec3 vWorldPosition;
layout(location = 3) flat out vec3 vLightDirection;
layout(location = 4) flat out vec3 vLightColor;
layout(location = 5) flat out vec3 vAmbientColor;
layout(location = 6) out vec4 vLightSpacePosition[4];
layout(location = 10) flat out vec4 vShadowSettings;
layout(location = 11) flat out vec4 vCascadeSplits;
layout(location = 12) flat out uint vCascadeCount;
layout(location = 13) flat out vec4 vBaseColorFactor;
layout(location = 14) flat out uvec4 vTextureIndices;
layout(location = 15) flat out vec4 vEmissiveFactor;
layout(location = 16) flat out vec4 vMaterialParams;
// Depth along the *camera's* view axis, not the probe's. The shadow cascades are
// fitted to the camera frustum, so cascade selection has to ask where a surface
// is relative to the camera even when nothing is being rendered from there.
layout(location = 17) out float vCameraViewDepth;

void main()
{
    ObjectFrameData objectData = pc.objectFrameData.objects[gl_InstanceIndex & kObjectIndexMask];

    vec4 worldPosition = objectData.model * vec4(inPosition, 1.0);
    vWorldPosition = worldPosition.xyz;

    // Normal matrix from the model matrix. Non-uniform scale is rare in this
    // scene, and the capture only needs the normal for a lambert term.
    vNormal = normalize(mat3(objectData.model) * inNormal);
    vUV = inUV;

    vLightDirection = pc.frameConstants.values.lightDirection.xyz;
    vLightColor = pc.frameConstants.values.lightColor.rgb;
    vAmbientColor = pc.frameConstants.values.ambientColor.rgb;
    vShadowSettings = pc.frameConstants.values.shadowSettings;
    vCascadeSplits = pc.frameConstants.values.cascadeSplits;
    vCascadeCount = uint(pc.frameConstants.values.cameraForward.w + 0.5);
    vBaseColorFactor = objectData.baseColorFactor;
    vTextureIndices = objectData.textureIndices;
    vEmissiveFactor = objectData.emissiveFactor;
    vMaterialParams = objectData.materialParams;

    for (int cascade = 0; cascade < 4; ++cascade) {
        vLightSpacePosition[cascade] = objectData.lightMvp[cascade] * vec4(inPosition, 1.0);
    }

    vCameraViewDepth = dot(worldPosition.xyz - pc.frameConstants.values.cameraPosition.xyz, pc.frameConstants.values.cameraForward.xyz);

    gl_Position = pc.faceViewProjection * worldPosition;
}
