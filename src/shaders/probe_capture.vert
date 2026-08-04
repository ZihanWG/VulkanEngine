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

struct ObjectFrameData {
    mat4 mvp;
    mat4 model;
    mat4 lightMvp[4];
    vec4 lightDirection;
    vec4 lightColor;
    vec4 ambientColor;
    vec4 cascadeSplits;
    vec4 shadowSettings;
    vec4 baseColorFactor;
    vec4 materialParams;
    vec4 cameraPosition;
    vec4 cameraForward;
    uvec4 textureIndices;
    vec4 emissiveFactor;
    mat4 currMvpNoJitter;
    mat4 prevMvpNoJitter;
};

layout(buffer_reference, std430) readonly buffer ObjectFrameDataBuffer {
    ObjectFrameData objects[];
};

// The mat4 sits at offset 16, not 8: push-constant layout rounds the 8-byte
// buffer reference up to the matrix's 16-byte alignment. The C++
// ProbeCapturePushConstants mirror carries explicit padding to match.
layout(push_constant) uniform PushConstants {
    ObjectFrameDataBuffer objectFrameData;
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

    vLightDirection = objectData.lightDirection.xyz;
    vLightColor = objectData.lightColor.rgb;
    vAmbientColor = objectData.ambientColor.rgb;
    vShadowSettings = objectData.shadowSettings;
    vCascadeSplits = objectData.cascadeSplits;
    vCascadeCount = uint(objectData.cameraForward.w + 0.5);
    vBaseColorFactor = objectData.baseColorFactor;
    vTextureIndices = objectData.textureIndices;
    vEmissiveFactor = objectData.emissiveFactor;
    vMaterialParams = objectData.materialParams;

    for (int cascade = 0; cascade < 4; ++cascade) {
        vLightSpacePosition[cascade] = objectData.lightMvp[cascade] * vec4(inPosition, 1.0);
    }

    vCameraViewDepth = dot(worldPosition.xyz - objectData.cameraPosition.xyz, objectData.cameraForward.xyz);

    gl_Position = pc.faceViewProjection * worldPosition;
}
