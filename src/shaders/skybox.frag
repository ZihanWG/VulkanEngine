#version 460

layout(set = 0, binding = 0) uniform samplerCube uEnvironmentMap;
// Integrated volumetric fog. Sampled unconditionally: when fog is off the
// volume is held at its neutral clear (0 scattering, 1 transmittance), so this
// costs one fetch and changes nothing. The skybox push constants are already at
// the 128-byte minimum, so there is no room for an enable flag.
layout(set = 0, binding = 1) uniform sampler3D uFogVolume;

layout(push_constant) uniform SkyboxPushConstants {
    mat4 inverseViewProjection;
    mat4 prevViewProjection;
} pc;

layout(location = 0) in vec3 vDirection;
layout(location = 1) in vec2 vNdc;
layout(location = 0) out vec4 outColor;
layout(location = 1) out vec2 outVelocity;
layout(location = 2) out vec4 outNormalRoughness;

// Rotation-only sky reprojection: projecting the view direction with w = 0
// drops the camera translation, so the velocity tracks camera rotation only
// (correct for an infinitely distant sky).
vec2 computeSkyVelocity()
{
    vec4 prevClip = pc.prevViewProjection * vec4(normalize(vDirection), 0.0);
    if (prevClip.w <= 0.0) {
        return vec2(0.0);
    }
    vec2 prevNdc = prevClip.xy / prevClip.w;
    return (vNdc - prevNdc) * 0.5;
}

void main()
{
    vec3 environmentColor = texture(uEnvironmentMap, normalize(vDirection)).rgb;

    // The sky is infinitely far, so it takes the volume's far slice -- the total
    // fog along the whole view ray. Without this the world fogs out while the
    // sky behind it stays perfectly crisp, which reads as a hole in the fog.
    vec2 fogUv = vNdc * 0.5 + 0.5;
    vec4 fog = texture(uFogVolume, vec3(fogUv, 1.0));
    environmentColor = environmentColor * fog.a + fog.rgb;

    outColor = vec4(environmentColor, 1.0);
    outVelocity = computeSkyVelocity();
    // Sky pixels carry no surface: max roughness makes the SSR trace skip them
    // (it also rejects them by far-plane depth).
    outNormalRoughness = vec4(0.5, 0.5, 1.0, 0.0);
}
