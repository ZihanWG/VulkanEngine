#ifndef VE_PROBE_CUBE_GLSL
#define VE_PROBE_CUBE_GLSL

// The cube a probe gathers through: which direction each capture texel stands
// for, and how much of the sphere it covers. Shared by the convolution, which
// integrates the capture over these directions, and the ray-traced gather, which
// fires its rays along them -- so the traced capture lands in exactly the texels
// the convolution reads it back from, under the same sub-texel jitter. Needs
// irradiance_probes.glsl included first.

// Mirrors ve::renderer::pointShadowFaceDirection. Face order +X, -X, +Y, -Y,
// +Z, -Z, shared with the punctual shadow cube.
vec3 cubeFaceForward(int face)
{
    if (face == 0) { return vec3(1.0, 0.0, 0.0); }
    if (face == 1) { return vec3(-1.0, 0.0, 0.0); }
    if (face == 2) { return vec3(0.0, 1.0, 0.0); }
    if (face == 3) { return vec3(0.0, -1.0, 0.0); }
    if (face == 4) { return vec3(0.0, 0.0, 1.0); }
    return vec3(0.0, 0.0, -1.0);
}

// Mirrors ve::renderer::probeCubeTexelDirection. The basis is the one
// glm::lookAt builds inside the face projection, including the reference-axis
// switch on the two vertical faces; a CPU test projects these directions back
// through the real matrix to pin the two together.
vec3 probeCubeTexelDirection(int face, vec2 texel, vec2 jitter)
{
    float size = float(kProbeCaptureFaceResolution);
    // Jitter subtracted, matching ve::renderer::probeCubeTexelDirection: the
    // projection shifts the image, so the direction under texel x is the one
    // that was at x minus the shift.
    float u = (texel.x + 0.5 - jitter.x) / size * 2.0 - 1.0;
    float v = (texel.y + 0.5 - jitter.y) / size * 2.0 - 1.0;

    vec3 forward = cubeFaceForward(face);
    vec3 worldUp = abs(forward.y) > 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(0.0, 1.0, 0.0);
    vec3 right = normalize(cross(forward, worldUp));
    vec3 up = cross(right, forward);

    return normalize(forward + right * u + up * v);
}

// Mirrors ve::renderer::probeCubeTexelSolidAngle. Cube texels shrink toward a
// face's corners by up to a factor of three; without this the corners of every
// face are over-counted and the gathered irradiance is smoothly biased.
float probeCubeTexelSolidAngleAt(vec2 texel)
{
    float size = float(kProbeCaptureFaceResolution);
    float u = (texel.x + 0.5) / size * 2.0 - 1.0;
    float v = (texel.y + 0.5) / size * 2.0 - 1.0;
    float denominator = 1.0 + u * u + v * v;
    return (4.0 / (size * size)) / (denominator * sqrt(denominator));
}

// --- Rotated rays ---------------------------------------------------------------

// Mirrors ve::renderer::probeRayDirection: ray `ray` of a spherical Fibonacci set
// of `rayCount`, turned by the unit quaternion `rotation` (xyz vector, w scalar).
vec3 probeRayDirection(uint ray, uint rayCount, vec4 rotation)
{
    float count = float(max(rayCount, 1u));
    float index = float(min(ray, max(rayCount, 1u) - 1u));
    float turns = index * 0.6180339887;
    float phi = 6.2831853 * (turns - floor(turns));
    float cosTheta = 1.0 - (2.0 * index + 1.0) / count;
    float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
    vec3 direction = vec3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
    vec3 inner = cross(rotation.xyz, direction) + rotation.w * direction;
    return direction + 2.0 * cross(rotation.xyz, inner);
}

// Mirrors ve::renderer::probeRayTexel.
ivec2 probeRayTexel(uint slot, uint ray)
{
    return ivec2(int(ray % kProbeRayAtlasRowTexels),
                 int(slot) * kProbeCaptureFaceResolution + int(ray / kProbeRayAtlasRowTexels));
}

#endif // VE_PROBE_CUBE_GLSL
