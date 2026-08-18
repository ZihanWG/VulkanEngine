#include "renderer/BrdfIntegration.h"

#include "core/JobSystem.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ve::renderer {

namespace {

constexpr float kPi = 3.14159265359f;
constexpr float kEpsilon = 0.0001f;

struct Vec2 {
    float x = 0.0f;
    float y = 0.0f;
};

struct Vec3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

Vec3 operator+(Vec3 lhs, Vec3 rhs)
{
    return {lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z};
}

Vec3 operator-(Vec3 lhs, Vec3 rhs)
{
    return {lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z};
}

Vec3 operator*(Vec3 value, float scale)
{
    return {value.x * scale, value.y * scale, value.z * scale};
}

float dot(Vec3 lhs, Vec3 rhs)
{
    return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

Vec3 cross(Vec3 lhs, Vec3 rhs)
{
    return {
        lhs.y * rhs.z - lhs.z * rhs.y,
        lhs.z * rhs.x - lhs.x * rhs.z,
        lhs.x * rhs.y - lhs.y * rhs.x
    };
}

Vec3 normalize(Vec3 value)
{
    const float length = std::sqrt(std::max(dot(value, value), 0.0f));
    if (length <= 0.0f) {
        return {0.0f, 0.0f, 1.0f};
    }

    const float inverseLength = 1.0f / length;
    return value * inverseLength;
}

float radicalInverseVdc(uint32_t bits)
{
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return static_cast<float>(bits) * 2.3283064365386963e-10f;
}

Vec2 hammersley(uint32_t index, uint32_t sampleCount)
{
    return {
        static_cast<float>(index) / static_cast<float>(sampleCount),
        radicalInverseVdc(index)
    };
}

Vec3 importanceSampleGgx(Vec2 xi, Vec3 normal, float roughness)
{
    const float alpha = roughness * roughness;
    const float phi = 2.0f * kPi * xi.x;
    const float cosTheta = std::sqrt(
        (1.0f - xi.y) / std::max(1.0f + (alpha * alpha - 1.0f) * xi.y, kEpsilon));
    const float sinTheta = std::sqrt(std::max(1.0f - cosTheta * cosTheta, 0.0f));

    const Vec3 halfwayTangent{
        std::cos(phi) * sinTheta,
        std::sin(phi) * sinTheta,
        cosTheta
    };

    const Vec3 up = std::abs(normal.z) < 0.999f ? Vec3{0.0f, 0.0f, 1.0f} : Vec3{1.0f, 0.0f, 0.0f};
    const Vec3 tangent = normalize(cross(up, normal));
    const Vec3 bitangent = cross(normal, tangent);

    return normalize(
        tangent * halfwayTangent.x
        + bitangent * halfwayTangent.y
        + normal * halfwayTangent.z);
}

float geometrySchlickGgx(float normalDirection, float roughness)
{
    const float alpha = roughness * roughness;
    const float k = alpha / 2.0f;
    return normalDirection / std::max(normalDirection * (1.0f - k) + k, kEpsilon);
}

float geometrySmith(float normalView, float normalLight, float roughness)
{
    return geometrySchlickGgx(normalView, roughness) * geometrySchlickGgx(normalLight, roughness);
}

uint8_t floatToByte(float value)
{
    return static_cast<uint8_t>(std::clamp(value * 255.0f, 0.0f, 255.0f));
}

// One row of the table. Split out so the serial and parallel paths run byte for
// byte the same code, rather than two loops that could drift apart.
void integrateRow(uint32_t y, uint32_t size, std::vector<uint8_t>& pixels)
{
    const float roughness = (static_cast<float>(y) + 0.5f) / static_cast<float>(size);
    for (uint32_t x = 0; x < size; ++x) {
        const float normalView = (static_cast<float>(x) + 0.5f) / static_cast<float>(size);
        const std::array<float, 2> brdf = integrateBrdf(normalView, roughness);
        const size_t offset = (static_cast<size_t>(y) * size + x) * kBrdfLutChannels;

        pixels[offset + 0] = floatToByte(brdf[0]);
        pixels[offset + 1] = floatToByte(brdf[1]);
    }
}

} // namespace

std::array<float, 2> integrateBrdf(float normalView, float roughness)
{
    const Vec3 normal{0.0f, 0.0f, 1.0f};
    const Vec3 viewDirection{
        std::sqrt(std::max(1.0f - normalView * normalView, 0.0f)),
        0.0f,
        normalView
    };

    float scale = 0.0f;
    float bias = 0.0f;

    for (uint32_t i = 0; i < kBrdfIntegrationSampleCount; ++i) {
        const Vec2 xi = hammersley(i, kBrdfIntegrationSampleCount);
        const Vec3 halfway = importanceSampleGgx(xi, normal, roughness);
        const Vec3 lightDirection = normalize(halfway * (2.0f * dot(viewDirection, halfway)) - viewDirection);

        const float normalLight = std::max(lightDirection.z, 0.0f);
        const float normalHalf = std::max(halfway.z, 0.0f);
        const float viewHalf = std::max(dot(viewDirection, halfway), 0.0f);

        if (normalLight > 0.0f) {
            const float geometry = geometrySmith(normalView, normalLight, roughness);
            const float geometryVisibility =
                geometry * viewHalf / std::max(normalHalf * normalView, kEpsilon);
            const float fresnel = std::pow(1.0f - viewHalf, 5.0f);

            scale += (1.0f - fresnel) * geometryVisibility;
            bias += fresnel * geometryVisibility;
        }
    }

    const float inverseSampleCount = 1.0f / static_cast<float>(kBrdfIntegrationSampleCount);
    return {scale * inverseSampleCount, bias * inverseSampleCount};
}

std::vector<uint8_t> makeSplitSumBrdfLut(uint32_t size, JobSystem* jobSystem)
{
    if (size == 0) {
        throw std::runtime_error("Cannot create a zero-sized BRDF LUT.");
    }

    std::vector<uint8_t> pixels(static_cast<size_t>(size) * size * kBrdfLutChannels);

    // Rows are the natural unit: each is `size` independent texels, and every
    // texel writes only its own two bytes, so chunks never overlap and need no
    // locking. Cost per row is uniform, unlike the LOD chains, so parallelFor's
    // equal contiguous chunks are the right fit here.
    if (jobSystem != nullptr && size > 1) {
        jobSystem->parallelFor(size, 1, [size, &pixels](size_t begin, size_t end) {
            for (size_t y = begin; y < end; ++y) {
                integrateRow(static_cast<uint32_t>(y), size, pixels);
            }
        });
    } else {
        for (uint32_t y = 0; y < size; ++y) {
            integrateRow(y, size, pixels);
        }
    }

    return pixels;
}

} // namespace ve::renderer
