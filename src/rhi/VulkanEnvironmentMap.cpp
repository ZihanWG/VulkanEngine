#include "rhi/VulkanEnvironmentMap.h"

#include "core/JobSystem.h"

#include "rhi/VulkanBuffer.h"
#include "rhi/VulkanCommandContext.h"
#include "rhi/VulkanContext.h"

#include <stb_image.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ve::rhi {

namespace {

constexpr uint32_t kCubeFaceCount = 6;
constexpr uint32_t kRgbaChannels = 4;
constexpr uint32_t kRgba8TexelBytes = 4;
constexpr uint32_t kRgba16fTexelBytes = 8;
constexpr uint32_t kRgba32fTexelBytes = 16;
constexpr uint32_t kSpecularPrefilterSampleCount = 96;
constexpr float kByteScale = 1.0f / 255.0f;
constexpr float kIrradianceScale = 0.55f;
constexpr float kPi = 3.14159265358979323846f;

struct FaceGradient {
    std::array<uint8_t, 3> horizon;
    std::array<uint8_t, 3> zenith;
};

struct Vec3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

struct StbiFloatDeleter {
    void operator()(float* pixels) const
    {
        stbi_image_free(pixels);
    }
};

using StbiFloatPixels = std::unique_ptr<float, StbiFloatDeleter>;

constexpr std::array<FaceGradient, kCubeFaceCount> kProceduralFaceGradients = {{
    {{{76, 92, 108}}, {{126, 150, 172}}},
    {{{72, 88, 102}}, {{116, 138, 158}}},
    {{{112, 136, 160}}, {{168, 190, 210}}},
    {{{54, 62, 70}}, {{84, 94, 104}}},
    {{{78, 94, 112}}, {{132, 156, 178}}},
    {{{58, 70, 86}}, {{104, 124, 148}}},
}};

constexpr std::array<Vec3, kCubeFaceCount> kCubeFaceDirections = {{
    {1.0f, 0.0f, 0.0f},
    {-1.0f, 0.0f, 0.0f},
    {0.0f, 1.0f, 0.0f},
    {0.0f, -1.0f, 0.0f},
    {0.0f, 0.0f, 1.0f},
    {0.0f, 0.0f, -1.0f},
}};

float lerpFloat(float a, float b, float t)
{
    return a * (1.0f - t) + b * t;
}

uint32_t calculateMipLevels(uint32_t faceSize)
{
    return static_cast<uint32_t>(std::floor(std::log2(static_cast<double>(faceSize)))) + 1;
}

uint32_t mipFaceSize(uint32_t faceSize, uint32_t mipLevel)
{
    return std::max(1u, faceSize >> mipLevel);
}

size_t mipFaceTexelCount(uint32_t faceSize, uint32_t mipLevel)
{
    const uint32_t size = mipFaceSize(faceSize, mipLevel);
    return static_cast<size_t>(size) * size;
}

size_t mipFaceByteCount(uint32_t faceSize, uint32_t mipLevel, size_t texelByteCount)
{
    return mipFaceTexelCount(faceSize, mipLevel) * texelByteCount;
}

size_t mipChainByteCount(uint32_t faceSize, uint32_t mipLevels, size_t texelByteCount)
{
    size_t byteCount = 0;
    for (uint32_t mipLevel = 0; mipLevel < mipLevels; ++mipLevel) {
        byteCount += mipFaceByteCount(faceSize, mipLevel, texelByteCount) * kCubeFaceCount;
    }
    return byteCount;
}

size_t mipChainFloatCount(uint32_t faceSize, uint32_t mipLevels)
{
    return mipChainByteCount(faceSize, mipLevels, kRgba32fTexelBytes) / sizeof(float);
}

size_t formatTexelByteCount(VkFormat format)
{
    switch (format) {
    case VK_FORMAT_R8G8B8A8_UNORM:
    case VK_FORMAT_R8G8B8A8_SRGB:
        return kRgba8TexelBytes;
    case VK_FORMAT_R16G16B16A16_SFLOAT:
        return kRgba16fTexelBytes;
    case VK_FORMAT_R32G32B32A32_SFLOAT:
        return kRgba32fTexelBytes;
    default:
        break;
    }

    throw std::runtime_error("Unsupported environment map texel format.");
}

uint8_t floatToByte(float value)
{
    return static_cast<uint8_t>(std::clamp(value * 255.0f, 0.0f, 255.0f));
}

float sanitizeHdrChannel(float value)
{
    if (!std::isfinite(value)) {
        return 0.0f;
    }

    return std::max(value, 0.0f);
}

Vec3 sanitizeHdrColor(Vec3 color)
{
    return {sanitizeHdrChannel(color.x), sanitizeHdrChannel(color.y), sanitizeHdrChannel(color.z)};
}

uint16_t floatToHalfBits(float value)
{
    value = std::clamp(sanitizeHdrChannel(value), 0.0f, 65504.0f);

    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));

    const uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mantissa = bits & 0x7fffffu;

    if (exponent <= 0) {
        if (exponent < -10) {
            return static_cast<uint16_t>(sign);
        }

        mantissa = (mantissa | 0x800000u) >> (1 - exponent);
        return static_cast<uint16_t>(sign | ((mantissa + 0x1000u) >> 13));
    }

    if (exponent >= 31) {
        return static_cast<uint16_t>(sign | 0x7bffu);
    }

    mantissa += 0x1000u;
    if ((mantissa & 0x800000u) != 0) {
        mantissa = 0;
        ++exponent;
        if (exponent >= 31) {
            return static_cast<uint16_t>(sign | 0x7bffu);
        }
    }

    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) | (mantissa >> 13));
}

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

Vec3 lerp(Vec3 a, Vec3 b, float t)
{
    return a * (1.0f - t) + b * t;
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
        lhs.x * rhs.y - lhs.y * rhs.x,
    };
}

Vec3 normalize(Vec3 value)
{
    const float length = std::sqrt(dot(value, value));
    if (length <= 0.0f) {
        return {0.0f, 1.0f, 0.0f};
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

std::array<float, 2> hammersley(uint32_t index, uint32_t sampleCount)
{
    return {
        static_cast<float>(index) / static_cast<float>(sampleCount),
        radicalInverseVdc(index),
    };
}

Vec3 importanceSampleGgx(const std::array<float, 2>& xi, Vec3 normal, float roughness)
{
    const float alpha = roughness * roughness;
    const float phi = 2.0f * kPi * xi[0];
    const float cosTheta =
        std::sqrt((1.0f - xi[1]) / std::max(1.0f + (alpha * alpha - 1.0f) * xi[1], 0.0001f));
    const float sinTheta = std::sqrt(std::max(1.0f - cosTheta * cosTheta, 0.0f));

    const Vec3 halfwayTangent{
        std::cos(phi) * sinTheta,
        std::sin(phi) * sinTheta,
        cosTheta,
    };

    const Vec3 up = std::abs(normal.y) < 0.999f ? Vec3{0.0f, 1.0f, 0.0f} : Vec3{1.0f, 0.0f, 0.0f};
    const Vec3 tangent = normalize(cross(up, normal));
    const Vec3 bitangent = cross(normal, tangent);
    return normalize(tangent * halfwayTangent.x + bitangent * halfwayTangent.y + normal * halfwayTangent.z);
}

Vec3 cubemapTexelDirection(uint32_t face, float u, float v)
{
    const float sc = u * 2.0f - 1.0f;
    const float tc = v * 2.0f - 1.0f;

    switch (face) {
    case 0:
        return normalize({1.0f, -tc, -sc});
    case 1:
        return normalize({-1.0f, -tc, sc});
    case 2:
        return normalize({sc, 1.0f, tc});
    case 3:
        return normalize({sc, -1.0f, -tc});
    case 4:
        return normalize({sc, -tc, 1.0f});
    case 5:
        return normalize({-sc, -tc, -1.0f});
    default:
        return {0.0f, 1.0f, 0.0f};
    }
}

struct CubemapSampleCoordinate {
    uint32_t face = 0;
    float u = 0.5f;
    float v = 0.5f;
};

CubemapSampleCoordinate cubemapSampleCoordinate(Vec3 direction)
{
    direction = normalize(direction);
    const float absX = std::abs(direction.x);
    const float absY = std::abs(direction.y);
    const float absZ = std::abs(direction.z);

    float sc = 0.0f;
    float tc = 0.0f;
    uint32_t face = 0;
    if (absX >= absY && absX >= absZ) {
        if (direction.x >= 0.0f) {
            face = 0;
            sc = -direction.z / absX;
            tc = -direction.y / absX;
        } else {
            face = 1;
            sc = direction.z / absX;
            tc = -direction.y / absX;
        }
    } else if (absY >= absZ) {
        if (direction.y >= 0.0f) {
            face = 2;
            sc = direction.x / absY;
            tc = direction.z / absY;
        } else {
            face = 3;
            sc = direction.x / absY;
            tc = -direction.z / absY;
        }
    } else {
        if (direction.z >= 0.0f) {
            face = 4;
            sc = direction.x / absZ;
            tc = -direction.y / absZ;
        } else {
            face = 5;
            sc = -direction.x / absZ;
            tc = -direction.y / absZ;
        }
    }

    return {
        face,
        std::clamp(sc * 0.5f + 0.5f, 0.0f, 1.0f),
        std::clamp(tc * 0.5f + 0.5f, 0.0f, 1.0f),
    };
}

Vec3 proceduralFaceColor(uint32_t face, float u, float v)
{
    const FaceGradient& gradient = kProceduralFaceGradients.at(face);
    const float sideShade = 0.9f + 0.1f * (1.0f - std::abs(u * 2.0f - 1.0f));

    return {
        lerpFloat(static_cast<float>(gradient.horizon[0]), static_cast<float>(gradient.zenith[0]) * sideShade, v) *
            kByteScale,
        lerpFloat(static_cast<float>(gradient.horizon[1]), static_cast<float>(gradient.zenith[1]) * sideShade, v) *
            kByteScale,
        lerpFloat(static_cast<float>(gradient.horizon[2]), static_cast<float>(gradient.zenith[2]) * sideShade, v) *
            kByteScale,
    };
}

Vec3 averageProceduralFaceColor(uint32_t face)
{
    constexpr uint32_t kSampleCount = 4;
    Vec3 color{};

    for (uint32_t y = 0; y < kSampleCount; ++y) {
        const float v = (static_cast<float>(y) + 0.5f) / static_cast<float>(kSampleCount);
        for (uint32_t x = 0; x < kSampleCount; ++x) {
            const float u = (static_cast<float>(x) + 0.5f) / static_cast<float>(kSampleCount);
            color = color + proceduralFaceColor(face, u, v);
        }
    }

    return color * (1.0f / static_cast<float>(kSampleCount * kSampleCount));
}

Vec3 sampleProceduralEnvironment(Vec3 direction)
{
    const CubemapSampleCoordinate coordinate = cubemapSampleCoordinate(direction);
    return proceduralFaceColor(coordinate.face, coordinate.u, coordinate.v);
}

void writeRgba(std::vector<uint8_t>& pixels, size_t offset, Vec3 color)
{
    pixels[offset + 0] = floatToByte(color.x);
    pixels[offset + 1] = floatToByte(color.y);
    pixels[offset + 2] = floatToByte(color.z);
    pixels[offset + 3] = 255;
}

void writeRgba32f(std::vector<float>& pixels, size_t offset, Vec3 color)
{
    color = sanitizeHdrColor(color);
    pixels[offset + 0] = color.x;
    pixels[offset + 1] = color.y;
    pixels[offset + 2] = color.z;
    pixels[offset + 3] = 1.0f;
}

uint32_t wrapTexelCoordinate(int32_t coordinate, uint32_t size)
{
    const int32_t signedSize = static_cast<int32_t>(size);
    int32_t wrapped = coordinate % signedSize;
    if (wrapped < 0) {
        wrapped += signedSize;
    }

    return static_cast<uint32_t>(wrapped);
}

Vec3 readRgba32fTexel(std::span<const float> pixels,
                      uint32_t faceSize,
                      uint32_t face,
                      uint32_t x,
                      uint32_t y)
{
    const size_t offset =
        ((static_cast<size_t>(face) * faceSize * faceSize) + (static_cast<size_t>(y) * faceSize + x)) *
        kRgbaChannels;
    return sanitizeHdrColor({pixels[offset + 0], pixels[offset + 1], pixels[offset + 2]});
}

Vec3 sampleRgba32fFace(std::span<const float> pixels, uint32_t faceSize, uint32_t face, float u, float v)
{
    const float clampedU = std::clamp(u, 0.0f, 1.0f);
    const float clampedV = std::clamp(v, 0.0f, 1.0f);
    const float sampleX = clampedU * static_cast<float>(faceSize - 1);
    const float sampleY = clampedV * static_cast<float>(faceSize - 1);
    const uint32_t x0 = static_cast<uint32_t>(std::floor(sampleX));
    const uint32_t y0 = static_cast<uint32_t>(std::floor(sampleY));
    const uint32_t x1 = std::min(x0 + 1, faceSize - 1);
    const uint32_t y1 = std::min(y0 + 1, faceSize - 1);
    const float tx = sampleX - static_cast<float>(x0);
    const float ty = sampleY - static_cast<float>(y0);

    const Vec3 c00 = readRgba32fTexel(pixels, faceSize, face, x0, y0);
    const Vec3 c10 = readRgba32fTexel(pixels, faceSize, face, x1, y0);
    const Vec3 c01 = readRgba32fTexel(pixels, faceSize, face, x0, y1);
    const Vec3 c11 = readRgba32fTexel(pixels, faceSize, face, x1, y1);

    return lerp(lerp(c00, c10, tx), lerp(c01, c11, tx), ty);
}

Vec3 sampleRgba32fCubemap(std::span<const float> pixels, uint32_t faceSize, Vec3 direction)
{
    const CubemapSampleCoordinate coordinate = cubemapSampleCoordinate(direction);
    return sampleRgba32fFace(pixels, faceSize, coordinate.face, coordinate.u, coordinate.v);
}

Vec3 readEquirectangularTexel(std::span<const float> pixels,
                              uint32_t width,
                              uint32_t x,
                              uint32_t y)
{
    const size_t offset = (static_cast<size_t>(y) * width + x) * kRgbaChannels;
    return sanitizeHdrColor({pixels[offset + 0], pixels[offset + 1], pixels[offset + 2]});
}

Vec3 sampleEquirectangular(std::span<const float> pixels, uint32_t width, uint32_t height, Vec3 direction)
{
    direction = normalize(direction);
    const float longitude = std::atan2(direction.z, direction.x);
    float u = longitude / (2.0f * kPi) + 0.5f;
    u -= std::floor(u);
    const float v = std::acos(std::clamp(direction.y, -1.0f, 1.0f)) / kPi;

    const float sampleX = u * static_cast<float>(width) - 0.5f;
    const float sampleY = v * static_cast<float>(height) - 0.5f;
    const int32_t x0 = static_cast<int32_t>(std::floor(sampleX));
    const int32_t y0 = static_cast<int32_t>(std::floor(sampleY));
    const uint32_t wrappedX0 = wrapTexelCoordinate(x0, width);
    const uint32_t wrappedX1 = wrapTexelCoordinate(x0 + 1, width);
    const uint32_t clampedY0 =
        static_cast<uint32_t>(std::clamp(y0, 0, static_cast<int32_t>(height - 1)));
    const uint32_t clampedY1 =
        static_cast<uint32_t>(std::clamp(y0 + 1, 0, static_cast<int32_t>(height - 1)));
    const float tx = sampleX - static_cast<float>(x0);
    const float ty = sampleY - static_cast<float>(y0);

    const Vec3 c00 = readEquirectangularTexel(pixels, width, wrappedX0, clampedY0);
    const Vec3 c10 = readEquirectangularTexel(pixels, width, wrappedX1, clampedY0);
    const Vec3 c01 = readEquirectangularTexel(pixels, width, wrappedX0, clampedY1);
    const Vec3 c11 = readEquirectangularTexel(pixels, width, wrappedX1, clampedY1);

    return lerp(lerp(c00, c10, tx), lerp(c01, c11, tx), ty);
}

void validateRgba32fFaceData(uint32_t faceSize, std::span<const float> pixels)
{
    if (faceSize == 0) {
        throw std::runtime_error("Cannot create a zero-sized HDR environment map.");
    }

    const size_t expectedFloatCount =
        static_cast<size_t>(faceSize) * faceSize * kCubeFaceCount * kRgbaChannels;
    if (pixels.size() != expectedFloatCount) {
        throw std::runtime_error("HDR environment cubemap RGBA32F data has the wrong float count.");
    }
}

std::vector<float> makeHdrEnvironmentFaces(std::span<const float> equirectangularPixels,
                                           uint32_t width,
                                           uint32_t height,
                                           uint32_t faceSize)
{
    if (width == 0 || height == 0 || faceSize == 0) {
        throw std::runtime_error("Cannot create an HDR environment cubemap from zero-sized image data.");
    }

    std::vector<float> pixels(static_cast<size_t>(faceSize) * faceSize * kCubeFaceCount * kRgbaChannels);

    for (uint32_t face = 0; face < kCubeFaceCount; ++face) {
        for (uint32_t y = 0; y < faceSize; ++y) {
            const float v = (static_cast<float>(y) + 0.5f) / static_cast<float>(faceSize);
            for (uint32_t x = 0; x < faceSize; ++x) {
                const float u = (static_cast<float>(x) + 0.5f) / static_cast<float>(faceSize);
                const Vec3 direction = cubemapTexelDirection(face, u, v);
                const Vec3 color = sampleEquirectangular(equirectangularPixels, width, height, direction);
                const size_t offset =
                    ((static_cast<size_t>(face) * faceSize * faceSize) + (static_cast<size_t>(y) * faceSize + x)) *
                    kRgbaChannels;
                writeRgba32f(pixels, offset, color);
            }
        }
    }

    return pixels;
}

std::vector<uint8_t> makeProceduralEnvironmentFaces(uint32_t faceSize)
{
    if (faceSize == 0) {
        throw std::runtime_error("Cannot create a zero-sized procedural environment map.");
    }

    std::vector<uint8_t> pixels(static_cast<size_t>(faceSize) * faceSize * kCubeFaceCount * kRgbaChannels);

    // Layer order follows Vulkan cube-map convention: +X, -X, +Y, -Y, +Z, -Z.
    for (uint32_t face = 0; face < kCubeFaceCount; ++face) {
        for (uint32_t y = 0; y < faceSize; ++y) {
            const float v = faceSize == 1 ? 1.0f : static_cast<float>(y) / static_cast<float>(faceSize - 1);
            for (uint32_t x = 0; x < faceSize; ++x) {
                const float u = faceSize == 1 ? 0.5f : static_cast<float>(x) / static_cast<float>(faceSize - 1);
                const size_t offset =
                    ((static_cast<size_t>(face) * faceSize * faceSize) + (static_cast<size_t>(y) * faceSize + x)) *
                    kRgbaChannels;

                writeRgba(pixels, offset, proceduralFaceColor(face, u, v));
            }
        }
    }

    return pixels;
}

std::vector<uint8_t> makeProceduralDiffuseIrradianceFaces(uint32_t faceSize)
{
    if (faceSize == 0) {
        throw std::runtime_error("Cannot create a zero-sized procedural diffuse irradiance map.");
    }

    std::array<Vec3, kCubeFaceCount> faceColors{};
    for (uint32_t face = 0; face < kCubeFaceCount; ++face) {
        faceColors.at(face) = averageProceduralFaceColor(face);
    }

    std::vector<uint8_t> pixels(static_cast<size_t>(faceSize) * faceSize * kCubeFaceCount * kRgbaChannels);

    for (uint32_t face = 0; face < kCubeFaceCount; ++face) {
        for (uint32_t y = 0; y < faceSize; ++y) {
            const float v = (static_cast<float>(y) + 0.5f) / static_cast<float>(faceSize);
            for (uint32_t x = 0; x < faceSize; ++x) {
                const float u = (static_cast<float>(x) + 0.5f) / static_cast<float>(faceSize);
                const Vec3 normal = cubemapTexelDirection(face, u, v);
                Vec3 irradiance{};
                float totalWeight = 0.0f;

                for (uint32_t sourceFace = 0; sourceFace < kCubeFaceCount; ++sourceFace) {
                    const float weight = std::max(dot(normal, kCubeFaceDirections.at(sourceFace)), 0.0f);
                    irradiance = irradiance + faceColors.at(sourceFace) * weight;
                    totalWeight += weight;
                }

                if (totalWeight > 0.0f) {
                    irradiance = irradiance * (kIrradianceScale / totalWeight);
                }

                const size_t offset =
                    ((static_cast<size_t>(face) * faceSize * faceSize) + (static_cast<size_t>(y) * faceSize + x)) *
                    kRgbaChannels;
                writeRgba(pixels, offset, irradiance);
            }
        }
    }

    return pixels;
}

template <typename SampleEnvironment>
Vec3 prefilterSpecularDirection(Vec3 reflectionDirection, float roughness, SampleEnvironment sampleEnvironment)
{
    const Vec3 normal = normalize(reflectionDirection);
    if (roughness <= 0.001f) {
        return sampleEnvironment(normal);
    }

    const Vec3 viewDirection = normal;
    Vec3 prefilteredColor{};
    float totalWeight = 0.0f;

    for (uint32_t sampleIndex = 0; sampleIndex < kSpecularPrefilterSampleCount; ++sampleIndex) {
        const std::array<float, 2> xi = hammersley(sampleIndex, kSpecularPrefilterSampleCount);
        const Vec3 halfway = importanceSampleGgx(xi, normal, roughness);
        const Vec3 lightDirection = normalize(halfway * (2.0f * dot(viewDirection, halfway)) - viewDirection);
        const float normalLight = std::max(dot(normal, lightDirection), 0.0f);

        if (normalLight > 0.0f) {
            prefilteredColor = prefilteredColor + sampleEnvironment(lightDirection) * normalLight;
            totalWeight += normalLight;
        }
    }

    if (totalWeight > 0.0f) {
        return prefilteredColor * (1.0f / totalWeight);
    }

    return sampleEnvironment(normal);
}

std::vector<uint8_t> makeProceduralPrefilteredSpecularFaces(uint32_t faceSize, JobSystem* jobSystem)
{
    if (faceSize == 0) {
        throw std::runtime_error("Cannot create a zero-sized procedural prefiltered environment map.");
    }

    const uint32_t mipLevels = calculateMipLevels(faceSize);

    std::vector<uint8_t> pixels(mipChainByteCount(faceSize, mipLevels, kRgba8TexelBytes));
    size_t mipOffset = 0;

    for (uint32_t mipLevel = 0; mipLevel < mipLevels; ++mipLevel) {
        const uint32_t size = mipFaceSize(faceSize, mipLevel);
        const float roughness =
            mipLevels == 1 ? 0.0f : static_cast<float>(mipLevel) / static_cast<float>(mipLevels - 1);

        // One row of one face. Every texel writes only its own bytes, so rows can
        // run in any order and the output is byte-identical either way; the mip
        // loop stays serial only because `mipOffset` walks the chain.
        const auto integrateRow = [&](size_t rowIndex) {
            const auto face = static_cast<uint32_t>(rowIndex / size);
            const auto y = static_cast<uint32_t>(rowIndex % size);
            const float v = size == 1 ? 0.5f : static_cast<float>(y) / static_cast<float>(size - 1);
            for (uint32_t x = 0; x < size; ++x) {
                const float u = size == 1 ? 0.5f : static_cast<float>(x) / static_cast<float>(size - 1);
                const Vec3 direction = cubemapTexelDirection(face, u, v);
                const Vec3 color = prefilterSpecularDirection(
                    direction,
                    roughness,
                    [](Vec3 sampleDirection) { return sampleProceduralEnvironment(sampleDirection); });
                const size_t offset =
                    mipOffset + ((static_cast<size_t>(face) * size * size) + (static_cast<size_t>(y) * size + x)) *
                                    kRgbaChannels;
                writeRgba(pixels, offset, color);
            }
        };

        const size_t rowCount = static_cast<size_t>(kCubeFaceCount) * size;
        if (jobSystem != nullptr && rowCount > 1) {
            jobSystem->parallelFor(rowCount, 1, [&integrateRow](size_t begin, size_t end) {
                for (size_t row = begin; row < end; ++row) {
                    integrateRow(row);
                }
            });
        } else {
            for (size_t row = 0; row < rowCount; ++row) {
                integrateRow(row);
            }
        }

        mipOffset += mipFaceByteCount(faceSize, mipLevel, kRgba8TexelBytes) * kCubeFaceCount;
    }

    return pixels;
}

Vec3 averageRgba32fFaceColor(uint32_t face, uint32_t sourceFaceSize, std::span<const float> sourcePixels)
{
    Vec3 color{};
    const float sampleScale = 1.0f / static_cast<float>(sourceFaceSize * sourceFaceSize);

    for (uint32_t y = 0; y < sourceFaceSize; ++y) {
        for (uint32_t x = 0; x < sourceFaceSize; ++x) {
            color = color + readRgba32fTexel(sourcePixels, sourceFaceSize, face, x, y);
        }
    }

    return color * sampleScale;
}

std::vector<float> makeHdrDiffuseIrradianceFaces(uint32_t sourceFaceSize,
                                                 std::span<const float> sourcePixels,
                                                 uint32_t faceSize)
{
    validateRgba32fFaceData(sourceFaceSize, sourcePixels);
    if (faceSize == 0) {
        throw std::runtime_error("Cannot create a zero-sized HDR diffuse irradiance map.");
    }

    std::array<Vec3, kCubeFaceCount> faceColors{};
    for (uint32_t face = 0; face < kCubeFaceCount; ++face) {
        faceColors.at(face) = averageRgba32fFaceColor(face, sourceFaceSize, sourcePixels);
    }

    std::vector<float> pixels(static_cast<size_t>(faceSize) * faceSize * kCubeFaceCount * kRgbaChannels);

    for (uint32_t face = 0; face < kCubeFaceCount; ++face) {
        for (uint32_t y = 0; y < faceSize; ++y) {
            const float v = (static_cast<float>(y) + 0.5f) / static_cast<float>(faceSize);
            for (uint32_t x = 0; x < faceSize; ++x) {
                const float u = (static_cast<float>(x) + 0.5f) / static_cast<float>(faceSize);
                const Vec3 normal = cubemapTexelDirection(face, u, v);
                Vec3 irradiance{};
                float totalWeight = 0.0f;

                for (uint32_t sourceFace = 0; sourceFace < kCubeFaceCount; ++sourceFace) {
                    const float weight = std::max(dot(normal, kCubeFaceDirections.at(sourceFace)), 0.0f);
                    irradiance = irradiance + faceColors.at(sourceFace) * weight;
                    totalWeight += weight;
                }

                if (totalWeight > 0.0f) {
                    irradiance = irradiance * (kIrradianceScale / totalWeight);
                }

                const size_t offset =
                    ((static_cast<size_t>(face) * faceSize * faceSize) + (static_cast<size_t>(y) * faceSize + x)) *
                    kRgbaChannels;
                writeRgba32f(pixels, offset, irradiance);
            }
        }
    }

    return pixels;
}

std::vector<float> makeHdrPrefilteredSpecularFaces(uint32_t sourceFaceSize,
                                                   std::span<const float> sourcePixels,
                                                   uint32_t faceSize)
{
    validateRgba32fFaceData(sourceFaceSize, sourcePixels);
    if (faceSize == 0) {
        throw std::runtime_error("Cannot create a zero-sized HDR prefiltered environment map.");
    }

    const uint32_t mipLevels = calculateMipLevels(faceSize);

    std::vector<float> pixels(mipChainFloatCount(faceSize, mipLevels));
    size_t mipOffset = 0;

    for (uint32_t mipLevel = 0; mipLevel < mipLevels; ++mipLevel) {
        const uint32_t size = mipFaceSize(faceSize, mipLevel);
        const float roughness =
            mipLevels == 1 ? 0.0f : static_cast<float>(mipLevel) / static_cast<float>(mipLevels - 1);

        for (uint32_t face = 0; face < kCubeFaceCount; ++face) {
            for (uint32_t y = 0; y < size; ++y) {
                const float v = size == 1 ? 0.5f : static_cast<float>(y) / static_cast<float>(size - 1);
                for (uint32_t x = 0; x < size; ++x) {
                    const float u = size == 1 ? 0.5f : static_cast<float>(x) / static_cast<float>(size - 1);
                    const Vec3 direction = cubemapTexelDirection(face, u, v);
                    const Vec3 color = prefilterSpecularDirection(
                        direction,
                        roughness,
                        [sourcePixels, sourceFaceSize](Vec3 sampleDirection) {
                            return sampleRgba32fCubemap(sourcePixels, sourceFaceSize, sampleDirection);
                        });
                    const size_t offset =
                        mipOffset + ((static_cast<size_t>(face) * size * size) + (static_cast<size_t>(y) * size + x)) *
                                        kRgbaChannels;
                    writeRgba32f(pixels, offset, color);
                }
            }
        }

        mipOffset += mipFaceTexelCount(faceSize, mipLevel) * kCubeFaceCount * kRgbaChannels;
    }

    return pixels;
}

bool supportsEnvironmentFormat(VkPhysicalDevice physicalDevice, VkFormat format)
{
    VkFormatProperties properties{};
    vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &properties);

    constexpr VkFormatFeatureFlags requiredFeatures =
        VK_FORMAT_FEATURE_TRANSFER_DST_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
    return (properties.optimalTilingFeatures & requiredFeatures) == requiredFeatures;
}

void validateEnvironmentFormatSupport(VkPhysicalDevice physicalDevice, VkFormat format)
{
    if (!supportsEnvironmentFormat(physicalDevice, format)) {
        throw std::runtime_error("Environment map format does not support sampled cube image uploads.");
    }
}

VkFormat chooseHdrEnvironmentFormat(VkPhysicalDevice physicalDevice)
{
    if (supportsEnvironmentFormat(physicalDevice, VK_FORMAT_R16G16B16A16_SFLOAT)) {
        return VK_FORMAT_R16G16B16A16_SFLOAT;
    }
    if (supportsEnvironmentFormat(physicalDevice, VK_FORMAT_R32G32B32A32_SFLOAT)) {
        return VK_FORMAT_R32G32B32A32_SFLOAT;
    }

    throw std::runtime_error("No supported sampled float format was found for HDR environment maps.");
}

bool supportsSampledImageLinearFiltering(VkPhysicalDevice physicalDevice, VkFormat format)
{
    VkFormatProperties properties{};
    vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &properties);
    return (properties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT) != 0;
}

VkImageMemoryBarrier2 environmentBarrier(VkImage image,
                                         VkImageLayout oldLayout,
                                         VkImageLayout newLayout,
                                         VkPipelineStageFlags2 srcStageMask,
                                         VkAccessFlags2 srcAccessMask,
                                         VkPipelineStageFlags2 dstStageMask,
                                         VkAccessFlags2 dstAccessMask,
                                         uint32_t levelCount)
{
    VkImageMemoryBarrier2 barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    barrier.srcStageMask = srcStageMask;
    barrier.srcAccessMask = srcAccessMask;
    barrier.dstStageMask = dstStageMask;
    barrier.dstAccessMask = dstAccessMask;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = levelCount;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = kCubeFaceCount;
    return barrier;
}

void recordImageBarrier(VkCommandBuffer commandBuffer, const VkImageMemoryBarrier2& barrier)
{
    VkDependencyInfo dependencyInfo{};
    dependencyInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependencyInfo.imageMemoryBarrierCount = 1;
    dependencyInfo.pImageMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(commandBuffer, &dependencyInfo);
}

} // namespace

VulkanEnvironmentMap::~VulkanEnvironmentMap()
{
    reset();
}

VulkanEnvironmentMap::VulkanEnvironmentMap(VulkanEnvironmentMap&& other) noexcept
{
    moveFrom(other);
}

VulkanEnvironmentMap& VulkanEnvironmentMap::operator=(VulkanEnvironmentMap&& other) noexcept
{
    if (this != &other) {
        reset();
        moveFrom(other);
    }

    return *this;
}

HdrEnvironmentCubeData VulkanEnvironmentMap::loadHdrEquirectangularFaces(const std::filesystem::path& path,
                                                                         uint32_t faceSize)
{
    if (faceSize == 0) {
        throw std::runtime_error("Cannot create a zero-sized HDR environment cubemap.");
    }

    int loadedWidth = 0;
    int loadedHeight = 0;
    int sourceChannels = 0;
    const std::string filename = path.string();
    StbiFloatPixels loadedPixels(
        stbi_loadf(filename.c_str(), &loadedWidth, &loadedHeight, &sourceChannels, STBI_rgb_alpha));

    if (!loadedPixels) {
        const char* failureReason = stbi_failure_reason();
        throw std::runtime_error("Failed to load HDR environment '" + filename +
                                 "': " + (failureReason ? failureReason : "unknown stb_image error"));
    }
    if (loadedWidth <= 0 || loadedHeight <= 0) {
        throw std::runtime_error("HDR environment has invalid dimensions: " + filename);
    }
    if (sourceChannels < 3) {
        throw std::runtime_error("HDR environment must contain at least RGB channels: " + filename);
    }

    const uint32_t width = static_cast<uint32_t>(loadedWidth);
    const uint32_t height = static_cast<uint32_t>(loadedHeight);
    const size_t floatCount = static_cast<size_t>(width) * height * kRgbaChannels;
    std::span<const float> equirectangularPixels(loadedPixels.get(), floatCount);

    HdrEnvironmentCubeData cubeData{};
    cubeData.faceSize = faceSize;
    cubeData.rgba32fPixels = makeHdrEnvironmentFaces(equirectangularPixels, width, height, faceSize);
    return cubeData;
}

void VulkanEnvironmentMap::createProcedural(VulkanContext& context,
                                            const VulkanCommandContext& commandContext,
                                            uint32_t faceSize)
{
    const std::vector<uint8_t> pixels = makeProceduralEnvironmentFaces(faceSize);
    createFromRgba8Faces(context,
                         commandContext,
                         faceSize,
                         std::span<const uint8_t>(pixels.data(), pixels.size()),
                         VK_FORMAT_R8G8B8A8_UNORM);
}

void VulkanEnvironmentMap::createProceduralDiffuseIrradiance(VulkanContext& context,
                                                             const VulkanCommandContext& commandContext,
                                                             uint32_t faceSize)
{
    const std::vector<uint8_t> pixels = makeProceduralDiffuseIrradianceFaces(faceSize);
    createFromRgba8Faces(context,
                         commandContext,
                         faceSize,
                         std::span<const uint8_t>(pixels.data(), pixels.size()),
                         VK_FORMAT_R8G8B8A8_UNORM);
}

void VulkanEnvironmentMap::createProceduralPrefilteredSpecular(VulkanContext& context,
                                                               const VulkanCommandContext& commandContext,
                                                               uint32_t faceSize,
                                                               JobSystem* jobSystem)
{
    const std::vector<uint8_t> pixels = makeProceduralPrefilteredSpecularFaces(faceSize, jobSystem);
    createFromRgba8MipFaces(context,
                            commandContext,
                            faceSize,
                            calculateMipLevels(faceSize),
                            std::span<const uint8_t>(pixels.data(), pixels.size()),
                            VK_FORMAT_R8G8B8A8_UNORM);
}

void VulkanEnvironmentMap::createFromRgba8Faces(VulkanContext& context,
                                                const VulkanCommandContext& commandContext,
                                                uint32_t faceSize,
                                                std::span<const uint8_t> pixels,
                                                VkFormat format)
{
    createFromRgba8MipFaces(context, commandContext, faceSize, 1, pixels, format);
}

void VulkanEnvironmentMap::createFromRgba8MipFaces(VulkanContext& context,
                                                   const VulkanCommandContext& commandContext,
                                                   uint32_t faceSize,
                                                   uint32_t mipLevels,
                                                   std::span<const uint8_t> pixels,
                                                   VkFormat format)
{
    reset();

    if (faceSize == 0) {
        throw std::runtime_error("Cannot create a zero-sized environment map.");
    }
    if (mipLevels == 0) {
        throw std::runtime_error("Cannot create an environment map with zero mip levels.");
    }

    const size_t expectedByteCount = mipChainByteCount(faceSize, mipLevels, kRgba8TexelBytes);
    if (pixels.size_bytes() != expectedByteCount) {
        throw std::runtime_error("Environment cubemap RGBA8 mip data has the wrong byte count.");
    }

    context_ = &context;
    faceSize_ = faceSize;
    mipLevels_ = mipLevels;
    format_ = format;
    layout_ = VK_IMAGE_LAYOUT_UNDEFINED;

    validateEnvironmentFormatSupport(context.physicalDevice(), format_);

    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = format_;
    imageInfo.extent = {faceSize_, faceSize_, 1};
    imageInfo.mipLevels = mipLevels_;
    imageInfo.arrayLayers = kCubeFaceCount;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo allocationInfo{};
    allocationInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    allocationInfo.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    VK_CHECK(vmaCreateImage(context.allocator(), &imageInfo, &allocationInfo, &image_, &allocation_, nullptr));

    uploadMipFaces(context, commandContext, std::as_bytes(pixels));
    createImageView();
    createSampler();
}

void VulkanEnvironmentMap::createFromHdrEquirectangular(VulkanContext& context,
                                                        const VulkanCommandContext& commandContext,
                                                        const std::filesystem::path& path,
                                                        uint32_t faceSize)
{
    const HdrEnvironmentCubeData cubeData = loadHdrEquirectangularFaces(path, faceSize);
    createFromRgba32fFaces(context,
                           commandContext,
                           cubeData.faceSize,
                           std::span<const float>(cubeData.rgba32fPixels.data(), cubeData.rgba32fPixels.size()));
}

void VulkanEnvironmentMap::createFromRgba32fFaces(VulkanContext& context,
                                                  const VulkanCommandContext& commandContext,
                                                  uint32_t faceSize,
                                                  std::span<const float> pixels)
{
    createFromRgba32fMipFaces(context, commandContext, faceSize, 1, pixels);
}

void VulkanEnvironmentMap::createFromRgba32fMipFaces(VulkanContext& context,
                                                     const VulkanCommandContext& commandContext,
                                                     uint32_t faceSize,
                                                     uint32_t mipLevels,
                                                     std::span<const float> pixels)
{
    reset();

    if (faceSize == 0) {
        throw std::runtime_error("Cannot create a zero-sized HDR environment map.");
    }
    if (mipLevels == 0) {
        throw std::runtime_error("Cannot create an HDR environment map with zero mip levels.");
    }

    const size_t expectedFloatCount = mipChainFloatCount(faceSize, mipLevels);
    if (pixels.size() != expectedFloatCount) {
        throw std::runtime_error("HDR environment cubemap RGBA32F mip data has the wrong float count.");
    }

    context_ = &context;
    faceSize_ = faceSize;
    mipLevels_ = mipLevels;
    format_ = chooseHdrEnvironmentFormat(context.physicalDevice());
    layout_ = VK_IMAGE_LAYOUT_UNDEFINED;

    std::vector<uint16_t> rgba16fPixels;
    std::vector<float> rgba32fPixels;
    std::span<const std::byte> uploadPixels;
    if (format_ == VK_FORMAT_R16G16B16A16_SFLOAT) {
        rgba16fPixels.resize(pixels.size());
        for (size_t index = 0; index < pixels.size(); ++index) {
            rgba16fPixels[index] = floatToHalfBits(pixels[index]);
        }
        uploadPixels = std::as_bytes(std::span<const uint16_t>(rgba16fPixels.data(), rgba16fPixels.size()));
    } else {
        rgba32fPixels.resize(pixels.size());
        for (size_t index = 0; index < pixels.size(); ++index) {
            rgba32fPixels[index] = sanitizeHdrChannel(pixels[index]);
        }
        uploadPixels = std::as_bytes(std::span<const float>(rgba32fPixels.data(), rgba32fPixels.size()));
    }

    const size_t expectedByteCount = mipChainByteCount(faceSize, mipLevels, formatTexelByteCount(format_));
    if (uploadPixels.size_bytes() != expectedByteCount) {
        throw std::runtime_error("HDR environment upload data has the wrong byte count for the selected format.");
    }

    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = format_;
    imageInfo.extent = {faceSize_, faceSize_, 1};
    imageInfo.mipLevels = mipLevels_;
    imageInfo.arrayLayers = kCubeFaceCount;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo allocationInfo{};
    allocationInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    allocationInfo.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    VK_CHECK(vmaCreateImage(context.allocator(), &imageInfo, &allocationInfo, &image_, &allocation_, nullptr));

    uploadMipFaces(context, commandContext, uploadPixels);
    createImageView();
    createSampler();
}

void VulkanEnvironmentMap::createDiffuseIrradianceFromRgba32fFaces(VulkanContext& context,
                                                                   const VulkanCommandContext& commandContext,
                                                                   uint32_t sourceFaceSize,
                                                                   std::span<const float> sourcePixels,
                                                                   uint32_t faceSize)
{
    const std::vector<float> pixels = makeHdrDiffuseIrradianceFaces(sourceFaceSize, sourcePixels, faceSize);
    createFromRgba32fFaces(context,
                           commandContext,
                           faceSize,
                           std::span<const float>(pixels.data(), pixels.size()));
}

void VulkanEnvironmentMap::createPrefilteredSpecularFromRgba32fFaces(VulkanContext& context,
                                                                     const VulkanCommandContext& commandContext,
                                                                     uint32_t sourceFaceSize,
                                                                     std::span<const float> sourcePixels,
                                                                     uint32_t faceSize)
{
    const uint32_t mipLevels = calculateMipLevels(faceSize);
    const std::vector<float> pixels = makeHdrPrefilteredSpecularFaces(sourceFaceSize, sourcePixels, faceSize);
    createFromRgba32fMipFaces(context,
                              commandContext,
                              faceSize,
                              mipLevels,
                              std::span<const float>(pixels.data(), pixels.size()));
}

void VulkanEnvironmentMap::reset()
{
    if (!context_) {
        return;
    }

    if (sampler_) {
        vkDestroySampler(context_->vkDevice(), sampler_, nullptr);
        sampler_ = VK_NULL_HANDLE;
    }

    if (imageView_) {
        vkDestroyImageView(context_->vkDevice(), imageView_, nullptr);
        imageView_ = VK_NULL_HANDLE;
    }

    if (image_) {
        vmaDestroyImage(context_->allocator(), image_, allocation_);
        image_ = VK_NULL_HANDLE;
        allocation_ = VK_NULL_HANDLE;
    }

    context_ = nullptr;
    faceSize_ = 0;
    mipLevels_ = 0;
    format_ = VK_FORMAT_UNDEFINED;
    layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
}

void VulkanEnvironmentMap::uploadMipFaces(VulkanContext& context,
                                          const VulkanCommandContext& commandContext,
                                          std::span<const std::byte> pixels)
{
    VulkanBuffer stagingBuffer;
    VulkanBufferCreateInfo stagingInfo{};
    stagingInfo.size = static_cast<VkDeviceSize>(pixels.size_bytes());
    stagingInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    stagingInfo.memoryUsage = VMA_MEMORY_USAGE_AUTO;
    stagingInfo.allocationFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
    stagingBuffer.createBuffer(context, stagingInfo);
    stagingBuffer.upload(pixels);

    VkCommandBufferAllocateInfo allocateInfo{};
    allocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocateInfo.commandPool = commandContext.commandPool();
    allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocateInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateCommandBuffers(context.vkDevice(), &allocateInfo, &commandBuffer));

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(commandBuffer, &beginInfo));

    const VkImageMemoryBarrier2 toTransfer = environmentBarrier(image_,
                                                                VK_IMAGE_LAYOUT_UNDEFINED,
                                                                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                                                VK_PIPELINE_STAGE_2_NONE,
                                                                VK_ACCESS_2_NONE,
                                                                VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                                                VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                                                mipLevels_);
    recordImageBarrier(commandBuffer, toTransfer);

    std::vector<VkBufferImageCopy> copyRegions;
    copyRegions.reserve(static_cast<size_t>(mipLevels_) * kCubeFaceCount);

    VkDeviceSize bufferOffset = 0;
    const size_t texelByteCount = formatTexelByteCount(format_);
    for (uint32_t mipLevel = 0; mipLevel < mipLevels_; ++mipLevel) {
        const uint32_t size = mipFaceSize(faceSize_, mipLevel);
        const VkDeviceSize faceByteSize =
            static_cast<VkDeviceSize>(mipFaceByteCount(faceSize_, mipLevel, texelByteCount));

        for (uint32_t face = 0; face < kCubeFaceCount; ++face) {
            VkBufferImageCopy copyRegion{};
            copyRegion.bufferOffset = bufferOffset + faceByteSize * face;
            copyRegion.bufferRowLength = 0;
            copyRegion.bufferImageHeight = 0;
            copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            copyRegion.imageSubresource.mipLevel = mipLevel;
            copyRegion.imageSubresource.baseArrayLayer = face;
            copyRegion.imageSubresource.layerCount = 1;
            copyRegion.imageOffset = {0, 0, 0};
            copyRegion.imageExtent = {size, size, 1};
            copyRegions.push_back(copyRegion);
        }

        bufferOffset += faceByteSize * kCubeFaceCount;
    }

    vkCmdCopyBufferToImage(commandBuffer,
                           stagingBuffer.buffer(),
                           image_,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                           static_cast<uint32_t>(copyRegions.size()),
                           copyRegions.data());

    const VkImageMemoryBarrier2 toShaderRead = environmentBarrier(image_,
                                                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                                                  VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                                                  VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                                                  VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                                                                  VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                                                                  mipLevels_);
    recordImageBarrier(commandBuffer, toShaderRead);

    VK_CHECK(vkEndCommandBuffer(commandBuffer));

    VkCommandBufferSubmitInfo commandBufferInfo{};
    commandBufferInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
    commandBufferInfo.commandBuffer = commandBuffer;

    VkSubmitInfo2 submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
    submitInfo.commandBufferInfoCount = 1;
    submitInfo.pCommandBufferInfos = &commandBufferInfo;

    VK_CHECK(vkQueueSubmit2(context.graphicsQueue(), 1, &submitInfo, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(context.graphicsQueue()));
    vkFreeCommandBuffers(context.vkDevice(), commandContext.commandPool(), 1, &commandBuffer);

    layout_ = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
}

void VulkanEnvironmentMap::createImageView()
{
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = image_;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
    viewInfo.format = format_;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = mipLevels_;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = kCubeFaceCount;

    VK_CHECK(vkCreateImageView(context_->vkDevice(), &viewInfo, nullptr, &imageView_));
}

void VulkanEnvironmentMap::createSampler()
{
    const bool linearFiltering = supportsSampledImageLinearFiltering(context_->physicalDevice(), format_);

    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = linearFiltering ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
    samplerInfo.minFilter = linearFiltering ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
    samplerInfo.mipmapMode = linearFiltering ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.maxAnisotropy = 1.0f;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = mipLevels_ > 0 ? static_cast<float>(mipLevels_ - 1) : 0.0f;
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;

    VK_CHECK(vkCreateSampler(context_->vkDevice(), &samplerInfo, nullptr, &sampler_));
}

void VulkanEnvironmentMap::moveFrom(VulkanEnvironmentMap& other) noexcept
{
    context_ = std::exchange(other.context_, nullptr);
    image_ = std::exchange(other.image_, VK_NULL_HANDLE);
    allocation_ = std::exchange(other.allocation_, VK_NULL_HANDLE);
    imageView_ = std::exchange(other.imageView_, VK_NULL_HANDLE);
    sampler_ = std::exchange(other.sampler_, VK_NULL_HANDLE);
    faceSize_ = std::exchange(other.faceSize_, 0);
    mipLevels_ = std::exchange(other.mipLevels_, 0);
    format_ = std::exchange(other.format_, VK_FORMAT_UNDEFINED);
    layout_ = std::exchange(other.layout_, VK_IMAGE_LAYOUT_UNDEFINED);
}

} // namespace ve::rhi
