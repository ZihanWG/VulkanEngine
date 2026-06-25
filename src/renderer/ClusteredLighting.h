#pragma once

#include "rhi/VulkanBuffer.h"

#include <cstddef>
#include <cstdint>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <vector>

namespace ve::rhi {
class VulkanContext;
}

namespace ve::renderer {

// Punctual light record shared with the lighting shaders. std430 stores each
// vec4 on a 16-byte boundary, so the four members give a 64-byte runtime-array
// stride that must match the GpuLight struct declared in the GLSL.
struct GpuLight {
    glm::vec4 positionRange{0.0f, 0.0f, 0.0f, 10.0f};  // xyz = world position, w = range
    glm::vec4 colorIntensity{1.0f, 1.0f, 1.0f, 1.0f};  // rgb = color, a = intensity
    glm::vec4 directionType{0.0f, -1.0f, 0.0f, 0.0f};  // xyz = spot direction, w = type
    glm::vec4 spotScaleOffset{-1.0f, 1.0f, 0.0f, 0.0f}; // x = cos(outer), y = 1/(cos(inner)-cos(outer))
};

static_assert(sizeof(GpuLight) == 64);
static_assert(offsetof(GpuLight, positionRange) == 0);
static_assert(offsetof(GpuLight, colorIntensity) == 16);
static_assert(offsetof(GpuLight, directionType) == 32);
static_assert(offsetof(GpuLight, spotScaleOffset) == 48);

enum class LightType : uint32_t {
    Point = 0,
    Spot = 1,
};

// Owns the CPU light list plus one host-visible, buffer-device-address light
// buffer per frame-in-flight. The main HDR fragment shader reads the buffer
// through a push-constant address, mirroring how ObjectFrameData is delivered,
// which keeps per-frame light updates free of descriptor-set hazards.
//
// Phase 1 evaluates every light per fragment (brute force). Phase 2 extends this
// subsystem with the froxel cluster grid and the GPU light-culling compute pass.
class ClusteredLighting final {
public:
    static constexpr uint32_t kMaxLights = 1024;

    void create(rhi::VulkanContext& context, uint32_t frameCount);

    [[nodiscard]] std::vector<GpuLight>& lights() { return lights_; }
    [[nodiscard]] const std::vector<GpuLight>& lights() const { return lights_; }
    [[nodiscard]] uint32_t lightCount() const;
    void clear() { lights_.clear(); }

    void addPointLight(const glm::vec3& position, const glm::vec3& color, float intensity, float range);
    void addSpotLight(const glm::vec3& position,
                      const glm::vec3& direction,
                      const glm::vec3& color,
                      float intensity,
                      float range,
                      float innerAngleRadians,
                      float outerAngleRadians);

    void upload(uint32_t frameIndex);
    [[nodiscard]] VkDeviceAddress lightBufferAddress(uint32_t frameIndex) const;

private:
    std::vector<GpuLight> lights_;
    std::vector<rhi::VulkanBuffer> lightBuffers_;
};

} // namespace ve::renderer
