#include "renderer/ClusteredLighting.h"

#include "rhi/VulkanContext.h"

#include <algorithm>
#include <cmath>
#include <glm/geometric.hpp>
#include <span>

namespace ve::renderer {

void ClusteredLighting::create(rhi::VulkanContext& context, uint32_t frameCount)
{
    lightBuffers_.clear();
    lightBuffers_.resize(frameCount);

    for (uint32_t frameIndex = 0; frameIndex < frameCount; ++frameIndex) {
        rhi::VulkanBufferCreateInfo bufferInfo{};
        bufferInfo.size = static_cast<VkDeviceSize>(kMaxLights * sizeof(GpuLight));
        bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        bufferInfo.memoryUsage = VMA_MEMORY_USAGE_AUTO;
        bufferInfo.allocationFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
        bufferInfo.requestDeviceAddress = true;
        lightBuffers_[frameIndex].createBuffer(context, bufferInfo);
    }
}

uint32_t ClusteredLighting::lightCount() const
{
    return static_cast<uint32_t>(std::min<size_t>(lights_.size(), kMaxLights));
}

void ClusteredLighting::addPointLight(const glm::vec3& position,
                                      const glm::vec3& color,
                                      float intensity,
                                      float range)
{
    GpuLight light{};
    light.positionRange = glm::vec4(position, range);
    light.colorIntensity = glm::vec4(color, intensity);
    light.directionType = glm::vec4(0.0f, -1.0f, 0.0f, static_cast<float>(LightType::Point));
    // cos(outer) = -1 keeps the spot cone fully open; the shader ignores it for
    // point lights, but a sane value avoids surprises if the type is changed.
    light.spotScaleOffset = glm::vec4(-1.0f, 1.0f, 0.0f, 0.0f);
    lights_.push_back(light);
}

void ClusteredLighting::addSpotLight(const glm::vec3& position,
                                     const glm::vec3& direction,
                                     const glm::vec3& color,
                                     float intensity,
                                     float range,
                                     float innerAngleRadians,
                                     float outerAngleRadians)
{
    const float cosInner = std::cos(innerAngleRadians);
    const float cosOuter = std::cos(outerAngleRadians);
    const float invDelta = 1.0f / std::max(cosInner - cosOuter, 1.0e-4f);

    GpuLight light{};
    light.positionRange = glm::vec4(position, range);
    light.colorIntensity = glm::vec4(color, intensity);
    light.directionType = glm::vec4(glm::normalize(direction), static_cast<float>(LightType::Spot));
    light.spotScaleOffset = glm::vec4(cosOuter, invDelta, 0.0f, 0.0f);
    lights_.push_back(light);
}

void ClusteredLighting::upload(uint32_t frameIndex)
{
    if (frameIndex >= lightBuffers_.size() || lights_.empty()) {
        return;
    }

    const size_t count = std::min<size_t>(lights_.size(), kMaxLights);
    lightBuffers_[frameIndex].upload(std::as_bytes(std::span<const GpuLight>(lights_.data(), count)));
}

VkDeviceAddress ClusteredLighting::lightBufferAddress(uint32_t frameIndex) const
{
    if (frameIndex >= lightBuffers_.size()) {
        return 0;
    }
    return lightBuffers_[frameIndex].deviceAddress();
}

} // namespace ve::renderer
