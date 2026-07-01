#include "rhi/VulkanComputePipeline.h"

#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ve::rhi {

namespace {

std::vector<uint32_t> readSpirvFile(const std::filesystem::path& path)
{
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open shader file: " + path.string());
    }

    const auto fileSize = static_cast<size_t>(file.tellg());
    if (fileSize == 0 || (fileSize % sizeof(uint32_t)) != 0) {
        throw std::runtime_error("Shader file is empty or not valid SPIR-V: " + path.string());
    }

    std::vector<uint32_t> words(fileSize / sizeof(uint32_t));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(words.data()), static_cast<std::streamsize>(fileSize));
    if (!file) {
        throw std::runtime_error("Failed to read shader file: " + path.string());
    }

    return words;
}

} // namespace

VulkanComputePipeline::~VulkanComputePipeline()
{
    reset();
}

VulkanComputePipeline::VulkanComputePipeline(VulkanComputePipeline&& other) noexcept
{
    moveFrom(other);
}

VulkanComputePipeline& VulkanComputePipeline::operator=(VulkanComputePipeline&& other) noexcept
{
    if (this != &other) {
        reset();
        moveFrom(other);
    }

    return *this;
}

void VulkanComputePipeline::create(VkDevice device, const VulkanComputePipelineCreateInfo& createInfo)
{
    reset();

    device_ = device;
    shaderModule_ = createShaderModule(createInfo.shaderPath);

    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount = static_cast<uint32_t>(createInfo.descriptorSetLayouts.size());
    layoutInfo.pSetLayouts = createInfo.descriptorSetLayouts.data();
    layoutInfo.pushConstantRangeCount = static_cast<uint32_t>(createInfo.pushConstantRanges.size());
    layoutInfo.pPushConstantRanges = createInfo.pushConstantRanges.data();
    VK_CHECK(vkCreatePipelineLayout(device_, &layoutInfo, nullptr, &layout_));

    VkPipelineShaderStageCreateInfo shaderStage{};
    shaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    shaderStage.module = shaderModule_;
    shaderStage.pName = "main";

    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = shaderStage;
    pipelineInfo.layout = layout_;

    VK_CHECK(vkCreateComputePipelines(device_, createInfo.pipelineCache, 1, &pipelineInfo, nullptr, &pipeline_));
}

void VulkanComputePipeline::reset()
{
    if (pipeline_) {
        vkDestroyPipeline(device_, pipeline_, nullptr);
        pipeline_ = VK_NULL_HANDLE;
    }

    if (layout_) {
        vkDestroyPipelineLayout(device_, layout_, nullptr);
        layout_ = VK_NULL_HANDLE;
    }

    if (shaderModule_) {
        vkDestroyShaderModule(device_, shaderModule_, nullptr);
        shaderModule_ = VK_NULL_HANDLE;
    }

    device_ = VK_NULL_HANDLE;
}

VkShaderModule VulkanComputePipeline::createShaderModule(const std::filesystem::path& path) const
{
    const std::vector<uint32_t> code = readSpirvFile(path);

    VkShaderModuleCreateInfo moduleInfo{};
    moduleInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    moduleInfo.codeSize = code.size() * sizeof(uint32_t);
    moduleInfo.pCode = code.data();

    VkShaderModule shaderModule = VK_NULL_HANDLE;
    VK_CHECK(vkCreateShaderModule(device_, &moduleInfo, nullptr, &shaderModule));
    return shaderModule;
}

void VulkanComputePipeline::moveFrom(VulkanComputePipeline& other) noexcept
{
    device_ = std::exchange(other.device_, VK_NULL_HANDLE);
    shaderModule_ = std::exchange(other.shaderModule_, VK_NULL_HANDLE);
    layout_ = std::exchange(other.layout_, VK_NULL_HANDLE);
    pipeline_ = std::exchange(other.pipeline_, VK_NULL_HANDLE);
}

} // namespace ve::rhi
