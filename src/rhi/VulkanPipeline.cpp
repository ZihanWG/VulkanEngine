#include "rhi/VulkanPipeline.h"

#include <array>
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

VulkanPipeline::~VulkanPipeline()
{
    reset();
}

VulkanPipeline::VulkanPipeline(VulkanPipeline&& other) noexcept
{
    moveFrom(other);
}

VulkanPipeline& VulkanPipeline::operator=(VulkanPipeline&& other) noexcept
{
    if (this != &other) {
        reset();
        moveFrom(other);
    }

    return *this;
}

void VulkanPipeline::create(VkDevice device, const VulkanPipelineCreateInfo& createInfo)
{
    reset();

    if (createInfo.colorFormat == VK_FORMAT_UNDEFINED) {
        throw std::runtime_error("Cannot create graphics pipeline with an undefined color format.");
    }

    device_ = device;
    vertexShaderModule_ = createShaderModule(createInfo.vertexShaderPath);
    fragmentShaderModule_ = createShaderModule(createInfo.fragmentShaderPath);

    VkPipelineShaderStageCreateInfo vertexStage{};
    vertexStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertexStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertexStage.module = vertexShaderModule_;
    vertexStage.pName = "main";

    VkPipelineShaderStageCreateInfo fragmentStage{};
    fragmentStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragmentStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragmentStage.module = fragmentShaderModule_;
    fragmentStage.pName = "main";

    const std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages = {
        vertexStage,
        fragmentStage
    };

    VkPipelineVertexInputStateCreateInfo vertexInput{};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInput.vertexBindingDescriptionCount = static_cast<uint32_t>(createInfo.vertexBindings.size());
    vertexInput.pVertexBindingDescriptions = createInfo.vertexBindings.data();
    vertexInput.vertexAttributeDescriptionCount = static_cast<uint32_t>(createInfo.vertexAttributes.size());
    vertexInput.pVertexAttributeDescriptions = createInfo.vertexAttributes.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterization{};
    rasterization.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterization.polygonMode = VK_POLYGON_MODE_FILL;
    rasterization.cullMode = VK_CULL_MODE_NONE;
    rasterization.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterization.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = createInfo.enableDepth ? VK_TRUE : VK_FALSE;
    depthStencil.depthWriteEnable = createInfo.enableDepth ? VK_TRUE : VK_FALSE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;

    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT
        | VK_COLOR_COMPONENT_G_BIT
        | VK_COLOR_COMPONENT_B_BIT
        | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo colorBlend{};
    colorBlend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlend.attachmentCount = 1;
    colorBlend.pAttachments = &colorBlendAttachment;

    const std::array<VkDynamicState, 2> dynamicStates = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR
    };

    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
    dynamicState.pDynamicStates = dynamicStates.data();

    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    VK_CHECK(vkCreatePipelineLayout(device_, &layoutInfo, nullptr, &layout_));

    VkFormat colorFormat = createInfo.colorFormat;
    VkPipelineRenderingCreateInfo renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachmentFormats = &colorFormat;
    renderingInfo.depthAttachmentFormat = createInfo.enableDepth ? createInfo.depthFormat : VK_FORMAT_UNDEFINED;

    // Dynamic Rendering has no VkRenderPass compatibility object, so the pipeline
    // declares the attachment formats it will render into through this pNext struct.
    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.pNext = &renderingInfo;
    pipelineInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
    pipelineInfo.pStages = shaderStages.data();
    pipelineInfo.pVertexInputState = &vertexInput;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterization;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.pColorBlendState = &colorBlend;
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.layout = layout_;
    pipelineInfo.renderPass = VK_NULL_HANDLE;

    VK_CHECK(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline_));
}

void VulkanPipeline::reset()
{
    if (pipeline_) {
        vkDestroyPipeline(device_, pipeline_, nullptr);
        pipeline_ = VK_NULL_HANDLE;
    }

    if (layout_) {
        vkDestroyPipelineLayout(device_, layout_, nullptr);
        layout_ = VK_NULL_HANDLE;
    }

    if (fragmentShaderModule_) {
        vkDestroyShaderModule(device_, fragmentShaderModule_, nullptr);
        fragmentShaderModule_ = VK_NULL_HANDLE;
    }

    if (vertexShaderModule_) {
        vkDestroyShaderModule(device_, vertexShaderModule_, nullptr);
        vertexShaderModule_ = VK_NULL_HANDLE;
    }

    device_ = VK_NULL_HANDLE;
}

VkShaderModule VulkanPipeline::createShaderModule(const std::filesystem::path& path) const
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

void VulkanPipeline::moveFrom(VulkanPipeline& other) noexcept
{
    device_ = std::exchange(other.device_, VK_NULL_HANDLE);
    vertexShaderModule_ = std::exchange(other.vertexShaderModule_, VK_NULL_HANDLE);
    fragmentShaderModule_ = std::exchange(other.fragmentShaderModule_, VK_NULL_HANDLE);
    layout_ = std::exchange(other.layout_, VK_NULL_HANDLE);
    pipeline_ = std::exchange(other.pipeline_, VK_NULL_HANDLE);
}

} // namespace ve::rhi
