#include "rhi/VulkanPipelineStore.h"

#include "rhi/VulkanDebugUtils.h"

#include <utility>

namespace ve::rhi {

PipelineRef
VulkanPipelineStore::get(VkDevice device, const VulkanPipelineCreateInfo& createInfo, std::string_view debugName)
{
    PipelineKey key = PipelineKey::from(createInfo);

    if (const auto existing = entries_.find(key); existing != entries_.end()) {
        ++hits_;
        existing->second.debugNames.emplace_back(debugName);
        return PipelineRef(*this, existing->second.pipeline, generation_);
    }

    // Built into a local first so a throw leaves entries_ untouched. Several
    // callers treat a failed pipeline as "feature unavailable" and carry on, so
    // this path is reached in normal operation, not only on a broken build.
    VulkanPipeline pipeline;
    pipeline.create(device, createInfo);

    const std::string name(debugName);
    debug::setObjectName(device, pipeline.pipeline(), VK_OBJECT_TYPE_PIPELINE, name);
    debug::setObjectName(device, pipeline.layout(), VK_OBJECT_TYPE_PIPELINE_LAYOUT, name + "Layout");

    ++misses_;
    Entry entry{};
    entry.pipeline = std::move(pipeline);
    entry.debugNames.push_back(name);

    const auto inserted = entries_.emplace(std::move(key), std::move(entry));
    return PipelineRef(*this, inserted.first->second.pipeline, generation_);
}

ComputePipelineRef
VulkanPipelineStore::get(VkDevice device, const VulkanComputePipelineCreateInfo& createInfo, std::string_view debugName)
{
    ComputePipelineKey key = ComputePipelineKey::from(createInfo);

    if (const auto existing = computeEntries_.find(key); existing != computeEntries_.end()) {
        ++hits_;
        existing->second.debugNames.emplace_back(debugName);
        return ComputePipelineRef(*this, existing->second.pipeline, generation_);
    }

    VulkanComputePipeline pipeline;
    pipeline.create(device, createInfo);

    const std::string name(debugName);
    debug::setObjectName(device, pipeline.pipeline(), VK_OBJECT_TYPE_PIPELINE, name);
    debug::setObjectName(device, pipeline.layout(), VK_OBJECT_TYPE_PIPELINE_LAYOUT, name + "Layout");

    ++misses_;
    ComputeEntry entry{};
    entry.pipeline = std::move(pipeline);
    entry.debugNames.push_back(name);

    const auto inserted = computeEntries_.emplace(std::move(key), std::move(entry));
    return ComputePipelineRef(*this, inserted.first->second.pipeline, generation_);
}

void VulkanPipelineStore::reset()
{
    entries_.clear();
    computeEntries_.clear();
    hits_ = 0;
    misses_ = 0;
    // Invalidates every ref issued from the entries just destroyed. A ref the
    // caller reissues picks up the new generation; one it forgets to reissue
    // reads as empty instead of as a dangling pointer.
    ++generation_;
}

std::vector<std::vector<std::string>> VulkanPipelineStore::entryDebugNames() const
{
    std::vector<std::vector<std::string>> names;
    names.reserve(entries_.size() + computeEntries_.size());
    for (const auto& entry : entries_) {
        names.push_back(entry.second.debugNames);
    }
    for (const auto& entry : computeEntries_) {
        names.push_back(entry.second.debugNames);
    }

    return names;
}

} // namespace ve::rhi
