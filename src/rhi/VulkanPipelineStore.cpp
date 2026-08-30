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
        return PipelineRef(existing->second.pipeline);
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
    return PipelineRef(inserted.first->second.pipeline);
}

void VulkanPipelineStore::reset()
{
    entries_.clear();
    hits_ = 0;
    misses_ = 0;
}

std::vector<std::vector<std::string>> VulkanPipelineStore::entryDebugNames() const
{
    std::vector<std::vector<std::string>> names;
    names.reserve(entries_.size());
    for (const auto& entry : entries_) {
        names.push_back(entry.second.debugNames);
    }

    return names;
}

} // namespace ve::rhi
