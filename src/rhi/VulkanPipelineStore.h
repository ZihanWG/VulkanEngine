#pragma once

#include "rhi/PipelineKey.h"
#include "rhi/VulkanPipeline.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

// Owns graphics pipelines keyed by their state, so callers ask for "a pipeline
// that does this" instead of declaring "a pipeline named that".
//
// Distinct from rhi::VulkanPipelineCache despite the similar name: that one is the
// on-disk VkPipelineCache blob (a driver-level compile cache, persisted between
// runs and validated against the device). This one is an in-memory map from
// PipelineKey to a live VulkanPipeline, and it is what makes two callers that
// describe the same state share one VkPipeline instead of compiling two. The two
// compose -- entries created here still pass createInfo.pipelineCache down to
// vkCreateGraphicsPipelines.
//
// ## Lifetime, and why it is the whole safety argument
//
// PipelineKey holds VkDescriptorSetLayout *handles*. If a layout were destroyed
// and recreated while an entry keyed on the old handle stayed live, a later
// lookup could either collide with a recycled handle value or hand back a pipeline
// whose layout no longer exists. Nothing in the key can detect that.
//
// The rule that removes the hazard: **the store is reset() wherever pipelines are
// rebuilt wholesale.** In this renderer that is the top of
// Renderer::createPipeline(), which is already the single point where every
// descriptor set layout, attachment format and shader is re-derived. The store
// therefore never outlives a layout recreation, and no eviction policy is needed
// -- entries live exactly one pipeline-build generation.
//
// Do not add a "keep entries across reset if they look unchanged" optimization
// without first giving layouts a generation counter. The saving would be a handful
// of pipeline compiles behind an already-warm VkPipelineCache; the risk is a stale
// handle that validation cannot see.
//
// ## Debug names
//
// A collapsed entry keeps the name its *first* requester gave it, because that is
// the name already on the VkPipeline. Every requested name is recorded so the
// sharing is visible (entryDebugNames()) rather than showing up in a capture as a
// pass mysteriously labelled with another pass's name.
namespace ve::rhi {

class VulkanPipelineStore {
public:
    VulkanPipelineStore() = default;
    ~VulkanPipelineStore() = default;

    VulkanPipelineStore(const VulkanPipelineStore&) = delete;
    VulkanPipelineStore& operator=(const VulkanPipelineStore&) = delete;
    VulkanPipelineStore(VulkanPipelineStore&&) = delete;
    VulkanPipelineStore& operator=(VulkanPipelineStore&&) = delete;

    // Returns the pipeline for createInfo's state, creating it on first request.
    //
    // The reference is stable until reset(): the map is node-based, so neither
    // rehashing nor later insertions move the stored VulkanPipeline. Callers may
    // therefore hold the address across frames -- which is exactly how the renderer
    // keeps its per-pipeline members.
    //
    // Throws whatever VulkanPipeline::create() throws (a missing SPIR-V file, an
    // undefined attachment format, a driver failure). On a throw the store is
    // unchanged: the pipeline is built into a local and only moved in on success,
    // so a failed optional feature leaves no half-built entry behind.
    [[nodiscard]] const VulkanPipeline&
    get(VkDevice device, const VulkanPipelineCreateInfo& createInfo, std::string_view debugName);

    // Destroys every pipeline. See the lifetime note above for when this must run.
    void reset();

    [[nodiscard]] std::size_t size() const
    {
        return entries_.size();
    }
    [[nodiscard]] uint32_t hits() const
    {
        return hits_;
    }
    [[nodiscard]] uint32_t misses() const
    {
        return misses_;
    }

    // One inner vector per live pipeline, holding every debug name requested for
    // it in request order. An inner vector with more than one name is a collapse:
    // those callers described identical state. Ordering between entries is
    // unspecified (it is a hash map); sort before printing if stability matters.
    [[nodiscard]] std::vector<std::vector<std::string>> entryDebugNames() const;

private:
    struct Entry {
        VulkanPipeline pipeline;
        std::vector<std::string> debugNames;
    };

    std::unordered_map<PipelineKey, Entry> entries_;
    uint32_t hits_ = 0;
    uint32_t misses_ = 0;
};

} // namespace ve::rhi
