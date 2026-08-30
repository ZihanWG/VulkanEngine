#pragma once

#include "rhi/PipelineKey.h"
#include "rhi/VulkanComputePipeline.h"
#include "rhi/VulkanPipeline.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

// Owns pipelines keyed by their state, so callers ask for "a pipeline that does
// this" instead of declaring "a pipeline named that". Graphics and compute both,
// in two maps behind one reset and one generation.
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
// That rule has a sharp edge, and it has already drawn blood once
// (createProbeCapturePipeline, which the constructor called and createPipeline()
// did not): **only put a pipeline in this store if createPipeline() rebuilds it.**
// A ref that the rebuild does not reissue points at a destroyed entry. The
// generation stamp on PipelineRefT turns that into a null handle rather than a
// use-after-free, but a silently missing pass is still a bug -- the guard is a
// backstop, not a licence.
//
// This is why the compute pipelines owned by ClusteredLighting, GpuCulling,
// PunctualShadows, VolumetricFogPass, VirtualShadowMapPass and the probe volume
// are *not* here: they are built inside those subsystems' createResources() calls,
// their lifetime is their subsystem's resources rather than the renderer's
// pipeline rebuild, and a reset would destroy them with nothing to rebuild them.
// The line is lifetime, not graphics-versus-compute.
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

class VulkanPipelineStore;

// A non-owning handle to a pipeline the store owns.
//
// It exists so a caller can hold "the pipeline for this pass" as a member the
// same way it used to hold a VulkanPipeline, without owning it. The accessors
// deliberately mirror VulkanPipeline's and return VK_NULL_HANDLE when empty,
// because an empty ref is how this renderer already spells "this feature is not
// available in this configuration" -- the bindless fallback, a missing punctual
// atlas, a VSM page pool that was never allocated. Every existing
// `x.pipeline() != VK_NULL_HANDLE` guard keeps working unchanged, and none of
// them has to learn about null.
//
// Templated because VulkanPipeline and VulkanComputePipeline are unrelated types
// with the same two accessors, and the generation check below must exist once.
//
// A ref is only obtainable from VulkanPipelineStore::get(), so its lifetime
// question has exactly one answer: it is valid until that store is reset(). It
// carries the store's generation so that answer is enforced rather than trusted.
template <typename Pipeline>
class PipelineRefT {
public:
    PipelineRefT() = default;
    PipelineRefT(const VulkanPipelineStore& store, const Pipeline& pipeline, uint32_t generation)
        : store_(&store), pipeline_(&pipeline), generation_(generation)
    {}

    // VK_NULL_HANDLE both when empty and when the store has been reset since this
    // ref was issued. The second case is the backstop: every ref must be reissued
    // by the rebuild that reset the store, and one that is not would otherwise be
    // a pointer into a destroyed entry. Degrading to the null handle turns that
    // into the "feature unavailable" path every call site already handles, which
    // is a visibly missing effect rather than a use-after-free.
    [[nodiscard]] VkPipeline pipeline() const { return current() ? pipeline_->pipeline() : VK_NULL_HANDLE; }

    [[nodiscard]] VkPipelineLayout layout() const { return current() ? pipeline_->layout() : VK_NULL_HANDLE; }

    // Forgets the pipeline; it stays alive in the store. Named reset() to match
    // what the owning members it replaced were called.
    void reset() { *this = PipelineRefT{}; }

    [[nodiscard]] bool valid() const { return current(); }

private:
    // Defined after VulkanPipelineStore, whose generation it reads.
    [[nodiscard]] bool current() const;

    const VulkanPipelineStore* store_ = nullptr;
    const Pipeline* pipeline_ = nullptr;
    uint32_t generation_ = 0;
};

using PipelineRef = PipelineRefT<VulkanPipeline>;
using ComputePipelineRef = PipelineRefT<VulkanComputePipeline>;

class VulkanPipelineStore {
public:
    VulkanPipelineStore() = default;
    ~VulkanPipelineStore() = default;

    VulkanPipelineStore(const VulkanPipelineStore&) = delete;
    VulkanPipelineStore& operator=(const VulkanPipelineStore&) = delete;
    VulkanPipelineStore(VulkanPipelineStore&&) = delete;
    VulkanPipelineStore& operator=(VulkanPipelineStore&&) = delete;

    // Returns a ref to the pipeline for createInfo's state, creating it on first
    // request.
    //
    // The ref stays valid until reset(): the map is node-based, so neither
    // rehashing nor later insertions move the stored VulkanPipeline. Callers may
    // therefore hold it across frames -- which is exactly how the renderer keeps
    // its per-pipeline members.
    //
    // Throws whatever VulkanPipeline::create() throws (a missing SPIR-V file, an
    // undefined attachment format, a driver failure). On a throw the store is
    // unchanged: the pipeline is built into a local and only moved in on success,
    // so a failed optional feature leaves no half-built entry behind.
    [[nodiscard]] PipelineRef
    get(VkDevice device, const VulkanPipelineCreateInfo& createInfo, std::string_view debugName);

    // The compute overload. Same contract, same generation, same reset -- a
    // compute pipeline whose lifetime is the renderer's pipeline rebuild belongs
    // in the same store as the graphics ones, and one whose lifetime is its
    // subsystem's resources does not (see the class comment).
    [[nodiscard]] ComputePipelineRef
    get(VkDevice device, const VulkanComputePipelineCreateInfo& createInfo, std::string_view debugName);

    // Destroys every pipeline. See the lifetime note above for when this must run.
    void reset();

    // Graphics and compute together: the log line reports what the renderer
    // holds, and the split is not what a reader of it wants to reconcile.
    [[nodiscard]] std::size_t size() const
    {
        return entries_.size() + computeEntries_.size();
    }
    // Bumped by reset(). Refs compare against it to notice they were not reissued.
    [[nodiscard]] uint32_t generation() const
    {
        return generation_;
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
    template <typename Pipeline>
    struct EntryT {
        Pipeline pipeline;
        std::vector<std::string> debugNames;
    };

    using Entry = EntryT<VulkanPipeline>;
    using ComputeEntry = EntryT<VulkanComputePipeline>;

    std::unordered_map<PipelineKey, Entry> entries_;
    std::unordered_map<ComputePipelineKey, ComputeEntry> computeEntries_;
    uint32_t hits_ = 0;
    uint32_t misses_ = 0;
    uint32_t generation_ = 0;
};

template <typename Pipeline>
inline bool PipelineRefT<Pipeline>::current() const
{
    return pipeline_ != nullptr && store_ != nullptr && store_->generation() == generation_;
}

} // namespace ve::rhi
