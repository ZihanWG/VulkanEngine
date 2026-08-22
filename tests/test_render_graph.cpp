// Render graph pass culling: the backward liveness sweep that decides which
// declared passes actually run. It is the graph's one piece of non-trivial pure
// logic, and a mistake in it either drops work that was needed (missing shadows,
// a black screen) or keeps work that was not, silently, with no validation error
// either way.

#include "renderer/RenderGraph.h"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <span>
#include <string>
#include <vector>

using ve::renderer::cullUnusedPasses;
using ve::renderer::RenderPassNode;
using ve::renderer::RenderResourceAccess;
using ve::renderer::RenderResourceUsage;
using ve::renderer::RGResourceKind;

namespace {

RenderResourceUsage texture(uint32_t index, RenderResourceAccess access)
{
    RenderResourceUsage usage{};
    usage.resource.name = "texture" + std::to_string(index);
    usage.resource.kind = RGResourceKind::Texture;
    usage.resource.index = index;
    usage.access = access;
    return usage;
}

RenderResourceUsage buffer(uint32_t index, RenderResourceAccess access)
{
    RenderResourceUsage usage{};
    usage.resource.name = "buffer" + std::to_string(index);
    usage.resource.kind = RGResourceKind::Buffer;
    usage.resource.index = index;
    usage.access = access;
    return usage;
}

RenderResourceUsage historyTexture(uint32_t index)
{
    RenderResourceUsage usage = texture(index, RenderResourceAccess::Read);
    usage.historyRead = true;
    return usage;
}

RenderPassNode pass(std::string name, std::vector<RenderResourceUsage> usages, bool sideEffect = false)
{
    RenderPassNode node{};
    node.name = std::move(name);
    node.resourceUsages = std::move(usages);
    node.sideEffect = sideEffect;
    return node;
}

} // namespace

TEST_CASE("A pass whose writes nothing reads is culled", "[rendergraph]")
{
    std::vector<RenderPassNode> passes{pass("Orphan", {texture(0, RenderResourceAccess::Write)})};

    cullUnusedPasses(passes, 1, 0);

    CHECK(passes[0].culled);
    CHECK_FALSE(passes[0].cullReason.empty());
}

TEST_CASE("A side effect keeps a pass alive with no readers", "[rendergraph]")
{
    // Presentation and readbacks leave the graph, so their writes have no reader
    // inside it. Without this the final pass of every frame would be culled.
    std::vector<RenderPassNode> passes{pass("Present", {texture(0, RenderResourceAccess::Write)}, true)};

    cullUnusedPasses(passes, 1, 0);

    CHECK_FALSE(passes[0].culled);
}

TEST_CASE("A pass feeding a live reader survives", "[rendergraph]")
{
    std::vector<RenderPassNode> passes{
        pass("Producer", {texture(0, RenderResourceAccess::Write)}),
        pass("Consumer", {texture(0, RenderResourceAccess::Read), texture(1, RenderResourceAccess::Write)}, true),
    };

    cullUnusedPasses(passes, 2, 0);

    CHECK_FALSE(passes[0].culled);
    CHECK_FALSE(passes[1].culled);
}

TEST_CASE("A pass that produces nothing is culled even when it reads live data", "[rendergraph]")
{
    // Reading without writing and without a side effect cannot affect the frame,
    // so the pass is dead however useful its input is.
    std::vector<RenderPassNode> passes{pass("ReadOnly", {texture(0, RenderResourceAccess::Read)})};

    cullUnusedPasses(passes, 1, 0);

    CHECK(passes[0].culled);
}

TEST_CASE("Culling is transitive back through a chain", "[rendergraph]")
{
    // A -> B -> nothing. B is dead because its output is unread, and A must then
    // die too: a culled pass does not propagate its reads, so B's read of
    // texture 0 must not keep A alive. Getting this wrong leaves a chain of
    // passes running to feed a result that is thrown away.
    std::vector<RenderPassNode> passes{
        pass("A", {texture(0, RenderResourceAccess::Write)}),
        pass("B", {texture(0, RenderResourceAccess::Read), texture(1, RenderResourceAccess::Write)}),
    };

    cullUnusedPasses(passes, 2, 0);

    CHECK(passes[1].culled);
    CHECK(passes[0].culled);
}

TEST_CASE("A write that is overwritten before any read culls the earlier pass", "[rendergraph]")
{
    // Two passes write the same texture and only the later result is read, so
    // the first write is dead. This is the case that needs a write to *clear*
    // the liveness flag rather than leave it set.
    std::vector<RenderPassNode> passes{
        pass("FirstWrite", {texture(0, RenderResourceAccess::Write)}),
        pass("SecondWrite", {texture(0, RenderResourceAccess::Write)}),
        pass("Reader", {texture(0, RenderResourceAccess::Read)}, true),
    };

    cullUnusedPasses(passes, 1, 0);

    CHECK(passes[0].culled);
    CHECK_FALSE(passes[1].culled);
    CHECK_FALSE(passes[2].culled);
}

TEST_CASE("ReadWrite keeps the producer alive", "[rendergraph]")
{
    // A read-modify-write both consumes and produces the resource, so it must
    // leave it live for whoever produced it. If the sweep applied the write
    // after the read instead of before, the flag would end up cleared and the
    // producer would be culled out from under the pass that reads it.
    std::vector<RenderPassNode> passes{
        pass("Producer", {texture(0, RenderResourceAccess::Write)}),
        pass("Accumulate", {texture(0, RenderResourceAccess::ReadWrite)}, true),
    };

    cullUnusedPasses(passes, 1, 0);

    CHECK_FALSE(passes[0].culled);
    CHECK_FALSE(passes[1].culled);
}

TEST_CASE("Buffers and textures have independent liveness", "[rendergraph]")
{
    // Both index spaces start at 0, so a sweep that ignored the kind would let a
    // read of buffer 0 keep a pass writing texture 0 alive.
    std::vector<RenderPassNode> passes{
        pass("WritesTexture", {texture(0, RenderResourceAccess::Write)}),
        pass("ReadsBuffer", {buffer(0, RenderResourceAccess::Read), buffer(1, RenderResourceAccess::Write)}, true),
    };

    cullUnusedPasses(passes, 1, 2);

    CHECK(passes[0].culled);
    CHECK_FALSE(passes[1].culled);
}

TEST_CASE("A buffer producer survives a later buffer reader", "[rendergraph]")
{
    std::vector<RenderPassNode> passes{
        pass("FillBuffer", {buffer(0, RenderResourceAccess::Write)}),
        pass("ConsumeBuffer", {buffer(0, RenderResourceAccess::Read), buffer(1, RenderResourceAccess::Write)}, true),
    };

    cullUnusedPasses(passes, 0, 2);

    CHECK_FALSE(passes[0].culled);
    CHECK_FALSE(passes[1].culled);
}

TEST_CASE("Usages past the resource count do not keep a pass alive", "[rendergraph]")
{
    // An index the graph never imported is ignored rather than treated as live,
    // so a pass whose only write is out of range still counts as writing and is
    // culled for having no reachable output.
    std::vector<RenderPassNode> passes{pass("Stale", {texture(99, RenderResourceAccess::Write)})};

    cullUnusedPasses(passes, 1, 0);

    CHECK(passes[0].culled);
}

TEST_CASE("Culling an empty graph is a no-op", "[rendergraph]")
{
    std::vector<RenderPassNode> passes;
    CHECK_NOTHROW(cullUnusedPasses(passes, 0, 0));
    CHECK(passes.empty());
}

TEST_CASE("A pass with no declared usages is culled unless it has a side effect", "[rendergraph]")
{
    std::vector<RenderPassNode> passes{pass("Empty", {}), pass("EmptyWithEffect", {}, true)};

    cullUnusedPasses(passes, 0, 0);

    CHECK(passes[0].culled);
    CHECK_FALSE(passes[1].culled);
}

TEST_CASE("A live consumer revives a whole producer chain", "[rendergraph]")
{
    // The shape the real frame graph has: depth -> pyramid -> cull -> draw, with
    // only the last pass having an external effect.
    std::vector<RenderPassNode> passes{
        pass("Depth", {texture(0, RenderResourceAccess::Write)}),
        pass("Pyramid", {texture(0, RenderResourceAccess::Read), texture(1, RenderResourceAccess::Write)}),
        pass("Cull", {texture(1, RenderResourceAccess::Read), buffer(0, RenderResourceAccess::Write)}),
        pass("Draw", {buffer(0, RenderResourceAccess::Read), texture(2, RenderResourceAccess::Write)}, true),
    };

    cullUnusedPasses(passes, 3, 1);

    for (const RenderPassNode& node : passes) {
        INFO("pass " << node.name << " reason: " << node.cullReason);
        CHECK_FALSE(node.culled);
    }
}

// ---------------------------------------------------------------------------
// Barrier derivation: what a declared access means in layout/stage/access terms.
//
// The trap here is the image aspect. A depth image asks for a depth layout where
// a colour image asks for the general shader-read one, and whether stencil is
// present picks between the combined and depth-only variants. Getting it wrong
// is a validation error at best and a driver-dependent correctness bug at worst.

TEST_CASE("Shader-read layout depends on the image aspect", "[rendergraph][barriers]")
{
    using ve::renderer::RGAccess;
    using ve::renderer::textureAccessState;

    const auto colorState =
        textureAccessState(VK_IMAGE_ASPECT_COLOR_BIT, RGAccess::ShaderRead, VK_IMAGE_LAYOUT_UNDEFINED);
    CHECK(colorState.layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    // A depth image sampled as a texture (Hi-Z, SSR, GTAO all do this) must use a
    // depth read-only layout, not the colour one.
    const auto depthState =
        textureAccessState(VK_IMAGE_ASPECT_DEPTH_BIT, RGAccess::ShaderRead, VK_IMAGE_LAYOUT_UNDEFINED);
    CHECK(depthState.layout == VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL);

    // With stencil present it is the combined layout instead.
    const auto depthStencilState = textureAccessState(
        VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT, RGAccess::ShaderRead, VK_IMAGE_LAYOUT_UNDEFINED);
    CHECK(depthStencilState.layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);

    // The scopes are the same either way -- only the layout is aspect-dependent.
    CHECK(colorState.access == VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    CHECK(depthState.access == VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
}

TEST_CASE("Depth attachment layout depends on stencil presence", "[rendergraph][barriers]")
{
    using ve::renderer::RGAccess;
    using ve::renderer::textureAccessState;

    const auto depthOnly = textureAccessState(
        VK_IMAGE_ASPECT_DEPTH_BIT, RGAccess::DepthStencilAttachmentWrite, VK_IMAGE_LAYOUT_UNDEFINED);
    CHECK(depthOnly.layout == VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);

    const auto withStencil = textureAccessState(VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT,
                                                RGAccess::DepthStencilAttachmentWrite,
                                                VK_IMAGE_LAYOUT_UNDEFINED);
    CHECK(withStencil.layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

    // Depth is tested in both fragment-test stages, so both must be in scope or a
    // write can race the early test of the next pass.
    CHECK((depthOnly.stage & VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT) != 0);
    CHECK((depthOnly.stage & VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT) != 0);
}

TEST_CASE("Storage image access always uses the general layout", "[rendergraph][barriers]")
{
    using ve::renderer::RGAccess;
    using ve::renderer::textureAccessState;

    for (const RGAccess access :
         {RGAccess::StorageImageRead, RGAccess::StorageImageWrite, RGAccess::StorageImageReadWrite}) {
        // Even for a depth-aspect image: storage access has no depth variant.
        CHECK(textureAccessState(VK_IMAGE_ASPECT_COLOR_BIT, access, VK_IMAGE_LAYOUT_UNDEFINED).layout ==
              VK_IMAGE_LAYOUT_GENERAL);
        CHECK(textureAccessState(VK_IMAGE_ASPECT_DEPTH_BIT, access, VK_IMAGE_LAYOUT_UNDEFINED).layout ==
              VK_IMAGE_LAYOUT_GENERAL);
    }

    const auto readWrite =
        textureAccessState(VK_IMAGE_ASPECT_COLOR_BIT, RGAccess::StorageImageReadWrite, VK_IMAGE_LAYOUT_UNDEFINED);
    CHECK((readWrite.access & VK_ACCESS_2_SHADER_STORAGE_READ_BIT) != 0);
    CHECK((readWrite.access & VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT) != 0);
}

TEST_CASE("Present asks for the present layout and no scopes", "[rendergraph][barriers]")
{
    using ve::renderer::RGAccess;
    using ve::renderer::textureAccessState;

    // Presentation is synchronized by the semaphore, not by this barrier, so the
    // transition must carry the layout without claiming a stage or access scope.
    const auto state =
        textureAccessState(VK_IMAGE_ASPECT_COLOR_BIT, RGAccess::Present, VK_IMAGE_LAYOUT_UNDEFINED);
    CHECK(state.layout == VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
    CHECK(state.stage == VK_PIPELINE_STAGE_2_NONE);
    CHECK(state.access == VK_ACCESS_2_NONE);
    CHECK(state.declaredAccess == RGAccess::Present);
}

TEST_CASE("A buffer access on a texture falls back to the current layout", "[rendergraph][barriers]")
{
    using ve::renderer::RGAccess;
    using ve::renderer::textureAccessState;

    // Buffer-shaped accesses say nothing about image layout, so the texture keeps
    // whatever it already had rather than being transitioned to UNDEFINED, which
    // would discard its contents.
    const auto state = textureAccessState(
        VK_IMAGE_ASPECT_COLOR_BIT, RGAccess::StorageBufferRead, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    CHECK(state.layout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    CHECK(state.stage == VK_PIPELINE_STAGE_2_NONE);
    CHECK(state.access == VK_ACCESS_2_NONE);
}

TEST_CASE("Transfer accesses map to their own layouts", "[rendergraph][barriers]")
{
    using ve::renderer::RGAccess;
    using ve::renderer::textureAccessState;

    const auto src =
        textureAccessState(VK_IMAGE_ASPECT_COLOR_BIT, RGAccess::TransferSrc, VK_IMAGE_LAYOUT_UNDEFINED);
    CHECK(src.layout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    CHECK(src.access == VK_ACCESS_2_TRANSFER_READ_BIT);

    const auto dst =
        textureAccessState(VK_IMAGE_ASPECT_COLOR_BIT, RGAccess::TransferDst, VK_IMAGE_LAYOUT_UNDEFINED);
    CHECK(dst.layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    CHECK(dst.access == VK_ACCESS_2_TRANSFER_WRITE_BIT);
}

TEST_CASE("Indirect buffer reads sync against the draw-indirect stage", "[rendergraph][barriers]")
{
    using ve::renderer::bufferAccessState;
    using ve::renderer::RGAccess;

    // GPU culling writes these and the draw consumes them; the wrong stage here
    // races the command fetch, which reads as sporadically missing geometry.
    const auto state = bufferAccessState(RGAccess::IndirectRead);
    CHECK(state.stage == VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT);
    CHECK(state.access == VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT);
}

TEST_CASE("Storage buffer stages cover every shader stage that binds them", "[rendergraph][barriers]")
{
    using ve::renderer::bufferAccessState;
    using ve::renderer::RGAccess;

    // These buffers are bound by vertex, fragment, and compute shaders alike, so
    // omitting any one stage leaves a real hazard unsynchronized.
    const auto state = bufferAccessState(RGAccess::StorageBufferReadWrite);
    CHECK((state.stage & VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT) != 0);
    CHECK((state.stage & VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT) != 0);
    CHECK((state.stage & VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT) != 0);
    CHECK((state.access & VK_ACCESS_2_SHADER_STORAGE_READ_BIT) != 0);
    CHECK((state.access & VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT) != 0);
}

TEST_CASE("Host reads sync against the host stage", "[rendergraph][barriers]")
{
    using ve::renderer::bufferAccessState;
    using ve::renderer::RGAccess;

    // The readback path depends on this; without it the CPU can observe stale
    // contents after the fence.
    const auto state = bufferAccessState(RGAccess::HostRead);
    CHECK(state.stage == VK_PIPELINE_STAGE_2_HOST_BIT);
    CHECK(state.access == VK_ACCESS_2_HOST_READ_BIT);
}

TEST_CASE("An image access on a buffer emits no barrier", "[rendergraph][barriers]")
{
    using ve::renderer::bufferAccessState;
    using ve::renderer::RGAccess;

    // transitionBuffer skips when declaredAccess is Unknown or the stage is NONE.
    // The buffer mapping resets declaredAccess as well as the scopes, so an
    // image-shaped access on a buffer is dropped rather than emitting a barrier
    // with empty scopes.
    for (const RGAccess access : {RGAccess::ShaderRead,
                                  RGAccess::ColorAttachmentWrite,
                                  RGAccess::DepthStencilAttachmentWrite,
                                  RGAccess::StorageImageReadWrite,
                                  RGAccess::Present}) {
        const auto state = bufferAccessState(access);
        CHECK(state.declaredAccess == RGAccess::Unknown);
        CHECK(state.stage == VK_PIPELINE_STAGE_2_NONE);
        CHECK(state.access == VK_ACCESS_2_NONE);
    }
}

TEST_CASE("Buffer transfer directions use distinct stages", "[rendergraph][barriers]")
{
    using ve::renderer::bufferAccessState;
    using ve::renderer::RGAccess;

    CHECK(bufferAccessState(RGAccess::TransferSrc).access == VK_ACCESS_2_TRANSFER_READ_BIT);
    CHECK(bufferAccessState(RGAccess::TransferDst).access == VK_ACCESS_2_TRANSFER_WRITE_BIT);
}

// ---------------------------------------------------------------------------
// Barrier necessity: whether a transition emits anything at all.
//
// Emitting a barrier that was not needed only costs performance. Skipping one
// that was needed is a race — and on tile-based hardware it often still renders
// correctly on the machine it was written on, which is the worst failure mode
// this file guards against.

TEST_CASE("Read after read needs no buffer barrier", "[rendergraph][barriers]")
{
    using ve::renderer::bufferBarrierRequired;
    using ve::renderer::RGAccess;

    // Two reads cannot race, however many passes touch the buffer.
    CHECK_FALSE(bufferBarrierRequired(/*usedThisFrame=*/true,
                                      RGAccess::StorageBufferRead,
                                      VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
                                      VK_ACCESS_2_SHADER_STORAGE_READ_BIT));
}

TEST_CASE("Any write on either side needs a buffer barrier", "[rendergraph][barriers]")
{
    using ve::renderer::bufferBarrierRequired;
    using ve::renderer::RGAccess;

    // write -> read
    CHECK(bufferBarrierRequired(
        true, RGAccess::StorageBufferWrite, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT));
    // read -> write
    CHECK(bufferBarrierRequired(
        true, RGAccess::StorageBufferRead, VK_ACCESS_2_SHADER_STORAGE_READ_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT));
    // write -> write
    CHECK(bufferBarrierRequired(
        true, RGAccess::StorageBufferWrite, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT));
}

TEST_CASE("An untouched buffer needs no barrier on its first use", "[rendergraph][barriers]")
{
    using ve::renderer::bufferBarrierRequired;
    using ve::renderer::RGAccess;

    // Nothing has run against it yet, so there is nothing to order against even
    // though the incoming access writes.
    CHECK_FALSE(bufferBarrierRequired(
        /*usedThisFrame=*/false, RGAccess::Unknown, VK_ACCESS_2_NONE, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT));

    // But either half of "has been touched" is enough to require one: a
    // resource carried over from a previous frame has usedThisFrame false while
    // still holding a real last access.
    CHECK(bufferBarrierRequired(
        false, RGAccess::StorageBufferWrite, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT));
    CHECK(bufferBarrierRequired(
        true, RGAccess::Unknown, VK_ACCESS_2_NONE, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT));
}

TEST_CASE("A layout change always needs a texture barrier", "[rendergraph][barriers]")
{
    using ve::renderer::RGAccess;
    using ve::renderer::textureBarrierRequired;

    // Even read-to-read, and even untouched: the layout transition itself is the
    // work, and nothing else performs it.
    CHECK(textureBarrierRequired(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                 VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                 /*usedThisFrame=*/false,
                                 RGAccess::Unknown,
                                 VK_ACCESS_2_NONE,
                                 VK_ACCESS_2_NONE));
}

TEST_CASE("Same layout falls back to the write rule", "[rendergraph][barriers]")
{
    using ve::renderer::RGAccess;
    using ve::renderer::textureBarrierRequired;

    CHECK_FALSE(textureBarrierRequired(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                       true,
                                       RGAccess::ShaderRead,
                                       VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                                       VK_ACCESS_2_SHADER_SAMPLED_READ_BIT));

    // Two storage-image passes both sit in GENERAL, so the layout is unchanged
    // and only the access rule separates them.
    CHECK(textureBarrierRequired(VK_IMAGE_LAYOUT_GENERAL,
                                 VK_IMAGE_LAYOUT_GENERAL,
                                 true,
                                 RGAccess::StorageImageWrite,
                                 VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                                 VK_ACCESS_2_SHADER_STORAGE_READ_BIT));
}

TEST_CASE("An UNDEFINED old layout orders against nothing", "[rendergraph][barriers]")
{
    using ve::renderer::RGAccess;
    using ve::renderer::textureBarrierRequired;

    // UNDEFINED means the contents are not preserved, so there is no prior
    // state to synchronize with. Reaching here at all means the desired layout
    // is UNDEFINED too, since any real target would be a layout change above.
    CHECK_FALSE(textureBarrierRequired(VK_IMAGE_LAYOUT_UNDEFINED,
                                       VK_IMAGE_LAYOUT_UNDEFINED,
                                       true,
                                       RGAccess::ColorAttachmentWrite,
                                       VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                                       VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT));
}

// ---------------------------------------------------------------------------
// Declaration validation. The declarations are the graph's description of what
// each pass touches; until something checks them, a pass can read a transient
// nothing produced and the only symptom is wrong pixels -- no validation error,
// no crash. These are the three ways a declaration set can be wrong that are
// decidable without a device.

namespace {

using ve::renderer::RGDeclarationIssue;
using ve::renderer::RGResourceValidationInfo;
using ve::renderer::validateDeclarations;

constexpr RGResourceValidationInfo kImported{false, false};
constexpr RGResourceValidationInfo kTransient{true, false};
constexpr RGResourceValidationInfo kAliasedTransient{true, true};

} // namespace

TEST_CASE("Reading a transient nothing produced is reported", "[rendergraph][declarations]")
{
    std::vector<RenderPassNode> passes{pass("Consumer", {texture(0, RenderResourceAccess::Read)}, true)};
    const std::vector<RGResourceValidationInfo> textures{kTransient};

    const auto issues = validateDeclarations(passes, textures, {});

    REQUIRE(issues.size() == 1);
    CHECK(issues[0].passIndex == 0);
    CHECK(issues[0].issue == RGDeclarationIssue::ReadsContentNoPassProduced);
    CHECK(issues[0].resourceIndex == 0);
}

TEST_CASE("Reading an imported resource before writing it is fine", "[rendergraph][declarations]")
{
    // An imported resource keeps its contents by contract -- the swapchain, the
    // shadow maps, the probe atlases -- so there is nothing to produce.
    std::vector<RenderPassNode> passes{pass("Consumer", {texture(0, RenderResourceAccess::Read)}, true)};
    const std::vector<RGResourceValidationInfo> textures{kImported};

    CHECK(validateDeclarations(passes, textures, {}).empty());
}

TEST_CASE("A declared history read is not an issue", "[rendergraph][declarations]")
{
    // The main pass reading the previous frame's ambient occlusion, which GTAO
    // only writes later in the frame.
    std::vector<RenderPassNode> passes{
        pass("MainHDR", {historyTexture(0), texture(1, RenderResourceAccess::Write)}, true),
        pass("GtaoBlur", {texture(0, RenderResourceAccess::Write)}, true),
    };
    const std::vector<RGResourceValidationInfo> textures{kTransient, kTransient};

    CHECK(validateDeclarations(passes, textures, {}).empty());
}

TEST_CASE("A history read of a pool-bound resource is reported", "[rendergraph][declarations]")
{
    // Aliasing gives the bytes to another resource between frames and the
    // handoff barrier discards the contents, so there is no previous frame left
    // to read. This is the rule that says which transients may be aliased.
    std::vector<RenderPassNode> passes{pass("MainHDR", {historyTexture(0)}, true)};
    const std::vector<RGResourceValidationInfo> textures{kAliasedTransient};

    const auto issues = validateDeclarations(passes, textures, {});

    REQUIRE(issues.size() == 1);
    CHECK(issues[0].issue == RGDeclarationIssue::HistoryReadOfAliasedResource);
}

TEST_CASE("A producer earlier in the frame satisfies the read", "[rendergraph][declarations]")
{
    std::vector<RenderPassNode> passes{
        pass("Producer", {texture(0, RenderResourceAccess::Write)}),
        pass("Consumer", {texture(0, RenderResourceAccess::Read)}, true),
    };
    const std::vector<RGResourceValidationInfo> textures{kTransient};

    CHECK(validateDeclarations(passes, textures, {}).empty());
}

TEST_CASE("A producer later in the frame does not satisfy the read", "[rendergraph][declarations]")
{
    // The ordering mistake the check exists for: the consumer runs first and
    // samples whatever was in the image, not what the producer will put there.
    std::vector<RenderPassNode> passes{
        pass("Consumer", {texture(0, RenderResourceAccess::Read)}, true),
        pass("Producer", {texture(0, RenderResourceAccess::Write)}, true),
    };
    const std::vector<RGResourceValidationInfo> textures{kTransient};

    const auto issues = validateDeclarations(passes, textures, {});

    REQUIRE(issues.size() == 1);
    CHECK(issues[0].passIndex == 0);
    CHECK(issues[0].issue == RGDeclarationIssue::ReadsContentNoPassProduced);
}

TEST_CASE("A read-modify-write is judged on what existed before the pass", "[rendergraph][declarations]")
{
    // The pass's own write must not retroactively satisfy its own read, or an
    // accumulate over an unproduced target would look well-formed.
    std::vector<RenderPassNode> passes{pass("Accumulate", {texture(0, RenderResourceAccess::ReadWrite)}, true)};
    const std::vector<RGResourceValidationInfo> textures{kTransient};

    const auto issues = validateDeclarations(passes, textures, {});

    REQUIRE(issues.size() == 1);
    CHECK(issues[0].issue == RGDeclarationIssue::ReadsContentNoPassProduced);
}

TEST_CASE("A culled pass neither produces nor reports", "[rendergraph][declarations]")
{
    // Its declarations describe work that will not run, so it cannot satisfy a
    // later read, and its own reads cannot be wrong.
    std::vector<RenderPassNode> passes{
        pass("DeadProducer", {texture(0, RenderResourceAccess::Write)}),
        pass("DeadConsumer", {texture(1, RenderResourceAccess::Read)}),
        pass("LiveConsumer", {texture(0, RenderResourceAccess::Read)}, true),
    };
    passes[0].culled = true;
    passes[1].culled = true;
    const std::vector<RGResourceValidationInfo> textures{kTransient, kTransient};

    const auto issues = validateDeclarations(passes, textures, {});

    REQUIRE(issues.size() == 1);
    CHECK(issues[0].passIndex == 2);
}

TEST_CASE("Declaring one resource twice in a pass is reported", "[rendergraph][declarations]")
{
    // Two barriers for one resource cannot share a dependency info, so this
    // costs a barrier submission on top of being a likely copy-paste slip.
    std::vector<RenderPassNode> passes{
        pass("Producer", {texture(0, RenderResourceAccess::Write)}),
        pass("Consumer", {texture(0, RenderResourceAccess::Read), texture(0, RenderResourceAccess::Read)}, true),
    };
    const std::vector<RGResourceValidationInfo> textures{kTransient};

    const auto issues = validateDeclarations(passes, textures, {});

    REQUIRE(issues.size() == 1);
    CHECK(issues[0].passIndex == 1);
    CHECK(issues[0].issue == RGDeclarationIssue::ResourceDeclaredTwice);
}

TEST_CASE("Texture and buffer declarations are checked independently", "[rendergraph][declarations]")
{
    // Both index spaces start at 0, so a check that ignored the kind would let a
    // write to texture 0 satisfy a read of buffer 0.
    std::vector<RenderPassNode> passes{
        pass("WritesTexture", {texture(0, RenderResourceAccess::Write)}),
        pass("ReadsBuffer", {buffer(0, RenderResourceAccess::Read)}, true),
    };
    const std::vector<RGResourceValidationInfo> textures{kTransient};
    const std::vector<RGResourceValidationInfo> buffers{kTransient};

    const auto issues = validateDeclarations(passes, textures, buffers);

    REQUIRE(issues.size() == 1);
    CHECK(issues[0].resourceKind == RGResourceKind::Buffer);
    CHECK(issues[0].issue == RGDeclarationIssue::ReadsContentNoPassProduced);
}

TEST_CASE("Usages past the resource count are ignored", "[rendergraph][declarations]")
{
    std::vector<RenderPassNode> passes{pass("Stale", {texture(99, RenderResourceAccess::Read)}, true)};
    const std::vector<RGResourceValidationInfo> textures{kTransient};

    CHECK(validateDeclarations(passes, textures, {}).empty());
}

TEST_CASE("Validating an empty graph is a no-op", "[rendergraph][declarations]")
{
    const std::vector<RenderPassNode> passes;

    CHECK(validateDeclarations(passes, {}, {}).empty());
}

TEST_CASE("Every issue has a distinct name", "[rendergraph][declarations]")
{
    using ve::renderer::renderGraphDeclarationIssueName;

    const std::string produced = renderGraphDeclarationIssueName(RGDeclarationIssue::ReadsContentNoPassProduced);
    const std::string aliased = renderGraphDeclarationIssueName(RGDeclarationIssue::HistoryReadOfAliasedResource);
    const std::string twice = renderGraphDeclarationIssueName(RGDeclarationIssue::ResourceDeclaredTwice);

    CHECK_FALSE(produced.empty());
    CHECK(produced != aliased);
    CHECK(aliased != twice);
    CHECK(produced != twice);
}

// ---------------------------------------------------------------------------
// Barrier batching. A pass's inferred barriers go into one vkCmdPipelineBarrier2,
// which is only sound while no two of them target the same resource: barriers
// inside one dependency info are unordered relative to each other, so a resource
// transitioned twice in a pass would have its two transitions race. Getting this
// wrong produces no validation error, just a resource occasionally read in the
// wrong layout.

TEST_CASE("An empty batch never needs a flush", "[rendergraph][barriers]")
{
    using ve::renderer::barrierBatchNeedsFlush;

    const std::vector<uint32_t> batched{};

    CHECK_FALSE(barrierBatchNeedsFlush(batched, 0));
    CHECK_FALSE(barrierBatchNeedsFlush(batched, 7));
}

TEST_CASE("Distinct resources share one barrier submission", "[rendergraph][barriers]")
{
    using ve::renderer::barrierBatchNeedsFlush;

    // The common case, and the entire point of batching: a pass reading four
    // different textures emits one barrier call, not four.
    const std::vector<uint32_t> batched{0, 1, 2};

    CHECK_FALSE(barrierBatchNeedsFlush(batched, 3));
}

TEST_CASE("A resource already in the batch forces a flush", "[rendergraph][barriers]")
{
    using ve::renderer::barrierBatchNeedsFlush;

    const std::vector<uint32_t> batched{4, 9, 2};

    CHECK(barrierBatchNeedsFlush(batched, 4));
    CHECK(barrierBatchNeedsFlush(batched, 9));
    CHECK(barrierBatchNeedsFlush(batched, 2));
}

TEST_CASE("Texture and buffer indices do not collide", "[rendergraph][barriers]")
{
    using ve::renderer::barrierBatchNeedsFlush;

    // Textures and buffers are separate tables, so index 3 in one says nothing
    // about index 3 in the other. The caller keeps a list per kind; this only
    // checks that the rule itself carries no cross-kind assumption.
    const std::vector<uint32_t> batchedTextures{3};
    const std::vector<uint32_t> batchedBuffers{};

    CHECK(barrierBatchNeedsFlush(batchedTextures, 3));
    CHECK_FALSE(barrierBatchNeedsFlush(batchedBuffers, 3));
}

// ---------------------------------------------------------------------------
// Resource lifetimes for the transient memory allocator. An interval that is too
// short lets two live resources share bytes, which corrupts the frame; one that
// is too long is safe but silently prevents the sharing the allocator exists to
// do. Neither shows up as a validation error.

TEST_CASE("A texture used by one pass has a single-pass lifetime")
{
    std::vector<RenderPassNode> passes = {
        pass("write", {texture(0, RenderResourceAccess::Write)}, /*sideEffect=*/true),
    };

    const auto lifetimes = ve::renderer::computeTextureLifetimes(passes, 1);

    REQUIRE(lifetimes.size() == 1);
    REQUIRE(lifetimes[0].used);
    REQUIRE(lifetimes[0].firstPass == 0);
    REQUIRE(lifetimes[0].lastPass == 0);
}

TEST_CASE("A lifetime spans the first and last pass that touch a texture")
{
    std::vector<RenderPassNode> passes = {
        pass("produce", {texture(0, RenderResourceAccess::Write)}),
        pass("unrelated", {texture(1, RenderResourceAccess::Write)}, /*sideEffect=*/true),
        pass("consume", {texture(0, RenderResourceAccess::Read)}, /*sideEffect=*/true),
    };

    const auto lifetimes = ve::renderer::computeTextureLifetimes(passes, 2);

    REQUIRE(lifetimes[0].firstPass == 0);
    REQUIRE(lifetimes[0].lastPass == 2);
    REQUIRE(lifetimes[1].firstPass == 1);
    REQUIRE(lifetimes[1].lastPass == 1);
}

TEST_CASE("An untouched texture reports an empty lifetime")
{
    std::vector<RenderPassNode> passes = {
        pass("write", {texture(0, RenderResourceAccess::Write)}, /*sideEffect=*/true),
    };

    const auto lifetimes = ve::renderer::computeTextureLifetimes(passes, 3);

    REQUIRE_FALSE(lifetimes[1].used);
    // The convention TransientAllocationRequest uses to drop a resource.
    REQUIRE(lifetimes[1].firstPass > lifetimes[1].lastPass);
}

TEST_CASE("Culled passes do not extend a lifetime")
{
    std::vector<RenderPassNode> passes = {
        pass("live", {texture(0, RenderResourceAccess::Write)}, /*sideEffect=*/true),
        pass("dead", {texture(0, RenderResourceAccess::Read)}),
        pass("tail", {texture(1, RenderResourceAccess::Write)}, /*sideEffect=*/true),
    };
    passes[1].culled = true;

    const auto lifetimes = ve::renderer::computeTextureLifetimes(passes, 2);

    // Counting the culled pass would stretch texture0 to pass 1 and stop it
    // sharing bytes with anything that starts there.
    REQUIRE(lifetimes[0].lastPass == 0);
}

TEST_CASE("Lifetimes ignore handles past the texture count")
{
    std::vector<RenderPassNode> passes = {
        pass("write", {texture(0, RenderResourceAccess::Write), texture(9, RenderResourceAccess::Read)},
             /*sideEffect=*/true),
    };

    const auto lifetimes = ve::renderer::computeTextureLifetimes(passes, 1);

    REQUIRE(lifetimes.size() == 1);
    REQUIRE(lifetimes[0].used);
}

TEST_CASE("Buffer usages never appear in texture lifetimes")
{
    std::vector<RenderPassNode> passes = {
        pass("compute", {buffer(0, RenderResourceAccess::Write)}, /*sideEffect=*/true),
        pass("draw", {texture(0, RenderResourceAccess::Write)}, /*sideEffect=*/true),
    };

    const auto lifetimes = ve::renderer::computeTextureLifetimes(passes, 1);

    REQUIRE(lifetimes[0].firstPass == 1);
    REQUIRE(lifetimes[0].lastPass == 1);
}

TEST_CASE("Lifetimes run after culling, matching how the graph will call them")
{
    std::vector<RenderPassNode> passes = {
        pass("orphan", {texture(0, RenderResourceAccess::Write)}),
        pass("present", {texture(1, RenderResourceAccess::Write)}, /*sideEffect=*/true),
    };

    cullUnusedPasses(passes, 2, 0);
    const auto lifetimes = ve::renderer::computeTextureLifetimes(passes, 2);

    // texture0's only writer was culled, so nothing is live for it at all.
    REQUIRE(passes[0].culled);
    REQUIRE_FALSE(lifetimes[0].used);
    REQUIRE(lifetimes[1].used);
}
