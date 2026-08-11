// Render graph pass culling: the backward liveness sweep that decides which
// declared passes actually run. It is the graph's one piece of non-trivial pure
// logic, and a mistake in it either drops work that was needed (missing shadows,
// a black screen) or keeps work that was not, silently, with no validation error
// either way.

#include "renderer/RenderGraph.h"

#include <catch2/catch_test_macros.hpp>

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
