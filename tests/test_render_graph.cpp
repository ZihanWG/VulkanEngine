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
