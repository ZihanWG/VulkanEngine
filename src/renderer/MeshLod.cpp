#include "renderer/MeshLod.h"

#include "core/Logger.h"

#include <meshoptimizer.h>

#include <glm/geometric.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

namespace ve::renderer {

MeshletBuild buildMeshlets(std::vector<uint32_t>& indices,
                           std::span<const MeshLod> lods,
                           const float* vertexPositions,
                           size_t vertexCount,
                           size_t vertexStride,
                           const MeshletBuildSettings& settings)
{
    MeshletBuild build;
    if (vertexPositions == nullptr || vertexCount == 0 || lods.empty()) {
        return build;
    }
    build.rangesPerLod.assign(lods.size(), glm::uvec2(0));
    std::vector<Meshlet>& meshlets = build.meshlets;

    // meshopt's documented limits. Violating them is undefined rather than
    // diagnosed, so they are enforced here instead of trusted from the caller.
    const size_t maxVertices = std::clamp<size_t>(settings.maxVertices, 3, 255);
    // Must stay divisible by 4, which is why this rounds down rather than
    // clamping: a caller asking for 100 gets 100, one asking for 126 gets 124.
    const size_t maxTriangles = std::min<size_t>((settings.maxTriangles / 4U) * 4U, 512);
    if (maxTriangles == 0) {
        return build;
    }

    std::vector<meshopt_Meshlet> rawMeshlets;
    std::vector<uint32_t> meshletVertices;
    std::vector<unsigned char> meshletTriangles;
    std::vector<uint32_t> reordered;

    for (size_t levelIndex = 0; levelIndex < lods.size(); ++levelIndex) {
        const MeshLod& level = lods[levelIndex];
        const size_t rangeEnd = static_cast<size_t>(level.firstIndex) + level.indexCount;
        if (level.indexCount < 3 || rangeEnd > indices.size()) {
            continue;
        }

        const uint32_t* levelIndices = indices.data() + level.firstIndex;
        const size_t bound = meshopt_buildMeshletsBound(level.indexCount, maxVertices, maxTriangles);
        rawMeshlets.resize(bound);
        meshletVertices.resize(bound * maxVertices);
        meshletTriangles.resize(bound * maxTriangles * 3);

        const size_t produced = meshopt_buildMeshlets(rawMeshlets.data(),
                                                      meshletVertices.data(),
                                                      meshletTriangles.data(),
                                                      levelIndices,
                                                      level.indexCount,
                                                      vertexPositions,
                                                      vertexCount,
                                                      vertexStride,
                                                      maxVertices,
                                                      maxTriangles,
                                                      settings.coneWeight);
        if (produced == 0) {
            continue;
        }

        // Rebuild the level's range grouped by meshlet. Written into scratch and
        // copied back rather than shuffled in place: the expansion below reads
        // the meshlet's own vertex table, which indexes the ORIGINAL range, so
        // overwriting as we go would read indices we had already replaced.
        reordered.clear();
        reordered.reserve(level.indexCount);

        const auto meshletBase = static_cast<uint32_t>(meshlets.size());
        for (size_t index = 0; index < produced; ++index) {
            const meshopt_Meshlet& raw = rawMeshlets[index];
            const auto firstIndex = static_cast<uint32_t>(level.firstIndex + reordered.size());

            const uint32_t* localVertices = meshletVertices.data() + raw.vertex_offset;
            const unsigned char* localTriangles = meshletTriangles.data() + raw.triangle_offset;
            for (uint32_t triangle = 0; triangle < raw.triangle_count * 3U; ++triangle) {
                reordered.push_back(localVertices[localTriangles[triangle]]);
            }

            Meshlet meshlet{};
            meshlet.firstIndex = firstIndex;
            meshlet.indexCount = raw.triangle_count * 3U;

            // Computed from the emitted contiguous range rather than from the
            // meshlet's local tables: computeClusterBounds takes exactly the
            // (indices, count) pair that was just written, so the bounds
            // describe the geometry the GPU will actually draw for this range.
            const meshopt_Bounds bounds = meshopt_computeClusterBounds(reordered.data() + (firstIndex - level.firstIndex),
                                                                       meshlet.indexCount,
                                                                       vertexPositions,
                                                                       vertexCount,
                                                                       vertexStride);
            meshlet.centerRadius = glm::vec4(bounds.center[0], bounds.center[1], bounds.center[2], bounds.radius);
            // meshopt reports an unusable cone as cutoff 1, which is already the
            // "never reject" value, so this needs no special case -- but it is
            // clamped because a cutoff above 1 would make the GPU test
            // nonsensical rather than merely permissive.
            meshlet.coneAxisCutoff = glm::vec4(bounds.cone_axis[0],
                                               bounds.cone_axis[1],
                                               bounds.cone_axis[2],
                                               std::min(bounds.cone_cutoff, 1.0f));
            meshlets.push_back(meshlet);
        }

        // The grouping is a permutation of the level's triangles, so the range
        // must come back exactly the size it went in. Anything else would move
        // every later level and silently corrupt the whole buffer.
        if (reordered.size() != level.indexCount) {
            meshlets.resize(meshletBase);
            continue;
        }

        std::copy(reordered.begin(), reordered.end(), indices.begin() + level.firstIndex);
        build.rangesPerLod[levelIndex] = glm::uvec2(meshletBase, static_cast<uint32_t>(produced));
    }

    return build;
}

bool meshletConeCulled(const glm::vec3& center,
                       float radius,
                       const glm::vec3& coneAxis,
                       float coneCutoff,
                       const glm::vec3& viewPosition)
{
    // 1 is the "no usable cone" value and would otherwise be widened past 1 by
    // the radius term below into something that rejects nothing anyway -- but
    // taking the early out keeps that an explicit contract rather than an
    // accident of the arithmetic.
    if (coneCutoff >= 1.0f) {
        return false;
    }

    const glm::vec3 toCenter = center - viewPosition;
    const float distance = glm::length(toCenter);
    // Inside the meshlet's own sphere there is no single view direction to test
    // against, and the cone says nothing useful. Never reject.
    if (!(distance > radius) || !(distance > 0.0f)) {
        return false;
    }

    // Widened by the angle the bounding sphere subtends, so a meshlet whose
    // cone only just faces away is kept rather than wrongly dropped.
    return glm::dot(toCenter / distance, coneAxis) >= coneCutoff + radius / distance;
}

LodChainBuild buildLodChainDetached(std::span<const uint32_t> sourceIndices,
                                    const float* vertexPositions,
                                    size_t vertexCount,
                                    size_t vertexStride,
                                    std::string_view debugName,
                                    const LodBuildSettings& settings)
{
    LodChainBuild build;
    if (sourceIndices.empty()) {
        return build;
    }

    build.sourceIndexCount = static_cast<uint32_t>(sourceIndices.size());
    if (sourceIndices.size() < settings.minIndexCount || vertexCount == 0 || vertexPositions == nullptr) {
        return build;
    }

    std::vector<uint32_t> simplified;
    size_t previousCount = sourceIndices.size();

    for (uint32_t level = 1; level < settings.maxLods; ++level) {
        // Keep the target a multiple of 3 so the simplifier is never asked for a
        // partial triangle.
        const double ratio = std::pow(settings.indexRatio, static_cast<double>(level));
        const size_t targetCount = (static_cast<size_t>(static_cast<double>(sourceIndices.size()) * ratio) / 3U) * 3U;
        if (targetCount < settings.minIndexCount) {
            break;
        }

        simplified.resize(sourceIndices.size());
        float resultError = 0.0f;
        const size_t resultCount = meshopt_simplify(simplified.data(),
                                                    sourceIndices.data(),
                                                    sourceIndices.size(),
                                                    vertexPositions,
                                                    vertexCount,
                                                    vertexStride,
                                                    targetCount,
                                                    settings.targetError,
                                                    /*options=*/0,
                                                    &resultError);
        simplified.resize(resultCount);

        // Stop as soon as the simplifier stalls. An extra level that barely
        // removes triangles still costs index memory and a LOD-table entry, and
        // buys a switch that changes nothing on screen.
        if (resultCount == 0 ||
            static_cast<double>(resultCount) > static_cast<double>(previousCount) * (1.0 - settings.minReduction)) {
            break;
        }

        meshopt_optimizeVertexCache(simplified.data(), simplified.data(), resultCount, vertexCount);

        // Relative to this build's own buffer. appendLodChain rebases it.
        build.simplifiedLods.push_back(
            {static_cast<uint32_t>(build.simplifiedIndices.size()), static_cast<uint32_t>(resultCount)});
        build.simplifiedIndices.insert(build.simplifiedIndices.end(), simplified.begin(), simplified.end());
        previousCount = resultCount;
    }

    // Composed, not printed: this can run on a worker and Logger has no mutex.
    if (!build.simplifiedLods.empty() && !debugName.empty()) {
        build.logMessage =
            "LOD chain for '" + std::string(debugName) + "': L0=" + std::to_string(build.sourceIndexCount / 3) + "tri";
        for (size_t level = 0; level < build.simplifiedLods.size(); ++level) {
            build.logMessage += " L" + std::to_string(level + 1) + "=" +
                                std::to_string(build.simplifiedLods[level].indexCount / 3) + "tri";
        }
    }

    return build;
}

std::vector<MeshLod> appendLodChain(std::vector<uint32_t>& indices, uint32_t firstIndex, const LodChainBuild& build)
{
    std::vector<MeshLod> lods;
    if (build.sourceIndexCount == 0) {
        return lods;
    }

    lods.reserve(build.simplifiedLods.size() + 1);
    lods.push_back({firstIndex, build.sourceIndexCount});

    // Captured before the insert: every simplified level is offset from where this
    // build's block starts, and appending would move the end.
    const auto base = static_cast<uint32_t>(indices.size());
    for (const MeshLod& level : build.simplifiedLods) {
        lods.push_back({base + level.firstIndex, level.indexCount});
    }

    indices.insert(indices.end(), build.simplifiedIndices.begin(), build.simplifiedIndices.end());
    return lods;
}

std::vector<MeshLod> buildLodChain(std::vector<uint32_t>& indices,
                                   uint32_t firstIndex,
                                   uint32_t indexCount,
                                   const float* vertexPositions,
                                   size_t vertexCount,
                                   size_t vertexStride,
                                   std::string_view debugName,
                                   const LodBuildSettings& settings)
{
    if (indexCount == 0) {
        return {};
    }

    // A range that runs past the buffer cannot be simplified, but level 0 is still
    // what the caller asked for.
    const size_t rangeEnd = static_cast<size_t>(firstIndex) + indexCount;
    if (rangeEnd > indices.size()) {
        return {{firstIndex, indexCount}};
    }

    const LodChainBuild build =
        buildLodChainDetached(std::span<const uint32_t>(indices.data() + firstIndex, indexCount),
                              vertexPositions,
                              vertexCount,
                              vertexStride,
                              debugName,
                              settings);

    std::vector<MeshLod> lods = appendLodChain(indices, firstIndex, build);
    if (!build.logMessage.empty()) {
        Logger::info(build.logMessage);
    }

    return lods;
}

float projectedScreenRadius(float radius, float distance, float projScaleY)
{
    if (radius <= 0.0f || projScaleY <= 0.0f) {
        return 0.0f;
    }

    // Camera inside (or on) the bounding sphere: the object fills the view, so
    // report something large enough that selection lands on level 0.
    constexpr float kMinDistance = 1.0e-4f;
    if (distance <= kMinDistance) {
        return std::numeric_limits<float>::max();
    }

    return radius / distance * projScaleY;
}

uint32_t selectLodIndex(float projectedRadiusPixels, uint32_t lodCount, const LodSelectionSettings& settings)
{
    if (lodCount <= 1) {
        return 0;
    }

    const uint32_t maxLod = lodCount - 1;
    if (settings.forcedLod >= 0) {
        return std::min(static_cast<uint32_t>(settings.forcedLod), maxLod);
    }

    // Degenerate or vanishingly small on screen: nothing to preserve, take the
    // cheapest level.
    if (!(projectedRadiusPixels > 0.0f)) {
        return maxLod;
    }

    const float reference = std::max(settings.referenceRadiusPixels, 1.0f);
    // Each level covers one halving of the on-screen radius, so the level is just
    // how many times the projected radius fits into the reference, in octaves.
    const float level = std::log2(reference / projectedRadiusPixels) + settings.bias;
    if (!(level > 0.0f)) {
        return 0;
    }
    if (level >= static_cast<float>(maxLod)) {
        return maxLod;
    }

    return static_cast<uint32_t>(level);
}

} // namespace ve::renderer
