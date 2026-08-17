#include "renderer/MeshLod.h"

#include "core/Logger.h"

#include <meshoptimizer.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

namespace ve::renderer {

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
        const size_t targetCount =
            (static_cast<size_t>(static_cast<double>(sourceIndices.size()) * ratio) / 3U) * 3U;
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
            static_cast<double>(resultCount) >
                static_cast<double>(previousCount) * (1.0 - settings.minReduction)) {
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
        build.logMessage = "LOD chain for '" + std::string(debugName) +
                           "': L0=" + std::to_string(build.sourceIndexCount / 3) + "tri";
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

    const LodChainBuild build = buildLodChainDetached(
        std::span<const uint32_t>(indices.data() + firstIndex, indexCount),
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
