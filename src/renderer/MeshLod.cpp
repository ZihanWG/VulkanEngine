#include "renderer/MeshLod.h"

#include "core/Logger.h"

#include <meshoptimizer.h>

#include <cmath>
#include <string>

namespace ve::renderer {

std::vector<MeshLod> buildLodChain(std::vector<uint32_t>& indices,
                                   uint32_t firstIndex,
                                   uint32_t indexCount,
                                   const float* vertexPositions,
                                   size_t vertexCount,
                                   size_t vertexStride,
                                   std::string_view debugName,
                                   const LodBuildSettings& settings)
{
    std::vector<MeshLod> lods;
    if (indexCount == 0) {
        return lods;
    }

    lods.push_back({firstIndex, indexCount});

    const size_t rangeEnd = static_cast<size_t>(firstIndex) + indexCount;
    if (indexCount < settings.minIndexCount || vertexCount == 0 || vertexPositions == nullptr ||
        rangeEnd > indices.size()) {
        return lods;
    }

    // The simplified levels are appended to the very buffer the source range
    // lives in, so copy the source out first — appending can reallocate.
    const std::vector<uint32_t> sourceIndices(indices.begin() + firstIndex, indices.begin() + rangeEnd);
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

        lods.push_back({static_cast<uint32_t>(indices.size()), static_cast<uint32_t>(resultCount)});
        indices.insert(indices.end(), simplified.begin(), simplified.end());
        previousCount = resultCount;
    }

    if (lods.size() > 1 && !debugName.empty()) {
        std::string message = "LOD chain for '" + std::string(debugName) + "':";
        for (size_t level = 0; level < lods.size(); ++level) {
            message += " L" + std::to_string(level) + "=" + std::to_string(lods[level].indexCount / 3) + "tri";
        }
        Logger::info(message);
    }

    return lods;
}

} // namespace ve::renderer
