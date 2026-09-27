#pragma once

// TAA disocclusion detection, the GPU-free half.
//
// A pixel is disoccluded when the surface it shows now was hidden last frame --
// behind something that has since moved away. Its reprojected history then holds
// the occluder, not the surface, and neighbourhood clamping cannot always tell:
// an occluder whose colour happens to sit inside the current neighbourhood
// survives the clamp and trails as a ghost. Depth can tell, because the
// occluder was at a different distance.
//
// So the resolve records, in the history's alpha channel, the view depth of the
// surface each history texel shows, and the next frame compares it with the
// depth the current surface *would* have had last frame. That second quantity
// needs no extra buffer: for a point of this frame's depth buffer it is the w
// of the previous frame's clip position, and both it and this frame's view
// depth reduce to two rows of matrix algebra the CPU folds once per frame.
//
// "Would have had" assumes the surface stood still: only the camera is
// reprojected, since velocity is two-dimensional and says nothing about depth.
// An object that moves along the view direction by more than the tolerance in
// one frame therefore fails the test against its own history.
//
// taa_disocclusion.glsl is the shader half and mirrors every function here; the
// constants are compared by tools/check_shader_constants.py.

#include <algorithm>
#include <array>
#include <glm/geometric.hpp>
#include <glm/mat4x4.hpp>
#include <glm/matrix.hpp>
#include <glm/vec2.hpp>
#include <glm/vec4.hpp>
#include <limits>

namespace ve::renderer {

// The history's alpha holds kTaaHistoryDepthOffset + view depth, so every value
// the resolve writes is at least 1. Two things follow: the render-target preview,
// which draws the history through ImGui's alpha blend, stays opaque; and a
// history written before depth was recorded -- alpha exactly 1 -- decodes to a
// depth of 0, which means "unknown" and never rejects anything.
inline constexpr float kTaaHistoryDepthOffset = 1.0f;

// The view depth recorded for sky: farther than any surface, and still
// representable in the history's 16-bit float alpha once the offset is added.
inline constexpr float kTaaHistorySkyDepth = 60000.0f;

// Two rows that turn a clip-space point of the current, jittered frame --
// c = (ndc.x, ndc.y, depth, 1) -- into view depths without a matrix inverse per
// pixel:
//
//   previous-frame view depth = dot(numerator, c) / dot(denominator, c)
//   current-frame view depth  = 1 / dot(denominator, c)
//
// The denominator is the w row of the inverse jittered view-projection, which
// divides the homogeneous world position; the numerator is the previous
// view-projection's w row -- clip w is view depth under a perspective
// projection -- carried through that same inverse. Jitter only shears x and y,
// so reconstructing through the jittered inverse is what matches the depth
// buffer, and it leaves the depths themselves untouched.
struct TaaPreviousDepthRows {
    glm::vec4 numerator{0.0f, 0.0f, 0.0f, 0.0f};
    glm::vec4 denominator{0.0f, 0.0f, 0.0f, 1.0f};
};

[[nodiscard]] inline TaaPreviousDepthRows taaPreviousDepthRows(const glm::mat4& jitteredViewProjection,
                                                               const glm::mat4& previousViewProjection)
{
    const glm::mat4 inverse = glm::inverse(jitteredViewProjection);
    const glm::vec4 previousWRow = glm::vec4(previousViewProjection[0][3],
                                             previousViewProjection[1][3],
                                             previousViewProjection[2][3],
                                             previousViewProjection[3][3]);
    TaaPreviousDepthRows rows;
    // Row vector times matrix: component j is the row dotted with column j.
    rows.numerator = previousWRow * inverse;
    rows.denominator = glm::vec4(inverse[0][3], inverse[1][3], inverse[2][3], inverse[3][3]);
    return rows;
}

[[nodiscard]] inline float taaPreviousViewDepth(const TaaPreviousDepthRows& rows, glm::vec2 ndc, float depth)
{
    const glm::vec4 clip(ndc, depth, 1.0f);
    return glm::dot(rows.numerator, clip) / glm::dot(rows.denominator, clip);
}

[[nodiscard]] inline float taaCurrentViewDepth(const TaaPreviousDepthRows& rows, glm::vec2 ndc, float depth)
{
    return 1.0f / glm::dot(rows.denominator, glm::vec4(ndc, depth, 1.0f));
}

[[nodiscard]] inline float taaEncodeHistoryDepth(float viewDepth)
{
    return kTaaHistoryDepthOffset + std::clamp(viewDepth, 0.0f, kTaaHistorySkyDepth);
}

// Zero or less means the history texel carries no depth and cannot be judged.
[[nodiscard]] inline float taaDecodeHistoryDepth(float alpha)
{
    return alpha - kTaaHistoryDepthOffset;
}

// True when none of the four history texels around the reprojected position can
// be the surface the current neighbourhood shows.
//
// Four texels and a neighbourhood range rather than one depth against one depth,
// because the test runs on silhouettes too: the history's bilinear footprint
// straddles an edge, and so does the current 3x3. Demanding that every texel
// match would reject the history exactly where anti-aliasing needs it. Only when
// no texel falls inside the range the current surfaces span -- widened by
// `tolerance`, relative -- is the history showing something that is not there
// any more.
//
// A neighbourhood that contains sky leaves the range open above, since sky
// history matches it. A neighbourhood with no surface at all (expectedMin above
// expectedMax) and a footprint with no recorded depth both return false: with
// nothing to compare, the history is kept.
[[nodiscard]] inline bool taaHistoryDisoccluded(const std::array<float, 4>& historyDepths,
                                                float expectedMin,
                                                float expectedMax,
                                                bool neighbourhoodHasSky,
                                                float tolerance)
{
    if (expectedMin > expectedMax) {
        return false;
    }
    const float lower = expectedMin * (1.0f - tolerance);
    const float upper = neighbourhoodHasSky ? std::numeric_limits<float>::max() : expectedMax * (1.0f + tolerance);

    bool anyKnown = false;
    for (const float depth : historyDepths) {
        if (depth <= 0.0f) {
            continue;
        }
        anyKnown = true;
        if (depth >= lower && depth <= upper) {
            return false;
        }
    }
    return anyKnown;
}

} // namespace ve::renderer
