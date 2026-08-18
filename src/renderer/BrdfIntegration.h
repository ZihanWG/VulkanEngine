#pragma once

// Split-sum BRDF integration (Karis 2013) for the IBL lookup table.
//
// GPU-free and kept out of the Vulkan translation unit for two reasons: every
// texel is an independent pure function of its coordinates, so the table can be
// built across the job system, and the result can then be pinned by headless
// tests -- the same split used by CascadeMath.h and MeshLod.h.
//
// The cost is why it lives here at all. At the default 256x256 with 128 samples
// per texel this is 8.4 million integration steps of sin/cos/pow, which measured
// ~97 ms on one thread: 59% of scene creation, paid by every scene including an
// empty one.

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ve {
class JobSystem;
}

namespace ve::renderer {

// Scale and bias, so the LUT is two channels.
inline constexpr uint32_t kBrdfLutChannels = 2;
inline constexpr uint32_t kBrdfIntegrationSampleCount = 128;

// One texel: the split-sum integral for a view angle and roughness. Pure, and
// depends on nothing but its arguments -- which is what makes the table
// identical on every run and every machine.
[[nodiscard]] std::array<float, 2> integrateBrdf(float normalView, float roughness);

// The whole table, row major, kBrdfLutChannels bytes per texel. Rows are
// independent, so `jobSystem` spreads them; nullptr computes them serially.
//
// The bytes are identical either way. Each texel writes only its own two bytes
// and reads nothing another texel wrote, so there is no ordering to preserve --
// unlike the LOD chains, which needed a fixed append order.
//
// Throws std::runtime_error when `size` is zero.
[[nodiscard]] std::vector<uint8_t> makeSplitSumBrdfLut(uint32_t size, JobSystem* jobSystem = nullptr);

} // namespace ve::renderer
