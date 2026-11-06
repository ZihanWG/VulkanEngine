#ifndef VE_IRRADIANCE_PROBES_GLSL
#define VE_IRRADIANCE_PROBES_GLSL

// Shader-side copy of the irradiance-probe layout constants.
//
// ve::renderer (renderer/IrradianceProbes.h) is the unit-tested reference copy;
// this is the duplicate the shaders read, the same arrangement
// ClusterGrid.h/cluster_grid.glsl and VirtualShadowMap.h/virtual_shadow_map.glsl
// already use. tools/check_shader_constants.py compares the two, so a divergence
// is a failed build rather than a wrong tile.
//
// This header exists because there used to be five copies. probe_capture.frag,
// probe_convolve.comp, probe_border.comp, probe_debug_fill.comp and
// simple_bindless.frag each declared their own, 36 declarations in all, and
// nothing compared any of them to the C++ or to each other. The sharp edge was
// kProbeAtlasTilesX: it is *derived* on the C++ side (kProbeGridX * kProbeGridY)
// and was written out as a literal 32 in four shaders, so changing the probe
// grid would have left four shaders addressing the wrong tile -- with no
// validation error, and nothing on screen but subtly wrong indirect light.
//
// The integer types match the C++ in value, not in spelling: the grid extents
// are uint because the shaders index with uint there, and the tile geometry is
// int because it is used in signed texel arithmetic. Changing a type here is a
// compile error at the use site, which is the point.

// Probe grid extents. kProbeCount is kProbeGridX * kProbeGridY * kProbeGridZ.
const uint kProbeGridX = 8u;
const uint kProbeGridY = 4u;
const uint kProbeGridZ = 8u;
const uint kProbeCount = kProbeGridX * kProbeGridY * kProbeGridZ;

// Octahedral tile resolutions, before the border texel.
const int kProbeIrradianceResolution = 8;
const int kProbeDepthResolution = 16;

// One texel of border per side, so hardware bilinear filtering at a tile edge
// samples the wrapped neighbour rather than the tile next door.
const int kProbeBorderTexels = 1;

// Atlas tile counts. DERIVED on the C++ side -- kProbeAtlasTilesX is
// kProbeGridX * kProbeGridY and kProbeAtlasTilesY is kProbeGridZ -- so these two
// move when the grid does. That is exactly the coupling the old per-shader
// literals hid.
const int kProbeAtlasTilesX = 32;
const int kProbeAtlasTilesY = 8;

// The cube captured per probe before it is convolved into octahedral tiles.
const int kProbeCaptureFaceCount = 6;
const int kProbeCaptureFaceResolution = 16;

// Far plane for probe depth, and the units every stored distance is in.
const float kProbeMaxDistance = 64.0;

// Near plane of a raster capture face, which the traced gather uses as its rays'
// minimum distance so both gathers ignore the same sliver around the probe.
const float kProbeCaptureNearPlane = 0.05;

// Cap on probes captured per frame, which sizes the capture atlas.
const uint kMaxProbesPerFrame = 16u;

// The ray-traced gather (probe_trace.comp): threads per side of a workgroup --
// one cube face -- how far a hit's sun ray starts off the surface, the angle one
// capture texel spans, and the floor under the cosine that stretches a cone's
// footprint at grazing hits. See ve::renderer::probeRayConeTextureLod.
const uint kProbeTraceLocalSize = 16u;
const float kProbeTraceSurfaceOffset = 0.02;
const float kProbeTraceConeSpread = 0.125;
const float kProbeTraceMinCosine = 0.1;

// Rotated rays (gi.raysPerProbe > 0): at most a capture slot's worth per probe,
// wrapped through the slot's rows of the capture atlas this many to a row. See
// ve::renderer::probeRayDirection.
const uint kProbeMaxRaysPerProbe = 1536u;
const uint kProbeRayAtlasRowTexels = 96u;

// Sharpness of the depth lobe in the Chebyshev visibility test: DDGI's 50, as
// measured in docs/irradiance_probes.md#measuring-leaks. The convolution takes
// the runtime value (gi.depthLobeExponent) from its push block.
const float kProbeDepthLobeExponent = 50.0;

// Floors that keep a probe from going fully black: the smallest visibility
// weight a probe may contribute, and the weight a back-facing probe keeps.
const float kProbeMinVisibility = 0.02;
const float kProbeBackfaceFloor = 0.0;

// A traced probe more than this share of whose rays hit the back of a
// single-sided surface is inside geometry, and shading leaves it out. DDGI's
// value. See ve::renderer::probeIsInsideGeometry.
const float kProbeInsideBackFaceFraction = 0.25;

// Relocation (gi.probeRelocation): a probe nearer a front face than this share
// of the smallest spacing is moved away from it, and no offset reaches the
// second share of the spacing on any axis. See ve::renderer::probeRelocationStep.
const float kProbeMinFrontFaceFraction = 0.2;
const float kProbeMaxOffsetFraction = 0.45;

#endif // VE_IRRADIANCE_PROBES_GLSL
