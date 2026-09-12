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

// Sharpness of the depth lobe in the Chebyshev visibility test. Blunter than
// DDGI's 50 on purpose; see docs/irradiance_probes.md.
const float kProbeDepthLobeExponent = 20.0;

// Floors that keep a probe from going fully black: the smallest visibility
// weight a probe may contribute, and the weight a back-facing probe keeps.
const float kProbeMinVisibility = 0.02;
const float kProbeBackfaceFloor = 0.2;

#endif // VE_IRRADIANCE_PROBES_GLSL
