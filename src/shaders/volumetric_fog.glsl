#ifndef VE_VOLUMETRIC_FOG_GLSL
#define VE_VOLUMETRIC_FOG_GLSL

// Shader-side copy of the froxel-volume constants.
//
// ve::renderer (renderer/VolumetricFog.h) is the unit-tested reference copy;
// this is the duplicate the shaders read, the same arrangement
// ClusterGrid.h/cluster_grid.glsl already uses.
// tools/check_shader_constants.py compares the two.
//
// Three copies became one. fog_inject.comp and fog_integrate.comp each declared
// the block, and simple_bindless.frag declared kFogNearPlane a third time,
// function-locally, where nothing was ever going to notice it: the parity check
// was reading the two fog shaders by name and had no reason to look in the main
// shading pass.

// Froxel volume dimensions. kFogFroxelCount is X * Y * Z.
const uint kFogGridX = 160u;
const uint kFogGridY = 90u;
const uint kFogGridZ = 64u;

// Near plane of the exponential slice distribution. The main pass inverts the
// same distribution to find the slice a shaded pixel falls in, so it needs this
// number as much as the injection pass does -- which is why it lives here and
// not in a fog-only block.
const float kFogNearPlane = 0.5;

#endif // VE_VOLUMETRIC_FOG_GLSL
