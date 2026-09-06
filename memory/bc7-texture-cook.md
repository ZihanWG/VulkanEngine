---
name: bc7-texture-cook
description: "P1 of the asset pipeline (BC7/BC5 texture cook) — COMPLETE and fully MERGED to main (d5e83be + 1ecabe2, both --no-ff, CI green). MEASURED on Sponza: 365.98 -> 91.06 MiB (4.02x), 72 of 77 block compressed, decode wait to zero, upload -46%. The async transfer queue is now unblocked."
metadata: 
  node_type: memory
  type: project
  originSessionId: 65c520fd-110a-4da1-9bfd-08132047b63e
  modified: 2026-08-17T15:13:35.518Z
---

P1 of [[asset-pipeline-initiative]]: cook textures to BC7/BC5 with baked mips so
the runtime stops uploading 366 MiB of RGBA8.

**The offline half is MERGED** — PR #8 merged `--no-ff` as **`d5e83be`**, pushed;
`main` == `origin/main`, clean, 330 tests pass. The four commits are `7afbbd5`
vendor encoder, `7af4b67` KTX2 container, `880df9f` format policy, `8a0b8d1` the
`vecook` tool. All 4 CI checks were green before merge (both MSVC jobs, Ubuntu,
lavapipe render+validation gate). The cross-platform worry about
`bc7decomp.cpp`'s `<immintrin.h>` (guarded out on this ARM Mac, live on x86
runners) **did not materialise** — the vendored encoder and decoder compile clean
on MSVC and Ubuntu with no per-platform CMake work.

Both feature branches were deleted after merging; `main` is the only branch.

## What exists now

- `external/bc7enc/` — bc7enc_rdo pinned at `e6990bc1`, **8 files** (the 6
  planned plus `bc7decomp.cpp/.h` for `--verify`). Own CMake target, warning
  exempt like imgui. Only `vecook` links it.
- `src/assets/Ktx2.h/.cpp` in Core — writer **and parser**. The parser was not
  scope creep: it is the test instrument, and it hands out level offsets rather
  than copies, which is what the future per-mip `vkCmdCopyBufferToImage` wants.
- `src/assets/TextureCook.h/.cpp` in Core — `cookedFormatForUsage`,
  `chooseTextureFormat`, `cookedFormatUsable`, `generateMipChainRgba8`,
  `extractRgba8Block`, `encodeRgba8BlockRows`.
- `tools/vecook/main.cpp` — the host tool. No Vulkan link, by design.
- `docs/asset_system.md` has the full contract table.

## Decisions that are now settled, don't re-litigate

- **Format follows the material slot, not the file.** baseColor/emissive →
  BC7_SRGB, metallicRoughness/occlusion → BC7_UNORM, normal → BC5_UNORM.
- **BC7 and BC5 caps are separate booleans**, because a device can have one
  without the other. RGBA8 fallback **keeps the sRGB decode**.
- **Mips average in linear light for sRGB.** Test pins 191 (wrong) vs 225
  (right) on one-black-three-white.
- **Non-multiple-of-4 extents clamp**, not zero-fill, not resize.
- Perceptual YCbCr error weights for colour, linear weights for data maps.

## Measured on M3, Release (this machine, not assumed)

Synthetic 1024²: base level **exactly 4.00x** (4.00 MiB → 1.00 MiB), matching the
earlier standalone de-risk. **3.00x with the mip chain**, because baked mips add
~1/3 — quote the 3.00x when talking about upload size, the 4.00x is base only.
Full BC7 chain ~100 ms across the JobSystem pool; **BC5 is ~10x cheaper than
BC7** (7-10 ms). The earlier "370 ms single-threaded" figure is not comparable —
different image, and `--threads 1` still runs the calling thread too.

## The verification trick that worked

There is **no KTX2 validator installed on this machine** (no `ktx`, not in the
Vulkan SDK, not in brew). `vecook --verify` replaces it: reads the file back off
disk, decodes every level, scores PSNR against the source chain, fails under
20 dB. **Calibrated against a real negative control** — a deliberately transposed
block order measured **7.5 dB** where the correct encode measures 24.1 dB, and
the unit test caught the same transposition independently. If you touch the
container or the block loop, re-run that negative control rather than trusting
the tests alone.

## The runtime half — MERGED (`1ecabe2`, PR #9, `--no-ff`)

`ab7174f` copy plan + sidecar naming, `5e0abc5` `createFromKtx2` + wiring,
`162de84` cook driver + measurement. 334 tests, `full` green, validation tally
0/0 on the GPU, all 4 CI checks green including the lavapipe gate. `main` is
clean at `origin/main`; both feature branches still exist.

**Measured on Sponza** (Release, M3, A/B/A with the control repeated and
returning inside 3%):

| | uncooked | cooked |
| --- | --- | --- |
| texture device bytes | 365.98 MiB | **91.06 MiB (4.02x)** |
| block compressed | 0 of 77 | **72 of 77** |
| decode wait | 58.88 ms | **0.00 ms** |
| texture upload | 89.90 ms | 48.70 ms (−46%) |
| device-local used | 877.68 MiB | 621.68 MiB |

**4.02x is the whole-chain number; vecook's per-file 3.00x is NOT the same
figure** (both columns here include mips; vecook compares a cooked chain against
an uncompressed base level). Decode wait hitting exactly zero is categorical —
the decode job is never dispatched.

Design facts worth not re-deriving:
- The upload got **simpler**: 2 barriers instead of 2+2*(N-1), and
  `TRANSFER_SRC` drops off the image usage entirely. That last part is the real
  unlock — `vkCmdBlitImage` needs a graphics queue, `vkCmdCopyBufferToImage` does
  not, so the **async transfer queue is now legal**.
- Sidecars are `foo.<slot>.ktx2` beside the source, gitignored. Slot is in the
  name because one image in two slots needs two formats.
- Every failure falls back to the source, and all three were exercised on the
  GPU: stale (source touched), wrong-slot (BC7 planted as a normal map), and
  truncated. The format is re-checked **inside** `createFromKtx2`, not only at
  the call site, because a mis-slotted file is a wrong image that still renders.
- `tools/cook_textures.py` reads a **glTF, not a directory** — a directory has no
  slot information and guessing from file names is the corruption case.

**Two things I could not verify, don't claim them:** there is no visual A/B. The
Sponza startup camera frames the building from outside, and the default portfolio
scene *loads* the checker textures but never samples them, so cooking them
changes **exactly 0 pixels** — which is also why the committed golden is immune
to whether a cook ran. Judging BC7 quality needs the editor camera inside Sponza.

**Don't compare absolute timings to the older Sponza column in
`docs/asset_load_baseline.md`** (7105 ms init, 3944 ms glTF import). Byte counts
agree exactly so it is the same scene; the wall clock was almost certainly a cold
file cache right after the fetch. See [[back-to-back-or-dont-claim]].
