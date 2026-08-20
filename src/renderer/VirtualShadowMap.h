#pragma once

// GPU-free core of the directional-light virtual shadow map.
//
// Everything here is pure math over a clipmap of fixed-size pages: no Vulkan,
// no renderer state, so it is unit-testable on the CPU the same way
// CascadeMath.h, ClusterGrid.h, MeshLod.h and PunctualShadowAtlas.h are. The
// page-marking compute shader mirrors these functions, exactly as cull.comp
// mirrors MeshLod.h -- keep the two in sync.
//
// The model in one paragraph: the light's XY plane is tiled by an *absolute*
// grid of square pages, one grid per clipmap level, with level L's pages twice
// the world size of level L-1's. A page's world rect therefore never depends on
// the camera -- only on the light basis and the level. The camera decides which
// finite window of that infinite grid is addressable, and a page's slot inside
// the physical pool is its absolute coordinate taken modulo the window size.
//
// That absolute-grid choice is the whole reason page caching can work. A
// clipmap centred on the camera and snapped to *texels* would give every page a
// slightly different world rect every frame, so every page's content would
// change every frame and nothing would ever be reusable. Here a page either
// stays addressable (and its contents remain exactly correct) or scrolls out of
// the window entirely.

#include <cstddef>
#include <cstdint>
#include <vector>

#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

namespace ve::renderer {

// --- Fixed geometry -------------------------------------------------------

// Texels along one edge of a page. 128 matches the granularity Unreal's VSM
// uses; it is large enough that the per-page bookkeeping is cheap and small
// enough that a page is rarely wasted on geometry that only partly covers it.
inline constexpr uint32_t kVsmPageSize = 128;

// Pages along one edge of a clipmap level's addressable window. Sixteen gives a
// 2048x2048 virtual texel grid per level.
//
// This is a coverage constant, not a quality one, and eight was measured to be
// too small. A point at distance d selects a level whose texel is about
// d/projScaleY across, so that level's window spans roughly
// (axis/2) * kVsmPageSize * d / projScaleY. For the point to fall inside the
// window that selected it, the window has to be at least d wide, which needs
//
//     axis >= 2 * projScaleY / kVsmPageSize
//
// At 1080p that is about 12. With axis = 8 the geometry-stress scene requested
// 38 pages at one texel per pixel and ZERO at a quarter of that -- every page it
// wanted was outside its own level's window, which in the sampling phase is not
// a blurry shadow but no shadow at all. vsmSelectLevel now also enforces the
// coverage bound directly, so this constant only sets how much quality survives
// rather than whether anything is addressable.
inline constexpr uint32_t kVsmPagesPerLevelAxis = 16;
inline constexpr uint32_t kVsmPagesPerLevel = kVsmPagesPerLevelAxis * kVsmPagesPerLevelAxis;
inline constexpr uint32_t kVsmLevelResolution = kVsmPagesPerLevelAxis * kVsmPageSize;

// Ceiling on clipmap levels. Twelve levels at a 4 m level-0 extent reach ~8 km,
// which is past any distance this renderer's shadow settings expose.
inline constexpr uint32_t kVsmMaxClipmapLevels = 12;

// Every page that can exist at once, across every level.
inline constexpr uint32_t kVsmMaxVirtualPages = kVsmMaxClipmapLevels * kVsmPagesPerLevel;

// The page-request set is a bitmask the marking dispatch atomicOr's into, one
// bit per virtual page -- a few hundred bytes, small enough to read back every
// frame without a staging strategy.
inline constexpr uint32_t kVsmPageRequestWordCount = (kVsmMaxVirtualPages + 31u) / 32u;

// --- Physical pool --------------------------------------------------------

// One depth image holding the rendered pages. 4096x4096 at 128px pages is 1024
// physical pages against kVsmMaxVirtualPages virtual ones.
//
// The virtual set is deliberately larger than the pool, so the page table is a
// real indirection and the rendering phase needs a residency policy. That is not
// a design preference: the coverage bound above forces enough pages per level
// that a one-to-one mapping would need a pool no GPU here has the memory for.
// What makes it tractable is the measurement -- the resident set on both the
// default and geometry-stress scenes is around 40 pages, so an allocator over
// 1024 slots is never under pressure.
inline constexpr uint32_t kVsmPagePoolSize = 4096;
inline constexpr uint32_t kVsmPagePoolPagesPerAxis = kVsmPagePoolSize / kVsmPageSize;
inline constexpr uint32_t kVsmPagePoolPageCount = kVsmPagePoolPagesPerAxis * kVsmPagePoolPagesPerAxis;

// --- Per-frame budget -----------------------------------------------------

// Pages rendered in one frame. The page pass is shaped like the punctual shadow
// atlas pass (one rendering scope, one viewport/scissor and one indirect draw
// per slot), so PunctualShadows::kMaxGpuCulledSlots = 64 was the starting guess.
//
// Measured up from it: the page-marking pass requests 99 pages on the default
// scene and 60 on geometry stress, both at 1080p. A budget below the resident
// set is not wrong -- caching is what keeps the per-frame redraw far under it --
// but a COLD start has to fill the whole set, and at 64 that takes two frames of
// visibly wrong shadow. 128 covers both measured scenes outright.
inline constexpr uint32_t kMaxVsmPagesPerFrame = 128;

// Casters whose commands fit in one page's region of the compacted indirect
// buffer. A page covers a small world rect, so this is generous in practice --
// and going over is COUNTED and reported rather than silently dropped, the same
// contract FrameCapacity uses for over-cap geometry.
inline constexpr uint32_t kMaxVsmCastersPerPage = 256;

// --- Settings -------------------------------------------------------------

struct VsmClipmapSettings {
    // Active clipmap levels, clamped into [1, kVsmMaxClipmapLevels].
    uint32_t levelCount = 8;
    // World units spanned by the whole virtual texel grid of level 0. The finest
    // texel is therefore level0Extent / kVsmLevelResolution across.
    float level0Extent = 4.0f;
    // Multiplies the world-space texel size a pixel asks for. Above 1.0 selects
    // coarser levels (cheaper, blurrier); below 1.0 selects finer ones.
    float texelsPerPixel = 1.0f;
    // Half-thickness of every level's ortho depth range, in world units along
    // the light direction. Casters outside it do not shadow.
    float depthRange = 250.0f;
};

// Clamps a settings block into its supported ranges. Free function so the
// clamping is part of the tested surface rather than living in the UI.
[[nodiscard]] VsmClipmapSettings clampVsmClipmapSettings(const VsmClipmapSettings& settings);

// --- Light basis ----------------------------------------------------------

// View matrix whose XY plane the clipmap grids tile and whose -Z runs along the
// light direction. Translation-free on purpose: the grids are absolute, so the
// basis must not move with the camera or every page's world rect would move
// with it.
//
// `lightDirection` is the direction the light travels along and is normalized
// internally; a degenerate or up-parallel direction falls back to a fixed basis
// rather than producing NaNs.
[[nodiscard]] glm::mat4 vsmLightView(const glm::vec3& lightDirection);

// --- Level geometry -------------------------------------------------------

// World units across one page at this level. Level L doubles level L-1.
[[nodiscard]] float vsmPageWorldSize(const VsmClipmapSettings& settings, uint32_t level);

// World units across one texel at this level.
[[nodiscard]] float vsmTexelWorldSize(const VsmClipmapSettings& settings, uint32_t level);

// Coarsest level whose addressable window can still reach a point this far from
// the camera.
//
// The window is kVsmPagesPerLevelAxis pages across, centred on the camera's own
// page, so the guaranteed reach from the camera is one page short of half of
// that -- the camera sits somewhere inside its page rather than at its centre.
// A level finer than this has no slot for the point at all, which is why
// selection takes the maximum of this and the quality level rather than trusting
// quality alone.
[[nodiscard]] uint32_t vsmMinLevelForCoverage(const VsmClipmapSettings& settings, float distanceToCamera);

// Finest level that both resolves and reaches a point at `distanceToCamera`.
//
// Quality side: the finest level whose texel is no larger than the world-space
// footprint one screen pixel covers there. `projScaleY` is
// viewportHeight * 0.5 * |proj[1][1]|, the same quantity mesh LOD selection and
// punctual shadow tile sizing already use, so "how big does this look" means one
// thing across the engine. The abs() is the caller's responsibility for the same
// reason it is in cull.comp: these projections carry the Vulkan Y-flip, so
// proj[1][1] is negative.
//
// Coverage side: vsmMinLevelForCoverage above. Taking the maximum is what makes
// the result always addressable -- without it, a low texelsPerPixel selects
// levels whose windows do not reach the geometry and the page set comes back
// empty, which was measured before this bound existed.
//
// Returns 0 for a non-positive projScaleY or a degenerate distance, and never
// returns past levelCount - 1.
[[nodiscard]] uint32_t vsmSelectLevel(const VsmClipmapSettings& settings,
                                      float distanceToCamera,
                                      float projScaleY);

// --- Absolute page grid ---------------------------------------------------

// Absolute page coordinates of the page containing a light-space XY position.
// These are signed and unbounded: they name a page in the infinite grid, not a
// slot in the pool.
[[nodiscard]] glm::ivec2 vsmAbsolutePageCoords(const VsmClipmapSettings& settings,
                                               uint32_t level,
                                               const glm::vec2& lightSpaceXy);

// Minimum corner of the addressable window at this level: the kVsmPagesPerLevelAxis
// square of absolute pages centred on the camera.
[[nodiscard]] glm::ivec2 vsmWindowOrigin(const VsmClipmapSettings& settings,
                                         uint32_t level,
                                         const glm::vec2& cameraLightSpaceXy);

// Whether an absolute page is inside the window that starts at `windowOrigin`.
// A page outside it has no slot and cannot be sampled -- which is exactly what
// makes the sampler fall back to a coarser level.
[[nodiscard]] bool vsmPageInWindow(const glm::ivec2& absolutePage, const glm::ivec2& windowOrigin);

// The one absolute page inside `windowOrigin`'s window that occupies `slot`.
//
// The inverse of vsmSlotIndex within a window, and the reason the page-request
// bitmask needs no absolute coordinates of its own: a slot plus the window it
// was marked against names exactly one page.
[[nodiscard]] glm::ivec2 vsmAbsolutePageForSlot(const glm::ivec2& windowOrigin, uint32_t slot);

// Slot of an absolute page inside its level, as a toroidal wrap rather than a
// window-relative offset.
//
// This is the second half of why caching works. Window-relative indexing would
// renumber every slot the moment the window scrolls by one page, so a scroll of
// one page would invalidate all 64. Wrapping means a scroll invalidates only the
// row or column that actually changed identity.
[[nodiscard]] uint32_t vsmSlotIndex(const glm::ivec2& absolutePage);

// Flat page id across every level, and its inverse.
[[nodiscard]] uint32_t vsmPageId(uint32_t level, uint32_t slot);
[[nodiscard]] uint32_t vsmPageLevel(uint32_t pageId);
[[nodiscard]] uint32_t vsmPageSlot(uint32_t pageId);

// --- Physical pool addressing --------------------------------------------

// A page's texel rect inside the physical pool, in the same shape the page pass
// turns into a viewport and scissor.
struct VsmPageRect {
    uint32_t x = 0;
    uint32_t y = 0;
    uint32_t size = kVsmPageSize;
};

[[nodiscard]] VsmPageRect vsmPagePoolRect(uint32_t physicalPage);

// UV offset/scale of a physical page inside the pool, the pair the sampling
// path maps page-local UV through. Mirrors shadowAtlasRectUvOffsetScale.
[[nodiscard]] glm::vec4 vsmPagePoolUvOffsetScale(const VsmPageRect& rect);

// --- Page table and residency --------------------------------------------

// No physical page is bound to this virtual page.
inline constexpr uint32_t kVsmInvalidPhysicalPage = 0xFFFFFFFFu;

// One virtual page's entry, as both the allocator and the shaders read it.
//
// The absolute coordinates are the load-bearing field. A slot is a toroidal
// wrap, so when the window scrolls the same slot comes to name a *different*
// absolute page -- and an entry that recorded only "resident" would then hand
// the sampler depth rendered for somewhere else. Storing the identity lets every
// reader check it, which is what makes scroll invalidation free rather than a
// CPU-side sweep that has to be kept in step with the window.
//
// std430: four 4-byte scalars, so the runtime-array stride is 16 with no
// padding, and the GLSL mirror is the same four fields in the same order.
struct VsmPageTableEntry {
    uint32_t physicalPage = kVsmInvalidPhysicalPage;
    // Absolute page this physical page currently holds. Meaningless when
    // physicalPage is invalid.
    int32_t absoluteX = 0;
    int32_t absoluteY = 0;
    // 1 once the page has actually been rendered. Allocation and rendering are
    // separate steps -- a page can own physical space for a frame before its
    // depth exists -- and sampling an allocated-but-undrawn page is sampling
    // whatever the previous occupant left there.
    uint32_t rendered = 0;
};

static_assert(sizeof(VsmPageTableEntry) == 16);
static_assert(offsetof(VsmPageTableEntry, physicalPage) == 0);
static_assert(offsetof(VsmPageTableEntry, absoluteX) == 4);
static_assert(offsetof(VsmPageTableEntry, absoluteY) == 8);
static_assert(offsetof(VsmPageTableEntry, rendered) == 12);

// What one acquire() did, so the caller can tell a cache hit from work.
struct VsmPageAcquireResult {
    uint32_t physicalPage = kVsmInvalidPhysicalPage;
    // The page already holds depth for exactly these absolute coordinates.
    bool alreadyRendered = false;
    // A different virtual page lost its physical page to this one.
    bool evicted = false;
};

// Residency for the physical page pool: which virtual page owns which physical
// page, and what to throw out when the pool is full.
//
// GPU-free and unit-tested, like the rest of this header. The pool is 1024 pages
// and the measured resident set is around 100, so eviction is a correctness
// backstop rather than a hot path -- but it has to exist, because the virtual
// page set (kVsmMaxVirtualPages) is deliberately larger than the pool.
class VsmPageAllocator final {
public:
    VsmPageAllocator();

    // Drops all residency. For anything that changes what a page's depth would
    // contain but is not visible in a page's identity: a scene switch, the light
    // direction moving, a clipmap settings change.
    void reset();

    // Starts a frame. `frameCounter` only has to increase; it is what makes
    // "least recently used" mean something.
    void beginFrame(uint64_t frameCounter);

    // Binds a physical page to this virtual page, evicting the least recently
    // used one if the pool is full. A page acquired earlier this frame is never
    // evicted, so a frame can never starve itself.
    //
    // Returns the physical page and whether its depth is already correct for
    // these absolute coordinates. A hit needs no rendering; anything else does.
    VsmPageAcquireResult acquire(uint32_t pageId, const glm::ivec2& absolutePage);

    // Records that this page's depth has been drawn. Separate from acquire so a
    // frame that runs out of its per-frame draw budget leaves the page allocated
    // but unrendered rather than claiming depth that was never written.
    void markRendered(uint32_t pageId);

    // The table, indexed by virtual page id. Uploaded verbatim.
    [[nodiscard]] const std::vector<VsmPageTableEntry>& entries() const { return entries_; }

    [[nodiscard]] uint32_t residentPages() const { return residentPages_; }
    [[nodiscard]] uint32_t evictionsThisFrame() const { return evictionsThisFrame_; }

private:
    [[nodiscard]] uint32_t allocatePhysicalPage();

    std::vector<VsmPageTableEntry> entries_;
    // Virtual page that owns each physical page, or kVsmInvalidPhysicalPage.
    std::vector<uint32_t> physicalOwner_;
    // Frame each physical page was last acquired on.
    std::vector<uint64_t> physicalLastUsed_;
    uint64_t frameCounter_ = 0;
    uint32_t residentPages_ = 0;
    uint32_t evictionsThisFrame_ = 0;
};

// --- Page projection ------------------------------------------------------

// Light view-projection that rasterizes exactly one page: an orthographic box
// over that page's absolute world rect, with a depth range of
// [-depthRange, +depthRange] around the light-space origin.
//
// Depth is mapped to [0, 1] with the engine's normal-Z convention (nearer is
// smaller), matching every other shadow projection here.
[[nodiscard]] glm::mat4 vsmPageViewProjection(const VsmClipmapSettings& settings,
                                              const glm::mat4& lightView,
                                              uint32_t level,
                                              const glm::ivec2& absolutePage);

// --- Page request bitmask -------------------------------------------------

// Word and bit a page id occupies in the request bitmask. Both the marking
// shader and the CPU reader use these, so an off-by-one shows up as a failing
// unit test rather than as pages that are silently never requested.
[[nodiscard]] uint32_t vsmRequestWordIndex(uint32_t pageId);
[[nodiscard]] uint32_t vsmRequestBitMask(uint32_t pageId);

// What one frame's marking dispatch asked for. Reported in the debug UI and in
// a log line, because a GPU-side number that only exists in ImGui cannot be
// verified from a headless session.
struct VsmPageRequestStats {
    uint32_t requestedPages = 0;
    uint32_t requestedPerLevel[kVsmMaxClipmapLevels]{};
    uint32_t highestRequestedLevel = 0;
    uint32_t lowestRequestedLevel = 0;
};

// Decodes a read-back request bitmask. `words` must point at
// kVsmPageRequestWordCount uints.
[[nodiscard]] VsmPageRequestStats vsmDecodeRequestStats(const uint32_t* words, uint32_t levelCount);

} // namespace ve::renderer
