#pragma once

#include "rhi/VulkanBuffer.h"

#include <volk.h>

#include <cstdint>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

namespace ve::rhi {
class VulkanContext;
class VulkanSwapchain;
} // namespace ve::rhi

namespace ve::renderer {

// Portfolio screenshot capture *mechanism*: owns per-frame host-visible readback
// buffers, records the swapchain->buffer copy (with the required layout barriers)
// between the composite and ImGui passes, and writes the captured image to PNG
// after a frame of latency. Capture *policy* -- when to capture and the
// scene/lighting setup -- stays in Renderer, which drives this via recordCopy().
class ScreenshotCapture final {
public:
    ScreenshotCapture() = default;
    ~ScreenshotCapture() = default;

    ScreenshotCapture(const ScreenshotCapture&) = delete;
    ScreenshotCapture& operator=(const ScreenshotCapture&) = delete;
    ScreenshotCapture(ScreenshotCapture&&) = delete;
    ScreenshotCapture& operator=(ScreenshotCapture&&) = delete;

    void initialize(rhi::VulkanContext& context, uint32_t frameCount, std::filesystem::path outputDirectory);
    void shutdown();

    [[nodiscard]] bool hasPending() const;

    // Records, for the given in-flight frame, a copy of the swapchain color image
    // into a host-visible readback buffer and queues it for CPU readback. Callers
    // must have validated that the swapchain supports transfer-src and that its
    // format/extent are screenshot-capable (Renderer does this in its policy gate).
    //
    // explicitOutputPath overrides the timestamped + "_latest" pair with a single
    // named file. Regression capture uses it so a CI run cannot clobber the one
    // tracked portfolio screenshot.
    void recordCopy(VkCommandBuffer commandBuffer,
                    uint32_t frameIndex,
                    rhi::VulkanSwapchain& swapchain,
                    uint32_t imageIndex,
                    const std::filesystem::path& explicitOutputPath = {});

    // Downloads a previously queued readback (if ready) and writes it to PNG.
    void processReadback(uint32_t frameIndex);

    [[nodiscard]] const std::string& status() const { return status_; }
    void setStatus(std::string status) { status_ = std::move(status); }
    [[nodiscard]] const std::filesystem::path& lastSavedPath() const { return lastSavedPath_; }

private:
    struct Readback {
        rhi::VulkanBuffer buffer;
        VkExtent2D extent{};
        VkFormat format = VK_FORMAT_UNDEFINED;
        std::filesystem::path timestampedPath;
        std::filesystem::path latestPath;
        bool pending = false;
    };

    rhi::VulkanContext* context_ = nullptr;
    std::vector<Readback> readbacks_;
    std::filesystem::path outputDirectory_;
    std::string status_ = "No capture requested.";
    std::filesystem::path lastSavedPath_;
};

} // namespace ve::renderer
