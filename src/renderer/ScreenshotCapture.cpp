#include "renderer/ScreenshotCapture.h"

#include "core/Logger.h"
#include "core/PngWriter.h"
#include "rhi/VulkanContext.h"
#include "rhi/VulkanDebugUtils.h"
#include "rhi/VulkanSwapchain.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <exception>
#include <iomanip>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ve::renderer {

namespace {

std::string portfolioTimestamp()
{
    const std::time_t now = std::time(nullptr);
    std::tm localTime{};
#if defined(_WIN32)
    localtime_s(&localTime, &now);
#else
    localtime_r(&now, &localTime);
#endif

    std::ostringstream stream;
    stream << std::put_time(&localTime, "%Y%m%d_%H%M%S");
    return stream.str();
}

std::vector<uint8_t> convertScreenshotToRgba8(std::span<const std::byte> source, VkExtent2D extent, VkFormat format)
{
    const size_t pixelCount = static_cast<size_t>(extent.width) * extent.height;
    const size_t byteCount = pixelCount * 4U;
    if (source.size_bytes() < byteCount) {
        throw std::runtime_error("Screenshot readback buffer is smaller than the captured image.");
    }

    std::vector<uint8_t> rgba(byteCount);
    const auto* input = reinterpret_cast<const uint8_t*>(source.data());
    for (size_t pixel = 0; pixel < pixelCount; ++pixel) {
        const size_t offset = pixel * 4U;
        switch (format) {
        case VK_FORMAT_B8G8R8A8_UNORM:
        case VK_FORMAT_B8G8R8A8_SRGB:
            rgba[offset + 0] = input[offset + 2];
            rgba[offset + 1] = input[offset + 1];
            rgba[offset + 2] = input[offset + 0];
            rgba[offset + 3] = input[offset + 3];
            break;
        case VK_FORMAT_R8G8B8A8_UNORM:
        case VK_FORMAT_R8G8B8A8_SRGB:
            rgba[offset + 0] = input[offset + 0];
            rgba[offset + 1] = input[offset + 1];
            rgba[offset + 2] = input[offset + 2];
            rgba[offset + 3] = input[offset + 3];
            break;
        default:
            // Renderer's policy gate only forwards screenshot-capable formats, so
            // this defensive path is unreachable in practice.
            throw std::runtime_error("Unsupported screenshot swapchain format: " +
                                     std::to_string(static_cast<int>(format)));
        }
    }

    return rgba;
}

} // namespace

void ScreenshotCapture::initialize(rhi::VulkanContext& context, uint32_t frameCount, std::filesystem::path outputDirectory)
{
    context_ = &context;
    outputDirectory_ = std::move(outputDirectory);
    readbacks_.clear();
    readbacks_.resize(frameCount);
}

void ScreenshotCapture::shutdown()
{
    // VulkanBuffer is RAII; clearing the readbacks frees the GPU allocations.
    readbacks_.clear();
    context_ = nullptr;
}

bool ScreenshotCapture::hasPending() const
{
    for (const Readback& readback : readbacks_) {
        if (readback.pending) {
            return true;
        }
    }

    return false;
}

void ScreenshotCapture::recordCopy(VkCommandBuffer commandBuffer,
                                   uint32_t frameIndex,
                                   rhi::VulkanSwapchain& swapchain,
                                   uint32_t imageIndex)
{
    if (frameIndex >= readbacks_.size()) {
        status_ = "Screenshot failed: frame readback slots are not initialized.";
        Logger::warn(status_);
        return;
    }

    const VkExtent2D extent = swapchain.extent();
    const VkFormat format = swapchain.colorFormat();

    Readback& readback = readbacks_[frameIndex];
    if (readback.pending) {
        status_ = "Screenshot capture skipped: previous readback is still pending.";
        Logger::warn(status_);
        return;
    }

    const VkDeviceSize byteSize = static_cast<VkDeviceSize>(extent.width) * extent.height * 4U;
    if (!readback.buffer.valid() || readback.buffer.size() != byteSize) {
        rhi::VulkanBufferCreateInfo bufferInfo{};
        bufferInfo.size = byteSize;
        bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bufferInfo.memoryUsage = VMA_MEMORY_USAGE_AUTO;
        bufferInfo.allocationFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
        readback.buffer.createBuffer(*context_, bufferInfo);
        rhi::debug::setObjectName(context_->vkDevice(),
                                  readback.buffer.buffer(),
                                  VK_OBJECT_TYPE_BUFFER,
                                  "PortfolioScreenshotReadbackBuffer" + std::to_string(frameIndex));
    }

    // Screenshot capture sits between CompositePass and ImGuiPass. The swapchain
    // image is copied as a transfer source, then returned to color attachment
    // layout so the normal overlay/present path can continue unchanged.
    const VkImageLayout oldLayout = swapchain.imageLayout(imageIndex);
    VkImageMemoryBarrier2 toTransferBarrier{};
    toTransferBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    toTransferBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    toTransferBarrier.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    toTransferBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    toTransferBarrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
    toTransferBarrier.oldLayout = oldLayout;
    toTransferBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    toTransferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toTransferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toTransferBarrier.image = swapchain.image(imageIndex);
    toTransferBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    toTransferBarrier.subresourceRange.baseMipLevel = 0;
    toTransferBarrier.subresourceRange.levelCount = 1;
    toTransferBarrier.subresourceRange.baseArrayLayer = 0;
    toTransferBarrier.subresourceRange.layerCount = 1;

    VkDependencyInfo toTransferDependency{};
    toTransferDependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    toTransferDependency.imageMemoryBarrierCount = 1;
    toTransferDependency.pImageMemoryBarriers = &toTransferBarrier;
    vkCmdPipelineBarrier2(commandBuffer, &toTransferDependency);
    swapchain.setImageLayout(imageIndex, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

    VkBufferImageCopy copyRegion{};
    copyRegion.bufferOffset = 0;
    copyRegion.bufferRowLength = 0;
    copyRegion.bufferImageHeight = 0;
    copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copyRegion.imageSubresource.mipLevel = 0;
    copyRegion.imageSubresource.baseArrayLayer = 0;
    copyRegion.imageSubresource.layerCount = 1;
    copyRegion.imageOffset = {0, 0, 0};
    copyRegion.imageExtent = {extent.width, extent.height, 1};
    vkCmdCopyImageToBuffer(commandBuffer,
                           swapchain.image(imageIndex),
                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           readback.buffer.buffer(),
                           1,
                           &copyRegion);

    VkBufferMemoryBarrier2 hostReadBarrier{};
    hostReadBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
    hostReadBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    hostReadBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    hostReadBarrier.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
    hostReadBarrier.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
    hostReadBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    hostReadBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    hostReadBarrier.buffer = readback.buffer.buffer();
    hostReadBarrier.offset = 0;
    hostReadBarrier.size = byteSize;

    VkImageMemoryBarrier2 toColorBarrier{};
    toColorBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    toColorBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    toColorBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
    toColorBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    toColorBarrier.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    toColorBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    toColorBarrier.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    toColorBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toColorBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toColorBarrier.image = swapchain.image(imageIndex);
    toColorBarrier.subresourceRange = toTransferBarrier.subresourceRange;

    std::array<VkImageMemoryBarrier2, 1> imageBarriers{toColorBarrier};
    VkDependencyInfo afterCopyDependency{};
    afterCopyDependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    afterCopyDependency.bufferMemoryBarrierCount = 1;
    afterCopyDependency.pBufferMemoryBarriers = &hostReadBarrier;
    afterCopyDependency.imageMemoryBarrierCount = static_cast<uint32_t>(imageBarriers.size());
    afterCopyDependency.pImageMemoryBarriers = imageBarriers.data();
    vkCmdPipelineBarrier2(commandBuffer, &afterCopyDependency);
    swapchain.setImageLayout(imageIndex, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    readback.extent = extent;
    readback.format = format;
    readback.timestampedPath = outputDirectory_ / ("vulkan_engine_portfolio_" + portfolioTimestamp() + ".png");
    readback.latestPath = outputDirectory_ / "vulkan_engine_portfolio_latest.png";
    readback.pending = true;
    status_ = "Screenshot readback queued from final composite before ImGui overlay.";
}

void ScreenshotCapture::processReadback(uint32_t frameIndex)
{
    if (frameIndex >= readbacks_.size()) {
        return;
    }

    Readback& readback = readbacks_[frameIndex];
    if (!readback.pending || !readback.buffer.valid()) {
        return;
    }

    try {
        std::vector<std::byte> pixels(static_cast<size_t>(readback.buffer.size()));
        readback.buffer.download(std::span<std::byte>(pixels.data(), pixels.size()));

        const std::vector<uint8_t> rgba = convertScreenshotToRgba8(pixels, readback.extent, readback.format);
        const uint32_t rowStride = readback.extent.width * 4U;
        writePngRgba8(readback.timestampedPath, readback.extent.width, readback.extent.height, rgba, rowStride);
        writePngRgba8(readback.latestPath, readback.extent.width, readback.extent.height, rgba, rowStride);

        lastSavedPath_ = readback.timestampedPath;
        status_ = "Saved portfolio screenshot: " + readback.timestampedPath.string();
        Logger::info(status_);
    } catch (const std::exception& error) {
        status_ = std::string("Screenshot save failed: ") + error.what();
        Logger::warn(status_);
    }

    readback.pending = false;
}

} // namespace ve::renderer
