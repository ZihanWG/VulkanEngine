#include "renderer/FrameCaptureRequest.h"

#include <cstdio>

namespace ve::renderer {

std::string frameSequenceFileName(uint32_t index)
{
    char name[32]{};
    std::snprintf(name, sizeof(name), "frame_%04u.png", static_cast<unsigned>(index));
    return name;
}

std::filesystem::path FrameCaptureRequest::outputPathFor(uint64_t frame) const
{
    if (!sequence) {
        return output;
    }
    return output / frameSequenceFileName(static_cast<uint32_t>(frame - firstFrame));
}

} // namespace ve::renderer
