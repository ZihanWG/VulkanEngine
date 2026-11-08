#include "core/Logger.h"

#include <iostream>

namespace ve {

namespace {

const char* levelPrefix(Logger::Level level)
{
    switch (level) {
    case Logger::Level::Trace:
        return "[Trace]";
    case Logger::Level::Info:
        return "[Info ]";
    case Logger::Level::Warning:
        return "[Warn ]";
    case Logger::Level::Error:
        return "[Error]";
    }

    return "[Log  ]";
}

} // namespace

void Logger::log(Level level, std::string_view message)
{
    std::ostream& stream = (level == Level::Error || level == Level::Warning) ? std::cerr : std::cout;
    // Flushed per message. std::cerr is unit-buffered already; std::cout is
    // fully buffered whenever it is redirected to a file, which is how every CI
    // leg and measurement run captures it, so a crash used to take the last few
    // kilobytes of [Info] lines with it -- exactly the lines that say how far
    // the run got. One flush per message is at most one per frame
    // (--gpu-timings-every-frame logs one block a frame), and it also keeps the
    // two streams in order when both go to the same file.
    stream << levelPrefix(level) << ' ' << message << '\n' << std::flush;
}

void Logger::trace(std::string_view message)
{
    log(Level::Trace, message);
}

void Logger::info(std::string_view message)
{
    log(Level::Info, message);
}

void Logger::warn(std::string_view message)
{
    log(Level::Warning, message);
}

void Logger::error(std::string_view message)
{
    log(Level::Error, message);
}

} // namespace ve
