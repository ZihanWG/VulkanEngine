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
    stream << levelPrefix(level) << ' ' << message << '\n';
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
