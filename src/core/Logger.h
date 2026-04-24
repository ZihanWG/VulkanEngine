#pragma once

#include <string_view>

namespace ve {

class Logger final {
public:
    enum class Level {
        Trace,
        Info,
        Warning,
        Error
    };

    static void log(Level level, std::string_view message);
    static void trace(std::string_view message);
    static void info(std::string_view message);
    static void warn(std::string_view message);
    static void error(std::string_view message);
};

} // namespace ve
