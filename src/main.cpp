#include "core/Application.h"
#include "core/CommandLine.h"

int main(int argc, char** argv)
{
    ve::LaunchOptions options{};
    if (!ve::parseLaunchOptions(argc, argv, options)) {
        return -1;
    }

    ve::Application app(std::move(options));
    return app.run();
}
