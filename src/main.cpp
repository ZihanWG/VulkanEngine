#include "core/Application.h"

int main(int argc, char** argv)
{
    ve::Application::Config config{};
    if (!ve::Application::parseArguments(argc, argv, config)) {
        return -1;
    }

    ve::Application app(std::move(config));
    return app.run();
}
