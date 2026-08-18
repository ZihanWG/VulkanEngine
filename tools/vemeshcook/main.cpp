// Offline mesh cook: one glTF in, one .vemesh with baked LOD chains out.
// See docs/mesh_lod.md.
//
//   vemeshcook <scene.gltf> [--force] [--threads N] [--verify]
//
// Exit codes: 0 cooked or already up to date, 1 cook failure, 2 usage/IO error.
//
// A measurement is why this exists. Sponza's glTF import is ~349 ms, of which
// ~296 ms is LOD simplification and ~35 ms vertex assembly -- against ~2 ms of
// actual JSON parsing. Parallelising the simplification made it 3.3x faster but
// only spread it across cores; baking it removes it, and removes it by the same
// amount on a machine with two cores as on one with eight.
//
// Links VulkanEngineCore only. All the expensive work lives in
// renderer/GltfGeometry (parse, assembly, LOD) and renderer/MeshCache (the
// container), neither of which touches Vulkan -- which is what keeps this tool
// off the Vulkan loader and off SDL3.

#include "core/JobSystem.h"
#include "renderer/GltfGeometry.h"
#include "renderer/MeshCache.h"

#include <charconv>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Options {
    std::filesystem::path scene;
    bool force = false;
    bool verify = false;
    size_t threadCount = 0;
};

[[noreturn]] void usage()
{
    std::fprintf(stderr, "usage: vemeshcook <scene.gltf> [--force] [--threads N] [--verify]\n");
    std::exit(2);
}

Options parseArguments(int argc, char** argv)
{
    Options options{};
    std::vector<std::string> positional;

    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        if (argument == "--force") {
            options.force = true;
        } else if (argument == "--verify") {
            options.verify = true;
        } else if (argument == "--threads") {
            if (index + 1 >= argc) {
                usage();
            }
            const std::string_view value(argv[++index]);
            unsigned int threads = 0;
            const auto result = std::from_chars(value.data(), value.data() + value.size(), threads);
            if (result.ec != std::errc{} || result.ptr != value.data() + value.size()) {
                usage();
            }
            options.threadCount = threads;
        } else if (argument.rfind("--", 0) == 0) {
            std::fprintf(stderr, "vemeshcook: unknown option %s\n", std::string(argument).c_str());
            usage();
        } else {
            positional.emplace_back(argument);
        }
    }

    if (positional.size() != 1) {
        usage();
    }

    options.scene = std::filesystem::path(positional.front());
    return options;
}

std::vector<std::byte> readWholeFile(const std::filesystem::path& path)
{
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        return {};
    }

    const std::streamsize size = input.tellg();
    if (size <= 0) {
        return {};
    }

    input.seekg(0);
    std::vector<std::byte> bytes(static_cast<size_t>(size));
    input.read(reinterpret_cast<char*>(bytes.data()), size);
    if (!input) {
        return {};
    }

    return bytes;
}

// Empty when the two agree. Reports the first difference rather than a bool so a
// failure names what drifted.
std::string firstMismatch(const std::vector<ve::renderer::CpuMeshData>& built,
                          const std::vector<ve::renderer::CpuMeshData>& decoded)
{
    if (built.size() != decoded.size()) {
        return "mesh count " + std::to_string(decoded.size()) + " != " + std::to_string(built.size());
    }

    for (size_t index = 0; index < built.size(); ++index) {
        const ve::renderer::CpuMeshData& a = built[index];
        const ve::renderer::CpuMeshData& b = decoded[index];
        const std::string where = "mesh " + std::to_string(index) + ": ";

        if (a.debugName != b.debugName) {
            return where + "debug name";
        }
        if (a.indexCount != b.indexCount) {
            return where + "authored index count";
        }
        if (a.indices != b.indices) {
            return where + "index buffer";
        }
        if (a.vertices.size() != b.vertices.size()) {
            return where + "vertex count";
        }
        if (!a.vertices.empty()
            && std::memcmp(a.vertices.data(), b.vertices.data(), a.vertices.size() * sizeof(ve::renderer::Vertex))
                   != 0) {
            return where + "vertex data";
        }
        if (a.primitives.size() != b.primitives.size()
            || (!a.primitives.empty()
                && std::memcmp(a.primitives.data(),
                               b.primitives.data(),
                               a.primitives.size() * sizeof(ve::renderer::MeshPrimitive))
                       != 0)) {
            return where + "primitive table";
        }
        if (a.lods.size() != b.lods.size()
            || (!a.lods.empty()
                && std::memcmp(a.lods.data(), b.lods.data(), a.lods.size() * sizeof(ve::renderer::MeshLod)) != 0)) {
            return where + "LOD table";
        }
        if (a.localBounds.min != b.localBounds.min || a.localBounds.max != b.localBounds.max) {
            return where + "bounds";
        }
    }

    return {};
}

std::string formatMiB(size_t bytes)
{
    char text[64];
    std::snprintf(text, sizeof(text), "%.2f MiB", static_cast<double>(bytes) / (1024.0 * 1024.0));
    return text;
}

} // namespace

int main(int argc, char** argv)
{
    const Options options = parseArguments(argc, argv);

    std::error_code existsError;
    if (!std::filesystem::exists(options.scene, existsError) || existsError) {
        std::fprintf(stderr, "vemeshcook: scene does not exist: %s\n", options.scene.string().c_str());
        return 2;
    }

    ve::renderer::MeshCacheExpectation expectation{};
    if (!ve::renderer::makeMeshCacheExpectation(options.scene, expectation)) {
        std::fprintf(stderr, "vemeshcook: cannot stat %s\n", options.scene.string().c_str());
        return 2;
    }

    const std::filesystem::path output = ve::renderer::meshCacheSidecarPath(options.scene);

    // The status check is the same one the runtime does, so "up to date" here
    // means exactly "the runtime would use this file".
    if (!options.force) {
        std::ifstream existing(output, std::ios::binary | std::ios::ate);
        if (existing) {
            const std::streamsize size = existing.tellg();
            existing.seekg(0);
            std::vector<std::byte> blob(static_cast<size_t>(std::max<std::streamsize>(size, 0)));
            existing.read(reinterpret_cast<char*>(blob.data()), size);
            if (existing && ve::renderer::meshCacheStatus(blob, expectation) == ve::renderer::MeshCacheStatus::Usable) {
                std::printf("vemeshcook: %s is up to date\n", output.string().c_str());
                return 0;
            }
        }
    }

    try {
        const auto started = std::chrono::steady_clock::now();
        ve::JobSystem jobs(options.threadCount);
        const ve::renderer::GltfGeometry geometry = ve::renderer::loadGltfGeometry(options.scene, &jobs);
        const auto builtAt = std::chrono::steady_clock::now();

        const std::vector<std::byte> blob = ve::renderer::writeMeshCache(geometry.meshes, expectation);

        if (output.has_parent_path()) {
            std::filesystem::create_directories(output.parent_path());
        }
        std::ofstream out(output, std::ios::binary);
        if (!out) {
            std::fprintf(stderr, "vemeshcook: cannot open %s for writing\n", output.string().c_str());
            return 1;
        }
        out.write(reinterpret_cast<const char*>(blob.data()), static_cast<std::streamsize>(blob.size()));
        // Closed explicitly, not left to scope exit: --verify reads this path back
        // off disk, and a still-buffered stream hands it a truncated file.
        out.close();
        if (!out) {
            std::fprintf(stderr, "vemeshcook: failed to write %s\n", output.string().c_str());
            return 1;
        }

        size_t vertexCount = 0;
        size_t indexCount = 0;
        size_t lodCount = 0;
        for (const ve::renderer::CpuMeshData& mesh : geometry.meshes) {
            vertexCount += mesh.vertices.size();
            indexCount += mesh.indices.size();
            lodCount += mesh.lods.size();
        }

        const auto buildMs =
            std::chrono::duration_cast<std::chrono::milliseconds>(builtAt - started).count();

        std::printf("vemeshcook: %s -> %s\n", options.scene.string().c_str(), output.string().c_str());
        std::printf("vemeshcook: %zu mesh(es), %zu vertices, %zu indices, %zu LOD level(s), built in %lld ms\n",
                    geometry.meshes.size(),
                    vertexCount,
                    indexCount,
                    lodCount,
                    static_cast<long long>(buildMs));
        std::printf("vemeshcook: wrote %s\n", formatMiB(blob.size()).c_str());

        if (options.verify) {
            // The equivalence check the runtime cannot make. Once geometry is
            // cooked the LOD chains are never rebuilt, so the log comparison that
            // verified the parallel LOD change is unavailable here -- there is
            // nothing left to compare against. Reading the file back off disk and
            // matching it field by field against what was just built is what
            // takes its place.
            const std::vector<std::byte> reread = readWholeFile(output);
            if (ve::renderer::meshCacheStatus(reread, expectation) != ve::renderer::MeshCacheStatus::Usable) {
                std::fprintf(stderr, "vemeshcook: verification FAILED, the file just written is not usable\n");
                return 1;
            }

            const std::vector<ve::renderer::CpuMeshData> decoded = ve::renderer::readMeshCache(reread);
            const std::string mismatch = firstMismatch(geometry.meshes, decoded);
            if (!mismatch.empty()) {
                std::fprintf(stderr, "vemeshcook: verification FAILED: %s\n", mismatch.c_str());
                return 1;
            }

            std::printf("vemeshcook: verified %zu mesh(es) round-trip exactly\n", decoded.size());
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "vemeshcook: %s\n", error.what());
        return 1;
    }

    return 0;
}
