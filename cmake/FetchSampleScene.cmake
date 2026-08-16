# Optional download of a production-scale glTF scene.
#
# The repository ships 68 KiB of assets, which is enough to exercise the renderer
# but not enough to measure it: the asset-load baseline in docs/asset_load_baseline.md
# found texture upload below timer resolution and no glTF being loaded at all.
# This fetches a real scene so that measurement, GPU culling, and portfolio
# screenshots have something to work on.
#
# OFF by default, deliberately. With it off nothing changes: no download, no
# configure-time network access, CI stays fast, and the committed golden image
# keeps comparing against the small deterministic scene it was captured from.
#
#   cmake --preset debug -DVULKAN_ENGINE_FETCH_SAMPLE_SCENE=ON
#
# Files land in the build tree, not the source tree, so nothing has to be
# gitignored and a clean build directory removes them.
#
# Sponza, from KhronosGroup/glTF-Sample-Assets, pinned to a commit rather than a
# branch so the bytes cannot move underneath a measurement. CC BY 4.0: see the
# LICENSE.md fetched alongside it.

option(VULKAN_ENGINE_FETCH_SAMPLE_SCENE
       "Download the Sponza sample scene at configure time (off by default; adds a network fetch)" OFF)

set(VULKAN_ENGINE_SAMPLE_SCENE_COMMIT "5bad5aaa0bbb5d0f9cdc934e626f27d0df1e79b8")
set(VULKAN_ENGINE_SAMPLE_SCENE_SHA256 "646c10cbc8fab990ca29f363e90e2d65155f3a3569506852eb1434a9465b9501")

function(ve_fetch_sample_scene OUT_SCENE_PATH)
    set(${OUT_SCENE_PATH} "" PARENT_SCOPE)
    if(NOT VULKAN_ENGINE_FETCH_SAMPLE_SCENE)
        return()
    endif()

    set(base_url
        "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/${VULKAN_ENGINE_SAMPLE_SCENE_COMMIT}/Models/Sponza/glTF")
    set(destination "${CMAKE_BINARY_DIR}/fetched-assets/sponza")
    set(scene_file "${destination}/Sponza.gltf")

    file(MAKE_DIRECTORY "${destination}")

    # The entry point is hash-checked. Everything else is addressed by the same
    # pinned commit, so a mismatch here is the signal that the pin moved or the
    # download truncated.
    if(NOT EXISTS "${scene_file}")
        message(STATUS "Fetching sample scene: Sponza.gltf")
        file(DOWNLOAD "${base_url}/Sponza.gltf" "${scene_file}"
             EXPECTED_HASH SHA256=${VULKAN_ENGINE_SAMPLE_SCENE_SHA256}
             TLS_VERIFY ON
             STATUS download_status)
        list(GET download_status 0 download_code)
        if(NOT download_code EQUAL 0)
            list(GET download_status 1 download_message)
            file(REMOVE "${scene_file}")
            message(WARNING "Sample scene download failed (${download_message}); building without it.")
            return()
        endif()
    endif()

    # The glTF lists its own dependencies, so the buffer and image URIs are read
    # out of it rather than duplicated here. A hand-maintained file list would
    # drift the first time the upstream model changed a texture.
    file(READ "${scene_file}" scene_json)
    set(referenced_uris "")

    foreach(collection "buffers" "images")
        string(JSON collection_length ERROR_VARIABLE json_error LENGTH "${scene_json}" ${collection})
        if(json_error OR NOT collection_length)
            continue()
        endif()
        math(EXPR last_index "${collection_length} - 1")
        foreach(index RANGE ${last_index})
            string(JSON uri ERROR_VARIABLE uri_error GET "${scene_json}" ${collection} ${index} "uri")
            if(NOT uri_error AND uri)
                list(APPEND referenced_uris "${uri}")
            endif()
        endforeach()
    endforeach()

    list(REMOVE_DUPLICATES referenced_uris)
    list(LENGTH referenced_uris uri_count)
    message(STATUS "Sample scene references ${uri_count} buffer/image file(s)")

    set(missing_count 0)
    foreach(uri IN LISTS referenced_uris)
        set(target_file "${destination}/${uri}")
        if(EXISTS "${target_file}")
            continue()
        endif()
        file(DOWNLOAD "${base_url}/${uri}" "${target_file}" TLS_VERIFY ON STATUS file_status)
        list(GET file_status 0 file_code)
        if(NOT file_code EQUAL 0)
            file(REMOVE "${target_file}")
            math(EXPR missing_count "${missing_count} + 1")
            message(WARNING "Sample scene file failed to download: ${uri}")
        endif()
    endforeach()

    if(missing_count GREATER 0)
        # A partial scene renders with missing textures and would quietly poison
        # any measurement taken against it, so it is refused rather than shipped.
        message(WARNING "Sample scene is incomplete (${missing_count} file(s) missing); building without it.")
        return()
    endif()

    if(NOT EXISTS "${destination}/LICENSE.md")
        file(DOWNLOAD "${base_url}/../LICENSE.md" "${destination}/LICENSE.md" TLS_VERIFY ON)
    endif()

    message(STATUS "Sample scene ready: ${scene_file}")
    set(${OUT_SCENE_PATH} "${scene_file}" PARENT_SCOPE)
endfunction()
