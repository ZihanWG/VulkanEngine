# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "E:/VulkanEngine/build/_deps/vma-src")
  file(MAKE_DIRECTORY "E:/VulkanEngine/build/_deps/vma-src")
endif()
file(MAKE_DIRECTORY
  "E:/VulkanEngine/build/_deps/vma-build"
  "E:/VulkanEngine/build/_deps/vma-subbuild/vma-populate-prefix"
  "E:/VulkanEngine/build/_deps/vma-subbuild/vma-populate-prefix/tmp"
  "E:/VulkanEngine/build/_deps/vma-subbuild/vma-populate-prefix/src/vma-populate-stamp"
  "E:/VulkanEngine/build/_deps/vma-subbuild/vma-populate-prefix/src"
  "E:/VulkanEngine/build/_deps/vma-subbuild/vma-populate-prefix/src/vma-populate-stamp"
)

set(configSubDirs Debug)
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "E:/VulkanEngine/build/_deps/vma-subbuild/vma-populate-prefix/src/vma-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "E:/VulkanEngine/build/_deps/vma-subbuild/vma-populate-prefix/src/vma-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
