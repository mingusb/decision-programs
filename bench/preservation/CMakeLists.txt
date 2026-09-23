cmake_minimum_required(VERSION 3.30)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86-real CACHE STRING "CUDA architectures to build")
endif()
if(NOT DEFINED CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")
endif()
project(histogram_preservation LANGUAGES CXX CUDA)
find_package(CUDAToolkit 13.4 REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA" AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.4)
  set(CMAKE_CUDA23_STANDARD_COMPILE_OPTION "-std=c++23")
  set(CMAKE_CUDA23_EXTENSION_COMPILE_OPTION "-std=c++23")
endif()
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 23)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

get_filename_component(GH_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(GH_ARCHIVED_SOURCES "${CMAKE_CURRENT_BINARY_DIR}/sources")
execute_process(
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/prepare_sources.py"
          --repo-root "${GH_REPO_ROOT}" --output-dir "${GH_ARCHIVED_SOURCES}"
  COMMAND_ERROR_IS_FATAL ANY
)
# Reconfigure and revalidate if the archive or its manifest changes.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/prepare_sources.py"
  "${GH_REPO_ROOT}/results/a5000-custom-only/source.tar.gz"
  "${GH_REPO_ROOT}/results/a5000-custom-only/source-manifest.json"
  "${GH_REPO_ROOT}/results/a5000-large-bins/final-source.tar.gz"
  "${GH_REPO_ROOT}/results/a5000-large-bins/final-source-manifest.json"
)

foreach(backend IN ITEMS old new)
  set(archive "${GH_ARCHIVED_SOURCES}/${backend}")
  set(sources
    "${archive}/src/config.cpp"
    "${archive}/src/defaults.cpp"
    "${archive}/src/histogram.cu"
    "${archive}/src/bitplane.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/backend.cpp"
  )
  if(backend STREQUAL "new")
    list(APPEND sources "${archive}/src/global_narrow.cu" "${archive}/src/shared_overflow.cu")
  endif()
  add_library(paired_${backend} STATIC ${sources})
  target_include_directories(paired_${backend} PRIVATE "${archive}/include")
  target_include_directories(paired_${backend} SYSTEM PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
  # Equal-length namespaces prevent cross-version symbol interposition. No
  # definition is exported to the harness, which sees only backend.hpp.
  target_compile_definitions(paired_${backend} PRIVATE
    gh=gh_${backend}
    GH_BACKEND_ENTRY=paired_${backend}_api
    GH_BACKEND_NAME="${backend}"
  )
  target_compile_options(paired_${backend} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo;--ptxas-options=-v>
  )
endforeach()

add_executable(histogram_preservation main.cpp)
target_link_libraries(histogram_preservation PRIVATE
  paired_old paired_new CUDA::cudart CUDA::cuda_driver ${CMAKE_DL_LIBS}
)
# Both complete custom libraries share this process's single dynamic runtime.
# No reference histogram source or target participates in this diagnostic.
