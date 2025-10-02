cmake_minimum_required(VERSION 3.16)

# Suppress developer warnings
set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS ON CACHE BOOL "Suppress developer warnings")

project(proxi_flat_cpp)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Ensure all targets use PIC for shared libraries
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find OpenMP package
find_package(OpenMP)

# Fetch pybind11
include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.11.1
)
FetchContent_MakeAvailable(pybind11)

# Include directories
include_directories(include)

# Create a static library for shared sources
add_library(proxi_common STATIC
    src/priority_queue.cc
    src/proxi_flat.cc
)
target_compile_options(proxi_common PRIVATE 
    -O3 -march=native -DNDEBUG -ffast-math -ftree-vectorize -funroll-loops
)
if(OpenMP_CXX_FOUND)
    target_link_libraries(proxi_common PRIVATE OpenMP::OpenMP_CXX)
endif()

# Define the proxi_flat module
pybind11_add_module(proxi_flat_cpp MODULE
    bindings/proxi_flat_binding.cc
)
target_link_libraries(proxi_flat_cpp PRIVATE proxi_common)
if(OpenMP_CXX_FOUND)
    target_link_libraries(proxi_flat_cpp PRIVATE OpenMP::OpenMP_CXX)
endif()

# Define the proxi_knn module
pybind11_add_module(proxi_knn_cpp MODULE
    src/proxi_knn.cc
    bindings/proxi_knn_binding.cc
)
target_link_libraries(proxi_knn_cpp PRIVATE proxi_common)
if(OpenMP_CXX_FOUND)
    target_link_libraries(proxi_knn_cpp PRIVATE OpenMP::OpenMP_CXX)
endif()

# Use Wild as the linker via -fuse-ld
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=wild")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fuse-ld=wild")
    message(STATUS "Using Wild as the linker with -fuse-ld=wild")
endif()

# Set target properties for .so naming
set_target_properties(proxi_flat_cpp PROPERTIES PREFIX "" SUFFIX ".so")
set_target_properties(proxi_knn_cpp PROPERTIES PREFIX "" SUFFIX ".so")

# Debug output for compile options
get_target_property(COMPILE_OPTIONS proxi_flat_cpp COMPILE_OPTIONS)
message(STATUS "Final compile options: ${COMPILE_OPTIONS}")

# Installation rules
install(TARGETS proxi_flat_cpp LIBRARY DESTINATION .)
install(TARGETS proxi_knn_cpp LIBRARY DESTINATION .)
