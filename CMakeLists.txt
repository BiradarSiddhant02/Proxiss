cmake_minimum_required(VERSION 3.15)
project(proxi)

set(CMAKE_CXX_STANDARD 20)

# Set compiler flags for Release/Debug
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ffast-math -flto -mavx -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG   "-O0 -g")

# Download pybind11 from GitHub if not found
find_package(pybind11 REQUIRED)

if(NOT TARGET pybind11::module)
    message(STATUS "Downloading pybind11 from GitHub...")
    include(ExternalProject)
    ExternalProject_Add(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG master
        PREFIX ${CMAKE_BINARY_DIR}/external
        CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/external/pybind11
    )
    add_dependencies(proxi pybind11)
endif()

# Find OpenMP
find_package(OpenMP REQUIRED)

# Include headers
include_directories(include)

# Build the pybind11 module
pybind11_add_module(proxi 
    bindings/proxi_binding.cc
    src/proxi.cc
)

# Link against OpenMP runtime
target_link_libraries(proxi PUBLIC OpenMP::OpenMP_CXX)
