cmake_minimum_required(VERSION 3.15)

# Set default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

project(proxi)

set(CMAKE_CXX_STANDARD 20)

# --- Platform-specific optimization flags ---
if(MSVC)
    # For Visual Studio (MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /arch:AVX /DNDEBUG")
else()
    # For GCC/Clang
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ffast-math -flto -mavx -DNDEBUG")
endif()

# --- pybind11 integration ---
find_package(pybind11 QUIET)
if(NOT pybind11_FOUND)
    message(STATUS "Downloading pybind11...")
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG master
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# --- Include directories ---
include_directories(include)

# --- pybind11 module ---
pybind11_add_module(proxi 
    bindings/proxi_binding.cc
    src/proxi.cc
)

# --- OpenMP support ---
find_package(OpenMP)

if(OpenMP_FOUND)
    target_link_libraries(proxi PRIVATE OpenMP::OpenMP_CXX)
    
    if(MSVC)
        target_compile_options(proxi PRIVATE /openmp)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        target_compile_options(proxi PRIVATE -fopenmp)
    endif()
endif()
