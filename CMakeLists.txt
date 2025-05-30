cmake_minimum_required(VERSION 3.16)

project(proxi_flat_cpp)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

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

# Define the module
pybind11_add_module(proxi_flat_cpp MODULE
    src/proxi_flat.cc
    bindings/proxi_flat_binding.cc
)

# Set include directories for the target
# (already included above)

# GCC/Clang optimizations (Linux only)
include(CheckCXXCompilerFlag)
target_compile_options(proxi_flat_cpp PRIVATE 
    -O3 -DNDEBUG -ffast-math
)

# Check for march=native support
check_cxx_compiler_flag("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if(COMPILER_SUPPORTS_MARCH_NATIVE)
    target_compile_options(proxi_flat_cpp PRIVATE -march=native)
    message(STATUS "Using -march=native for optimal CPU optimization")
else()
    # Fallback to basic AVX
    check_cxx_compiler_flag("-mavx" COMPILER_SUPPORTS_AVX)
    if(COMPILER_SUPPORTS_AVX)
        target_compile_options(proxi_flat_cpp PRIVATE -mavx)
    endif()
endif()

# Apply OpenMP to the target if found
if(OpenMP_CXX_FOUND)
    target_link_libraries(proxi_flat_cpp PRIVATE OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP found and enabled")
else()
    message(WARNING "OpenMP not found. Building without OpenMP support.")
endif()

# Set properties and handle platform-specific naming conventions
set_target_properties(proxi_flat_cpp PROPERTIES PREFIX "")
set_target_properties(proxi_flat_cpp PROPERTIES SUFFIX ".so")

# Debug output to verify compilation flags
get_target_property(COMPILE_OPTIONS proxi_flat_cpp COMPILE_OPTIONS)
message(STATUS "Final compile options: ${COMPILE_OPTIONS}")

# Installation rules
install(TARGETS proxi_flat_cpp LIBRARY DESTINATION .)