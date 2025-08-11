cmake_minimum_required(VERSION 3.16)

# Suppress developer warnings
set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS ON CACHE BOOL "Suppress developer warnings")

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

# Define the proxi_flat module
pybind11_add_module(proxi_flat_cpp MODULE
    src/proxi_flat.cc
    bindings/proxi_flat_binding.cc
)

# Define the proxi_knn module
pybind11_add_module(proxi_knn_cpp MODULE
    src/proxi_knn.cc
    bindings/proxi_knn_binding.cc
)

# Define the proxi_cluster module
pybind11_add_module(proxi_cluster_cpp MODULE
    src/proxi_cluster.cc
    bindings/proxi_cluster_binding.cc
)

# Set include directories for the target
# (already included above)

# GCC/Clang optimizations (Linux only)
include(CheckCXXCompilerFlag)
target_compile_options(proxi_flat_cpp PRIVATE 
    -O3 -march=native -DNDEBUG -ffast-math -ftree-vectorize -funroll-loops -fslp-vectorize
)
target_compile_options(proxi_knn_cpp PRIVATE 
    -O3 -march=native -DNDEBUG -ffast-math -ftree-vectorize -funroll-loops -fslp-vectorize
)
target_compile_options(proxi_cluster_cpp PRIVATE 
    -O3 -march=native -DNDEBUG -ffast-math -ftree-vectorize -funroll-loops -fslp-vectorize
)

# Apply OpenMP to the targets if found
if(OpenMP_CXX_FOUND)
    target_link_libraries(proxi_flat_cpp PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(proxi_knn_cpp PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(proxi_cluster_cpp PRIVATE OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP found and enabled")
else()
    message(WARNING "OpenMP not found. Building without OpenMP support.")
endif()

# Set properties and handle platform-specific naming conventions
set_target_properties(proxi_flat_cpp PROPERTIES PREFIX "")
set_target_properties(proxi_flat_cpp PROPERTIES SUFFIX ".so")
set_target_properties(proxi_knn_cpp PROPERTIES PREFIX "")
set_target_properties(proxi_knn_cpp PROPERTIES SUFFIX ".so")
set_target_properties(proxi_cluster_cpp PROPERTIES PREFIX "")
set_target_properties(proxi_cluster_cpp PROPERTIES SUFFIX ".so")

# Debug output to verify compilation flags
get_target_property(COMPILE_OPTIONS proxi_flat_cpp COMPILE_OPTIONS)
message(STATUS "Final compile options: ${COMPILE_OPTIONS}")

# Installation rules
install(TARGETS proxi_flat_cpp LIBRARY DESTINATION .)
install(TARGETS proxi_knn_cpp LIBRARY DESTINATION .)
install(TARGETS proxi_cluster_cpp LIBRARY DESTINATION .)