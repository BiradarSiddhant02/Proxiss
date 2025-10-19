cmake_minimum_required(VERSION 3.16)

# ==============================================================================
# Auto-dependency Management CMake Configuration
# ==============================================================================
# This CMakeLists.txt automatically handles:
# 1. Forces clang++ compiler usage (downloads if not found)
# 2. Auto-installs OpenMP if not available
# 3. Auto-downloads pybind11 via FetchContent
# 4. Supports multiple package managers: apt, yum, dnf, pacman, brew
# ==============================================================================

# Suppress developer warnings
set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS ON CACHE BOOL "Suppress developer warnings")

# Force clang++ compiler
if(NOT CMAKE_CXX_COMPILER)
    # Try to find clang++ first
    find_program(CLANG_CXX_EXECUTABLE clang++)
    
    if(NOT CLANG_CXX_EXECUTABLE)
        message(STATUS "clang++ not found, attempting to install...")
        
        # Try different package managers to install clang
        find_program(APT_EXECUTABLE apt-get)
        find_program(YUM_EXECUTABLE yum)
        find_program(DNF_EXECUTABLE dnf)
        find_program(PACMAN_EXECUTABLE pacman)
        find_program(BREW_EXECUTABLE brew)
        
        if(APT_EXECUTABLE)
            message(STATUS "Installing clang via apt-get...")
            execute_process(COMMAND sudo apt-get update)
            execute_process(COMMAND sudo apt-get install -y clang)
        elseif(YUM_EXECUTABLE)
            message(STATUS "Installing clang via yum...")
            execute_process(COMMAND sudo yum install -y clang)
        elseif(DNF_EXECUTABLE)
            message(STATUS "Installing clang via dnf...")
            execute_process(COMMAND sudo dnf install -y clang)
        elseif(PACMAN_EXECUTABLE)
            message(STATUS "Installing clang via pacman...")
            execute_process(COMMAND sudo pacman -S --noconfirm clang)
        elseif(BREW_EXECUTABLE)
            message(STATUS "Installing clang via brew...")
            execute_process(COMMAND brew install llvm)
        else()
            message(WARNING "No supported package manager found. Please install clang++ manually.")
        endif()
        
        # Try to find clang++ again after installation
        find_program(CLANG_CXX_EXECUTABLE clang++)
    endif()
    
    if(CLANG_CXX_EXECUTABLE)
        set(CMAKE_CXX_COMPILER ${CLANG_CXX_EXECUTABLE})
        message(STATUS "Using clang++ compiler: ${CMAKE_CXX_COMPILER}")
    else()
        message(FATAL_ERROR "clang++ compiler not found and could not be installed automatically")
    endif()
endif()

project(proxi_flat_cpp)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Verify compiler is clang++
if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(WARNING "Compiler is not clang++: ${CMAKE_CXX_COMPILER_ID}. This build is optimized for clang++.")
endif()

# Ensure all targets use PIC for shared libraries
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Add compiler-specific optimizations for clang++
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(STATUS "Using clang++ optimizations")
endif()

# Find OpenMP package
find_package(OpenMP)

if(NOT OpenMP_CXX_FOUND)
    message(STATUS "OpenMP not found, attempting to install...")
    
    # Try different package managers to install OpenMP
    find_program(APT_EXECUTABLE apt-get)
    find_program(YUM_EXECUTABLE yum)
    find_program(DNF_EXECUTABLE dnf)
    find_program(PACMAN_EXECUTABLE pacman)
    find_program(BREW_EXECUTABLE brew)
    
    if(APT_EXECUTABLE)
        message(STATUS "Installing OpenMP via apt-get...")
        execute_process(COMMAND sudo apt-get install -y libomp-dev)
    elseif(YUM_EXECUTABLE)
        message(STATUS "Installing OpenMP via yum...")
        execute_process(COMMAND sudo yum install -y libomp-devel)
    elseif(DNF_EXECUTABLE)
        message(STATUS "Installing OpenMP via dnf...")
        execute_process(COMMAND sudo dnf install -y libomp-devel)
    elseif(PACMAN_EXECUTABLE)
        message(STATUS "Installing OpenMP via pacman...")
        execute_process(COMMAND sudo pacman -S --noconfirm openmp)
    elseif(BREW_EXECUTABLE)
        message(STATUS "Installing OpenMP via brew...")
        execute_process(COMMAND brew install libomp)
    else()
        message(WARNING "No supported package manager found for OpenMP installation")
    endif()
    
    # Try to find OpenMP again after installation
    find_package(OpenMP)
    
    if(NOT OpenMP_CXX_FOUND)
        message(WARNING "OpenMP still not found after installation attempt. Building without OpenMP support.")
    endif()
endif()

if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found: ${OpenMP_CXX_LIBRARIES}")
else()
    message(STATUS "Building without OpenMP support")
endif()

# Find Eigen3 package (required for PCA)
find_package(Eigen3 QUIET)

if(NOT Eigen3_FOUND)
    message(STATUS "Eigen3 not found, attempting to install...")
    
    # Try different package managers to install Eigen3
    find_program(APT_EXECUTABLE apt-get)
    find_program(YUM_EXECUTABLE yum)
    find_program(DNF_EXECUTABLE dnf)
    find_program(PACMAN_EXECUTABLE pacman)
    find_program(BREW_EXECUTABLE brew)
    
    if(APT_EXECUTABLE)
        message(STATUS "Installing Eigen3 via apt-get...")
        execute_process(COMMAND sudo apt-get update)
        execute_process(COMMAND sudo apt-get install -y libeigen3-dev)
    elseif(YUM_EXECUTABLE)
        message(STATUS "Installing Eigen3 via yum...")
        execute_process(COMMAND sudo yum install -y eigen3-devel)
    elseif(DNF_EXECUTABLE)
        message(STATUS "Installing Eigen3 via dnf...")
        execute_process(COMMAND sudo dnf install -y eigen3-devel)
    elseif(PACMAN_EXECUTABLE)
        message(STATUS "Installing Eigen3 via pacman...")
        execute_process(COMMAND sudo pacman -S --noconfirm eigen)
    elseif(BREW_EXECUTABLE)
        message(STATUS "Installing Eigen3 via brew...")
        execute_process(COMMAND brew install eigen)
    else()
        message(WARNING "No supported package manager found for Eigen3 installation")
    endif()
    
    # Try to find Eigen3 again after installation
    find_package(Eigen3 QUIET)
    
    if(NOT Eigen3_FOUND)
        message(FATAL_ERROR "Eigen3 is required for PCA functionality but could not be found or installed")
    endif()
endif()

if(Eigen3_FOUND)
    message(STATUS "Eigen3 found: ${EIGEN3_INCLUDE_DIR}")
    include_directories(${EIGEN3_INCLUDE_DIR})
else()
    message(FATAL_ERROR "Eigen3 is required but not found")
endif()

# Try to find pybind11 first, only fetch if not found
find_package(pybind11 QUIET)

if(NOT pybind11_FOUND)
    message(STATUS "pybind11 not found locally, fetching from GitHub...")
    include(FetchContent)
    FetchContent_Declare(
      pybind11
      GIT_REPOSITORY https://github.com/pybind/pybind11.git
      GIT_TAG v2.11.1
    )
    FetchContent_MakeAvailable(pybind11)
    message(STATUS "pybind11 fetched and ready")
else()
    message(STATUS "Using existing pybind11 installation")
endif()

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

# Define the proxi_pca module (includes PCA and ProxiPCA sources, like KNN includes its source)
pybind11_add_module(proxi_pca_cpp MODULE
    src/pca.cc
    src/proxi_pca.cc
    bindings/proxi_pca_binding.cc
)
target_compile_options(proxi_pca_cpp PRIVATE 
    -O3 -march=native -DNDEBUG -ffast-math -ftree-vectorize -funroll-loops
)
target_link_libraries(proxi_pca_cpp PRIVATE proxi_common)
if(OpenMP_CXX_FOUND)
    target_link_libraries(proxi_pca_cpp PRIVATE OpenMP::OpenMP_CXX)
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
set_target_properties(proxi_pca_cpp PROPERTIES PREFIX "" SUFFIX ".so")

# Debug output for compile options
get_target_property(COMPILE_OPTIONS proxi_flat_cpp COMPILE_OPTIONS)
message(STATUS "Final compile options: ${COMPILE_OPTIONS}")

# Installation rules
install(TARGETS proxi_flat_cpp LIBRARY DESTINATION .)
install(TARGETS proxi_knn_cpp LIBRARY DESTINATION .)
install(TARGETS proxi_pca_cpp LIBRARY DESTINATION .)
