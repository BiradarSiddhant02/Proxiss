cmake_minimum_required(VERSION 3.16)

project(proxi_cpp)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find OpenMP package
find_package(OpenMP)

# Platform-specific compiler flags
if(MSVC)
    set(CMAKE_CXX_FLAGS "/O2 /DNDEBUG")
    
    # Check for AVX support in MSVC
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("/arch:AVX" COMPILER_SUPPORTS_AVX)
    if(COMPILER_SUPPORTS_AVX)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX")
    endif()
else()
    set(CMAKE_CXX_FLAGS "-O3 -DNDEBUG")
    
    # Check for AVX support in GCC/Clang
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-mavx" COMPILER_SUPPORTS_AVX)
    if(COMPILER_SUPPORTS_AVX)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx")
    endif()
endif()

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
pybind11_add_module(proxi_cpp MODULE
    src/proxi_flat.cc
    bindings/proxi_binding.cc
)

# Set include directories for the target
target_include_directories(proxi_cpp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Apply compiler options to the target
if(MSVC)
    target_compile_options(proxi_cpp PRIVATE /O2)
    if(COMPILER_SUPPORTS_AVX)
        target_compile_options(proxi_cpp PRIVATE /arch:AVX)
    endif()
else()
    target_compile_options(proxi_cpp PRIVATE -O3 -march=native -mavx)
endif()

# Apply OpenMP to the target appropriately for each platform
if(OpenMP_CXX_FOUND)
    target_link_libraries(proxi_cpp PRIVATE OpenMP::OpenMP_CXX)
else()
    # Special case for macOS which may not have OpenMP by default
    if(APPLE)
        message(STATUS "MacOS detected, checking for Homebrew OpenMP...")
        find_program(HOMEBREW_EXECUTABLE brew)
        if(HOMEBREW_EXECUTABLE)
            execute_process(
                COMMAND ${HOMEBREW_EXECUTABLE} --prefix libomp
                RESULT_VARIABLE HOMEBREW_LIBOMP
                OUTPUT_VARIABLE HOMEBREW_LIBOMP_PREFIX
                OUTPUT_STRIP_TRAILING_WHITESPACE
            )
            if(HOMEBREW_LIBOMP EQUAL 0 AND EXISTS "${HOMEBREW_LIBOMP_PREFIX}")
                set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
                set(OpenMP_CXX_LIB_NAMES "omp")
                set(OpenMP_omp_LIBRARY ${HOMEBREW_LIBOMP_PREFIX}/lib/libomp.dylib)
                target_compile_options(proxi_cpp PRIVATE ${OpenMP_CXX_FLAGS})
                target_link_libraries(proxi_cpp PRIVATE ${OpenMP_omp_LIBRARY})
                target_include_directories(proxi_cpp PRIVATE ${HOMEBREW_LIBOMP_PREFIX}/include)
                message(STATUS "Using Homebrew OpenMP")
            else()
                message(WARNING "OpenMP not found. Building without OpenMP support. To enable OpenMP on macOS, install libomp with Homebrew.")
            endif()
        else()
            message(WARNING "Homebrew not found. Building without OpenMP support. To enable OpenMP on macOS, install Homebrew and libomp.")
        endif()
    else()
        message(WARNING "OpenMP not found. Building without OpenMP support.")
    endif()
endif()

# Set properties and handle platform-specific naming conventions
set_target_properties(proxi_cpp PROPERTIES PREFIX "")

if(WIN32)
    set_target_properties(proxi_cpp PROPERTIES SUFFIX ".pyd")
elseif(APPLE)
    set_target_properties(proxi_cpp PROPERTIES SUFFIX ".so")
endif()

# Installation rules
install(TARGETS proxi_cpp LIBRARY DESTINATION .)