cmake_minimum_required(VERSION 3.16)

project(proxi_cpp)

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
pybind11_add_module(proxi_cpp MODULE
    src/proxi_flat.cc
    bindings/proxi_binding.cc
)

# Set include directories for the target
target_include_directories(proxi_cpp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Apply compiler options to the target - consolidated approach
if(MSVC)
    # Check for AVX2 support first
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("/arch:AVX2" COMPILER_SUPPORTS_AVX2)
    check_cxx_compiler_flag("/arch:AVX" COMPILER_SUPPORTS_AVX)
    
    # Base optimization flags
    target_compile_options(proxi_cpp PRIVATE 
        /O2 /Ob2 /Oi /Ot /Oy /GL /fp:fast /DNDEBUG
        /favor:INTEL64 /GS- /Gy /GT /MP
    )
    
    # Add AVX support
    if(COMPILER_SUPPORTS_AVX2)
        target_compile_options(proxi_cpp PRIVATE /arch:AVX2)
        message(STATUS "Using AVX2 instructions on Windows")
    elseif(COMPILER_SUPPORTS_AVX)
        target_compile_options(proxi_cpp PRIVATE /arch:AVX)
        message(STATUS "Using AVX instructions on Windows")
    endif()
    
    # Link-time optimization
    set_target_properties(proxi_cpp PROPERTIES 
        LINK_FLAGS "/LTCG /OPT:REF /OPT:ICF"
    )
else()
    # GCC/Clang optimizations
    include(CheckCXXCompilerFlag)
    
    target_compile_options(proxi_cpp PRIVATE 
        -O3 -DNDEBUG -ffast-math
    )
    
    # Check for march=native support
    check_cxx_compiler_flag("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
    if(COMPILER_SUPPORTS_MARCH_NATIVE)
        target_compile_options(proxi_cpp PRIVATE -march=native)
        message(STATUS "Using -march=native for optimal CPU optimization")
    else()
        # Fallback to basic AVX
        check_cxx_compiler_flag("-mavx" COMPILER_SUPPORTS_AVX)
        if(COMPILER_SUPPORTS_AVX)
            target_compile_options(proxi_cpp PRIVATE -mavx)
        endif()
    endif()
endif()

# Apply OpenMP to the target appropriately for each platform
if(OpenMP_CXX_FOUND)
    target_link_libraries(proxi_cpp PRIVATE OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP found and enabled")
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

# Debug output to verify compilation flags
get_target_property(COMPILE_OPTIONS proxi_cpp COMPILE_OPTIONS)
message(STATUS "Final compile options: ${COMPILE_OPTIONS}")

# Installation rules
install(TARGETS proxi_cpp LIBRARY DESTINATION .)