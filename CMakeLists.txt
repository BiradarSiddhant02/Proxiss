cmake_minimum_required(VERSION 3.16)

project(proxi)

set(CMAKE_CXX_STANDARD 20)

if(MSVC)
    set(CMAKE_CXX_FLAGS "/O2 /arch:AVX /DNDEBUG")
else()
    set(CMAKE_CXX_FLAGS "-O3 -march=native -mavx -DNDEBUG -fopenmp")
endif()

include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.11.1
)
FetchContent_MakeAvailable(pybind11)

include_directories(include)

pybind11_add_module(proxi MODULE
    src/proxi_flat.cc
    bindings/proxi_binding.cc
)

target_include_directories(proxi PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_compile_options(proxi PRIVATE -O3 -march=native)

set_target_properties(proxi PROPERTIES PREFIX "")

install(TARGETS proxi LIBRARY DESTINATION .)