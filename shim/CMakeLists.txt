cmake_minimum_required(VERSION 3.21)
project(amarbaro_shim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

list(APPEND CMAKE_PREFIX_PATH /opt/rocm)
find_package(hip REQUIRED)
find_package(hipblaslt REQUIRED)

add_library(amarbaro_shim SHARED src/amarbaro_shim.cpp)
target_include_directories(amarbaro_shim PUBLIC include)
target_link_libraries(amarbaro_shim PRIVATE hip::host roc::hipblaslt)
