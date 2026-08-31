cmake_minimum_required(VERSION 3.21)
project(baro_shim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

list(APPEND CMAKE_PREFIX_PATH /opt/rocm)
find_package(hip REQUIRED)
find_package(hipblaslt REQUIRED)

add_library(baro_shim SHARED src/baro_shim.cpp)
target_include_directories(baro_shim PUBLIC include)
target_link_libraries(baro_shim PRIVATE hip::host roc::hipblaslt)
