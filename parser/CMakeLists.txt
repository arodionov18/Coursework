cmake_minimum_required(VERSION 3.10)

find_library(PTHRED_LIBRARY pthread)
find_library(DL_LIBRARY dl)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED on)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

include_directories(../include)
link_directories(../bin)

find_library(HALIDE_LIBRARY Halide)

add_executable(layer layer.cpp)
target_link_libraries(layer ${PTHREAD_LIBRARY} ${DL_LIBRARY} -lHalide)

