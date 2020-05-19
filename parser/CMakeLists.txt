cmake_minimum_required(VERSION 3.10)
find_package(Protobuf REQUIRED)
MESSAGE("${Protobuf_FOUND}")
MESSAGE("${Protobuf_VERSION}")
MESSAGE("${Protobuf_LIBRARIES}")
find_library(PTHREAD_LIBRARY pthread)
find_library(DL_LIBRARY dl)
find_library(PROTOBUF protobuf)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED on)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

# Set your own path to halide
set(HALIDE_PATH "/home/anton/Downloads/halide")

include_directories(${HALIDE_PATH}/include)
#include_directories(${HALIDE_PATH}/tools)
link_directories(${HALIDE_PATH}/bin)

include_directories(include)
include_directories(generated)


#PROTOBUF
FILE(TO_NATIVE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/proto PROTOMODEL_PATH)
FILE(TO_NATIVE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/generated PROTOBINDING_PATH)

FILE(GLOB DATAMODEL_PROTOS "${CMAKE_CURRENT_SOURCE_DIR}/proto/caffe2.proto")

FOREACH(proto ${DATAMODEL_PROTOS})
FILE(TO_NATIVE_PATH ${proto} proto_native)
EXECUTE_PROCESS(COMMAND ${PROTOBUF_PROTOC_EXECUTABLE} --proto_path=${PROTOMODEL_PATH} --cpp_out=${PROTOBINDING_PATH} ${proto_native}
RESULT_VARIABLE rv)
# Optional, but that can show the user if something have gone wrong with the proto generation
IF(${rv})
MESSAGE("Generation of data model returned ${rv} for proto ${proto_native}")
ENDIF()
ENDFOREACH(proto)
## List generated sources files
FILE(GLOB HDRS "generated/*.h")
FILE(GLOB SRCS "generated/*.cc")
MESSAGE("HDRS: ${HDRS}")
MESSAGE("SRCS: ${SRCS}")
ADD_LIBRARY(protoBinding ${HDRS} ${SRCS})
# ${PROTOBUF_LIBRARIES} should be defined by FIND_PACKAGE(Protobuf REQUIRED)
TARGET_LINK_LIBRARIES(protoBinding ${PROTOBUF_LIBRARIES})

set(layers_list
    src/convolutional.cpp
    src/fully_connected.cpp
    src/concat.cpp
    src/data.cpp
    src/relu.cpp
    src/flatten.cpp
    src/maxpooling.cpp
    src/softmax.cpp
    src/dropout.cpp)

set(HALIDE_DISTRIB_DIR ${HALIDE_PATH})
include("${HALIDE_PATH}/halide.cmake")
add_executable(layer src/layer.cpp src/main.cpp src/net.cpp src/common.cpp ${layers_list})
target_link_libraries(layer ${PTHREAD_LIBRARY} ${DL_LIBRARY} ${PROTOBUF} -lHalide -lpng -ljpeg protoBinding glog)

