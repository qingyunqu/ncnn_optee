
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Installation Directory")
endif()
message(STATUS "CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")

# This is the minimum cmake version supporting the modern cmake OpenMP module below
# modern cmake requires 3.0 anyway and android cmake requires 3.6
cmake_minimum_required(VERSION 3.1.3)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE release CACHE STRING "Choose the type of build" FORCE)
endif()

set(COVERAGE OFF CACHE BOOL "Coverage")

project(benchncnn CXX)

option(NCNN_OPENMP "openmp support" OFF)
option(NCNN_STDIO "load model from external file" ON)
option(NCNN_STRING "plain and verbose string" ON)
option(NCNN_OPENCV "minimal opencv structure emulation" OFF)
option(NCNN_BENCHMARK "print benchmark information for every layer" OFF)
option(NCNN_PIXEL "convert and resize from/to image pixel" ON)
option(NCNN_PIXEL_ROTATE "rotate image pixel orientation" OFF)
# teec option
option(NCNN_TEEC "run ncnn forward on optee client and optee os" ON)


if(NCNN_OPENMP)
    find_package(OpenMP)
    # For CMake < 3.9, we need to make the target ourselves
    if(NOT TARGET OpenMP::OpenMP_CXX)
        find_package(Threads REQUIRED)
        add_library(OpenMP::OpenMP_CXX IMPORTED INTERFACE)
        set_property(TARGET OpenMP::OpenMP_CXX
          PROPERTY INTERFACE_COMPILE_OPTIONS ${OpenMP_CXX_FLAGS})
        # Only works if the same flag is passed to the linker; use CMake 3.9+ otherwise (Intel, AppleClang)
        set_property(TARGET OpenMP::OpenMP_CXX
          PROPERTY INTERFACE_LINK_LIBRARIES ${OpenMP_CXX_FLAGS} Threads::Threads)
    endif()
endif()

#    add_definitions(-Wall -Wextra -Wno-unused-function)
#    add_definitions(-fPIC)
#    if(PI3)
#        add_definitions(-march=native)
#        add_definitions(-mfpu=neon)
#        add_definitions(-mfloat-abi=hard)
#        add_definitions(-D__ARM_NEON)
#        add_definitions(-D__ANDROID__)
#    endif()
#    # add_definitions(-march=native)
#    # add_definitions(-flto)
#    add_definitions(-fvisibility=hidden -fvisibility-inlines-hidden)

#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti -fno-exceptions")


set(CMAKE_CXX_FLAGS "-std=c++11 -march=armv7-a ${CMAKE_CXX_FLAGS}")
#add_definitions(-mfloat-abi=hard)
#add_definitions(-D__ARM_NEON)
#-mfloat-abi=hard -mfpu=neon-vfpv4 #-fopenmp
#add_definitions(-D__ANDROID__)

add_subdirectory(src)

set(SRC benchncnn.cpp)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/src)
add_executable (${PROJECT_NAME} ${SRC})
target_link_libraries(${PROJECT_NAME} ncnn)
install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_BINDIR})
install(FILES 
	param/alexnet.param
	param/googlenet.param
	param/mnasnet.param
	param/mobilenet.param
	param/mobilenet_ssd.param
	param/mobilenet_v2.param
	param/mobilenet_yolo.param
	param/mobilenet_yolov3.param
	param/proxylessnasnet.param
	param/resnet18.param
	param/shufflenet.param
	param/squeezenet.param
	param/squeezenet_ssd.param
	param/vgg16.param
	DESTINATION ${CMAKE_INSTALL_BINDIR})
