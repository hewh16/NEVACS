cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 14)

project(yolov5_pro)

add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-11.4)
set(CMAKE_BUILD_TYPE Debug)
set(THREADS_PREFER_PTHREAD_FLAG ON)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
set(THREADS_HAVE_PTHREAD_ARG 1)

find_package(CUDA 11.3 REQUIRED)
find_package(MetavisionSDK COMPONENTS core driver ui REQUIRED)
find_package(Boost COMPONENTS program_options REQUIRED)
find_package(OpenCV COMPONENTS core highgui REQUIRED)
find_package(Threads REQUIRED)
find_package(PkgConfig)
pkg_check_modules(SERIAL libserial)

if(WIN32)
enable_language(CUDA)
endif(WIN32)

include_directories(${PROJECT_SOURCE_DIR}/include)

include_directories(/usr/include/eigen3)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
#include_directories(/usr/include/x86_64-linux-gnu/)
#link_directories(/usr/lib/x86_64-linux-gnu/)
include_directories(/home/weihua/TensorRT-8.0.3.4/include/)
link_directories(/home/weihua/TensorRT-8.0.3.4/targets/x86_64-linux-gnu/lib/)

if(CMAKE_VERSION VERSION_LESS "2.8.11")
    # Add OpenCV headers location to your include paths
    include_directories(${OpenCV_INCLUDE_DIRS})
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall -Ofast -g -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")
cuda_add_library(myplugins SHARED yololayer.cu)
target_link_libraries(myplugins nvinfer cudart)

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

cuda_add_executable(yolov5_pro calibrator.cpp yolov5_pro.cpp preprocess.cu)

target_link_libraries(yolov5_pro nvinfer)
target_link_libraries(yolov5_pro cudart)
target_link_libraries(yolov5_pro pthread)
target_link_libraries(yolov5_pro myplugins)
target_link_libraries(yolov5_pro ${OpenCV_LIBS})
target_link_libraries(yolov5_pro MetavisionSDK::core MetavisionSDK::driver MetavisionSDK::ui Boost::program_options)
target_include_directories(yolov5_pro PRIVATE ${SERIAL_INCLUDE_DIRS})
target_link_libraries(yolov5_pro ${SERIAL_LDFLAGS})
target_link_libraries(yolov5_pro ${CMAKE_THREAD_LIBS_INIT})


include_directories(${PROJECT_SOURCE_DIR} ./DeepSORT/KalmanFilter ./DeepSORT/DeepAppearanceDescriptor ./DeepSORT/MunkresAssignment ./DeepSORT/MunkresAssignment/munkres)
file(GLOB SOURCE_FILES "./DeepSORT/*.cpp" "./DeepSORT/KalmanFilter/*.cpp" "./DeepSORT/DeepAppearanceDescriptor/*.cpp" "./DeepSORT/MunkresAssignment/*.cpp" "./DeepSORT/MunkresAssignment/munkres/*.cpp" )
add_library(deepsort ${SOURCE_FILES} ./DeepSORT/track_deepsort.cpp)
target_link_libraries(deepsort  /usr/lib/x86_64-linux-gnu/libuuid.so)
target_link_libraries(yolov5_pro Metavision::HAL_discovery Boost::program_options ${OpenCV_LIBS} Threads::Threads)
target_link_libraries(yolov5_pro deepsort)

if(UNIX)
add_definitions(-O2 -pthread)
endif(UNIX)


