cmake_minimum_required(VERSION 3.4.1)

project(test)

# 设置为发行版
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)
add_definitions(-O3)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# skip 3rd-party lib dependencies
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-shlib-undefined")


set(CMAKE_SKIP_INSTALL_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

if (CMAKE_C_COMPILER MATCHES "aarch64")
  set(LIB_ARCH aarch64)
else()
  set(LIB_ARCH armhf)
endif()

include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/3rdparty)
include_directories(${CMAKE_SOURCE_DIR}/3rdparty/plog)





# opencv
set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/3rdparty/opencv/opencv-linux-aarch64/share/OpenCV)
find_package(OpenCV REQUIRED)





#ffmpeg
set(FFMPEG_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/FFmpeg4.4)
set(FFMPEG_LIB ${FFMPEG_DIR}/lib)
include_directories(${FFMPEG_DIR}/include)


# mpp
set(MPP_PATH ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/mpp)
set(MPP_LIBS ${MPP_PATH}/lib/librockchip_mpp.so.0)
include_directories(${MPP_PATH}/include)



aux_source_directory(./src DIR_SRCS)
add_executable(test
    ${DIR_SRCS}
)


target_link_libraries(test
  ${OpenCV_LIBS}
  ${MPP_LIBS}
  ${FFMPEG_LIB}/libavformat.a
  ${FFMPEG_LIB}/libavdevice.a
  ${FFMPEG_LIB}/libavcodec.a
  ${FFMPEG_LIB}/libavutil.a
  ${FFMPEG_LIB}/libavfilter.a
  pthread
  ${FFMPEG_LIB}/libswscale.a
  ${FFMPEG_LIB}/libswresample.a
)
