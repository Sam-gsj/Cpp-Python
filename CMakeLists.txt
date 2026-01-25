cmake_minimum_required(VERSION 3.4.1)

project(rknn_yolov5_demo)

# 设置 Debug 编译选项
set(CMAKE_BUILD_TYPE Debug)

# 为调试模式添加编译器标志
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")  # -g 添加调试信息，-O0 禁用优化（可以帮助调试）

# 设置 Debug 模式下的链接器标志
set(CMAKE_EXE_LINKER_FLAGS_DEBUG "-g")
# 启用堆栈跟踪
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer")



set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-pthread")

# skip 3rd-party lib dependencies
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-shlib-undefined")

# install target and libraries
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/rknn_yolov5_demo_${CMAKE_SYSTEM_NAME})

set(CMAKE_SKIP_INSTALL_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")


set(LIB_ARCH aarch64)
set(RKNN_RT_LIB ${CMAKE_SOURCE_DIR}/lib/librknnrt.so)

#rga
set(RGA_PATH ${CMAKE_SOURCE_DIR}/include/3rdparty/rga/RK3588)
set(RGA_LIB ${RGA_PATH}/lib/Linux//${LIB_ARCH}/librga.so)



include_directories(${CMAKE_SOURCE_DIR}/include/3rdparty)
include_directories(${RGA_PATH}/include)

# opencv
find_package(OpenCV REQUIRED)


include_directories( ${CMAKE_SOURCE_DIR}/include)



add_executable( resnet18
        src/main.cpp
        src/postprocess.cc
        src/preprocess.cc
        src/rkResnet.cpp
        src/utils.cpp
        src/ilogger.cpp
)

target_link_libraries(resnet18
  ${RKNN_RT_LIB}
  ${OpenCV_LIBS}
  ${RGA_LIB}
)


add_executable( resnet18_thread
        src/main_thread.cpp
        src/postprocess.cc
        src/preprocess.cc
        src/rkResnet.cpp
        src/utils.cpp
        src/ilogger.cpp
)

target_link_libraries(resnet18_thread
  ${RKNN_RT_LIB}
  ${OpenCV_LIBS}
  ${RGA_LIB}
)
