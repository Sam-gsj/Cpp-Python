get_filename_component(PARENT_DIR "${PROJECT_SOURCE_DIR}" DIRECTORY)
set(RKNPU_RUNTIME_PATH ${PARENT_DIR})
# include lib
if (EXISTS ${RKNPU_RUNTIME_PATH})
    set(RKNN_RT_LIB ${RKNPU_RUNTIME_PATH}/lib/librknnrt.so)
    include_directories(${RKNPU_RUNTIME_PATH}/include)
else ()
    message(FATAL_ERROR "[rknpu2.cmake] RKNPU_RUNTIME_PATH does not exist.")
endif ()
