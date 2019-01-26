cmake_minimum_required(VERSION 2.8)

set(CMAKE_C_COMPILER /usr/bin/gcc-5)
set(CMAKE_CXX_COMPILER /usr/bin/g++-5)

execute_process(COMMAND python3 -c "import tensorflow as tf; print(tf.sysconfig.get_lib())" OUTPUT_VARIABLE Tensorflow_INCLUDE_LIBS)

execute_process(COMMAND python3 -c "import tensorflow; print(tensorflow.sysconfig.get_include())" OUTPUT_VARIABLE Tensorflow_INCLUDE_DIRS)


# C++11 required for tensorflow
set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "-O3 ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "-shared ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "-fPIC ${CMAKE_CXX_FLAGS}")

set(CMAKE_CXX_FLAGS "-I${VIRTUAL_ENV}/lib/python3.6/site-packages/tensorflow/include/external/nsync/public  ${CMAKE_CXX_FLAGS}")

set(CMAKE_CXX_FLAGS "-L${VIRTUAL_ENV}/lib/python3.6/site-packages/tensorflow/ -ltensorflow_framework ${CMAKE_CXX_FLAGS}")

set(CMAKE_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0 ${CMAKE_CXX_FLAGS}")

include_directories(${Tensorflow_INCLUDE_DIRS})
include_directories(${Tensorflow_INCLUDE_LIBS})

#add_library(zero_out SHARED zero_out.cc)

#include_directories(${Tensorflow_INCLUDE_DIRS})
#add_library(zero_out_grad SHARED zero_out_grad.cc)

SET(CUDA_PROPAGATE_HOST_FLAGS ON)
set(CUDA_NVCC_FLAGS "-arch=sm_61")
set(CUDA_NVCC_FLAGS "--expt-relaxed-constexpr ${CUDA_NVCC_FLAGS}")

include_directories(${Tensorflow_INCLUDE_DIRS})
FIND_PACKAGE(CUDA)
#CUDA_ADD_LIBRARY(
#    zero_out SHARED
#    zero_out.cu
#    zero_out_grad.cc
#    zero_out.cc)

CUDA_ADD_LIBRARY(
    interp SHARED
    grid_interpolate.cc
    grid_interpolate.cu
)
