cmake_minimum_required(VERSION 3.5.0)
project(static_poly)

set(WARNING_FLAGS "-Wall -Wextra -Wfatal-errors -Werror")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${WARNING_FLAGS} -g")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g3")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DNDEBUG -O3")
set(CMAKE_EXE_LINKER_FLAGS "-g")

set(CMAKE_CXX_STANDARD 14)

find_package(CUDA QUIET REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/include)
link_directories("/usr/local/cuda/lib64")

set(CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    #"-arch=sm_35"
    "-g -G"
    "-std=c++11"
    "-Xptxas -v"
    )

SET(CUDA_PROPAGATE_HOST_FLAGS OFF)

cuda_compile(DRIVER_O ${CMAKE_SOURCE_DIR}/main.cu)
cuda_add_executable(monte_carlo_gpu ${DRIVER_O})
target_link_libraries(monte_carlo_gpu -lcurand)

