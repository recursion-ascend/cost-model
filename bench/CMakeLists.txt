# 访存常数微基准。构建方式照抄 mega_moe/include/CMakeLists.txt 的 ascendc 用法。
#
#   source /your/cann/set_env.sh
#   cmake -S bench -B build_bench -DSOC_VERSION=ascend950pr_957c
#   cmake --build build_bench -j
#   ./build_bench/bench_mem 0 > bench_mem.csv
#   python bench/analyze.py bench_mem.csv
#
# SOC_VERSION 需与你的卡一致 (megamoe 用的是 ascend950pr_957c)。
cmake_minimum_required(VERSION 3.16)
project(BenchMem CXX)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(SOC_VERSION ascend950pr_957c CACHE STRING "Target SoC")
set(ASCEND_CANN_PACKAGE_PATH $ENV{ASCEND_HOME_PATH})
if(NOT ASCEND_CANN_PACKAGE_PATH)
    message(FATAL_ERROR "ASCEND_HOME_PATH 未设置 —— 先 source set_env.sh")
endif()
include(${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/tikcpp/ascendc_kernel_cmake/ascendc.cmake)

set(SDK_COMMON "${ASCEND_CANN_PACKAGE_PATH}/opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/common")
set(INCS
    "${ASCEND_CANN_PACKAGE_PATH}/include"
    "${ASCEND_CANN_PACKAGE_PATH}/include/op_common"
    "${ASCEND_CANN_PACKAGE_PATH}/asc/include"
    "${ASCEND_CANN_PACKAGE_PATH}/asc/include/adv_api"
    "${ASCEND_CANN_PACKAGE_PATH}/asc/include/utils"
    "${SDK_COMMON}"
    "${SDK_COMMON}/tensor_api"
    "${SDK_COMMON}/tensor_api/include")

ascendc_library(bench_mem_kernel SHARED bench_mem_kernel.cpp)
ascendc_include_directories(bench_mem_kernel PRIVATE ${INCS})

add_executable(bench_mem bench_mem_main.cpp)
target_compile_options(bench_mem PRIVATE -O2 -std=c++17)
target_include_directories(bench_mem PRIVATE "${ASCEND_CANN_PACKAGE_PATH}/include")
target_link_directories(bench_mem PRIVATE "${ASCEND_CANN_PACKAGE_PATH}/lib64")
target_link_libraries(bench_mem PRIVATE bench_mem_kernel ascendcl)
