cmake_minimum_required(VERSION 3.24)
project(nvfp4_sm100 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 100a)
set(CUTLASS_DIR "" CACHE PATH "Path to a CUTLASS 4.x checkout")
if(NOT EXISTS "${CUTLASS_DIR}/include/cutlass/cutlass.h")
  message(FATAL_ERROR "Set CUTLASS_DIR to a CUTLASS checkout containing include/cutlass")
endif()

find_package(CUDAToolkit REQUIRED)
add_executable(nvfp4_sm100 nvfp4_sm100.cu)
target_include_directories(nvfp4_sm100 PRIVATE
  "${CUTLASS_DIR}/include"
  "${CUTLASS_DIR}/examples/common"
  "${CUTLASS_DIR}/tools/util/include"
)
target_link_libraries(nvfp4_sm100 PRIVATE CUDA::cudart)
target_compile_options(nvfp4_sm100 PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-O3>
)
