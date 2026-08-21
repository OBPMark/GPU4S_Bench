# =======================================================================
# File:         CMakeLists.txt (./memory_bandwidth_bench)
# Description:  Build targets for Memory Bandwidth benchmark
# Target:       OpenCL, CUDA, HIP
# License:      ESA-PL Strong Copyleft – v2.5
# =======================================================================

cmake_minimum_required(VERSION 3.24)
project(memory_bandwidth CXX) 

# ====== Include cmake module ======
list(APPEND CMAKE_MODULE_PATH 
    "${CMAKE_CURRENT_SOURCE_DIR}/../common/cmake/"
    "${CMAKE_CURRENT_SOURCE_DIR}/../common/cmake/module"
)
# Global:
include(setup)
include(compileBlueprint)
# Module:
include(findHIP)

# show the configuration of the project
set(NO_OPENMP_TARGET true) #For the list of backend
if(PROJECT_IS_TOP_LEVEL)
    showConfig()
endif()

# ====== Compilation of the targets ======

compile_target(${PROJECT_NAME}_cpu
    BENCH_DIR               ${CMAKE_CURRENT_SOURCE_DIR}
    SOURCES_FILES           cpu/lib_cpu.cpp 
    SHORTCUTS_NAMES         cpu CPU
)

# --- Android Targets---
if(ANDROID)

    if(ANDROID_OPENCL_LIB_INC)
    # --- OpenCL ---
    compile_target(${PROJECT_NAME}_opencl
        BENCH_DIR          ${CMAKE_CURRENT_SOURCE_DIR}
        SOURCES_FILES      opencl/lib_opencl.cpp
        COMPILE_DEFS       OPENCL
                           CL_HPP_TARGET_OPENCL_VERSION=${OPENCL_VERSION}
                             
        INCLUDES           ${ANDROID_INC}
        LIBRARIES          ${ANDROID_LIB}/${ANDROID_ABI}/libOpenCL.so
        SHORTCUTS_NAMES    cl OpenCL
    )
    endif()

endif(ANDROID)

# --- Computeur targets ---
if(NOT ANDROID)
    # --- OpenCL Targets ---
    if(OpenCL_FOUND)
        # --- OpenCL ---
        compile_target(${PROJECT_NAME}_opencl
            BENCH_DIR          ${CMAKE_CURRENT_SOURCE_DIR}
            SOURCES_FILES      opencl/lib_opencl.cpp
            COMPILE_DEFS       OPENCL
                               CL_HPP_TARGET_OPENCL_VERSION=${OPENCL_VERSION}

            LIBRARIES          OpenCL::OpenCL 
            SHORTCUTS_NAMES    cl OpenCL
        )

    endif()

    # --- CUDA Targets---
    if(CUDAToolkit_FOUND)
        # --- CUDA ---
        compile_target(${PROJECT_NAME}_cuda
            BENCH_DIR          ${CMAKE_CURRENT_SOURCE_DIR}
            SOURCES_FILES      cuda/lib_cuda.cu
            COMPILE_DEFS       CUDA
            SET_CUDA           1
            LIBRARIES          CUDA::cudart CUDA::cuda_driver
            SHORTCUTS_NAMES    cuda CUDA
        )

    endif()

    # --- HIP targets---
    if(hip_FOUND)
        # --- HIP ---
        compile_target(${PROJECT_NAME}_hip
            BENCH_DIR          ${CMAKE_CURRENT_SOURCE_DIR}
            SOURCES_FILES       hip/lib_hip.cpp 
            SET_HIP_FILES       hip/lib_hip.cpp 
            COMPILE_DEFS        HIP
            LIBRARIES           hip::host
            SHORTCUTS_NAMES     hip- HIP
        )

    endif()
endif(NOT ANDROID)


# ====== Shorcuts ======

# --- all-opencl (Android) ---
if(ANDROID AND ANDROID_OPENCL_LIB_INC)
    add_custom_target(${SHORTCUT_PREFIX}all-opencl DEPENDS 
        ${PROJECT_NAME}_opencl
    )
endif()

# --- all-opencl---
if(NOT ANDROID AND OpenCL_FOUND)
    add_custom_target(${SHORTCUT_PREFIX}all-opencl DEPENDS 
        ${PROJECT_NAME}_opencl
    )
endif()


# --- all-hip---
if(NOT ANDROID AND hip_FOUND)
    add_custom_target(${SHORTCUT_PREFIX}all-hip DEPENDS 
        ${PROJECT_NAME}_hip 
    )
endif()


# --- all-cuda---
if(NOT ANDROID AND CUDAToolkit_FOUND)
    add_custom_target(${SHORTCUT_PREFIX}all-cuda DEPENDS 
        ${PROJECT_NAME}_cuda 
    )
endif()

