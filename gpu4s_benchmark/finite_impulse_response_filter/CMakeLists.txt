# =======================================================================
# File:         CMakeLists.txt (./finite_impulse_response_filter)
# Description:  Build targets for Finite Impulse Response Filter benchmark
# Target:       CPU, OpenMP, OpenCL, CUDA, HIP
# License:      ESA-PL Strong Copyleft – v2.5
# =======================================================================

cmake_minimum_required(VERSION 3.24)
project(FIR_filter CXX) 

# ====== Include cmake module ======
list(APPEND CMAKE_MODULE_PATH 
    "${CMAKE_CURRENT_SOURCE_DIR}/../common/cmake/"
    "${CMAKE_CURRENT_SOURCE_DIR}/../common/cmake/module"
)
# Global:
include(setup)
include(compileBlueprint)
# Module:
include(findOpenMP)
include(findHIP)
include(findOpenBLAS)

# show the configuration of the project
if(PROJECT_IS_TOP_LEVEL)
    showConfig()
endif()

# ====== Compilation of the targets ======

# --- CPU target ---
compile_target(${PROJECT_NAME}_cpu
    BENCH_DIR          ${CMAKE_CURRENT_SOURCE_DIR}
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

    # --- OpenMP---
    compile_target(${PROJECT_NAME}_openmp
        BENCH_DIR          ${CMAKE_CURRENT_SOURCE_DIR}
        SOURCES_FILES      openmp/lib_omp.cpp
        COMPILE_DEFS       OPENMP 
        COMPILE_OPTIONS    -fopenmp  
        LIBRARIES          -static-openmp   #openmp flags
                           -fopenmp  
                           -lm              # add math lib

        SHORTCUTS_NAMES    openmp OpenMP
    )


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

    # --- OpenMP Targets ---
    if(OpenMP_CXX_FOUND)
        # --- OpenMP ---
        compile_target(${PROJECT_NAME}_openmp
            BENCH_DIR          ${CMAKE_CURRENT_SOURCE_DIR}
            SOURCES_FILES      openmp/lib_omp.cpp
            COMPILE_DEFS       OPENMP

            LIBRARIES          OpenMP::OpenMP_CXX # openmp flags
                               m   # equivalent to -lm

            SHORTCUTS_NAMES    openmp OpenMP 
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
            LIBRARIES          CUDA::cudart 
            SHORTCUTS_NAMES    cuda CUDA
        )

    endif()

    # --- HIP targets---
    if(hip_FOUND)
        # --- HIP ---²
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
if(ANDROID AND ANDROID_OPENCL_LIB_INC )
    add_custom_target(${SHORTCUT_PREFIX}all-opencl DEPENDS 
        ${PROJECT_NAME}_opencl
    )
endif()

# --- all-opencl---
if(NOT ANDROID AND OpenCL_FOUND )
    add_custom_target(${SHORTCUT_PREFIX}all-opencl DEPENDS 
        ${PROJECT_NAME}_opencl
    )
endif()


# --- all-openmp (Android) ---
if(ANDROID AND ANDROID_OPENBLAS_LIB_INC  )
    add_custom_target(${SHORTCUT_PREFIX}all-openmp DEPENDS 
        ${PROJECT_NAME}_openmp 
    )
endif()


# --- all-openmp---
if(NOT ANDROID AND OpenMP_CXX_FOUND )
    add_custom_target(${SHORTCUT_PREFIX}all-openmp DEPENDS 
        ${PROJECT_NAME}_openmp 
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