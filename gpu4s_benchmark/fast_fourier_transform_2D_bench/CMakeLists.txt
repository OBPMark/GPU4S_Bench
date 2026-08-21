# =======================================================================
# File:         CMakeLists.txt (./fast_fourier_transform_2D_bench)
# Description:  Build targets for 2D Fast Fourier Transform benchmark
# Target:       CPU, OpenCL, CUDA,
# License:      ESA-PL Strong Copyleft – v2.5
# ======================================================================

cmake_minimum_required(VERSION 3.24)
project(FFT_2D CXX) 

# ====== Include cmake module ======
list(APPEND CMAKE_MODULE_PATH 
    "${CMAKE_CURRENT_SOURCE_DIR}/../common/cmake/"
    "${CMAKE_CURRENT_SOURCE_DIR}/../common/cmake/module"
)
# Global:
include(setup)
include(compileBlueprint)
# Module:
include(findVkFFT)
include(findFFTW)

# show the configuration of the project
set(NO_OPENMP_TARGET true) #For the list of backend
if(PROJECT_IS_TOP_LEVEL)
    showConfig()
endif()


# ====== Compilation of the targets ======


# --- Android Targets---
if(ANDROID)

    if(ANDROID_FFTW_LIB_INC)
        # --- CPU target ---
        compile_target(${PROJECT_NAME}_cpu
            BENCH_DIR               ${CMAKE_CURRENT_SOURCE_DIR}
            SOURCES_FILES           cpu/lib_cpu_lib.cpp 
            INCLUDES                ${ANDROID_INC}
            LIBRARIES               ${ANDROID_LIB}/${ANDROID_ABI}/libfftw3.a  
            SHORTCUTS_NAMES         cpu CPU
        )

        if(ANDROID_OPENCL_LIB_INC AND VKFFT_FOUND)
            # --- OpenCL-lib ---
            compile_target(${PROJECT_NAME}_opencl_lib
                BENCH_DIR           ${CMAKE_CURRENT_SOURCE_DIR}
                SOURCES_FILES       opencl/lib_opencl_lib.cpp
                COMPILE_DEFS        OPENCL   
                                    OPENCL VKFFT_BACKEND=3
                                    CL_HPP_TARGET_OPENCL_VERSION=${OPENCL_VERSION} 

                COMPILE_OPTIONS     -Wno-deprecated-declarations   #hide warning from vkfft 
                INCLUDES            ${ANDROID_INC} 
                                    ${VKFFT_INCLUDE_DIR}

                LIBRARIES           ${ANDROID_LIB}/${ANDROID_ABI}/libOpenCL.so  
                                    ${ANDROID_LIB}/${ANDROID_ABI}/libfftw3.a                        
                SHORTCUTS_NAMES    opencl-lib OpenCL-lib
            )
        endif()
    endif()
endif(ANDROID)

# --- Computeur targets ---
if(NOT ANDROID)

    if(FFTW3_FOUND)
        # --- CPU target ---
        compile_target(${PROJECT_NAME}_cpu
            BENCH_DIR               ${CMAKE_CURRENT_SOURCE_DIR}
            SOURCES_FILES           cpu/lib_cpu_lib.cpp 
            LIBRARIES               fftw3
            SHORTCUTS_NAMES         cpu CPU
        )

        # --- OpenCL Targets ---
        if(OpenCL_FOUND AND VKFFT_FOUND)
            # --- OpenCL-lib ---
            compile_target(${PROJECT_NAME}_opencl_lib
                BENCH_DIR           ${CMAKE_CURRENT_SOURCE_DIR}
                SOURCES_FILES       opencl/lib_opencl_lib.cpp
                COMPILE_DEFS        OPENCL
                                    OPENCL VKFFT_BACKEND=3
                                    CL_HPP_TARGET_OPENCL_VERSION=${OPENCL_VERSION}

                COMPILE_OPTIONS     -Wno-deprecated-declarations   #hide warning from vkfft 
                INCLUDES            ${VKFFT_INCLUDE_DIR}
                LIBRARIES           OpenCL::OpenCL fftw3 
                SHORTCUTS_NAMES     opencl-lib OpenCL-lib
            )
        endif()

    


        # --- CUDA Targets---
        if(CUDAToolkit_FOUND)
            # --- CUDA-lib ---
            compile_target(${PROJECT_NAME}_cuda_lib
                BENCH_DIR           ${CMAKE_CURRENT_SOURCE_DIR}
                SOURCES_FILES       cuda/lib_cuda_lib.cu
                COMPILE_DEFS        CUDA
                SET_CUDA            1
                LIBRARIES           CUDA::cudart CUDA::cufft fftw3
                SHORTCUTS_NAMES     cuda-lib CUDA-lib
            )
        endif()
    endif()
endif(NOT ANDROID)


# ====== Shorcuts ======

# --- all-opencl (Android) ---
if(ANDROID AND ANDROID_OPENCL_LIB_INC)
    add_custom_target(${SHORTCUT_PREFIX}all-opencl DEPENDS 
        ${PROJECT_NAME}_opencl_lib
    )
endif()

# --- all-opencl---
if(NOT ANDROID AND OpenCL_FOUND )
    add_custom_target(${SHORTCUT_PREFIX}all-opencl DEPENDS 
        ${PROJECT_NAME}_opencl_lib
    )
endif()


# --- all-cuda---
if(NOT ANDROID AND CUDAToolkit_FOUND)
    add_custom_target(${SHORTCUT_PREFIX}all-cuda DEPENDS 
        ${PROJECT_NAME}_cuda_lib
    )
endif()