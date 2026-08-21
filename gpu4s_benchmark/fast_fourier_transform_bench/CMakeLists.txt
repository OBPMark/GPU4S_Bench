# =======================================================================
# File:         CMakeLists.txt (./fast_fourier_transform_bench)
# Description:  Build targets for Fast Fourier Transform benchmark
# Target:       CPU, OpenMP, OpenCL, CUDA, HIP
# License:      ESA-PL Strong Copyleft – v2.5
# =======================================================================

cmake_minimum_required(VERSION 3.24)
project(FFT CXX) 

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
include(findVkFFT)
include(findFFTW)

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
            BENCH_DIR           ${CMAKE_CURRENT_SOURCE_DIR}
            SOURCES_FILES       opencl/lib_opencl.cpp
                                opencl/lib_opencl_common.cpp

            COMPILE_DEFS        OPENCL
                                CL_HPP_TARGET_OPENCL_VERSION=${OPENCL_VERSION}
                                
            INCLUDES            ${ANDROID_INC}
            LIBRARIES           ${ANDROID_LIB}/${ANDROID_ABI}/libOpenCL.so
            SHORTCUTS_NAMES     cl OpenCL
        )
        # --- OpenCL-opt ---
        compile_target(${PROJECT_NAME}_opencl_opt
            BENCH_DIR           ${CMAKE_CURRENT_SOURCE_DIR}
            SOURCES_FILES       opencl/lib_opencl_opt.cpp
                                opencl/lib_opencl_common.cpp

            COMPILE_DEFS        OPENCL   
                                CL_HPP_TARGET_OPENCL_VERSION=${OPENCL_VERSION} 

            INCLUDES            ${ANDROID_INC}
            LIBRARIES           ${ANDROID_LIB}/${ANDROID_ABI}/libOpenCL.so
            SHORTCUTS_NAMES     opencl-opt OpenCL-opt
        )


        if(VKFFT_FOUND)
            # --- OpenCL-lib ---
            compile_target(${PROJECT_NAME}_opencl_lib
                BENCH_DIR           ${CMAKE_CURRENT_SOURCE_DIR}
                SOURCES_FILES       opencl/lib_opencl_lib.cpp
                                    opencl/lib_opencl_common.cpp

                COMPILE_DEFS        OPENCL   
                                    OPENCL VKFFT_BACKEND=3
                                    CL_HPP_TARGET_OPENCL_VERSION=${OPENCL_VERSION} 

                COMPILE_OPTIONS     -Wno-deprecated-declarations    #hide warning from vkfft 
                                    -Wno-comment                    #hide comment error

                INCLUDES            ${ANDROID_INC} 
                                    ${VKFFT_INCLUDE_DIR}

                LIBRARIES           ${ANDROID_LIB}/${ANDROID_ABI}/libOpenCL.so  
                                    ${ANDROID_LIB}/${ANDROID_ABI}/libfftw3.a                        
                SHORTCUTS_NAMES    opencl-lib OpenCL-lib
            )
        endif()
    endif()

    # --- OpenMP---
    compile_target(${PROJECT_NAME}_openmp
        BENCH_DIR           ${CMAKE_CURRENT_SOURCE_DIR}
        SOURCES_FILES       openmp/lib_omp.cpp
                            openmp/omp_common.cpp

        COMPILE_DEFS        OPENMP 
        COMPILE_OPTIONS     -fopenmp  
        LIBRARIES           -static-openmp   #openmp flags
                            -fopenmp  
                            -lm              # add math lib

        SHORTCUTS_NAMES     openmp OpenMP
    )

    # --- OpenMP-opt ---
    compile_target(${PROJECT_NAME}_openmp_opt
        BENCH_DIR           ${CMAKE_CURRENT_SOURCE_DIR}
        SOURCES_FILES       openmp/lib_omp_opt.cpp
                            openmp/omp_common.cpp
                            
        COMPILE_DEFS        OPENMP 
        COMPILE_OPTIONS     -fopenmp  
        LIBRARIES           -static-openmp   #openmp flags
                            -fopenmp  
                            -lm              # add math lib

        SHORTCUTS_NAMES     openmp-opt OpenMP-opt
    )

    if(ANDROID_FFTW_LIB_INC)    
        # --- OpenMP-lib ---
        compile_target(${PROJECT_NAME}_openmp_lib
            BENCH_DIR           ${CMAKE_CURRENT_SOURCE_DIR}
            SOURCES_FILES       openmp/lib_omp_opt.cpp
                                openmp/omp_common.cpp

            COMPILE_DEFS        OPENMP 
            COMPILE_OPTIONS     -fopenmp  
            LIBRARIES           -static-openmp   #openmp flags
                                -fopenmp  
                                -lm              # add math lib
                                ${ANDROID_LIB}/${ANDROID_ABI}/libfftw3_omp.a  
                                ${ANDROID_LIB}/${ANDROID_ABI}/libfftw3.a   


            SHORTCUTS_NAMES     openmp-lib OpenMP-lib
        )
    endif()

endif(ANDROID)

# --- Computeur targets ---
if(NOT ANDROID)
    # --- OpenCL Targets ---
    if(OpenCL_FOUND)
        # --- OpenCL ---
        compile_target(${PROJECT_NAME}_opencl
            BENCH_DIR           ${CMAKE_CURRENT_SOURCE_DIR}
            SOURCES_FILES       opencl/lib_opencl.cpp
                                opencl/lib_opencl_common.cpp

            COMPILE_DEFS        OPENCL
                                CL_HPP_TARGET_OPENCL_VERSION=${OPENCL_VERSION}

            LIBRARIES           OpenCL::OpenCL
            SHORTCUTS_NAMES     cl OpenCL
        )

        # --- OpenCL-opt ---
        compile_target(${PROJECT_NAME}_opencl_opt
            BENCH_DIR           ${CMAKE_CURRENT_SOURCE_DIR}
            SOURCES_FILES       opencl/lib_opencl_opt.cpp
                                opencl/lib_opencl_common.cpp

            COMPILE_DEFS        OPENCL
                                CL_HPP_TARGET_OPENCL_VERSION=${OPENCL_VERSION}

            LIBRARIES           OpenCL::OpenCL
            SHORTCUTS_NAMES     opencl-opt OpenCL-opt
        )

        if(VKFFT_FOUND AND FFTW3_FOUND)        
            # --- OpenCL-lib ---
            compile_target(${PROJECT_NAME}_opencl_lib
                BENCH_DIR           ${CMAKE_CURRENT_SOURCE_DIR}
                SOURCES_FILES       opencl/lib_opencl_lib.cpp
                                    opencl/lib_opencl_common.cpp

                COMPILE_DEFS        OPENCL
                                    OPENCL VKFFT_BACKEND=3
                                    CL_HPP_TARGET_OPENCL_VERSION=${OPENCL_VERSION}

                COMPILE_OPTIONS     -Wno-deprecated-declarations   #hide warning from vkfft 
                INCLUDES            ${VKFFT_INCLUDE_DIR}
                LIBRARIES           OpenCL::OpenCL fftw3 
                SHORTCUTS_NAMES     opencl-lib OpenCL-lib
            )
        endif()
      
    endif()

    # --- OpenMP Targets ---
    if(OpenMP_CXX_FOUND)
        # --- OpenMP ---
        compile_target(${PROJECT_NAME}_openmp
            BENCH_DIR           ${CMAKE_CURRENT_SOURCE_DIR}
            SOURCES_FILES       openmp/lib_omp.cpp
                                openmp/omp_common.cpp

            COMPILE_DEFS        OPENMP

            LIBRARIES           OpenMP::OpenMP_CXX # openmp flags
                                m   # equivalent to -lm

            SHORTCUTS_NAMES     openmp OpenMP 
        )

        # --- OpenMP-opt ---
        compile_target(${PROJECT_NAME}_openmp_opt
            BENCH_DIR           ${CMAKE_CURRENT_SOURCE_DIR}
            SOURCES_FILES       openmp/lib_omp_opt.cpp
                                openmp/omp_common.cpp

            COMPILE_DEFS        OPENMP

            LIBRARIES           OpenMP::OpenMP_CXX #  openmp flags
                                m   # equivalent to -lm 

            SHORTCUTS_NAMES     openmp-opt OpenMP-opt 
        )

        if(FFTW3_FOUND)    
            # --- OpenMP-lib ---
            compile_target(${PROJECT_NAME}_openmp_lib
                BENCH_DIR           ${CMAKE_CURRENT_SOURCE_DIR}
                SOURCES_FILES       openmp/lib_omp_opt.cpp
                                    openmp/omp_common.cpp

                COMPILE_DEFS        OPENMP

                LIBRARIES           OpenMP::OpenMP_CXX # openmp flags
                                    m                  # Corresponds to -lm
                                    fftw3_omp         # Corresponds to -lfftw3_omps
                                    fftw3             # Corresponds to -lfftw3
                                
                SHORTCUTS_NAMES     openmp-lib OpenMP-lib
            )
        endif()

    endif()


    # --- CUDA Targets---
    if(CUDAToolkit_FOUND)
        # --- CUDA ---
        compile_target(${PROJECT_NAME}_cuda
            BENCH_DIR           ${CMAKE_CURRENT_SOURCE_DIR}
            SOURCES_FILES       cuda/lib_cuda.cu
                                cuda/cuda_common.cu

            COMPILE_DEFS        CUDA
            SET_CUDA            1
            LIBRARIES           CUDA::cudart 
            SHORTCUTS_NAMES     cuda CUDA
        )

        # --- CUDA-opt ---
        compile_target(${PROJECT_NAME}_cuda_opt
            BENCH_DIR           ${CMAKE_CURRENT_SOURCE_DIR}
            SOURCES_FILES       cuda/lib_cuda_opt.cu
                                cuda/cuda_common.cu

            COMPILE_DEFS        CUDA
            SET_CUDA            1
            LIBRARIES           CUDA::cudart 
            SHORTCUTS_NAMES     cuda-opt CUDA-opt
        )

        # --- CUDA-lib ---
        compile_target(${PROJECT_NAME}_cuda_lib
            BENCH_DIR           ${CMAKE_CURRENT_SOURCE_DIR}
            SOURCES_FILES       cuda/lib_cuda_lib.cu
                                cuda/cuda_common.cu

            COMPILE_DEFS        CUDA
            SET_CUDA            1
            LIBRARIES           CUDA::cudart CUDA::cufft fftw3
            SHORTCUTS_NAMES     cuda-lib CUDA-lib
        )
    endif()

    # --- HIP targets---
    if(hip_FOUND)
        # --- HIP ---²
        compile_target(${PROJECT_NAME}_hip
            BENCH_DIR           ${CMAKE_CURRENT_SOURCE_DIR}
            SOURCES_FILES       hip/lib_hip.cpp
                                hip/hip_common.cpp 

            SET_HIP_FILES       hip/lib_hip.cpp 
                                hip/hip_common.cpp 

            COMPILE_DEFS        HIP
            LIBRARIES           hip::host
            SHORTCUTS_NAMES     hip- HIP
        )

        # --- HIP-opt ---
        compile_target(${PROJECT_NAME}_hip_opt
            BENCH_DIR           ${CMAKE_CURRENT_SOURCE_DIR}
            SOURCES_FILES       hip/lib_hip_opt.cpp 
                                hip/hip_common.cpp 

            SET_HIP_FILES       hip/lib_hip_opt.cpp 
                                hip/hip_common.cpp 
                                
            COMPILE_DEFS        HIP
            LIBRARIES           hip::host
            SHORTCUTS_NAMES     hip-opt HIP-opt
        )
    endif()
endif(NOT ANDROID)


# ====== Shorcuts ======

# --- all-opencl (Android) ---
if(ANDROID AND ANDROID_OPENCL_LIB_INC )
    add_custom_target(${SHORTCUT_PREFIX}all-opencl DEPENDS 
        ${PROJECT_NAME}_opencl
        ${PROJECT_NAME}_opencl_opt 
        ${PROJECT_NAME}_opencl_lib
    )
endif()

# --- all-opencl---
if(NOT ANDROID AND OpenCL_FOUND )
    add_custom_target(${SHORTCUT_PREFIX}all-opencl DEPENDS 
        ${PROJECT_NAME}_opencl
        ${PROJECT_NAME}_opencl_opt 
        ${PROJECT_NAME}_opencl_lib
    )
endif()


# --- all-openmp (Android) ---
if(ANDROID AND ANDROID_OPENBLAS_LIB_INC  )
    add_custom_target(${SHORTCUT_PREFIX}all-openmp DEPENDS 
        ${PROJECT_NAME}_openmp 
        ${PROJECT_NAME}_openmp_opt 
        ${PROJECT_NAME}_openmp_lib
    )
endif()


# --- all-openmp---
if(NOT ANDROID AND OpenMP_CXX_FOUND )
    add_custom_target(${SHORTCUT_PREFIX}all-openmp DEPENDS 
        ${PROJECT_NAME}_openmp 
        ${PROJECT_NAME}_openmp_opt 
        ${PROJECT_NAME}_openmp_lib
    )
endif()


# --- all-hip---
if(NOT ANDROID AND hip_FOUND)
    add_custom_target(${SHORTCUT_PREFIX}all-hip DEPENDS 
        ${PROJECT_NAME}_hip 
        ${PROJECT_NAME}_hip_opt 
    )
endif()


# --- all-cuda---
if(NOT ANDROID AND CUDAToolkit_FOUND)
    add_custom_target(${SHORTCUT_PREFIX}all-cuda DEPENDS 
        ${PROJECT_NAME}_cuda 
        ${PROJECT_NAME}_cuda_opt 
        ${PROJECT_NAME}_cuda_lib
    )
endif()