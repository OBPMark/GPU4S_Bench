# =======================================================================
# File:         CMakeLists.txt (./matrix_multiplication_tensor_bench)
# Description:  Build targets for Matrix Multiplication Tensor benchmark
# Target:       OpenCL, CUDA
# License:      ESA-PL Strong Copyleft – v2.5
# =======================================================================

cmake_minimum_required(VERSION 3.24)
project(matrix_mult_tensor CXX) 

# ====== Include cmake module ======
list(APPEND CMAKE_MODULE_PATH 
    "${CMAKE_CURRENT_SOURCE_DIR}/../common/cmake/"
    "${CMAKE_CURRENT_SOURCE_DIR}/../common/cmake/module"
)
# Global:
include(setup)
include(compileBlueprint)
# Module:
include(findCLBlast)

# show the configuration of the project
set(NO_OPENMP_TARGET true) #For the list of backend
if(PROJECT_IS_TOP_LEVEL)
    showConfig()
endif()

# ====== Compilation ======

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
    endif()

    if(ANDROID_OPENCL_LIB_INC AND ANDROID_CLBLAST_LIB_INC)
    # --- OpenCL-lib ---
    compile_target(${PROJECT_NAME}_opencl_lib   
        BENCH_DIR           ${CMAKE_CURRENT_SOURCE_DIR}
        SOURCES_FILES       opencl/lib_opencl_lib.cpp
                            opencl/lib_opencl_common.cpp

        COMPILE_DEFS        OPENCL   
                            CL_HPP_TARGET_OPENCL_VERSION=${OPENCL_VERSION} 

        INCLUDES            ${ANDROID_INC}
        LIBRARIES           ${ANDROID_LIB}/${ANDROID_ABI}/libOpenCL.so
                            ${ANDROID_LIB}/${ANDROID_ABI}/libclblast.a

        SHORTCUTS_NAMES     opencl-lib OpenCL-lib
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

        # --- OpenCL-lib ---
        if(CLBlast_FOUND)
            compile_target(${PROJECT_NAME}_opencl_lib
                BENCH_DIR           ${CMAKE_CURRENT_SOURCE_DIR}
                SOURCES_FILES       opencl/lib_opencl_lib.cpp
                                    opencl/lib_opencl_common.cpp

                COMPILE_DEFS        OPENCL
                                    CL_HPP_TARGET_OPENCL_VERSION=${OPENCL_VERSION}

                LIBRARIES           OpenCL::OpenCL clblast
                SHORTCUTS_NAMES     opencl-lib OpenCL-lib
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

            COMPILE_DEFS        CUDA ${DATATYPEGPU}
            SET_CUDA            1
            LIBRARIES           CUDA::cudart 
            SHORTCUTS_NAMES     cuda-opt CUDA-opt
        )

        # --- CUDA-lib ---
        compile_target(${PROJECT_NAME}_cuda_lib
            BENCH_DIR           ${CMAKE_CURRENT_SOURCE_DIR}
            SOURCES_FILES       cuda/lib_cuda_lib.cu
                                cuda/cuda_common.cu

            COMPILE_DEFS        CUDA ${DATATYPEGPU}
            SET_CUDA            1
            LIBRARIES           CUDA::cudart CUDA::cublas
            SHORTCUTS_NAMES     cuda-lib CUDA-lib
        )
    endif()

endif(NOT ANDROID)


# ====== Shorcuts ======

# --- all-opencl (Android) ---
if(ANDROID AND ANDROID_OPENCL_LIB_INC AND ANDROID_CLBLAST_LIB_INC )
    add_custom_target(${SHORTCUT_PREFIX}all-opencl DEPENDS 
        ${PROJECT_NAME}_opencl
        ${PROJECT_NAME}_opencl_opt 
        ${PROJECT_NAME}_opencl_lib
    )
endif()

# --- all-opencl---
if(NOT ANDROID AND OpenCL_FOUND AND CLBlast_FOUND )
    add_custom_target(${SHORTCUT_PREFIX}all-opencl DEPENDS 
        ${PROJECT_NAME}_opencl
        ${PROJECT_NAME}_opencl_opt 
        ${PROJECT_NAME}_opencl_lib
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



