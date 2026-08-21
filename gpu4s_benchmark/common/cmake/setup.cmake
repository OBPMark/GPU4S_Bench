# =======================================================================
# File:         setup.cmake
# Description:  Global configuration script managing variables, flags,
#               and hardware dependency configurations (OpenCL/CUDA) for
#               both host and Android environments.
# License:      ESA-PL Strong Copyleft – v2.5
# =======================================================================

# ====== CMake Configuration ======
include(FetchContent)
include(showConfig)

# --- Global Variable ---
set(OPENCL_VERSION 300        CACHE STRING "API OpenCL Version : 200, 210, 220, 300, 310")
set(ENDIANFLAGS    "little"         CACHE STRING "ENDIANFLAGS : LITTLENDIAN, BIGENDIAN")
set(OPT_FLAG       "-O3"      CACHE STRING "Compiler optimization level : -O2, -O3, -Ofast")
set(CUDA_ARCH      "native"   CACHE STRING "API CUDA version: native, sm_72-86")
set(BLA_VENDOR     "OpenBLAS" CACHE STRING "BLAS lib : ATLAS, OpenBLAS")

if(NOT BLOCKSIZE)
    set(BLOCKSIZE       16        CACHE STRING "Block size for tiled kernels")
    set(BLOCK_MESSAGE true)
endif()

if(NOT DATATYPE)
    set(DATATYPE       "FLOAT"    CACHE STRING "Data type: FLOAT, DOUBLE, INT")
    set(DATA_MESSAGE true)
endif()

# extern dir
set(EXTERN_DIR     ${CMAKE_CURRENT_SOURCE_DIR}/../common/extern/)

# --- Android Variable ---
if(ANDROID)
    set(ANDROID_INC ${CMAKE_CURRENT_SOURCE_DIR}/../common/android/include/)
    set(ANDROID_LIB ${CMAKE_CURRENT_SOURCE_DIR}/../common/android/libs/) 
endif(ANDROID)


# --- Define Output directory ---
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/)

# --- Global Settings ---
set(CMAKE_CXX_STANDARD 17)    # force clang 17 or g++ 17 compiler
set(CMAKE_BUILD_TYPE Release) # add production-level optimization
set(CMAKE_POSITION_INDEPENDENT_CODE ON) # set -fPIC flag
add_compile_options(${OPT_FLAG})



# ====== Package/language Handler ======
if(NOT ANDROID)
    # Searching for package installation
    find_package(OpenCL QUIET) 

    # --- CUDA ---
    find_package(CUDAToolkit QUIET)

    if(CUDAToolkit_FOUND AND CUDAToolkit_NVCC_EXECUTABLE)
        set(CMAKE_CUDA_COMPILER ${CUDAToolkit_NVCC_EXECUTABLE})
        enable_language(CUDA)
    endif()
    

endif()

# ====== Global user warning ======
if(NOT ANDROID)

    if(NOT OpenCL_FOUND)
        message(WARNING "OpenCL installation was not found on this system. Skipping opencl targets.")
    endif()

    if(NOT CUDAToolkit_FOUND)
        message(WARNING "CUDA installation was not found on this system. Skipping CUDA targets.")
    endif()

else()
    # --- Download OpenCL Headers ---
    if(NOT EXISTS "${ANDROID_INC}/CL/opencl.hpp")
        message(STATUS "Downloading v2026.05.29 OpenCL C++ headers...")
        
        # Download Khronos OpenCL C headers
        FetchContent_Declare(opencl_headers
            GIT_REPOSITORY https://github.com/KhronosGroup/OpenCL-Headers.git
            GIT_TAG        v2026.05.29
            GIT_SHALLOW    TRUE
        )
        FetchContent_MakeAvailable(opencl_headers)
        
        # cp the CL directory into ${ANDROID_INC}
        file(COPY "${opencl_headers_SOURCE_DIR}/CL" DESTINATION "${ANDROID_INC}")

        # Download the modern C++ wrapper (opencl.hpp)
        file(DOWNLOAD 
            "https://raw.githubusercontent.com/KhronosGroup/OpenCL-CLHPP/v2026.05.29/include/CL/opencl.hpp"
            "${ANDROID_INC}/CL/opencl.hpp"
            SHOW_PROGRESS
        )
    endif()

    # --- Check for depandancy files ---
    if(EXISTS ${ANDROID_LIB}${ANDROID_ABI}/libOpenCL.so)
            set(ANDROID_OPENCL_LIB_INC TRUE)
    else()
        set(ANDROID_OPENCL_LIB_INC FALSE)
        message(WARNING "Opencl android libs file was not found. Skipping OpenCL targets. See README.md")
    endif()
endif()



# ====== Shorcuts ======

# Classic shortcut (framework)
set(SHORTCUT_PREFIX "" CACHE INTERNAL "Prefix for shortcut targets")

# --- check if global cmake is used ---
if(NOT PROJECT_IS_TOP_LEVEL)
    # compex shortcut (bench + framework)
    set(SHORTCUT_PREFIX "${PROJECT_NAME}-")
endif()
