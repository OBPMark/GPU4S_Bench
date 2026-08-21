# =======================================================================
# File:         showCOnfig.cmake
# Description:  Global configuration script managing variables, flags,
#               and hardware dependency configurations (OpenCL/CUDA) for
#               both host and Android environments.
# License:      ESA-PL Strong Copyleft – v2.5
# =======================================================================

# --- Define the color---
string(ASCII 27 Esc)
set(ColorReset "${Esc}[0m")
set(ColorBold  "${Esc}[1m")
set(ColorRed   "${Esc}[1;31m")
set(ColorGreen "${Esc}[1;32m")
set(ColorBlue  "${Esc}[1;34m")
set(ColorCyan  "${Esc}[1;36m")


if(ANDROID)
    set(TARGET_PLATFORM "Android (${ANDROID_ABI}, ${ANDROID_PLATFORM})")
else()
    set(TARGET_PLATFORM "Computer (${CMAKE_SYSTEM_PROCESSOR})")
endif()

function(showConfig)

    # --- check the active targets ---
    set(ACTIVE_TARGETS "CPU") # CPU target is always compiled

    if(ANDROID)
        if(ANDROID_OPENCL_LIB_INC)
            string(APPEND ACTIVE_TARGETS ", OpenCL")
        endif()
        if(NOT NO_OPENMP_TARGET)
            string(APPEND ACTIVE_TARGETS ", OpenMP")
        endif()
    else()
        if(OpenCL_FOUND)
            string(APPEND ACTIVE_TARGETS ", OpenCL")
        endif()
        if(OpenMP_CXX_FOUND)
            string(APPEND ACTIVE_TARGETS ", OpenMP")
        endif()
        if(CUDAToolkit_FOUND)
            string(APPEND ACTIVE_TARGETS ", CUDA")
        endif()
        if(hip_FOUND)
            string(APPEND ACTIVE_TARGETS ", HIP")
        endif()
    endif()

# --- show all the configuration ---
    message(STATUS "${ColorCyan}============ ${ColorReset}${ColorBold}GPU4S Benchmark Configuration 🚀 ${ColorCyan}============${ColorReset}")
    message(STATUS "  Benchmark             : ${ColorBlue}${PROJECT_NAME}${ColorReset}")
    message(STATUS "  Target Platform       : ${ColorBlue}${TARGET_PLATFORM}${ColorReset}")
    if(NOT ${PROJECT_NAME} STREQUAL gpu4s_benchmark_global)
    message(STATUS "  Active Backends       : ${ColorBlue}${ACTIVE_TARGETS}${ColorReset}")
    endif()
    message(STATUS "  Data Type             : ${ColorBold}${DATATYPE}${ColorReset}")
    message(STATUS "  Block Size            : ${ColorBold}${BLOCKSIZE}${ColorReset}")
    if(NSTREAMS)
        if(${PROJECT_NAME} STREQUAL gpu4s_benchmark_global)
            message(STATUS "  Number of Sreams      : ${ColorBold}${NSTREAMS}${ColorReset}  //Used in CIFAR_10_MUTIPLE")
        else()
            message(STATUS "  Number of Sreams      : ${ColorBold}${NSTREAMS}${ColorReset}")
        endif()
    endif()
    message(STATUS "  Optimization Level    : ${ColorRed}${OPT_FLAG}${ColorReset}")
    message(STATUS "  CUDA Architecture     : ${ColorRed}${CUDA_ARCH}${ColorReset}")
    message(STATUS "  OpenCL Version        : ${ColorRed}${OPENCL_VERSION}${ColorReset}")
    # show the list of benchark 
    message(STATUS "${ColorCyan}==========================================================${ColorReset}")
endfunction(showConfig)
