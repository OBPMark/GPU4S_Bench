# =======================================================================
# File:         findCUDNN.cmake
# Description:  look if CUDNN is already installed
# License:      ESA-PL Strong Copyleft – v2.5
# =======================================================================

if(NOT ANDROID)
    find_package(CUDNN QUIET) 
    find_library(CUDNN_LIBRARY 
        NAMES cudnn libcudnn
        HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64 
              /usr/local/cuda/lib64 
              /usr/lib64 
              /usr/lib/x86_64-linux-gnu
    )

    # --- User warning ---
    if(NOT CUDNN_LIBRARY)
        set(CUDNN_FOUND false)
        message(WARNING "CUDNN installation was not found on this system. Skipping CUDNN targets.")
    else()
        set(CUDNN_FOUND true)
    endif()

endif(NOT ANDROID)
