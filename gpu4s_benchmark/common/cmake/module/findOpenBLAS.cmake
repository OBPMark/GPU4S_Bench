# =======================================================================
# File:         findOpenBLAS.cmake
# Description:  Locates host OpenBLAS installation and verifies
#               OpenBLAS dependencies for Android cross-compilation 
# License:      ESA-PL Strong Copyleft – v2.5
# =======================================================================
if(NOT ANDROID) # Computer
    # --- find package ATLAS OR OPENBLAS ---
    find_package(BLAS QUIET) 
    if(BLAS_FOUND)
        find_path(BLAS_INCLUDE_DIRS 
            NAMES cblas.h cblas_atlas.h
            PATH_SUFFIXES openblas blas atlas
        )
    endif()

    # --- User warning ---
    if(NOT BLAS_INCLUDE_DIRS)
        message(WARNING "${BLA_VENDOR} installation was not found on this system. Skipping OpenMP-lib target.")
    endif()

else() # ANDROID

    # --- Check for depandancy files ---
    if(EXISTS ${ANDROID_LIB}${ANDROID_ABI}/libopenblas.a
    AND EXISTS ${ANDROID_INC}${ANDROID_ABI}/openblas_config.h
    AND BLA_VENDOR STREQUAL "OpenBLAS")
        set(ANDROID_OPENBLAS_LIB_INC TRUE)
    else()
        set(ANDROID_OPENBLAS_LIB_INC FALSE)
        message(WARNING "Openblast android libs/headers files was not found. Skipping OpenMP-lib target. See README.md")
    endif()   

endif()