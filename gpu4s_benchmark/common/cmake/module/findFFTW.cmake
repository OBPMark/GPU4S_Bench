# =======================================================================
# File:         findFFTW.cmake
# Description:  Locates host findFFTW installation and verifies
#               findFFTW dependencies for Android cross-compilation 
# License:      ESA-PL Strong Copyleft – v2.5
# =======================================================================
if(NOT ANDROID) # Computer
    # --- find package FFTW3 ---
    find_package(FFTW3 QUIET) 

    # --- User warning ---
    if(NOT FFTW3_FOUND)
        message(WARNING "FFTW3 installation was not found on this system. Skipping FFTW3 targets.")
    endif()

else() # ANDROID

    if(NOT EXISTS "${ANDROID_INC}/fftw3.h")
    # Download the FFTW3 header (fftw3.h)
        file(DOWNLOAD 
            "https://raw.githubusercontent.com/FFTW/fftw3/fftw-3.3.11/api/fftw3.h"
            "${ANDROID_INC}/fftw3.h"
            SHOW_PROGRESS
        )
    endif()


    # --- Check for depandancy files ---
    if( EXISTS ${ANDROID_LIB}${ANDROID_ABI}/libfftw3.a
    AND EXISTS ${ANDROID_LIB}${ANDROID_ABI}/libfftw3_omp.a)
        set(ANDROID_FFTW_LIB_INC TRUE)
    else()
        set(ANDROID_FFTW_LIB_INC FALSE)
        message(WARNING "FFTW3 android libs/headers files was not found. Skipping FFTW3 target. See README.md")
    endif()   

endif()