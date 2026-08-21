# =======================================================================
# File:         findCLBlast.cmake
# Description:  Locates host CLBlast installation and verifies
#               CLBlast dependencies for Android cross-compilation 
# License:      ESA-PL Strong Copyleft – v2.5
# =======================================================================
if(NOT ANDROID) # Computer
    # --- find package CLBlast ---
    find_package(CLBlast QUIET) 

    # --- User warning ---
    if(NOT CLBlast_FOUND)
        message(WARNING "CLBlast installation was not found on this system. Skipping OpenCL-lib target.")
    endif()

else() # ANDROID

    # --- Download clblast Headers ---
    if(NOT EXISTS "${ANDROID_INC}/clblast.h")
        message(STATUS "Downloading 1.7.0 clblast.h header...")
        file(DOWNLOAD 
            "https://raw.githubusercontent.com/CNugteren/CLBlast/1.7.0/include/clblast.h"
            "${ANDROID_INC}/clblast.h"
            SHOW_PROGRESS
        )
    endif()


    # --- Check for depandancy files ---
    if(EXISTS ${ANDROID_LIB}${ANDROID_ABI}/libclblast.a)
        set(ANDROID_CLBLAST_LIB_INC TRUE)
    else()
        set(ANDROID_CLBLAST_LIB_INC FALSE)
        message(WARNING "clblast android libs file was not found. Skipping OpenCL-lib target. See README.md")
    endif()

endif()