# =======================================================================
# File:         findHIP.cmake
# Description:  Locates host HIP installation 
# License:      ESA-PL Strong Copyleft – v2.5
# =======================================================================
if(NOT ANDROID)
    # --- find package HIP ---
    find_package(hip QUIET)
    if(hip_FOUND)
        enable_language(HIP)
    endif()

    # --- User warning ---
    if(NOT hip_FOUND)
        message(WARNING "HIP installation was not found on this system. Skipping HIP targets.")
    endif()

endif(NOT ANDROID)


