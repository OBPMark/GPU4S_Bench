# =======================================================================
# File:         findOpenMP.cmake
# Description:  Locates host OpenMP installation 
# License:      ESA-PL Strong Copyleft – v2.5
# =======================================================================
if(NOT ANDROID) # Computer
    # --- find package OpenMP ---
    find_package(OpenMP QUIET)

    # --- User warning ---
    if(NOT OpenMP_CXX_FOUND)
            message(WARNING "OpenMP installation was not found on this system. Skipping OpenMP-lib target.")
    endif()

else() # ANDROID
    # Nothing to do already include in the SDK
endif()