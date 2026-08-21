# =======================================================================
# File:         findVkFFT.cmake
# Description:  Locates or downloads VkFFT header-only library
#               for host and Android cross-compilation
# License:      ESA-PL Strong Copyleft – v2.5
# =======================================================================

# --- Download VkFFT Headers ---
if(NOT EXISTS "${EXTERN_DIR}VkFFT/vkFFT.h")
    message(STATUS "Downloading  v1.3.4 VkFFT headers...")
    
    # Download VkFFT headers
    FetchContent_Declare(VkFFT
        GIT_REPOSITORY https://github.com/DTolm/VkFFT.git
        GIT_TAG        v1.3.4
        GIT_SHALLOW    TRUE
        EXCLUDE_FROM_ALL     # don't build cmake list
    )
    FetchContent_Populate(VkFFT)

    # cp the VkFFT directory into ${FOLDER_DIR}
    file(COPY "${vkfft_SOURCE_DIR}/vkFFT/vkFFT.h" DESTINATION "${EXTERN_DIR}VkFFT")
    file(COPY "${vkfft_SOURCE_DIR}/vkFFT/vkFFT"   DESTINATION "${EXTERN_DIR}VkFFT")

endif()

# set DIR and FOUND
set(VKFFT_INCLUDE_DIR "${EXTERN_DIR}/VkFFT/")
set(VKFFT_FOUND TRUE)

