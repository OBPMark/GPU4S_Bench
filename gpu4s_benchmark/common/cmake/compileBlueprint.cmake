# =======================================================================
# File:         compileBlueprint.cmake
# Description:  Defines compile_target() function used by all
#               benchmark CMakeLists.txt to build targets
# License:      ESA-PL Strong Copyleft – v2.5
# =======================================================================

# =======================================================================
# compile_target() — Main build function for benchmark targets
#
# Usage:
#  compile_target(<TARGET_NAME>
#      BENCH_DIR        <path>          # Root directory of the benchmark
#      SOURCES_FILES    <files...>      # Backend source files (opencl, cuda, hip, omp...)
#      SET_HIP_FILES    <files...>      # Files to explicitly set as HIP language
#      COMPILE_DEFS     <defs...>       # Compiler definitions (e.g. OPENCL, CUDA, FLOAT)
#      COMPILE_OPTIONS  <flags...>      # Extra compiler flags (e.g. -fopenmp)
#      SET_CUDA         <0|1>           # Enable CUDA architecture property
#      INCLUDES         <dirs...>       # Additional include directories
#      LIBRARIES        <libs...>       # Libraries to link against
#      SHORTCUTS_NAMES  <names...>      # Custom shortcut target aliases
#  )
#
# =======================================================================
function(compile_target TARGET_NAME)

# Parse arguments
set(singleArgs   BENCH_DIR SET_CUDA) 
set(multipleArgs SOURCES_FILES SET_HIP_FILES COMPILE_DEFS SHORTCUTS_NAMES LIBRARIES INCLUDES COMPILE_OPTIONS) 

cmake_parse_arguments(ARG "" "${singleArgs}" "${multipleArgs}" ${ARGN})

    # --- add bench dir to src dir ---
    foreach(SRC IN LISTS ARG_SOURCES_FILES)
        list(APPEND ABSOLUTE_SOURCES "${ARG_BENCH_DIR}/${SRC}")
    endforeach()

    # --- Add all source file ---
    add_executable(${TARGET_NAME}
        ${ARG_BENCH_DIR}/main.cpp
        ${ABSOLUTE_SOURCES}
        ${ARG_BENCH_DIR}/cpu_functions/cpu_functions.cpp
    )

    # --- Set specefic files as HIP API files ---
    if(ARG_SET_HIP_FILES)
        foreach(HIP_FILE IN LISTS ARG_SET_HIP_FILES)
            set_source_files_properties(${ARG_BENCH_DIR}/${HIP_FILE} PROPERTIES LANGUAGE HIP)
        endforeach()
    endif()

    # --- Add compile define ---
    target_compile_definitions(${TARGET_NAME} PRIVATE
        ${DATATYPE}
        BLOCK_SIZE=${BLOCKSIZE}
        ${ARG_COMPILE_DEFS}
        ${ENDIANFLAGS}
        ${PROFILING}    
    )

    # --- Set specific compiler flags ---
    if(ARG_COMPILE_OPTIONS)
        target_compile_options(${TARGET_NAME} PRIVATE ${ARG_COMPILE_OPTIONS})
    endif()

    # --- Set target CUDA ARCHITECTURES ---
    if(ARG_SET_CUDA)
        set_target_properties(${TARGET_NAME} PROPERTIES CUDA_ARCHITECTURES ${CUDA_ARCH})
    endif()
    
    # --- Includes header directories ---
    target_include_directories(${TARGET_NAME} PRIVATE 
        ${ARG_INCLUDES}
        "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/.."
    )
    
    # --- Link required libraries ---
    if(ARG_LIBRARIES)
        target_link_libraries(${TARGET_NAME} PRIVATE ${ARG_LIBRARIES}) 
    endif()

    # --- Create custom shortcut targets ---
    foreach(SHORTCUT IN LISTS ARG_SHORTCUTS_NAMES)
        if(PROJECT_IS_TOP_LEVEL)
            # Classic shortcut (framework)
            add_custom_target(${SHORTCUT} DEPENDS ${TARGET_NAME})
        else()
            # compex shortcut (bench + framework)
            add_custom_target("${PROJECT_NAME}-${SHORTCUT}" DEPENDS ${TARGET_NAME})
        endif()
    endforeach()

endfunction()