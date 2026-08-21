/** * ====================================================================
 * @file        lib_opencl_common.h
 * @brief       Shared data structures, macros, and universal helpers 
 *              for hardware acceleration benchmarks.
 * @paragraph   License
 * ESA-PL Strong Copyleft – v2.5
 * ======================================================================= */
#pragma once
#include "benchmark_common.h"

// ============ global opencl function ============
inline bool openclError(const char* txt, const cl_int err){
    if (err != CL_SUCCESS) {
        fprintf(stderr, "%s (OpenCL error code %d)\n", txt, err);
        return true; // Error detected
    }
    return false;
}


#ifdef UMA_COMPATIBILITY
// ============ UMA struct ============

/**
 * @brief Encapsulates a mapping association between a host pointer address,
 *       an OpenCL device buffer, and an event handle.
 * 
 */
struct BufferMapCL {
    bench_t**   hostBuffer;   /**< Pointer to the host-side memory pointer address */
    cl::Buffer* deviceBuffer; /**< Pointer to the OpenCL device buffer object */
    cl::Event*  deviceEvent;  /**< Pointer to the OpenCL device event object used for profiling timing */
};
// ============ UMA function ============

/**
 * @brief Maps device memory buffers into the host's virtual address space,
 *        granting the CPU direct write access to the shared memory.
 * 
 * @tparam MapCL 
 * @param device_object 
 * @param memSize 
 * @param mapCL 
 */
template <typename... MapCL>
inline void map_unified_memory(GraficCommon* device_object, unsigned memSize, MapCL... mapCL) {
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    Clock mapCLK;
    cl_int err = CL_SUCCESS;
    cl_int lastErr = CL_SUCCESS;

    mapCLK.start();

    // --- C++17 Fold Expression Unrolled at compile-time  ---
    // For each MapCL buffer host buffer map device memory into the CPU's address space
    (( 
        // Map the buffer with CL_MAP_WRITE so CPU can directly use it
        *(mapCL.hostBuffer) = static_cast<bench_t*>(
            deviceObj->queue->enqueueMapBuffer(
                *(mapCL.deviceBuffer), CL_TRUE, CL_MAP_WRITE, 0, memSize, nullptr, mapCL.deviceEvent, &lastErr
            )
        ),
        // Save first error, so no failures are silently ignored
        (lastErr != CL_SUCCESS ? err = lastErr : CL_SUCCESS)
    ), ...);

    if (openclError("Failed to map buffer!", err)) return;

    deviceObj->queue->finish();
    mapCLK.end();

    // store the hd2h time
    deviceObj->h2d_elapsed_time += mapCLK.getElapsedNS();
}

/**
 * @brief Give CPU control of the shared memory back to the GPU, 
 *        ensuring the device has exclusive access to the buffers 
 * 
 * @tparam MapCL 
 * @param device_object 
 * @param mapCL 
 */
template <typename... MapCL>
inline void unmap_unified_memory(GraficCommon* device_object, MapCL... mapCL) {
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    Clock unmapCLK;
    cl_int err = CL_SUCCESS;
    cl_int lastErr = CL_SUCCESS;
    
    unmapCLK.start();

    // --- C++17 Fold Expression Unrolled at compile-time  ---
    // For each MapCL buffer unmap host buffer to device buffer
    ((
        lastErr = deviceObj->queue->enqueueUnmapMemObject(
            *(mapCL.deviceBuffer), *(mapCL.hostBuffer), NULL, mapCL.deviceEvent
        ),
        // Save first error, so no failures are silently ignored
        (lastErr != CL_SUCCESS ? err = lastErr : CL_SUCCESS)
    ), ...);

    if (openclError("Failed to unmap buffer!", err)) return;

    deviceObj->queue->finish();
    unmapCLK.end();

    // store the h2d time
    deviceObj->h2d_elapsed_time += unmapCLK.getElapsedNS();
}

/**
 * @brief Synchronizes the output memory back to the host,
 *        give CPU direct access to the GPU's results.
 * 
 * @tparam device_object 
 * @param d_C 
 * @param buff_size 
 */
template <typename... MapCL>
inline void map_unified_memory_to_host(GraficCommon* device_object, unsigned memSize, MapCL... mapCL) {
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    Clock mapCLK;
    cl_int err = CL_SUCCESS;
    cl_int lastErr = CL_SUCCESS;
    
    mapCLK.start();

    // --- C++17 Fold Expression Unrolled at compile-time  ---
    // For each MapCL buffer host buffer map device memory into the CPU's address space
    (( 
        *(mapCL.hostBuffer) = static_cast<bench_t*>(
            deviceObj->queue->enqueueMapBuffer(
                *(mapCL.deviceBuffer), CL_TRUE, CL_MAP_READ, 0, memSize, nullptr, mapCL.deviceEvent, &lastErr
            )
        ),
        // Save first error, so no failures are silently ignored
        (lastErr != CL_SUCCESS ? err = lastErr : CL_SUCCESS)
    ), ...);

    if (openclError("Failed to map buffer to host!", err)) return;

    deviceObj->queue->finish();
    mapCLK.end();

    // store the d2h time
    deviceObj->d2h_elapsed_time = mapCLK.getElapsedNS();
}
#endif




