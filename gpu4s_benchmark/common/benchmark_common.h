/** * ====================================================================
 * @file        benchmark_common.h
 * @brief       Shared data structures, macros, and universal helpers 
 *              for hardware acceleration benchmarks.
 * @paragraph   License
 * ESA-PL Strong Copyleft – v2.5
 * ======================================================================= */
#pragma once

// --- Standard lib ---
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>

// --- project lib ---
#include "Clock.h"

// --- Specefic framework lib ---
#ifdef CUDA
    // CUDA lib
    #include <cuda_runtime.h>
#elif OPENCL
    // OpenCL lib
    #include <CL/opencl.hpp>
    // #include <CL/cl.hpp> // OLD framework
#elif HIP
	// HIP part
    #include <hip/hip_runtime.h>

    inline void hipDumbSync() 
    {
            void* dumb_ptr;
            hipError_t err = hipMalloc(&dumb_ptr, 4);
                    err = hipMemset(dumb_ptr, 0, 4);
                    err = hipFree(dumb_ptr);

            if (err != hipSuccess)
            {
                fprintf(stderr, "Enable to create the dumb obj (error code %s)!\n", hipGetErrorString(err));
                return;
            }
    }
#elif OPENMP
    // OpenMP lib
    #include <omp.h>
#else
    //CPU part
#endif

// --- UMA + profiling mangement ---
#if defined(ANDROID) && defined(OPENCL) 
    #define FORCE_PROFILING_CLOCK
    #define UMA_COMPATIBILITY 
#endif



// ======= Commmon variable =======
// --- Core Data Types ---
#ifdef INT
	#define __ptype "%d"
    typedef int bench_t;
#elif FLOAT
	#define __ptype "%f"
    typedef float bench_t;
#elif DOUBLE
	#define __ptype "%f"
    typedef double bench_t;
#endif

// --- OpenCL Runtime Kernel Code  ---
#ifdef OPENCL
    #ifdef INT
        static const std::string type_kernel_common = "typedef int bench_t;\n";
    #elif FLOAT
        static const std::string type_kernel_common = "typedef float bench_t;\n";
    #elif DOUBLE
        static const std::string type_kernel_common = "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\ntypedef double bench_t;\n";
    #endif
#endif


// --- Commmon struct ---
struct GraficCommon{
	#ifdef CUDA 
        // --- CUDA Variable ---
        cudaEvent_t *start_memory_copy_device;
        cudaEvent_t *stop_memory_copy_device;
        cudaEvent_t *start_memory_copy_host;
        cudaEvent_t *stop_memory_copy_host;
        cudaEvent_t *start;
        cudaEvent_t *stop;
   	#elif OPENCL
        // --- OpenCL variable ---
        cl::Context *context;
        cl::CommandQueue *queue;
        cl::Device default_device;
	#elif HIP
		// --- Hip variable ---
		hipEvent_t *start_memory_copy_device;
		hipEvent_t *stop_memory_copy_device;
		hipEvent_t *start_memory_copy_host;
		hipEvent_t *stop_memory_copy_host;
		hipEvent_t *start;
		hipEvent_t *stop;
	#elif OPENMP
		// --- OpenMP part----
	#else
		// --- CPU variable ---
	#endif
    // --- clock profiling ---
	float h2d_elapsed_time  = 0.0f;
	float d2h_elapsed_time  = 0.0f;
    float elapsed_time      = 0.0f;
	bool  profiling_clock   = false;
};


// ====== Fonction Prototype ======
// --- Standard initialization use by every benchmarks ---
void init(GraficCommon *device_object, char* device_name);
void init(GraficCommon *device_object, int platform, int device, char* device_name);


// --- Initialization of the memory ---
// Overload for Correlation2D, LRN, max_pooling, memory_bandwidth, relu, softmax, wavelet_transform 
bool device_memory_init(GraficCommon *device_object, unsigned int size_a_matrix, unsigned int size_b_matrix);
// Overload for Convolution2D, FIR, matrix_mult:(naïve/FP16/tensor) 
bool device_memory_init(GraficCommon *device_object, unsigned int size_a_matrix, unsigned int size_b_matrix, unsigned int size_c_matrix);


// --- Copy RAM memory to GPU memory ---
// Overload for LRN, max_pooling, memory_bandwidth, relu, softmax, wavelet_transform 
void copy_memory_to_device(GraficCommon *device_object, bench_t* h_A, unsigned int size_a);
// Overload for Convolution2D, FIR, matrix_mult:(naïve/FP16/tensor) 
void copy_memory_to_device(GraficCommon *device_object, bench_t* h_A, bench_t* h_B, unsigned int size_a, unsigned int size_b);


// --- Launch the benchmarks ---
// Overload for Correlation2D, wavelet_transform, Memory bandwidth 
void execute_kernel(GraficCommon *device_object, unsigned int n);
// Overload for LRN, matrix_mult:(naïve/FP16/tensor), relu, softmax 
void execute_kernel(GraficCommon *device_object, unsigned int n, unsigned int m, unsigned int w);


// --- Copy back to CPU RAM memory ---
// Overload for Cifar_10, Convolution2D, FIR, LRN, matrix_mult:(naïve/FP16/tensor), max_pooling, memory_bandwidth, relu, softmax, wavelet_transform 
void copy_memory_to_host(GraficCommon *device_object, bench_t* h_C, int size);


// --- return the duration of the benchmark ---
// Overload for matrix_mult:(FP16/tensor), memory_bandwidth 
float get_elapsed_time(GraficCommon *device_object, bool csv_format);

// Standard prototype  of get_elapsed_time used by most of the benchmarks :
// Overload for Cifar_10(naïve/mutiple), Convolution2D, Correlation2D, fft:(Naïve/2D/window), FIR, LRN, matrix_mult:(naïve), max_pooling, relu, softmax, wavelet_transform 
float get_elapsed_time(GraficCommon *device_object, bool csv_format, bool csv_format_timestamp, long int timestamp);

// Standard clean prototype used by every the benchmarks
void clean(GraficCommon *device_object);


/// --- UMA memory function ---
#ifdef UMA_COMPATIBILITY

    // --- 1 buffer ---
    /**
     * @brief Maps one device buffer into host-visible memory (blocking write-map).
     * @param device_object Pointer to the device common structure
     * @param A Reference to receive the mapped host pointer
     * @param memSize Size of the buffer to map, in bytes
     */
    void get_unified_memory_pointers(GraficCommon* device_object, bench_t* &A, unsigned int memSize);
    /**
     * @brief Unmaps a single buffer, blocking until the device regains ownership.
     * @param device_object Pointer to the device common structure
     * @param A Reference to the mapped host pointer to unmap
     */
    void sync_unified_memory_to_device(GraficCommon* device_object, bench_t* &A);

    // --- 2 buffer ---
    /**
     * @brief Maps two equal-sized device buffers into host-visible memory 
     * @param device_object Pointer to the device common structure
     * @param A Reference to receive the first mapped host pointer
     * @param B Reference to receive the second mapped host pointer
     * @param memSize Size of EACH buffer to map, in bytes - both buffers share this one size
     */
    void get_unified_memory_pointers(GraficCommon* device_object, bench_t* &A, bench_t* &B, unsigned int memSize);
    /**
     * @brief Unmaps two buffers, blocked for host until the device give aigain ownership 
     * @param device_object Pointer to the device common structure
     * @param A Reference to the first mapped host pointer to unmap
     * @param B Reference to the second mapped host pointer to unmap
     */
    void sync_unified_memory_to_device(GraficCommon* device_object, bench_t* &A, bench_t* &B);

    // --- 3 buffer ---
    /**
     * @brief Maps three equal-sized device buffers into host-visible memory
     * @param device_object Pointer to the device common structure
     * @param A Reference to receive the first mapped host pointer
     * @param B Reference to receive the second mapped host pointer
     * @param C Reference to receive the third mapped host pointer
     * @param memSize Size of EACH buffer, in bytes. 
     */
    void get_unified_memory_pointers(GraficCommon* device_object, bench_t* &A, bench_t* &B, bench_t* &C, unsigned int memSize);
    /**
     * @brief Unmaps three buffers, blocked for host until the device give aigain ownership
     * @param device_object Pointer to the device common structure
     * @param A Reference to the first mapped host pointer to unmap
     * @param B Reference to the second mapped host pointer to unmap
     * @param C Reference to the third mapped host pointer to unmap
     */
    void sync_unified_memory_to_device(GraficCommon* device_object, bench_t* &A, bench_t* &B, bench_t* &C);

    /**
     * @brief Maps output result buffer back to host
     * @param device_object Pointer to the device common structure
     * @param d_output Reference to receive the mapped output host pointer
     * @param size_output Size of the output buffer to map, in bytes
     */
    void sync_unified_memory_to_host(GraficCommon* device_object, bench_t* &d_output, unsigned int size_output);
#endif