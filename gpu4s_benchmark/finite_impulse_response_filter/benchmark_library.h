/** * ====================================================================
 * @file        benchmark_library.h (./finite_impulse_response_filter)
 * @brief       Specific memory structures and function overloads 
 *              for the Finite Impulse Response Filter benchmark.
 * @paragraph   License
 * ESA-PL Strong Copyleft – v2.5
 * ======================================================================= */
#pragma once
// Include all the benchmark common variable, struct, prototype, lib
#include "benchmark_common.h"

// ======= Benchmark local variable =======
// --- Nothing ---

struct GraficObject : public GraficCommon {
	#ifdef CUDA 
		// CUDA PART
		bench_t* d_A;
		bench_t* d_B;
		bench_t* kernel;
   	#elif OPENCL
		// OpenCL PART
		cl::Event *evt_copyA;
		cl::Event *evt_copyB;
		cl::Event *evt_copyC;
		cl::Event *evt;
		cl::Buffer *d_A;
		cl::Buffer *d_B;
		cl::Buffer *kernel;
	#elif HIP
		// Hip part --
		bench_t* d_A;
		bench_t* d_B;
		bench_t* kernel;
	#elif OPENMP
		// OpenMP part
		bench_t* d_A;
		bench_t* d_B;
		bench_t* kernel;
	#else
		// CUDA PART
		bench_t* d_A;
		bench_t* d_B;
		bench_t* kernel;
	#endif
};

// --- Specefic overload of benchmarking function ---
void execute_kernel(GraficCommon* device_object, unsigned int n, unsigned int m, unsigned int w, unsigned int kernel_size);


#ifdef UMA_COMPATIBILITY
/**
 * @brief Maps three device buffers of three independent sizes into host-visible memory
 * 
 * @param device_object Pointer to the device common structure
 * @param A Reference to receive the mapped input host pointer (d_A)
 * @param B Reference to receive the mapped kernel-weights host pointer (kernel)
 * @param C Reference to receive the mapped output host pointer (d_B)
 * @param memSize Size of A, in bytes
 * @param memSize2 Size of B, in bytes
 * @param memSize3 Size of C, in bytes
 */
void get_unified_memory_pointers(GraficCommon* device_object, bench_t* &A, bench_t* &B, bench_t* &C, unsigned sizeA, unsigned sizeB, unsigned sizeC);

/**
 * @brief Unmaps all three buffers, blocked for host until the device give aigain ownership 
 *        
 * @param device_object Pointer to the device common structure
 * @param A Reference to the mapped input host pointer to unmap
 * @param B Reference to the mapped kernel-weights host pointer to unmap
 * @param C Reference to the mapped output host pointer to unmap
 */
void sync_unified_memory_to_device(GraficCommon* device_object, bench_t* &A, bench_t* &B, bench_t* &C);

#endif