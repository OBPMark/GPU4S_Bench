/** * ====================================================================
 * @file        benchmark_library.h (./fast_fourier_transform_window_bench)
 * @brief       Specific memory structures and function overloads 
 *              for the Fast Fourier Transform Window benchmark.
 * @paragraph   License
 * ESA-PL Strong Copyleft – v2.5
 * ======================================================================= */
#pragma once
// Include all the benchmark common variable, struct, prototype, lib
#include "benchmark_common.h"

// ======= Benchmark local variable =======
// --- CUDA Data types  ---
#ifdef CUDA
	// CUDA lib
	#include <cufft.h>
	#ifdef FLOAT
		typedef cufftComplex bench_cuda_complex;
	#elif DOUBLE 
		typedef cufftDoubleComplex bench_cuda_complex;
	#endif
#endif

struct GraficObject : public GraficCommon {
	#ifdef CUDA
		// CUDA PART
		#ifdef LIB
			bench_cuda_complex* d_A;
			bench_cuda_complex* d_B;
		#else
			bench_t* d_A;
			bench_t* d_B;
		#endif
   	#elif OPENCL
		// OpenCL PART
		cl::Event *evt_copyA;
		cl::Event *evt_copyB;
		cl::Event *evt;
		cl::Event *evt_end;
		cl::Buffer *d_A;
		cl::Buffer *d_B;
	#elif HIP
		// Hip part 
		bench_t* d_A;
		bench_t* d_B;
	#elif OPENMP
		bench_t* d_A;
		bench_t* d_B;
		bench_t* d_Br;
	#else
		//CPU PART
		bench_t* d_A;
		bench_t* d_B;
		bench_t* d_Br;
	#endif
};


// --- Specefic overload of benchmarking function ---
bool device_memory_init(GraficCommon* device_object, int64_t size_a_array, int64_t size_b_array);
void copy_memory_to_device(GraficCommon* device_object, bench_t* h_A,int64_t size);
void execute_kernel(GraficCommon* device_object,int64_t window, int64_t n);
void copy_memory_to_host(GraficCommon* device_object, bench_t* h_B, int64_t size);
