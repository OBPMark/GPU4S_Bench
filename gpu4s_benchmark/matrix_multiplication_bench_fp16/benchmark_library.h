/** * ====================================================================
 * @file        benchmark_library.h (./matrix_multiplication_bench_fp16)
 * @brief       Specific memory structures and function overloads 
 *              for the Matrix Multiplication FP16 benchmark.
 * @paragraph   License
 * ESA-PL Strong Copyleft – v2.5
 * ======================================================================= */
#pragma once

// ======= Benchmark local variable =======
// --- Core Data Types ---
#ifdef INT
	typedef int bench_t_gpu;
#elif FLOAT16
	typedef float bench_t; //not in benchmark_common.h
#elif FLOAT
	typedef float bench_t_gpu;
#elif DOUBLE
	typedef double bench_t_gpu;
#endif

// Include all the benchmark common variable, struct, prototype, lib
#include "benchmark_common.h"


// --- OpenCL Runtime Kernel Code  ---
#ifdef OPENCL
	#ifdef FLOAT16
		// OpenCL float16 lib
		static const std::string type_kernel = 
			"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
			"typedef half bench_t;\n";
	#else 
		// Fallback for the other data type
		static const std::string type_kernel = type_kernel_common;
    #endif
#endif

// --- CUDA Runtime lib Code  ---
#ifdef CUDA
	#ifdef FLOAT16
		// CUDA float16 lib
		#include <cuda_fp16.h>
		typedef half bench_t_gpu;
    #endif
#endif

struct GraficObject : public GraficCommon {
	#ifdef CUDA
		// CUDA PART
		bench_t* d_A;
		bench_t* d_B;
		#ifdef FLOAT16
			bench_t_gpu* d_half_A;
			bench_t_gpu* d_half_B;
			bench_t_gpu* d_half_C;
		#endif
		bench_t* d_C;
	#elif OPENCL
		// OpenCL PART
		cl::Event *evt_copyA;
		cl::Event *evt_copyB;
		cl::Event *evt_copyC;
		cl::Event *evt;
		cl::Buffer *d_A;
		cl::Buffer *d_B;
		cl::Buffer *d_C;
		#ifdef FLOAT16
			cl::Buffer *d_half_A;
			cl::Buffer *d_half_B;
			cl::Buffer *d_half_C;
		#endif
	#else
		//CPU PART
	#endif

};

// --- Specefic overload of benchmarking function ---

