/** * ====================================================================
 * @file        benchmark_library.h (./LRN_bench)
 * @brief       Specific memory structures and function overloads 
 *              for the LRN benchmark.
 * @paragraph   License
 * ESA-PL Strong Copyleft – v2.5
 * ======================================================================= */
#pragma once
// Include all the benchmark common variable, struct, prototype, lib
#include "benchmark_common.h"

// ======= Benchmark local variable =======
// --- Compute ---
const bench_t K = 2;
const bench_t ALPHA = 10e-4;
const bench_t BETA = 0.75;

struct GraficObject : public GraficCommon {
	#ifdef CUDA
		// CUDA PART
		bench_t* d_A;
		bench_t* d_B;
   	#elif OPENCL
		// OpenCL PART
		cl::Event *evt_copyA;
		cl::Event *evt_copyB;
		cl::Event *evt;
		cl::Buffer *d_A;
		cl::Buffer *d_B;
	#elif HIP
		// Hip part 
		bench_t* d_A;
		bench_t* d_B;	
	#elif OPENMP
		// OpenMP part
		bench_t* d_A;
		bench_t* d_B;
	#else
		// CPU part
		bench_t* d_A;
		bench_t* d_B;
	#endif
};

// --- Specefic overload of benchmarking function ---
