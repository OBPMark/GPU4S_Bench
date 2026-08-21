/** * ====================================================================
 * @file        benchmark_library.h (./max_pooling_bench)
 * @brief       Specific memory structures and function overloads 
 *              for the Max Pooling benchmark.
 * @paragraph   License
 * ESA-PL Strong Copyleft – v2.5
 * ======================================================================= */
#pragma once
// Include all the benchmark common variable, struct, prototype, lib
#include "benchmark_common.h"

// ======= Benchmark local variable =======
// --- Nothing for now --

struct GraficObject : public GraficCommon {
	#ifdef CUDA
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
		// CPU PART
		bench_t* d_A;
		bench_t* d_B;
	#endif
};

// --- Specefic overload of benchmarking function ---
void execute_kernel(GraficCommon* device_object, unsigned int n, unsigned int m, unsigned int w, unsigned int stride, unsigned int size_lateral);
