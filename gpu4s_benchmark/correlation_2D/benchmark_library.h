/** * ====================================================================
 * @file        benchmark_library.h (./correlation_2D)
 * @brief       Specific memory structures and function overloads 
 *              for the Correlation 2D benchmark.
 * @paragraph   License
 * ESA-PL Strong Copyleft – v2.5
 * ======================================================================= */
#pragma once
// Include all the benchmark common variable, struct, prototype, lib
#include "benchmark_common.h"

// ======= Benchmark local variable =======
// --- Core Data Types ---
#ifdef INT
	typedef float result_bench_t;
#elif FLOAT
	typedef float result_bench_t;
#elif DOUBLE
	typedef double result_bench_t;
#endif

// --- OpenCL Runtime Kernel Code  ---
#ifdef OPENCL
    #ifdef INT
        static const std::string type_kernel = type_kernel_common + "typedef float result_bench_t;\n";
    #elif FLOAT
        static const std::string type_kernel = type_kernel_common + "typedef float result_bench_t;\n";
    #elif DOUBLE
        static const std::string type_kernel = type_kernel_common + "typedef double result_bench_t;\n";
    #endif
#endif

struct GraficObject : public GraficCommon {
	#ifdef CUDA
		// CUDA PART
		bench_t* d_A;
		bench_t* d_B;
		result_bench_t* d_R;
		result_bench_t* mean_A; // axuliar values for the mean of matrix A
		result_bench_t* mean_B; // axuliar values for the mean of matrix B
		result_bench_t* acumulate_value_a_b; // auxiliar values for the acumulation
		result_bench_t* acumulate_value_a_a; // auxiliar values for the acumulation
		result_bench_t* acumulate_value_b_b; // auxiliar values for the acumulation
   	#elif OPENCL
		// OpenCL PART
		cl::Event *evt_copyA;
		cl::Event *evt_copyB;
		cl::Event *evt_copyAB;
		cl::Event *evt_copyAA;
		cl::Event *evt_copyBB;
		cl::Event *evt;
		cl::Event *evt_mean;
		cl::Buffer *d_A;
		cl::Buffer *d_B;
		cl::Buffer *d_R;
		cl::Buffer *mean_A; // axuliar values for the mean of matrix A
		cl::Buffer *mean_B; // axuliar values for the mean of matrix B
		cl::Buffer *acumulate_value_a_b; // auxiliar values for the acumulation
		cl::Buffer *acumulate_value_a_a; // auxiliar values for the acumulation
		cl::Buffer *acumulate_value_b_b; // auxiliar values for the acumulation
	#elif HIP
		// Hip part --
		bench_t* d_A;
		bench_t* d_B;
		result_bench_t* d_R;
		result_bench_t* mean_A; // axuliar values for the mean of matrix A
		result_bench_t* mean_B; // axuliar values for the mean of matrix B
		result_bench_t* acumulate_value_a_b; // auxiliar values for the acumulation
		result_bench_t* acumulate_value_a_a; // auxiliar values for the acumulation
		result_bench_t* acumulate_value_b_b; // auxiliar values for the acumulation
	#elif OPENMP
		// OpenMP part
		bench_t* d_A;
		bench_t* d_B;
		result_bench_t d_R;
		result_bench_t mean_A; // axuliar values for the mean of matrix A
		result_bench_t mean_B; // axuliar values for the mean of matrix B
		result_bench_t acumulate_value_a_b; // auxiliar values for the acumulation
		result_bench_t acumulate_value_a_a; // auxiliar values for the acumulation
		result_bench_t acumulate_value_b_b; // auxiliar values for the acumulation
	#else
		// CPU part
		bench_t* d_A;
		bench_t* d_B;
		result_bench_t d_R;
		result_bench_t mean_A; // axuliar values for the mean of matrix A
		result_bench_t mean_B; // axuliar values for the mean of matrix B
		result_bench_t acumulate_value_a_b; // auxiliar values for the acumulation
		result_bench_t acumulate_value_a_a; // auxiliar values for the acumulation
		result_bench_t acumulate_value_b_b; // auxiliar values for the acumulation
	#endif
};

// --- Specefic overload of benchmarking function ---
void copy_memory_to_device(GraficCommon* device_object, bench_t* h_A, unsigned int size_a, bench_t* h_B, unsigned int size_b);
void copy_memory_to_host(GraficCommon* device_object, result_bench_t* h_R);
